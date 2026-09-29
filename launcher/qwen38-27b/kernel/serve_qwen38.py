"""
Serve Qwen3.8-27B (bf16) on a Kaggle TPU v5e-8 with vLLM.

This script is pushed to Kaggle as a script kernel by ../launch.py, which fills
in the CFG line below. It also runs standalone with defaults (e.g. pasted into
a Kaggle notebook/script in the UI) — then it just prints instead of using ntfy.

Steps (each one is announced in the log):
  1/6  runtime  — venv with vllm-tpu (pinned, CPU torch) built by uv in ~30 s,
                  resolution pinned to the env dataset's build date; + the MTP fix
  2/6  cache    — restore the pre-built XLA compile cache from the env dataset
  3/6  weights  — find the mounted weights dataset (or download from HF to /tmp)
  4/6  server   — start vLLM (TP=8, text-only, MTP speculative decoding)
  5/6  tunnel   — open a public cloudflared URL (printed before the server is
                  live so you can prepare your client)
  6/6  ready    — READY banner + self-test, then keep serving until
                  keepalive_min elapses

With both datasets attached the endpoint is live in ~22 minutes (~12 with
text_only, ~6 with fast_start). Without the env dataset the compile is cold (+15 min).
"""
import base64
import collections
import struct
import zlib
import glob
import gzip
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line)

DEFAULTS = {
    "vllm_tpu_version": "0.28.0",
    "weights_dataset": "rahim3/qwen3-8-27b-bf16",     # HF mirror of Qwen/Qwen3.8-27B
    "env_dataset": "rahim3/qwen38-tpu-env-v5e8",       # XLA cache + cloudflared + manifest
    "hf_model_id": "Qwen/Qwen3.8-27B",                # fallback download source
    "max_model_len": 262144,       # native context; drop to 131072 + max_num_seqs 16 for throughput
    "max_num_seqs": 4,
    "mtp_tokens": 3,               # MTP spec decoding (+34% in our A/B test). Stock vllm-tpu
                                   # 0.28.0 corrupts outputs with it (missing GDN state
                                   # rollback); we apply patches/mtp-rollback-v0280.diff
                                   # (a port of upstream PR #3178) before serving —
                                   # verified lossless, 12/12 greedy exact-match.
    "async_scheduling": False,     # vllm-tpu 0.28.0 + MTP crashes on some requests with async scheduling.
    "reasoning_effort_default": "xhigh",   # server-side default: xhigh | medium | low
    "tool_call_parser": "qwen3_coder",  # matches Qwen3.8's XML tool format; "" disables
    "text_only": False,            # True: skip the vision tower + its TPU graphs (saves ~8 min,
                                   # image inputs then error out)
    "min_token_bucket": 64,        # smallest padded batch (tokens); 16 = more graphs to compile
    "precompile_workers": 4,       # parallel XLA compile threads (1 = sequential)
    "fast_start": False,           # True: skip precompile -> READY in ~4 min (needs the env
                                   # dataset's cache); the script then warms the common
                                   # request shapes itself; rare shapes stall once (~1 min)
    "keepalive_min": 480,          # auto-shutdown guard (Kaggle TPU caps at 9h anyway)
    "api_key": "",                 # generated if empty
    "ntfy_topic": "",              # optional: publish progress to ntfy.sh/<topic>
    "tunnel_token": "",            # Cloudflare named-tunnel token -> static hostname (see SETUP.md)
    "tunnel_hostname": "",         # public hostname routed to that tunnel, e.g. "qwen.example.com"
    "served_model_name": "qwen3.8-27b",
    "verbose": False,              # show every vLLM log line (always saved to vllm.log)
    "build_bundle": False,         # maintainer mode: build the env dataset instead of serving
}
CFG = {**DEFAULTS, **(CFG or {})}
# Notebook flow: drop overrides in a serve_config.json next to this script
# (falls back to the working directory when pasted into a notebook, where
# __file__ does not exist).
try:
    _cfg_file = Path(__file__).resolve().parent / "serve_config.json"
except NameError:
    _cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

PORT = 8000
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
XLA_CACHE = "/tmp/xla_cache"
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
RAW_LOG = WORK / "vllm.log"          # every line vLLM/pip print, for debugging
CLOUDFLARED = Path("/tmp/cloudflared")
CLOUDFLARED_BUNDLED = Path("/tmp/cloudflared-bundled")
T0 = time.time()
PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"

os.environ["HF_HOME"] = "/tmp/hf"                 # /kaggle/working is only ~21 GB
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["VLLM_XLA_CACHE_PATH"] = XLA_CACHE
os.environ["MIN_TOKEN_BUCKET"] = str(CFG["min_token_bucket"])
os.environ["NUM_PRECOMPILE_WORKERS"] = str(CFG["precompile_workers"])
if CFG["fast_start"]:
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
# The venv ships its own libtpu; don't let the image's TPU_LIBRARY_PATH override it.
os.environ.pop("TPU_LIBRARY_PATH", None)

_raw = open(RAW_LOG, "a", buffering=1)


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    print(line, flush=True)
    _raw.write(line + "\n")


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    """Progress event: always logged; also pushed to ntfy if a topic is set."""
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {phase}",
                "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag, show=None, env=None):
    """Run a command; stream its output to vllm.log (and to the console when
    show/verbose). Returns the exit code."""
    show = CFG["verbose"] if show is None else show
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env)
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        _raw.write(f"[{tag}] {line}\n")
        if show:
            print(f"[{tag}] {line[:400]}", flush=True)
    rc = p.wait()
    if rc != 0 and not show:
        log(f"[{tag}] exited with code {rc}; last lines:")
        for ln in list(tail)[-15:]:
            print("    " + ln[:300], flush=True)
    return rc


def find_input(*patterns):
    """Datasets mount at /kaggle/input/<slug> (UI) or /kaggle/input/datasets/<owner>/<slug> (API push)."""
    for pat in patterns:
        hits = glob.glob(f"/kaggle/input/{pat}") + glob.glob(f"/kaggle/input/datasets/*/{pat}")
        if hits:
            return hits[0]
    return None


def fetch_cloudflared():
    if CLOUDFLARED.exists():
        return
    try:
        downloaded = CLOUDFLARED.with_suffix(".download")
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            downloaded)
        downloaded.replace(CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
        log("   downloaded fresh cloudflared")
    except Exception as e:
        log(f"(cloudflared download failed: {e})")


# gzip+base64 of patches/mtp-rollback-v0280.diff + patches/spec-draft-rows-v0280.diff
# (applied in that order); regenerated by tools/embed_patch.py
MTP_PATCH_B64 = "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"  # __EMBEDDED_PATCH__


def apply_mtp_patch():
    """Port of upstream PR #3178 (GDN state rollback on rejected draft tokens).
    Without it ANY speculative decoding corrupts outputs on this model."""
    if not MTP_PATCH_B64:
        return True
    Path("/tmp/mtpfix.diff").write_text(
        gzip.decompress(base64.b64decode(MTP_PATCH_B64)).decode())
    origin = subprocess.check_output(
        [PY, "-c", "import importlib.util as u; print(u.find_spec('tpu_inference').origin)"],
        text=True).strip()
    pkg_root = os.path.dirname(os.path.dirname(origin))
    p = subprocess.run(["patch", "-p1", "-d", pkg_root, "-i", "/tmp/mtpfix.diff",
                        "--no-backup-if-mismatch", "-N"], capture_output=True, text=True)
    _raw.write(p.stdout + p.stderr)
    if p.returncode == 0 or "previously applied" in p.stdout:
        return True
    log(p.stdout[-1500:], p.stderr[-500:])
    return False


def runtime_ok():
    r = subprocess.run([PY, "-c", "import importlib.util as u, jax, torch\n"
                        "assert u.find_spec('vllm') and u.find_spec('tpu_inference')\n"
                        "print(jax.__version__, torch.__version__)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        log(f"   runtime check OK (jax {r.stdout.split()[0]}, torch {r.stdout.split()[1]})")
        return True
    log("   runtime check FAILED:", (r.stderr or r.stdout)[-800:])
    return False


def install_runtime(built=None):
    """Fresh venv with vllm-tpu pinned. CPU torch (what vllm-tpu's own Docker
    image uses) — the default PyPI torch drags in ~3 GB of CUDA libraries that
    a TPU never uses. `built` (a date from the env dataset's manifest) pins the
    dependency resolution to that day so the compile cache keeps matching."""
    ver = CFG["vllm_tpu_version"]
    shutil.rmtree(VENV, ignore_errors=True)
    pin = ["--exclude-newer", f"{built}T23:59:59Z"] if built else []
    log("   building venv with uv" + (f" (packages as of {built})" if built else "") + "...")
    if (sh([sys.executable, "-m", "pip", "install", "-q", "uv"], "pip") == 0
            and sh([sys.executable, "-m", "uv", "venv", VENV, "--python", sys.executable, "-q"], "uv") == 0
            and sh([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
                    "--torch-backend=cpu", *pin, f"vllm-tpu=={ver}"], "uv") == 0):
        return "uv"
    log("   uv failed; falling back to pip (slower)")
    shutil.rmtree(VENV, ignore_errors=True)
    if sh([sys.executable, "-m", "venv", "--without-pip", VENV], "venv") != 0:
        return None
    rc = sh([sys.executable, "-m", "pip", "--python", PY, "install", "-q",
             "--extra-index-url", "https://download.pytorch.org/whl/cpu",
             f"vllm-tpu=={ver}"], "pip")
    return "pip" if rc == 0 else None


# ---------------- 1. runtime ----------------
banner(1, "Python runtime", f"vllm-tpu {CFG['vllm_tpu_version']}")
threading.Thread(target=fetch_cloudflared, daemon=True).start()
bundle_root = find_input(CFG["env_dataset"].split("/")[-1], "qwen38-tpu-env*")
bundle, manifest = None, {}
if bundle_root:
    # Kaggle may keep the files at the top level or under the kernel's output folder
    hits = glob.glob(f"{bundle_root}/manifest.json") + glob.glob(f"{bundle_root}/*/manifest.json")
    if hits:
        bundle = os.path.dirname(hits[0])
        manifest = json.loads(Path(hits[0]).read_text())
    else:
        bundle = bundle_root
if bundle and Path(bundle, "cloudflared").exists():
    shutil.copy(Path(bundle, "cloudflared"), CLOUDFLARED_BUNDLED)
    CLOUDFLARED_BUNDLED.chmod(0o755)
if manifest and (manifest.get("python") != PY_VER
                 or manifest.get("vllm_tpu_version") != CFG["vllm_tpu_version"]):
    log(f"   env dataset was built for python {manifest.get('python')} / vllm-tpu "
        f"{manifest.get('vllm_tpu_version')}; this session has python {PY_VER} and wants "
        f"vllm-tpu {CFG['vllm_tpu_version']} -> its compile cache will not match")
    manifest = {}
if not bundle:
    log(f"   env dataset not attached (expected {CFG['env_dataset']}) -> cold compile later")

t = time.time()
publish("install", vllm_tpu=CFG["vllm_tpu_version"])
log("   installer output goes to", RAW_LOG)
runtime = install_runtime(manifest.get("built"))
if runtime is None or not runtime_ok():
    publish("failed", step="install")
    sys.exit(1)
publish("installed", secs=int(time.time() - t), via=runtime)
if apply_mtp_patch():
    publish("mtp-patch-applied")
elif CFG["mtp_tokens"] > 0:
    publish("mtp-patch-failed", note="disabling MTP: unsafe without the rollback patch")
    CFG["mtp_tokens"] = 0
log(f"   runtime ready in {int(time.time() - t)} s")

# ---------------- 2. XLA compile cache ----------------
banner(2, "XLA compile cache")
t = time.time()
cache_tar = (Path(bundle, "xla_cache.tar") if bundle and Path(bundle, "xla_cache.tar").exists()
             else find_input("*/xla_cache*.tar.gz", "xla_cache*.tar.gz"))
cache_dir = find_input("*/*/xla_cache", "*/xla_cache", "xla_cache")
if cache_tar:
    flags = "-xf" if str(cache_tar).endswith(".tar") else "-xzf"
    sh(["tar", flags, str(cache_tar), "-C", "/tmp"], "tar")
elif cache_dir:
    sh(["cp", "-r", cache_dir, "/tmp/"], "cp")
    sh(["chmod", "-R", "u+w", XLA_CACHE], "chmod")
n_entries = len(glob.glob(XLA_CACHE + "/*"))
cache_configs = manifest.get("configs", [])
this_config = [CFG["max_model_len"], CFG["max_num_seqs"], CFG["mtp_tokens"], CFG["text_only"]]
if n_entries:
    covered = (not cache_configs) or (this_config in cache_configs)
    publish("cache-restored", entries=n_entries, secs=int(time.time() - t),
            covers_this_config=covered)
    if not covered:
        log(f"   note: the cache was built for [ctx, seqs, mtp, text_only] in {cache_configs}; "
            f"this run uses {this_config} -> its graphs compile cold (add ~10-15 min)")
    else:
        log("   compiled TPU graphs for this exact config are cached -> fast start")
else:
    publish("cache-missing", note="cold compile: expect ~10 extra minutes")

# ---------------- 3. weights ----------------
banner(3, "Model weights", "55 GB bf16 safetensors")
weights_slug = CFG["weights_dataset"].split("/")[-1]
model_path = find_input(weights_slug)
if model_path and os.path.exists(os.path.join(model_path, "config.json")):
    publish("weights-mounted", path=model_path)
else:
    publish("weights-download", model=CFG["hf_model_id"],
            note="attach the weights dataset to skip this (~5 min parallel download)")
    t = time.time()
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(CFG["hf_model_id"], allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "vocab*", "merges*"])
    publish("weights-downloaded", secs=int(time.time() - t))


# ---------------- 4. vLLM server ----------------
NOISE = ("vllm._C", "metadata.google.internal", "Triton is installed", "Transparent hugepages",
         "Pin memory is not supported", "Expect torch.Tensor", "Inductor compilation",
         "cloud_tpu_init.py", "SyntaxWarning", "Compilation of worker", "AOT lower skipped",
         "torch_dtype", "UserWarning", "warnings.warn", "resource_tracker", "Precompile worker0 sample",
         "Precompile worker0 gather", "Precompile worker0 compute_and_gather")


def server_args(cfg):
    args = [PY, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--tensor-parallel-size", "8",
            "--max-model-len", str(cfg["max_model_len"]),
            "--max-num-seqs", str(cfg["max_num_seqs"]),
            "--port", str(PORT),
            "--api-key", cfg["api_key"],
            "--served-model-name", cfg["served_model_name"],
            "--reasoning-parser", "qwen3"]
    if cfg.get("async_scheduling") is not None:
        args.append("--async-scheduling" if cfg["async_scheduling"] else "--no-async-scheduling")
    if cfg["text_only"]:
        # Qwen3.8 is a vision-language checkpoint; we only serve text. This skips
        # the vision tower and roughly halves the number of TPU graphs to compile.
        args += ["--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    if cfg["mtp_tokens"] > 0:
        args += ["--speculative-config",
                 json.dumps({"method": "mtp", "num_speculative_tokens": cfg["mtp_tokens"]})]
    if cfg["tool_call_parser"]:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", cfg["tool_call_parser"]]
    if cfg["reasoning_effort_default"] != "xhigh":
        # The chat template defaults reasoning_effort to 'xhigh'; ship a copy with a
        # different default so the server-side default changes without client changes.
        tc = json.loads(Path(model_path, "tokenizer_config.json").read_text())
        template = tc.get("chat_template")
        marker = "reasoning_effort|default('xhigh')"
        if not isinstance(template, str) or marker not in template:
            log("   WARNING: could not override reasoning_effort (chat_template missing, "
                "not a plain string, or without the default-effort marker) -> serving "
                "with the checkpoint's own default (xhigh)")
        else:
            Path("/tmp/chat_template.jinja").write_text(template.replace(
                marker, f"reasoning_effort|default('{cfg['reasoning_effort_default']}')"))
            args += ["--chat-template", "/tmp/chat_template.jinja"]
    return args


def n_token_graphs():
    n, b = 1, CFG["min_token_bucket"]
    while b < 2048:  # vLLM's default max_num_batched_tokens on TPU
        b *= 2
        n += 1
    return n


def make_translator():
    """Turns vLLM's firehose into a handful of human lines. Everything raw still
    lands in vllm.log."""
    st = {"graph": 0, "loads": 0, "said": set()}
    n_graphs = n_token_graphs()

    def once(key, msg):
        if key not in st["said"]:
            st["said"].add(key)
            log(msg)

    def tr(line):
        if CFG["verbose"]:
            print(f"[vllm] {line[:500]}", flush=True)
            return
        if any(k in line for k in NOISE):
            return
        m = re.search(r"Loading weights took ([\d.]+) seconds", line)
        if m:
            st["loads"] += 1
            if st["loads"] == 1:
                log(f"   weights read from the dataset in {float(m.group(1)):.0f} s")
            return
        m = re.search(r"load model weights from storage to TPU: ([\d.]+)", line)
        if m:
            if st["loads"] <= 1:
                log(f"   weights sharded across the 8 TPU chips ({float(m.group(1)):.0f} s)")
            else:
                log("   MTP draft head loaded")
            return
        m = re.search(r"KV cache size: ([\d,]+) tokens", line)
        if m:
            log(f"   KV cache fits {m.group(1)} tokens")
            return
        if "Precompile all the subgraphs" in line:
            log(f"   compiling TPU graphs — {n_graphs} text graphs"
                + ("" if CFG["text_only"] else ", the same again for image inputs,")
                + " + helpers (~20 s each if cached, ~1 min if not)")
            return
        m = re.search(r"Precompile worker\d+ backbone --> \{'num_tokens': (\d+)", line)
        if m:
            st["graph"] += 1
            log(f"     graph {st['graph']}/{n_graphs}: batches of {m.group(1)} tokens")
            return
        if "embed_multimodal" in line or "input_embeddings_merger" in line:
            once("vision-enc", "     warming the image encoder (~5 min; \"text_only\": true skips it)")
            return
        if "backbone with embeds" in line:
            once("vision", "     compiling image-input graphs (~3 min)")
            return
        m = re.search(r"Warm-up call pass finished in ([\d.]+) \[secs\] over (\d+) tasks", line)
        if m:
            if float(m.group(1)) > 5:
                log(f"     warm-up run of {m.group(2)} graphs done ({float(m.group(1)):.0f} s)")
            return
        if "Precompile" in line and "drafter" in line:
            once("mtp", "     compiling speculative-decoding (MTP) graphs")
            return
        if "Precompile" in line or "Compilation of" in line:
            once("helpers", "     compiling sampler / helper graphs")
            return
        if "Application startup complete" in line:
            return
        if " ERROR " in line or "Traceback" in line or "Error:" in line or "rror(" in line:
            print(time.strftime("[%H:%M:%S] ") + f"   [vllm] {line[:400]}", flush=True)
    return tr


def launch_server(cfg):
    publish("server-launch", max_model_len=cfg["max_model_len"],
            max_num_seqs=cfg["max_num_seqs"], mtp=cfg["mtp_tokens"],
            text_only=cfg["text_only"], min_token_bucket=cfg["min_token_bucket"])
    tail = collections.deque(maxlen=200)
    p = subprocess.Popen(server_args(cfg), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    tr = make_translator()

    def pump():
        for line in p.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                _raw.write(f"[vllm] {line}\n")
                tr(line)
    threading.Thread(target=pump, daemon=True).start()
    p.tail = tail
    return p


def healthy(cfg):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models",
                                     headers={"Authorization": f"Bearer {cfg['api_key']}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(server, cfg, expect_min):
    t = time.time()
    while time.time() - t < 5400:
        if server.poll() is not None:
            tail = "\n".join(list(server.tail)[-100:])
            log(f"server exited rc={server.returncode}; last output:\n{tail}")
            log(f"full log: {RAW_LOG}")
            publish("failed", step="server", rc=server.returncode, tail=tail[-2500:])
            sys.exit(1)
        if healthy(cfg):
            return int(time.time() - t)
        el = int(time.time() - t)
        if el and el % 120 < 6:
            publish("compiling", elapsed_s=el)
            log(f"   ... {el // 60} min into startup (typically ~{expect_min} min)")
        time.sleep(5)
    publish("failed", step="health-timeout", tail="\n".join(list(server.tail)[-60:])[-2500:])
    sys.exit(1)


def stop_server(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=30)
    time.sleep(10)  # let the TPU runtime free the chips


def completion(cfg, prompt, max_tokens, stream=False, timeout=900):
    body = {"model": cfg["served_model_name"], "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0}
    if stream:
        body.update(stream=True, ignore_eos=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    t0 = time.time(); ttft = None; gen = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                gen += 1
            if obj.get("usage"):
                gen = obj["usage"].get("completion_tokens", gen)
    return ttft, time.time() - t0, gen


def test_png(w=256, h=256, rgb=(200, 30, 30)):
    """A solid-colour PNG without PIL, for the image self-test."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def chat(cfg, messages, max_tokens=32, timeout=600):
    body = {"model": cfg["served_model_name"], "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def self_test(cfg):
    """Warms the remaining lazy paths, checks an image request when images are
    enabled, and reports single-stream decode speed."""
    tps = None
    try:
        completion(cfg, "Hello", 8)
        txt = completion(cfg, "The capital of France is", 8)["choices"][0]["text"]
        ttft, total, gen = completion(cfg, "Write a short story about a lighthouse.", 192, stream=True)
        tps = (gen - 1) / (total - ttft) if gen > 1 else 0.0
        publish("benchmark", decode_tok_s=round(tps, 1), sanity=txt.strip()[:60])
        log(f"   self-test: {tps:.1f} tok/s single-stream decode; "
            f"'The capital of France is' -> {txt.strip()[:40]!r}")
    except Exception as e:
        publish("benchmark-error", err=str(e)[:200])
    if not cfg["text_only"]:
        try:
            img = "data:image/png;base64," + base64.b64encode(test_png()).decode()
            ans = chat(cfg, [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": "What colour is this image? One word."}]}])
            publish("image-test", answer=ans.strip()[:40])
            log(f"   image request works (a red square -> {ans.strip()[:30]!r})")
        except Exception as e:
            publish("image-test-failed", err=str(e)[:200])
            log(f"   IMAGE REQUEST FAILED: {str(e)[:200]}")
    return tps


def exercise(cfg, quiet=False):
    """Hit the shapes a real client hits: short and long prompts, streaming, a
    small concurrent batch. In build mode this puts their graphs in the cache;
    in fast_start mode it loads them so users don't hit the one-time stalls."""
    steps = [("short prompt", lambda: completion(cfg, "Hello", 8)),
             ("4k-token prompt", lambda: completion(
                 cfg, "The quick brown fox jumps over the lazy dog. " * 400, 16, timeout=1200)),
             ("streaming", lambda: completion(
                 cfg, "Write a short story about a lighthouse.", 64, stream=True))]
    n = min(cfg["max_num_seqs"], 8)
    errs = []

    def one():
        try:
            completion(cfg, "Count from one to twenty in words.", 48)
        except Exception as e:
            errs.append(str(e)[:200])

    def batch():
        ths = [threading.Thread(target=one) for _ in range(n)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    steps.append((f"{n} parallel requests", batch))
    for name, fn in steps:
        t = time.time()
        try:
            fn()
        except Exception as e:
            errs.append(f"{name}: {str(e)[:200]}")
        if not quiet:
            log(f"   warmed: {name} ({time.time() - t:.0f} s)")
    return errs


# ---------------- maintainer mode: build the env dataset ----------------
BUILD_CONFIGS = [
    {"max_model_len": 262144, "max_num_seqs": 4, "mtp_tokens": 3, "text_only": False},
    {"max_model_len": 131072, "max_num_seqs": 16, "mtp_tokens": 3, "text_only": False},
    {"max_model_len": 262144, "max_num_seqs": 4, "mtp_tokens": 3, "text_only": True},
]
if CFG["build_bundle"]:
    # fast_start must not leak into the cache-populating launches, or the
    # packed xla_cache.tar would be mostly empty.
    os.environ.pop("SKIP_JAX_PRECOMPILE", None)
    CFG["fast_start"] = False
    banner(4, "BUILD MODE", "serving each config once to populate the XLA cache")
    log("   TPU-related env:", {k: v for k, v in os.environ.items() if "TPU" in k or "PJRT" in k})
    results = {}
    for c in BUILD_CONFIGS:
        cfg = {**CFG, **c}
        key = (f"{c['max_model_len']}/{c['max_num_seqs']}/mtp{c['mtp_tokens']}"
               + ("/text" if c["text_only"] else "/mm"))
        server = launch_server(cfg)
        secs = wait_healthy(server, cfg, 30)
        errs = exercise(cfg, quiet=True)
        tps = self_test(cfg)
        stop_server(server)
        results[key] = {"startup_secs": secs, "decode_tok_s": tps, "errors": errs}
        publish("build-config-done", config=key, **results[key])
    # probe: how fast is a start with SKIP_JAX_PRECOMPILE=1 now that the cache is warm?
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 5)
    lat = []
    for i in range(3):
        t = time.time()
        try:
            completion(cfg, ["Hello", "Say hi.", "Name a color."][i], 8, timeout=1800)
            lat.append(round(time.time() - t, 1))
        except Exception as e:
            lat.append(str(e)[:100])
    tps = self_test(cfg)
    stop_server(server)
    os.environ.pop("SKIP_JAX_PRECOMPILE")
    publish("probe-skip-precompile", startup_secs=secs, first_request_secs=lat, decode_tok_s=tps)
    # also a warm re-start of the default config (what users will see)
    cfg = {**CFG, **BUILD_CONFIGS[0]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 15)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart", startup_secs=secs, decode_tok_s=tps)
    cfg = {**CFG, **BUILD_CONFIGS[2]}
    server = launch_server(cfg)
    secs = wait_healthy(server, cfg, 8)
    tps = self_test(cfg)
    stop_server(server)
    publish("probe-warm-restart-text-only", startup_secs=secs, decode_tok_s=tps)

    out = WORK / "bundle"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    banner(5, "packing the bundle", str(out))
    # (no venv tarball: uv rebuilds the identical env in ~30 s, and Kaggle would
    #  unpack a tar into 100k files anyway — some with '[' in the name, which it rejects)
    sh(["tar", "-cf", str(out / "xla_cache.tar"), "-C", "/tmp", "xla_cache"], "tar")
    fetch_cloudflared()
    if CLOUDFLARED.exists():
        shutil.copy(CLOUDFLARED, out / "cloudflared")
    pkgs = subprocess.run([sys.executable, "-m", "uv", "pip", "list", "--python", PY,
                           "--format=json"], capture_output=True, text=True)
    try:
        pkgs = {d["name"]: d["version"] for d in json.loads(pkgs.stdout)}
    except Exception:
        pkgs = {}
    manifest = {
        "built": time.strftime("%Y-%m-%d"),
        "python": PY_VER,
        "vllm_tpu_version": CFG["vllm_tpu_version"],
        "mtp_patch": "applied at runtime",
        "min_token_bucket": CFG["min_token_bucket"],
        "configs": [[c["max_model_len"], c["max_num_seqs"], c["mtp_tokens"], c["text_only"]]
                    for c in BUILD_CONFIGS],
        "results": results,
        "accelerator": "TPU v5e-8 (Kaggle)",
        "packages": pkgs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sizes = {p.name: round(p.stat().st_size / 1e9, 2) for p in out.iterdir()}
    publish("bundle-built", sizes_gb=sizes, results=results)
    sys.exit(0)

# ---------------- 4. launch ----------------
expect_min = (10 if n_entries else 20) + (0 if CFG["text_only"] else (10 if n_entries else 15))
if CFG["fast_start"]:
    expect_min = 5 if n_entries else 7
    if not n_entries:
        log("   fast_start without a compile cache: every new request shape will compile "
            "cold (~1 min each) — attach the env dataset for this mode to make sense")
banner(4, "Starting vLLM", f"TP=8, ctx {CFG['max_model_len']}, {CFG['max_num_seqs']} seqs, "
       f"MTP k={CFG['mtp_tokens']}, {'text-only' if CFG['text_only'] else 'multimodal'}")
log(f"   expect ~{expect_min} min; progress lines below, full vLLM log in {RAW_LOG}")
server = launch_server(CFG)

# ---------------- 5. tunnel (in parallel with the server start) ----------------
banner(5, "Public URL")
url = None
tunnel = None
for _ in range(60):  # cloudflared download runs in the background from step 1
    if CLOUDFLARED.exists():
        break
    time.sleep(2)
if not CLOUDFLARED.exists() and CLOUDFLARED_BUNDLED.exists():
    shutil.copy(CLOUDFLARED_BUNDLED, CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)
    log("   using bundled cloudflared because the fresh download was unavailable")
if CLOUDFLARED.exists():
    version = subprocess.run([str(CLOUDFLARED), "--version"], capture_output=True,
                             text=True, timeout=10)
    log(f"   cloudflared: {(version.stdout or version.stderr).strip()[:200]}")
if CLOUDFLARED.exists() and CFG["tunnel_token"]:
    # Named tunnel: static hostname, set up once in the Cloudflare dashboard
    # (Zero Trust -> Networks -> Tunnels). No trycloudflare.com scraping needed —
    # we already know the hostname; just wait for cloudflared to come up.
    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "run",
                               "--token", CFG["tunnel_token"]],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def pump_named():
        for line in tunnel.stdout:
            _raw.write(f"[cloudflared] {line}")
    threading.Thread(target=pump_named, daemon=True).start()
    time.sleep(3)  # let cloudflared fail fast on a bad token/route rather than
                   # silently reporting a hostname that will just 502 forever
    if tunnel.poll() is None:
        url = f"https://{CFG['tunnel_hostname']}"
    else:
        _raw.write("[cloudflared] named tunnel exited immediately — check "
                    "tunnel_token / tunnel_hostname\n")
elif CLOUDFLARED.exists():
    # Quick tunnel (default): free, zero setup, but a new random
    # trycloudflare.com URL every boot.
    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                               "--no-autoupdate"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    lines = []

    def pump_cf():
        for line in tunnel.stdout:
            lines.append(line.rstrip())
            _raw.write(f"[cloudflared] {line}")
    threading.Thread(target=pump_cf, daemon=True).start()
    deadline = time.time() + 180
    while time.time() < deadline and url is None:
        for ln in lines:
            m = pat.search(ln)
            if m:
                url = m.group(0).rstrip("/")
                break
        time.sleep(1)
if url:
    log(f"   your endpoint will be  {url}/v1")
    log("   (not live yet — it answers 502 until the READY banner below)")
    publish("tunnel-url", endpoint=f"{url}/v1")
else:
    tunnel_detail = "server still reachable inside the kernel on :8000"
    if tunnel is not None and tunnel.poll() is not None:
        tunnel_detail += f"; cloudflared exited with code {tunnel.returncode}"
    publish("tunnel-failed", note=tunnel_detail)

# ---------------- 6. wait, announce, self-test, keep alive ----------------
startup = wait_healthy(server, CFG, expect_min)
publish("serving", startup_secs=startup)
log("")
log("#" * 70)
log(f"#  READY — the server is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {url + '/v1' if url else 'http://127.0.0.1:8000/v1 (tunnel failed)'}")
log(f"#  API KEY  : {CFG['api_key']}")
log(f"#  MODEL    : {CFG['served_model_name']}   (context {CFG['max_model_len']}, "
    f"{CFG['max_num_seqs']} parallel requests)")
log("#" * 70)
log("#  Try it:")
log(f"#    curl {url + '/v1' if url else 'http://127.0.0.1:8000/v1'}/chat/completions \\")
log(f"#      -H 'Authorization: Bearer {CFG['api_key']}' -H 'Content-Type: application/json' \\")
log("#      -d '{\"model\": \"" + CFG["served_model_name"] + "\", \"messages\": [{\"role\": \"user\", "
    "\"content\": \"Hello!\"}], \"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}'")
log(f"#  Serving for up to {CFG['keepalive_min']} min, then this cell exits on its own.")
log("#" * 70)
publish("ready", endpoint=(f"{url}/v1" if url else None), api_key=CFG["api_key"],
        model=CFG["served_model_name"], max_model_len=CFG["max_model_len"],
        keepalive_min=CFG["keepalive_min"], startup_secs=startup)

if CFG["fast_start"]:
    banner(6, "Warm-up", "loading the common request shapes; the endpoint is usable meanwhile")
    log("   (fast_start: a request with a new shape waits ~1 min the first time)")
    exercise(CFG)
else:
    banner(6, "Self-test", "one short generation; the endpoint is usable meanwhile")
self_test(CFG)

t_serve = time.time()
while time.time() - t_serve < CFG["keepalive_min"] * 60:
    time.sleep(120)
    if server.poll() is not None:
        publish("stopped", reason="server-exit", rc=server.returncode)
        sys.exit(1)
    up = int((time.time() - t_serve) / 60)
    if up % 10 < 2:
        publish("heartbeat", up_min=up, endpoint=(f"{url}/v1" if url else None))
        log(f"   still serving ({up} min) — {url + '/v1' if url else ''}")
publish("auto-shutdown", served_min=CFG["keepalive_min"])
server.terminate()
sys.exit(0)
