"""Serve-side machinery: kernel build/push, ntfy event watch, status/stop.

Moved from launch.py (which is now a thin dispatcher). Shared state:
  STATE_FILE (~/.kaggle-tpu-lab.json) — RUNTIME record of the current/last
  session (kernel, topic, per-boot api_key, model); written by `serve`, read by
  `status`/`stop`/`env` (direct-mode key).
  ktl_common config.json — DURABLE record (model, relay URL, relay secrets).

`push_and_watch()` is the reusable entry point the setup wizard calls.
"""
import base64
import io
import json
import os
import re
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import ktl_common
from ktl_common import MODELS, say

STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"
# Written by `serve` on READY so the `env` command can resolve base URL /
# client key / model later without re-asking. chmod 600 (may hold a key).
ENV_STATE_FILE = ktl_common.legacy_state_path()

# Friendly one-liners for each phase the kernel publishes.
PHASE_TEXT = {
    # --- Qwen phases ---
    "install":            "Building the Python runtime with uv (~30 s)...",
    "installed":          "Runtime ready.",
    "mtp-patch-applied":  "MTP state-rollback patch applied.",
    "mtp-patch-failed":   "MTP patch did not apply — speculative decoding disabled for safety.",
    "cache-restored":     None,  # rendered below (depends on config coverage)
    "cache-missing":      "No compile cache found — cold compile, add ~10 min.",
    "weights-mounted":    "Weights found mounted (no download needed).",
    "weights-download":   "Downloading weights from Hugging Face (~5 min)...",
    "weights-downloaded": "Weights downloaded.",
    "server-launch":      "Starting vLLM — loading weights, then TPU graph compile...",
    "serving":            "Server is HEALTHY.",
    "benchmark":          None,
    # --- GLM phases ---
    "loading":            "Loading weights onto the TPU chips...",
    "loaded":             "Weights loaded.",
    "warmed":             "Warm-up complete (prefill buckets, decode programs, snapshots).",
    # --- Shared phases ---
    "tunnel-url":         None,
    "compiling":          None,  # rendered with elapsed time below
    "ready":              None,
    "heartbeat":          None,
    "failed":             None,
    "auto-shutdown":      "Keepalive window ended — kernel shut down cleanly.",
    "stopped":            "Server exited unexpectedly.",
}


def check_auth():
    ok, detail = ktl_common.kaggle_authed()
    if not ok:
        sys.exit("Kaggle CLI is not working or not authenticated.\n"
                 "Install with `pip install kaggle`, then create a token at\n"
                 f"{ktl_common.KAGGLE_TOKEN_URL}\n"
                 "(or run `kaggle auth login` for OAuth).\n\n"
                 f"Error was:\n{detail}")


def kaggle_username(cli_arg):
    user = ktl_common.kaggle_username(cli_arg)
    if not user:
        sys.exit("Could not detect your Kaggle username — pass it with --user <name>.")
    return user


def embed_engine(model_cfg):
    """Base64 tar.gz of the engine package for kernels that carry an __ENGINE__
    slot (GLM). The kernel extracts it to /kaggle/working/glm53 — the same
    package the notebook's engine cell writes out. None for models without a
    slot (Qwen)."""
    engine_dir = model_cfg.get("engine_dir")
    if not engine_dir:
        return None
    engine_dir = Path(engine_dir)
    if not engine_dir.is_dir():
        sys.exit(f"engine package not found at {engine_dir} — cannot embed it in the kernel")
    files = [p for p in sorted(engine_dir.rglob("*.py"))
             if "tests" not in p.parts and "__pycache__" not in p.parts]
    if not files:
        sys.exit(f"no engine files found under {engine_dir}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in files:
            tf.add(p, arcname=str(Path(engine_dir.name) / p.relative_to(engine_dir)))
    say(f"Embedding engine package: {len(files)} files, "
        f"{len(buf.getvalue()) / 1e6:.2f} MB compressed")
    return base64.b64encode(buf.getvalue()).decode()


def build_kernel(model_cfg, cfg):
    """Kernel source with the launcher config (and the engine package, if any)
    injected. Returns (source, kernel_name)."""
    src = model_cfg["kernel_src"].read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{model_cfg['kernel_name']} is missing the __LAUNCHER_CONFIG__ line")
    engine_b64 = embed_engine(model_cfg)
    if engine_b64 is not None:
        src, m = re.subn(r"^ENGINE_B64 = .*__ENGINE__.*$",
                         f'ENGINE_B64 = "{engine_b64}"', src, count=1, flags=re.M)
        if m != 1:
            sys.exit(f"{model_cfg['kernel_name']} is missing the __ENGINE__ line")
    return src, model_cfg["kernel_name"]


def push_and_watch(model_key, user, slug=None, relay=None, keepalive_min=480,
                   max_model_len=262144, max_num_seqs=4, mtp=3,
                   reasoning_effort=None, weights_dataset=None,
                   no_tools=False, text_only=False, verbose=False,
                   fast_start=False, streams=None, vision=None,
                   tunnel=None,
                   watch_progress=True, stop_after_ready=False):
    """Push the serving kernel and watch it come up. Returns the runtime state
    dict. ``stop_after_ready`` makes the watch loop return right after the
    READY banner (used by `setup` so the wizard can continue).

    ``relay`` (when given) is {"url", "secret"} — the per-boot Kaggle key is
    attached here and POSTed to /update-config on READY."""
    model_cfg = MODELS[model_key]
    slug = slug or model_cfg["default_slug"]
    topic = "ktl-" + uuid.uuid4().hex[:20]
    # KTL_API_KEY lets you pin a permanent key instead of getting a fresh
    # random one every boot. Falls back to the original random behavior.
    api_key = os.environ.get("KTL_API_KEY") or "sk-" + secrets.token_hex(16)
    if relay is not None:
        relay = dict(relay)
        relay["api_key"] = api_key

    # Build the CFG dict that gets injected into the kernel script.
    cfg = {
        "ntfy_topic": topic,
        "api_key": api_key,
        "served_model_name": model_cfg["served_model_name"],
        "keepalive_min": keepalive_min,
    }

    # Model-specific config keys
    if model_key == "qwen":
        cfg["max_model_len"] = max_model_len
        cfg["max_num_seqs"] = max_num_seqs
        cfg["mtp_tokens"] = mtp
        cfg["async_scheduling"] = False
        cfg["reasoning_effort_default"] = reasoning_effort
        cfg["weights_dataset"] = weights_dataset or model_cfg["weights_dataset"]
        if tunnel is not None:
            cfg["tunnel_token"] = tunnel["token"]
            cfg["tunnel_hostname"] = tunnel["hostname"]
        if no_tools:
            cfg["tool_call_parser"] = ""
        if text_only:
            cfg["text_only"] = True
        if verbose:
            cfg["verbose"] = True
        if fast_start:
            cfg["fast_start"] = True
    elif model_key == "glm":
        cfg["reasoning_effort_default"] = reasoning_effort
        if max_model_len:
            cfg["max_len"] = max_model_len
        if streams:
            cfg["streams"] = streams
        if vision is not None:
            cfg["vision"] = vision
        if verbose:
            cfg["verbose"] = True

    src, kernel_name = build_kernel(model_cfg, cfg)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / kernel_name).write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": kernel_name,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "dataset_sources": model_cfg["dataset_sources"],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Pushing kernel {user}/{slug} ({model_key}, TPU v5e-8)...")
        r = ktl_common.run_kaggle("kernels", "push", "-p", str(td), timeout=1800)
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"WARNING: {line.strip()} — the kernel will still run, "
                    "but may need to download weights / compile cold.")

    state = {"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key,
             "model": model_key}
    STATE_FILE.write_text(json.dumps(state))
    say("Pushed. Kaggle takes a few minutes to provision the TPU and attach the "
        "datasets; the endpoint is usually live ~22 min after the kernel starts.")
    say("Watching progress (Ctrl-C is safe — the server keeps running; "
        "`python launch.py status` re-attaches, `... stop` kills it).")
    if watch_progress:
        watch(state["kernel"], topic, relay, model_key, model_cfg,
              stop_after_ready=stop_after_ready)
    return state


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
    # --- Option A: named Cloudflare Tunnel -> permanent hostname, no relay needed.
    # Set both or neither. See docs/setup-manual.md.
    tunnel_token = os.environ.get("KTL_TUNNEL_TOKEN", "")
    tunnel_hostname = os.environ.get("KTL_TUNNEL_HOSTNAME", "")
    if tunnel_token and not tunnel_hostname:
        sys.exit("KTL_TUNNEL_TOKEN is set but KTL_TUNNEL_HOSTNAME is not — set both (see docs/setup-manual.md).")
    tunnel = {"token": tunnel_token, "hostname": tunnel_hostname} if tunnel_token else None

    # --- Option B: Cloudflare Worker relay -> permanent URL in front of the
    # quick tunnel, auto-updated on every boot. Precedence: env > config.json.
    cfg = ktl_common.load_config(migrate=False)
    relay_url = ktl_common.resolve(
        None, os.environ.get("KTL_RELAY_URL", ""),
        cfg.get("cloudflare", {}).get("relay_url", ""))
    relay_secret = ktl_common.resolve(
        None, os.environ.get("KTL_RELAY_UPDATE_SECRET", ""),
        cfg.get("secrets", {}).get("update_secret", ""))
    relay = None
    if relay_url and relay_secret:
        relay = {"url": relay_url.rstrip("/"), "secret": relay_secret}
    elif relay_url or relay_secret:
        sys.exit("Set both KTL_RELAY_URL and KTL_RELAY_UPDATE_SECRET "
                 "(or run `python launch.py setup` to store them in ~/.ktl/config.json), "
                 "or neither (see docs/setup-manual.md).")

    push_and_watch(
        args.model, user, slug=args.slug, relay=relay,
        keepalive_min=args.keepalive_min, max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, mtp=args.mtp,
        reasoning_effort=args.reasoning_effort,
        weights_dataset=args.weights_dataset, no_tools=args.no_tools,
        text_only=args.text_only, verbose=args.verbose, fast_start=args.fast_start,
        streams=args.streams, vision=args.vision, tunnel=tunnel)


def read_events(topic, since):
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return []
    events = []
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        try:
            events.append((e["time"], json.loads(e.get("message", "{}"))))
        except Exception:
            continue
    return events


def register_with_relay(relay, endpoint):
    """POST the freshly-booted Kaggle endpoint + static key to the Worker relay
    (Option B). No-op if relay wasn't configured or no endpoint was reserved."""
    if not relay or not endpoint:
        return
    kaggle_root = endpoint[:-3] if endpoint.endswith("/v1") else endpoint
    payload = json.dumps({"kaggle_url": kaggle_root, "kaggle_key": relay["api_key"]}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{relay['url']}/update-config", data=payload,
                headers={"Content-Type": "application/json",
                         "User-Agent": "kaggle-tpu-lab/1.0",
                         "Authorization": f"Bearer {relay['secret']}"},
                method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                say(f"Relay updated ({r.status}) — your permanent endpoint is live.")
                return
        except Exception as e:
            if attempt == 2:
                say(f"WARNING: could not update the relay after 3 tries ({e}). "
                    "Your permanent URL will keep pointing at the previous session "
                    "until this succeeds.")
            else:
                time.sleep(3)


def relay_from_env(api_key):
    cfg = ktl_common.load_config(migrate=False)
    relay_url = ktl_common.resolve(
        None, os.environ.get("KTL_RELAY_URL", ""),
        cfg.get("cloudflare", {}).get("relay_url", ""))
    relay_secret = ktl_common.resolve(
        None, os.environ.get("KTL_RELAY_UPDATE_SECRET", ""),
        cfg.get("secrets", {}).get("update_secret", ""))
    if relay_url and relay_secret:
        return {"url": relay_url.rstrip("/"), "secret": relay_secret, "api_key": api_key}
    return None


def write_env_state(endpoint, model_key, model_cfg, relay=None):
    """Persist the client-facing base URL + model data + mode so `ktl env` can
    resolve them later.

    SECURITY: no API key is stored here (keeps secrets out of a plaintext
    file). The key is read from the environment at `ktl env` time —
    KTL_CLIENT_API_KEY for a relay session, KTL_API_KEY for a direct one.
    `mode` records which, so the two are never mixed up.

    Written atomically (temp + rename) into a 0700 dir, file 0600. Also updates
    the durable model record in ~/.ktl/config.json.
    """
    try:
        mode = "relay" if relay is not None else "direct"
        if relay is not None:
            base_url = relay["url"].rstrip("/")
        else:
            base_url = (endpoint[:-3] if endpoint.endswith("/v1") else endpoint).rstrip("/")
        state = {
            "base_url": base_url,
            "endpoint": endpoint,
            "model": model_key,
            "api_model": model_cfg.get("api_model", model_cfg.get("served_model_name")),
            "context": model_cfg.get("context"),
            "max_output": model_cfg.get("max_output"),
            "mode": mode,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        d = ENV_STATE_FILE.parent
        d.mkdir(parents=True, exist_ok=True)
        try:
            d.chmod(0o700)
        except OSError:
            pass
        tmp = ENV_STATE_FILE.with_name(ENV_STATE_FILE.name + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, ENV_STATE_FILE)   # atomic on POSIX
        ktl_common.update_config(model=model_key)
    except OSError:
        pass


def render_event(ev, relay=None, model_key=None, model_cfg=None):
    phase = ev.get("phase", "?")
    if phase == "compiling":
        # Qwen sends elapsed_s; GLM sends what + secs
        if "elapsed_s" in ev:
            say(f"Loading / compiling... {ev['elapsed_s'] // 60} min elapsed "
                "(typically ~20 min with the env dataset, ~35 min without)")
        elif "what" in ev:
            say(f"Compiling {ev['what']} ({ev.get('secs', 0)} s)")
        else:
            say(f"Compiling... {json.dumps({k: v for k, v in ev.items() if k != 'phase'})}")
    elif phase == "cache-restored":
        if ev.get("covers_this_config", True):
            say("XLA compile cache restored for this exact config — fast start.")
        else:
            say("XLA compile cache restored, but not for this config — its graphs "
                "compile cold (add ~10 min).")
    elif phase == "tunnel-url":
        say(f"Endpoint URL reserved: {ev.get('endpoint')}  (not live yet — wait for the banner)")
        register_with_relay(relay, ev.get("endpoint"))
    elif phase == "tunnel-failed":
        say("PUBLIC TUNNEL FAILED — server may be healthy, but it is not reachable from the internet.")
        if ev.get("note"):
            say(f"Tunnel detail: {ev['note']}")
    elif phase == "serving":
        say(f"Server is HEALTHY after {ev.get('startup_secs', 0) // 60} min.")
    elif phase == "benchmark":
        say(f"Quick benchmark: {ev.get('decode_tok_s', '?')} tok/s single-stream decode "
            f"(sanity: {ev['sanity']!r})")
    elif phase == "loading":
        say(f"Loading weights ({ev.get('note', '')})...")
    elif phase == "loaded":
        say(f"Weights loaded ({ev.get('minutes', '?')} min; HBM {ev.get('hbm_gb', '?')} GB/chip).")
    elif phase == "warmed":
        say(f"Warm-up done ({ev.get('minutes', '?')} min; HBM {ev.get('hbm_gb', '?')} GB/chip).")
    elif phase == "ready":
        if not ev.get("endpoint"):
            say("Server is healthy, but no public endpoint was created; relay registration was skipped.")
            return
        register_with_relay(relay, ev.get("endpoint"))
        if model_cfg is not None:
            write_env_state(ev["endpoint"], model_key, model_cfg, relay)
            say(f"Saved endpoint info to {ENV_STATE_FILE} — run `python launch.py env` "
                "to print ready-to-paste settings for your AI tools.")
        print("\n" + "=" * 66)
        print("  YOUR ENDPOINT IS LIVE")
        print(f"  base URL : {ev['endpoint']}")
        print(f"  API key  : {ev['api_key']}")
        print(f"  model    : {ev['model']}   (context: {ev.get('max_model_len', '?')})")
        print("=" * 66)
        print("""
Try it:
  curl $BASE/chat/completions -H "Authorization: Bearer $KEY" \\
    -H "Content-Type: application/json" -d '{
      "model": "MODEL_NAME",
      "messages": [{"role": "user", "content": "Hello!"}]
    }'

See the README for hooking this into Claude Code, Codex CLI, opencode, etc.
""")
        say(f"The kernel keeps serving for up to {ev.get('keepalive_min', '?')} min. "
            "Ctrl-C here does NOT stop it; use `python launch.py stop`.")
    elif phase == "heartbeat":
        say(f"Still serving ({ev.get('up_min', '?')} min up) — {ev.get('endpoint', '')}")
    elif phase == "failed":
        say(f"FAILED at step {ev.get('step', '?')}.")
        if ev.get("tail"):
            print("--- last server output ---")
            print(ev["tail"])
        say("Full log: `python launch.py status` after the kernel exits, or the "
            "kernel page on kaggle.com.")
    else:
        text = PHASE_TEXT.get(phase)
        say(text if text else f"{phase} {json.dumps({k: v for k, v in ev.items() if k != 'phase'})}")


def watch(kernel, topic, relay=None, model_key=None, model_cfg=None,
          stop_after_ready=False):
    since = int(time.time()) - 600
    last_status = None
    seen_boot = False
    try:
        while True:
            for ts, ev in read_events(topic, since):
                since = max(since, ts)
                seen_boot = True
                render_event(ev, relay, model_key, model_cfg)
                if ev.get("phase") == "ready" and stop_after_ready:
                    return
                if ev.get("phase") in ("failed", "auto-shutdown", "stopped"):
                    return
            since = max(since, int(time.time()) - 1) if seen_boot else since
            r = ktl_common.run_kaggle("kernels", "status", kernel)
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
            status = m.group(1) if m else "UNKNOWN"
            if status != last_status:
                if status == "QUEUED":
                    say("Kaggle: queued — waiting for a TPU v5e-8 slot...")
                elif status == "RUNNING" and not seen_boot:
                    say("Kaggle: provisioning the VM and attaching datasets "
                        "(a few minutes)...")
                elif status in ("ERROR", "CANCELACKNOWLEDGED", "COMPLETE"):
                    say(f"Kernel finished with status {status}.")
                    return
                last_status = status
            time.sleep(30)
    except KeyboardInterrupt:
        say("Detached. The kernel keeps running — `python launch.py status` to "
            "re-attach, `python launch.py stop` to kill it.")


def cmd_build_env(args):
    """Maintainer flow. When the kernel finishes:
        kaggle kernels output <user>/<slug> -p bundle_out
        then create/version the dataset from bundle_out/bundle (see README)."""
    check_auth()
    user = kaggle_username(args.user)
    topic = "ktl-" + uuid.uuid4().hex[:20]
    model_cfg = MODELS["qwen"]  # build-env is Qwen-specific for now
    cfg = {"build_bundle": True, "ntfy_topic": topic, "weights_dataset": args.weights_dataset}
    src, _ = build_kernel(model_cfg, cfg)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "build_env.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{args.slug}", "title": args.slug, "code_file": "build_env.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "false", "enable_tpu": "true", "enable_internet": "true",
            "dataset_sources": [args.weights_dataset],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = ktl_common.run_kaggle("kernels", "push", "-p", str(td), timeout=1800)
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
    STATE_FILE.write_text(json.dumps({"kernel": f"{user}/{args.slug}", "topic": topic,
                                      "api_key": "", "model": "qwen"}))
    say(f"Pushed {user}/{args.slug}. It serves each config once (~1.5 h total) and "
        "leaves xla_cache.tar / cloudflared / manifest.json in its output.")
    watch(f"{user}/{args.slug}", topic)


def load_state():
    if not STATE_FILE.exists():
        sys.exit("No launch state found — run `python launch.py serve` first.")
    return json.loads(STATE_FILE.read_text())


def cmd_status(args):
    st = load_state()
    relay = relay_from_env(st["api_key"])
    model_key = st.get("model", "qwen")
    model_cfg = MODELS.get(model_key)
    say(f"Kernel: {st['kernel']}  (model: {model_key})")
    r = ktl_common.run_kaggle("kernels", "status", st["kernel"])
    say(((r.stdout or "") + (r.stderr or "")).strip())
    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    for _, ev in events[-8:]:
        render_event(ev, relay, model_key, model_cfg)
    if any(ev.get("phase") == "ready" for _, ev in events):
        say(f"API key: {st['api_key']}")
    if args.follow:
        watch(st["kernel"], st["topic"], relay, model_key, model_cfg)


def cmd_stop(args):
    st = load_state()
    say(f"Deleting kernel {st['kernel']} (terminates the TPU session)...")
    try:
        p = subprocess.run([str(c) for c in ktl_common.kaggle_cmd() +
                            ["kernels", "delete", st["kernel"]]],
                           input="yes\n", capture_output=True, text=True, timeout=120)
        say((p.stdout + p.stderr).strip() or "done")
    except subprocess.TimeoutExpired:
        say("kaggle delete timed out — the kernel may still be terminating; "
            "re-run `python launch.py stop`.")
