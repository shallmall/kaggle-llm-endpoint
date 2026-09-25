#!/usr/bin/env python3
"""
kaggle-tpu-lab launcher — serve LLMs on a free Kaggle TPU from your terminal.

    python launch.py serve                          # Qwen3.8-27B (default)
    python launch.py serve --model glm              # GLM-5.3-Flash
    python launch.py serve --model glm --reasoning-effort low
    python launch.py status                         # one-shot status + recent events
    python launch.py stop                           # kill the TPU session

Requires the Kaggle CLI, authenticated:  pip install kaggle   (see README).
Only the Python standard library is used here.
"""
import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"

# ---------------------------------------------------------------------------
# Model-specific configuration
# ---------------------------------------------------------------------------
MODELS = {
    "qwen": {
        "kernel_src": HERE / "qwen38-27b" / "kernel" / "serve_qwen38.py",
        "kernel_name": "serve_qwen38.py",
        "default_slug": "qwen38-tpu-serve",
        "served_model_name": "qwen3.8-27b",
        "dataset_sources": [
            "rahim3/qwen3-8-27b-bf16",
            "rahim3/qwen38-tpu-env-v5e8",
        ],
        "weights_dataset": "rahim3/qwen3-8-27b-bf16",
        "env_dataset": "rahim3/qwen38-tpu-env-v5e8",
    },
    "glm": {
        "kernel_src": HERE / "glm53-flash" / "kernel" / "serve_glm53.py",
        "kernel_name": "serve_glm53.py",
        "default_slug": "glm53-tpu-serve",
        "served_model_name": "glm-5.3-flash",
        "dataset_sources": [
            "rahim3/glm53-flash-iq3xxs-1",
            "rahim3/glm53-flash-iq3xxs-2",
            "rahim3/glm53-flash-serve",
        ],
        "weights_dataset": None,
        "env_dataset": None,
    },
}

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


def kaggle(*args, capture=True):
    cmd = [sys.executable, "-m", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=capture, text=True)
    return r


def say(msg):
    print(time.strftime("[%H:%M] "), msg, flush=True)


def check_auth():
    r = kaggle("kernels", "list", "-m", "--page-size", "1")
    if r.returncode != 0:
        sys.exit("Kaggle CLI is not working or not authenticated.\n"
                 "Install with `pip install kaggle`, then put your API token in place\n"
                 "(https://www.kaggle.com/settings -> Create New Token).\n\n"
                 f"Error was:\n{(r.stderr or r.stdout).strip()}")


def kaggle_username(cli_arg):
    if cli_arg:
        return cli_arg
    r = kaggle("config", "view")
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    sys.exit("Could not detect your Kaggle username — pass it with --user <name>.")


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
    model_cfg = MODELS[args.model]
    slug = args.slug or model_cfg["default_slug"]
    topic = "ktl-" + uuid.uuid4().hex[:20]
    # KTL_API_KEY lets you pin a permanent key instead of getting a fresh
    # random one every boot. Falls back to the original random behavior.
    api_key = os.environ.get("KTL_API_KEY") or "sk-" + secrets.token_hex(16)

    # --- Option A: named Cloudflare Tunnel -> permanent hostname, no relay needed.
    # Set both or neither. See SETUP.md.
    tunnel_token = os.environ.get("KTL_TUNNEL_TOKEN", "")
    tunnel_hostname = os.environ.get("KTL_TUNNEL_HOSTNAME", "")
    if tunnel_token and not tunnel_hostname:
        sys.exit("KTL_TUNNEL_TOKEN is set but KTL_TUNNEL_HOSTNAME is not — set both (see SETUP.md).")

    # --- Option B: Cloudflare Worker relay -> permanent URL in front of the
    # quick tunnel, auto-updated on every boot. Set both or neither.
    relay_url = os.environ.get("KTL_RELAY_URL", "").rstrip("/")
    relay_secret = os.environ.get("KTL_RELAY_UPDATE_SECRET", "")
    relay = None
    if relay_url and relay_secret:
        relay = {"url": relay_url, "secret": relay_secret, "api_key": api_key}
    elif relay_url or relay_secret:
        sys.exit("Set both KTL_RELAY_URL and KTL_RELAY_UPDATE_SECRET, or neither (see SETUP.md).")

    # Build the CFG dict that gets injected into the kernel script.
    cfg = {
        "ntfy_topic": topic,
        "api_key": api_key,
        "served_model_name": model_cfg["served_model_name"],
        "keepalive_min": args.keepalive_min,
    }

    # Model-specific config keys
    if args.model == "qwen":
        cfg["max_model_len"] = args.max_model_len
        cfg["max_num_seqs"] = args.max_num_seqs
        cfg["mtp_tokens"] = args.mtp
        cfg["async_scheduling"] = False
        cfg["reasoning_effort_default"] = args.reasoning_effort
        cfg["weights_dataset"] = args.weights_dataset or model_cfg["weights_dataset"]
        if tunnel_token:
            cfg["tunnel_token"] = tunnel_token
            cfg["tunnel_hostname"] = tunnel_hostname
        if args.no_tools:
            cfg["tool_call_parser"] = ""
        if args.text_only:
            cfg["text_only"] = True
        if args.verbose:
            cfg["verbose"] = True
        if args.fast_start:
            cfg["fast_start"] = True
    elif args.model == "glm":
        cfg["reasoning_effort_default"] = args.reasoning_effort
        if args.max_model_len:
            cfg["max_len"] = args.max_model_len
        if args.streams:
            cfg["streams"] = args.streams
        if args.vision is not None:
            cfg["vision"] = args.vision
        if args.verbose:
            cfg["verbose"] = True

    src = model_cfg["kernel_src"].read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{model_cfg['kernel_name']} is missing the __LAUNCHER_CONFIG__ line")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / model_cfg["kernel_name"]).write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": model_cfg["kernel_name"],
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "dataset_sources": model_cfg["dataset_sources"],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Pushing kernel {user}/{slug} ({args.model}, TPU v5e-8)...")
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"WARNING: {line.strip()} — the kernel will still run, "
                    "but may need to download weights / compile cold.")

    STATE_FILE.write_text(json.dumps(
        {"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key,
         "model": args.model}))
    say("Pushed. Kaggle takes a few minutes to provision the TPU and attach the "
        "datasets; the endpoint is usually live ~22 min after the kernel starts.")
    say("Watching progress (Ctrl-C is safe — the server keeps running; "
        "`python launch.py status` re-attaches, `... stop` kills it).")
    watch(f"{user}/{slug}", topic, relay)


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
    relay_url = os.environ.get("KTL_RELAY_URL", "").rstrip("/")
    relay_secret = os.environ.get("KTL_RELAY_UPDATE_SECRET", "")
    if relay_url and relay_secret:
        return {"url": relay_url, "secret": relay_secret, "api_key": api_key}
    return None


def render_event(ev, relay=None):
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
            f"(sanity: {ev.get('sanity', '')!r})")
    elif phase == "loading":
        say(f"Loading weights ({ev.get('note', '')})...")
    elif phase == "loaded":
        say(f"Weights loaded in {ev.get('minutes', '?')} min (HBM {ev.get('hbm_gb', '?')} GB/chip).")
    elif phase == "warmed":
        say(f"Warm-up done in {ev.get('minutes', '?')} min (HBM {ev.get('hbm_gb', '?')} GB/chip).")
    elif phase == "ready":
        if not ev.get("endpoint"):
            say("Server is healthy, but no public endpoint was created; relay registration was skipped.")
            return
        register_with_relay(relay, ev.get("endpoint"))
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


def watch(kernel, topic, relay=None):
    since = int(time.time()) - 600
    last_status = None
    seen_boot = False
    try:
        while True:
            for ts, ev in read_events(topic, since):
                since = max(since, ts)
                seen_boot = True
                render_event(ev, relay)
                if ev.get("phase") in ("failed", "auto-shutdown", "stopped"):
                    return
            since = max(since, int(time.time()) - 1) if seen_boot else since
            r = kaggle("kernels", "status", kernel)
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
    src = model_cfg["kernel_src"].read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{model_cfg['kernel_name']} is missing the __LAUNCHER_CONFIG__ line")
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
        r = kaggle("kernels", "push", "-p", str(td))
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
    say(f"Kernel: {st['kernel']}  (model: {st.get('model', 'qwen')})")
    r = kaggle("kernels", "status", st["kernel"])
    say(((r.stdout or "") + (r.stderr or "")).strip())
    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    for _, ev in events[-8:]:
        render_event(ev, relay)
    if any(ev.get("phase") == "ready" for _, ev in events):
        say(f"API key: {st['api_key']}")
    if args.follow:
        watch(st["kernel"], st["topic"], relay)


def cmd_stop(args):
    st = load_state()
    say(f"Deleting kernel {st['kernel']} (terminates the TPU session)...")
    p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "delete",
                        st["kernel"]], input="yes\n", capture_output=True, text=True)
    say((p.stdout + p.stderr).strip() or "done")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- serve ---
    s = sub.add_parser("serve", help="push the serving kernel and watch it come up")
    s.add_argument("--model", choices=["qwen", "glm"], default="qwen",
                   help="which model to serve (default: qwen)")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default=None,
                   help="kernel name (default: per-model slug)")
    s.add_argument("--max-model-len", type=int, default=None,
                   help="context length (Qwen default: 262144; GLM default: 262144)")
    s.add_argument("--max-num-seqs", type=int, default=4,
                   help="max concurrent sequences (Qwen only)")
    s.add_argument("--streams", type=int, default=None,
                   help="concurrent decode streams (GLM only)")
    s.add_argument("--mtp", type=int, default=3,
                   help="MTP speculative tokens (Qwen only; 0 disables)")
    s.add_argument("--reasoning-effort", default=None,
                   help="server-side default reasoning effort "
                        "(Qwen: xhigh|medium|low; GLM: low|high)")
    s.add_argument("--keepalive-min", type=int, default=480,
                   help="auto-shutdown after this many minutes of serving")
    s.add_argument("--weights-dataset", default=None,
                   help="override the weights dataset (Qwen only)")
    s.add_argument("--no-tools", action="store_true",
                   help="disable tool-calling support (Qwen only)")
    s.add_argument("--text-only", action="store_true",
                   help="skip the vision tower (Qwen only)")
    s.add_argument("--vision", type=lambda x: x.lower() in ("true", "1", "yes"),
                   default=None,
                   help="enable/disable vision tower (GLM only)")
    s.add_argument("--verbose", action="store_true",
                   help="show verbose log lines in the kernel log")
    s.add_argument("--fast-start", action="store_true",
                   help="skip TPU graph precompile (Qwen only)")
    s.set_defaults(fn=cmd_serve)

    # --- build-env (Qwen maintainer flow) ---
    s = sub.add_parser("build-env", help="(maintainers) push a kernel that builds the "
                       "env dataset: venv + XLA cache + cloudflared")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-env-bundle")
    s.add_argument("--weights-dataset", default="rahim3/qwen3-8-27b-bf16")
    s.set_defaults(fn=cmd_build_env)

    # --- status ---
    s = sub.add_parser("status", help="show current kernel status + recent events")
    s.add_argument("--follow", "-f", action="store_true", help="keep watching")
    s.set_defaults(fn=cmd_status)

    # --- stop ---
    s = sub.add_parser("stop", help="terminate the TPU session")
    s.set_defaults(fn=cmd_stop)

    args = ap.parse_args()

    # Set model-specific defaults for reasoning_effort
    if args.cmd == "serve" and args.reasoning_effort is None:
        args.reasoning_effort = "xhigh" if args.model == "qwen" else "low"
    if args.cmd == "serve" and args.max_model_len is None:
        args.max_model_len = 262144

    args.fn(args)


if __name__ == "__main__":
    main()
