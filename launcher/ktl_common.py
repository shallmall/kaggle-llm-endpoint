"""Shared plumbing for the ktl toolchain (setup / doctor / env / serve).

Paths, the single config file (~/.ktl/config.json, KTL_HOME override), key
generation, secret masking/redaction, a subprocess helper that times out and
redacts, and prompt helpers that honour --yes / --dry-run.

Stdlib only, Python 3.10+.
"""
from __future__ import annotations

import copy
import json
import os
import re
import secrets as _secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants (pinned facts — sources in docs/setup-verification.md)
# ---------------------------------------------------------------------------
CONFIG_VERSION = 1

ENV_VAR = "KTL_CLIENT_API_KEY"
PLACEHOLDER = "<YOUR_CLIENT_API_KEY>"

# wrangler major we pin npx to (latest stable line, 2026-09-29: 4.143.0).
WRANGLER_MAJOR = 4
# wrangler 4's engines field requires Node >= 22.0.0 (npm registry, 2026-09-29).
NODE_MIN = (22, 0, 0)
# The Worker rejects secrets shorter than 32 chars (worker/worker.js);
# token_urlsafe(32) yields 43 chars, comfortably above the minimum.
KEY_BYTES = 32

STEP_NAMES = ("prereqs", "kaggle", "cloudflare", "secrets",
              "worker", "serve", "clients")

DEFAULT_WORKER_NAME = "kaggle-tpu-relay"   # matches worker/wrangler.jsonc

KAGGLE_TOKEN_URL = "https://www.kaggle.com/settings/api"
NODE_INSTALL_URL = "https://nodejs.org"


def say(msg):
    print(time.strftime("[%H:%M] "), msg, flush=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def ktl_home() -> Path:
    """Config/state directory: $KTL_HOME, else ~/.ktl."""
    override = os.environ.get("KTL_HOME", "").strip()
    return Path(override) if override else Path.home() / ".ktl"


def config_path() -> Path:
    return ktl_home() / "config.json"


def legacy_state_path() -> Path:
    """Pre-setup state file written by `serve`; migrated on first config read."""
    return ktl_home() / "state.json"


def log_path() -> Path:
    return ktl_home() / "setup.log"


# ---------------------------------------------------------------------------
# Config file — the single source of truth
# ---------------------------------------------------------------------------
DEFAULTS = {
    "version": CONFIG_VERSION,
    "model": "qwen",
    "kaggle": {"username": ""},
    "cloudflare": {"worker_name": "", "relay_url": ""},
    "secrets": {"client_api_key": "", "update_secret": ""},
    "steps": {s: "pending" for s in STEP_NAMES},
    "updated_at": "",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            d.chmod(0o700)
        except OSError:
            pass


def _atomic_write(path: Path, content: str) -> None:
    """temp file + rename so a crash never leaves a half-written file."""
    _ensure_dir(path.parent)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(content)
    if os.name == "posix":
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _kaggle_username_from_file() -> str:
    p = Path.home() / ".kaggle" / "kaggle.json"
    try:
        return str(json.loads(p.read_text()).get("username", "") or "")
    except (OSError, ValueError):
        return ""


def migrate_legacy_state(cfg: dict) -> dict:
    """Pull durable facts out of the legacy state.json into ``cfg`` (in place).

    Mapped: model, and — for relay-mode sessions — the relay URL. The direct
    quick-tunnel endpoint is ephemeral (changes every boot) and is NOT copied.
    The legacy file itself is left untouched."""
    st = _read_json(legacy_state_path())
    if not st:
        return cfg
    if st.get("model") in ("qwen", "glm"):
        cfg["model"] = st["model"]
    if st.get("mode") == "relay" and st.get("base_url"):
        cfg.setdefault("cloudflare", {})["relay_url"] = st["base_url"].rstrip("/")
    if not cfg.get("kaggle", {}).get("username"):
        user = _kaggle_username_from_file()
        if user:
            cfg.setdefault("kaggle", {})["username"] = user
    return cfg


def load_config(migrate: bool = True) -> dict:
    """Load config.json (missing/corrupt -> defaults).

    On first read (no config.json) the legacy state.json is migrated into a new
    config.json and saved, then left untouched afterwards. Callers apply
    precedence on top via resolve() — this returns config values over defaults."""
    p = config_path()
    stored = _read_json(p) if p.exists() else {}
    if migrate and not p.exists() and legacy_state_path().exists():
        cfg = copy.deepcopy(DEFAULTS)
        migrate_legacy_state(cfg)
        save_config(cfg)
        stored = _read_json(p)
    return _deep_merge(DEFAULTS, stored)


def save_config(cfg: dict) -> None:
    cfg = dict(cfg)
    cfg["version"] = CONFIG_VERSION
    cfg["steps"] = _deep_merge(DEFAULTS["steps"], cfg.get("steps", {}))
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write(config_path(), json.dumps(cfg, indent=2) + "\n")


def update_config(**partial) -> dict:
    """Load, apply top-level key updates, save, return the new config."""
    cfg = load_config(migrate=False)
    for k, v in partial.items():
        cfg[k] = v
    save_config(cfg)
    return cfg


def set_step(cfg: dict, step: str, status: str) -> None:
    cfg.setdefault("steps", {})[step] = status


def step_status(cfg: dict, step: str) -> str:
    return cfg.get("steps", {}).get(step, "pending")


def resolve(cli, env, config_value, default=None):
    """Precedence everywhere: CLI flag > environment variable > config.json > default."""
    for v in (cli, env, config_value, default):
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def client_key_from_config(cfg: dict, env_key: str = "") -> tuple:
    """(key, source) for relay-mode clients: env var wins over config.json."""
    if env_key.strip():
        return env_key, "env"
    key = cfg.get("secrets", {}).get("client_api_key", "")
    if key:
        return key, "config"
    return PLACEHOLDER, "placeholder"


# ---------------------------------------------------------------------------
# Keys + masking
# ---------------------------------------------------------------------------
def gen_key() -> str:
    return _secrets.token_urlsafe(KEY_BYTES)


def mask(value: str) -> str:
    """'abcd…wxyz' style display. Short values are fully hidden."""
    if not value:
        return "(none)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]}"


_SECRET_NAME = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*?(?:TOKEN|KEY|SECRET|PASSWORD|PASSWD)[A-Za-z0-9_]*=)(\S+)")


def redact(text: str) -> str:
    """Mask secret-looking NAME=value tokens (dotenv-style lines, incl. diff
    context lines) so subprocess output and logs never carry credentials."""
    return _SECRET_NAME.sub(lambda m: m.group(1) + mask(m.group(2)), text or "")


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
class SubprocessError(RuntimeError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg + (f"\nHint: {hint}" if hint else ""))
        self.hint = hint


def hint_for_output(out: str) -> str:
    o = (out or "").lower()
    checks = (
        ("not logged in", "Not logged in to Cloudflare — run: python launch.py setup --only cloudflare"),
        ("login required", "Not logged in to Cloudflare — run: python launch.py setup --only cloudflare"),
        ("waiting for authorization code",
         "Cloudflare login timed out (OAuth callback to localhost:8976 never returned). "
         "On a remote/headless machine use an API token: export CLOUDFLARE_API_TOKEN=… "
         "CLOUDFLARE_ACCOUNT_ID=… (token: dash.cloudflare.com/profile/api-tokens)"),
        ("unauthorized", "Authentication failed — check your keys (relay: python launch.py setup --rotate)"),
        ("401", "Authentication failed (401) — check keys (relay: python launch.py setup --rotate)"),
        ("403", "Forbidden (403) — this account has no access to that Worker"),
        ("enotfound", "DNS lookup failed — check network connectivity and the URL"),
        ("econnrefused", "Connection refused — the endpoint is not reachable"),
        ("timed out", "Timed out — the endpoint may still be booting (python launch.py env --test --wait 600)"),
        ("requires node", "wrangler needs Node >= 22 — install from https://nodejs.org"),
        ("engines", "Node too old for wrangler 4 (needs >= 22) — install from https://nodejs.org"),
        ("quota", "Kaggle quota exceeded — free TPU hours reset weekly (kaggle.com/account)"),
        ("kaggle.com/settings", "Kaggle credentials missing or expired — create a token at " + KAGGLE_TOKEN_URL),
    )
    for needle, hint in checks:
        if needle in o:
            return hint
    return ""


def run_logged(cmd: list, *, stdin: str | None = None, timeout: int = 300,
               cwd: str | Path | None = None, inherit_stdio: bool = False,
               log_line=None, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run ``cmd`` with a timeout; captured output is redacted before it is
    echoed/logged. Raises SubprocessError (with a friendly hint) on failure.

    ``log_line(text)`` optionally receives each redacted output line. With
    ``inherit_stdio`` the child shares the terminal (interactive flows like
    `wrangler login` that open a browser). ``quiet`` suppresses the echo
    (for read-only probes whose output would pollute structured output)."""
    def _log(text: str):
        if log_line:
            log_line(text)

    if inherit_stdio:
        _log("$ " + " ".join(str(c) for c in cmd))
        r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                           timeout=timeout)
        if r.returncode != 0:
            raise SubprocessError(f"`{' '.join(str(c) for c in cmd)}` exited "
                                  f"{r.returncode}", hint_for_output(""))
        return r

    try:
        r = subprocess.run([str(c) for c in cmd], input=stdin,
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(cwd) if cwd else None)
    except subprocess.TimeoutExpired as e:
        raise SubprocessError(
            f"`{cmd[0]}` timed out after {timeout}s",
            "The service may still be starting — re-run the step or check `doctor`.") from e
    out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    for line in redact(out).splitlines():
        if line.strip():
            if not quiet:
                print(line, flush=True)
            _log(line)
    if r.returncode != 0:
        raise SubprocessError(
            f"`{' '.join(str(c) for c in cmd)}` failed (exit {r.returncode})",
            hint_for_output(out))
    return r


# ---------------------------------------------------------------------------
# Tool resolution (Windows-safe: shutil.which, no hardcoded paths)
# ---------------------------------------------------------------------------
def kaggle_cmd() -> list:
    """Prefer a `kaggle` executable on PATH; fall back to the module of the
    current interpreter (pip-installed case)."""
    p = shutil.which("kaggle")
    if p:
        return [p]
    return [sys.executable, "-m", "kaggle"]


def run_kaggle(*args, capture=True, timeout=300) -> subprocess.CompletedProcess:
    """Run the kaggle CLI with a sane timeout so a wedged/offline CLI can't
    hang doctor/wizard/status indefinitely (the CLI has ~30 s internal
    timeouts, but a stuck process is possible). A timeout or a missing/corrupt
    binary becomes a synthetic failure (returncode -9 / -1) that callers treat
    like any other non-zero exit. Callers doing a big upload (kernel push)
    pass a larger timeout."""
    cmd = [str(c) for c in kaggle_cmd() + list(args)]
    try:
        return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            list(args), returncode=-9, stdout="", stderr="kaggle CLI timed out")
    except OSError as e:
        return subprocess.CompletedProcess(
            list(args), returncode=-1, stdout="", stderr=f"kaggle CLI not runnable: {e}")


def kaggle_version() -> str:
    """CLI version string, or '' when the kaggle CLI is absent/unrunnable."""
    try:
        r = run_kaggle("--version", capture=True)
        if r.returncode != 0:
            return ""
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        return m.group(1) if m else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def kaggle_authed() -> tuple:
    """Cheap authenticated probe (list one of your own kernels). -> (ok, detail)."""
    r = run_kaggle("kernels", "list", "-m", "--page-size", "1", capture=True)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode == 0:
        return True, "ok"
    return False, redact(out)[:400]


def kaggle_username(cli_arg: str | None = None) -> str:
    """Detect the Kaggle username ('' when unknown)."""
    if cli_arg:
        return cli_arg
    user = _kaggle_username_from_file()
    if user:
        return user
    r = run_kaggle("config", "view", capture=True)
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    return ""


def node_version() -> tuple | None:
    p = shutil.which("node")
    if not p:
        return None
    try:
        r = subprocess.run([p, "--version"], capture_output=True, text=True,
                           timeout=15)
        m = re.search(r"v(\d+)\.(\d+)\.(\d+)", r.stdout or "")
        return tuple(int(x) for x in m.groups()) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def npx_cmd() -> str | None:
    """npx ships with Node; on Windows this resolves to npx.cmd."""
    return shutil.which("npx")


def wrangler_cmd(*args) -> list:
    npx = npx_cmd()
    if not npx:
        raise SubprocessError("npx not found",
                              f"Install Node >= {NODE_MIN[0]} from {NODE_INSTALL_URL}")
    return [npx, "-y", f"wrangler@{WRANGLER_MAJOR}", *[str(a) for a in args]]


# ---------------------------------------------------------------------------
# Prompts (--yes / --dry-run aware)
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    yes: bool = False
    dry_run: bool = False
    verbose: bool = False
    log_lines: list = field(default_factory=list)

    def log(self, text: str):
        self.log_lines.append(text)


def confirm(ctx: Ctx, prompt: str, default: bool = False) -> bool:
    if ctx.dry_run:
        ctx.log(f"[dry-run] would ask: {prompt}")
        return default
    if ctx.yes:
        ctx.log(f"auto-approved (--yes): {prompt}")
        return True
    try:
        ans = input(f"{prompt} [{'y/N' if not default else 'Y/n'}] ").strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


def ask(ctx: Ctx, prompt: str, default: str = "") -> str:
    if ctx.dry_run:
        ctx.log(f"[dry-run] would ask: {prompt}")
        return default
    if ctx.yes:
        ctx.log(f"auto-answered (--yes): {prompt} -> {default or '(none)'}")
        return default
    try:
        ans = input(f"{prompt} [{default}] ").strip()
    except EOFError:
        return default
    return ans or default


def choose(ctx: Ctx, prompt: str, options: list) -> list:
    """Multi-select: prints numbered options, returns the chosen subset (in
    original order). --yes selects everything; --dry-run selects nothing."""
    if not options:
        return []
    if ctx.dry_run:
        ctx.log(f"[dry-run] would ask: {prompt}")
        return []
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    if ctx.yes:
        ctx.log(f"auto-selected all (--yes): {prompt}")
        return list(options)
    try:
        raw = input(f"{prompt} (numbers, comma-separated, or 'all') ").strip()
    except EOFError:
        return []
    if raw.lower() in ("all", "*"):
        return list(options)
    picked = []
    for part in re.split(r"[,\s]+", raw):
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= len(options):
            picked.append(options[int(part) - 1])
    return picked


def wait_for_enter(ctx: Ctx, prompt: str) -> None:
    if ctx.dry_run:
        ctx.log(f"[dry-run] would wait: {prompt}")
        return
    try:
        input(prompt)
    except EOFError:
        pass


# ---------------------------------------------------------------------------
# Git-tree secret guard
# ---------------------------------------------------------------------------
def _git(*args, cwd) -> str:
    if not shutil.which("git"):
        return ""
    try:
        r = subprocess.run(["git", *[str(a) for a in args]], cwd=str(cwd),
                           capture_output=True, text=True, timeout=10)
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def inside_git_worktree(path) -> bool:
    d = Path(path).expanduser()
    if not d.exists():
        d = d.parent
    return _git("rev-parse", "--is-inside-work-tree", cwd=d) == "true"


def gitignore_guard(ctx: Ctx, target: str, literal_value: str) -> None:
    """Before a literal secret lands in a file inside a git working tree, warn
    and offer to gitignore it (never silently leave a commit-able secret)."""
    if not literal_value or literal_value == PLACEHOLDER:
        return
    if not inside_git_worktree(target):
        return
    d = Path(target).expanduser()
    if not d.is_dir():
        d = d.parent
    root = _git("rev-parse", "--show-toplevel", cwd=d)
    if not root:
        return
    t = Path(target).expanduser()
    r = Path(root)
    try:
        rel = t.relative_to(r).as_posix()
    except ValueError:
        rel = t.name
    print(f"\nwarning: a literal key will be written to {rel}, which is inside a "
          f"git working tree ({root}).")
    offer = f"Add {rel} to {r / '.gitignore'}?"
    if ctx.dry_run:
        ctx.log(f"[dry-run] {offer}")
        return
    if ctx.yes or confirm(ctx, offer, default=True):
        gi = r / ".gitignore"
        entries = gi.read_text().splitlines() if gi.exists() else []
        if rel not in entries:
            with open(gi, "a") as f:
                f.write(("\n" if entries and entries[-1].strip() else "") + rel + "\n")
            print(f"added {rel} to .gitignore")
    else:
        print(f"noted — remember to keep {rel} out of version control.")


# ---------------------------------------------------------------------------
# Model catalog (shared by ktl_serve [kernel fields] and ktl_env [api fields])
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
        "engine_dir": None,
        # env-command / client-config data (read by ktl_env, never hardcoded there)
        "api_model": "qwen3.8-27b",
        "context": 262144,
        "max_output": 65536,
        "cost": {"input": 0.45, "output": 3.2, "cache_read": 0.05},
        # Does this backend serve POST /v1/responses (what Codex wire_api="responses"
        # needs)? vLLM exposes the OpenAI Responses API -> True (confirm with a live
        # `ktl env --test`). The GLM engine does not (404) -> see below.
        "responses_api": True,
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
        # the kernel carries an __ENGINE__ slot: the glm53 package ships inside
        # the pushed script (base64 tar.gz), so the kernel needs no glm53/ folder
        "engine_dir": HERE / "glm53-flash" / "engine" / "glm53",
        # env-command / client-config data (read by ktl_env, never hardcoded there)
        "api_model": "glm-5.3-flash",
        "context": 262144,
        "max_output": 65536,
        "cost": {"input": 0.15, "output": 0.5, "cache_read": 0.03},
        # The GLM engine's do_POST only handles /v1/{models,messages,
        # chat/completions,completions}; anything else (incl. /v1/responses)
        # falls through to 404. So Codex (wire_api="responses") CANNOT work on
        # GLM. Source-confirmed in kernel/serve_glm53.py.
        "responses_api": False,
    },
}
