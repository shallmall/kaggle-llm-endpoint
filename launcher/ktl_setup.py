"""
Guided setup wizard (`python launch.py setup`) and read-only health check
(`python launch.py doctor`).

The wizard takes a fresh machine from clone to a working, tested endpoint.
Each step is idempotent and resumable; progress lives in ~/.ktl/config.json
(see ktl_common). Only the Python standard library is used.

Design notes:
  - Subprocesses run through _run(), which honours --dry-run (prints, never
    executes) and --yes (prompts are auto-approved but still logged).
  - Secrets are generated locally (token_urlsafe(32)) and sent to wrangler
    on STDIN only — never argv, never logs (ktl_common.redact as backstop).
  - doctor() has ZERO side effects: it never writes config or state, it only
    prints a redacted report, and exits 0 only when everything is healthy.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import ktl_common
import ktl_env
import ktl_serve
from ktl_common import Ctx, SubprocessError, confirm, say

WORKER_DIR = ktl_common.HERE.parent / "worker"

WIZARD_STEPS = ("prereqs", "kaggle", "cloudflare", "secrets",
                "worker", "serve", "verify", "clients")

_WORKER_URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.workers\.dev")

# (binary on PATH, ktl_env client id)
_CLIENT_BINARIES = (("claude", "claude-code"), ("codex", "codex"),
                    ("opencode", "opencode"), ("hermes", "hermes"),
                    ("aider", "aider"))
_CLIENT_CMD = {
    "claude-code": "claude",
    "codex": "codex -c model_provider=kaggle-tpu -c model=<MODEL>",
    "opencode": "opencode",
    "hermes": "hermes   (alt: hermes -m <MODEL>)",
    "aider": "aider   (use model openai/<MODEL>)",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _DryResult:
    """Stand-in for subprocess.CompletedProcess under --dry-run."""
    stdout = ""
    stderr = ""
    returncode = 0


def _run(ctx: Ctx, cmd: list, **kw):
    """Run an external command, honouring --dry-run (log only, never execute).

    All stdout/stderr is redacted by ktl_common.run_logged; the (redacted)
    lines are appended to ctx.log_lines, which the wizard appends to
    ~/.ktl/setup.log after each completed step.
    """
    if ctx.dry_run:
        line = "[dry-run] would run: " + " ".join(shlex.quote(str(c)) for c in cmd)
        ctx.log(line)
        print("        " + line)
        return _DryResult()
    return ktl_common.run_logged(list(cmd), log_line=ctx.log, **kw)


def _save(ctx: Ctx, cfg: dict):
    """Persist progress (no-op under --dry-run) and flush the log lines."""
    if ctx.dry_run:
        return
    ktl_common.save_config(cfg)
    if ctx.log_lines:
        p = ktl_common.log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for line in ctx.log_lines:
                f.write(ktl_common.redact(line) + "\n")
        ctx.log_lines.clear()


def _whoami(quiet: bool = False) -> str:
    r = ktl_common.run_logged(ktl_common.wrangler_cmd("whoami"), timeout=180,
                              quiet=quiet)
    return r.stdout or ""


def _parse_whoami(out: str) -> str:
    """Extract the account name from `wrangler whoami`. Handles both the plain
    `Account Name: <name>` line (piped / older wrangler) and the box-drawing
    table wrangler 4.x prints (the first data row after the header)."""
    for line in out.splitlines():
        m = re.search(r"Account Name:\s*(.*?)\s*$", line)
        if m:
            return m.group(1).strip()
    cells = re.findall(r"^\s*│\s*(.*?)\s*│\s*(.*?)\s*│\s*$", out, re.MULTILINE)
    for name, _acct in cells:
        nm = name.strip()
        if nm not in ("Account Name", "Account ID"):
            return nm
    m = re.search(r"[Ll]ogged in as:?\s*(\S+)", out)
    return m.group(1).strip() if m else ""


def _kaggle_credential_source() -> str | None:
    """Which credential mechanism is present (never returns the value)."""
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return "KAGGLE_API_TOKEN env var"
    if (Path.home() / ".kaggle" / "access_token").exists():
        return "~/.kaggle/access_token"
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return "~/.kaggle/kaggle.json (legacy credentials file)"
    return None


def _worker_name(cfg: dict, args) -> str:
    return ktl_common.resolve(getattr(args, "worker_name", None), None,
                              cfg.get("cloudflare", {}).get("worker_name", ""),
                              ktl_common.DEFAULT_WORKER_NAME)


def _resolved_for(cfg: dict, client: str = "setup"):
    """Build a ktl_env.Resolved for the current config.

    Same precedence as `launch.py env`: base URL cli-flag > KTL_RELAY_URL >
    config.json > legacy serve state; relay mode uses the relay client key
    (env > config.json), direct mode uses KTL_API_KEY > serve-state key.
    Returns None when no endpoint is known yet.
    """
    st = ktl_env._legacy_state()
    relay_cfg_url = (cfg.get("cloudflare", {}).get("relay_url", "") or "").strip()
    env_url = os.environ.get("KTL_RELAY_URL", "").strip()

    if env_url:
        base_url, url_source = ktl_env.normalize_root(env_url), "env"
    elif relay_cfg_url:
        base_url, url_source = ktl_env.normalize_root(relay_cfg_url), "config"
    elif st.get("base_url"):
        base_url, url_source = ktl_env.normalize_root(st["base_url"]), "state"
    else:
        return None

    if env_url or relay_cfg_url:
        mode = "relay"
    elif st.get("mode") in ("relay", "direct"):
        mode = st["mode"]
    else:
        mode = "direct"

    if mode == "relay":
        key, src = ktl_common.client_key_from_config(
            cfg, os.environ.get(ktl_common.ENV_VAR, "").strip())
        key_source = {"env": "env (relay)", "config": "config (relay)"}[src] \
            if src != "placeholder" else "placeholder"
        client_key, key_ph = key, src == "placeholder"
    else:  # direct
        env_key = os.environ.get("KTL_API_KEY", "").strip()
        if env_key:
            client_key, key_source, key_ph = env_key, "env (direct)", False
        elif ktl_env._SERVE_STATE_FILE.exists():
            try:
                saved = json.loads(ktl_env._SERVE_STATE_FILE.read_text()).get("api_key", "")
            except Exception:
                saved = ""
            if saved:
                client_key, key_source, key_ph = saved, "state (direct)", False
            else:
                client_key, key_source, key_ph = ktl_env.PLACEHOLDER, "placeholder", True
        else:
            client_key, key_source, key_ph = ktl_env.PLACEHOLDER, "placeholder", True

    model_key = cfg.get("model", "qwen")
    mc = ktl_common.MODELS[model_key]
    return ktl_env.Resolved(
        client=client, model_key=model_key, api_model=mc["api_model"],
        context=mc["context"], max_output=mc["max_output"], cost=mc["cost"],
        base_url=base_url, client_key=client_key, key_is_placeholder=key_ph,
        key_source=key_source, url_source=url_source, shell=ktl_env.detect_shell(),
        reveal=False, responses_api=mc.get("responses_api", True))


def _http_get_json(url: str, key: str, timeout: int = 8):
    """GET with a bearer token -> (status_code_or_None, body_text)."""
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + key,
                      "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # URLError, timeout, DNS, ...
        return None, str(e)


# --------------------------------------------------------------------------
# wizard steps
# --------------------------------------------------------------------------

def _step_prereqs(ctx: Ctx, cfg: dict, args):
    py = sys.version_info
    if (py.major, py.minor) < (3, 10):
        sys.exit(f"Python 3.10+ is required (you have {py.major}.{py.minor}).")
    if (py.major, py.minor) < (3, 11):
        say(f"note  : kaggle CLI now recommends Python 3.11+ — this interpreter "
            f"is {py.major}.{py.minor}; if `pip install kaggle` fails, upgrade Python.")

    if ctx.dry_run:
        ctx.log("[dry-run] would check: kaggle CLI on PATH (pip install kaggle if missing)")
        ctx.log(f"[dry-run] would check: Node.js >= v{'.'.join(str(x) for x in ktl_common.NODE_MIN)} + npx")
        say(f"ok    : prerequisites (dry run — {py.major}.{py.minor} checked, "
            "kaggle/Node checks skipped)")
        return

    # kaggle CLI
    if shutil.which("kaggle") or ktl_common.kaggle_version():
        say(f"ok    : kaggle CLI {ktl_common.kaggle_version()} ({ktl_common.kaggle_cmd()[0]})")
    else:
        print("missing : kaggle CLI (Kaggle login + TPU kernel push)")
        print(f"install : {sys.executable} -m pip install kaggle")
        if ctx.dry_run:
            ctx.log("[dry-run] would offer to run: pip install kaggle")
        elif confirm(ctx, "Install the kaggle CLI now?", default=True):
            _run(ctx, [sys.executable, "-m", "pip", "install", "kaggle"], timeout=600)
            if not ktl_common.kaggle_version():
                sys.exit("kaggle CLI is still not on PATH after install — "
                         "open a new shell and re-run setup.")
            say(f"ok    : kaggle CLI {ktl_common.kaggle_version()}")
        else:
            sys.exit("kaggle CLI is required. Install it (command above), "
                     "then re-run: python launch.py setup")

    # Node.js + npx (needed for the Cloudflare Worker relay; never auto-installed)
    need = ".".join(str(x) for x in ktl_common.NODE_MIN)
    nv = ktl_common.node_version()
    if nv is None:
        print(f"missing : Node.js (>= v{need} required for wrangler {ktl_common.WRANGLER_MAJOR})")
        print(f"install : {ktl_common.NODE_INSTALL_URL}   (pick the LTS build for your OS)")
        sys.exit("Node.js is required for the Cloudflare Worker relay. "
                 "Install it, then re-run: python launch.py setup")
    if nv < ktl_common.NODE_MIN:
        sys.exit(f"Node is too old: v{'.'.join(str(x) for x in nv)} < v{need}. "
                 f"Install a newer LTS from {ktl_common.NODE_INSTALL_URL}.")
    npx = ktl_common.npx_cmd()
    if not npx:
        sys.exit("npx was not found next to node — reinstall Node from nodejs.org.")
    say(f"ok    : Node v{'.'.join(str(x) for x in nv)} (>= {need}), npx {npx}")


def _step_kaggle(ctx: Ctx, cfg: dict, args):
    cred = _kaggle_credential_source()
    if ctx.dry_run:
        if cred:
            say(f"found : Kaggle credentials via {cred} (dry run — auth check skipped)")
        else:
            say("missing: Kaggle credentials (dry run — would run: kaggle auth login)")
        return

    if cred:
        say(f"found : Kaggle credentials via {cred}")
    else:
        print("No Kaggle credentials found. Options:")
        print("  A) OAuth (opens a browser):   kaggle auth login")
        print(f"  B) API token: create one at   {ktl_common.KAGGLE_TOKEN_URL}")
        print("     then either export KAGGLE_API_TOKEN=<token>, or save it to")
        print("     ~/.kaggle/access_token (or the legacy ~/.kaggle/kaggle.json).")
        if confirm(ctx, "Run `kaggle auth login` now? (opens a browser window)",
                   default=False):
            _run(ctx, ktl_common.kaggle_cmd() + ["auth", "login"],
                 inherit_stdio=True, timeout=600)
        else:
            ktl_common.wait_for_enter(
                ctx, "Set up your Kaggle credentials, then press Enter to continue... ")

    ok, detail = ktl_common.kaggle_authed()
    if not ok:
        print("Kaggle authentication check failed:")
        for ln in detail.splitlines():
            print("  " + ln)
        sys.exit("Fix your Kaggle credentials (https://www.kaggle.com/settings/api — "
                 "TPU access also requires a phone-verified account), then re-run: "
                 "python launch.py setup")

    user = cfg.get("kaggle", {}).get("username") or ktl_common.kaggle_username()
    if not user:
        user = ktl_common.ask(
            ctx, "Could not detect your Kaggle username — enter it "
                 "(it is in your kaggle.com profile URL): ", default="")
        if not user:
            sys.exit("Kaggle username is required (your kaggle.com profile URL).")
    cfg.setdefault("kaggle", {})["username"] = user
    say(f"ok    : Kaggle authenticated as {user}")


def _device_login(ctx: Ctx) -> str:
    """OAuth 2.0 Device Authorization Grant (RFC 8628). This is the ONLY login
    the wizard uses: wrangler prints a verification URL and a one-time code,
    then polls until it is approved in a browser — on any machine, over SSH,
    headless, or on a desktop. No localhost callback server, nothing else can
    time out. Returns the account name."""
    say("using the DEVICE login (OAuth 2.0 device flow) — approve it in any browser")
    print("1) open https://dash.cloudflare.com/oauth2/device in a browser")
    print("   on the machine you normally use (your own computer).")
    print("2) enter the code wrangler prints below and click Approve.")
    print("   This terminal waits and finishes automatically.")
    try:
        _run(ctx, ktl_common.wrangler_cmd("login", "--device"),
             inherit_stdio=True, timeout=600)
    except SubprocessError:
        sys.exit("The device login did not complete (see the error above). "
                 "Open the URL, enter the code and click Approve, then "
                 "re-run: python launch.py setup --only cloudflare")
    try:
        who = _parse_whoami(_whoami())
    except SubprocessError:
        who = ""
    if not who:
        sys.exit("The device login did not complete. Re-run: "
                 "python launch.py setup --only cloudflare")
    return who


def _step_cloudflare(ctx: Ctx, cfg: dict, args):
    if ctx.dry_run:
        ctx.log("[dry-run] would run: wrangler whoami (and `wrangler login` if needed)")
        say("ok    : Cloudflare (dry run — login check skipped)")
        return
    have_token = bool(os.environ.get("CLOUDFLARE_API_TOKEN", "").strip())
    if have_token:
        ctx.log("using Cloudflare API token from CLOUDFLARE_API_TOKEN env var")
    try:
        who = _parse_whoami(_whoami())
    except SubprocessError:
        who = ""
    if who:
        say(f"ok    : Cloudflare — logged in as {who}")
        return
    if have_token:
        sys.exit("CLOUDFLARE_API_TOKEN is set, but `wrangler whoami` did not "
                 "return an account. Check the token (dash.cloudflare.com/"
                 "profile/api-tokens) and re-run: "
                 "python launch.py setup --only cloudflare")

    who = _device_login(ctx)
    say(f"ok    : Cloudflare — logged in as {who}")


def _put_worker_secrets(ctx: Ctx, cfg: dict):
    """Set both Worker secrets. Values travel on STDIN only — never argv."""
    name = _worker_name(cfg, None)
    sec = cfg["secrets"]
    for var, val in (("CLIENT_API_KEY", sec["client_api_key"]),
                     ("UPDATE_SECRET", sec["update_secret"])):
        say(f"setting Worker secret {var} (via stdin — never argv, never logged)")
        _run(ctx, ktl_common.wrangler_cmd("secret", "put", var, "--name", name),
             cwd=WORKER_DIR, stdin=val + "\n", timeout=180)


def _step_secrets(ctx: Ctx, cfg: dict, args):
    sec = cfg.setdefault("secrets", {})
    if args.rotate:
        if not confirm(
                ctx,
                "Regenerate CLIENT_API_KEY and UPDATE_SECRET?\n"
                "  NOTE: rotating UPDATE_SECRET makes the Worker's stored config\n"
                "  unreadable until the next `serve` re-registers it (~30 s blip).",
                default=False):
            sys.exit("rotation cancelled (no changes made)")
        sec["client_api_key"] = ktl_common.gen_key()
        sec["update_secret"] = ktl_common.gen_key()
        say(f"rotated: client_api_key {ktl_common.mask(sec['client_api_key'])}")
        say(f"        update_secret  {ktl_common.mask(sec['update_secret'])}")
    else:
        if sec.get("client_api_key"):
            say(f"kept    : client_api_key {ktl_common.mask(sec['client_api_key'])}")
        else:
            sec["client_api_key"] = ktl_common.gen_key()
            say(f"generated: client_api_key {ktl_common.mask(sec['client_api_key'])}")
        if sec.get("update_secret"):
            say(f"kept    : update_secret  {ktl_common.mask(sec['update_secret'])}")
        else:
            sec["update_secret"] = ktl_common.gen_key()
            say(f"generated: update_secret  {ktl_common.mask(sec['update_secret'])}")
    if args.rotate and ktl_common.step_status(cfg, "worker") == "done":
        _put_worker_secrets(ctx, cfg)


def _step_worker(ctx: Ctx, cfg: dict, args):
    cf = cfg.setdefault("cloudflare", {})
    name = _worker_name(cfg, args)
    cf["worker_name"] = name

    if ctx.dry_run:
        ctx.log(f"[dry-run] would run: wrangler deploy --name {name} (from {WORKER_DIR})")
        ctx.log(f"[dry-run] would run: wrangler secret put CLIENT_API_KEY/UPDATE_SECRET "
                f"--name {name} (values on stdin)")
        say(f"ok    : Worker {name} (dry run — nothing deployed)")
        return

    if not (cfg.get("secrets", {}).get("client_api_key")
            and cfg.get("secrets", {}).get("update_secret")):
        sys.exit("no relay secrets generated yet — run: "
                 "python launch.py setup --only secrets")

    try:
        account = _parse_whoami(_whoami())
    except SubprocessError:
        account = ""
    if not account:
        sys.exit("Not logged in to Cloudflare — run: "
                 "python launch.py setup --only cloudflare")

    if args.adopt_worker:
        url = args.adopt_worker.strip().rstrip("/")
        if not re.search(r"\.workers\.dev/?$", url):
            sys.exit(f"--adopt-worker URL does not look like a workers.dev URL: "
                     f"{args.adopt_worker!r}")
        if not confirm(ctx,
                       f"Adopt the existing Worker {name!r} at {url}?\n"
                       "  (no deploy; just sets its 2 secrets and saves the URL)",
                       default=True):
            sys.exit("adopt cancelled (no changes made)")
        say(f"adopting: {url} (deploy skipped)")
    else:
        print()
        print("APPROVAL 1/4 — deploy the Cloudflare Worker relay")
        print(f"  account     : {account}")
        print(f"  worker name : {name}")
        print(f"  code        : {WORKER_DIR}")
        print("  also sets   : CLIENT_API_KEY + UPDATE_SECRET "
              "(generated above, sent via stdin)")
        if not confirm(ctx, "Deploy the Worker now?", default=True):
            sys.exit("deploy cancelled (nothing was deployed)")
        r = _run(ctx, ktl_common.wrangler_cmd("deploy", "--name", name),
                 cwd=WORKER_DIR, timeout=600)
        m = _WORKER_URL_RE.search(r.stdout or "")
        url = m.group(0).rstrip("/") if m else None
        if not url:
            url = ktl_common.ask(
                ctx, "Could not find the workers.dev URL in the deploy output — "
                     "paste the full URL: ", default="").strip().rstrip("/")
        if not url:
            sys.exit("no Worker URL found. Check the deploy output above, then "
                     "re-run (or use --adopt-worker URL).")
        print()
        print("APPROVAL 2/4 — confirm the relay URL")
        print(f"  your permanent endpoint will be: {url}")
        if not confirm(ctx, "Save this URL as the relay endpoint?", default=True):
            sys.exit("URL confirmation cancelled — the Worker IS deployed; "
                     "re-run setup or pass --adopt-worker URL.")

    cf["relay_url"] = url
    _put_worker_secrets(ctx, cfg)
    say(f"ok    : Worker {name} — relay {url}, secrets set")


def _step_serve(ctx: Ctx, cfg: dict, args):
    model = cfg.get("model", "qwen")
    mc = ktl_common.MODELS[model]
    relay_url = (cfg.get("cloudflare", {}).get("relay_url", "") or "").strip()
    relay_secret = (cfg.get("secrets", {}).get("update_secret", "") or "").strip()
    relay = {"url": relay_url.rstrip("/"), "secret": relay_secret} \
        if (relay_url and relay_secret) else None

    warn = (f"Starting a TPU session uses your Kaggle weekly TPU quota (~20 h/week).\n"
            f"  model      : {model} ({mc['served_model_name']})\n"
            f"  relay      : {relay['url'] if relay else '(none — per-boot quick-tunnel URL)'}\n"
            f"  keep-alive : 480 min (auto-shutdown when idle/over quota)")
    ctx.log("QUOTA WARNING (logged even with --yes): " + " | ".join(warn.splitlines()))

    if ctx.dry_run:
        print(warn)
        ctx.log("[dry-run] would start the TPU session "
                "(ktl_serve.push_and_watch, wait for READY)")
        say("ok    : TPU session (dry run — nothing started)")
        return

    print()
    print("APPROVAL 3/4 — start the TPU session")
    print(warn)
    if not confirm(ctx, "Start the TPU session now?", default=True):
        sys.exit("serve cancelled — Worker/relay state is saved; re-run: "
                 "python launch.py setup")
    user = cfg.get("kaggle", {}).get("username", "") or ktl_common.kaggle_username()
    if not user:
        sys.exit("Kaggle username unknown — run: python launch.py setup --only kaggle")
    ktl_serve.check_auth()
    ktl_serve.push_and_watch(model, user, relay=relay, stop_after_ready=True)
    say("ok    : TPU session READY (serving in the background on Kaggle)")


def _step_verify(ctx: Ctx, cfg: dict, args):
    if ctx.dry_run:
        ctx.log("[dry-run] would run the live compatibility matrix (env --test)")
        say("ok    : verification (dry run — matrix not run)")
        return
    r = _resolved_for(cfg, client="setup-verify")
    if r is None:
        sys.exit("no endpoint known yet — finish the serve step first "
                 "(python launch.py setup).")
    if r.key_is_placeholder:
        sys.exit("no client API key available — the relay needs a generated key "
                 "(step 'secrets') or $KTL_CLIENT_API_KEY.")
    say(f"running the live compatibility matrix against {r.base_url} ...")
    rows = ktl_env.run_matrix(r)
    print(ktl_env.format_matrix(rows, r))
    code = ktl_env.matrix_exit_code(rows)
    if code != 0:
        sys.exit(f"verification not fully green (exit {code}). If the TPU is still "
                 f"booting, wait and retry: python launch.py env --test --wait 600")
    say(f"ok    : endpoint verified at {r.base_url} (all critical routes pass)")


def _step_clients(ctx: Ctx, cfg: dict, args):
    detected = [(b, c) for b, c in _CLIENT_BINARIES if shutil.which(b)]
    if not detected:
        say("no AI clients found on PATH (looked for: "
            + ", ".join(b for b, _ in _CLIENT_BINARIES) + ")")
        print("  You can still generate configs manually later:  python launch.py env")
        return
    mc = ktl_common.MODELS[cfg.get("model", "qwen")]
    print()
    print("APPROVAL 4/4 — configure your AI clients to use the relay")
    for b, c in detected:
        print(f"  found: {c:<11} ({b})")
    picked = ktl_common.choose(
        ctx, "Select the clients to configure "
             "(each config file gets a .ktl-bak-* backup): ",
        [c for _, c in detected])
    if not picked:
        say("no clients selected — you can configure them later: python launch.py env")
        return
    for client in picked:
        r = _resolved_for(cfg, client=client)
        if r is None:
            print(f"  {client}: no endpoint known yet — skipped "
                  "(finish the serve step first)")
            continue
        report = ktl_env.apply_write(r, ctx.dry_run, ctx)
        print(f"  {client}: {report}")
        cmd = _CLIENT_CMD[client].replace("<MODEL>", mc["api_model"])
        print(f"        launch: {cmd}")
    say("ok    : clients configured")


_STEP_FNS = {
    "prereqs": _step_prereqs,
    "kaggle": _step_kaggle,
    "cloudflare": _step_cloudflare,
    "secrets": _step_secrets,
    "worker": _step_worker,
    "serve": _step_serve,
    "verify": _step_verify,
    "clients": _step_clients,
}


# --------------------------------------------------------------------------
# setup (wizard entry point)
# --------------------------------------------------------------------------

def cmd_setup(args) -> int:
    ctx = Ctx(yes=args.yes, dry_run=args.dry_run, verbose=args.verbose)

    if args.reset:
        p = ktl_common.config_path()
        if p.exists():
            if not confirm(ctx,
                           f"Delete {p} and start over?\n"
                           "  (does NOT delete the deployed Cloudflare Worker)",
                           default=True):
                return 1
            if ctx.dry_run:
                ctx.log(f"[dry-run] would delete {p}")
                print(f"  [dry-run] would delete {p}")
            else:
                p.unlink()
                print(f"deleted {p}")
        else:
            print(f"nothing to reset: {p} does not exist")

    cfg = ktl_common.load_config()
    model = ktl_common.resolve(args.model, None, cfg.get("model"), "qwen")
    if model not in ktl_common.MODELS:
        print(f"unknown model {model!r} (choose from: {', '.join(ktl_common.MODELS)})")
        return 2
    cfg["model"] = model

    steps = (args.only,) if args.only else WIZARD_STEPS

    print()
    if ctx.dry_run:
        print("kaggle-tpu-lab setup — DRY RUN (nothing will be changed)")
    else:
        print(f"kaggle-tpu-lab setup — model: {model} "
              f"({ktl_common.MODELS[model]['served_model_name']})")
        print(f"  state file: {ktl_common.config_path()}   "
              f"log: {ktl_common.log_path()}")
    print()

    ok = True
    for step in steps:
        persisted = step in ktl_common.STEP_NAMES  # verify is not persisted
        if persisted and not args.only and ktl_common.step_status(cfg, step) == "done":
            print(f"  skip  : {step:<11} (already done; --reset to redo from scratch)")
            continue
        print(f"  step  : {step}")
        try:
            _STEP_FNS[step](ctx, cfg, args)
        except SystemExit as e:
            code = e.code
            if isinstance(code, str) and code:
                print(str(code))   # sys.exit("message") must not be swallowed
            if code not in (None, 0) and persisted:
                ktl_common.set_step(cfg, step, "failed")
                _save(ctx, cfg)
            ok = False
            break
        except SubprocessError as e:
            if persisted:
                ktl_common.set_step(cfg, step, "failed")
                _save(ctx, cfg)
            print(f"\nerror : {e}")
            if e.hint:
                print(f"hint  : {e.hint}")
            ok = False
            break
        except KeyboardInterrupt:
            print()
            print("interrupted — progress saved. "
                  "Re-run `python launch.py setup` to resume.")
            break
        if persisted and not ctx.dry_run:
            ktl_common.set_step(cfg, step, "done")
            _save(ctx, cfg)
        print(f"  done  : {step}")

    # flush any log lines (also on failure/interrupt paths)
    if not ctx.dry_run and ctx.log_lines:
        p = ktl_common.log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for line in ctx.log_lines:
                f.write(ktl_common.redact(line) + "\n")
        ctx.log_lines.clear()

    print()
    if ok:
        relay = (cfg.get("cloudflare", {}).get("relay_url", "") or "").strip()
        key = (cfg.get("secrets", {}).get("client_api_key", "") or "").strip()
        mk = cfg.get("model", "qwen")
        print("setup complete.")
        if relay:
            print(f"  relay      : {relay}")
        if key:
            print(f"  client key : {ktl_common.mask(key)}   "
                  f"(stored in {ktl_common.config_path()}; clients use $KTL_CLIENT_API_KEY)")
        print(f"  model      : {mk} ({ktl_common.MODELS[mk]['api_model']})")
        print("  next       : python launch.py env --test     "
              "(or: python launch.py doctor)")
    else:
        print("setup stopped — see the error above. Re-run "
              "`python launch.py setup` to resume where it left off.")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# doctor (read-only health report)
# --------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    """Read-only health report. ZERO side effects: never writes config or
    state, never mutates anything; only prints (secrets redacted).
    Exit 0 = healthy, 1 = one or more problems (with fix hints)."""
    lines: list = []
    problems: list = []
    hints: list = []
    matrix_rows: list = []

    def line(status: str, text: str, hint: str = None):
        lines.append((status, text))
        if status in ("FAIL", "MISSING"):
            problems.append(text)
            if hint:
                hints.append(hint)
        elif status == "WARN" and hint:
            hints.append(hint)

    print(f"ktl doctor — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}   "
          f"(read-only, secrets redacted)")

    # 1. Python
    py = sys.version_info
    pys = f"{py.major}.{py.minor}.{py.micro}"
    if (py.major, py.minor) < (3, 10):
        line("FAIL", f"Python {pys} (need >= 3.10)",
             "Install Python 3.10+ and re-run: python launch.py setup")
    elif (py.major, py.minor) < (3, 11):
        line("WARN", f"Python {pys} (kaggle CLI now recommends 3.11+)",
             "If the kaggle CLI misbehaves, upgrade Python.")
    else:
        line("OK", f"Python {pys}")

    # 2. kaggle CLI + credentials + auth
    have_bin = bool(shutil.which("kaggle"))
    if have_bin:
        line("OK", f"kaggle CLI {ktl_common.kaggle_version()} at {ktl_common.kaggle_cmd()[0]}")
    else:
        line("FAIL", "kaggle CLI not on PATH",
             f"Install: {sys.executable} -m pip install kaggle")
    cred = _kaggle_credential_source()
    if cred:
        line("OK", f"Kaggle credentials present (via {cred}) — value not shown")
    else:
        line("MISSING", "Kaggle credentials not found",
             "Run `kaggle auth login` (browser), or create an API token at "
             + ktl_common.KAGGLE_TOKEN_URL)
    if have_bin:
        ok, detail = ktl_common.kaggle_authed()
        if ok:
            line("OK", f"Kaggle auth: ok (user: {ktl_common.kaggle_username() or 'unknown'})")
        else:
            line("FAIL", "Kaggle auth check failed",
                 "Check credentials at https://www.kaggle.com/settings/api "
                 "(TPU access also requires a phone-verified account)")

    # 3. Node.js + npx
    need = ".".join(str(x) for x in ktl_common.NODE_MIN)
    nv = ktl_common.node_version()
    if nv is None:
        line("MISSING", "Node.js not found",
             f"Install the LTS from {ktl_common.NODE_INSTALL_URL} "
             f"(wrangler needs Node >= v{need})")
    elif nv < ktl_common.NODE_MIN:
        line("FAIL", f"Node v{'.'.join(str(x) for x in nv)} too old (need >= v{need})",
             f"Upgrade from {ktl_common.NODE_INSTALL_URL}")
    else:
        npx = ktl_common.npx_cmd()
        line("OK", f"Node v{'.'.join(str(x) for x in nv)} (>= v{need}), "
                   f"npx: {npx or 'NOT FOUND'}")
        if not npx:
            line("FAIL", "npx not found next to node",
                 "Reinstall Node from nodejs.org")

    # 4. Cloudflare login (read-only whoami; skipped when Node is absent)
    if nv is not None and nv >= ktl_common.NODE_MIN:
        who = ""
        try:
            who = _parse_whoami(_whoami(quiet=True))
        except SubprocessError:
            pass
        if who:
            line("OK", f"Cloudflare login: {who}")
        else:
            line("MISSING", "Cloudflare: not logged in",
                 "Run: python launch.py setup --only cloudflare")

    # 5. config file
    cfg = ktl_common.load_config(migrate=False)
    cp = ktl_common.config_path()
    if cp.exists():
        mode = format(stat.S_IMODE(cp.stat().st_mode), "03o")
        line("OK", f"config: {cp} (mode {mode})")
        if mode != "600" and os.name != "nt":
            line("WARN", f"config file mode is {mode} (expected 600)",
                 f"Run: chmod 600 {cp}")
        steps = cfg.get("steps", {})
        line("OK", "steps: " + (", ".join(f"{k}={v}" for k, v in steps.items()) or "none"))
        sec = cfg.get("secrets", {})
        if sec.get("client_api_key") and sec.get("update_secret"):
            line("OK", "secrets: client_api_key set, update_secret set (values hidden)")
        else:
            line("WARN", "secrets: "
                 + ", ".join(k + " missing" for k in ("client_api_key", "update_secret")
                             if not sec.get(k)),
                 "Run: python launch.py setup --only secrets")
    else:
        line("MISSING", f"config file not found: {cp}",
             "Run: python launch.py setup")

    # 6. endpoint
    relay_url = (cfg.get("cloudflare", {}).get("relay_url", "") or "").strip().rstrip("/")
    if relay_url:
        line("OK", f"relay URL (saved): {relay_url}")
        key = os.environ.get(ktl_common.ENV_VAR, "").strip() \
            or cfg.get("secrets", {}).get("client_api_key", "")
        if not key:
            line("WARN", "relay client key not available "
                 "(set $KTL_CLIENT_API_KEY or run setup --only secrets)",
                 "Run: python launch.py setup --only secrets")
        else:
            code, body = _http_get_json(relay_url + "/v1/models", key, timeout=8)
            if code == 200:
                line("OK", "GET /v1/models -> 200 (endpoint reachable)")
                r = _resolved_for(cfg, client="doctor")
                if r is not None and not r.key_is_placeholder:
                    matrix_rows = ktl_env.run_matrix(r)
                    if not args.json:
                        print(ktl_env.format_matrix(matrix_rows, r))
                    mcode = ktl_env.matrix_exit_code(matrix_rows)
                    if mcode == 0:
                        line("OK", "compatibility matrix: all critical routes pass")
                    else:
                        line("WARN",
                             f"compatibility matrix: {mcode} route(s) not passing",
                             "The model may still be booting — wait and retry: "
                             "python launch.py env --test --wait 600")
            elif code is None:
                line("FAIL", f"relay unreachable: {body}",
                     "Is the Worker deployed? Check the Cloudflare dashboard, then "
                     "re-check: python launch.py doctor")
            else:
                line("FAIL", f"GET /v1/models -> HTTP {code}",
                     "Wrong key or Worker misconfigured — "
                     "python launch.py setup --only worker --rotate")
    elif ktl_env._SERVE_STATE_FILE.exists():
        try:
            st = json.loads(ktl_env._SERVE_STATE_FILE.read_text())
            line("OK", f"direct endpoint (serve state): {st.get('base_url', '?')} "
                 f"[{st.get('mode', 'direct')}]")
        except Exception:
            line("WARN", "serve state file exists but is unreadable",
                 "Run: python launch.py serve")
    else:
        line("MISSING", "no endpoint configured (no relay URL, no running serve)",
             "Run: python launch.py setup   (or: python launch.py serve for direct mode)")

    rc = 1 if problems else 0
    print()
    if args.json:
        doc = {"healthy": not problems,
               "checks": [{"status": s, "text": t} for s, t in lines],
               "hints": hints}
        if matrix_rows:
            doc["matrix"] = matrix_rows
        print(json.dumps(doc, indent=2))
    else:
        for status, text in lines:
            tag = {"OK": "ok", "WARN": "warn"}.get(status, status)
            print(f"  [{tag:<7}] {text}")
        print()
        if problems:
            print(f"RESULT: {len(problems)} problem(s)")
            if hints:
                print("how to fix:")
                for h in hints:
                    print("  - " + h)
        else:
            print("RESULT: healthy")
    return rc
