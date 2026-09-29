"""ktl_env — shared AI-client config generator + compatibility matrix.

Stdlib only (Python 3.10+). No third-party deps, no `tomllib` (3.11+), no PyYAML.

This module is imported by ``launch.py`` (``ktl env``). It is deliberately
self-contained: it never imports ``launch`` (avoiding a circular import) and never
hardcodes model names — everything model-specific is passed in via ``Resolved``
(sourced by launch.py from the MODELS registry).

Secret hygiene
--------------
By default generators emit an *environment reference* to the client key
(``$KTL_CLIENT_API_KEY``), never the literal — wherever the client's format
supports it.  ``--reveal`` embeds the literal.

Some clients only accept a static config file that cannot reference an env var
(Claude ``settings.json``, opencode ``opencode.json``, Hermes ``.env``).  For
those the literal is written to the file, which is ``chmod 600`` on POSIX and
always produces a warning.  When no key is available the placeholder
``<YOUR_CLIENT_API_KEY>`` is emitted instead.

Clients authenticate to the relay with the Worker's ``CLIENT_API_KEY`` — never
``UPDATE_SECRET`` or ``KTL_API_KEY``.  This module never reads Worker secrets.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import ktl_common
from ktl_common import (PLACEHOLDER, ENV_VAR, MODELS, say)   # re-exported for compat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIENTS = ["claude-code", "codex", "opencode", "hermes", "aider", "curl", "python"]

PROVIDER_ID = "kaggle-tpu"          # opencode provider id (match the live config)
PROVIDER_NAME = "Kaggle TPU Relay"
OPENCODE_NPM = "@ai-sdk/openai-compatible"
ANTHROPIC_VERSION = "2023-06-01"

_STATE_DISPLAY = "~/.ktl/state.json"

# Per-shell token for "the KTL_CLIENT_API_KEY env var".
_SHELL_REF = {
    "bash": "$KTL_CLIENT_API_KEY",
    "zsh": "$KTL_CLIENT_API_KEY",
    "fish": "$KTL_CLIENT_API_KEY",
    "powershell": "$env:KTL_CLIENT_API_KEY",
    "cmd": "%KTL_CLIENT_API_KEY%",
}

# Client -> [(config file, write kind)]. aider/curl/python are print-only.
# Hermes manages two files: the .env key line + the config.yaml model: block.
WRITE_TARGETS = {
    "claude-code": [("~/.claude/settings.json", "json_env")],
    "codex": [("~/.codex/config.toml", "toml_block")],
    "opencode": [("~/.config/opencode/opencode.json", "json_provider")],
    "hermes": [("~/.hermes/.env", "hermes_env"), ("~/.hermes/config.yaml", "hermes_model")],
}

# Critical rows for the --test exit code.
_TEST_TIMEOUT = 60
_MAX_TOKENS = 16


# ---------------------------------------------------------------------------
# Resolved endpoint
# ---------------------------------------------------------------------------
@dataclass
class Resolved:
    client: str
    model_key: str                 # "qwen" | "glm"
    api_model: str                 # e.g. "qwen3.8-27b" (from registry)
    context: int
    max_output: int
    cost: dict
    base_url: str                  # root, no /v1, no trailing slash
    client_key: str                # literal key or PLACEHOLDER
    key_is_placeholder: bool
    key_source: str                # env | state | placeholder
    url_source: str                # cli | env | state
    shell: str = "bash"
    reveal: bool = False
    responses_api: bool = True    # does the backend serve POST /v1/responses (Codex)
    notes: list = field(default_factory=list)

    @property
    def openai_base(self) -> str:
        return self.base_url + "/v1"


def normalize_root(url: str) -> str:
    """Strip trailing slashes and a trailing /v1 so we always hold the root."""
    u = (url or "").strip()
    u = u.rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


def detect_shell() -> str:
    if os.environ.get("PSModulePath"):
        return "powershell"
    name = os.path.basename(os.environ.get("SHELL", "")).lower()
    if name in ("zsh", "bash", "fish"):
        return name
    if os.name == "nt":
        return "powershell"
    return "bash"


# ---------------------------------------------------------------------------
# Shell + key helpers
# ---------------------------------------------------------------------------
def _sq(s: str) -> str:
    """Single-quote for POSIX shells (escape embedded single quotes)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def shell_export(shell: str, name: str, use_ref: bool, literal: str) -> str:
    """One export line for the given shell. ``use_ref`` emits the env reference."""
    if use_ref:
        ref = _SHELL_REF[shell]
        if shell in ("bash", "zsh"):
            return f"export {name}={ref}"
        if shell == "fish":
            return f"set -x {name} {ref}"
        if shell == "powershell":
            return f"$env:{name} = {ref}"
        return f"set {name}={ref}"
    if shell in ("bash", "zsh"):
        return f"export {name}={_sq(literal)}"
    if shell == "fish":
        return f"set -x {name} {literal}"
    if shell == "powershell":
        return f'$env:{name} = "{literal}"'
    return f"set {name}={literal}"


def _print_key(r: Resolved):
    """(use_ref, literal) for print contexts. Placeholder always shows literally."""
    if r.key_is_placeholder:
        return (False, r.client_key)
    if r.reveal:
        return (False, r.client_key)
    return (True, r.client_key)


def _file_key(r: Resolved) -> str:
    """Key token for file-based clients (static config, no env-ref support).

    By default we write the PLACEHOLDER so no real secret lands in a file or
    the printed preview; ``--reveal`` opts in to the literal (which ``--write``
    then stores chmod 600 with a warning). This keeps ``ktl env`` secret-free
    unless you explicitly ask for the literal.
    """
    if r.key_is_placeholder:
        return r.client_key          # already a placeholder
    if r.reveal:
        return r.client_key          # real literal, opt-in
    return PLACEHOLDER               # hidden by default


def _write_key(r: Resolved, existing_key: str):
    """(set_it, value) for a file write. A real key is only ever written with
    --reveal; otherwise we keep the user's existing key, or leave a placeholder
    slot when there is none. Never clobbers a real key with a placeholder."""
    if (not r.key_is_placeholder) and r.reveal:
        return (True, r.client_key)
    if existing_key in (None, ""):
        return (True, PLACEHOLDER)
    return (False, existing_key)     # preserve what's already there


def _header(r: Resolved) -> list:
    key_desc = {
        "env": "env $KTL_CLIENT_API_KEY",
        "state": f"state ({_STATE_DISPLAY})",
        "placeholder": "PLACEHOLDER — set KTL_CLIENT_API_KEY",
    }.get(r.key_source, r.key_source)
    hidden = "" if r.reveal else "   [hidden; pass --reveal to print the literal]"
    return [
        f"# {r.client}  (model: {r.api_model}, context: {r.context})",
        f"# base URL : {r.base_url}   (source: {r.url_source})",
        f"# API key  : {key_desc}{hidden}",
        f"# shell    : {r.shell}",
        "",
    ]


# ---------------------------------------------------------------------------
# opencode provider (golden-locked)
# ---------------------------------------------------------------------------
def opencode_model_block(r: Resolved) -> dict:
    return {
        "name": r.api_model,
        "limit": {"context": r.context, "output": r.max_output},
        "cost": dict(r.cost),
        "tool_call": True,
        "reasoning": True,
        "temperature": True,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
    }


def opencode_provider(r: Resolved) -> dict:
    """The ``{PROVIDER_ID: {...}}`` block (sits under ``provider.`` in the file)."""
    return {
        PROVIDER_ID: {
            "npm": OPENCODE_NPM,
            "name": PROVIDER_NAME,
            "options": {
                "baseURL": r.openai_base,
                "apiKey": _file_key(r),
            },
            "models": {r.api_model: opencode_model_block(r)},
        }
    }


def opencode_provider_json(r: Resolved) -> str:
    return json.dumps(opencode_provider(r), indent=2) + "\n"


# ---------------------------------------------------------------------------
# Per-client print generators
# ---------------------------------------------------------------------------
def _claude_model_env(r: Resolved) -> dict:
    """Model-selection env vars for Claude Code.

    The model is the BARE id (vLLM validates the name against the served model,
    and the relay does no name rewriting — a ``kaggle-tpu/`` prefix is rejected).
    Background tasks (titles, etc.) use the haiku slot, so point it at the same
    model or they 404 on a default Anthropic haiku id. We set both the current
    (ANTHROPIC_DEFAULT_HAIKU_MODEL) and deprecated (ANTHROPIC_SMALL_FAST_MODEL)
    vars so it works across Claude Code versions. Sonnet/opus aliases are pinned
    too: /model sonnet|opus would otherwise switch to default Anthropic ids,
    which the relay does not serve.
    """
    m = r.api_model
    return {
        "ANTHROPIC_MODEL": m,
        "ANTHROPIC_SMALL_FAST_MODEL": m,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": m,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": m,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": m,
    }


def gen_claude_code(r: Resolved) -> str:
    key = _file_key(r)
    cfg = {
        "env": {
            "ANTHROPIC_BASE_URL": r.base_url,   # root — Claude appends /v1/messages
            "ANTHROPIC_AUTH_TOKEN": key,        # Bearer (vLLM is Bearer-only)
            **_claude_model_env(r),
        }
    }
    body = json.dumps(cfg, indent=2)
    return "\n".join([
        "Merge into ~/.claude/settings.json (the 'env' object):",
        body,
        "",
        "# base URL is the ROOT — the Claude client appends /v1/messages itself.",
        "# ANTHROPIC_AUTH_TOKEN (Bearer), not ANTHROPIC_API_KEY (x-api-key).",
        "# model is the bare id — vLLM validates the name, so no provider prefix.",
        f"# SMALL_FAST / DEFAULT_HAIKU route background tasks at the same model.",
        f"# run: claude   (model {r.api_model})",
        _warn_file_key(r),
    ])


def gen_codex(r: Resolved) -> str:
    # This export *defines* KTL_CLIENT_API_KEY (which codex reads via env_key),
    # so it must carry a literal — never a self-referencing $KTL_CLIENT_API_KEY.
    # Without --reveal we print the placeholder so the real key is not leaked.
    keyval = r.client_key if (r.reveal or r.key_is_placeholder) else PLACEHOLDER
    lines = [
        "1) Export the key (codex reads it via env_key at runtime):",
        "   " + shell_export(r.shell, ENV_VAR, False, keyval),
        "",
        "2) Append this managed block to ~/.codex/config.toml:",
        toml_managed_block(r),
        "",
        "3) Run with the model passed on the command line",
        "   (top-level model keys are intentionally NOT written, so your",
        "   existing codex config keeps working):",
        f"   codex -c model_provider={PROVIDER_ID} -c model={r.api_model}",
        "",
    ]
    if not r.responses_api:
        lines += [
            "!!! NOT SUPPORTED FOR THIS MODEL !!!",
            f"# This backend does NOT implement POST /v1/responses, which Codex",
            f'# requires (wire_api = "responses"). The config above will NOT work',
            f"# for {r.api_model}. Use Claude Code, opencode, Hermes, aider, or the",
            f"# curl/Python OpenAI examples instead. Confirm with: ktl env --test --model {r.model_key}",
        ]
    else:
        lines += [
            "# wire_api = \"responses\" needs the backend's /v1/responses route",
            f"# (this model serves it); confirm with: ktl env --test --model {r.model_key}",
        ]
    return "\n".join(lines)


def gen_opencode(r: Resolved) -> str:
    return "\n".join([
        "Merge into ~/.config/opencode/opencode.json under 'provider':",
        opencode_provider_json(r).rstrip("\n"),
        "",
        "# base URL is root + /v1 (OpenAI-compatible provider).",
        _warn_file_key(r),
    ])


def hermes_model_block(r: Resolved) -> str:
    """The top-level ``model:`` block for ~/.hermes/config.yaml.

    ``provider: custom`` + ``api_mode: chat_completions`` makes Hermes call
    POST {base_url}/chat/completions with the bare model id. The api_key is a
    ${KTL_CLIENT_API_KEY} reference — .env is loaded before config.yaml, so it
    expands without the secret ever landing in the YAML file.
    """
    return "\n".join([
        "model:",
        f"  default: {r.api_model}",
        "  provider: custom",
        f"  base_url: {r.openai_base}",
        "  api_key: ${" + ENV_VAR + "}",
        "  api_mode: chat_completions",
    ])


def gen_hermes(r: Resolved) -> str:
    return "\n".join([
        "1) ~/.hermes/.env (loaded first; ${...} in config.yaml expands from it):",
        f"{ENV_VAR}={_file_key(r)}",
        "",
        "2) Replace the top-level `model:` block in ~/.hermes/config.yaml:",
        hermes_model_block(r),
        "",
        "# Verified against the installed hermes-agent source (~/.hermes/hermes-agent):",
        "# - bare `openai` is aliased to openrouter, and `openai-api` uses the",
        "#   codex_responses transport (/v1/responses) — OPENAI_API_KEY /",
        "#   OPENAI_BASE_URL do NOT route here, so an earlier version's lines",
        "#   for them are now unused (safe to delete from .env).",
        "# - provider: custom + api_mode: chat_completions -> POST /v1/chat/completions.",
        f"# run: hermes   (or: hermes -m {r.api_model} to override per invocation)",
        _warn_file_key(r),
    ])


def gen_aider(r: Resolved) -> str:
    # Aider only routes a model to an OpenAI-compatible endpoint when the name
    # carries the `openai/` prefix, and the endpoint var is OPENAI_API_BASE
    # (verified: aider.chat/docs/llms/openai-compat.html).
    use_ref, literal = _print_key(r)
    m = "openai/" + r.api_model
    return "\n".join([
        "Aider (print-only — no file is written):",
        "",
        shell_export(r.shell, "AIDER_MODEL", False, m),
        shell_export(r.shell, "OPENAI_API_KEY", use_ref, literal),
        shell_export(r.shell, "OPENAI_API_BASE", False, r.openai_base),
        "",
        "# or .aider.conf.yml:",
        "model: " + m,
        f"openai-api-key: {_SHELL_REF[r.shell] if use_ref else literal}",
        f"openai-api-base: {r.openai_base}",
        "",
        "# run: aider   (model shown as openai/" + r.api_model + ")",
    ])


def gen_curl(r: Resolved) -> str:
    use_ref, literal = _print_key(r)
    key = _SHELL_REF[r.shell] if use_ref else literal
    openai_body = json.dumps(
        {"model": r.api_model,
         "messages": [{"role": "user", "content": "Hello!"}]}, separators=(",", ":"))
    anthro_body = json.dumps(
        {"model": r.api_model, "max_tokens": _MAX_TOKENS,
         "messages": [{"role": "user", "content": "Hello!"}]}, separators=(",", ":"))
    return "\n".join([
        "curl (print-only):",
        "",
        "# OpenAI-compatible chat:",
        f'curl -sS {r.openai_base}/chat/completions \\',
        f'  -H "Authorization: Bearer {key}" \\',
        f'  -H "Content-Type: application/json" -d \'{openai_body}\'',
        "",
        "# Anthropic Messages:",
        f'curl -sS {r.openai_base}/messages \\',
        f'  -H "x-api-key: {key}" \\',
        f'  -H "anthropic-version: {ANTHROPIC_VERSION}" \\',
        f'  -H "Content-Type: application/json" -d \'{anthro_body}\'',
    ])


def gen_python(r: Resolved) -> str:
    use_ref, literal = _print_key(r)
    key_expr = f'os.environ["{ENV_VAR}"]' if use_ref else json.dumps(literal)
    return "\n".join([
        "Python (print-only):",
        "",
        "# OpenAI SDK (base_url = root + /v1):",
        "import os",
        "from openai import OpenAI",
        f"client = OpenAI(base_url={json.dumps(r.openai_base)}, api_key={key_expr})",
        f"r = client.chat.completions.create(model={json.dumps(r.api_model)}, "
        "messages=[{'role': 'user', 'content': 'Hi'}])",
        "",
        "# Anthropic SDK (base_url = root, NO /v1):",
        "from anthropic import Anthropic",
        f"client = Anthropic(base_url={json.dumps(r.base_url)}, api_key={key_expr})",
    ])


_GENERATORS = {
    "claude-code": gen_claude_code,
    "codex": gen_codex,
    "opencode": gen_opencode,
    "hermes": gen_hermes,
    "aider": gen_aider,
    "curl": gen_curl,
    "python": gen_python,
}


def _warn_file_key(r: Resolved) -> str:
    if r.key_is_placeholder:
        return "# note: no key available — fill in " + PLACEHOLDER + " (or re-run with the env set)."
    return "# warning: the literal key is written to a file — it is chmod 600 on POSIX."


def render(r: Resolved) -> str:
    """Full printed output (header + client-specific snippet)."""
    return "\n".join([*_header(r), _GENERATORS[r.client](r)])


# ---------------------------------------------------------------------------
# Write mode
# ---------------------------------------------------------------------------
def toml_managed_block(r: Resolved) -> str:
    return "\n".join([
        "# >>> ktl managed >>>",
        "[model_providers." + PROVIDER_ID + "]",
        f'name = "{PROVIDER_NAME}"',
        f'base_url = "{r.openai_base}"',
        'wire_api = "responses"',
        f'env_key = "{ENV_VAR}"',
        "# <<< ktl managed <<<",
    ])


def _home(p: str) -> str:
    return os.path.expanduser(p)


def _backups_for(path: str):
    import glob
    return sorted(glob.glob(path + ".ktl-bak-*"))


def backup(path: str) -> str:
    """Copy ``path`` to ``path.ktl-bak-<ts>``, keeping the 3 most recent."""
    stamp = time.strftime("%Y%m%d%H%M%S")
    dest = f"{path}.ktl-bak-{stamp}"
    n = 1
    while os.path.exists(dest):          # same-second writes get -1, -2, ...
        dest = f"{path}.ktl-bak-{stamp}-{n}"
        n += 1
    with open(path, "rb") as f:
        data = f.read()
    with open(dest, "wb") as f:
        f.write(data)
    old = _backups_for(path)[3:]
    for o in old:
        try:
            os.remove(o)
        except OSError:
            pass
    return dest


def restore(path: str) -> str:
    """Restore the most recent backup of ``path`` (if any)."""
    backs = _backups_for(path)
    if not backs:
        return f"no backup found for {path}"
    latest = backs[-1]
    backup(path)  # snapshot the current state before overwriting
    with open(latest, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data)
    return f"restored {path} from {latest}"


def _json_load_or_refuse(path: str):
    """Load a JSON file; if it exists but is unparseable, back it up + raise."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        b = backup(path)
        raise RefuseWrite(f"{path} is not valid JSON ({e}); backed up to {b}, not writing.")


class RefuseWrite(RuntimeError):
    pass


def _write_json_provider(path: str, r: Resolved, dry_run: bool):
    existing = _json_load_or_refuse(path)
    ktl = existing.setdefault("provider", {}).setdefault(PROVIDER_ID, {})
    ktl["npm"] = OPENCODE_NPM
    ktl["name"] = PROVIDER_NAME
    opts = ktl.setdefault("options", {})
    opts["baseURL"] = r.openai_base
    set_it, val = _write_key(r, opts.get("apiKey"))
    if set_it:
        opts["apiKey"] = val
    ktl.setdefault("models", {})[r.api_model] = opencode_model_block(r)
    return _emit(path, json.dumps(existing, indent=2) + "\n", dry_run)


def _write_json_env(path: str, r: Resolved, dry_run: bool):
    existing = _json_load_or_refuse(path)
    env = existing.setdefault("env", {})
    env["ANTHROPIC_BASE_URL"] = r.base_url
    env.update(_claude_model_env(r))
    set_it, val = _write_key(r, env.get("ANTHROPIC_AUTH_TOKEN"))
    if set_it:
        env["ANTHROPIC_AUTH_TOKEN"] = val
    return _emit(path, json.dumps(existing, indent=2) + "\n", dry_run)


def _write_toml_block(path: str, r: Resolved, dry_run: bool):
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        if not _toml_parses(content):
            b = backup(path)
            raise RefuseWrite(f"{path} is not valid TOML; backed up to {b}, not writing.")
        # Replace any existing managed block (idempotent).
        start = "# >>> ktl managed >>>"
        end = "# <<< ktl managed <<<"
        if start in content and end in content:
            i, j = content.index(start), content.index(end) + len(end)
            content = content[:i] + toml_managed_block(r) + content[j:]
        else:
            content = content.rstrip("\n") + "\n\n" + toml_managed_block(r) + "\n"
    else:
        content = toml_managed_block(r) + "\n"
    return _emit(path, content, dry_run)


def _toml_parses(content: str) -> bool:
    try:
        import tomllib  # 3.11+; only used to VALIDATE, not to write
        tomllib.loads(content)
        return True
    except ModuleNotFoundError:
        # No validator on 3.10 — fall back to a lightweight sanity check so we
        # still refuse obviously-broken files (duplicate top-level keys).
        import re
        tops = re.findall(r"^([A-Za-z0-9_.\-]+)\s*=", content, flags=re.M)
        tops = [t for t in tops if not t.startswith("[")]
        return len(tops) == len(set(tops))
    except Exception:
        return False


def _write_hermes_env(path: str, r: Resolved, dry_run: bool):
    """Manage only the KTL_CLIENT_API_KEY line in ~/.hermes/.env; other lines
    (incl. stale OPENAI_* from earlier versions) are left untouched."""
    existing_lines = []
    if os.path.exists(path):
        with open(path) as f:
            existing_lines = f.read().splitlines()
    key_line = next((ln for ln in existing_lines if ln.startswith(ENV_VAR + "=")), None)
    existing_key = key_line.split("=", 1)[1] if key_line else ""
    set_it, val = _write_key(r, existing_key)
    lines = [ln for ln in existing_lines if not ln.startswith(ENV_VAR + "=")]
    if set_it:
        lines.append(f"{ENV_VAR}={val}")
    elif key_line:
        lines.append(key_line)          # preserve the user's existing key
    content = "\n".join(lines).rstrip("\n") + "\n" if lines else ""
    return _emit(path, content, dry_run)


def _write_hermes_model(path: str, r: Resolved, dry_run: bool):
    """Replace the top-level ``model:`` block in config.yaml (or append one when
    absent). Line-based surgery: the block runs from the unindented ``model:``
    line to the next unindented line; everything else is byte-preserved."""
    block = hermes_model_block(r).splitlines()
    lines = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("model:")), None)
    if start is not None:
        end = start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
            end += 1
        lines[start:end] = block
    else:
        if lines:
            lines.append("")
        lines.extend(block)
    content = "\n".join(lines).rstrip("\n") + "\n"
    return _emit(path, content, dry_run)


def _redact(text: str) -> str:
    """Mask key values in a diff so a real secret is never printed, while leaving
    the placeholder slot visible to the user. Env-style lines are masked for ANY
    secret-looking name (*TOKEN/*KEY/*SECRET/*PASSWORD) — dotenv diffs include
    context lines with the user's unrelated credentials."""
    def mask(val: str) -> str:
        return val if val == PLACEHOLDER else "***REDACTED***"
    text = re.sub(r'("apiKey"\s*:\s*")([^"]*)(")',
                  lambda m: m.group(1) + mask(m.group(2)) + m.group(3), text)
    text = re.sub(r'("ANTHROPIC_AUTH_TOKEN"\s*:\s*")([^"]*)(")',
                  lambda m: m.group(1) + mask(m.group(2)) + m.group(3), text)
    text = re.sub(
        r'([A-Za-z_][A-Za-z0-9_]*?(?:TOKEN|KEY|SECRET|PASSWORD|PASSWD)[A-Za-z0-9_]*=)(\S+)',
        lambda m: m.group(1) + mask(m.group(2)), text)
    return text


def unified_diff(old: str, new: str, label: str) -> str:
    import difflib
    return _redact("\n".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=label + " (current)", tofile=label + " (new)")))


def _emit(path: str, new_content: str, dry_run: bool) -> str:
    exists = os.path.exists(path)
    if exists:
        with open(path) as f:
            old = f.read()
    else:
        old = ""
    if new_content == old:
        return f"no change: {path}"
    if dry_run:
        diff = unified_diff(old, new_content, path)
        return f"[dry-run] would update {path}:\n" + (diff or "  (empty -> file)")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if exists:
        backup(path)
    with open(path, "w") as f:
        f.write(new_content)
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return f"wrote {path}" + (" (backup: " + path + ".ktl-bak-*)" if exists else " (new file)")


_WRITERS = {
    "json_provider": _write_json_provider,
    "json_env": _write_json_env,
    "toml_block": _write_toml_block,
    "hermes_env": _write_hermes_env,
    "hermes_model": _write_hermes_model,
}


def apply_write(r: Resolved, dry_run: bool, ctx=None) -> str:
    if r.client not in WRITE_TARGETS:
        return f"{r.client} is print-only — there is nothing to --write."
    ctx = ctx or ktl_common.Ctx(dry_run=dry_run)
    writing_literal = (not r.key_is_placeholder) and r.reveal
    if writing_literal and not dry_run:
        for path, _kind in WRITE_TARGETS[r.client]:
            ktl_common.gitignore_guard(ctx, _home(path), r.client_key)
    return "\n".join(_WRITERS[kind](_home(path), r, dry_run)
                     for path, kind in WRITE_TARGETS[r.client])


# ---------------------------------------------------------------------------
# --test compatibility matrix
# ---------------------------------------------------------------------------
def _http(method: str, url: str, headers: dict, body: dict | None,
          stream: bool = False, timeout: int = _TEST_TIMEOUT):
    """Return (status, first_chunk_or_body, headers, elapsed_ms, ttfb_ms)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    t0 = time.time()
    ttfb = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        status = resp.status
        rheaders = dict(resp.headers)
        if stream:
            line = resp.readline()
            ttfb = (time.time() - t0) * 1000
            # read a couple more lines to confirm the stream is producing data
            buf = [line]
            for _ in range(3):
                l2 = resp.readline()
                if not l2:
                    break
                buf.append(l2)
            resp.close()
            return status, b"".join(buf), rheaders, (time.time() - t0) * 1000, ttfb
        raw = resp.read()
        elapsed = (time.time() - t0) * 1000
        return status, raw, rheaders, elapsed, (time.time() - t0) * 1000
    except urllib.error.HTTPError as e:
        raw = e.read() if hasattr(e, "read") else b""
        return e.code, raw, dict(e.headers or {}), (time.time() - t0) * 1000, (time.time() - t0) * 1000
    except (urllib.error.URLError, OSError) as e:
        return None, str(e).encode(), {}, (time.time() - t0) * 1000, None


def run_matrix(r: Resolved) -> list:
    """Run the compatibility matrix; return a list of row dicts.

    Each row: {name, critical, ok, status, detail}. ``status`` is an HTTP code,
    or None on a connection-level failure (still-booting / unreachable).
    """
    key = r.client_key
    rows = []
    base = r.openai_base

    def add(name, critical, res, check):
        status, body, _h, elapsed, ttfb = res
        ok = False
        detail = ""
        if status is None:
            detail = f"unreachable: {body.decode(errors='replace')[:120]}"
        else:
            detail = check(status, body, elapsed, ttfb)
            ok = detail is None or detail == ""
            if detail is None:
                detail = f"{status} {elapsed:.0f} ms"
        rows.append({"name": name, "critical": critical, "ok": bool(ok),
                     "status": status, "detail": detail})

    def _json_body(body):
        try:
            return json.loads(body.decode())
        except Exception:
            return None

    # 1. GET /v1/models
    st, body, _h, el, tt = _http("GET", base + "/models",
                                 {"Authorization": f"Bearer {key}"}, None)
    def _c_models(status, body, el, tt):
        if status != 200:
            return f"expected 200, got {status}"
        j = _json_body(body)
        ids = [m.get("id") for m in (j.get("data") or [])] if isinstance(j, dict) else []
        if not ids:
            return "model list empty"
        if r.api_model not in ids:
            return f"served model {r.api_model!r} not in /v1/models list"
        return None
    add("GET /v1/models", True, (st, body, _h, el, tt), _c_models)

    chat_body = {"model": r.api_model,
                 "messages": [{"role": "user", "content": "Say hi"}],
                 "max_tokens": _MAX_TOKENS, "stream": False}
    def _c_chat(status, body, el, tt):
        if status != 200:
            return f"expected 200, got {status}: {body.decode(errors='replace')[:120]}"
        j = _json_body(body)
        if not (isinstance(j, dict) and j.get("choices")):
            return "no choices in response"
        return None
    # 2. chat non-stream
    res = _http("POST", base + "/chat/completions", {"Authorization": f"Bearer {key}"}, chat_body)
    add("POST /v1/chat/completions (non-stream)", True, res, _c_chat)
    # 3. chat stream
    chat_body["stream"] = True
    res = _http("POST", base + "/chat/completions", {"Authorization": f"Bearer {key}"},
                chat_body, stream=True)
    def _c_stream(status, body, el, tt):
        if status != 200:
            return f"expected 200, got {status}"
        if b"data:" not in body and b"chat.completion" not in body:
            return "no SSE data received"
        return None
    add("POST /v1/chat/completions (stream)", True, res, _c_stream)

    msg_body = {"model": r.api_model, "max_tokens": _MAX_TOKENS,
                "messages": [{"role": "user", "content": "Say hi"}]}
    def _c_msg(status, body, el, tt):
        if status != 200:
            return f"expected 200, got {status}: {body.decode(errors='replace')[:120]}"
        j = _json_body(body)
        if isinstance(j, dict) and (j.get("content") or j.get("type") or j.get("message")):
            return None
        return f"unexpected messages body: {body.decode(errors='replace')[:120]}"
    # 4. Anthropic messages (Bearer)
    res = _http("POST", base + "/messages",
                {"Authorization": f"Bearer {key}", "anthropic-version": ANTHROPIC_VERSION},
                msg_body)
    add("POST /v1/messages (Bearer)", True, res, _c_msg)
    # 5. Anthropic messages (x-api-key)
    res = _http("POST", base + "/messages",
                {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}, msg_body)
    add("POST /v1/messages (x-api-key)", True, res, _c_msg)

    # 6. OpenAI Responses API (Codex wire_api) — flags a missing /v1/responses
    resp_body = {"model": r.api_model, "input": "Say hi", "max_output_tokens": _MAX_TOKENS}
    res = _http("POST", base + "/responses", {"Authorization": f"Bearer {key}"}, resp_body)
    def _c_responses(status, body, el, tt):
        if status == 404:
            return "NOT IMPLEMENTED — Codex (wire_api=responses) will fail"
        if status != 200:
            return f"expected 200, got {status}: {body.decode(errors='replace')[:120]}"
        return None
    add("POST /v1/responses (Codex)", True, res, _c_responses)

    # 7. tool-calling probe
    tool_body = {
        "model": r.api_model,
        "messages": [{"role": "user", "content": "What is 2+2? Use the tool."}],
        "max_tokens": _MAX_TOKENS,
        "tools": [{"type": "function", "function": {
            "name": "add", "description": "add two numbers",
            "parameters": {"type": "object",
                           "properties": {"a": {"type": "integer"},
                                          "b": {"type": "integer"}}}}}],
    }
    res = _http("POST", base + "/chat/completions", {"Authorization": f"Bearer {key}"}, tool_body)
    def _c_tools(status, body, el, tt):
        if status != 200:
            return f"expected 200, got {status}"
        j = _json_body(body)
        ch = (j or {}).get("choices") or [{}]
        msg = (ch[0] or {}).get("message") or {}
        if msg.get("tool_calls"):
            return None
        if msg.get("content"):
            return "no tool_calls (model answered directly)"
        return "no message in tool probe"
    add("tool-calling probe", False, res, _c_tools)

    # 8. latency / TTFB (re-use a non-stream chat)
    t0 = time.time()
    res = _http("POST", base + "/chat/completions", {"Authorization": f"Bearer {key}"}, chat_body)
    status, body, _h, elapsed, ttfb = res
    def _c_latency(status, body, el, tt):
        if status != 200:
            return f"expected 200, got {status}"
        return f"TTFB {tt:.0f} ms, total {el:.0f} ms"
    rows.append({"name": "latency / TTFB", "critical": False,
                 "ok": status == 200, "status": status,
                 "detail": (f"TTFB {ttfb:.0f} ms, total {elapsed:.0f} ms"
                            if ttfb is not None else f"status {status}")})
    return rows


def matrix_exit_code(rows: list):
    """0 all critical pass; 1 some fail; 2 unreachable/auth; 3 still booting."""
    crit = [r for r in rows if r["critical"]]
    if not crit:
        return 0
    if all(r["ok"] for r in crit):
        return 0
    statuses = [r["status"] for r in crit]
    if all(s is None for s in statuses):
        return 2          # nothing reachable at all
    if all(s in (None, 401, 403) for s in statuses):
        return 2          # auth failure
    if any(s == 503 for s in statuses) and not any(
            s in (200, 400, 404, 422) for s in statuses):
        return 3          # still booting / warming
    return 1


def format_matrix(rows: list, r: Resolved) -> str:
    lines = [f"Compatibility matrix for {r.base_url} ({r.api_model}):", ""]
    for row in rows:
        mark = "PASS" if row["ok"] else "FAIL"
        lines.append(f"  [{mark}] {row['name']:<42} {row['detail']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# `ktl env` command (moved from launch.py; launch.py is now a thin dispatcher)
# ---------------------------------------------------------------------------
_SERVE_STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"


def _legacy_state() -> dict:
    """The pre-setup state file (~/.ktl/state.json), written by `serve`."""
    p = ktl_common.legacy_state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def cmd_env(args) -> int:
    """`ktl env` — print / write AI-client config pointing at the endpoint.

    Resolution precedence (flag > env var > config.json > legacy state > default):
      base URL : --url > $KTL_RELAY_URL > ~/.ktl/config.json > legacy state.json
      mode     : relay when $KTL_RELAY_URL or config relay_url is set, else state
      client key (relay)  : $KTL_CLIENT_API_KEY > config secrets.client_api_key
      client key (direct) : $KTL_API_KEY > ~/.kaggle-tpu-lab.json (serve's record)
      model    : --model > config > legacy state > qwen
    Returns the --test exit code (0/1/2/3), or 0 otherwise.
    """
    # --- restore mode ---
    if args.restore:
        targets = ([c for c in CLIENTS if c in WRITE_TARGETS]
                   if args.restore == "all" else [args.restore])
        for c in targets:
            if c not in WRITE_TARGETS:
                sys.exit(f"{c} is print-only — nothing to --restore.")
            for path, _kind in WRITE_TARGETS[c]:
                print(restore(_home(path)))
        return 0

    st = _legacy_state()
    cfg = ktl_common.load_config()
    relay_cfg_url = cfg.get("cloudflare", {}).get("relay_url", "")

    # --- base URL: --url > KTL_RELAY_URL > config.json > legacy state ---
    if args.url:
        base_url, url_source = normalize_root(args.url), "cli"
    elif os.environ.get("KTL_RELAY_URL", "").strip():
        base_url, url_source = normalize_root(os.environ["KTL_RELAY_URL"]), "env"
    elif relay_cfg_url.strip():
        base_url, url_source = normalize_root(relay_cfg_url), "config"
    elif st.get("base_url"):
        base_url, url_source = normalize_root(st["base_url"]), "state"
    else:
        sys.exit("No base URL. Pass --url, set KTL_RELAY_URL, run `python launch.py "
                 "setup`, or run `serve` first (it saves the endpoint to ~/.ktl).")

    # --- mode: relay vs direct (decides WHICH key the client uses) ---
    if os.environ.get("KTL_RELAY_URL", "").strip() or relay_cfg_url.strip():
        mode = "relay"
    elif st.get("mode") in ("relay", "direct"):
        mode = st["mode"]
    else:
        mode = "direct"

    # --- client key (never in legacy state; relay key: env > config.json) ---
    if mode == "relay":
        key, src = ktl_common.client_key_from_config(
            cfg, os.environ.get(ENV_VAR, "").strip())
        key_source = {"env": "env (relay)", "config": "config (relay)"}[src] \
            if src != "placeholder" else "placeholder"
        client_key, key_ph = key, src == "placeholder"
    else:  # direct
        env_key = os.environ.get("KTL_API_KEY", "").strip()
        if env_key:
            client_key, key_source, key_ph = env_key, "env (direct)", False
        elif os.path.exists(_SERVE_STATE_FILE):
            try:
                saved = json.loads(_SERVE_STATE_FILE.read_text()).get("api_key", "")
            except Exception:
                saved = ""
            if saved:
                client_key, key_source, key_ph = saved, "state (direct)", False
            else:
                client_key, key_source, key_ph = PLACEHOLDER, "placeholder", True
        else:
            client_key, key_source, key_ph = PLACEHOLDER, "placeholder", True

    # --- model: --model > config.json > legacy state > qwen ---
    if args.model:
        model_key, model_source = args.model, "cli"
    elif cfg.get("model") in MODELS:
        model_key, model_source = cfg["model"], "config"
    elif st.get("model") in MODELS:
        model_key, model_source = st["model"], "state"
    else:
        model_key, model_source = "qwen", "default"
    mc = MODELS[model_key]

    shell = args.shell or detect_shell()

    clients = CLIENTS if args.client == "all" else [args.client]

    resolved = {
        c: Resolved(
            client=c, model_key=model_key, api_model=mc["api_model"],
            context=mc["context"], max_output=mc["max_output"], cost=mc["cost"],
            base_url=base_url, client_key=client_key, key_is_placeholder=key_ph,
            key_source=key_source, url_source=url_source, shell=shell,
            reveal=args.reveal, responses_api=mc.get("responses_api", True))
        for c in clients
    }

    # --- --test compatibility matrix ---
    if args.test:
        r = resolved[clients[0]]
        deadline = time.time() + (args.wait or 0)
        while True:
            rows = run_matrix(r)
            code = matrix_exit_code(rows)
            if args.json:
                print(json.dumps({"rows": rows, "exit_code": code}, indent=2))
            else:
                print(format_matrix(rows, r))
            if code == 0:
                return 0
            if code in (2, 3) and time.time() < deadline:
                note = ("still booting/warming" if code == 3 else "unreachable")
                msg = f"{note} — retrying in 10 s " \
                      f"({int(max(0, deadline - time.time()))} s of --wait left)..."
                if args.json:
                    print(msg, file=sys.stderr)   # keep stdout a pure JSON doc
                else:
                    say(msg)
                time.sleep(10)
                continue
            return code

    # --- scope filter for --write (comma list of clients) ---
    scope = None
    if args.scope:
        scope = [s.strip() for s in args.scope.split(",") if s.strip()]
        bad = [s for s in scope if s not in WRITE_TARGETS]
        if bad:
            sys.exit(f"--scope has non-writable client(s): {', '.join(bad)} "
                     f"(writable: {', '.join(WRITE_TARGETS)})")

    # --- render + write ---
    blocks = []
    for c in clients:
        if len(clients) > 1:
            blocks += ["=" * 66, f" {c}", "=" * 66, ""]
        blocks.append(render(resolved[c]))
        blocks.append("")

    write_reports = []
    if args.write or args.dry_run:
        for c in clients:
            if scope and c not in scope:
                continue
            rep = apply_write(resolved[c], args.dry_run)
            write_reports.append(rep)

    if args.json:
        print(json.dumps({
            "resolution": {
                "mode": mode,
                "model": model_key, "api_model": mc["api_model"], "model_source": model_source,
                "base_url": base_url, "url_source": url_source,
                "key_source": key_source, "shell": shell,
            },
            "outputs": {c: "\n".join(b for b in [render(resolved[c])]) for c in clients},
            "write": write_reports,
        }, indent=2))
    else:
        say(f"mode     : {mode}   "
            + ("(clients use KTL_CLIENT_API_KEY)" if mode == "relay"
               else "(clients use the key serve printed / KTL_API_KEY)"))
        say(f"model    : {model_key} -> {mc['api_model']}   (source: {model_source})")
        say(f"base URL : {base_url}   (source: {url_source})")
        key_desc = {
            "env (relay)": "env $KTL_CLIENT_API_KEY",
            "env (direct)": "env $KTL_API_KEY",
            "config (relay)": "config (~/.ktl/config.json)",
            "state (direct)": "state (~/.kaggle-tpu-lab.json)",
            "placeholder": ("PLACEHOLDER — set KTL_CLIENT_API_KEY"
                            if mode == "relay" else "PLACEHOLDER — set KTL_API_KEY"),
        }[key_source]
        say(f"API key  : {key_desc}" + ("" if args.reveal else "   [hidden; --reveal to show]"))
        say(f"shell    : {shell}")
        print()
        print("\n".join(blocks))
        if write_reports:
            print("--- write ---")
            print("\n".join(write_reports))
    return 0
