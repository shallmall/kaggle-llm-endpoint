# Plan: guided installer (`setup`) + `doctor` + unified config

Status: READY TO EXECUTE. All Step-0 inspection done; external facts verified;
both design questions answered by the user (unittest for new tests; serve code
goes to `ktl_serve.py`). Awaiting plan-mode exit to start writing code.

## Step 0 findings (short plan, as required)

```
ENV VARS (all kept working): KTL_RELAY_URL, KTL_RELAY_UPDATE_SECRET, KTL_API_KEY (direct key),
  KTL_CLIENT_API_KEY (relay key), KTL_TUNNEL_TOKEN/HOSTNAME (named tunnel), KAGGLE_API_TOKEN (kaggle's).
FLAGS (existing, unchanged): env: --model/--url/--shell/--write/--scope/--dry-run/--reveal/--test/
  --wait/--json/--restore; serve: --model/--user/--slug/--keepalive-min + all model flags.
ktl_env.py exposes: Resolved, render(), apply_write(), WRITE_TARGETS (list of (path,kind) pairs),
  run_matrix()/matrix_exit_code()/format_matrix(), PLACEHOLDER, ENV_VAR, backup()/restore(), _redact().
MOVE to ktl_common.py (NEW): KTL_HOME + config.json load/save/migrate (atomic 0600), key gen
  (secrets.token_urlsafe(32)), mask ("abcd…wxyz"), redact(), subprocess helper (timeout + captured
  stderr + redacted logs to ~/.ktl/setup.log + friendly hint mapping), prompts (confirm/ask/choose,
  --yes/--dry-run aware), kaggle_cmd() [shutil.which "kaggle" -> fallback sys.executable -m kaggle;
  current -m-only call BREAKS where kaggle is a standalone bin (true on this machine: CLI 2.2.4 at
  ~/.local/bin/kaggle, not importable by system python)], node_version()/npx_cmd()/wrangler_cmd().
ADD ktl_setup.py (NEW): setup wizard (8 steps, 4 approval gates) + doctor (read-only).
ADD ktl_serve.py (NEW): serve/build-env/status/stop + kernel build/watch move out of launch.py;
  exposes push_and_watch(...) for the wizard; watch gains stop_after_ready for setup.
REFRCTOR ktl_env.py: import PLACEHOLDER/ENV_VAR + config/URL resolution from ktl_common (re-exported
  so tests still see ktl_env.PLACEHOLDER); cmd_env() moves into ktl_env.py; behavior unchanged.
CONFIG: ~/.ktl/config.json (KTL_HOME override), precedence flag > env > config > default, legacy
  ~/.ktl/state.json migrated on first read then untouched; .gitignore += .ktl/ *.ktl-bak-*;
  git-tree literal-secret warning + gitignore offer.
```

## Verified external facts (all 2026-09-29, sources for docs/setup-verification.md)

| Fact | Value | Source (VERIFIED) |
| --- | --- | --- |
| wrangler latest stable | 4.143.0 → pin `WRANGLER_MAJOR = 4` (`npx -y wrangler@4 …`) | registry.npmjs.org/wrangler/latest |
| Node minimum for wrangler 4 | >= 22.0.0 (`NODE_MIN = (22,0,0)`) | same registry metadata, `engines` field |
| `wrangler secret put <KEY>` accepts piped STDIN | documented: `echo … \| wrangler secret put KEY` | developers.cloudflare.com/workers/wrangler/commands/workers/ (updated 2026-09-22) |
| `wrangler deploy` | never deletes secrets; prints `https://<sub>.workers.dev` (regex `https://[a-z0-9.-]+\.workers\.dev`, fallback: paste prompt) | same page |
| Kaggle auth mechanisms (4) | `kaggle auth login` (OAuth), `KAGGLE_API_TOKEN` env, `~/.kaggle/access_token` file, legacy `~/.kaggle/kaggle.json`; token page = kaggle.com/settings/api | github.com/Kaggle/kaggle-api/blob/master/docs/README.md |
| Kaggle CLI Python requirement | docs state Python 3.11+ (tool itself stays 3.10+; warn only) | same kaggle-api docs |
| Worker secrets | `CLIENT_API_KEY` + `UPDATE_SECRET`, both must be ≥ 32 chars | worker/worker.js source (source-confirmed) |
| `whoami` output format | INFERRED (parse account name defensively; never hard-required) | — |

Local machine facts: Python 3.12.3, kaggle CLI 2.2.4 (standalone bin), Node v26.7.0, npx 11.19.0,
`~/.kaggle/kaggle.json` present.

## Files

### 1. `ktl_common.py` (NEW, ~330 lines) — drafted in full
- Constants: `CONFIG_VERSION=1`, `ENV_VAR`, `PLACEHOLDER`, `WRANGLER_MAJOR=4`,
  `NODE_MIN=(22,0,0)`, `KEY_BYTES=32`, `STEP_NAMES`, `DEFAULT_WORKER_NAME`,
  `KAGGLE_TOKEN_URL`, `NODE_INSTALL_URL`.
- Paths: `ktl_home()` ($KTL_HOME else ~/.ktl), `config_path()`,
  `legacy_state_path()`, `log_path()`.
- Config: `DEFAULTS` (exact schema from the task), `_deep_merge`, `load_config(migrate=True)`
  (missing config + legacy state → `migrate_legacy_state()` → save → re-read; corrupt → defaults),
  `save_config()` (atomic tmp+rename, 0700 dir / 0600 file on POSIX, `version`, `updated_at` ISO),
  `update_config(**partial)`, `set_step/step_status`,
  `migrate_legacy_state()` (model; relay-mode `base_url` → `cloudflare.relay_url`; username from
  `~/.kaggle/kaggle.json`; direct quick-tunnel URL NOT copied — ephemeral; state.json untouched),
  `resolve(cli, env, config_value, default)`,
  `client_key_from_config(cfg, env_key) -> (key, source)`.
- Keys/masking: `gen_key() = secrets.token_urlsafe(32)` (43 chars ≥ Worker's 32),
  `mask()` → `abcd…wxyz` (≤8 chars → `****`), `redact()` (same secret-name regex as
  ktl_env._redact, masks `*TOKEN/*KEY/*SECRET/*PASSWORD/*PASSWD` =value tokens).
- Subprocess: `SubprocessError(msg, hint)`, `hint_for_output()` (not-logged-in / 401 / 403 /
  network / timeout / old Node / Kaggle quota / kaggle credentials → friendly one-liners),
  `run_logged(cmd, stdin=, timeout=, cwd=, inherit_stdio=, log_line=)` — timeout → friendly error;
  captured output printed + forwarded redacted; `inherit_stdio` for interactive flows
  (`wrangler login`).
- Tools: `kaggle_cmd()` (which → `-m` fallback), `run_kaggle()`, `kaggle_version()`,
  `kaggle_authed() -> (ok, detail)` (cheap `kaggle kernels list -m --page-size 1`),
  `kaggle_username(cli_arg)` (file → `kaggle config view` regex → ""), `node_version()`,
  `npx_cmd()` (Windows resolves npx.cmd via PATHEXT), `wrangler_cmd(*args)`.
- Prompts: `Ctx` dataclass (`yes`, `dry_run`, `verbose`, in-memory `log_lines` + `log()`),
  `confirm/ask/choose/wait_for_enter` — dry-run: log the question, no-op; --yes: log
  auto-approval, default answer / select-all.
- Git guard: `inside_git_worktree()`, `gitignore_guard(ctx, target, literal)` — literal (not
  placeholder) target inside a git tree → warn + offer `.gitignore` entry (append at repo root;
  --yes auto-applies; never blocks, never rewrites existing entries).

### 2. `ktl_serve.py` (NEW, ~450 lines) — verbatim extraction from launch.py
Moves: `MODELS` → actually goes to **ktl_common.py** (shared by serve+env), `PHASE_TEXT`,
`check_auth()` (now wraps `ktl_common.kaggle_authed()` + sys.exit on failure), `kaggle_username`
(exit variant), `embed_engine`, `build_kernel`, `read_events`, `register_with_relay`,
`relay_from_env`, `write_env_state`, `render_event`, `watch`, `load_state`,
`cmd_serve/cmd_build_env/cmd_status/cmd_stop`, `STATE_FILE` (~/.kaggle-tpu-lab.json — kept as
serve's RUNTIME record: kernel/topic/api_key/model).
New shape: `cmd_serve(args)` → `push_and_watch(model_key, user, slug, relay, keepalive_min,
…flag params…, watch=True, stop_after_ready=False)` (existing body; returns state dict).
`watch(..., stop_after_ready=False)` — returns right after the READY banner (used by setup so the
wizard can continue to verify/clients). `write_env_state` additionally calls
`ktl_common.update_config(model=model_key)` (config.json becomes the durable record; state.json
still written for compat/direct mode). `kaggle()` helper replaced by `ktl_common.run_kaggle`
(which fixes the standalone-bin case). `say()` imported from ktl_common.

### 3. `ktl_env.py` (REFACTORED, behavior unchanged, tests stay green)
- `from ktl_common import PLACEHOLDER, ENV_VAR, resolve, client_key_from_config, ktl_home,
  legacy_state_path, redact, mask, gen_key, say` — re-exported so `ktl_env.PLACEHOLDER` etc. keep
  working (tests import them).
- `cmd_env(args)` moves here (launch.py no longer holds it). Resolution order (all existing
  behavior + config.json slot inserted per precedence rule):
  - mode: `KTL_RELAY_URL` env → config `cloudflare.relay_url` present → "relay" → legacy
    state `mode` → "direct"
  - base URL: `--url` > `KTL_RELAY_URL` > config `cloudflare.relay_url` > legacy state `base_url`
    > exit with the current error message
  - client key (relay): `KTL_CLIENT_API_KEY` env > config `secrets.client_api_key` > placeholder
    (key_source labels gain "config (~/.ktl/config.json)")
  - client key (direct): `KTL_API_KEY` env > `~/.kaggle-tpu-lab.json` api_key > placeholder
  - model: `--model` > config `model` > legacy state `model` > "qwen"
- `gitignore_guard` hook into `_emit` when a LITERAL (non-placeholder) value is being written
  (protects `--reveal` file writes inside git trees).
- `--test` logic, writers, `_redact` (keeps `***REDACTED***` token — its tests assert it),
  matrix code: untouched.

### 4. `launch.py` (SHRINKS 846 → ~200 lines)
argparse dispatcher only: `serve | build-env | status | stop | env | setup | doctor`.
Keeps: serve default-filling (`reasoning_effort`, `max_model_len`), the
`rc = args.fn(args); if rc is not None: sys.exit(rc)` exit-code contract, per-subcommand flag
definitions VERBATIM (env/serve flags unchanged). Docstring lists all commands.

### 5. `ktl_setup.py` (NEW, ~450 lines)
`cmd_setup(args)`:
- `--reset` first: confirm → delete config.json only (never the Worker) → continue fresh.
- `Ctx` from flags; `cfg = load_config()`; `model = resolve(args.model, None, cfg["model"], "qwen")`;
  `--only STEP` restricts; each step: skip when `step_status == "done"` (resume), save config
  AFTER each completed/failed step (Ctrl-C safe), wizard log → `~/.ktl/setup.log` (redacted,
  appended).
- **1 prereqs** — Python ≥3.10 (running is proof; warn <3.11 re kaggle CLI), `kaggle` CLI
  (which → `-m`; version via `kaggle --version`; missing → print `pip install kaggle` + offer to
  run it with approval — never auto-install Node), Node ≥ 22 + npx (missing/too-old → print
  nodejs.org; no auto-install).
- **2 kaggle** — credential presence: `KAGGLE_API_TOKEN` env → `~/.kaggle/access_token` →
  `~/.kaggle/kaggle.json`; if none: show KAGGLE_TOKEN_URL + offer `kaggle auth login` (OAuth,
  inherit stdio) or wait-for-Enter after user creates/places creds; VALIDATE via
  `kaggle_authed()`; save username only (`kaggle_username()`).
- **3 cloudflare** — `wrangler whoami` (timeout 120; first npx run downloads); not logged in →
  `wrangler login` (browser, inherit stdio, wait) → re-whoami; display account name (parsed
  defensively).
- **4 secrets** — generate `client_api_key` + `update_secret` if empty (masked display, save).
  `--rotate`: approval (warning: invalidates stored Worker config until next serve) → new values →
  re-put via Worker (deferred if worker step not done).
- **5 worker** — name: `--worker-name` > config > `kaggle-tpu-relay`.
  **GATE 1** (account + worker name + what deploys: worker/ dir, 2 secrets set after).
  Deploy `npx -y wrangler@4 deploy --name NAME` cwd=worker/ (redacted capture) → URL regex,
  fallback paste prompt → **GATE 2** (confirm URL) → `wrangler secret put CLIENT_API_KEY` /
  `UPDATE_SECRET` with value on **STDIN only** (never argv — tests assert) → save
  `cloudflare.{worker_name,relay_url}`.
  `--adopt-worker URL`: no deploy; confirm; put both secrets; save URL.
- **6 serve** — **GATE 3** quota warning (always logged even with --yes): "starting a TPU
  session consumes weekly quota; model: X" → `ktl_serve.push_and_watch(...)` with relay from
  config (direct fallback + notice if none), `stop_after_ready=True`; done on READY.
- **7 verify** — build `Resolved` (relay mode when relay set), `ktl_env.run_matrix()` +
  `format_matrix()`, print, report pass/fail (not persisted in steps).
- **8 clients** — `shutil.which` for claude/codex/opencode/hermes/aider → **GATE 4** multi-select
  (choose) → per client: build `Resolved` (same resolution as `env`), `ktl_env.apply_write()`
  (backups/managed blocks/idempotent) → print launch command per client
  (`claude` / `codex -c model_provider=kaggle-tpu -c model=…` / `opencode` / `hermes` / `aider`).
- Final summary: endpoint URL, masked client key, "re-run changes nothing", `python launch.py env`.

`cmd_doctor(args)` — ZERO side effects (stdout only, no config writes, no setup.log):
  python version • kaggle CLI presence + version + credential presence (yes/no, never values) +
  `kaggle_authed()` • node/npx versions • `wrangler whoami` (login state/account) • config path +
  permissions (octal) + steps state + secrets present (yes/no) • saved relay URL •
  `GET {relay}/v1/models` with stored client key (5 s timeout; only when URL+key exist) • full
  `--test` matrix when reachable • per-failure "fix hint" (hint_for_output + step-specific) •
  exit 0 = healthy (prereqs + kaggle auth + [if relay configured: worker reachable & critical
  matrix rows pass]; missing config/relay → exit 1 with "run setup" hint).

Flags (setup): `--yes --dry-run --only STEP --model --worker-name --adopt-worker --rotate
--reset --verbose`. `--dry-run`: every action printed, nothing written (config, files,
subprocesses). `--only`: one of prereqs|kaggle|cloudflare|secrets|worker|serve|verify|clients
(verify allowed for one-shot re-test).

### 6. `.gitignore`
Add `.ktl/` and `*.ktl-bak-*` (`.env`/`*.log`/`.kaggle-tpu-lab.json` already present).

## Tests (unittest, stdlib, no network, temp `KTL_HOME` + temp PATH everywhere)

`tests/test_ktl_common.py`
- config round-trip, defaults merge, atomic (no `.tmp-*` left), POSIX 0600, version/updated_at.
- `resolve()` precedence (flag > env > config > default; empty strings skipped).
- `mask()` shapes; `redact()` masks `GITHUB_TOKEN=…`/`KTL_CLIENT_API_KEY=…`, keeps URLs/placeholder.
- migration: relay state.json → config (model + relay_url), direct state.json → no relay_url,
  state.json untouched (byte-identical), migrate-once (second load doesn't clobber edits).
- `gen_key()` ≥ 32 chars, unique.
- `kaggle_cmd()`: fake `kaggle` on temp PATH wins; removed → `[sys.executable, "-m", "kaggle"]`.
- `node_version()` via fake `node` (v22.1.0 → (22,1,0); v18.0.0 → below min); `npx_cmd()`;
  Windows: mocked `os.name="nt"` + `shutil.which` → `npx.cmd` path handled list-form.
- `run_logged()`: exit-code → SubprocessError with hint mapping (401/not-logged-in/quota/timeout);
  redaction (fake cmd echoes a secret → stdout/log show masked); `stdin=` passes value without
  argv exposure (fake binary records argv to a file); `inherit_stdio` path.
- prompts: confirm/ask/choose with Ctx(yes=True) / Ctx(dry_run=True) / mocked `input`.
- `gitignore_guard()`: temp git repo + literal → .gitignore entry appended; placeholder → no-op;
  non-repo → no-op; idempotent (no duplicate entry). (skipUnless git present)

`tests/test_ktl_setup.py`
- Fake `npx`/`kaggle`/`node` executables (shell scripts) on a temp PATH: emit canned outputs
  driven by a scenario file; RECORD argv + stdin + cwd to JSON for assertions.
- `--dry-run --yes` full run: asserts config.json absent/unchanged, no fake invocations, all
  actions printed.
- secrets step: two keys ≥32 chars, masked in captured stdout (literal absent), config file 0600
  (POSIX), secrets present in config.
- worker step: deploy output WITH url → saved; WITHOUT url → paste prompt (mocked input);
  `secret put` argv == [npx, -y, wrangler@4, secret, put, NAME] (NO secret in argv) and recorded
  stdin == the secret; GATE 1: input "n" → deploy never invoked, step not done; `--yes` → passes.
- adopt: `--adopt-worker URL` → deploy not called, both secrets put, URL saved.
- `--rotate`: new values ≠ old, secret re-put, old value never printed.
- resume: pre-seeded steps {prereqs:done, kaggle:done} → their fake commands NOT invoked; a
  failing cloudflare (scenario) → step failed + config saved; re-run success → continues.
- idempotency: all-done config → re-run prints skip messages, config bytes identical (no save).
- `--reset`: confirm "y" → config.json deleted; "n" → kept.
- `--only secrets` → only that step runs.
- doctor: dir snapshot (files + mtimes) before/after → identical (zero side effects); report
  contains masked (not literal) keys; unreachable URL → fix hint + exit 1; local
  `http.server` (localhost only) serving /v1/models + /v1/chat/completions → matrix section
  printed, exit 0.

Existing 72 ktl_env tests must stay green (import shims + behavior preservation).

## Docs

- README: Quick start → `clone → cd kaggle-llm-endpoint → python launch.py setup` (fresh machine
  reaches passing --test with 4 approval prompts + browser/token logins); manual path
  (current "What you need" + relay-by-hand details) moved to **docs/setup-manual.md**; new
  "Where your config lives" section (~/.ktl/config.json paths, 0600/0700, what IS stored
  [username, worker name, relay URL, 2 keys] and what is NEVER stored [Kaggle token/creds,
  direct-session key], how to reset: `setup --reset`, `env --restore <client>`, key rotation).
- **docs/setup-verification.md**: the fact table above (VERIFIED URL + date vs INFERRED).
- docs/clients*.md untouched (client formats unchanged).

## Acceptance mapping

| Criterion | How met |
| --- | --- |
| Fresh machine → passing --test with 4 prompts + 2 logins | wizard steps 1-8; gates 1-4; browser logins = `wrangler login` + Kaggle token/OAuth |
| Re-run changes nothing and says so | done-step skip + idempotency test (config bytes identical) |
| doctor output contains no secrets | redaction pipeline + test asserting masked-only |
| launch.py shorter than before | 846 → ~200 lines (serve→ktl_serve, env→ktl_env, setup/doctor→ktl_setup) |
| env behavior unchanged | 72 existing tests green; resolution order adds config.json slot between env and legacy state |
| End: verified vs inferred list | docs/setup-verification.md + final chat summary |

## Execution order

1. `ktl_common.py` → 2. `ktl_serve.py` (extract) → 3. `ktl_env.py` (shims + cmd_env) →
   4. `launch.py` (thin dispatcher) → 5. `ktl_setup.py` → 6. `.gitignore` →
   7. tests (run suite after each stage; green at every stop) → 8. docs →
   9. final verification: full suite, `doctor` smoke (no config → exit 1 + hint),
   `setup --dry-run --yes` smoke, argparse for all 7 subcommands, no-secrets sweep of all
   printed output (grep for the 3 known leaked fragments + github_pat). No live endpoints
   touched, no commits.
