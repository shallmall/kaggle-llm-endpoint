# Setup & doctor — verification notes

What `python launch.py setup` / `doctor` rely on, and **what was actually
verified vs inferred**, so the wizard isn't built on unchecked assumptions.
Re-verify when wrangler or the Kaggle CLI changes a behavior the wizard
parses.

**Last verified: 2026-09-29**

## Verified (external, this cycle)

| Fact | Evidence | Used by |
|------|----------|---------|
| wrangler stable = **4.143.0** | npm registry `wrangler` dist-tags (2026-09-29) | pinned `WRANGLER_MAJOR = 4` → `npx -y wrangler@4 …` |
| wrangler requires **Node ≥ 22.0.0** | npm registry `wrangler` `engines.node` (2026-09-29) | `NODE_MIN = (22, 0, 0)` gate in prereqs/doctor |
| `wrangler secret put` **accepts the value on STDIN** | official Cloudflare docs (command reference) | `_put_worker_secrets` — values never touch argv |
| `wrangler deploy` **prints `https://<name>.<acct>.workers.dev`** | official docs + CLI output format | URL capture regex in the worker step (with paste fallback) |
| `wrangler whoami` prints **`Account Name: …` / `Logged in as: …`**; non-zero exit when logged out | CLI behavior | `_parse_whoami` + login detection (parsed defensively) |
| Kaggle auth mechanisms: **`kaggle auth login`** (OAuth), **`KAGGLE_API_TOKEN`** env var, **`~/.kaggle/access_token`**, legacy **`~/.kaggle/kaggle.json`** | Kaggle CLI docs + local `kaggle 2.2.4` install (2026-09-29) | credential detection in the kaggle step / doctor |
| API token page: **kaggle.com/settings/api** | Kaggle settings UI | all "create a token" hints |
| kaggle CLI recommends **Python 3.11+** | Kaggle CLI docs (2026-09-29) | prereqs note (warns below 3.11, hard-fails below 3.10) |
| Free TPU quota ≈ **20 h/week**, session cap ~9 h | Kaggle docs / ToS notes | GATE 3 quota warning (always logged), `keepalive_min` 480 |

## Verified (this machine, 2026-09-29)

| Fact | Evidence |
|------|----------|
| Python 3.12.3 system interpreter | `python3 --version` |
| kaggle CLI **2.2.4** installed as a standalone bin (`~/.local/bin/kaggle`), **not** importable by the system python | `kaggle --version`; `python3 -m kaggle` fails |
| Node **v26.7.0** + npx 11.19.0 | `node --version`, `npx --version` |
| `~/.kaggle/kaggle.json` (legacy creds) present | local home |

The "standalone bin, not `-m`-importable" row is why `ktl_common.kaggle_cmd()`
prefers `shutil.which("kaggle")` and only falls back to
`[sys.executable, "-m", "kaggle"]`.

## Inferred / not verifiable offline

| Item | Status | Why it's fine |
|------|--------|---------------|
| `wrangler login` behavior when the browser window is closed | INFERRED | wizard re-runs `whoami` after login and fails with a clear re-run hint on any mismatch |
| Exact Kaggle quota number (20 h/week) | INFERRED | wizard wording says "~20 h/week"; the warning is always logged regardless |
| Qwen serving `/v1/responses` | mock-confirmed only | same open item as the env matrix — run `env --test --model qwen` live to lock it in |
| Real-client behavior of each `env` config | per-client docs verified 2026-09-29 | see `docs/clients-verification.md` (statuses + sources) |

## Offline verification of the wizard itself

The setup/doctor suite (`tests/test_setup_common.py`,
`tests/test_setup_wizard.py`) runs the wizard end-to-end **without network or
accounts**: fake `npx`/`kaggle`/`node` executables on a temp PATH record argv
and stdin; the TPU push is stubbed; the verify/doctor HTTP checks run against
a localhost mock server. Assertions include: secrets reach wrangler **only on
stdin** (never argv), the config file is 0600/atomic, gates block on "n",
re-runs are byte-identical no-ops, and `doctor` leaves the whole HOME/KTL_HOME
tree untouched (zero side effects).
