# Client compatibility — verification report

`python launch.py env <client>` generates config for 7 AI clients. This report
records, **per client**, whether the config format was actually checked against a
source this session, and what the live `--test` matrix shows.

**Date: 2026-09-29.**

**Method / what "status" means**

- **VERIFIED** — read the official docs *this session* (URL + date below) and/or
  read the backend source in this repo. Nothing is marked VERIFIED on memory alone.
- **INFERRED** — standard/well-known format from memory or third-party guides;
  not re-fetched against a live doc this session. The uncertainty is stated.

Two things are independent and both are tracked here:

1. **Is the *client config* correct?** (env var names, file shape, base-URL rule) —
   the table below.
2. **Does the *backend* actually serve the route the client needs?** — the `--test`
   matrices + the backend route evidence below. A correct config still fails if the
   backend doesn't implement the route (e.g. Codex needs `/v1/responses`).

---

## 1. Client config table

| Client | Config format / path(s) | Env var / key names | Base-URL rule | Status |
| --- | --- | --- | --- | --- |
| **Claude Code** | `~/.claude/settings.json` → `env` object (file override of shell) | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (sent as `Authorization: Bearer`), `ANTHROPIC_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL` (deprecated) + `ANTHROPIC_DEFAULT_HAIKU_MODEL` (current) for background tasks | **root, no `/v1`** — the client appends `/v1/messages` itself | **VERIFIED** — live doc `https://code.claude.com/docs/en/env-vars` read 2026-09-29; + repo `qwen38-27b/README.md:116-127` (Claude Code verified end-to-end: Bearer-only → `ANTHROPIC_AUTH_TOKEN`, bare model id) |
| **Codex CLI** | `~/.codex/config.toml` → managed `[model_providers.kaggle-tpu]` block (no top-level `model`/`model_provider` written); run via `-c model_provider=kaggle-tpu -c model=<id>` | key via `env_key = "KTL_CLIENT_API_KEY"` (never stored in the file) | **root + `/v1`** (`base_url`); `wire_api = "responses"` → client calls `POST /v1/responses` | **VERIFIED** — official reference `https://developers.openai.com/codex/config-reference` + `https://developers.openai.com/codex/developer-commands` read 2026-09-29: `model_providers.<id>` custom-provider table with `.name`/`.base_url` ("API base URL for the model provider")/`.env_key` ("Environment variable supplying the provider API key"); **`wire_api` — `responses` is the only supported value and the default**; `-c key=value` command-line overrides documented. |
| **opencode** | `~/.config/opencode/opencode.json` → `provider.kaggle-tpu` (`npm`, `name`, `options`, `models`) | `options.apiKey` (static file — **no env-ref support**, so the key is written to the file, `chmod 600`, or a placeholder) | **root + `/v1`** (`options.baseURL` includes `/v1`) | **VERIFIED** — live doc `https://opencode.ai/docs/providers/` read 2026-09-29 (Custom-provider + `@ai-sdk/openai-compatible` examples all show `baseURL …/v1`; model key must match `GET /v1/models` id) |
| **Hermes** | `~/.hermes/.env` (key line) + top-level `model:` block in `~/.hermes/config.yaml` (`provider: custom`, `api_mode: chat_completions`, `api_key: ${KTL_CLIENT_API_KEY}`) | `KTL_CLIENT_API_KEY` in `.env`, referenced from the YAML (`${...}` expands from env) | **root + `/v1`** (`model.base_url` — chat_completions transport calls `POST {base_url}/chat/completions`) | **VERIFIED** — read the *installed* hermes-agent source (`~/.hermes/hermes-agent`, 2026-09-29): `.env` is loaded before `config.yaml` (`hermes_cli/main.py:721`); bare `openai` is **aliased to openrouter** and `openai-api` uses the `codex_responses` transport (`hermes_cli/providers.py`), so `OPENAI_API_KEY`/`OPENAI_BASE_URL` do **not** route to a plain compat endpoint — the earlier generator was wrong; the `provider: custom` block is the working shape (matches the operator's own live `config.yaml`). Also confirmed `openai_api_models.yaml` (older docs) is referenced **nowhere** in the source — dead file. |
| **aider** | print-only (env vars, or `.aider.conf.yml`) | `AIDER_MODEL`, `OPENAI_API_KEY`, **`OPENAI_API_BASE`** (or `model`, `openai-api-key`, `openai-api-base`); model needs the **`openai/<name>`** prefix | **root + `/v1`** | **VERIFIED** — official doc `https://aider.chat/docs/llms/openai-compat.html` read 2026-09-29: `export OPENAI_API_BASE=<endpoint>` + `aider --model openai/<model-name>`. The earlier generator emitted `OPENAI_BASE_URL` + a bare model id — **both wrong**, fixed (§6). |
| **curl** | print-only (one OpenAI + one Anthropic command) | headers: `Authorization: Bearer` (OpenAI); `x-api-key` + `anthropic-version` (Anthropic) | OpenAI full URL `root+/v1/chat/completions`; Anthropic full URL `root+/v1/messages` | **INFERRED** — trivial raw HTTP; base-URL rule is the standard split |
| **python** | print-only (SDK snippets) | `OpenAI(base_url, api_key)`; `Anthropic(base_url, api_key)` | OpenAI SDK `base_url = root+/v1`; Anthropic SDK `base_url = root` (SDK appends `/v1/messages`) | **INFERRED** — standard SDK constructor semantics |

**The base-URL rule in one line** (unchanged, now verified for the two strict cases):
Anthropic-API clients (Claude Code, Anthropic Python SDK) use the **root** (they append
`/v1/messages`); every OpenAI-compatible client (Codex, opencode, Hermes, aider, OpenAI
SDK, curl) uses **root + `/v1`**.

---

## 2. Backend route evidence (what `--test` is actually hitting)

The relay Worker is a **blanket proxy** (`worker/worker.js`): it authenticates the
client against `CLIENT_API_KEY`, then forwards **every path and header unchanged** to
the registered Kaggle quick-tunnel, rewriting only the backend auth headers. It does
**no path translation and no model-name rewriting.** So which routes exist is decided
entirely by the **backend**:

- **Qwen backend** = vLLM (`vllm-tpu==0.28.0`, `vllm.entrypoints.openai.api_server`;
  `qwen38-27b/kernel/serve_qwen38.py:343`). Serves the OpenAI API. The project README
  (`qwen38-27b/README.md:116`) states the bundled vLLM **also exposes an
  Anthropic-compatible `/v1/messages`** and that Claude Code was verified end-to-end
  against it. `/v1/responses` (OpenAI Responses API) is expected from this vLLM
  version but **not** independently confirmed here → see §4.
- **GLM backend** = custom TPU engine (`glm53-flash/kernel/serve_glm53.py`). Its
  `do_POST` (`:972-995`) handles only `/v1/messages/count_tokens`, `/v1/messages`,
  `/v1/chat/completions`, `/v1/completions`; `do_GET` (`:954-962`) handles `/health`
  and `/v1/models`. **Anything else → 404** (`:995`). So **`/v1/responses` is 404 —
  source-confirmed**, not an assumption.

> Note on model names: the relay does not rewrite model ids, and vLLM validates the
> `model` field against the served model. So clients must send the **bare** id
> (`qwen3.8-27b` / `glm-5.3-flash`), not a `kaggle-tpu/…` prefix. The GLM engine
> ignores the model field (single loaded model); vLLM does not.

---

## 3. `--test` matrices (this session)

**Endpoint used: a local mock server** (`/tmp/opencode/mock_backend.py`), *not* a live
Kaggle session. Reason: `KTL_RELAY_URL` is set but `KTL_CLIENT_API_KEY` is not, so the
live relay returns 401, and no live TPU session is registered here. The mock is
configured to mirror each backend's **source-confirmed** route set:

- GLM mock: `/v1/responses` → **404** (matches `serve_glm53.py`), all other routes 200.
- Qwen mock: all routes 200 (incl. `/v1/responses` and `/v1/messages`), per the vLLM
  README claims. The Qwen `/v1/responses` row is therefore **assumed**, not proven —
  a live `--test` is required to confirm (see §5).

Run with the real `launch.py env --test` (exit code now propagates correctly):

### `python launch.py env --test --model glm` (mock) — exit 1
```
Compatibility matrix for http://127.0.0.1:44087 (glm-5.3-flash):

  [PASS] GET /v1/models                             200 66 ms
  [PASS] POST /v1/chat/completions (non-stream)     200 1 ms
  [PASS] POST /v1/chat/completions (stream)         200 1 ms
  [PASS] POST /v1/messages (Bearer)                 200 2 ms
  [PASS] POST /v1/messages (x-api-key)              200 1 ms
  [FAIL] POST /v1/responses (Codex)                 NOT IMPLEMENTED — Codex (wire_api=responses) will fail
  [PASS] tool-calling probe                         200 1 ms
  [PASS] latency / TTFB                             TTFB 2 ms, total 2 ms
```

### `python launch.py env --test --model qwen` (mock) — exit 0
```
Compatibility matrix for http://127.0.0.1:43615 (qwen3.8-27b):

  [PASS] GET /v1/models                             200 66 ms
  [PASS] POST /v1/chat/completions (non-stream)     200 1 ms
  [PASS] POST /v1/chat/completions (stream)         200 1 ms
  [PASS] POST /v1/messages (Bearer)                 200 2 ms
  [PASS] POST /v1/messages (x-api-key)              200 1 ms
  [PASS] POST /v1/responses (Codex)                 200 1 ms
  [PASS] tool-calling probe                         200 1 ms
  [PASS] latency / TTFB                             TTFB 2 ms, total 2 ms
```

---

## 4. Decision rule — Codex & `/v1/responses`

Rule: *mark Codex supported only where `--test` shows `/v1/responses` passing; never
advertise a client as working otherwise.*

- **GLM — `/v1/responses` FAILS (404, source-confirmed in `serve_glm53.py:995`).**
  **Decision: mark Codex UNSUPPORTED for GLM.** `ktl env codex --model glm` now prints
  a clear `!!! NOT SUPPORTED FOR THIS MODEL !!!` warning instead of a config that
  won't work, and the README reflects it.
  **Why not "add the route":** implementing the OpenAI **Responses API** (a distinct
  request/response/streaming shape) inside the custom GLM TPU engine is a substantial
  feature change, well outside the scope of this verification pass. Marking unsupported
  is honest, safe, and reversible.
- **Qwen — `/v1/responses` expected to PASS** (vLLM serves the Responses API), but this
  is **INFERRED** (vllm-tpu 0.28.0; the mock returns 200). It has not been confirmed by
  a live `--test`. So Codex is marked **"expected — confirm with a live `--test`"** for
  Qwen, not a hard "supported."
- **Config-side note (verified):** the Codex docs confirm `wire_api = "responses"` is the
  **only** supported value — current Codex cannot target a chat/completions-only backend
  at all, so "add a compat shim" was never a viable alternative for GLM; the only real
  options were implement `/v1/responses` in the engine or mark unsupported.

No other route failed for either backend in the matrices, so no other client needs to
be demoted on route grounds. (The Claude Code *config* had a model-id bug, fixed in
§6 — that is a config-correctness issue, not a missing route.)

---

## 5. What is still needed to upgrade INFERRED → VERIFIED

1. **Live `--test` for Qwen** (`KTL_CLIENT_API_KEY` set + a running `serve --model
   qwen` session) to confirm `/v1/responses` and `/v1/messages` really pass — this is
   what turns Qwen's Codex row from "expected" to "supported."
2. **Real-client runs** (one one-token prompt per client against a live session):
   mocks can't catch a client that changed its config schema.
3. **Key rotation** for the 3 previously leaked secrets (out of scope, still open).

*Done this session:* Codex, Hermes, and aider were each upgraded from INFERRED to
VERIFIED (official docs for Codex/aider; the installed hermes-agent source for
Hermes), and the two config bugs that surfaced (aider's env var + model prefix;
Hermes's whole routing mechanism) were fixed — see §6.

---

## 6. Changes made this session (within scope)

- **`launch.py main()`** — propagate `cmd_env`'s `--test` return value via `sys.exit`
  so the documented 0/1/2/3 exit codes actually reach the shell (was always 0).
- **`ktl_env.gen_claude_code` / `_write_json_env`** — `ANTHROPIC_MODEL` now uses the
  **bare** model id (was `kaggle-tpu/<id>`, which vLLM rejects), and adds
  `ANTHROPIC_SMALL_FAST_MODEL` + `ANTHROPIC_DEFAULT_HAIKU_MODEL` so Claude Code's
  background tasks don't 404 on a default Anthropic haiku id. Matches the verified
  `qwen38-27b/README.md` config.
- **`ktl_env.gen_codex`** — prints a clear "NOT SUPPORTED FOR THIS MODEL" warning when
  the backend lacks `/v1/responses` (driven by a new `responses_api` fact on each
  model in `launch.MODELS`: `qwen=True`, `glm=False`).

*Second pass (same day, after the audit) — bug fixes + verification upgrades:*

- **`ktl_env.gen_aider`** — **bug fix:** endpoint var is `OPENAI_API_BASE` (was
  `OPENAI_BASE_URL`, which aider ignores) and the model carries the `openai/` prefix
  (was bare, which does not route to the compat endpoint) — in both the env-var
  exports and the `.aider.conf.yml` snippet. Verified against
  `aider.chat/docs/llms/openai-compat.html`.
- **`ktl_env.gen_hermes`** — **rewritten (the old mechanism does not work):** `.env`
  now holds `KTL_CLIENT_API_KEY`; a new `hermes_model_block()` /
  `_write_hermes_model()` manages the top-level `model:` block in
  `~/.hermes/config.yaml` (`provider: custom`, `base_url: <root>/v1`,
  `api_key: ${KTL_CLIENT_API_KEY}`, `api_mode: chat_completions`, `default: <id>`).
  The old `OPENAI_API_KEY`/`OPENAI_BASE_URL` env-var route was wrong — verified in the
  installed hermes-agent source: bare `openai` is aliased to openrouter, and
  `openai-api` uses the `codex_responses` transport (`/v1/responses`). The old
  `openai_api_models.yaml` file (older docs) is referenced nowhere in the source.
  `--write` for hermes now manages both files; `--restore hermes` restores both.
- **`ktl_env._claude_model_env`** — also pins `ANTHROPIC_DEFAULT_SONNET_MODEL` /
  `ANTHROPIC_DEFAULT_OPUS_MODEL` to the bare model, so `/model sonnet|opus` aliases
  don't switch to default Anthropic ids the relay does not serve.
- **`ktl_env._redact`** — env-style masking widened from three hardcoded names to any
  secret-looking name (`*TOKEN`/`*KEY`/`*SECRET`/`*PASSWORD`): dotenv diffs include
  context lines, which would otherwise print the user's unrelated credentials (caught
  live: a `GITHUB_TOKEN` in `~/.hermes/.env` appeared in a dry-run diff).
- **`WRITE_TARGETS` restructured** to a list of `(path, kind)` pairs (hermes = 2
  files); `apply_write` and `--restore` loop over them.

Test suite: **72 passing** (`python3 -m unittest discover -s tests`); worker
`node --check` clean; all four writable clients' `--write --dry-run` outputs verified
secret-free (no leaked key fragments or unrelated credentials).
