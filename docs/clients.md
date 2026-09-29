# Client configuration — verified details

`python launch.py env <client>` generates config for each AI client. This file
records **what was verified against the current docs, and when**, so the
generators aren't built on unchecked assumptions. Re-verify whenever a client
changes its config format — mock tests can't catch that.

**Last verified: 2026-09-29** (Claude Code + opencode from live docs; Codex from
the official config reference; Hermes from the installed hermes-agent source;
aider from the official docs — see also `clients-verification.md`).

The one rule that matters most — **the base URL differs by API**, the key is
always the same:

- **Anthropic Messages API** (Claude Code, the Anthropic Python SDK): base URL
  is the **root, no `/v1`**. The client appends `/v1/messages` itself.
- **OpenAI-compatible API** (Codex, opencode, aider, Hermes, OpenAI SDK, curl):
  base URL is the **root + `/v1`**.
- Key: the relay's `CLIENT_API_KEY` (relay mode) or the key `serve` printed
  (direct mode). `ktl env` resolves the right one from `mode`.

---

## Claude Code — verified 2026-09-29 (live docs, code.claude.com)

Source: https://code.claude.com/docs/en/settings and .../env-vars

- Config file: `~/.claude/settings.json` (user scope). An `env` block is a valid
  settings key; a value in the file overrides a same-named shell export.
- `ANTHROPIC_BASE_URL` — "Override the API endpoint to route requests through a
  proxy or gateway." It is the **root**; Claude appends `/v1/messages`. (So do
  **not** put `/v1` in it.)
- `ANTHROPIC_AUTH_TOKEN` — "Custom value for the `Authorization` header (the
  value you set here will be prefixed with `Bearer `)." → sent as
  `Authorization: Bearer <key>`.
- `ANTHROPIC_API_KEY` — alternative; sent as the `X-Api-Key` header. (The relay
  accepts both `Bearer` and `x-api-key`.)
- `ANTHROPIC_MODEL` — the model name; overrides the `model` settings key. We set
  it to `kaggle-tpu/<api_model>`.
- `ANTHROPIC_BETAS` — comma-separated extra `anthropic-beta` header values (not
  needed here).

Generator emits the `env` object:
```json
{ "env": {
    "ANTHROPIC_BASE_URL": "<root>",
    "ANTHROPIC_AUTH_TOKEN": "<CLIENT_API_KEY>",
    "ANTHROPIC_MODEL": "kaggle-tpu/<api_model>" } }
```

## opencode — verified 2026-09-29 (live docs, opencode.ai)

Source: https://opencode.ai/docs/providers/ ("Custom provider" + the
`@ai-sdk/openai-compatible` examples for Atomic Chat / llama.cpp / LM Studio /
Ollama).

- Config file: `~/.config/opencode/opencode.json` (or `.jsonc`).
- Provider lives under the top-level `provider` object, keyed by a custom id
  (we use `kaggle-tpu`).
- Required fields (from the docs examples):
  - `npm`: `"@ai-sdk/openai-compatible"` (any OpenAI-compatible API).
  - `name`: display name.
  - `options.baseURL`: the endpoint — **includes `/v1`** (all docs examples
    show e.g. `http://host:port/v1`).
  - `models`: a map of **model id → config**. The id **must match the `id`
    returned by `GET /v1/models`** (so `qwen3.8-27b` / `glm-5.3-flash`).
    Each entry supports `name` and `limit: {context, output}`.
- `options.apiKey` is accepted (used by the current live config and by this
  generator). The docs' canonical key routes are `/connect` (stored in
  `~/.local/share/opencode/auth.json`) or a provider env var; `options.apiKey`
  is the self-contained option we use.
- Extra model fields we emit beyond the minimal docs example — `cost`,
  `tool_call`, `reasoning`, `temperature`, `modalities` — are valid
  models.dev-schema options (present in the live config); they're optional
  display/capability hints.

Generator merges under `provider.kaggle-tpu`, merging `models` by key so other
models already present are kept.

## Codex CLI — standard format (cross-checked, re-verify on Codex updates)

Source: Codex CLI `config.toml` (OpenAI). Re-verify against the installed
Codex version's docs before relying on it.

- Config file: `~/.codex/config.toml`.
- We append a **managed block** between `# >>> ktl managed >>>` /
  `# <<< ktl managed <<<` markers (idempotent replace).
- The block defines a provider table only:
  ```toml
  [model_providers.kaggle-tpu]
  name = "Kaggle TPU Relay"
  base_url = "<root>/v1"
  wire_api = "responses"
  env_key = "KTL_CLIENT_API_KEY"
  ```
  - `wire_api = "responses"` → Codex calls `POST /v1/responses`. **This is the
    route that must exist on the relay**; `ktl env --test` flags it if missing.
  - `env_key` tells Codex which env var holds the key, so the file never stores
    the key.
- We do **not** write top-level `model` / `model_provider` keys (that would
  conflict with the user's existing defaults). The model is passed at runtime:
  `codex -c model_provider=kaggle-tpu -c model=<api_model>`.

## Hermes — verified 2026-09-29 (installed hermes-agent source)

- `~/.hermes/.env`: `KTL_CLIENT_API_KEY=<key>` (loaded before `config.yaml`,
  `hermes_cli/main.py:721`).
- Top-level `model:` block in `~/.hermes/config.yaml`: `provider: custom`,
  `base_url: <root>/v1`, `api_key: ${KTL_CLIENT_API_KEY}`,
  `api_mode: chat_completions`, `default: <api_model>` →
  `POST {base_url}/chat/completions`.
- The earlier `OPENAI_API_KEY`/`OPENAI_BASE_URL` approach is **wrong**: bare
  `openai` is aliased to openrouter and `openai-api` uses the `codex_responses`
  transport (`hermes_cli/providers.py`). `openai_api_models.yaml` (older docs)
  is referenced nowhere in the source.

## aider — verified 2026-09-29 (aider.chat/docs/llms/openai-compat.html)

- `AIDER_MODEL=openai/<api_model>`, `OPENAI_API_KEY=<key>`,
  `OPENAI_API_BASE=<root>/v1` (NOT `OPENAI_BASE_URL`; model needs the
  `openai/` prefix to route to the compat endpoint).
- Or `.aider.conf.yml`: `model: openai/<api_model>`, `openai-api-key`,
  `openai-api-base`.
- Print-only (no file written by `--write`).

## curl / Python — print-only examples

- curl: one OpenAI-compatible (`POST /v1/chat/completions`, Bearer) and one
  Anthropic (`POST /v1/messages`, `x-api-key` + `anthropic-version`).
- Python: OpenAI SDK (`base_url=<root>/v1`) and Anthropic SDK
  (`base_url=<root>`, no `/v1`).

---

## Re-verification checklist (run after a client update)

1. `python launch.py env all --model glm --dry-run` — eyeball each block.
2. `python launch.py env --test` — live matrix (flags a missing `/v1/responses`).
3. **Real-client run:** start each client once against a live session and send
   a one-token prompt. Mocks can't catch a client that changed its config
   schema; only the real client can.
4. Update the "Last verified" date above and note any format change.
