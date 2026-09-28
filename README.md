# kaggle-tpu-lab

Run frontier-class open models on Kaggle's **free TPU v5e-8** and get a public
endpoint that speaks the **OpenAI and Anthropic APIs**. Point Claude Code,
Codex CLI, opencode or plain `curl` at it. No GPU, no cloud bill — about
twenty minutes from pressing Run to a live URL.

> **Based on** [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) —
> extended here with multi-model support in the launcher (`--model qwen|glm`),
> GLM engine packaging, and a Cloudflare Worker relay for permanent URLs.

## Models

| Model | Weights on the TPU | Context | One stream | Many streams | Prefill | Run → URL | Engine |
|-------|--------------------|---------|------------|--------------|---------|-----------|--------|
| [Qwen3.8-27B](kaggle-tpu-lab/qwen38-27b/) | bf16, no quantization | 262k | ~130 tok/s | ~540 tok/s at 8 | 10,300 tok/s | ~22 min | vllm-tpu + one patch |
| [GLM-5.3-Flash](kaggle-tpu-lab/glm53-flash/) (320B MoE) | 3-bit experts, int8 rest | 262k | ~64 tok/s | ~90 tok/s at 3 | ~1,600 tok/s | ~16 min | custom JAX engine |

Each model folder has a run-all Kaggle notebook, the kernel script behind it,
and a README with the numbers and the how.

**Exact model names for API calls** (use these in the `"model"` field or
`ANTHROPIC_MODEL`):

| Model | `"model"` value |
|-------|-----------------|
| Qwen3.8-27B | `qwen3.8-27b` |
| GLM-5.3-Flash | `glm-5.3-flash` |

Example:
```bash
curl <ENDPOINT>/v1/chat/completions -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## What you need

- A **Kaggle account** with phone verification (required for TPU access) and
  its free quota (~20 TPU hours/week).
- **Python 3.10+** and the Kaggle CLI:
  ```bash
  pip install kaggle
  ```
  Get your token at kaggle.com → Settings → API → **Create New Token**, then:
  - **Linux / macOS:** `~/.kaggle/kaggle.json`
    ```bash
    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    ```
  - **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
- *(optional)* A free **Cloudflare account** + Node.js/npm for the permanent
  URL relay (see below). Without it you get a new random URL each boot.

## Quick start

```bash
cd kaggle-tpu-lab

python launch.py serve                  # Qwen3.8-27B (default)
python launch.py serve --model glm      # GLM-5.3-Flash (engine auto-embedded)
```

`launch.py` pushes the kernel, watches progress live, and prints the endpoint
URL + API key when it's ready. That's it.

## Useful commands

```bash
python launch.py status --follow        # re-attach / follow a running session
python launch.py stop                   # terminate the TPU session

python launch.py serve --model glm --reasoning-effort high --streams 8 --vision false
python launch.py serve --model qwen --text-only --fast-start
```

| Flag | Model | Effect |
|------|-------|--------|
| `--model qwen\|glm` | both | which model (default: qwen) |
| `--reasoning-effort` | both | server-side default effort |
| `--max-model-len` | both | context length (default 262144) |
| `--streams` / `--vision` | GLM | concurrent streams / vision tower |
| `--max-num-seqs` / `--mtp` / `--text-only` / `--no-tools` / `--fast-start` | Qwen | concurrency, spec-decode, vision, tools, precompile |
| `--keepalive-min 360` | both | auto-shutdown timer (default 480 ≈ 8 h) |

## Permanent URL (optional, ~15 min one-time)

The quick tunnel gets a new `trycloudflare.com` URL every boot. The
`worker/` folder deploys a Cloudflare Worker that keeps **one permanent URL**
pointed at the current session — every future `serve` auto-registers, so your
agent settings never change:

```bash
cd worker
npm install -g wrangler && wrangler login && wrangler deploy   # prints your URL

wrangler secret put UPDATE_SECRET      # random string; only launch.py uses it
wrangler secret put CLIENT_API_KEY     # a DIFFERENT random string; your clients use this
```

Then set in your shell:
```bash
export KTL_API_KEY="<any random string>"
export KTL_RELAY_URL="https://kaggle-tpu-relay.<your-subdomain>.workers.dev"
export KTL_RELAY_UPDATE_SECRET="<same as UPDATE_SECRET>"
```

Point your agent at `<KTL_RELAY_URL>/v1` with key `CLIENT_API_KEY`:

```bash
ANTHROPIC_BASE_URL=$KTL_RELAY_URL/v1 ANTHROPIC_AUTH_TOKEN=<CLIENT_API_KEY> \
ANTHROPIC_MODEL=glm-5.3-flash claude
```

(For Codex CLI / opencode: same values as `OPENAI_BASE_URL` / `OPENAI_API_KEY`.)

## How a session works

`launch.py` injects your settings into the kernel script and pushes it; the
kernel attaches the public datasets, builds the engine on the 8 chips, opens
the tunnel, and serves until `keepalive_min` elapses (~9 h max per Kaggle's
cap). Boot again for a fresh session and URL.

## Troubleshooting (quick hits)

- **No "Relay updated" line** — check env vars are loaded: `echo $KTL_RELAY_URL $KTL_RELAY_UPDATE_SECRET`.
- **401 from the relay** — you must send `CLIENT_API_KEY`, not the other two secrets.
- **502/504 on chat** — registration is fine, the model is still booting. Wait for READY.
- **GLM fails at "no glm53/ package"** — your `launch.py` is outdated; pull latest.
- **Kernel died / 9 h cap** — just `serve` again; relay auto-updates.
- **More relay issues** — see the full section in `git history` or run `cd worker && wrangler tail`.

## Local verification (no TPU needed)

```bash
kaggle-tpu-lab/glm53-flash/tools/verify_local.sh          # launcher check + kernel smoke test
kaggle-tpu-lab/glm53-flash/tools/verify_local.sh --full   # + engine unit tests (slow)
```

## Adding a model

One folder inside `kaggle-tpu-lab/` named after the model: `README.md`,
`kernel/` (serving script with the `__LAUNCHER_CONFIG__` line), `notebook/`,
plus whatever the recipe needs. Then add an entry to `MODELS` in `launch.py`.

## Credits

- [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) — the
  original project: Qwen3.8-27B on vllm-tpu, the GLM-5.3-Flash JAX engine,
  and the Kaggle TPU serving infrastructure.
- [ntfy.sh](https://ntfy.sh) for progress events between kernel and launcher.

## License

MIT (see [LICENSE.md](LICENSE.md)). Model weights keep their own licenses;
each folder says which.
