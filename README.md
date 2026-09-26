# kaggle-tpu-lab

Run frontier-class open models on Kaggle's free TPU v5e-8 and get a public
endpoint that speaks the OpenAI and Anthropic APIs. Point Claude Code, Codex
CLI, opencode or anything else at it. No GPU, no cloud bill, about twenty
minutes from pressing Run to a URL.

> **Based on** [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) —
> this project extends the original with multi-model support in the launcher
> (`--model qwen|glm`, including the GLM engine packaging) and a Cloudflare
> Worker relay for permanent URLs.

Each model has its own folder with a run-all Kaggle notebook, the kernel
script behind it, and a write-up of how it works and what we measured.

| Model | Weights on the TPU | Context | One stream | Many streams | Prefill | Run → URL | Engine |
|-------|-------------------|---------|------------|--------------|---------|-----------|--------|
| [Qwen3.8-27B](kaggle-tpu-lab/qwen38-27b/) | bf16, no quantization | 262k | ~130 tok/s | ~540 tok/s at 8 | 10,300 tok/s | ~22 min | vllm-tpu + one patch |
| [GLM-5.3-Flash](kaggle-tpu-lab/glm53-flash/) (320B MoE) | 3-bit experts, int8 rest | 262k | ~64 tok/s | ~90 tok/s at 3 | ~1,600 tok/s | ~16 min | our own JAX engine |

Numbers are measured on the shipped configuration; the folder READMEs say how.
Qwen runs on vllm-tpu with one patch. GLM-5.3-Flash runs on an engine we wrote
in JAX for it; as far as we know it is the first to run that model on a TPU.

---

## What you need

- **A Kaggle account** with a phone number verified (required for TPU access)
  and its free quota (~20 TPU hours/week).
- **Python 3.10+** and the Kaggle CLI:
  ```bash
  pip install kaggle
  ```
  Go to kaggle.com → Settings → API → **Create New Token**. This downloads
  `kaggle.json`. Put it here:
  ```bash
  mkdir -p ~/.kaggle
  mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  ```
- **A free Cloudflare account** (no domain, no credit card) for the permanent
  URL relay.
- **Node.js + npm** (for `wrangler`, the Cloudflare CLI).

---

## Quick start (terminal)

```bash
# from this repo's kaggle-tpu-lab/ folder:
cd kaggle-tpu-lab

# Qwen3.8-27B (default):
python launch.py serve

# GLM-5.3-Flash (the launcher embeds its JAX engine into the pushed kernel):
python launch.py serve --model glm
```

`launch.py` pushes the kernel with the Kaggle CLI, watches its progress via
ntfy, and prints the endpoint + API key when it's live. `status` and `stop`
do what they say.

---

## Permanent URL: Cloudflare Worker relay

The quick tunnel (`trycloudflare.com`) gets a new random URL every boot. The
Worker relay gives you **one permanent URL** that always points at the current
Kaggle session. Total one-time setup: ~15 minutes.

### 1. Deploy the Worker

The `worker/` folder at the repo root is the wrangler project as-is.

```bash
cd worker
npm install -g wrangler
wrangler login        # opens a browser tab — click Allow
wrangler deploy
```

This prints your permanent URL, e.g.:
```
https://kaggle-tpu-relay.<your-subdomain>.workers.dev
```

### 2. Create two secrets

```bash
wrangler secret put UPDATE_SECRET
# paste a random string (uuidgen works) — only launch.py uses this

wrangler secret put CLIENT_API_KEY
# paste a DIFFERENT random string — this is what you give to your client
```

Verify:
```bash
wrangler secret list
# Expect: UPDATE_SECRET, CLIENT_API_KEY
```

### 3. Set environment variables

Add to `~/.bashrc` (or `~/.zshrc`):

```bash
export KTL_API_KEY="<any random string>"
export KTL_RELAY_URL="https://kaggle-tpu-relay.<your-subdomain>.workers.dev"
export KTL_RELAY_UPDATE_SECRET="<same as UPDATE_SECRET>"
```

Then in every terminal you'll run `launch.py` from:
```bash
source ~/.bashrc
echo $KTL_API_KEY $KTL_RELAY_URL $KTL_RELAY_UPDATE_SECRET   # verify all three print
```

| Variable | Purpose |
|----------|---------|
| `KTL_API_KEY` | Becomes the real API key on the Kaggle side. Your client never sees it. |
| `KTL_RELAY_URL` | The Worker URL, **no `/v1`**. |
| `KTL_RELAY_UPDATE_SECRET` | Must match `UPDATE_SECRET` on Cloudflare. |

### 4. Boot and verify

```bash
python launch.py serve
```

Watch for:
```
Endpoint URL reserved: https://xxxx-yyyy.trycloudflare.com/v1  (not live yet)
Relay updated (200) — your permanent endpoint is live.
```

**`Relay updated (200)` confirms the Worker now points at this session.**

Then verify end-to-end:
```bash
curl https://kaggle-tpu-relay.<your-subdomain>.workers.dev/v1/models \
  -H "Authorization: Bearer <CLIENT_API_KEY>"
```

### 5. Point your agent at it — once, forever

- **Base URL:** `https://kaggle-tpu-relay.<your-subdomain>.workers.dev/v1`
- **API key:** your `CLIENT_API_KEY` value

Every future `python launch.py serve` auto-registers the new tunnel with the
relay. Your agent settings never change.

For Claude Code:
```bash
ANTHROPIC_BASE_URL=https://kaggle-tpu-relay.<sub>.workers.dev/v1 \
ANTHROPIC_AUTH_TOKEN=<CLIENT_API_KEY> \
ANTHROPIC_MODEL=glm-5.3-flash \
claude
```

For Codex CLI / opencode, set the equivalent `OPENAI_BASE_URL` and
`OPENAI_API_KEY` to the same values.

---

## Launch commands

```bash
# Qwen3.8-27B (default model):
python launch.py serve

# GLM-5.3-Flash:
python launch.py serve --model glm

# GLM with high reasoning, 8 streams, no vision:
python launch.py serve --model glm --reasoning-effort high --streams 8 --vision false

# Qwen text-only, fast start:
python launch.py serve --model qwen --text-only --fast-start

# Check status / re-attach to a running session:
python launch.py status
python launch.py status --follow

# Stop the TPU session:
python launch.py stop
```

### Options

| Flag | Model | Description |
|------|-------|-------------|
| `--model qwen` / `--model glm` | both | Which model to serve (default: qwen) |
| `--reasoning-effort low` | both | Server-side default reasoning effort |
| `--max-model-len 131072` | both | Context length (default: 262144) |
| `--streams 8` | GLM | Concurrent decode streams |
| `--vision false` | GLM | Skip the vision tower |
| `--max-num-seqs 8` | Qwen | Max concurrent sequences |
| `--mtp 0` | Qwen | Disable MTP speculative decoding |
| `--text-only` | Qwen | Skip vision tower |
| `--no-tools` | Qwen | Disable tool-calling |
| `--fast-start` | Qwen | Skip TPU graph precompile |
| `--keepalive-min 360` | both | Auto-shutdown after N minutes |
| `--verbose` | both | Show all log lines |
| `--user <name>` | both | Kaggle username (auto-detected) |
| `--slug <name>` | both | Kernel name override |

---

## How a session works

1. `launch.py` builds a `CFG` dict, injects it into the kernel script, and
   pushes it to Kaggle via the CLI.
2. The kernel attaches public datasets (weights, compile cache), builds the
   engine across the 8 TPU chips, opens a Cloudflare quick tunnel, and starts
   an HTTP server on port 8000.
3. The kernel publishes progress events to ntfy; `launch.py` watches them and
   prints human-readable status.
4. When the tunnel URL appears, `launch.py` registers it with the Worker relay
   (if configured), giving you a permanent URL.
5. The kernel serves until `keepalive_min` elapses (default 480 min ≈ 8 h),
   then exits cleanly. Run `python launch.py serve` again for a new session.

---

## Troubleshooting

**No "Relay updated" line in `launch.py` output**
- Check for `Endpoint URL reserved` first. If missing, the tunnel failed.
- If the URL line is present but no relay line: your env vars aren't loaded.
  Run `echo $KTL_RELAY_URL $KTL_RELAY_UPDATE_SECRET` in that terminal.

**`curl .../v1/models` returns "No Kaggle session registered yet"**
The Worker never received a successful `/update-config`. Either `launch.py`
hasn't finished booting, or registration failed. Check its output.

**`curl .../v1/models` returns 401**
Wrong key. You must send `CLIENT_API_KEY`, not `UPDATE_SECRET` and not
`KTL_API_KEY` — all three are different.

**502 / 504 / timeout on `/v1/chat/completions`**
Registration succeeded but the model isn't serving yet. Wait for the READY
banner in the `launch.py serve` terminal.

**`cloudflared exited with code -11`**
cloudflared crash, not a vLLM or auth issue. The Qwen kernel prefers a freshly
downloaded binary and falls back to the one bundled in the env dataset; the
GLM kernel downloads one. Just re-run `launch.py serve`.

**`AttributeError: __delitem__` in vLLM**
The vllm-tpu 0.28.0 MTP + async scheduling bug. The launcher now passes
`--no-async-scheduling` automatically. If you see this, your `launch.py` is
out of date — pull the latest.

**GLM fails at step 1/6: `no glm53/ package next to this script`**
The pushed kernel did not get the engine package. `launch.py` embeds it into
the script at push time, so step 1/6 must print
`engine package extracted to /kaggle/working/glm53` before anything else
happens. If the line is missing, your `launch.py` is out of date — pull the
latest.

**Session died / Kaggle kernel timed out**
Kaggle kernels cap at ~9 h. Just run `python launch.py serve` again — a new
tunnel gets generated and auto-registered. Your agent settings don't change.

**Manual registration (session already running, don't want to reboot)**
```bash
python launch.py status   # prints the current endpoint + key
curl -X POST https://kaggle-tpu-relay.<sub>.workers.dev/update-config \
  -H "Authorization: Bearer <UPDATE_SECRET>" \
  -H "Content-Type: application/json" \
  -d '{"kaggle_url":"https://xxxx.trycloudflare.com","kaggle_key":"<key>"}'
```
**Critical:** `kaggle_url` must be the bare root, **no `/v1`**.

**`curl .../update-config` returns "Unauthorized"**
In order of likelihood:
1. `kaggle_url` has a trailing `/v1` — strip it.
2. The Authorization header has a typo — run with `-v` and check.
3. The secret doesn't match what's stored on Cloudflare. Reset it:
   ```bash
   wrangler secret put UPDATE_SECRET
   ```
   then update `~/.bashrc` to the same new value and `source` it.

To see exactly what the Worker received:
```bash
cd worker && wrangler tail
```

---

## Local verification (no TPU needed)

The GLM serving script and its JAX engine can be smoke-tested on a plain CPU
machine — no Kaggle account, no TPU:

```bash
kaggle-tpu-lab/glm53-flash/tools/verify_local.sh          # launcher build check + full kernel smoke test
kaggle-tpu-lab/glm53-flash/tools/verify_local.sh --full   # + the engine unit test suite (slow on a laptop)
```

The smoke test runs the actual `serve_glm53.py` end-to-end against a tiny
engine with the real GLM tokenizer: the kernel's self-test, OpenAI and
Anthropic streaming, tool calls, the thinking budget, `count_tokens`, and the
429 queue.

---

## Adding a model

One folder inside `kaggle-tpu-lab/`, named after the model:
- `README.md` — the numbers and the how
- `kernel/` — the serving script (with the `CFG = None  # __LAUNCHER_CONFIG__` line)
- `notebook/` — the run-all notebook generated from it
- plus whatever the recipe needs (a patch, an engine package)

Then add an entry to `MODELS` in `launch.py` with the kernel path, dataset
sources, and default slug. The launcher and the notebook share the same config
block, so a setting changed in one place means the same thing in the other.

---

## Credits

- Original project: [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab)
  (529 stars) — Qwen3.8-27B on vllm-tpu, GLM-5.3-Flash JAX engine, Kaggle
  TPU v5e-8 serving infrastructure.
- Cloudflare Worker relay pattern and Durable Object config store.
- ntfy.sh for progress events between the kernel and the launcher.

---

## License

The code here is MIT. Model weights keep their own licenses; each folder says
which.
