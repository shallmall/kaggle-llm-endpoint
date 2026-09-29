# Manual setup (without the wizard)

`python launch.py setup` does everything on this page automatically. Use it
unless you have a reason not to — it validates each step, stores the results
in `~/.ktl/config.json`, and leaves you a passing compatibility matrix. This
page is the fallback for: constrained environments, "I want to do it myself",
or debugging a step the wizard flagged.

Each section below maps to a wizard step, so you can run the parts you need
and hand the rest back to `setup --only <step>`.

---

## 1. Prerequisites (step `prereqs`)

- **Kaggle account** with phone verification (required for TPU access) and the
  free quota (~20 TPU hours/week).
- **Python 3.10+** (the kaggle CLI now recommends 3.11+).
- **Kaggle CLI**:
  ```bash
  pip install kaggle
  ```
- **Node.js ≥ 22** (LTS from https://nodejs.org) with `npx` — only needed for
  the Cloudflare Worker relay (steps 3–5 below). Skip it for direct mode.

The wizard never auto-installs Node; it prints the exact download page and
waits.

## 2. Kaggle credentials (step `kaggle`)

Pick one:

- **OAuth (browser):** `kaggle auth login`
- **API token:** create one at <https://www.kaggle.com/settings/api>, then
  either `export KAGGLE_API_TOKEN=<token>`, or save it to
  `~/.kaggle/access_token`, or use the legacy credentials file:
  ```bash
  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json     # Windows: %USERPROFILE%\.kaggle\kaggle.json
  ```

Verify: `kaggle kernels list --page-size 1` (the wizard runs exactly this cheap
probe). Only the **username** is ever copied out of your credentials — the
token stays in `~/.kaggle/`.

## 3. Permanent URL relay (step `cloudflare` + `worker`) — optional

Without a relay you get a fresh random `trycloudflare.com` URL per boot
(direct mode, section 4). For a stable URL:

### Permanent URL relay (by hand)

```bash
cd worker
npx -y wrangler@4 login      # opens a browser; logs you into Cloudflare
npx -y wrangler@4 deploy     # prints https://<name>.<acct>.workers.dev
```

`wrangler login` uses an OAuth callback to `localhost:8976`, so it only works
when your browser and this terminal are on the **same machine**. From a VM,
WSL or container, forward the callback back to the machine with the browser:

```bash
ssh -L 8976:localhost:8976 <you>@<machine-with-browser>
```

…or run the step where your browser is (native Windows/macOS), or export a
token for a non-interactive login:
`export CLOUDFLARE_API_TOKEN=<token>` (Account > Workers Scripts at
<https://dash.cloudflare.com/profile/api-tokens>).

Two secrets, **different random strings, ≥ 32 chars each** (e.g.
`openssl rand -base64 32`). The wizard generates these for you; by hand:

```bash
npx -y wrangler@4 secret put UPDATE_SECRET    # only launch.py uses it
npx -y wrangler@4 secret put CLIENT_API_KEY   # your clients use this one
```

(Values can be piped in: `echo "<value>" | wrangler secret put CLIENT_API_KEY`.)

Then teach the launcher about the relay — the wizard stores these in
`~/.ktl/config.json`; by hand, use env vars (they take precedence):

```bash
export KTL_RELAY_URL="https://<name>.<acct>.workers.dev"
export KTL_RELAY_UPDATE_SECRET="<the UPDATE_SECRET you put above>"
```

## 4. Start the session (step `serve`)

```bash
python launch.py serve               # Qwen3.8-27B (default)
python launch.py serve --model glm   # GLM-5.3-Flash
```

With a relay configured (config or `KTL_RELAY_URL`), the session registers
itself with your permanent URL on READY — the endpoint stops changing.
Without one, it falls back to a per-boot quick-tunnel URL (still fine for
one-off use). Boot takes ~16–22 min; follow with
`python launch.py status --follow`.

## 5. Verify (step `verify`)

```bash
python launch.py env --test --wait 600
```

Runs the live compatibility matrix (models, chat non-stream/stream, Anthropic
messages, Responses, tool probe, latency). Exit `0` = all critical routes
pass. `doctor` runs the same matrix as part of its report.

## 6. Point your clients at it (step `clients`)

```bash
python launch.py env all                 # print config for every client
python launch.py env opencode --write    # write the file (.ktl-bak-* backup)
python launch.py env --restore opencode  # put the original back
```

Manual fallback — the two base-URL rules:

- **Claude Code** (Anthropic Messages): base URL = **root, no `/v1`**, key via
  `ANTHROPIC_AUTH_TOKEN`.
- **OpenAI-compatible** (Codex, opencode, aider, Hermes, SDKs, curl): base
  URL = **root + `/v1`**. aider reads `OPENAI_API_BASE` and wants the model as
  `openai/<id>`.

Key is always the relay's `CLIENT_API_KEY` (relay mode) or the key `serve`
printed (direct mode).

## Health checks

```bash
python launch.py doctor            # read-only; exit 0 = healthy; hints per failure
python launch.py status            # TPU session state + recent events
```

`doctor` never writes anything and redacts secrets — safe to paste into bug
reports.

---

## What the wizard stores

`~/.ktl/config.json` (0600, dir 0700, atomic writes; `KTL_HOME` overrides the
location):

- Kaggle **username**, worker name, permanent relay URL
- the two relay secrets (`client_api_key`, `update_secret`)
- per-step progress (`steps`) — this is what makes re-runs resume

It **never** stores: Kaggle tokens/credentials, the direct-session API key
(`~/.kaggle-tpu-lab.json`, changes per boot), or anything else.

Reset: `python launch.py setup --reset` (deletes `config.json` only — the
deployed Worker is untouched). Rotate: `setup --only secrets --rotate`.
