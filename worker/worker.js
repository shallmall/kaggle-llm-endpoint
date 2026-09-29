// kaggle-tpu-lab relay — permanent URL in front of a Kaggle quick tunnel
// that changes address (and used to change key) on every boot.
//
// Two secrets, two jobs, neither ever seen by the other party:
//   UPDATE_SECRET   - lets ONLY your own `launch.py` register the current
//                      Kaggle URL/key after each boot (POST /update-config).
//                      Also derives (via HKDF-SHA256) the AES-256-GCM key
//                      that encrypts the stored config at rest.
//   CLIENT_API_KEY  - lets ONLY your own client (Hermes, curl, etc.) use the
//                      proxy. This is the key you put in Hermes' .env — it is
//                      never the real Kaggle key, so leaking it can't reveal
//                      a rotating secret, only access to your own relay.
//
// The stored config is never held in plaintext: kaggleUrl and kaggleKey are
// AES-256-GCM sealed before they enter the Durable Object (enc:v1: prefix),
// each bound to its own field via AAD so a value can't be swapped into the
// other slot. Legacy plaintext values from before this change read through
// transparently until the next boot re-registers (they are then rewritten
// encrypted). Rotating UPDATE_SECRET invalidates the stored config until the
// next `launch.py serve` re-registers — ciphertext is unreadable by design.
//
// Auth accepts the key as `Authorization: Bearer <key>` or `x-api-key: <key>`
// (Anthropic SDK style) and compares SHA-256 digests in constant time. Both
// secrets must be >= 32 chars; a shorter one is treated as misconfigured and
// every request is denied.
//
// Generation calls (POST /v1/chat/completions, /v1/messages, /v1/responses)
// are rate-limited per 60 s window — RATE_LIMIT_PER_MIN (default 120),
// counted when the request starts (streams are long-lived, so counting on
// completion would let bursts through), answered with 429 + Retry-After in
// the API's own error shape. Side calls don't count. Concurrency itself is
// capped by the engines, not here (GLM: --streams + wait queue -> 429/503;
// Qwen: --max-num-seqs + the vLLM scheduler).
//
// State lives in a Durable Object (not KV): reads are strongly consistent,
// so there is no "wait ~60s for the edge to catch up" propagation gap.
//
// Client base URL:  https://<your-worker>.workers.dev/v1
// (mirrors exactly what you'd otherwise point Hermes at on the raw
// trycloudflare.com URL — no path rewriting needed.)

import { DurableObject } from "cloudflare:workers";
import {
  DEFAULT_LIMIT,
  decryptField,
  deriveKey,
  digestKeyMatch,
  encryptField,
  rateLimit,
  secretUsable,
  shouldCount,
} from "./ktlcore.js";

export class ConfigStore extends DurableObject {
  async setConfig(kaggleUrl, kaggleKey) {
    await this.ctx.storage.put({
      kaggleUrl,
      kaggleKey,
      updatedAt: Date.now(),
    });
  }

  async getConfig() {
    const [kaggleUrl, kaggleKey, updatedAt] = await Promise.all([
      this.ctx.storage.get("kaggleUrl"),
      this.ctx.storage.get("kaggleKey"),
      this.ctx.storage.get("updatedAt"),
    ]);
    return { kaggleUrl, kaggleKey, updatedAt };
  }

  async limitGeneration(limitPerMin) {
    return rateLimit(this.ctx.storage, limitPerMin);
  }
}

// Accepts the key from either header shape:
//   Authorization: Bearer <key>   (Claude Code, OpenAI SDK, curl)
//   x-api-key: <key>              (Anthropic SDK clients)
// Compared as SHA-256 digests in constant time. Secrets shorter than
// MIN_SECRET_LEN are treated as misconfigured -> deny, never open-fail.
async function authMatches(request, expected) {
  if (!secretUsable(expected)) return false;
  const candidates = [];
  const bearer = request.headers.get("Authorization") || "";
  if (bearer.startsWith("Bearer ")) candidates.push(bearer.slice(7));
  const xApiKey = request.headers.get("x-api-key") || "";
  if (xApiKey) candidates.push(xApiKey);
  for (const candidate of candidates) {
    if (await digestKeyMatch(candidate, expected)) return true;
  }
  return false;
}

// 429 in the error shape of the API the client spoke, so agents see a clean
// rate-limit message instead of a parse error.
function rateLimitResponse(pathname, retryAfterS) {
  const headers = { "Content-Type": "application/json", "Retry-After": String(retryAfterS) };
  if (pathname === "/v1/messages") {
    return new Response(
      JSON.stringify({
        type: "error",
        error: { type: "rate_limit_error", message: "Rate limit exceeded — slow down and retry." },
      }),
      { status: 429, headers },
    );
  }
  return new Response(
    JSON.stringify({
      error: { message: "Rate limit exceeded — slow down and retry.", type: "rate_limit_exceeded" },
    }),
    { status: 429, headers },
  );
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const id = env.CONFIG_STORE.idFromName("singleton");
    const store = env.CONFIG_STORE.get(id);

    // ---- 1. Kaggle -> Worker: register this boot's URL + key -------------
    if (url.pathname === "/update-config") {
      if (request.method !== "POST") {
        return new Response("Method not allowed", { status: 405 });
      }
      if (!await authMatches(request, env.UPDATE_SECRET)) {
        return new Response("Unauthorized", { status: 401 });
      }
      let body;
      try {
        body = await request.json();
      } catch {
        return new Response("Body must be JSON", { status: 400 });
      }
      const { kaggle_url, kaggle_key } = body || {};
      if (!kaggle_url || !kaggle_key) {
        return new Response("kaggle_url and kaggle_key are required", { status: 400 });
      }
      if (!/^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/i.test(kaggle_url)) {
        // Tightened on purpose: only accept the exact shape the launcher
        // actually sends. If someone gets the update secret, this stops
        // them pointing your relay at an arbitrary origin.
        return new Response("kaggle_url is not a trycloudflare.com root URL", { status: 400 });
      }
      // Seal before storing: the Durable Object never holds the raw URL/key.
      const key = await deriveKey(env.UPDATE_SECRET);
      await store.setConfig(
        await encryptField(key, "kaggleUrl", kaggle_url),
        await encryptField(key, "kaggleKey", kaggle_key),
      );
      return new Response("ok", { status: 200 });
    }

    // ---- 2. Everything else: proxy to Kaggle, client-authenticated first --
    if (!await authMatches(request, env.CLIENT_API_KEY)) {
      return new Response("Unauthorized", { status: 401 });
    }

    const stored = await store.getConfig();
    if (!stored.kaggleUrl || !stored.kaggleKey) {
      return new Response("No Kaggle session registered yet — start one with launch.py.", {
        status: 503,
      });
    }
    // Stale-session guard: a kernel's keepalive_min caps how long it can run
    // (default 480 min / 8h). If nothing has re-registered in longer than
    // that, the session is almost certainly dead rather than just quiet.
    const STALE_MS = 9 * 60 * 60 * 1000;
    if (Date.now() - stored.updatedAt > STALE_MS) {
      return new Response("Registered Kaggle session looks stale — start a new one.", {
        status: 503,
      });
    }

    // Counted at request start, generation routes only, after auth.
    if (shouldCount(request.method, url.pathname)) {
      const limit = Number(env.RATE_LIMIT_PER_MIN) || DEFAULT_LIMIT;
      const verdict = await store.limitGeneration(limit);
      if (!verdict.ok) {
        return rateLimitResponse(url.pathname, verdict.retryAfterS);
      }
    }

    let kaggleUrl, kaggleKey;
    try {
      const key = await deriveKey(env.UPDATE_SECRET);
      kaggleUrl = await decryptField(key, "kaggleUrl", stored.kaggleUrl);
      kaggleKey = await decryptField(key, "kaggleKey", stored.kaggleKey);
    } catch {
      // Ciphertext no longer decrypts — UPDATE_SECRET was rotated after
      // registration. Only a fresh serve can fix it.
      return new Response(
        "Stored session config is unreadable — UPDATE_SECRET changed since registration. Run `launch.py serve` to re-register.",
        { status: 503 },
      );
    }

    const target = new URL(kaggleUrl + url.pathname + url.search);
    const outgoing = new Request(target, {
      method: request.method,
      headers: request.headers,   // passes through anthropic-version, anthropic-beta, etc.
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "manual",
    });
    // Authenticate the BACKEND with the Kaggle key in BOTH header shapes.
    // The client's own Authorization / x-api-key (its CLIENT_API_KEY) must not
    // reach the backend — for /v1/messages an Anthropic-style backend reads
    // x-api-key, so overwriting only Authorization would leak the wrong key.
    outgoing.headers.set("Authorization", `Bearer ${kaggleKey}`);
    outgoing.headers.set("x-api-key", kaggleKey);
    outgoing.headers.delete("Host");

    return fetch(outgoing);
  },
};
