// kaggle-tpu-lab relay — permanent URL in front of a Kaggle quick tunnel
// that changes address (and used to change key) on every boot.
//
// Two secrets, two jobs, neither ever seen by the other party:
//   UPDATE_SECRET   - lets ONLY your own `launch.py` register the current
//                      Kaggle URL/key after each boot (POST /update-config).
//   CLIENT_API_KEY  - lets ONLY your own client (Hermes, curl, etc.) use the
//                      proxy. This is the key you put in Hermes' .env — it is
//                      never the real Kaggle key, so leaking it can't reveal
//                      a rotating secret, only access to your own relay.
//
// State lives in a Durable Object (not KV): reads are strongly consistent,
// so there is no "wait ~60s for the edge to catch up" propagation gap.
//
// Client base URL:  https://<your-worker>.workers.dev/v1
// (mirrors exactly what you'd otherwise point Hermes at on the raw
// trycloudflare.com URL — no path rewriting needed.)

import { DurableObject } from "cloudflare:workers";

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
}

// Constant-time-ish string compare so a mistyped/partial key doesn't leak
// timing information. Overkill for a personal proxy, but it's free.
function safeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) {
    return false;
  }
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function bearerMatches(request, expected) {
  if (!expected) return false; // secret not configured -> deny, never open-fail
  const header = request.headers.get("Authorization") || "";
  return safeEqual(header, `Bearer ${expected}`);
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
      if (!bearerMatches(request, env.UPDATE_SECRET)) {
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
      await store.setConfig(kaggle_url, kaggle_key);
      return new Response("ok", { status: 200 });
    }

    // ---- 2. Everything else: proxy to Kaggle, client-authenticated first --
    if (!bearerMatches(request, env.CLIENT_API_KEY)) {
      return new Response("Unauthorized", { status: 401 });
    }

    const { kaggleUrl, kaggleKey, updatedAt } = await store.getConfig();
    if (!kaggleUrl) {
      return new Response("No Kaggle session registered yet — start one with launch.py.", {
        status: 503,
      });
    }
    // Stale-session guard: a kernel's keepalive_min caps how long it can run
    // (default 480 min / 8h). If nothing has re-registered in longer than
    // that, the session is almost certainly dead rather than just quiet.
    const STALE_MS = 9 * 60 * 60 * 1000;
    if (Date.now() - updatedAt > STALE_MS) {
      return new Response("Registered Kaggle session looks stale — start a new one.", {
        status: 503,
      });
    }

    const target = new URL(kaggleUrl + url.pathname + url.search);
    const outgoing = new Request(target, {
      method: request.method,
      headers: request.headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "manual",
    });
    outgoing.headers.set("Authorization", `Bearer ${kaggleKey}`);
    outgoing.headers.delete("Host");

    return fetch(outgoing);
  },
};
