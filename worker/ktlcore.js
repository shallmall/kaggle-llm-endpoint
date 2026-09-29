// ktlcore.js — shared relay helpers, pure Web Crypto (no Cloudflare imports)
// so they run under plain Node for tests.
//
//   * at-rest encryption of the stored session config: AES-256-GCM, key
//     derived from UPDATE_SECRET via HKDF-SHA256 (no third secret to manage)
//   * generation-rate limiting: fixed 60 s window, counted at request start

export const ENC_PREFIX = "enc:v1:";
const HKDF_SALT = new TextEncoder().encode("kaggle-tpu-relay/v1");
const HKDF_INFO = new TextEncoder().encode("ktl/do-config/v1");

// The only calls that count against the limit — actual model generation.
// Side calls (/v1/models, token counting, health) are exempt.
export const GENERATION_PATHS = [
  "/v1/chat/completions",
  "/v1/messages",
  "/v1/responses",
];
export const WINDOW_MS = 60_000;
export const DEFAULT_LIMIT = 120;

// Relay secrets shorter than this are treated as misconfigured: every request
// is denied (never open-fail). At 32+ random chars, key guessing is not a
// realistic attack, so no failed-auth throttling is layered on top.
export const MIN_SECRET_LEN = 32;

export function secretUsable(secret) {
  return typeof secret === "string" && secret.length >= MIN_SECRET_LEN;
}

// Constant-time key compare via SHA-256 digest (digests are fixed-length, so
// no early exit leaks the expected key's length).
export async function digestKeyMatch(provided, expected) {
  if (typeof provided !== "string" || typeof expected !== "string" || !provided) {
    return false;
  }
  const a = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(provided)),
  );
  const b = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(expected)),
  );
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a[i] ^ b[i];
  return diff === 0;
}

export async function deriveKey(secret) {
  const ikm = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    "HKDF",
    false,
    ["deriveKey"],
  );
  return crypto.subtle.deriveKey(
    { name: "HKDF", hash: "SHA-256", salt: HKDF_SALT, info: HKDF_INFO },
    ikm,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

// Fresh 12-byte nonce per call. Output: enc:v1:base64(nonce || ciphertext || tag).
// The field name is bound in as AES-GCM AAD, so a value stored under one
// slot (kaggleUrl) cannot be swapped into the other (kaggleKey) and still
// decrypt.
export async function encryptField(key, field, plaintext) {
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const aad = new TextEncoder().encode("ktl/" + field);
  const sealed = new Uint8Array(
    await crypto.subtle.encrypt(
      { name: "AES-GCM", iv: nonce, additionalData: aad },
      key,
      new TextEncoder().encode(plaintext),
    ),
  );
  const out = new Uint8Array(12 + sealed.length);
  out.set(nonce, 0);
  out.set(sealed, 12);
  return ENC_PREFIX + btoa(String.fromCharCode(...out));
}

// Values without the enc:v1: prefix are legacy plaintext and pass through
// untouched. Throws if the ciphertext is corrupt, the key no longer matches
// (e.g. UPDATE_SECRET was rotated after registration), or it was sealed under
// a different field name.
export async function decryptField(key, field, value) {
  if (typeof value !== "string" || !value.startsWith(ENC_PREFIX)) return value;
  const raw = Uint8Array.from(atob(value.slice(ENC_PREFIX.length)), (c) =>
    c.charCodeAt(0),
  );
  const aad = new TextEncoder().encode("ktl/" + field);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: raw.slice(0, 12), additionalData: aad },
    key,
    raw.slice(12),
  );
  return new TextDecoder().decode(plaintext);
}

export function shouldCount(method, pathname) {
  return method === "POST" && GENERATION_PATHS.includes(pathname);
}

// Fixed-window counter on a {get, put} storage object.
// Returns {ok: true} or {ok: false, retryAfterS}.
// The get/put pair is not atomic: two racing requests can admit one extra
// request past the limit — negligible at personal-relay volume.
export async function rateLimit(storage, limitPerMin) {
  const now = Date.now();
  const [windowStart, count] = await Promise.all([
    storage.get("rl_window"),
    storage.get("rl_count"),
  ]);
  if (!windowStart || now - windowStart >= WINDOW_MS) {
    await storage.put({ rl_window: now, rl_count: 1 });
    return { ok: true };
  }
  if (count >= limitPerMin) {
    return {
      ok: false,
      retryAfterS: Math.max(1, Math.ceil((windowStart + WINDOW_MS - now) / 1000)),
    };
  }
  await storage.put({ rl_count: count + 1 });
  return { ok: true };
}
