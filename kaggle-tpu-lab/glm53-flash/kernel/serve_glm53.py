"""
Serve GLM-5.3-Flash (320B MoE, 18B active) on a Kaggle TPU v5e-8 with our own JAX engine.

This script is pushed to Kaggle as a script kernel by ../../launch.py, which fills in the CFG line below and embeds
the engine package. It also runs standalone (pasted into a Kaggle notebook next to a `glm53/` folder written by the
previous cell) — then it prints instead of using ntfy.

Steps (each one is announced in the log):
  1/6  runtime  — pre-flight (datasets attached, Internet on, a real TPU: ~20 s, before anything slow), pinned libtpu
                  + a few pip packages (~1 min), cloudflared, the engine package
  2/6  weights  — the routed experts (3-bit codebook tables, Unsloth's UD-IQ3_XXS GGUF) and the non-expert weights
                  (int8) straight onto the eight chips (~6 min); the non-expert weights come from the serve dataset's
                  base store when it is attached, else from the FP8 checkpoint datasets
  3/6  vision   — the vision tower, sharded over the chips
  4/6  warm-up  — the prefill buckets, the batched decode programs and the snapshot programs (~9 min with the serve
                  dataset's compile cache, ~14 min cold: the rest is JAX tracing, which no cache skips)
  5/6  tunnel   — a public cloudflared URL (three attempts; a URL that never resolves is replaced once)
  6/6  ready    — READY banner + self-test, then keep serving until keepalive_min elapses

A cold run leaves `jax_cache/` (everything it compiled) and `base/` (the non-expert weights, vision tower and
tokenizer, ~11 GB) in /kaggle/working, so its output can be turned into the serve dataset ("New dataset" from the
kernel output) that later runs attach instead of the FP8 datasets.
"""
import base64, collections, glob, hashlib, io, json, os, queue, re, secrets, shutil, subprocess, sys, tarfile, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__  (launch.py replaces this line)

DEFAULTS = {
    "expert_datasets": ["rahim3/glm53-flash-iq3xxs-1", "rahim3/glm53-flash-iq3xxs-2"],   # Unsloth UD-IQ3_XXS GGUF, mirrored
    "base_datasets": ["rahim3/glm53-flash-fp8-1", "rahim3/glm53-flash-fp8-2",            # HF FP8 checkpoint, mirrored:
                      "rahim3/glm53-flash-fp8-3", "rahim3/glm53-flash-fp8-4"],           # non-expert weights, vision, tokenizer
    "serve_dataset": "rahim3/glm53-flash-serve",   # base/ (non-expert weights, vision, tokenizer) + jax_cache/ (compiled
                                     # programs); attached: the FP8 datasets are not needed and the warm-up is ~9 min instead of ~14
    "dump_base": True,               # a cold run writes base/ into its output (so the serve dataset can be made from it)
    "libtpu": "0.0.42.*",            # the image's runtime is 140x slower on gathers and cannot run Pallas kernels
    "max_len": 262144,               # context capacity (tokens); a multiple of 32
    "streams": 4,                    # requests decoded together (one program set per batch size, ~2.5 min compile each)
    "sets": 4,                       # cache sets on the chips: running streams + finished contexts kept for their next turn
    "piece": 512,                    # prefill piece (tokens): 512 keeps the prefill temporaries small enough for three
    "sched_piece": 512,              # 262k streams next to the engine (1024 is ~13 % faster prefill but fits two or three)
    "q_block": 32,                   # queries per attention block during prefill
    "cache_q8": True,                # int8 latent cache (a 262k set costs ~0.2 GB/chip instead of 0.3)
    "bucket_min": 256,               # smallest prefill bucket compiled (short prompts pad to it)
    "min_free_gb": 0.55,             # HBM headroom an admission needs (a cache set + the prefill temporaries of one piece)
    "max_queue": 8,                  # waiting requests beyond the streams before a 429
    "max_wait_s": 90.0,              # a request waiting longer than this gets a 503 (clients retry)
    "keepalive_s": 15,               # SSE ping when nothing was streamed for this long (a buffered tool call, a queued
                                     # request): Cloudflare drops a response that is silent for ~100 s
    "think_budget_default": 0,       # thinking tokens before  is forced, when the request sets no budget (0 = unlimited)
    "vision": True,                  # load the vision tower (images in both APIs); False saves ~1 min and 0.14 GB/chip
    "vision_max_tokens": 1024,       # 28x28-pixel tokens per image
    "reasoning_effort_default": "low",   # server-side default: low | high (anything else = the template's Max)
    "temperature": 1.0, "top_p": 0.95,   # generation_config defaults
    "max_new_default": 4096,
    "snap_host_gb": 48,              # host RAM for parked contexts (agent sessions that interleave)
    "snap_rows": 1024,               # snapshot row bucket (1024 rows = 8192 tokens sharded)
    "snap_min": 256,                 # contexts shorter than this are re-prefilled instead of parked
    "base_min": 512,                 # a system section at least this long is pinned for new sessions
    "snap_warm_tokens": 32768,       # warm the snapshot programs for contexts up to this many tokens
    "keepalive_min": 480,            # auto-shutdown guard (Kaggle TPU caps at 9h anyway)
    "api_key": "",                   # generated if empty
    "ntfy_topic": "",                # optional: publish progress to ntfy.sh/<topic> (launch.py watches it)
    "served_model_name": "glm-5.3-flash",
    "tunnel": True,                  # False: no cloudflared (local testing)
    "skip_runtime": False,           # True: no pip installs (local testing)
    "port": 8000,
}
CFG = {**DEFAULTS, **(CFG or {}), **globals().get("CFG_PRESET", {})}
_cfg_file = Path("serve_config.json")            # notebook flow: overrides next to this script
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    CFG["api_key"] = "glm-" + secrets.token_hex(12)

ENGINE_B64 = ""  # __ENGINE__  (launch.py embeds the glm53 package here; the notebook writes the files instead)

PORT = int(CFG["port"])
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("/tmp")
CACHE_DIR = WORK / "jax_cache"
CLOUDFLARED = Path("/tmp/cloudflared")
T0 = time.time()
LOG_LINES = []


def log(*parts):
    line = time.strftime("[%H:%M:%S] ") + " ".join(str(p) for p in parts)
    LOG_LINES.append(line)
    print(line, flush=True)


def elapsed():
    return f"{int(time.time() - T0) // 60} min {int(time.time() - T0) % 60:02d} s"


def banner(step, title, note=""):
    log("")
    log("=" * 70)
    log(f" STEP {step}/6  {title}" + (f"   ({note})" if note else "") + f"   [{elapsed()} so far]")
    log("=" * 70)


def publish(phase, **extra):
    """Progress event: always logged; also pushed to ntfy if a topic is set."""
    log(f"PHASE {phase}", json.dumps(extra) if extra else "")
    if not CFG["ntfy_topic"]:
        return
    try:
        body = {"topic": CFG["ntfy_topic"], "title": f"kaggle-tpu-lab {phase}", "message": json.dumps({"phase": phase, **extra})}
        req = urllib.request.Request("https://ntfy.sh", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # noqa: BLE001
        log(f"(ntfy publish failed: {e})")


def sh(cmd, tag):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    log(f"   {tag}: rc {r.returncode} in {time.time() - t:.0f}s" + (f" | {(r.stderr or '')[-200:].strip()}" if r.returncode else ""))
    return r.returncode


def mounts_of(names):
    """Dataset names ('owner/slug') -> mounted paths, in the given order (missing ones dropped)."""
    out = []
    for n in names:
        slug = n.split("/")[-1]
        hits = glob.glob(f"/kaggle/input/datasets/{n}") + glob.glob(f"/kaggle/input/{slug}")
        if hits:
            out.append(hits[0])
    return out


def hbm():
    st = jax.devices()[0].memory_stats() or {}
    return st.get("bytes_in_use", 0) / 1e9, (st.get("bytes_limit") or 0) / 1e9


def fail(step, msg):
    """Stop with a plain message (the launcher prints `step` and `tail` of a "failed" phase)."""
    log("   " + msg)
    publish("failed", step=step, tail=msg)
    sys.exit(1)


def preflight():
    """Look before the slow steps (~20 s): the datasets attached, Internet on (pip, cloudflared) and a real TPU present.
    Kaggle sometimes starts a "TPU" session with no TPU (a CPU-only container, most often on new or not-yet-verified
    accounts): jax then sees one CPU device and the build dies minutes later with a sharding error."""
    need = list(CFG["expert_datasets"]) + ([] if CFG["serve_dataset"] and mounts_of([CFG["serve_dataset"]]) else
                                           ([CFG["serve_dataset"]] if CFG["serve_dataset"] and not mounts_of(CFG["base_datasets"]) else list(CFG["base_datasets"])))
    missing = [n for n in need if not mounts_of([n])]
    if missing:
        fail("datasets", f"datasets not attached: {missing}. In the right sidebar, Add Input -> search each name -> attach, then run again.")
    try:
        urllib.request.urlopen("https://pypi.org/simple/pip/", timeout=20).read(1)
    except Exception as e:  # noqa: BLE001
        fail("no-internet", f"no Internet from this session ({str(e)[:120]}): Session options -> Internet ON (a phone-verified "
                            "Kaggle account is needed for that), then run again. The pip packages and the tunnel need it.")
    code = ("import jax\n"
            "try:\n"
            "    d = jax.devices()\n"
            "    print('TPU_CHECK', len(d), d[0].platform, getattr(d[0], 'device_kind', ''))\n"
            "except Exception as e:\n"
            "    print('TPU_CHECK 0 none', str(e).replace(chr(10), ' ')[:200])\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        m = re.search(r"TPU_CHECK (\d+) (\S+)(.*)", (r.stdout or "") + (r.stderr or ""))
    except Exception as e:  # noqa: BLE001
        log(f"   (TPU check skipped: {e})"); return
    if m is None:
        log("   (TPU check inconclusive: the image's jax did not answer; continuing)"); return
    n, platform, rest = int(m.group(1)), m.group(2), m.group(3).strip()
    if n == 8 and platform == "tpu":
        log(f"   pre-flight OK: datasets attached, Internet on, 8 TPU chips ({rest})"); return
    if n == 0 and not re.search(r"jellyfish|TPU initialization failed|initialize backend 'tpu'|No TPU|vfio", rest, re.I):
        log(f"   (TPU check inconclusive, continuing: {rest[:160]})"); return
    fail("no-tpu", f"this session has no working TPU: jax sees {n} {platform} device(s) {rest}. Kaggle sometimes starts a TPU "
                   "session without one (most often on new or not-yet-verified accounts); nothing in this notebook can fix "
                   "that. Stop the session and start it again; `import jax; print(jax.device_count())` in a fresh cell must "
                   "print 8 before this script is worth running.")


# ----------------------------------------------------------------------------- 1. runtime
if not CFG["skip_runtime"]:
    banner(1, "Runtime", "pre-flight, pinned libtpu + pip packages, cloudflared, the engine")
    preflight()
    sh([sys.executable, "-m", "pip", "install", "-q", "safetensors", "huggingface_hub", "transformers>=5.16", "pillow",
        "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple"], "pip packages")
    if CFG["libtpu"]:
        sh([sys.executable, "-m", "pip", "install", "-q", f"libtpu=={CFG['libtpu']}"], f"libtpu=={CFG['libtpu']}")
    if ENGINE_B64 and not ENGINE_B64.startswith("__"):
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ENGINE_B64)), mode="r:gz") as tf:
            tf.extractall(WORK)
        sys.path.insert(0, str(WORK))
        log(f"   engine package extracted to {WORK}/glm53")
    elif not Path("glm53").is_dir():
        sys.exit("no glm53/ package next to this script and nothing embedded — run the notebook's engine cell first")

if CFG["tunnel"] and not CLOUDFLARED.exists():
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CLOUDFLARED)
    CLOUDFLARED.chmod(0o755)
    log("   cloudflared downloaded")

import numpy as np                      # noqa: E402  (after the runtime pins: libtpu must be installed before jax loads)
import jax, jax.numpy as jnp            # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402

SERVE_MOUNT = (mounts_of([CFG["serve_dataset"]]) or [None])[0] if CFG["serve_dataset"] else None
BASE_DIR = os.path.join(SERVE_MOUNT, "base") if SERVE_MOUNT and os.path.isdir(os.path.join(SERVE_MOUNT, "base")) else None
if "eng" not in globals():              # (a test harness may pre-set eng/tok/ids/eos/cfg and skip the build)
    cache_src = os.path.join(SERVE_MOUNT, "jax_cache") if SERVE_MOUNT and os.path.isdir(os.path.join(SERVE_MOUNT, "jax_cache")) else None
    if cache_src and not CACHE_DIR.exists():
        t = time.time()
        shutil.copytree(cache_src, CACHE_DIR)
        log(f"   compile cache restored from {cache_src} in {time.time() - t:.0f}s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))
    try:                                 # a process that already initialised the cache elsewhere keeps writing there
        from jax._src import compilation_cache as _cc
        _cc.reset_cache()
    except Exception as e:  # noqa: BLE001
        log(f"   (compile cache reset skipped: {e!r})")
    log(f"   jax {jax.__version__}, {jax.device_count()} devices, compile cache at {CACHE_DIR}")

# ----------------------------------------------------------------------------- 2. weights -> the chips
if "eng" not in globals():
    banner(2, "Weights", "3-bit codebook experts + int8 non-expert weights onto the eight chips, ~9 min")
    from glm53 import basestore as BS
    from glm53 import checkpoint as C
    from glm53 import model as M
    from glm53.engine import AXIS
    from glm53 import gguf_reader as G
    from glm53.resident import ResidentFetch, ResidentLayerEngine, pack_layer_from_gguf, hbm_bytes
    from transformers import AutoConfig, AutoTokenizer
    gguf_mounts = mounts_of(CFG["expert_datasets"])
    fp8_mounts = [] if BASE_DIR else mounts_of(CFG["base_datasets"])
    if len(gguf_mounts) < len(CFG["expert_datasets"]) or (not BASE_DIR and len(fp8_mounts) < len(CFG["base_datasets"])):
        fail("datasets", f"datasets missing: found gguf {gguf_mounts}, fp8 {fp8_mounts}, serve {SERVE_MOUNT}; attach "
                         f"{CFG['expert_datasets']} + {CFG['serve_dataset'] or CFG['base_datasets']}")
    hf_dir = BASE_DIR or fp8_mounts[0]
    log(f"   experts from {gguf_mounts}; the rest from {'the base store ' + BASE_DIR if BASE_DIR else 'the FP8 checkpoint'}")
    publish("loading", note="weights")
    hf_cfg = AutoConfig.from_pretrained(hf_dir)
    tc = hf_cfg.text_config
    N_LAYERS = tc.num_hidden_layers
    cfg = M.Cfg.from_hf(tc, dtype=jnp.bfloat16)
    N_DEV = jax.device_count()
    mesh = Mesh(np.array(jax.devices()), (AXIS,))
    tbl_sh = NamedSharding(mesh, P(AXIS))
    gm = G.GGUFModel(G.open_mirror(gguf_mounts))
    t_load = time.time()
    if BASE_DIR:
        store = BS.load(BASE_DIR)
        params = {**store["top"], "layers": []}
        log(f"   base store read in {time.time() - t_load:.0f}s ({store['manifest'].get('n_layers')} layers)")
        r = None
    else:
        r = C.RawShardReader(":".join(fp8_mounts))
        top = C.load_top(r, np.float32)
        params = {"embed": top["embed"].astype(jnp.bfloat16), "norm": top["norm"], "lm_head": top["lm_head"].astype(jnp.bfloat16), "layers": []}
        del top
    qtypes, hbm_total = {}, 0
    for i in range(N_LAYERS):
        ti = time.time()
        if BASE_DIR:
            p = store["layers"][i]
        else:
            p = C.load_layer(r, i, tc.layer_types[i], tc.mlp_layer_types[i], np.float32)
            p = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if a.ndim >= 2 else a, p)
        if tc.mlp_layer_types[i] == "sparse":
            tables, qt = pack_layer_from_gguf(gm, i, N_DEV, threads=16)
            qtypes[i] = qt
            hbm_total += hbm_bytes(tables) * N_DEV
            for k, t in tables.items():                    # each table straight onto the chips: host RAM holds one layer
                p["mlp"][k] = jax.tree.map(lambda a: jax.device_put(a, tbl_sh), t)
            jax.block_until_ready([p["mlp"][k] for k in tables])
            del tables
        params["layers"].append(p)
        if r is not None:
            for f in C.layer_shards(":".join(fp8_mounts), i):
                r.release(f)
        if i % 5 == 0 or i == N_LAYERS - 1:
            log(f"   layer {i + 1}/{N_LAYERS} ({time.time() - ti:.0f}s) | HBM chip0 {hbm()[0]:.2f} GB")
    log(f"   expert tables {hbm_total / 1e9:.1f} GB across {N_DEV} chips; weights read in {(time.time() - t_load) / 60:.1f} min")
    fetch = ResidentFetch(qtypes, tc.hidden_size, tc.moe_intermediate_size // N_DEV, out_dtype=jnp.bfloat16, gather_max_rows=64, chunk=8,
                          use_pallas=True, sweep_mode="ragged", tm=32, combine="matmul")
    eng = ResidentLayerEngine(cfg, params, fetch, max_len=CFG["max_len"], layers_per_program=4, layers_per_program_prefill=1,
                              int8_nonexpert=False if BASE_DIR else "all", seq_shard=True, q_block=CFG["q_block"],
                              prefill_piece=CFG["piece"], cache_q8=CFG["cache_q8"])
    del params
    use, lim = hbm()
    log(f"   ENGINE READY in {(time.time() - t_load) / 60:.1f} min: HBM chip0 {use:.2f} / {lim:.2f} GB, context {CFG['max_len']}")
    publish("loaded", hbm_gb=round(use, 2), minutes=round((time.time() - t_load) / 60, 1))
    tok = AutoTokenizer.from_pretrained(hf_dir)
    ids = np.asarray(tok(tok.apply_chat_template([{"role": "user", "content": "Write a haiku about tensor processing units."}],
                                                 tokenize=False, add_generation_prompt=True, reasoning_effort="low"),
                         add_special_tokens=False).input_ids).reshape(1, -1)
    eos = set(json.load(open(os.path.join(hf_dir, "config.json")))["text_config"].get("eos_token_id", [tok.eos_token_id]))

# ----------------------------------------------------------------------------- 3. vision tower
VISION, VISION_FWD = globals().get("VISION"), globals().get("VISION_FWD")
_vp = None
if CFG["vision"] and VISION_FWD is None and "hf_dir" in globals():
    banner(3, "Vision", "the vision tower sharded over the chips")
    from glm53 import vision as VIS
    t = time.time()
    if BASE_DIR and store.get("vision") is not None:
        _vp = store["vision"]
    else:
        _vp = VIS.load_vision(C.RawShardReader(":".join(fp8_mounts)), dtype=np.float32)
        _vp = jax.tree.map(lambda a: a.astype(jnp.bfloat16) if getattr(a, "dtype", None) == np.float32 else a, _vp)
    VISION = VIS.to_sharded(_vp, eng.mesh, dtype=jnp.bfloat16)
    VISION_FWD = VIS.make_forward_sharded(VISION, eng.mesh, dtype=jnp.bfloat16, q_chunk=1024)
    from PIL import Image
    for target in [b for b in (256, 512, 1024, 2048, 4096) if b <= 4 * CFG["vision_max_tokens"]]:
        side = int(target ** 0.5) * 14 - 14
        img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (side, side, 3), dtype=np.uint8))
        patches, grid = VIS.preprocess(img, max_tokens=CFG["vision_max_tokens"])
        n = patches.shape[0]
        bucket = max(256, 1 << (n - 1).bit_length())
        grids = (grid,) if bucket == n else (grid, (1, 2, (bucket - n) // 2))
        if bucket > n:
            patches = np.concatenate([patches, np.zeros((bucket - n, patches.shape[1]), patches.dtype)], 0)
        jax.block_until_ready(VISION_FWD(patches, grids))
    log(f"   vision tower ready in {time.time() - t:.0f}s; HBM chip0 {hbm()[0]:.2f} GB")
if CFG["dump_base"] and not BASE_DIR and "hf_dir" in globals():
    t = time.time()
    BS.dump(str(WORK / "base"), eng.params, vision=_vp, hf_dir=hf_dir, log=log,
            meta={"model": CFG["served_model_name"], "int8_nonexpert": "all", "experts": "resident (not stored)"})
    log(f"   base store dumped in {time.time() - t:.0f}s (this run's output can become the serve dataset)")
del _vp

# ----------------------------------------------------------------------------- 4. the server
from glm53.resident import ResidentLayerEngine as _RLE   # noqa: E402
from glm53.engine import DeviceSampler                    # noqa: E402
from glm53.scheduler import Request, Scheduler, SnapStore # noqa: E402
from glm53 import vision as VIS                           # noqa: E402

API_KEY = CFG["api_key"]
MAX_NEW_DEFAULT = CFG["max_new_default"]
KEEPALIVE_S = float(CFG["keepalive_s"] or 0)
THINK_BUDGET_DEFAULT = int(CFG["think_budget_default"] or 0)
MAX_STREAMS, MAX_SETS = CFG["streams"], CFG["sets"]
SCHED_PIECE = CFG["sched_piece"]
MIN_FREE_GB, MAX_WAIT_S, MAX_QUEUE = CFG["min_free_gb"], CFG["max_wait_s"], CFG["max_queue"]
VISION_MAX_TOKENS = CFG["vision_max_tokens"]
DEFAULT_EFFORT = CFG["reasoning_effort_default"]
DEFAULT_TEMP, DEFAULT_TOP_P = CFG["temperature"], CFG["top_p"]
MODEL = CFG["served_model_name"]
BUCKET_MIN = CFG["bucket_min"]
if BUCKET_MIN:
    eng.PREFILL_BUCKETS = tuple(b for b in type(eng).PREFILL_BUCKETS if b >= BUCKET_MIN)
STATE = {"url": None, "requests": 0, "tokens": 0, "prefix_hits": 0, "prefix_tokens_reused": 0, "prefill_s": 0.0,
         "snap_hits": 0, "snap_parks": 0, "snap_pins": 0, "snap_entries": 0, "snap_bytes": 0, "snap_s": 0.0,
         "steps": 0, "step_tokens": 0}
T = {k: tok.convert_tokens_to_ids(k) for k in ("<think>", "", "", "", "<arg_key>", "</arg_key>",
                                                "<arg_value>", "</arg_value>", "<|user|>", "<|observation|>", "<|assistant|>",
                                                "<|image|>", "<|begin_of_image|>", "<|end_of_image|>")}
STOP_IDS = set(eos) | {T["<|user|>"], T["<|observation|>"]}

# ---- images
IMG_CACHE = collections.OrderedDict()                  # sha256 -> (n_tokens, embeddings f32 [n, D]); LRU
IMG_CACHE_MAX = 64


def embed_image(img_bytes):
    """Image bytes -> (n_tokens, embeddings [n, D] f32) through the vision tower; cached by content hash. The patch
    count is padded to a power of two with a dummy image segment (its rows are dropped) so few shapes compile."""
    from PIL import Image
    if VISION_FWD is None:
        raise ValueError("this server has no vision tower loaded (config: vision)")
    h = hashlib.sha256(img_bytes).hexdigest()
    if h in IMG_CACHE:
        IMG_CACHE.move_to_end(h)
        return IMG_CACHE[h]
    patches, grid = VIS.preprocess(Image.open(io.BytesIO(img_bytes)), max_tokens=VISION_MAX_TOKENS)
    n = patches.shape[0]
    bucket = max(256, 1 << (n - 1).bit_length())
    grids = (grid,) if bucket == n else (grid, (1, 2, (bucket - n) // 2))
    if bucket > n:
        patches = np.concatenate([patches, np.zeros((bucket - n, patches.shape[1]), patches.dtype)], 0)
    t = time.time()
    out = np.asarray(VISION_FWD(patches, grids), np.float32)[:VIS.n_tokens(grid)]
    STATE["vision_s"] = STATE.get("vision_s", 0.0) + time.time() - t
    STATE["images"] = STATE.get("images", 0) + 1
    IMG_CACHE[h] = (out.shape[0], out)
    while len(IMG_CACHE) > IMG_CACHE_MAX:
        IMG_CACHE.popitem(last=False)
    return IMG_CACHE[h]


def _image_bytes(block):
    """Anthropic image block or OpenAI image_url part -> raw bytes (base64 or data: URL inline; http(s) fetched)."""
    if block.get("type") == "image":
        src = block.get("source", {})
        if src.get("type") == "base64":
            return base64.b64decode(src["data"])
        url = src.get("url", "")
    else:
        u = block.get("image_url", "")
        url = u.get("url", "") if isinstance(u, dict) else u
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith("http://") or url.startswith("https://"):
        return urllib.request.urlopen(url, timeout=30).read()
    raise ValueError("unsupported image source")


def _img_sig(img_bytes):
    """Negative pseudo token id identifying an image in prompt signatures (all of its tokens carry it)."""
    return -(1 + int(hashlib.sha256(img_bytes).hexdigest()[:8], 16) % (1 << 30))


def _dec(ids):
    return tok.decode([T["<|image|>"] if int(i) < 0 else int(i) for i in ids])


def _feed(sig, imgs):
    """Signature ids (image tokens negative) -> (real ids int32 [n], embeds (idx, vec) or None) for the engine."""
    sig_a, off = sig
    ids = np.where(sig_a < 0, T["<|image|>"], sig_a).astype(np.int32)
    idx = np.nonzero(sig_a < 0)[0]
    if len(idx) == 0:
        return ids, None
    vec = np.concatenate([imgs[int(sig_a[i])][1][off[i]:off[i] + 1] for i in idx], 0)
    return ids, (idx, vec)


def _run_offsets(sig):
    """Per position: index within its run of equal negative ids (0 for text)."""
    sig = np.asarray(sig)
    off = np.zeros(len(sig), np.int64)
    for i in range(1, len(sig)):
        if sig[i] < 0 and sig[i] == sig[i - 1]:
            off[i] = off[i - 1] + 1
    return off


# ---- sampling of the first token (the rest are sampled on the device)
def sample(logits, temperature, top_p, rng, n_cand=2048):
    z = np.asarray(logits, dtype=np.float32).reshape(-1)
    if temperature <= 0:
        return int(z.argmax())
    n_cand = min(n_cand, z.size - 1)
    cand = np.argpartition(-z, n_cand)[:n_cand]
    zc = z[cand] / temperature
    zc = zc - zc.max()
    p = np.exp(zc); p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p); cum = np.cumsum(p[order])
        keep = order[: max(1, int((cum <= top_p).sum()) + 1)]
        q = np.zeros_like(p); q[keep] = p[keep]; p = q / q.sum()
    return int(cand[rng.choice(n_cand, p=p)])


BASE_MIN, SNAP_HOST_GB = CFG["base_min"], CFG["snap_host_gb"]
SNAP_ROWS, SNAP_MIN, SNAP_WARM = CFG["snap_rows"], CFG["snap_min"], CFG["snap_warm_tokens"]


def system_end(prompt):
    """Index of the first <|user|> token = end of the rendered system section (system prompt + tools)."""
    try:
        return prompt.index(T["<|user|>"])
    except ValueError:
        return 0


def _match_len(live_ids, pos, prompt, quiet=False):
    """Longest reuse of the live context (ids[:pos]) for `prompt`: (k, fed) where prompt[k:] must still be prefilled and
    `fed` = the ids the engine will have seen after that, or (0, None). Handles a dropped thinking block (the template
    renders `<think>` where the live context holds the reasoning) and re-tokenisation drift near the boundary."""
    THINK, END = T["<|assistant|>"], T[""]
    i = j = 0
    while True:
        n = min(pos - i, len(prompt) - j)
        if n > 0:
            neq = np.asarray(live_ids[i:i + n]) != np.asarray(prompt[j:j + n])
            d = int(np.argmax(neq)) if neq.any() else n
            i += d; j += d
        if i >= pos:
            return (j, list(live_ids[:pos]) + list(prompt[j:])) if j <= len(prompt) else (0, None)
        if j >= len(prompt):
            return 0, None
        if i > 0 and live_ids[i - 1] == T["<think>"] and prompt[j] == END and END in live_ids[i:pos]:
            i = live_ids.index(END, i) + 1; j += 1
            continue
        if pos - i > 256:
            return 0, None
        a = max(0, i - 32); ja = j - (i - a)
        t_live = _dec(live_ids[a:pos])
        for k in range(max(ja + 1, j - 24), min(len(prompt), j + (pos - i) + 24) + 1):
            if _dec(prompt[ja:k]) == t_live:
                return k, list(live_ids[:pos]) + list(prompt[k:])
        return 0, None


RNG = np.random.default_rng()


def sched_feed(req, a, b):
    return _feed((req.sig[a:b], req.off[a:b]), req.imgs or {})


_drop = [k for k in eng._progs if isinstance(k, tuple) and k[0] == "group" and k[3] == 1 and k[5]]   # batch-1 loop programs: unused here
for k in _drop:
    del eng._progs[k]
SNAPS = SnapStore(eng, int(SNAP_HOST_GB * 1e9), SNAP_ROWS, match_len=_match_len, log=log, state=STATE)
SCHED = Scheduler(eng, STOP_IDS, MAX_STREAMS, MAX_SETS, feed=sched_feed, match_len=_match_len, system_end=system_end,
                  first_sample=lambda z, t, p: sample(z, t, p, RNG), snaps=SNAPS, log=log, state=STATE,
                  base_min=BASE_MIN, snap_min=SNAP_MIN, piece=SCHED_PIECE, min_free_gb=MIN_FREE_GB, max_wait_s=MAX_WAIT_S)


class QueueFull(Exception):
    """Too many requests in flight (-> 429)."""


class ClientGone(Exception):
    """The client closed the connection mid-response."""


_IDLE = object()


def generate(prompt, max_new, temperature, top_p, on_token=None, imgs=None, rid=None, on_idle=None, budget=None):
    """One request through the scheduler -> (out ids, prefill_s, decode_s, reused, reason) with reason "stop" | "length"
    | "stop_sequence" (`on_token` returned True: cancelled there) | "cancelled". Tokens reach `on_token` on THIS thread;
    `on_idle` is called when no token arrived for KEEPALIVE_S (a queued request); `budget` = Request.budget."""
    prompt = [int(t) for t in prompt]
    if MAX_QUEUE and len(SCHED.pending) + len(SCHED.active) >= MAX_STREAMS + MAX_QUEUE:
        raise QueueFull(f"{len(SCHED.active)} requests running and {len(SCHED.pending)} waiting; retry later")
    q = queue.Queue()
    req = Request(prompt, max_new, temperature, top_p, imgs=imgs, rid=rid, budget=budget,
                  on_token=q.put if on_token else None, on_done=(lambda: q.put(None)) if on_token else None)
    req.sig = np.asarray(prompt, np.int64)
    req.off = _run_offsets(req.sig) if imgs else np.zeros(len(prompt), np.int64)
    SCHED.submit(req)
    stopped = False
    if on_token:
        while True:
            try:
                t = q.get(timeout=KEEPALIVE_S or None)
            except queue.Empty:
                t = _IDLE
            if t is None:
                break
            if stopped:
                continue
            try:
                if t is _IDLE:
                    if on_idle is not None:
                        on_idle()
                elif on_token(t):
                    stopped = True
                    req.cancel()
            except Exception:
                req.cancel()
                raise
    req.done.wait()
    if req.error is not None:
        raise req.error
    return req.out, req.prefill_s, req.decode_s, req.reused, ("stop_sequence" if stopped else req.stop_reason)


# ---- output parsing (token level)
class StopFilter:
    """`stop_sequences` on the TEXT events: `feed(text) -> (emit, hit)` holds back a tail that could begin a stop
    sequence; at a match returns the text before it and the sequence; `flush()` releases the held tail."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.buf, self.hit = "", None

    def feed(self, text):
        if not self.stops:
            return text, None
        self.buf += text
        best = None
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, s)
        if best is not None:
            out, self.buf, self.hit = self.buf[:best[0]], "", best[1]
            return out, best[1]
        keep = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), keep, -1):
                if self.buf.endswith(s[:k]):
                    keep = k; break
        cut = len(self.buf) - keep
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out, None

    def flush(self):
        out, self.buf = self.buf, ""
        return out


def parse_tool_call(body, tools):
    """body = tokens between ."""
    name_end = body.index(T["<arg_key>"]) if T["<arg_key>"] in body else len(body)
    name = tok.decode(body[:name_end]).strip()
    schema = {}
    for t in tools or []:
        f = t.get("function", t)
        if f.get("name") == name:
            schema = (f.get("parameters") or f.get("input_schema") or {}).get("properties", {}) or {}
    args, i = {}, name_end
    while i < len(body) and T["<arg_key>"] in body[i:]:
        a = body.index(T["<arg_key>"], i) + 1
        b = body.index(T["</arg_key>"], a) if T["</arg_key>"] in body[a:] else len(body)
        key = tok.decode(body[a:b]).strip()
        c = body.index(T["<arg_value>"], b) + 1 if T["<arg_value>"] in body[b:] else len(body)
        d = body.index(T["</arg_value>"], c) if T["</arg_value>"] in body[c:] else len(body)
        raw = tok.decode(body[c:d])
        typ = schema.get(key, {}).get("type")
        if typ == "string":
            val = raw
        else:
            try:
                val = json.loads(raw)
            except Exception:  # noqa: BLE001
                val = raw
        args[key] = val
        i = d + 1
    return {"id": "call_" + uuid.uuid4().hex[:12], "name": name, "input": args}


class TokenStream:
    """Incremental token -> (kind, text/tool) events: kind in {"thinking", "text", "tool"}; partial multibyte text
    is held back until it decodes cleanly; tool-call tokens are buffered until ."""

    def __init__(self, tools, mode="thinking"):
        self.tools, self.mode, self.buf, self.tool_buf = tools, mode, [], []

    def feed(self, t):
        if t in STOP_IDS:
            return self.flush()
        if self.mode == "tool":
            if t == T[""]:
                self.mode = "text"; call = parse_tool_call(self.tool_buf, self.tools); self.tool_buf = []
                return [("tool", call)]
            self.tool_buf.append(t); return []
        if t == T[""] and self.mode == "thinking":
            ev = self.flush(); self.mode = "text"; return ev
        if t == T[""] and self.mode == "text":
            self.mode = "tool"; self.tool_buf = []; return self.flush()
        self.buf.append(t)
        try:
            text = tok.decode(self.buf)
            self.buf = []
            return [("text", text)] if text else []
        except Exception:
            return []

    def flush(self):
        evs = []
        if self.mode == "tool" and self.tool_buf:
            call = parse_tool_call(self.tool_buf, self.tools)
            evs.append(("tool", call))
            self.tool_buf = []
        if self.buf:
            try:
                text = tok.decode(self.buf)
                if text:
                    evs.append(("text", text))
            except Exception:
                pass
            self.buf = []
        return evs


# ---- prompt building
def build_prompt(messages, tools=None, effort=None):
    """Chat messages -> (token_ids list, imgs dict or None). Handles text and image content blocks."""
    effort = effort or DEFAULT_EFFORT
    tmpl_msgs = []
    imgs = {}
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "image_url":
                    u = block.get("image_url", {})
                    url = u.get("url", "") if isinstance(u, dict) else u
                    if url.startswith("data:"):
                        img_bytes = base64.b64decode(url.split(",", 1)[1])
                    elif url.startswith("http"):
                        img_bytes = urllib.request.urlopen(url, timeout=30).read()
                    else:
                        continue
                    n_tok, emb = embed_image(img_bytes)
                    sig = _img_sig(img_bytes)
                    imgs[sig] = (n_tok, emb)
                    parts.append("<|image|>" * n_tok)
                elif btype == "image":
                    img_bytes = _image_bytes(block)
                    n_tok, emb = embed_image(img_bytes)
                    sig = _img_sig(img_bytes)
                    imgs[sig] = (n_tok, emb)
                    parts.append("<|image|>" * n_tok)
            content = "\n".join(parts)
        tmpl_msgs.append({"role": role, "content": content})
    rendered = tok.apply_chat_template(tmpl_msgs, tokenize=False, add_generation_prompt=True,
                                       reasoning_effort=effort, tools=tools)
    token_ids = tok(rendered, add_special_tokens=False).input_ids
    return token_ids, (imgs if imgs else None)


# ---- HTTP server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _auth_ok(self):
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {API_KEY}"

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, msg):
        self._send_json(code, {"error": {"message": msg, "type": "error"}})

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _sse_write(self, data):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _sse_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            if not self._auth_ok():
                self._send_error(401, "Invalid API key")
                return
            self._send_json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "glm"}]})
        elif self.path in ("/health", "/"):
            self._send_json(200, {"status": "ok", "model": MODEL})
        else:
            self._send_error(404, "Not found")

    def do_POST(self):
        if not self._auth_ok():
            self._send_error(401, "Invalid API key")
            return
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_openai_chat()
        elif self.path in ("/v1/messages", "/messages"):
            self._handle_anthropic_messages()
        else:
            self._send_error(404, "Not found")

    def _handle_openai_chat(self):
        body = self._read_body()
        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or MAX_NEW_DEFAULT
        temperature = body.get("temperature", DEFAULT_TEMP)
        top_p = body.get("top_p", DEFAULT_TOP_P)
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        effort = body.get("reasoning_effort") or (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]

        try:
            prompt_ids, imgs = build_prompt(messages, tools=tools, effort=effort)
        except Exception as e:
            self._send_error(400, f"Failed to build prompt: {e}")
            return

        stop_filter = StopFilter(stop)
        ts = TokenStream(tools)
        thinking_parts, text_parts, tool_calls = [], [], []
        n_out = 0

        def on_token(tid):
            nonlocal n_out
            n_out += 1
            STATE["tokens"] += 1
            for kind, val in ts.feed(tid):
                if kind == "thinking":
                    thinking_parts.append(val)
                elif kind == "text":
                    emit, hit = stop_filter.feed(val)
                    if emit:
                        text_parts.append(emit)
                    if hit:
                        return True
                elif kind == "tool":
                    tool_calls.append(val)
            return False

        def on_idle():
            if stream:
                try:
                    self._sse_write({"id": rid, "object": "chat.completion.chunk", "choices": []})
                except Exception:
                    raise ClientGone()

        try:
            out_ids, prefill_s, decode_s, reused, reason = generate(
                prompt_ids, max_tokens, temperature, top_p,
                on_token=on_token if stream else None,
                imgs=imgs, rid=rid, on_idle=on_idle if stream else None)
        except QueueFull as e:
            self._send_error(429, str(e))
            return
        except ClientGone:
            return
        except Exception as e:
            self._send_error(500, f"Generation failed: {e}")
            return

        for kind, val in ts.flush():
            if kind == "thinking":
                thinking_parts.append(val)
            elif kind == "text":
                emit, _ = stop_filter.feed(val)
                if emit:
                    text_parts.append(emit)
            elif kind == "tool":
                tool_calls.append(val)
        remaining = stop_filter.flush()
        if remaining:
            text_parts.append(remaining)

        thinking_text = "".join(thinking_parts)
        text_text = "".join(text_parts)
        finish = "stop" if reason in ("stop", "stop_sequence") else "length"
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": n_out, "total_tokens": len(prompt_ids) + n_out}

        if stream:
            try:
                self._sse_write({"id": rid, "object": "chat.completion.chunk",
                                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                                 "usage": usage})
                self._sse_done()
            except Exception:
                pass
        else:
            msg = {"role": "assistant", "content": text_text or None}
            if thinking_text:
                msg["reasoning_content"] = thinking_text
            if tool_calls:
                msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                      "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}}
                                     for tc in tool_calls]
            self._send_json(200, {
                "id": rid, "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": usage,
            })
        STATE["requests"] += 1
        STATE["prefill_s"] += prefill_s

    def _handle_anthropic_messages(self):
        body = self._read_body()
        messages = body.get("messages", [])
        system = body.get("system", "")
        tools = body.get("tools")
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", MAX_NEW_DEFAULT)
        temperature = body.get("temperature", DEFAULT_TEMP)
        top_p = body.get("top_p", DEFAULT_TOP_P)
        stop = body.get("stop_sequences")
        effort = body.get("reasoning_effort")
        rid = "msg_" + uuid.uuid4().hex[:24]

        oa_msgs = []
        if system:
            oa_msgs.append({"role": "system", "content": system if isinstance(system, str) else "\n".join(system)})
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "image":
                        img_bytes = _image_bytes(block)
                        n_tok, emb = embed_image(img_bytes)
                        sig = _img_sig(img_bytes)
                        parts.append(f"[image:{sig}]")
                content = "\n".join(parts)
            oa_msgs.append({"role": role, "content": content})

        try:
            prompt_ids, imgs = build_prompt(oa_msgs, tools=tools, effort=effort)
        except Exception as e:
            self._send_error(400, f"Failed to build prompt: {e}")
            return

        stop_filter = StopFilter(stop)
        ts = TokenStream(tools)
        thinking_parts, text_parts, tool_calls = [], [], []
        n_out = 0

        def on_token(tid):
            nonlocal n_out
            n_out += 1
            STATE["tokens"] += 1
            for kind, val in ts.feed(tid):
                if kind == "thinking":
                    thinking_parts.append(val)
                elif kind == "text":
                    emit, hit = stop_filter.feed(val)
                    if emit:
                        text_parts.append(emit)
                    if hit:
                        return True
                elif kind == "tool":
                    tool_calls.append(val)
            return False

        def on_idle():
            if stream:
                try:
                    self._sse_write({"type": "ping"})
                except Exception:
                    raise ClientGone()

        try:
            out_ids, prefill_s, decode_s, reused, reason = generate(
                prompt_ids, max_tokens, temperature, top_p,
                on_token=on_token if stream else None,
                imgs=imgs, rid=rid, on_idle=on_idle if stream else None)
        except QueueFull as e:
            self._send_error(429, str(e))
            return
        except ClientGone:
            return
        except Exception as e:
            self._send_error(500, f"Generation failed: {e}")
            return

        for kind, val in ts.flush():
            if kind == "thinking":
                thinking_parts.append(val)
            elif kind == "text":
                emit, _ = stop_filter.feed(val)
                if emit:
                    text_parts.append(emit)
            elif kind == "tool":
                tool_calls.append(val)
        remaining = stop_filter.flush()
        if remaining:
            text_parts.append(remaining)

        thinking_text = "".join(thinking_parts)
        text_text = "".join(text_parts)
        finish = "end_turn" if reason in ("stop", "stop_sequence") else "max_tokens"
        usage = {"input_tokens": len(prompt_ids), "output_tokens": n_out}

        if stream:
            try:
                self._sse_write({"type": "message_start", "message": {"id": rid, "type": "message", "role": "assistant",
                                                                       "content": [], "model": MODEL, "usage": usage}})
                idx = 0
                if thinking_text:
                    self._sse_write({"type": "content_block_start", "index": idx,
                                     "content_block": {"type": "thinking", "thinking": ""}})
                    self._sse_write({"type": "content_block_delta", "index": idx,
                                     "delta": {"type": "thinking_delta", "thinking": thinking_text}})
                    self._sse_write({"type": "content_block_stop", "index": idx})
                    idx += 1
                self._sse_write({"type": "content_block_start", "index": idx,
                                 "content_block": {"type": "text", "text": ""}})
                self._sse_write({"type": "content_block_delta", "index": idx,
                                 "delta": {"type": "text_delta", "text": text_text}})
                self._sse_write({"type": "content_block_stop", "index": idx})
                for i, tc in enumerate(tool_calls):
                    tidx = idx + 1 + i
                    self._sse_write({"type": "content_block_start", "index": tidx,
                                     "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}}})
                    self._sse_write({"type": "content_block_delta", "index": tidx,
                                     "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc["input"])}})
                    self._sse_write({"type": "content_block_stop", "index": tidx})
                self._sse_write({"type": "message_delta", "delta": {"stop_reason": finish, "stop_sequence": None},
                                 "usage": {"output_tokens": n_out}})
                self._sse_write({"type": "message_stop"})
            except Exception:
                pass
        else:
            content = []
            if thinking_text:
                content.append({"type": "thinking", "thinking": thinking_text})
            content.append({"type": "text", "text": text_text})
            for tc in tool_calls:
                content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
            self._send_json(200, {
                "id": rid, "type": "message", "role": "assistant", "model": MODEL,
                "content": content, "stop_reason": finish, "stop_sequence": None,
                "usage": usage,
            })
        STATE["requests"] += 1
        STATE["prefill_s"] += prefill_s


# ----------------------------------------------------------------------------- 5. tunnel
banner(5, "Tunnel", "public cloudflared URL")
URL = None
tunnel_proc = None
if CFG["tunnel"] and CLOUDFLARED.exists():
    tunnel_proc = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    lines = []

    def _pump_tunnel():
        for line in tunnel_proc.stdout:
            lines.append(line.rstrip())
            log(f"[cloudflared] {line.rstrip()[:200]}")
    threading.Thread(target=_pump_tunnel, daemon=True).start()
    deadline = time.time() + 120
    while time.time() < deadline and URL is None:
        for ln in lines:
            m = pat.search(ln)
            if m:
                URL = m.group(0).rstrip("/")
                break
        time.sleep(1)
    if URL:
        log(f"   endpoint: {URL}/v1")
        publish("tunnel-url", endpoint=f"{URL}/v1")
    else:
        log("   WARNING: could not get a tunnel URL in 120s")
        publish("tunnel-failed", note="cloudflared did not produce a URL in time")
elif CFG["tunnel"]:
    log("   WARNING: cloudflared not available; no public tunnel")
    publish("tunnel-failed", note="cloudflared binary not found")

# ----------------------------------------------------------------------------- 6. ready + keepalive
banner(6, "Ready", "self-test, then keep serving")

try:
    t0 = time.time()
    prompt_ids, _ = build_prompt([{"role": "user", "content": "Say hello in one word."}])
    out_ids, pf, dec, reused, reason = generate(prompt_ids, 16, 0.0, 1.0)
    test_text = tok.decode(out_ids[:16])
    t_total = time.time() - t0
    log(f"   self-test: {test_text.strip()[:60]!r} ({t_total:.1f}s, {len(out_ids)} tokens)")
    publish("warmed", minutes=round((time.time() - T0) / 60, 1), hbm_gb=round(hbm()[0], 2))
except Exception as e:
    log(f"   self-test failed: {e}")
    publish("warmed", minutes=round((time.time() - T0) / 60, 1), hbm_gb=round(hbm()[0], 2), note=f"self-test error: {str(e)[:100]}")

httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
log(f"   HTTP server listening on 0.0.0.0:{PORT}")

endpoint = f"{URL}/v1" if URL else None
log("")
log("#" * 70)
log(f"#  READY — GLM-5.3-Flash is live ({elapsed()} after start)")
log(f"#  ENDPOINT : {endpoint or 'http://127.0.0.1:' + str(PORT) + '/v1 (no tunnel)'}")
log(f"#  API KEY  : {API_KEY}")
log(f"#  MODEL    : {MODEL}   (context {CFG['max_len']}, {MAX_STREAMS} streams)")
log("#" * 70)
log("#  Try it:")
log(f"#    curl {endpoint or 'http://127.0.0.1:' + str(PORT) + '/v1'}/chat/completions \\")
log(f"#      -H 'Authorization: Bearer {API_KEY}' -H 'Content-Type: application/json' \\")
log(f"#      -d '{{\"model\": \"{MODEL}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")
log(f"#  Serving for up to {CFG['keepalive_min']} min, then this kernel exits on its own.")
log("#" * 70)
publish("ready", endpoint=endpoint, api_key=API_KEY, model=MODEL,
        max_model_len=CFG["max_len"], keepalive_min=CFG["keepalive_min"],
        startup_secs=int(time.time() - T0))

def _serve():
    httpd.serve_forever()

srv_thread = threading.Thread(target=_serve, daemon=True)
srv_thread.start()

t_serve = time.time()
while time.time() - t_serve < CFG["keepalive_min"] * 60:
    time.sleep(120)
    up = int((time.time() - t_serve) / 60)
    if up % 10 < 2:
        publish("heartbeat", up_min=up, endpoint=endpoint)
        log(f"   still serving ({up} min) — {endpoint or 'local only'}")

publish("auto-shutdown", served_min=CFG["keepalive_min"])
log("   keepalive window ended; shutting down.")
httpd.shutdown()
if tunnel_proc and tunnel_proc.poll() is None:
    tunnel_proc.terminate()
    try:
        tunnel_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        tunnel_proc.kill()
sys.exit(0)
