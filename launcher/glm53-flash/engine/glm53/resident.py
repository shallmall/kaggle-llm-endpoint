"""HBM-resident codebook-quantized experts (Unsloth UD-IQ*/Q2_K_XL GGUF) for the fully-jitted `Engine`.

Storage per sparse layer (chip-major, `P(AXIS)` on axis 0 hands each chip its slice; TP over moe_inter): every table
(gate_q, up_q, down_q) is a dict of PLANAR arrays `{plane: u32 [n_dev, E * rows_p, R]}` (glm53.planes): the GGUF block
fields split into dense 32-bit planes with the matrix row on the lane axis, so a Pallas kernel can dequantize an
expert in VMEM without byte slicing and XLA needs no relayout copies. gate/up: R = ml rows (this chip's slice of
moe_inter), inputs D; down: R = D rows, inputs ml.

Tables live in `params["layers"][i]["mlp"]`; `M.moe_dense` calls `ResidentFetch.apply`:
  decode (N*k <= gather_max_rows): `glm53.pallas_moe.moe_matvec` per matrix (fused dequant + VPU matvec, the expert
  planes DMA'd by index), or the XLA path (dynamic slices + glm53.planes.dequant_planes + einsum) when use_pallas=False;
  prefill: sweep over all experts in chunks (XLA dequant + dense masked matmuls) so no per-token weight copies exist.
Activations are permuted to the planes' "pm" input order (planes.pm_x / pm_flat); outputs are in natural row order.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from glm53 import iqquant as Q
from glm53 import model as M
from glm53 import planes as PL
from glm53 import pallas_moe as K

QK = Q.QK_K
KEYS = ("gate_q", "up_q", "down_q")


def pack_layer_from_gguf(gm, layer: int, n_dev: int, threads: int = 8):
    """Read one layer's routed experts from a `gguf_reader.GGUFModel` -> (tables dict, qtypes dict)."""
    out, qtypes = {}, {}
    for key, tname in (("gate_q", "gate"), ("up_q", "up"), ("down_q", "down")):
        name = f"blk.{layer}.ffn_{tname}_exps.weight"
        info = gm.info(name)
        ne0, ne1, E = info["dims"]                       # ne0 = input dim (blocks along it), ne1 = rows, E experts
        bb = Q.BLOCK_BYTES[info["type"]]
        raws = gm.read_experts(name, range(E), threads=threads)
        t = np.stack([np.frombuffer(r, np.uint8).reshape(ne1, ne0 // QK, bb) for r in raws])   # [E, rows, nblk, bb]
        if key == "down_q":                              # split blocks (the mi input dim) across chips
            t = t.reshape(E, ne1, n_dev, ne0 // QK // n_dev, bb).transpose(2, 0, 1, 3, 4)
        else:                                            # split rows (the mi output dim) across chips
            t = t.reshape(E, n_dev, ne1 // n_dev, ne0 // QK, bb).transpose(1, 0, 2, 3, 4)
        out[key] = pack_chip_planes(t, info["type"])
        qtypes[key] = info["type"]
    return out, qtypes


def pack_chip_planes(t, qtype, threads=None):
    """t uint8 [n_dev, E, rows, nblk, bb] -> {plane: u32 [n_dev, E*rows_p, rows]} (glm53.planes layout per chip).
    The chips' slices are packed in parallel threads (NumPy releases the GIL on the big slices; the packing, not the
    GGUF read, dominated the 9-minute build: the dataset mount serves ~650 MB/s to 8-16 readers, 2026-09-14)."""
    n = t.shape[0]
    threads = n if threads is None else max(1, int(threads))
    if threads > 1 and n > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(threads, n)) as ex:
            per_chip = list(ex.map(lambda c: PL.pack_planes(t[c], qtype), range(n)))
    else:
        per_chip = [PL.pack_planes(t[c], qtype) for c in range(n)]
    return {k: np.stack([pc[k] for pc in per_chip]) for k in per_chip[0]}


def hbm_bytes(tables: dict) -> int:
    """Bytes per chip of a layer's tables ({key: {plane: [n_dev, ...]}})."""
    return sum(int(a.nbytes) // int(a.shape[0]) for planes in tables.values() for a in planes.values())


class ResidentFetch:
    """Stored on the Engine; `M.moe_dense` calls `.apply(p, x, idx, w, layer, limit)` when present."""

    def __init__(self, qtypes: dict, D: int, ml: int, out_dtype=jnp.bfloat16, gather_max_rows: int = 64, chunk: int = 8,
                 use_pallas: bool = True, interpret: bool = False, sweep_mode: str = "ragged", tm: int = 32,
                 combine: str = "matmul"):
        self.qtypes = qtypes            # {layer: {"gate_q": "IQ2_S", "up_q": "IQ2_S", "down_q": "IQ3_S"}}
        self.rows = {"gate_q": ml, "up_q": ml, "down_q": D}     # logical rows per expert (this chip)
        self.nblk = {"gate_q": D // QK, "up_q": D // QK, "down_q": ml // QK}
        self.out_dtype = out_dtype
        self.gather_max_rows = gather_max_rows
        self.chunk = chunk
        self.use_pallas = use_pallas    # decode path: Pallas fused kernel (True) or XLA planes dequant + einsum
        self.interpret = interpret      # run the Pallas kernel in interpret mode (CPU tests)
        self.sweep_mode = sweep_mode    # prefill: "dense" = masked sweep over active experts, "ragged" = grouped GEMM
        self.tm = tm                    # ragged: rows per block (multiple of 16 for bf16)
        self.combine = combine          # ragged: per-token combine of slot outputs, "gather" (row gather) or "matmul"

    def _deq(self, layer, key, tbl, sel):
        """Dequantize the experts picked by `sel` from planes `tbl` -> [n, C*W, R] (pm order along inputs) in out_dtype."""
        qt = self.qtypes[layer][key]
        rows_p = PL.plane_rows(qt, self.nblk[key])
        picked = {k: sel(a, rows_p[k]).reshape(-1, rows_p[k], a.shape[-1]) for k, a in tbl.items()}
        return PL.dequant_planes(picked, qt, self.nblk[key]).astype(self.out_dtype)

    @staticmethod
    def _slice_experts(a, rows_p, flat):
        """a u32 [1, E*rows_p, R]; flat int32 [n] -> [n*rows_p, R] via n dynamic slices (DMA, never a generic gather
        and never a copy of the whole table)."""
        parts = [lax.dynamic_slice_in_dim(a, flat[j] * rows_p, rows_p, axis=1) for j in range(flat.shape[0])]
        return jnp.concatenate(parts, axis=1)[0]

    def apply(self, p, x, idx, w, layer, limit):
        """x [N,D]; idx [N,k] int32 into E; w [N,k] fp32 -> partial MoE output [N,D] (TP-reduced by the caller)."""
        gate_q, up_q, down_q = p["gate_q"], p["up_q"], p["down_q"]         # {plane: [1, E*rows_p, R]} on this chip
        N, k = idx.shape
        if N * k <= self.gather_max_rows:
            if self.use_pallas:
                return self._apply_kernel(x, idx, w, layer, gate_q, up_q, down_q, limit)
            return self._apply_gather(x, idx, w, layer, gate_q, up_q, down_q, limit)
        return self._apply_sweep(x, idx, w, layer, gate_q, up_q, down_q, limit)

    def _apply_kernel(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        N, k = idx.shape
        flat = idx.reshape(-1)
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        Xg = jnp.repeat(PL.pm_x(qt_gu, x.astype(jnp.float32)), k, axis=0)              # [Nk, W, C]
        mv = lambda planes, qt, key, X: K.moe_matvec(planes, qt, self.nblk[key], flat, X, interpret=self.interpret)
        g = mv(gate_q, qt_gu, "gate_q", Xg)                                             # [Nk, ml] f32
        u = mv(up_q, self.qtypes[layer]["up_q"], "up_q", Xg)
        h = M.swiglu_clamped(g, u, limit)                                               # [Nk, ml]
        y = mv(down_q, qt_dn, "down_q", PL.pm_x(qt_dn, h)).reshape(N, k, -1)            # [N, k, D]
        return jnp.einsum("nkd,nk->nd", y, w)

    def _apply_gather(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        """XLA decode path on the planar tables (no Pallas): per-expert dynamic slices + dequant + einsum."""
        N, k = idx.shape
        flat = idx.reshape(-1)
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        sel = lambda a, rows_p: self._slice_experts(a, rows_p, flat)
        g = self._deq(layer, "gate_q", gate_q, sel)                                     # [Nk, D_pm, ml]
        u = self._deq(layer, "up_q", up_q, sel)
        d = self._deq(layer, "down_q", down_q, sel)                                     # [Nk, ml_pm, D]
        xs = jnp.repeat(PL.pm_flat(qt_gu, x).astype(self.out_dtype), k, axis=0)         # [Nk, D_pm]
        h = M.swiglu_clamped(jnp.einsum("nd,ndm->nm", xs, g), jnp.einsum("nd,ndm->nm", xs, u), limit)   # [Nk, ml]
        h = PL.pm_flat(qt_dn, h).astype(d.dtype)
        y = jnp.einsum("nm,nmd->nd", h, d).astype(jnp.float32).reshape(N, k, -1)
        return jnp.einsum("nkd,nk->nd", y, w)

    def _apply_sweep(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        if self.use_pallas and self.sweep_mode == "ragged":
            return self._apply_ragged_kernel(x, idx, w, layer, gate_q, up_q, down_q, limit)
        if self.use_pallas:
            return self._apply_sweep_kernel(x, idx, w, layer, gate_q, up_q, down_q, limit)
        return self._apply_sweep_xla(x, idx, w, layer, gate_q, up_q, down_q, limit)

    def _apply_ragged_kernel(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        """Prefill through the grouped GEMM kernels: (token, expert) slots sorted by expert into `tm`-row blocks,
        each expert dequantized once and multiplied against its own rows only; the per-token combine gathers each
        token's k slot rows (or, `combine="matmul"`, multiplies by a [T, Rp] weight matrix)."""
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        E = gate_q["qs"].shape[1] // PL.plane_rows(qt_gu, self.nblk["gate_q"])["qs"]
        T, k = idx.shape
        plan = K.ragged_plan(idx, w, E, self.tm)
        tok = plan["tok_row"]
        xk = K.pm_mxu(qt_gu, x.astype(jnp.float32)).astype(self.out_dtype)                     # [T, D]
        xs = jnp.take(xk, jnp.maximum(tok, 0), axis=0, mode="clip")
        xs = jnp.where((tok >= 0)[:, None], xs, jnp.zeros((), xs.dtype))                       # [Rp, D]
        h = K.moe_ragged_gateup(gate_q, up_q, qt_gu, self.nblk["gate_q"], plan["blk_expert"], plan["n_blocks"], xs,
                                limit, tm=self.tm, interpret=self.interpret)                     # [Rp, ml]
        hk = K.pm_mxu(qt_dn, h.astype(jnp.float32)).astype(self.out_dtype)
        y = K.moe_ragged_down(down_q, qt_dn, self.nblk["down_q"], plan["blk_expert"], plan["n_blocks"], hk,
                              tm=self.tm, interpret=self.interpret)                              # [Rp, D]
        if self.combine == "matmul":
            wc = (tok[None, :] == jnp.arange(T, dtype=jnp.int32)[:, None]).astype(jnp.float32) * plan["w_row"][None, :]
            return jnp.dot(wc.astype(y.dtype), y, preferred_element_type=jnp.float32)
        ys = jnp.take(y, plan["row_slot"].reshape(-1), axis=0, mode="clip").reshape(T, k, -1).astype(jnp.float32)
        return jnp.einsum("tkd,tk->td", ys, w.astype(jnp.float32))

    SWEEP_TOKENS = 512     # token chunk per kernel launch (bounds the [E, T, ml] intermediate to ~75 MB per chip)

    def _apply_sweep_kernel(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        """Prefill through the Pallas sweep kernels: every routed expert is dequantized once into VMEM tiles and MXU-
        multiplied against all tokens; experts no token routes to are skipped (scalar-prefetched active slots)."""
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        E = gate_q["qs"].shape[1] // PL.plane_rows(qt_gu, self.nblk["gate_q"])["qs"]
        N = x.shape[0]
        ids, n_act = K.active_slots(idx, E)
        rw = jnp.einsum("nk,nke->ne", w.astype(jnp.float32),
                        (idx[:, :, None] == ids[None, None, :]).astype(jnp.float32))          # [N, E] per slot
        outs = []
        for t0 in range(0, N, self.SWEEP_TOKENS):
            xs = x[t0:t0 + self.SWEEP_TOKENS]
            xk = K.pm_mxu(qt_gu, xs.astype(jnp.float32)).astype(self.out_dtype)                # [T, D]
            h = K.moe_sweep_gateup(gate_q, up_q, qt_gu, self.nblk["gate_q"], ids, n_act, xk, limit,
                                   interpret=self.interpret)                                    # [E, T, ml] bf16
            hw = h.astype(jnp.float32) * jnp.swapaxes(rw[t0:t0 + self.SWEEP_TOKENS], 0, 1)[:, :, None]
            hk = K.pm_mxu(qt_dn, hw).astype(self.out_dtype)                                     # [E, T, ml]
            outs.append(K.moe_sweep_down(down_q, qt_dn, self.nblk["down_q"], ids, n_act, hk, interpret=self.interpret))
        return outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=0)

    def _apply_sweep_xla(self, x, idx, w, layer, gate_q, up_q, down_q, limit):
        E = gate_q["qs"].shape[1] // PL.plane_rows(self.qtypes[layer]["gate_q"], self.nblk["gate_q"])["qs"]
        CH = self.chunk
        assert E % CH == 0, (E, CH)
        qt_gu, qt_dn = self.qtypes[layer]["gate_q"], self.qtypes[layer]["down_q"]
        onehot = jax.nn.one_hot(idx, E, dtype=jnp.float32)                  # [N,k,E]
        rw = jnp.einsum("nk,nke->ne", w.astype(jnp.float32), onehot)        # [N,E] routing weight (0 if unused)
        xg = PL.pm_flat(qt_gu, x).astype(self.out_dtype)                    # [N, D_pm]

        def body(acc, c):
            c0 = c * CH
            sel = lambda a, rows_p: lax.dynamic_slice_in_dim(a, c0 * rows_p, CH * rows_p, axis=1)[0]
            g = self._deq(layer, "gate_q", gate_q, sel)                                          # [CH, D_pm, ml]
            u = self._deq(layer, "up_q", up_q, sel)
            d = self._deq(layer, "down_q", down_q, sel)                                          # [CH, ml_pm, D]
            h = M.swiglu_clamped(jnp.einsum("nd,edm->nem", xg, g), jnp.einsum("nd,edm->nem", xg, u), limit)  # [N,CH,ml]
            rws = lax.dynamic_slice_in_dim(rw, c0, CH, 1)                                        # [N, CH]
            h = PL.pm_flat(qt_dn, h.astype(jnp.float32) * rws[..., None]).astype(d.dtype)
            return acc + jnp.einsum("nem,emd->nd", h, d).astype(jnp.float32), None

        acc, _ = lax.scan(body, jnp.zeros((xg.shape[0], x.shape[1]), jnp.float32), jnp.arange(E // CH))
        return acc


# ------------------------------------------------------------------------------------- per-layer-kind engine
from glm53.engine import Engine, AXIS, R, shard_map, cache_specs, device_sample, DeviceSampler   # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P           # noqa: E402


class _HostShards:
    """A sharded device array parked in host memory as its per-device shards (a pytree leaf)."""

    def __init__(self, shards, devices, shape, dtype, sharding):
        self.shards, self.devices, self.shape, self.dtype, self.sharding = shards, devices, shape, dtype, sharding
        self.nbytes = sum(int(x.nbytes) for x in shards)

    @classmethod
    def of(cls, a):
        sh = sorted(a.addressable_shards, key=lambda s: s.device.id)
        return cls([np.asarray(s.data) for s in sh], [s.device for s in sh], a.shape, a.dtype, a.sharding)

    def to_device(self):
        parts = [jax.device_put(x, d) for x, d in zip(self.shards, self.devices)]
        return jax.make_array_from_single_device_arrays(self.shape, self.sharding, parts)


class ResidentLayerEngine(Engine):
    """Resident experts with ONE jitted program per layer *kind* (attention type, MLP type, expert quant types,
    indexer presence) instead of one 45-layer program: the layer's params are program arguments, so ~6 programs cover
    all layers and the compile stays small (a single unrolled program with 42 codebook dequants got the XLA compiler
    SIGKILLed on the Kaggle host). Per step: embed program, 45 layer programs, head program — all device-resident,
    no host gathers.

    Prefill is CHUNKED: the caches are allocated up front (`alloc_caches`) and the prompt is fed in pieces of
    `prefill_piece` tokens (the last one padded to a bucket) through cache-carrying programs, so no temporary scales
    with the prompt length; `prefill(tokens, caches, pos0)` continues an existing context (prefix caching). Cache
    buffers are DONATED to the layer programs (in-place update; never reuse a cache object after passing it in —
    `copy_caches` for snapshots)."""

    def __init__(self, cfg, params, expert_fetch, devices=None, max_len: int = 4096, layers_per_program: int = 1,
                 layers_per_program_prefill: int = 1, int8_nonexpert: bool = False, seq_shard: bool = False,
                 q_block: int = 128, prefill_piece: int = 2048, cache_q8: bool = False):
        super().__init__(cfg, params, None, devices, max_len, expert_fetch=expert_fetch, int8_nonexpert=int8_nonexpert,
                         seq_shard=seq_shard, q_block=q_block, cache_q8=cache_q8)
        self._progs = {}
        self._allocs = {}
        self.launches = 0
        assert prefill_piece in self.PREFILL_BUCKETS, prefill_piece
        self.prefill_piece = min(prefill_piece, max_len)
        # consecutive layer groups; a group's program is keyed by the tuple of its layer kinds (dispatch ≈ 2 ms per
        # launch on the Kaggle TPU VM, so 45 single-layer launches cost ~100 ms/token). Prefill keeps 1 layer per
        # program by default: a 4-layer prefill program (expert sweep scans inside) did not finish compiling in 50 min.
        self.lpp = {"decode": layers_per_program, "prefill": layers_per_program_prefill}
        self.groups = {m: [list(range(a, min(a + k, cfg.n_layers))) for a in range(0, cfg.n_layers, k)]
                       for m, k in self.lpp.items()}

    def _kind(self, i):
        qt = tuple(sorted(self.expert_fetch.qtypes.get(i, {}).items())) if self.cfg.mlp_types[i] == "sparse" else ()
        return (self.cfg.layer_types[i], self.cfg.mlp_types[i], qt, "indexer" in self.params["layers"][i]["attn"])

    def _prog_embed(self, B, T, override=False):
        """Token embeddings broadcast to the hc residual streams [B,T,hc,D]; with override=True the extra arguments
        (vectors [B,T,D], mask [B,T] bool) replace the table rows where mask is set (image tokens: glm53.vision)."""
        key = ("embed", B, T, override)
        if key not in self._progs:
            def prog(embed, tokens, *ov):
                x = self._embed(embed, tokens).astype(self.lcfg.dtype)
                if override:
                    vec, mask = ov
                    x = jnp.where(mask[..., None], vec.astype(x.dtype), x)
                return jnp.broadcast_to(x[:, :, None, :], (B, T, self.cfg.hc, x.shape[-1]))
            in_specs = (self.specs["embed"], R) + ((R, R) if override else ())
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=R, check_vma=False))
        return self._progs[key]

    def _prog_head(self, B, T, sample=False):
        """Logits [B,V] of the last valid position; with sample=True also the device-sampled token ids [B]
        (extra args: temperature, top_p, key — `DeviceSampler.bind(mesh)`), returned as (ids, logits, next key)."""
        key = ("head", B, T, sample)
        if key not in self._progs:
            def prog(norm, lm_head, streams, length, *samp):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
                last = lax.dynamic_index_in_dim(h, length - 1, axis=1, keepdims=False)
                zl = self._logits_local(lm_head, last)
                logits = lax.all_gather(zl, AXIS, axis=-1, tiled=True)
                if not sample:
                    return logits
                temp, top_p, k = samp
                k1, k2 = jax.random.split(k)
                return device_sample(zl, temp, top_p, k1), logits, k2
            in_specs = (R, self.specs["lm_head"], R, R) + ((R, R, R) if sample else ())
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=in_specs,
                                                 out_specs=(R, R, R) if sample else R, check_vma=False))
        return self._progs[key]

    def _hist_spec(self, i):
        """Sharding of the per-token histories a speculative verify step returns for layer i (None: no history)."""
        if self.cfg.layer_types[i] == "linear_attention":
            return {"state_hist": P(None, None, AXIS), "conv_hist": P(None, None, None, AXIS)}
        if "indexer" in self.params["layers"][i]["attn"]:
            return {"tk_hist": R, "tg_hist": R}
        return None

    def _prog_group(self, layers, B, T, has_cache, use_recurrent, hist=False):
        kinds = tuple(self._kind(i) for i in layers)
        key = ("group", kinds, B, T, has_cache, use_recurrent, hist)
        if key not in self._progs:
            lts = [self.cfg.layer_types[i] for i in layers]
            mts = [self.cfg.mlp_types[i] for i in layers]
            css = [self._cache_spec(i) for i in layers]
            hss = [self._hist_spec(i) for i in layers]
            rep = list(layers)                     # representative layer ids (only used to look up expert qtypes)

            from glm53 import quant8 as Q8

            def prog(ps, streams, pos0, length, caches):
                new, hs = [], []
                for j, p in enumerate(ps):
                    p = Q8.dequant_tree(p, self.lcfg.dtype)          # int8 non-expert matrices -> bf16 (fused into dots)
                    fetch = self.expert_fetch if mts[j] == "sparse" else None
                    streams, nc = M.decoder_layer(p, streams, self.lcfg, lts[j], mts[j],
                                                  caches[j] if has_cache else None, pos0, use_recurrent, fetch, "auto",
                                                  layer=rep[j], cap=self.max_len, length=None if use_recurrent else length,
                                                  hist=hist)
                    nc, h = M.split_hist(nc)
                    new.append(nc); hs.append(h if hist else None)
                return (streams, new, hs) if hist else (streams, new)
            in_specs = ([self.specs["layers"][i] for i in layers], R, R, R, css if has_cache else None)
            out_specs = (R, css, hss) if hist else (R, css)
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
            self._progs[key] = jax.jit(sm, donate_argnums=(4,) if has_cache else ())
        return self._progs[key]

    def _prog_hidden(self, B, T):
        """Final-norm hidden states of every position [B,T,D] (MTP prefill input)."""
        key = ("hidden", B, T)
        if key not in self._progs:
            def prog(norm, streams):
                return M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=(R, R), out_specs=R, check_vma=False))
        return self._progs[key]

    def _prog_head_all(self, B, T, sample=False):
        """Logits [B,T,V] and hidden [B,T,D] of every position (speculative verify; T small); with sample=True
        (extra args as in `_prog_head`) also the device-sampled ids [B,T]: (ids, logits, hidden, next key)."""
        key = ("head_all", B, T, sample)
        if key not in self._progs:
            def prog(norm, lm_head, streams, *samp):
                h = M.rmsnorm(streams.mean(2).astype(self.lcfg.dtype), norm, self.cfg.eps)
                zl = self._logits_local(lm_head, h)
                logits = lax.all_gather(zl, AXIS, axis=-1, tiled=True)
                if not sample:
                    return logits, h
                temp, top_p, k = samp
                k1, k2 = jax.random.split(k)
                ids = device_sample(zl.reshape(B * T, -1), temp, top_p, k1)
                return ids.reshape(B, T), logits, h, k2
            in_specs = (R, self.specs["lm_head"], R) + ((R, R, R) if sample else ())
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=in_specs,
                                                 out_specs=(R, R, R, R) if sample else (R, R), check_vma=False))
        return self._progs[key]

    def _prog_rollback(self, B, T):
        """per-token histories of a T-token verify + n_keep -> the small state entries (KDA state/conv, pool tails)
        as after the first n_keep tokens; spliced into the caches by the caller (the big positional arrays are
        not touched)."""
        key = ("rollback", B, T)
        if key not in self._progs:
            hss = [self._hist_spec(i) for i in range(self.cfg.n_layers)]
            oss = [None if h is None else {M.HIST_KEYS[k]: P(*v[1:]) for k, v in h.items()} for h in hss]   # drop the T axis
            dts = [{M.HIST_KEYS[k]: self.cfg.dtype if M.HIST_KEYS[k] != "state" else jnp.float32 for k in h} if h else None
                   for h in hss]
            sm = shard_map(lambda h, n: M.rollback_states(h, n, dts), mesh=self.mesh, in_specs=(hss, R), out_specs=oss,
                           check_vma=False)
            self._progs[key] = jax.jit(sm)
        return self._progs[key]

    mtp = None                       # MTP (NextN) layer params on device, or None (see set_mtp)

    def _cache_spec(self, i):
        if i == self.cfg.n_layers:                                             # the MTP layer's cache entry
            mla = P(None, AXIS) if self.lcfg.seq_shard > 1 else R
            spec = {"c": mla, "pk": mla, "tk": R, "tg": R, "h": R}
            if self.cfg.cache_q8:
                spec["cs"] = mla
            return spec
        return super()._cache_spec(i)

    def _is_kda(self, i):
        return i < self.cfg.n_layers and self.cfg.layer_types[i] == "linear_attention"

    def alloc_caches(self, B):
        """Zero caches for a new context at the engine's capacity, allocated directly on the chips (+ the MTP
        layer's entry, index n_layers, when an MTP layer is installed: its attention cache and the carried final
        hidden state "h" [B,1,D] of the last processed position)."""
        cfg = self.cfg
        out = []
        n_entries = cfg.n_layers + (1 if self.mtp is not None else 0)
        for i in range(n_entries):
            spec = self._cache_spec(i)
            if i == cfg.n_layers:
                shapes = {k: (sh, M.cache_dtype(cfg, k, cfg.dtype)) for k, sh in M.mla_cache_shapes(cfg, B, self.max_len, True).items()}
                shapes["h"] = ((B, 1, cfg.hidden), cfg.dtype)
            elif cfg.layer_types[i] == "linear_attention":
                shapes = {"conv": ((B, cfg.conv_k - 1, 3 * cfg.kda_heads * cfg.kda_hd), cfg.dtype),
                          "state": ((B, cfg.kda_heads, cfg.kda_hd, cfg.kda_hd), jnp.float32)}
            else:
                has_ix = "indexer" in self.params["layers"][i]["attn"]
                shapes = {k: (sh, M.cache_dtype(cfg, k, cfg.dtype)) for k, sh in M.mla_cache_shapes(cfg, B, self.max_len, has_ix).items()}
            c = {}
            for k, (shape, dtype) in shapes.items():
                key = (shape, jnp.dtype(dtype), spec[k])
                if key not in self._allocs:
                    self._allocs[key] = jax.jit(lambda shape=shape, dtype=dtype: jnp.zeros(shape, dtype),
                                                out_shardings=NamedSharding(self.mesh, spec[k]))
                c[k] = self._allocs[key]()
            out.append(c)
        return out

    @staticmethod
    def copy_caches(caches):
        """Device-side copy (the programs update caches in place)."""
        return jax.tree.map(jnp.copy, caches)

    def _prefix_rows(self, n_tokens):
        """Local cache rows that positions < n_tokens can occupy (interleaved layout: 4-token pools round-robin)."""
        kp, n = self.cfg.idx_kpool, self.lcfg.seq_shard
        return -(-n_tokens // (kp * n)) * kp if n > 1 else n_tokens

    def snapshot_prefix(self, caches, n_tokens, rows_bucket=1):
        """Compact snapshot of a context of n_tokens (the KDA states and only the used MLA/indexer cache rows), e.g.
        a shared system prompt: ~2 KB/token/chip instead of a full cache set. Restore with `restore_prefix`.
        `rows_bucket` rounds the row count up (bounds the number of slice/restore programs compiled for arbitrary
        context lengths; the extra rows are never read back)."""
        rows = self._prefix_rows(n_tokens)
        cap = self.max_len // max(self.lcfg.seq_shard, 1)
        rows = min(-(-rows // rows_bucket) * rows_bucket, cap)
        key = ("snap", rows)
        if key not in self._progs:
            self._progs[key] = {}
        out = []
        for i, c in enumerate(caches):
            if self._is_kda(i):
                out.append(jax.tree.map(jnp.copy, c)); continue
            spec = self._cache_spec(i)
            snap = {}
            for k, a in c.items():
                if k not in M.ROW_KEYS:                                       # tail buffers: whole copies
                    snap[k] = jnp.copy(a); continue
                rk = rows // self.cfg.idx_kpool if k == "pk" else rows
                pk = (k, a.shape, str(a.dtype))
                if pk not in self._progs[key]:
                    sp = spec[k]
                    self._progs[key][pk] = jax.jit(shard_map(lambda x, rk=rk: x[:, :rk], mesh=self.mesh, in_specs=(sp,),
                                                             out_specs=sp, check_vma=False))
                snap[k] = self._progs[key][pk](a)
            out.append(snap)
        return {"n_tokens": n_tokens, "rows": rows, "caches": out}

    def restore_prefix(self, snap, B=1):
        """Fresh caches at full capacity holding the snapshot's context (the snapshot stays valid)."""
        caches = self.alloc_caches(B)
        rows = snap.get("rows") or self._prefix_rows(snap["n_tokens"])
        key = ("restore", rows)
        if key not in self._progs:
            self._progs[key] = {}
        out = []
        for i, c in enumerate(caches):
            sc = snap["caches"][i]
            if self._is_kda(i):
                out.append(jax.tree.map(jnp.copy, sc)); continue
            spec = self._cache_spec(i)
            new = {}
            for k, a in c.items():
                if k not in M.ROW_KEYS:
                    new[k] = jnp.copy(sc[k]); continue
                pk = (k, a.shape, str(a.dtype))
                if pk not in self._progs[key]:
                    sp = spec[k]
                    self._progs[key][pk] = jax.jit(shard_map(lambda x, p: lax.dynamic_update_slice_in_dim(x, p, 0, axis=1),
                                                             mesh=self.mesh, in_specs=(sp, sp), out_specs=sp, check_vma=False),
                                                   donate_argnums=(0,))
                new[k] = self._progs[key][pk](a, sc[k])
            out.append(new)
        return out

    @staticmethod
    def snapshot_to_host(snap):
        """Move a compact snapshot into host memory so it holds no HBM; a parked context costs ~17 KB/token on the
        host. Every array is kept as its per-device shards (no host-side assembly of the 8 shards, which ran at
        ~1.5 GB/s); `snapshot_from_host` brings it back for `restore_prefix`."""
        host = jax.tree.map(_HostShards.of, snap["caches"])
        n_bytes = sum(h.nbytes for h in jax.tree.leaves(host, is_leaf=lambda x: isinstance(x, _HostShards)))
        return {"n_tokens": snap["n_tokens"], "rows": snap["rows"], "caches": host, "bytes": n_bytes}

    @staticmethod
    def snapshot_from_host(hsnap):
        """Device snapshot (as `snapshot_prefix` returns it) from a host snapshot; the host copy stays valid."""
        caches = jax.tree.map(lambda h: h.to_device(), hsnap["caches"], is_leaf=lambda x: isinstance(x, _HostShards))
        return {"n_tokens": hsnap["n_tokens"], "rows": hsnap["rows"], "caches": caches}

    def _run_layers(self, tokens, caches, pos0, use_recurrent, length, override=None):
        """embed + all layer groups -> (final residual streams, new caches). `override` = (vectors [B,T,D],
        mask [B,T]) replaces the token embeddings where mask is set."""
        B, T = tokens.shape
        if override is None:
            streams = self._prog_embed(B, T)(self.params["embed"], jnp.asarray(tokens))
        else:
            streams = self._prog_embed(B, T, True)(self.params["embed"], jnp.asarray(tokens), jnp.asarray(override[0]),
                                                   jnp.asarray(override[1]))
        new_caches = []
        has_cache = caches is not None
        pos0, length = jnp.int32(pos0), jnp.int32(length)
        for g in self.groups["decode" if use_recurrent else "prefill"]:
            prog = self._prog_group(g, B, T, has_cache, use_recurrent)
            streams, ncs = prog([self.params["layers"][i] for i in g], streams, pos0, length,
                                [caches[i] for i in g] if has_cache else None)
            new_caches.extend(ncs)
            self.launches += 1
        return streams, new_caches

    def _run(self, tokens, caches, pos0, use_recurrent, length=None, want_logits=True, want_streams=False, override=None):
        B, T = tokens.shape
        length = T if length is None else length
        streams, new_caches = self._run_layers(tokens, caches, pos0, use_recurrent, length, override)
        logits = None
        if want_logits:
            logits = self._prog_head(B, T)(self.params["norm"], self.params["lm_head"], streams, jnp.int32(length))
        return (logits, new_caches, streams) if want_streams else (logits, new_caches)

    def decode_sample(self, token, caches, pos, sampler: DeviceSampler):
        """`decode` with the next token sampled on the device: returns (ids [B] int32 device array, logits [B,V],
        caches, pos + 1). Only the ids need to reach the host (one small transfer instead of the [B,V] logits)."""
        B = token.shape[0]
        n = self.cfg.n_layers
        extra = caches[n:] if len(caches) > n else []
        streams, new = self._run_layers(np.asarray(token).reshape(B, 1), caches[:n], pos, True, 1)
        if getattr(self, "_one", None) is None:          # a replicated constant (no per-step scalar transfer); lazy so
            self._one = jax.device_put(jnp.int32(1), NamedSharding(self.mesh, P()))   # live engines survive a reload
        ids, logits, nk = self._prog_head(B, 1, True)(self.params["norm"], self.params["lm_head"], streams, self._one,
                                                      *sampler.bind(self.mesh))
        sampler.advance(nk)
        return ids, logits, new + extra, pos + 1

    # ---- batched decode of independent streams (continuous batching): one cache set per stream, per-row positions
    def _prog_embed_rows(self, B):
        """(embed, tokens [B] int32, pos [B] int32) -> (streams [B,1,hc,D], pos + 1): the positions stay on the
        device (no per-step host transfer), the program advances them."""
        key = ("embed_rows", B)
        if key not in self._progs:
            def prog(embed, tokens, pos):
                x = self._embed(embed, tokens[:, None]).astype(self.lcfg.dtype)
                return jnp.broadcast_to(x[:, :, None, :], (B, 1, self.cfg.hc, x.shape[-1])), pos + 1
            self._progs[key] = jax.jit(shard_map(prog, mesh=self.mesh, in_specs=(self.specs["embed"], R, R),
                                                 out_specs=(R, R), check_vma=False))
        return self._progs[key]

    def _prog_group_rows(self, layers, B):
        """Decode step of `layers` for B independent streams: (params, streams [B,1,hc,D], pos [B], caches) with
        caches[j][b] = layer j's cache of stream b (batch dim 1, donated) -> (streams, new caches[j][b])."""
        kinds = tuple(self._kind(i) for i in layers)
        key = ("group_rows", kinds, B)
        if key not in self._progs:
            lts = [self.cfg.layer_types[i] for i in layers]
            mts = [self.cfg.mlp_types[i] for i in layers]
            css = [[self._cache_spec(i)] * B for i in layers]
            rep = list(layers)
            from glm53 import quant8 as Q8

            def prog(ps, streams, pos, caches):
                pos_b = [pos[b] for b in range(B)]
                new = []
                for j, p in enumerate(ps):
                    p = Q8.dequant_tree(p, self.lcfg.dtype)
                    fetch = self.expert_fetch if mts[j] == "sparse" else None
                    streams, ncs = M.decoder_layer_rows(p, streams, self.lcfg, lts[j], mts[j], caches[j], pos_b, fetch,
                                                        "auto", layer=rep[j], cap=self.max_len)
                    new.append(ncs)
                return streams, new
            in_specs = ([self.specs["layers"][i] for i in layers], R, R, css)
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=(R, css), check_vma=False)
            self._progs[key] = jax.jit(sm, donate_argnums=(3,))
        return self._progs[key]

    def device_positions(self, pos):
        """Host positions (ints) -> one replicated int32 [B] device array (the form `decode_rows` carries)."""
        return jax.device_put(np.asarray(pos, np.int32).reshape(-1), NamedSharding(self.mesh, P()))

    def decode_rows(self, tokens, cache_sets, pos, sampler: DeviceSampler | None = None):
        """One decode step of B independent streams in one batch: `tokens` [B] (their last tokens; a device array
        from the previous step or host ints), `cache_sets[b]` = stream b's own cache list (as `prefill` /
        `restore_prefix` return them, batch dim 1; consumed), `pos` [B] int32 replicated device array
        (`device_positions`) or host ints. Returns (ids [B] device int32 or None without a sampler, logits [B,V],
        new cache sets, pos + 1 on the device). Streams may sit at any positions; the MoE/mHC/projections run
        batched, the attention per row. Compiles one program set per B."""
        B = len(cache_sets)
        n = self.cfg.n_layers
        assert all(len(c) >= n for c in cache_sets), "every stream needs a full cache set"
        if not isinstance(pos, jax.Array):
            pos = self.device_positions(pos)
        if not (isinstance(tokens, jax.Array) and tokens.shape == (B,) and tokens.dtype == jnp.int32):
            tokens = jnp.asarray(np.asarray(tokens, np.int32).reshape(B))   # (a device array from the last step passes through)
        streams, pos_next = self._prog_embed_rows(B)(self.params["embed"], tokens, pos)
        new_sets = [[] for _ in range(B)]
        for g in self.groups["decode"]:
            prog = self._prog_group_rows(g, B)
            streams, ncs = prog([self.params["layers"][i] for i in g], streams, pos,
                                [[cache_sets[b][i] for b in range(B)] for i in g])
            for j in range(len(g)):
                for b in range(B):
                    new_sets[b].append(ncs[j][b])
            self.launches += 1
        for b in range(B):                                   # extra entries (an MTP cache) pass through untouched
            new_sets[b].extend(cache_sets[b][n:])
        if sampler is None:
            logits = self._prog_head(B, 1)(self.params["norm"], self.params["lm_head"], streams, jnp.int32(1))
            return None, logits, new_sets, pos_next
        if getattr(self, "_one", None) is None:
            self._one = jax.device_put(jnp.int32(1), NamedSharding(self.mesh, P()))
        ids, logits, nk = self._prog_head(B, 1, True)(self.params["norm"], self.params["lm_head"], streams, self._one,
                                                      *sampler.bind(self.mesh))
        sampler.advance(nk)
        return ids, logits, new_sets, pos_next

    def prefill(self, tokens, caches=None, pos0=0, embeds=None):
        """Prefill `tokens` [B,T] into a fresh context (caches=None) or continue one at position pos0 (the caches
        are consumed). Returns (logits of the last token, caches, pos0 + T). With an MTP layer installed the MTP
        cache (entry n_layers) is prefilled too: MTP position p takes (token p+1, final hidden p), so a piece
        covers positions [pos0-1, pos0+L-1) (or [0, L-1) for the first piece) and the last hidden is carried.
        `embeds` = (idx [M] int, vec [M, D]) replaces the embeddings of tokens[0, idx] (B = 1; image tokens,
        glm53.vision) — the MTP layer's own embedding input keeps the table rows (drafts only; verify is exact)."""
        tokens = np.asarray(tokens)
        B, T = tokens.shape
        assert T >= 1 and pos0 + T <= self.max_len, (pos0, T, self.max_len)
        if caches is None:
            caches = self.alloc_caches(B)
        if embeds is not None:
            assert B == 1
            e_idx, e_vec = np.asarray(embeds[0], np.int64), np.asarray(embeds[1])
        logits = None
        main, mc = (caches[:self.cfg.n_layers], caches[self.cfg.n_layers]) if self.mtp is not None else (caches, None)
        for a in range(0, T, self.prefill_piece):
            seg = tokens[:, a:a + self.prefill_piece]
            L = seg.shape[1]
            Tp = self._bucket(L)
            padded = np.zeros((B, Tp), np.int32)
            padded[:, :L] = seg
            ov = None
            if embeds is not None:
                sel = (e_idx >= a) & (e_idx < a + L)
                if sel.any():
                    vec = np.zeros((B, Tp, e_vec.shape[-1]), e_vec.dtype)
                    mask = np.zeros((B, Tp), bool)
                    vec[0, e_idx[sel] - a] = e_vec[sel]
                    mask[0, e_idx[sel] - a] = True
                    ov = (vec, mask)
            if mc is None:
                logits, main = self._run(padded, main, pos0 + a, False, length=L, want_logits=a + L >= T, override=ov)
                continue
            logits, main, streams = self._run(padded, main, pos0 + a, False, length=L, want_logits=a + L >= T,
                                              want_streams=True, override=ov)
            hidden = self._prog_hidden(B, Tp)(self.params["norm"], streams)                 # [B,Tp,D]
            p0 = pos0 + a
            if p0 == 0:                                        # rows 0..L-2: token j+1 with hidden j
                mt = np.zeros((B, Tp), np.int32); mt[:, :L - 1] = seg[:, 1:]
                hp = jnp.concatenate([hidden[:, :Tp - 1], jnp.zeros_like(hidden[:, :1])], axis=1)
                start, length = 0, L - 1
            else:                                              # rows p0-1..p0+L-2: token p0+j with hidden p0+j-1
                mt = padded
                hp = jnp.concatenate([mc["h"], hidden[:, :Tp - 1]], axis=1)
                start, length = p0 - 1, L
            mcache = {k: v for k, v in mc.items() if k != "h"}
            if length > 0:
                _, _, mcache = self._prog_mtp(B, Tp, True)(self.mtp, self.params["embed"], self.params["lm_head"],
                                                           jnp.asarray(mt), hp, jnp.int32(start), jnp.int32(length), mcache)
            mc = {**mcache, "h": lax.dynamic_index_in_dim(hidden, L - 1, axis=1, keepdims=True)}
        return logits, (main + [mc] if mc is not None else main), pos0 + T

    # ---- speculative decoding with the MTP (NextN) layer
    def set_mtp(self, params, qtypes, k=1):
        """Install the MTP layer: `params` from checkpoint.load_mtp_layer (2-D matrices bf16) plus its expert planes
        (pack_layer_from_gguf) under mlp["gate_q"/"up_q"/"down_q"]; `qtypes` their formats; `k` drafts per step.
        The layer's attention cache + carried hidden state become cache entry n_layers (alloc_caches)."""
        col, row = P(None, AXIS), P(AXIS, None)
        attn = {"q_a": R, "q_a_norm": R, "q_b": col, "kv_a": R, "kv_a_norm": R, "kv_b": col, "o": row,
                "indexer": {kk: R for kk in params["attn"]["indexer"]}}
        mlp = {"router_w": R, "router_bias": R, "shared": {"gate": col, "up": col, "down": row}}
        for kk in ("gate_q", "up_q", "down_q"):
            mlp[kk] = {pk: P(AXIS) for pk in params["mlp"][kk]}
        spec = {"ln1": R, "ln2": R, "attn": attn, "mlp": mlp, "enorm": R, "hnorm": R, "eh_proj": R, "head_norm": R}
        self.expert_fetch.qtypes[self.cfg.n_layers] = qtypes
        self.mtp = jax.tree.map(lambda a, sp: a if hasattr(a, "sharding") and a.sharding == NamedSharding(self.mesh, sp)
                                else jax.device_put(a, NamedSharding(self.mesh, sp)), params, spec)
        self.mtp_spec, self.mtp_k = spec, k

    def _prog_mtp(self, B, T, has_cache=True):
        """(mtp params, embed, lm_head, tokens [B,T] = the token AFTER each MTP position, h_prev [B,T,D], pos0,
        length, cache) -> (logits of the last real position [B,V], hidden [B,T,D], new cache)."""
        key = ("mtp", B, T, has_cache)
        if key not in self._progs:
            from glm53 import quant8 as Q8
            cs = {k: v for k, v in self._cache_spec(self.cfg.n_layers).items() if k != "h"}

            def prog(mp, embed, lm_head, tokens, h_prev, pos0, length, cache):
                mp = Q8.dequant_tree(mp, self.lcfg.dtype)
                e = self._embed(embed, tokens).astype(self.lcfg.dtype)
                e = jnp.where((pos0 + jnp.arange(T))[None, :, None] == 0, jnp.zeros((), e.dtype), e)   # no position -1
                h, nc = M.mtp_layer(mp, e, h_prev, self.lcfg, cache, pos0, self.expert_fetch, "auto", self.max_len,
                                    length, layer=self.cfg.n_layers)
                last = lax.dynamic_index_in_dim(h, length - 1, axis=1, keepdims=False)
                return self._logits(lm_head, last), h, nc
            in_specs = (self.mtp_spec, self.specs["embed"], self.specs["lm_head"], R, R, R, R, cs)
            sm = shard_map(prog, mesh=self.mesh, in_specs=in_specs, out_specs=(R, R, cs), check_vma=False)
            self._progs[key] = jax.jit(sm, donate_argnums=(7,))
        return self._progs[key]

    def _run_verify(self, tokens, caches, pos0, sampler=None):
        """Recurrent T-token step returning (logits [B,T,V], caches, per-layer histories, hidden [B,T,D], ids):
        ids = the device-sampled token of every position [B,T] when a DeviceSampler is given, else None."""
        B, T = tokens.shape
        streams = self._prog_embed(B, T)(self.params["embed"], jnp.asarray(tokens))
        new, hists = [], []
        pos0 = jnp.int32(pos0)
        for g in self.groups["decode"]:
            prog = self._prog_group(g, B, T, True, True, hist=True)
            streams, ncs, hs = prog([self.params["layers"][i] for i in g], streams, pos0, jnp.int32(T), [caches[i] for i in g])
            new.extend(ncs); hists.extend(hs)
            self.launches += 1
        if sampler is None:
            logits, hidden = self._prog_head_all(B, T)(self.params["norm"], self.params["lm_head"], streams)
            return logits, new, hists, hidden, None
        ids, logits, hidden, nk = self._prog_head_all(B, T, True)(self.params["norm"], self.params["lm_head"], streams,
                                                                  *sampler.bind(self.mesh))
        sampler.advance(nk)
        return logits, new, hists, hidden, ids

    def spec_decode(self, token, caches, pos, k=None, sampler=None, draft_override=None, stop_ids=()):
        """One speculative step from the committed token `token` [B] at position `pos` (B = 1): the MTP layer
        drafts k tokens, the main model verifies them in one (k+1)-token recurrent step, the longest accepted prefix
        is kept (recurrent states and pool tails rolled back to it). `sampler` decides the target's token per
        position: None = argmax, a host callable `sampler(logits_row) -> int`, or a `DeviceSampler` (sampled inside
        the head program; only the ids cross to the host). A draft is accepted iff it equals the target's own choice,
        which is exact for greedy and a valid, if conservative, scheme for sampling. A draft in `stop_ids` ends the
        accepted run (it is emitted as the last, unfed token). Returns (emitted tokens [1..k+1] — all but the last
        were fed to the model, logits of the last emitted position [B,V], caches, new position)."""
        k = self.mtp_k if k is None else k
        n = self.cfg.n_layers
        B = token.shape[0]
        assert B == 1 and self.mtp is not None
        mc = caches[n]
        mcache = {kk: v for kk, v in mc.items() if kk != "h"}
        hp = mc["h"]
        tok = jnp.asarray(np.asarray(token, np.int32).reshape(B, 1))
        drafts, tails = [], []                                  # the draft chain stays on the device (one host sync per step)
        for j in range(k):
            tails.append((mcache["tk"], mcache["tg"]))          # before draft j (donated: keep the objects, not copies)
            lg, hd, mcache = self._prog_mtp(B, 1, True)(self.mtp, self.params["embed"], self.params["lm_head"],
                                                        tok, hp, jnp.int32(pos - 1 + j), jnp.int32(1),
                                                        {**mcache, "tk": jnp.copy(mcache["tk"]), "tg": jnp.copy(mcache["tg"])})
            d = jnp.asarray([[int(draft_override[j])]], jnp.int32) if draft_override is not None else jnp.argmax(lg, axis=-1)[:, None].astype(jnp.int32)
            drafts.append(d); tok = d; hp = hd
        seq = jnp.concatenate([jnp.asarray(np.asarray(token, np.int32).reshape(B, 1))] + drafts, axis=1)   # [B,k+1]
        dev = isinstance(sampler, DeviceSampler)
        logits, new, hists, hidden, ids = self._run_verify(seq, caches[:n], pos, sampler if dev else None)
        seq_h = np.asarray(seq)[0]
        drafts_h = [int(x) for x in seq_h[1:]]
        if dev:                                                 # only k+1 ids cross to the host, not [k+1, V] logits
            preds = [int(x) for x in np.asarray(ids[0])]
        else:
            lg_h = np.asarray(logits[0])
            preds = [int(np.argmax(lg_h[j])) if sampler is None else int(sampler(lg_h[j])) for j in range(k + 1)]
        a = 0                                                   # a stop token is never fed: it ends the accepted run
        while a < k and preds[a] == drafts_h[a] and drafts_h[a] not in stop_ids:
            a += 1
        if a < k:
            rolled = self._prog_rollback(B, k + 1)(hists, jnp.int32(a + 1))
            new = [({**c, **r} if r else c) for c, r in zip(new, rolled)]
            if a + 1 < k:                                       # drafts a+1.. were fed rejected tokens: tail as before draft a+1
                mcache = {**mcache, "tk": tails[a + 1][0], "tg": tails[a + 1][1]}
        emitted = drafts_h[:a] + [preds[a]]
        h_next = lax.dynamic_index_in_dim(hidden, a, axis=1, keepdims=True)
        return emitted, logits[:, a], new + [{**mcache, "h": h_next}], pos + a + 1

    def decode(self, token, caches, pos):
        B = token.shape[0]
        n = self.cfg.n_layers
        extra = caches[n:] if len(caches) > n else []
        logits, caches = self._run(np.asarray(token).reshape(B, 1), caches[:n], pos, True)
        return logits, caches + extra, pos + 1
