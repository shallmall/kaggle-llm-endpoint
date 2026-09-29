"""Pallas TPU kernel: fused codebook-dequant + matvec of routed experts straight from the planar HBM tables.

`moe_matvec(planes, qtype, nblk, idx, X)` computes, for every expert slot i (one (token, expert) pair):
    out[i, r] = sum_n  W_{idx[i]}[r, n] * x_i[n]              (W dequantized on the fly, never written anywhere)
where X[i, w, c] = x_i[input(w, c)] is the per-word activation matrix of glm53.planes (see `planes.pm_x`).

Grid = expert slots; the expert id comes from a scalar-prefetch operand so each grid step DMAs exactly the planes
of the chosen expert (double-buffered by Pallas). Inside, one (8, 128) vreg of the qs plane = 8 packed words x 128
matrix rows; the codebook index of every group is looked up with in-vreg lane gathers (`jnp.take_along_axis` along
lanes on the (8, 128)-tiled code table, one gather per 128-entry table row + a select tree over the high index
bits), the 2-/3-bit levels and sign bits are unpacked with shifts, and the values are multiplied-accumulated on the
VPU against the activation broadcast along lanes: acc[s, r] += v[s, r] * X[8b + s, c]. The sublane sum of the
accumulator over all words is the output row block. Everything is 32-bit (no packed dtypes), loads are (n, 128)
windows at 8-aligned sublane offsets, so the kernel also runs under `interpret=True` on CPU for the tests.

Codebook tables: `code_table(qtype)` packs each grid entry as GROUP_W[qtype] level indices (2 or 3 bits each) into
one int32; entries are laid out [n_entries // 128, 128].
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from glm53 import iqquant as Q
from glm53 import planes as PL
from glm53 import model as M

LANES = 128
SUB = 8


@functools.lru_cache(None)
def code_table(qtype):
    """int32 [n // 128, 128]: entry e -> sum_k level_k << (lb * k)."""
    grid, levels = Q.GRIDS[qtype], Q.LEVELS[qtype]
    lb = int(np.ceil(np.log2(len(levels))))
    lv = np.searchsorted(levels, grid)                                   # [n, w]
    assert np.array_equal(levels[lv], grid)
    code = np.zeros(grid.shape[0], np.int64)
    for k in range(grid.shape[1]):
        code |= lv[:, k].astype(np.int64) << (lb * k)
    return code.astype(np.int32).reshape(-1, LANES)


@functools.lru_cache(None)
def kv16_table():
    t = np.zeros((1, LANES), np.float32)
    t[0, :16] = Q.KVALUES_IQ4NL
    return t


def _f16_bits_to_f32(h):
    """u32 holding f16 bits (low 16) -> f32 (normal + subnormal + zero; no inf/nan expected)."""
    h = h & 0xFFFF
    s = (h >> 15) & 1
    e = (h >> 10) & 31
    m = h & 1023
    bits = (s << 31) | ((e + 112) << 23) | (m << 13)
    normal = lax.bitcast_convert_type(bits.astype(jnp.uint32), jnp.float32)
    sub = m.astype(jnp.int32).astype(jnp.float32) * (2.0 ** -24)
    sub = jnp.where(s == 1, -sub, sub)
    return jnp.where(e == 0, sub, normal)


def _rows8(src, reps):
    """src (n, 128) with n * reps == 8 -> (8, 128) whose sublane s is src[s // reps]."""
    n = src.shape[0]
    if n == 1:
        return jnp.broadcast_to(src, (SUB, LANES))
    row = lax.broadcasted_iota(jnp.int32, (SUB, LANES), 0)
    out = jnp.broadcast_to(src[n - 1:n], (SUB, LANES))
    for r in range(n - 2, -1, -1):
        out = jnp.where(row < (r + 1) * reps, jnp.broadcast_to(src[r:r + 1], (SUB, LANES)), out)
    return out


def _sub_iota():
    return lax.broadcasted_iota(jnp.uint32, (SUB, LANES), 0)


def _lookup(tb, idx):
    """tb: list of (8,128) int32 table rows (128 entries each); idx (8,128) int32 -> codes (8,128) int32."""
    lo = idx & (LANES - 1)
    hi = idx >> 7
    vals = [jnp.take_along_axis(t, lo, axis=1) for t in tb]
    bit = 0
    while len(vals) > 1:
        sel = ((hi >> bit) & 1) == 1
        vals = [jnp.where(sel, vals[2 * m + 1], vals[2 * m]) for m in range(len(vals) // 2)]
        bit += 1
    return vals[0]


def _signed(v_int, sgn_bits, k):
    v = v_int.astype(jnp.int32).astype(jnp.float32)
    return jnp.where(((sgn_bits >> k) & 1) == 1, -v, v)


# ------------------------------------------------------------------------------------------ per-format word decoders
# `rows(name, r0, n)` returns rows r0..r0+n-1 of that plane for the current expert as an (n, 128) value (static lanes).
# Each decoder returns a list of (scale (8,128) f32, [(c, value (8,128) f32), ...]) for the 8 words 8b..8b+7.
def _decode_iq2_s(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sgw = rows("sg", 8 * b, 8)
    sub = _sub_iota()
    shift8 = (sub & 3) * 8
    qhw = (_rows8(rows("qh", 2 * b, 2), 4) >> shift8) & 255                 # qh byte of word 8b+s
    scw = (_rows8(rows("sc", 2 * b, 2), 4) >> shift8) & 255                 # sc byte of word 8b+s
    dw = rows("d", b // 2, 1) >> (16 * (b % 2))                             # block b for all 8 words
    dv = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    scale = [dv * (0.5 + ((scw >> (4 * h)) & 15).astype(jnp.int32).astype(jnp.float32)) * 0.25 for h in range(2)]
    groups = [(scale[0], []), (scale[1], [])]
    for j in range(4):
        idx = (((q >> (8 * j)) & 255) | (((qhw >> (2 * j)) & 3) << 8)).astype(jnp.int32)
        code = _lookup(tb, idx)
        sgn = (sgw >> (8 * j)) & 255
        for k in range(8):
            lv = (code >> (2 * k)) & 3
            v = 8 + 17 * lv + (lv >> 1)
            groups[j // 2][1].append((8 * j + k, _signed(v, sgn, k)))
    return groups


def _decode_iq3_s(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    sgh = (_rows8(rows("sg", 4 * b, 4), 2) >> ((sub & 1) * 16)) & 0xFFFF   # 16 sign bits of word 8b+s
    qhn = (_rows8(rows("qh", b, 1), 8) >> (sub * 4)) & 15                    # 4 high bits of word 8b+s
    scn = (_rows8(rows("sc", b // 2, 1), 8) >> ((4 * (b % 2) + (sub >> 1)) * 4)) & 15
    dw = rows("d", b // 4, 1) >> (16 * ((b // 2) % 2))
    dv = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    scale = dv * (1.0 + 2.0 * scn.astype(jnp.int32).astype(jnp.float32))
    items = []
    for j in range(4):
        idx = (((q >> (8 * j)) & 255) | (((qhn >> j) & 1) << 8)).astype(jnp.int32)
        code = _lookup(tb, idx)
        sgn = (sgh >> (4 * j)) & 15
        for k in range(4):
            lv = (code >> (3 * k)) & 7
            items.append((4 * j + k, _signed(2 * lv + 1, sgn, k)))
    return [(scale, items)]


def _decode_iq4_xs(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    i = (2 * b + (sub >> 2)) % 8                                             # sub-block of word 8b+s within its block
    slw = (_rows8(rows("sl", b // 4, 1), 8) >> (4 * i)) & 15
    shw = (_rows8(rows("sh", b // 4, 1), 8) >> (2 * i)) & 3
    dw = rows("d", b // 8, 1) >> (16 * ((b // 4) % 2))
    dv = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    scale = dv * ((slw | (shw << 4)).astype(jnp.int32) - 32).astype(jnp.float32)
    kv = tb[0]                                                               # (8,128) f32, entries at lanes 0..15
    items = []
    for pos in range(2):
        for j in range(4):
            nib = ((q >> (8 * j + 4 * pos)) & 15).astype(jnp.int32)
            items.append((4 * pos + j, jnp.take_along_axis(kv, nib, axis=1)))
    return [(scale, items)]


def _kq_scales(rows, b, sub):
    """K-quant scale bytes for the 8 words of vreg b (half n = b % 2 of block b // 2): per shift s an (8,128) u32 with
    the scale byte `is` = 8 n + 2 s + (sublane >= 4) of the block's 16, plus the block's d word (1,128)."""
    blk, n = b // 2, b % 2
    scw = rows("sc", 4 * blk, 4)                                                 # (4,128): the 16 scale bytes
    hi = (sub >= 4).astype(jnp.uint32)
    out = []
    for s_ in range(4):
        r = 2 * n + s_ // 2
        row = jnp.broadcast_to(scw[r:r + 1], (SUB, LANES))
        out.append((row >> (8 * (2 * (s_ % 2) + hi))) & 255)
    return out, rows("d", blk, 1)


def _decode_q2_k(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    sc, dw = _kq_scales(rows, b, sub)
    d = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    dmin = jnp.broadcast_to(_f16_bits_to_f32(dw >> 16), (SUB, LANES))
    items = []
    for s_ in range(4):
        dl = d * (sc[s_] & 15).astype(jnp.int32).astype(jnp.float32)
        ml = dmin * (sc[s_] >> 4).astype(jnp.int32).astype(jnp.float32)
        for j in range(4):
            v = ((q >> (8 * j + 2 * s_)) & 3).astype(jnp.int32).astype(jnp.float32)
            items.append((4 * s_ + j, dl * v - ml))
    return [(jnp.ones((SUB, LANES), jnp.float32), items)]


def _decode_q3_k(b, rows, tb):
    q = rows("qs", 8 * b, 8)
    sub = _sub_iota()
    blk, n = b // 2, b % 2
    hm = rows("hm", 8 * blk, 8)                                                  # (8,128): u32 q = hmask bytes 4q..4q+3
    sc, dw = _kq_scales(rows, b, sub)
    d = jnp.broadcast_to(_f16_bits_to_f32(dw), (SUB, LANES))
    items = []
    for s_ in range(4):
        dl = d * ((sc[s_] & 255).astype(jnp.int32) - 32).astype(jnp.float32)
        for j in range(4):
            v = ((q >> (8 * j + 2 * s_)) & 3).astype(jnp.int32)
            hb = ((hm >> (8 * j + 4 * n + s_)) & 1).astype(jnp.int32)
            items.append((4 * s_ + j, dl * (v - 4 + 4 * hb).astype(jnp.float32)))
    return [(jnp.ones((SUB, LANES), jnp.float32), items)]


DECODE = {"IQ2_S": _decode_iq2_s, "IQ3_S": _decode_iq3_s, "IQ4_XS": _decode_iq4_xs, "Q2_K": _decode_q2_k,
          "Q3_K": _decode_q3_k}


def _table_arrays(qtype):
    if qtype == "IQ4_XS":
        return jnp.asarray(kv16_table())
    if qtype in ("Q2_K", "Q3_K"):
        return jnp.zeros((1, LANES), jnp.int32)                              # no codebook
    return jnp.asarray(code_table(qtype))


def _pick_rows(block, off, r0, n):
    """block (8, 128) value; rows off+r0 .. off+r0+n-1 where `off` is a traced scalar in [0, 8): select tree
    (Mosaic has no unaligned dynamic sublane loads)."""
    outs = []
    for r in range(n):
        want = off + r0 + r
        v = block[0:1]
        for q in range(1, SUB):
            v = jnp.where(want == q, block[q:q + 1], v)
        outs.append(v)
    return outs[0] if n == 1 else jnp.concatenate(outs, axis=0)


# ------------------------------------------------------------------------------------------------------- the kernel
def moe_matvec(planes, qtype, nblk, idx, X, *, interpret=False, lane_block=None):
    """planes: {k: u32 [E*rows_p, R] or [1, E*rows_p, R]}; idx int32 [Nk]; X f32 [Nk, W, C] -> f32 [Nk, R].

    Grid = (expert slot, lane block of `lane_block` matrix rows); every ref access uses static indices (Mosaic rejects
    dynamic sublane/lane offsets that are not provably tile aligned): planes with fewer than 8 rows per expert are
    fetched as their 8-row block and the expert's rows are picked with selects on the scalar-prefetched offset."""
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    first = planes[keys[0]]
    lead = first.ndim == 3
    R = first.shape[-1]
    Nk = idx.shape[0]
    assert X.shape == (Nk, W, C), (X.shape, (Nk, W, C))
    assert R % LANES == 0 and W % SUB == 0, (R, W)
    LB = lane_block or min(R, 1024)
    assert R % LB == 0 and LB % LANES == 0, (R, LB)
    n_lb = R // LB
    n_vreg = LB // LANES
    tbl = _table_arrays(qtype)
    n_tbl = tbl.shape[0]
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]

    def kernel(idx_ref, *refs):
        plane_refs = dict(zip(keys, refs[:len(keys)]))
        tbl_ref, x_ref, o_ref = refs[len(keys):]
        i = pl.program_id(0)
        e = idx_ref[i]
        offs = {k: (lax.rem(e * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        if qtype == "IQ4_XS":
            tb = [jnp.broadcast_to(tbl_ref[0:1, :], (SUB, LANES))]
        else:
            tb = [jnp.broadcast_to(tbl_ref[r:r + 1, :], (SUB, LANES)) for r in range(n_tbl)]
        for v in range(n_vreg):
            lanes = slice(v * LANES, (v + 1) * LANES)

            def rows(k, r0, n, lanes=lanes):
                if offs[k] is None:
                    return plane_refs[k][r0:r0 + n, lanes]
                return _pick_rows(plane_refs[k][0:SUB, lanes], offs[k], r0, n)

            out = jnp.zeros((SUB, LANES), jnp.float32)
            for b in range(W // SUB):
                xb = x_ref[SUB * b:SUB * (b + 1), :]                                    # (8, C)
                for scale, items in decode(b, rows, tb):
                    acc = jnp.zeros((SUB, LANES), jnp.float32)
                    for c, val in items:
                        acc = acc + val * jnp.broadcast_to(xb[:, c:c + 1], (SUB, LANES))
                    out = out + acc * scale
            o_ref[:, lanes] = jnp.sum(out, axis=0, keepdims=True)

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda i, cb, idx_ref: (0, (idx_ref[i] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda i, cb, idx_ref: ((idx_ref[i] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys]
    in_specs += [pl.BlockSpec(tbl.shape, lambda i, cb, idx_ref: (0, 0)),
                 pl.BlockSpec((None, W, C), lambda i, cb, idx_ref: (i, 0, 0))]
    out_spec = pl.BlockSpec((None, 1, LB), lambda i, cb, idx_ref: (i, 0, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(Nk, n_lb), in_specs=in_specs,
                                             out_specs=out_spec)
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((Nk, 1, R), jnp.float32),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")))
    args = [planes[k] for k in keys] + [tbl, X]
    return fn(idx, *args).reshape(Nk, R)


def moe_matvec_ref(planes, qtype, nblk, idx, X):
    """Pure-JAX reference (gathers the experts, dequantizes with glm53.planes, einsum)."""
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    sel = {}
    for k in keys:
        a = planes[k]
        a = a[0] if a.ndim == 3 else a
        R = a.shape[-1]
        a = a.reshape(-1, rows_p[k], R)
        sel[k] = jnp.take(a, idx, axis=0)                                    # [Nk, rows_p, R]
    V = PL.dequant_planes(sel, qtype, nblk)                                 # [Nk, C*W, R]
    x_pm = jnp.swapaxes(X, 1, 2).reshape(X.shape[0], -1)                   # [Nk, C*W]
    return jnp.einsum("ni,nir->nr", x_pm, V, precision="highest")


# ------------------------------------------------------------------------------------------ prefill (sweep) kernels
# Dequantize each expert ONCE into (K, 128) bf16 tiles of W^T in VMEM and MXU-matmul all T tokens against them.
# Tile rows are ordered (word block b, value plane c, word s): input(8b+s, c) -> row (b*C + c)*8 + s; `pm_mxu`
# permutes activations to that order. Expert slots come from scalar-prefetched `ids` (unique active experts first,
# inactive slots repeat the last active id so their planes are never re-DMA'd) and `n_active` gates the compute.
def pm_mxu(qtype, x):
    """x [..., n] natural -> [..., n] in the kernel tile order (b, c, s)."""
    X = PL.pm_x(qtype, x)                                                    # [..., W, C]
    lead = X.shape[:-2]
    W, C = X.shape[-2:]
    y = X.reshape(lead + (W // SUB, SUB, C))
    y = jnp.swapaxes(y, -1, -2)                                              # [..., W/8, C, 8]
    return y.reshape(lead + (W * C,))


def _tiles(decode, rows, tb, b0, nb, dtype=jnp.bfloat16):
    """(nb*C*8, 128) tile of scaled W^T rows for word blocks b0..b0+nb-1 at the current lanes, in `dtype`."""
    parts = []
    for b in range(b0, b0 + nb):
        items = []
        for scale, its in decode(b, rows, tb):
            items += [(c, v * scale) for c, v in its]
        items.sort(key=lambda t: t[0])
        parts += [v for _, v in items]
    return jnp.concatenate(parts, axis=0).astype(dtype)


def _dot(a, b):
    prec = lax.Precision.HIGHEST if a.dtype == jnp.float32 else None
    return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=prec)


def active_slots(idx, E):
    """idx int32 [N, k] -> (ids int32 [E], n_active int32): unique routed experts first (ascending), the rest of the
    slots repeat the last active id."""
    hit = (jax.nn.one_hot(idx.reshape(-1), E, dtype=jnp.int32).sum(0) > 0)          # [E] bool
    order = jnp.argsort(jnp.where(hit, 0, 1) * E + jnp.arange(E))                     # active ids first, ascending
    n_active = hit.sum().astype(jnp.int32)
    last = order[jnp.maximum(n_active - 1, 0)]
    ids = jnp.where(jnp.arange(E) < n_active, order, last).astype(jnp.int32)
    return ids, n_active


def _k_group(qtype):
    """word blocks per MXU tile so that K >= 128."""
    return max(1, 128 // (PL.PLANES_PER_WORD[qtype] * SUB))


def _probe_tiles(probe, tiles_fn, K, dtype):
    """Cost-split probes: 'mxu' replaces the dequantized tile by a constant (matmul cost only), 'deq' keeps the
    dequant but returns None so the caller skips the dot (dequant cost only)."""
    if probe == "mxu":
        return jnp.full((K, LANES), 0.01, dtype)
    return tiles_fn()


def moe_sweep_gateup(planes_g, planes_u, qtype, nblk, ids, n_active, xk, limit, *, interpret=False, lane_block=None,
                     t_block=512, vmem_mb=48, probe=None):
    """planes_{g,u}: {k: u32 [1, E*rows_p, R]} (same qtype); ids [E] int32; n_active int32 scalar; xk [T, n_in] in
    pm_mxu order (bf16 on TPU; f32 -> f32 tiles + HIGHEST-precision dots for the CPU tests) ->
    h [E, T, R] (xk.dtype) = swiglu_clamped(x @ gate_e^T, x @ up_e^T) for active slots, 0 elsewhere.
    `probe` (timing only, wrong numbers): 'mxu' = constant tiles, 'deq' = dequant without the matmul."""
    dtype = xk.dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    E = ids.shape[0]
    T, n_in = xk.shape
    assert n_in == nblk * PL.QK, (n_in, nblk)
    first = planes_g[keys[0]]
    lead = first.ndim == 3
    R = first.shape[-1]
    LB = lane_block or min(R, 1024)
    TB = min(T, t_block)
    assert T % TB == 0 and R % LB == 0
    n_lb, n_vreg, n_tb = R // LB, LB // LANES, T // TB
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    n_tbl = tbl.shape[0]
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(ids_ref, nact_ref, *refs):
        g_refs = dict(zip(keys, refs[:nk]))
        u_refs = dict(zip(keys, refs[nk:2 * nk]))
        tbl_ref, x_ref, o_ref, acc_g, acc_u = refs[2 * nk:]
        e = pl.program_id(1)
        eid = ids_ref[e]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        if qtype == "IQ4_XS":
            tb = [jnp.broadcast_to(tbl_ref[0:1, :], (SUB, LANES))]
        else:
            tb = [jnp.broadcast_to(tbl_ref[r:r + 1, :], (SUB, LANES)) for r in range(n_tbl)]

        @pl.when(e >= nact_ref[0])
        def _inactive():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

        @pl.when(e < nact_ref[0])
        def _active():
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows_of(prefs, lanes=lanes):
                    def rows(k, r0, n):
                        if offs[k] is None:
                            return prefs[k][r0:r0 + n, lanes]
                        return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)
                    return rows

                rg, ru = rows_of(g_refs), rows_of(u_refs)
                ag = jnp.zeros((TB, LANES), jnp.float32)
                au = jnp.zeros((TB, LANES), jnp.float32)
                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    xs = x_ref[:, k0:k0 + K]                                              # (TB, K) bf16
                    tg = _probe_tiles(probe, lambda: _tiles(decode, rg, tb, b0, nbg, dtype), K, dtype)
                    tu = _probe_tiles(probe, lambda: _tiles(decode, ru, tb, b0, nbg, dtype), K, dtype)
                    if probe == "deq":
                        ag = ag + jnp.sum(tg, axis=0, keepdims=True).astype(jnp.float32)
                        au = au + jnp.sum(tu, axis=0, keepdims=True).astype(jnp.float32)
                        continue
                    ag = ag + _dot(xs, tg)
                    au = au + _dot(xs, tu)
                acc_g[:, lanes] = ag
                acc_u[:, lanes] = au
            o_ref[...] = M.swiglu_clamped(acc_g[...], acc_u[...], limit).astype(o_ref.dtype)

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda t, e, cb, ids_ref, n_ref: (0, (ids_ref[e] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda t, e, cb, ids_ref, n_ref: ((ids_ref[e] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys] * 2
    in_specs += [pl.BlockSpec(tbl.shape, lambda t, e, cb, ids_ref, n_ref: (0, 0)),
                 pl.BlockSpec((TB, n_in), lambda t, e, cb, ids_ref, n_ref: (t, 0))]
    out_spec = pl.BlockSpec((None, TB, LB), lambda t, e, cb, ids_ref, n_ref: (e, t, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(n_tb, E, n_lb), in_specs=in_specs,
                                             out_specs=out_spec,
                                             scratch_shapes=[pltpu.VMEM((TB, LB), jnp.float32)] * 2)
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((E, T, R), dtype),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",) * 3,
                                                             vmem_limit_bytes=vmem_mb << 20))
    args = [planes_g[k] for k in keys] + [planes_u[k] for k in keys] + [tbl, xk]
    return fn(ids, n_active.reshape(1), *args)


def moe_sweep_down(planes, qtype, nblk, ids, n_active, hk, *, interpret=False, lane_block=None, t_block=512,
                   vmem_mb=48, probe=None):
    """planes {k: u32 [1, E*rows_p, R]}; hk [E, T, n_in] (pm_mxu order, routing weights folded in; bf16 or f32) ->
    y f32 [T, R] = sum over active slots of hk[e] @ down_e^T."""
    dtype = hk.dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    E, T, n_in = hk.shape
    assert n_in == nblk * PL.QK and ids.shape == (E,)
    first = planes[keys[0]]
    lead = first.ndim == 3
    R = first.shape[-1]
    LB = lane_block or min(R, 1024)
    TB = min(T, t_block)
    assert T % TB == 0 and R % LB == 0
    n_lb, n_vreg, n_tb = R // LB, LB // LANES, T // TB
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    n_tbl = tbl.shape[0]
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(ids_ref, nact_ref, *refs):
        prefs = dict(zip(keys, refs[:nk]))
        tbl_ref, h_ref, o_ref = refs[nk:]
        e = pl.program_id(2)
        eid = ids_ref[e]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        if qtype == "IQ4_XS":
            tb = [jnp.broadcast_to(tbl_ref[0:1, :], (SUB, LANES))]
        else:
            tb = [jnp.broadcast_to(tbl_ref[r:r + 1, :], (SUB, LANES)) for r in range(n_tbl)]

        @pl.when(e == 0)
        def _init():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

        @pl.when(e < nact_ref[0])
        def _active():
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows(k, r0, n, lanes=lanes):
                    if offs[k] is None:
                        return prefs[k][r0:r0 + n, lanes]
                    return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)

                acc = o_ref[:, lanes]
                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    td = _probe_tiles(probe, lambda: _tiles(decode, rows, tb, b0, nbg, dtype), K, dtype)
                    if probe == "deq":
                        acc = acc + jnp.sum(td, axis=0, keepdims=True).astype(jnp.float32)
                        continue
                    acc = acc + _dot(h_ref[:, k0:k0 + K], td)
                o_ref[:, lanes] = acc

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda t, cb, e, ids_ref, n_ref: (0, (ids_ref[e] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda t, cb, e, ids_ref, n_ref: ((ids_ref[e] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys]
    in_specs += [pl.BlockSpec(tbl.shape, lambda t, cb, e, ids_ref, n_ref: (0, 0)),
                 pl.BlockSpec((None, TB, n_in), lambda t, cb, e, ids_ref, n_ref: (e, t, 0))]
    out_spec = pl.BlockSpec((TB, LB), lambda t, cb, e, ids_ref, n_ref: (t, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(n_tb, n_lb, E), in_specs=in_specs,
                                             out_specs=out_spec)
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((T, R), jnp.float32),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",) * 3,
                                                             vmem_limit_bytes=vmem_mb << 20))
    return fn(ids, n_active.reshape(1), *[planes[k] for k in keys], tbl, hk)


# ------------------------------------------------------------------------------------- grouped (ragged) prefill GEMM
# The dense sweep multiplies every 512-token chunk against every active expert (masked): on real text ~all 288 experts
# are active, so ~36x the useful MXU work, each expert dequantized once per chunk, and an [E, T, ml] intermediate.
# Here the (token, expert) slots are SORTED BY EXPERT into blocks of `tm` rows (each expert's rows start at a block
# boundary, zero-padded); the kernels walk the blocks in order, DMA + dequantize an expert's planes ONCE (when the
# block's expert changes; the dequantized W^T tiles stay in VMEM scratch) and multiply only that expert's rows.
def ragged_plan(idx, w, E, tm):
    """Slot layout for the grouped kernels. idx int32 [T, k], w f32 [T, k] -> dict with
    blk_expert int32 [G] (expert of row block g; G = T*k//tm + E blocks of `tm` rows, blocks >= n_blocks are padding
    and repeat the last expert), n_blocks int32 [1], tok_row int32 [Rp] (token of every sorted row, -1 = padding),
    w_row f32 [Rp] (its routing weight), row_slot int32 [T, k] (the row of every (token, slot))."""
    T, k = idx.shape
    S = T * k
    G = -(-S // tm) + E
    Rp = G * tm
    flat = idx.reshape(-1).astype(jnp.int32)
    order = jnp.argsort(flat)
    se = flat[order]
    counts = jnp.sum(jax.nn.one_hot(flat, E, dtype=jnp.int32), axis=0)                    # [E]
    padded = -(-counts // tm) * tm
    pstart = jnp.cumsum(padded) - padded
    ustart = jnp.cumsum(counts) - counts
    row_sorted = pstart[se] + (jnp.arange(S, dtype=jnp.int32) - ustart[se])              # row of sorted slot j
    tok_row = jnp.full((Rp,), -1, jnp.int32).at[row_sorted].set((order // k).astype(jnp.int32))
    w_row = jnp.zeros((Rp,), jnp.float32).at[row_sorted].set(w.reshape(-1)[order].astype(jnp.float32))
    row_slot = jnp.zeros((S,), jnp.int32).at[order].set(row_sorted).reshape(T, k)
    nb = padded // tm
    bstart = jnp.cumsum(nb) - nb
    n_blocks = jnp.sum(nb).astype(jnp.int32)
    g = jnp.arange(G, dtype=jnp.int32)
    be = jnp.sum((g[:, None] >= (bstart + nb)[None, :]).astype(jnp.int32), axis=1)        # experts ended at or before g
    be = jnp.minimum(be, E - 1)
    last = be[jnp.maximum(n_blocks - 1, 0)]
    be = jnp.where(g < n_blocks, be, last).astype(jnp.int32)
    return {"blk_expert": be, "n_blocks": n_blocks.reshape(1), "tok_row": tok_row, "w_row": w_row, "row_slot": row_slot}


def _ragged_call(kernel, planes_list, keys, rows_p, blocks, tbl, act, out_dtype, R, LB, tm, G, scratch, interpret, vmem_mb,
                 be, n_blocks):
    """Shared pallas_call plumbing of the two ragged kernels: grid (lane block, row block)."""
    lead = planes_list[0][keys[0]].ndim == 3
    n_lb = R // LB
    Rp, n_in = act.shape

    def plane_spec(k):
        B = blocks[k]
        if lead:
            return pl.BlockSpec((None, B, LB), lambda cb, g, be_ref, nb_ref: (0, (be_ref[g] * rows_p[k]) // B, cb))
        return pl.BlockSpec((B, LB), lambda cb, g, be_ref, nb_ref: ((be_ref[g] * rows_p[k]) // B, cb))

    in_specs = [plane_spec(k) for k in keys] * len(planes_list)
    in_specs += [pl.BlockSpec(tbl.shape, lambda cb, g, be_ref, nb_ref: (0, 0)),
                 pl.BlockSpec((tm, n_in), lambda cb, g, be_ref, nb_ref: (jnp.minimum(g, nb_ref[0] - 1), 0))]
    out_spec = pl.BlockSpec((tm, LB), lambda cb, g, be_ref, nb_ref: (g, cb))
    grid_spec = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(n_lb, G), in_specs=in_specs,
                                             out_specs=out_spec, scratch_shapes=scratch)
    fn = pl.pallas_call(kernel, grid_spec=grid_spec, out_shape=jax.ShapeDtypeStruct((Rp, R), out_dtype),
                        interpret=interpret,
                        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary"),
                                                             vmem_limit_bytes=vmem_mb << 20))
    args = [p[k] for p in planes_list for k in keys] + [tbl, act]
    return fn(be, n_blocks, *args)


def moe_ragged_gateup(planes_g, planes_u, qtype, nblk, blk_expert, n_blocks, xs, limit, *, tm=32, interpret=False,
                      lane_block=None, vmem_mb=48):
    """Grouped prefill GEMM, gate + up. xs [Rp, n_in] in pm_mxu order, rows sorted by expert in blocks of `tm` rows
    (block g = rows g*tm.., expert blk_expert[g]; blocks >= n_blocks are padding) -> h [Rp, R] (xs.dtype) =
    swiglu_clamped(x @ gate_e^T, x @ up_e^T) per row with e = its block's expert; padding blocks are zero."""
    dtype = xs.dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    Rp, n_in = xs.shape
    assert n_in == nblk * PL.QK and n_in == W * C and Rp % tm == 0, (n_in, nblk, W, C, Rp, tm)
    G = Rp // tm
    assert blk_expert.shape == (G,), (blk_expert.shape, G)
    R = planes_g[keys[0]].shape[-1]
    LB = lane_block or min(R, 1024)
    assert R % LB == 0
    n_vreg = LB // LANES
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    n_tbl = tbl.shape[0]
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(be_ref, nb_ref, *refs):
        g_refs = dict(zip(keys, refs[:nk]))
        u_refs = dict(zip(keys, refs[nk:2 * nk]))
        tbl_ref, x_ref, o_ref, wg, wu = refs[2 * nk:]
        g = pl.program_id(1)
        eid = be_ref[g]
        prev = be_ref[jnp.maximum(g - 1, 0)]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        if qtype == "IQ4_XS":
            tb = [jnp.broadcast_to(tbl_ref[0:1, :], (SUB, LANES))]
        else:
            tb = [jnp.broadcast_to(tbl_ref[r:r + 1, :], (SUB, LANES)) for r in range(n_tbl)]

        @pl.when((g == 0) | (eid != prev))
        def _decode():                                     # this expert's W^T tiles for the lane block, once
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows_of(prefs, lanes=lanes):
                    def rows(k, r0, n):
                        if offs[k] is None:
                            return prefs[k][r0:r0 + n, lanes]
                        return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)
                    return rows

                rg, ru = rows_of(g_refs), rows_of(u_refs)
                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    wg[k0:k0 + K, lanes] = _tiles(decode, rg, tb, b0, nbg, dtype)
                    wu[k0:k0 + K, lanes] = _tiles(decode, ru, tb, b0, nbg, dtype)

        @pl.when(g < nb_ref[0])
        def _rows():
            x = x_ref[...]                                                                  # (tm, n_in)
            o_ref[...] = M.swiglu_clamped(_dot(x, wg[...]), _dot(x, wu[...]), limit).astype(o_ref.dtype)

        @pl.when(g >= nb_ref[0])
        def _pad():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

    scratch = [pltpu.VMEM((n_in, LB), dtype)] * 2
    return _ragged_call(kernel, [planes_g, planes_u], keys, rows_p, blocks, tbl, xs, dtype, R, LB, tm, G, scratch,
                        interpret, vmem_mb, blk_expert, n_blocks)


def moe_ragged_down(planes, qtype, nblk, blk_expert, n_blocks, hs, *, tm=32, interpret=False, lane_block=None,
                    vmem_mb=48, out_dtype=None):
    """Grouped prefill GEMM, down. hs [Rp, n_in] (pm_mxu order, sorted rows as in `moe_ragged_gateup`) ->
    y [Rp, R] (out_dtype, default hs.dtype) = h @ down_e^T per row; padding blocks are zero."""
    dtype = hs.dtype
    out_dtype = out_dtype or dtype
    keys = PL.PLANE_KEYS[qtype]
    rows_p = PL.plane_rows(qtype, nblk)
    W = PL.WORDS_PER_BLOCK[qtype] * nblk
    C = PL.PLANES_PER_WORD[qtype]
    Rp, n_in = hs.shape
    assert n_in == nblk * PL.QK and n_in == W * C and Rp % tm == 0, (n_in, nblk, W, C, Rp, tm)
    G = Rp // tm
    assert blk_expert.shape == (G,), (blk_expert.shape, G)
    R = planes[keys[0]].shape[-1]
    LB = lane_block or min(R, 1024)
    assert R % LB == 0
    n_vreg = LB // LANES
    nbg = _k_group(qtype)
    K = nbg * C * SUB
    tbl = _table_arrays(qtype)
    n_tbl = tbl.shape[0]
    blocks = {k: (rows_p[k] if rows_p[k] % SUB == 0 else SUB) for k in keys}
    decode = DECODE[qtype]
    nk = len(keys)

    def kernel(be_ref, nb_ref, *refs):
        prefs = dict(zip(keys, refs[:nk]))
        tbl_ref, h_ref, o_ref, wd = refs[nk:]
        g = pl.program_id(1)
        eid = be_ref[g]
        prev = be_ref[jnp.maximum(g - 1, 0)]
        offs = {k: (lax.rem(eid * rows_p[k], SUB) if rows_p[k] % SUB else None) for k in keys}
        if qtype == "IQ4_XS":
            tb = [jnp.broadcast_to(tbl_ref[0:1, :], (SUB, LANES))]
        else:
            tb = [jnp.broadcast_to(tbl_ref[r:r + 1, :], (SUB, LANES)) for r in range(n_tbl)]

        @pl.when((g == 0) | (eid != prev))
        def _decode():
            for v in range(n_vreg):
                lanes = slice(v * LANES, (v + 1) * LANES)

                def rows(k, r0, n, lanes=lanes):
                    if offs[k] is None:
                        return prefs[k][r0:r0 + n, lanes]
                    return _pick_rows(prefs[k][0:SUB, lanes], offs[k], r0, n)

                for b0 in range(0, W // SUB, nbg):
                    k0 = b0 * C * SUB
                    wd[k0:k0 + K, lanes] = _tiles(decode, rows, tb, b0, nbg, dtype)

        @pl.when(g < nb_ref[0])
        def _rows():
            o_ref[...] = _dot(h_ref[...], wd[...]).astype(o_ref.dtype)

        @pl.when(g >= nb_ref[0])
        def _pad():
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)

    scratch = [pltpu.VMEM((n_in, LB), dtype)]
    return _ragged_call(kernel, [planes], keys, rows_p, blocks, tbl, hs, out_dtype, R, LB, tm, G, scratch, interpret,
                        vmem_mb, blk_expert, n_blocks)
