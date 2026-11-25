# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-GLUEB: bitwise repacks of the QSA prefill glue (each default off; prefill steps only).

G1 IQ-DENSE (VLLM_TPU_QSA_PREFILL_IQ_DENSE): the iq call in ``qsa.produce`` only: ``qsa.index_rope_halfdense`` (default form,
    see G1_FORM) or ``qsa.index_rope_dense`` (the design's lane-dense views; adds relayout copies, not used).
G2 RING-PRODUCER (VLLM_TPU_QSA_PREFILL_RING_PRODUCER): ``ring_producer`` below writes the three gathers of
    ``qsa._indexer_updates``' ring update (non-SHIFT path) -- current group members ``raw[clip(chunk)]``, previous
    members ``ring[slot, members % rows]`` and the group's old position ``tags[slot, group_start % rows, 0]`` -- in one
    Pallas TensorCore kernel, exactly and in the gathers' shapes (bf16 [4T, 128] x 2, int32 [T]); every consumer (the
    pooled-mean fusion included) is unchanged.
G2b (same key): ``value_shift`` below = ``compressed_pairs``' value gather ``keys[clip(local)]`` exactly, in one small
    Pallas kernel (local = t + ratio - 1 - logical % (2 ratio) + j ratio, an integer identity: shifted rows and a select per
    token). The XLA form (8 static shifts per column + selects) compiled to two [6400,1,128] select fusions of ~102.6k
    estimated cycles each (DEV gbd1), a regression against the gather, so the kernel builds each token's 32-bit word
    (rows 2t, 2t + 1 = keys[clip(t + 3 - r)], keys[clip(t + 7 - r)]) from 32-bit row images instead.
G3 EPI-FOLD (VLLM_TPU_QSA_PREFILL_EPI_FOLD): the union walk's ``lengths > 0`` mask inside its walked / served branches
    (``fold_mask``) and the zero-prefix route's in-place mask dropped (qsa_union_dense.py / qsa_rpa.py).

Prefill steps only: ``prefill_step`` (rows above request slots x max_query_len, and >= 1024 rows); decode / verify
programs are unchanged byte for byte. Engagement lines ('q38 PSTEP-GLUEB G<n> ENGAGED ...') are logged at trace time
(Python only, nothing enters the program), one per traced use.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

MIN_ROWS = 1024
_HI = 0xFFFF0000


def prefill_step(tokens: int, requests: int, max_query_len: int) -> bool:
    """A prefill-sized step: more rows than request slots x max_query_len (the decode / verify bound) and >= 1024."""
    return int(tokens) >= MIN_ROWS and int(tokens) > int(requests) * int(max_query_len)


def engaged(edit: str, message: str, *args) -> None:
    """One INFO line per traced use (the car's per-arm engagement label; base programs print none)."""
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info("q38 PSTEP-GLUEB %s ENGAGED: " + message, edit, *args)


def iq_dense_applies(x, rotary_dim: int, prefill_rows: bool) -> bool:
    return bool(
        envs.VLLM_TPU_QSA_PREFILL_IQ_DENSE
        and prefill_rows
        and x.ndim == 3
        and rotary_dim % 2 == 0
        and x.shape[-1] >= rotary_dim
        and (x.shape[1] * (rotary_dim // 2)) % 128 == 0
    )


# G1 form: "half" = qsa.index_rope_halfdense (default: no relayout, 2x lane density); "dense" = the design's [T, H * half] views
# (qsa.index_rope_dense; kept for reference: XLA lays [T, 128] token-minor and adds 12 transposing copies, DEV gbd1 FL / F1).
G1_FORM = "half"


def tile_lanes(v, heads: int):
    """``[T, half] -> [T, heads * half]``, lane ``h * half + j`` = ``v[:, j]`` (the dense view's lane order)."""
    return jnp.tile(v, (1, heads))


def cs_lanes(c, s):
    """The half-dense G1 form's per-lane multipliers: ``[c | s]`` (with a) and ``[s | c]`` (with b), lanes [0, half) | [half, 2 half)."""
    return jnp.concatenate([c, s], -1), jnp.concatenate([s, c], -1)


def ring_producer_applies(raw, ring, tags, ratio: int, prefill_rows: bool) -> bool:
    return bool(
        envs.VLLM_TPU_QSA_PREFILL_RING_PRODUCER
        and prefill_rows
        and not envs.VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT
        and ratio == 4
        and raw.ndim == 2
        and raw.dtype == jnp.bfloat16
        and raw.shape[1] == 128
        and len(raw) % 128 == 0
        and ring.ndim == 3
        and ring.dtype == jnp.bfloat16
        and ring.shape[1] == 8
        and ring.shape[2] == raw.shape[1]
        and tuple(tags.shape) == (ring.shape[0], ring.shape[1], 2)
    )


def value_shift_applies(keys, ratio: int, prefill_rows: bool) -> bool:
    return bool(
        envs.VLLM_TPU_QSA_PREFILL_RING_PRODUCER
        and prefill_rows
        and ratio == 4
        and keys.ndim == 2
        and keys.dtype == jnp.bfloat16
        and keys.shape[1] % 128 == 0
        and len(keys) % 128 == 0
    )


def epi_fold() -> bool:
    return bool(envs.VLLM_TPU_QSA_PREFILL_EPI_FOLD)


def fold_mask(live, x):
    """G3's inner mask (``where(live, x, 0)``), applied before the walked branch's final RNE."""
    return jnp.where(live, x, 0)


# ----------------------------------------------------------------------------------------------------------------------
# Kernels. Token t (request r = req[t], slot s = safe_slots[r], logical position L = context[r] + t - starts[r]):
# G2 RING-PRODUCER, phase p = L & 3, half h = L & 4; token t owns output rows 4t + j, j = 0..3:
#   current[4t + j]  = raw[clip(t - p + j)]          (chunk_indices = t - p + j: an integer identity)
#   previous[4t + j] = ring[s, h + j]                ((group_start + j) % 8 = h + j)
#   old_position[t]  = tags[s, h, 0]                 (group_start % 8 = h)
# G2b VALUE-SHIFT, r8 = L & 7; token t owns rows 2t + j, j = 0, 1: values[2t + j] = keys[clip(t + 3 - r8 + 4 j)].
# Rows 2k, 2k + 1 of a bf16 array share one 32-bit word per lane (pltpu.bitcast's pairing): the kernels build each
# output word from 32-bit images of the bf16 rows (bits << 16; built once, at grid step 0, in a scratch padded by the
# edge rows, so clip() is a plain shifted read), never through a float rounding, and bitcast 32-bit blocks to the bf16
# output tiles (the producer stores its two words per token at sublane stride 2 first). ``previous``'s words are the
# ring's own words (8 s + h is even). Requests partition the tokens: r owns [starts[r], starts[r + 1]), request 0
# everything below starts[1] and the last slot everything from starts[S - 1] on (= _token_metadata's req, clipped).

_PAD = 8
_HALF_BITS = 16


def _req_of(starts_ref, t, slots: int):
    c = jnp.int32(0)
    for k in range(1, slots + 1):
        c = c + (starts_ref[k] <= t).astype(jnp.int32)
    return jnp.minimum(c, slots - 1)


def _req_bounds(starts_ref, r, slots: int):
    imin, imax = jnp.int32(-(2**31)), jnp.int32(2**31 - 1)
    lo = jnp.where(r > 0, starts_ref[r], imin)
    hi = jnp.where(r < slots - 1, starts_ref[jnp.minimum(r + 1, slots)], imax)
    return lo, hi


def _row_image(src_ref, ru_ref, tokens: int):
    """ru[_PAD + x] = the 32-bit image (bits << 16) of src[clip(x, 0, tokens - 1)], x in [-_PAD, tokens + _PAD)."""
    lanes = src_ref.shape[1]
    chunk = 128

    def body(k, carry):
        r0 = pl.multiple_of(k * chunk, chunk)
        x = src_ref[pl.ds(r0, chunk), :].astype(jnp.float32)   # bf16 -> f32 is the bit shift (exact for every pattern)
        ru_ref[pl.ds(r0 + _PAD, chunk), :] = lax.bitcast_convert_type(x, jnp.uint32)
        return carry

    lax.fori_loop(0, tokens // chunk, body, 0)
    head = ru_ref[pl.ds(_PAD, 8), :]
    ru_ref[pl.ds(0, _PAD), :] = jnp.broadcast_to(head[0:1], (_PAD, lanes))
    tail = ru_ref[pl.ds(tokens, 8), :]   # image rows of src[tokens - 8 .. tokens - 1]
    ru_ref[pl.ds(tokens + _PAD, _PAD), :] = jnp.broadcast_to(tail[7:8], (_PAD, lanes))


def _views(ru_ref, t0, bt: int, ds):
    """{d: V_d} with V_d[k] = image of src[clip(t0 + k + d)], |d| <= 8, from one aligned window and sublane rolls."""
    base = pl.multiple_of(t0 + _PAD - 8, 8)
    win = ru_ref[pl.ds(base, bt + 16), :]   # win[8 + d + k] = image of src[clip(t0 + k + d)]
    out = {}
    for d in ds:
        sh = (bt + 16 - (8 + d)) % (bt + 16)
        out[d] = (pltpu.roll(win, sh, 0) if sh else win)[:bt]
    return out


def _pack(lo, hi):
    """One 32-bit word from two row images: low half = lo's bf16 bits, high half = hi's."""
    return (lo >> _HALF_BITS) | (hi & jnp.uint32(_HI))


def _per_request(starts_ref, ctx_ref, t0, tv, slots: int, body, init, axis=0):
    """Folds ``body(r, mask, logical, carry)`` over the requests overlapping tokens [t0, t0 + bt) (tokens along ``axis``)."""
    r_lo = _req_of(starts_ref, t0, slots)
    r_hi = _req_of(starts_ref, t0 + tv.shape[axis] - 1, slots)

    def step(r, carry):
        lo, hi = _req_bounds(starts_ref, r, slots)
        m = (tv >= lo) & (tv < hi)
        logical = ctx_ref[r] + tv - starts_ref[r]
        return body(r, m, logical, carry)

    return lax.fori_loop(r_lo, r_hi + 1, step, init)


def _ring_producer_kernel(starts_ref, ctx_ref, slot_ref, tags_ref, raw_ref, ring_ref, cur_ref, prev_ref, old_ref,
                          ru_ref, rw_ref, cst_ref, pst_ref, *, tokens, slots, bt, rows, wide):
    i = pl.program_id(0)
    lanes = raw_ref.shape[1]

    @pl.when(i == 0)
    def _prologue():
        _row_image(raw_ref, ru_ref, tokens)
        rw_ref[...] = pltpu.bitcast(ring_ref[...], jnp.uint32)
        # old_position for every token, tokens along lanes ([tokens / 128, 128]); tags_ref = [tags[:, 0, 0], tags[:, 4, 0]]
        tl = (lax.broadcasted_iota(jnp.int32, old_ref.shape, 0) * 128
              + lax.broadcasted_iota(jnp.int32, old_ref.shape, 1))
        out = jnp.zeros(old_ref.shape, jnp.int32)
        for r in range(slots):
            lo, hi = _req_bounds(starts_ref, jnp.int32(r), slots)
            m = (tl >= lo) & (tl < hi)
            logical = ctx_ref[r] + tl - starts_ref[r]
            s = slot_ref[r]
            out = jnp.where(m, jnp.where((logical & 4) == 0, tags_ref[s], tags_ref[slots + s]), out)
        old_ref[...] = out

    t0 = pl.multiple_of(i * bt, bt)
    tv = t0 + lax.broadcasted_iota(jnp.int32, (bt, lanes), 0)

    def body(r, m, logical, carry):
        phase, qlo, qhi = carry
        w = slot_ref[r] * (rows // 2)
        h0 = (logical & 4) == 0
        phase = jnp.where(m, logical & 3, phase)
        qlo = jnp.where(m, jnp.where(h0, rw_ref[pl.ds(w, 1), :], rw_ref[pl.ds(w + 2, 1), :]), qlo)
        qhi = jnp.where(m, jnp.where(h0, rw_ref[pl.ds(w + 1, 1), :], rw_ref[pl.ds(w + 3, 1), :]), qhi)
        return phase, qlo, qhi

    zeros_u = jnp.zeros((bt, lanes), jnp.uint32)
    phase, qlo, qhi = _per_request(starts_ref, ctx_ref, t0, tv, slots, body,
                                   (jnp.zeros((bt, lanes), jnp.int32), zeros_u, zeros_u))
    v = _views(ru_ref, t0, bt, range(-3, 4))
    pair = {d: _pack(v[d], v[d + 1]) for d in range(-3, 3)}
    clo, chi = pair[0], pair[2]
    for p in range(1, 4):
        clo = jnp.where(phase == p, pair[-p], clo)
        chi = jnp.where(phase == p, pair[2 - p], chi)

    if wide:   # harness mutant M_G2SHAPE only (a producer in another shape): member j in lanes [128 j, 128 j + 128)
        members = []
        for j in range(4):
            mj = v[j]
            for p in range(1, 4):
                mj = jnp.where(phase == p, v[j - p], mj)
            members.append(lax.bitcast_convert_type(mj, jnp.float32).astype(jnp.bfloat16))
        cur_ref[...] = jnp.concatenate(members, axis=1)
        prev_ref[...] = jnp.zeros(prev_ref.shape, prev_ref.dtype)
        return
    cst_ref[pl.ds(0, bt, stride=2), :] = clo
    cst_ref[pl.ds(1, bt, stride=2), :] = chi
    pst_ref[pl.ds(0, bt, stride=2), :] = qlo
    pst_ref[pl.ds(1, bt, stride=2), :] = qhi
    cur_ref[...] = pltpu.bitcast(cst_ref[...], jnp.bfloat16)
    prev_ref[...] = pltpu.bitcast(pst_ref[...], jnp.bfloat16)


def _block_tokens(tokens: int) -> int:
    for bt in (256, 128):
        if tokens % bt == 0:
            return bt
    raise ValueError("GLUEB kernels: tokens must be a multiple of 128")


def ring_producer(raw, ring, tags, starts, context, safe_slots, *, wide: bool = False):
    """``(current [T, 4, D], previous [T, 4, D], old_position [T])`` of the ring update, exactly (see above)."""
    tokens, lanes = raw.shape
    slots, rows = ring.shape[0], ring.shape[1]
    bt = _block_tokens(tokens)
    kernel = functools.partial(_ring_producer_kernel, tokens=tokens, slots=slots, bt=bt, rows=rows, wide=wide)
    member_shape = (tokens, 4 * lanes) if wide else (4 * tokens, lanes)
    member_block = (bt, 4 * lanes) if wide else (4 * bt, lanes)
    # The two old-position candidates per slot (group_start % 8 in {0, 4}) as one [2 S] vector: a fusion, no relayout.
    tag_pick = jnp.concatenate([tags[:, 0, 0], tags[:, 4, 0]]).astype(jnp.int32)
    cur, prev, old = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(tokens // bt,),
            in_specs=[
                pl.BlockSpec((tokens, lanes), lambda i, *_: (0, 0)),
                pl.BlockSpec((slots * rows, lanes), lambda i, *_: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec(member_block, lambda i, *_: (i, 0)),
                pl.BlockSpec(member_block, lambda i, *_: (i, 0)),
                pl.BlockSpec((tokens // 128, 128), lambda i, *_: (0, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((tokens + 2 * _PAD, lanes), jnp.uint32),
                pltpu.VMEM((slots * rows // 2, lanes), jnp.uint32),
                pltpu.VMEM((2 * bt, lanes), jnp.uint32),
                pltpu.VMEM((2 * bt, lanes), jnp.uint32),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct(member_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(member_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens // 128, 128), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=48 * 1024 * 1024),
        name="q38_glueb_ring_producer",
    )(
        starts.astype(jnp.int32),
        context.astype(jnp.int32),
        safe_slots.astype(jnp.int32),
        tag_pick,
        raw,
        ring.reshape(slots * rows, lanes),
    )
    return cur.reshape(tokens, 4, lanes), prev.reshape(tokens, 4, lanes), old.reshape(tokens)


def _value_shift_kernel(starts_ref, ctx_ref, keys_ref, out_ref, ru_ref, *, tokens, slots, bt):
    i = pl.program_id(0)
    lanes = keys_ref.shape[1]

    @pl.when(i == 0)
    def _prologue():
        _row_image(keys_ref, ru_ref, tokens)

    t0 = pl.multiple_of(i * bt, bt)
    tv = t0 + lax.broadcasted_iota(jnp.int32, (bt, lanes), 0)

    def body(r, m, logical, r8):
        return jnp.where(m, logical & 7, r8)

    r8 = _per_request(starts_ref, ctx_ref, t0, tv, slots, body, jnp.zeros((bt, lanes), jnp.int32))
    v = _views(ru_ref, t0, bt, range(-4, 8))
    word = _pack(v[3], v[7])
    for r in range(1, 8):
        word = jnp.where(r8 == r, _pack(v[3 - r], v[7 - r]), word)
    out_ref[...] = pltpu.bitcast(word, jnp.bfloat16)


# G2b form: "tmin" (default) = the kernel reads keys token-minor (keys.T: the layout XLA gives the index-key rope, so nothing is
# relaid out for it); "rows" = row-major keys (_value_shift_rows; pulls the key rope row-major, DEV gbd2 F2).
G2B_FORM = "tmin"


def _value_shift_t_kernel(starts_ref, ctx_ref, kt_ref, out_ref, ru_ref, *, tokens, slots, bt):
    """G2b on keys.T ([D, T], tokens along lanes: the layout XLA gives the index-key rope, so no relayout feeds the kernel):
    the same words as _value_shift_kernel, built [D, bt] per block from lane-rolled 32-bit images, then transposed to
    [bt, D] and bitcast to the [2 bt, D] bf16 tile."""
    i = pl.program_id(0)
    feats = kt_ref.shape[0]
    lp = 128

    @pl.when(i == 0)
    def _prologue():
        def body(k, carry):
            c0 = pl.multiple_of(k * 128, 128)
            x = kt_ref[:, pl.ds(c0, 128)].astype(jnp.float32)
            ru_ref[:, pl.ds(c0 + lp, 128)] = lax.bitcast_convert_type(x, jnp.uint32)
            return carry

        lax.fori_loop(0, tokens // 128, body, 0)
        head = ru_ref[:, pl.ds(lp, 128)]
        ru_ref[:, pl.ds(0, lp)] = jnp.broadcast_to(head[:, 0:1], (feats, lp))
        tail = ru_ref[:, pl.ds(tokens, 128)]   # images of keys[tokens - 128 .. tokens - 1]
        ru_ref[:, pl.ds(tokens + lp, lp)] = jnp.broadcast_to(tail[:, 127:128], (feats, lp))

    t0 = pl.multiple_of(i * bt, bt)
    tv = t0 + lax.broadcasted_iota(jnp.int32, (feats, bt), 1)

    def body(r, m, logical, r8):
        return jnp.where(m, logical & 7, r8)

    r8 = _per_request(starts_ref, ctx_ref, t0, tv, slots, body, jnp.zeros((feats, bt), jnp.int32), axis=1)
    win = ru_ref[:, pl.ds(t0, bt + 2 * lp)]   # win[:, lp + k + d] = image of keys[clip(t0 + k + d)]
    wn = bt + 2 * lp
    v = {}
    for d in range(-4, 8):
        sh = (wn - d) % wn
        v[d] = (pltpu.roll(win, sh, 1) if sh else win)[:, lp:lp + bt]
    word = _pack(v[3], v[7])
    for r in range(1, 8):
        word = jnp.where(r8 == r, _pack(v[3 - r], v[7 - r]), word)
    out_ref[...] = pltpu.bitcast(word.T, jnp.bfloat16)


def value_shift(keys, starts, context):
    """``compressed_pairs``' values ``keys[clip(local)]`` ([T, 2, D]) for ratio 4, exactly (see above)."""
    if G2B_FORM == "tmin":
        return _value_shift_t(keys, starts, context)
    return _value_shift_rows(keys, starts, context)


def _value_shift_rows(keys, starts, context):
    """G2b on row-major keys: XLA then lays the index-key rope row-major too (lane-sparse [T, 32] f32 fusions and 4 c / s
    relayout copies, DEV gbd2 F2), so the default form is the token-minor one (_value_shift_t)."""
    tokens, lanes = keys.shape
    slots = len(context)
    bt = _block_tokens(tokens)
    out = pl.pallas_call(
        functools.partial(_value_shift_kernel, tokens=tokens, slots=slots, bt=bt),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(tokens // bt,),
            in_specs=[pl.BlockSpec((tokens, lanes), lambda i, *_: (0, 0))],
            out_specs=pl.BlockSpec((2 * bt, lanes), lambda i, *_: (i, 0)),
            scratch_shapes=[pltpu.VMEM((tokens + 2 * _PAD, lanes), jnp.uint32)],
        ),
        out_shape=jax.ShapeDtypeStruct((2 * tokens, lanes), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=48 * 1024 * 1024),
        name="q38_glueb_value_shift",
    )(starts.astype(jnp.int32), context.astype(jnp.int32), keys)
    return out.reshape(tokens, 2, lanes)


def _value_shift_t(keys, starts, context):
    tokens, feats = keys.shape
    slots = len(context)
    bt = _block_tokens(tokens)
    out = pl.pallas_call(
        functools.partial(_value_shift_t_kernel, tokens=tokens, slots=slots, bt=bt),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(tokens // bt,),
            in_specs=[pl.BlockSpec((feats, tokens), lambda i, *_: (0, 0))],
            out_specs=pl.BlockSpec((2 * bt, feats), lambda i, *_: (i, 0)),
            scratch_shapes=[pltpu.VMEM((feats, tokens + 256), jnp.uint32)],
        ),
        out_shape=jax.ShapeDtypeStruct((2 * tokens, feats), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=48 * 1024 * 1024),
        name="q38_glueb_value_shift",
    )(starts.astype(jnp.int32), context.astype(jnp.int32), keys.T)
    return out.reshape(tokens, 2, feats)
