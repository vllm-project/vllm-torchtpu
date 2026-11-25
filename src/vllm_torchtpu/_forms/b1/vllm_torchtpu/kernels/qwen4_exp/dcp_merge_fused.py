# SPDX-License-Identifier: Apache-2.0
"""q38 IDXX SEL-1P(c): the compressed-DCP pair merge's threshold, tie cut and
keep mask in one Pallas TensorCore kernel (VLLM_TPU_QSA_DCP_FUSED_MERGE).

``dcp_select.merge_threshold`` (``from_scores``) + ``dcp_select.keep_mask`` run
as about ten XLA fusions per QSA layer plus, on a row whose top-k boundary is
tied past its quota, a 31-step ``fori_loop`` over every row. This kernel takes
the same operands -- the DCP ranks' gathered local lists and the SparseCore
top-k's (slots, picked raw bits) over them -- and writes the same result: the
kept group ids (``-1`` dropped) of one rank's list (``compressed_dcp_own_blocks``)
or of every rank's list (``compressed_dcp_blocks`` before its sort). The tie
bisection runs inside the kernel only on 8-row groups that hold an ambiguous
row. Integer arithmetic only, the same formulas: bit-identical by construction
(tests/kernels/test_q38_idxx_fused_merge.py checks it against dcp_select).
"""

import contextlib
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import BITS_NEG_INF
from vllm_torchtpu.kernels.qwen4_exp.dcp_select import ALL_TIES, EMPTY

_BIG = jnp.iinfo(jnp.int32).max


def _kscope(name):
    """ANNOT's kernel-interior scope (q38k:<group>.<phase>) when that module is
    in the tree; a no-op otherwise (and whenever its key is off)."""
    try:
        from vllm_torchtpu.tracing import decode_markers
    except ImportError:
        return contextlib.nullcontext()
    return decode_markers.kscope(name)
SUB = 8  # rows per tie-bisection group (one sublane tile)


def _monotone(bits):
    return jnp.bitwise_xor(bits, jnp.bitwise_and(jnp.right_shift(bits, 31), 0x7FFFFFFF))


def _row_sum(mask):
    return jnp.sum(mask.astype(jnp.int32), axis=1, keepdims=True)


def _kernel(rank_ref, bits_ref, pos_ref, slots_ref, picked_ref, out_ref,
            thr_ref, cut_ref, *, d, k, rows, own):
    """One block of ``rows`` rows. bits / pos [d, rows, w]; slots / picked
    [rows, k]; out [rows, w] (own) or [d, rows, w]."""
    neg_inf = jnp.int32(BITS_NEG_INF)
    with _kscope("qsa_select.thresh"):
        thr_key, full, quota, ambiguous = _threshold(
            bits_ref, pos_ref, slots_ref, picked_ref, d=d, k=k, rows=rows
        )
    thr_ref[...] = thr_key
    cut_ref[...] = jnp.zeros((rows, 1), jnp.int32)
    with _kscope("qsa_select.tie"):
        _ties(bits_ref, pos_ref, cut_ref, thr_key, quota, ambiguous, d=d, rows=rows)
    with _kscope("qsa_select.keep"):
        _keep(rank_ref, bits_ref, pos_ref, out_ref, cut_ref, thr_key, full, ambiguous,
              d=d, own=own)


def _threshold(bits_ref, pos_ref, slots_ref, picked_ref, *, d, k, rows):
    neg_inf = jnp.int32(BITS_NEG_INF)
    slots = slots_ref[...]
    taken = slots >= 0
    thr_key = jnp.min(jnp.where(taken, _monotone(picked_ref[...]), _BIG), axis=1, keepdims=True)
    full = _row_sum(taken) >= k
    strict = jnp.zeros((rows, 1), jnp.int32)
    tied = jnp.zeros((rows, 1), jnp.int32)
    for r in range(d):
        pos = pos_ref[r]
        filled = pos != EMPTY
        key = _monotone(jnp.where(filled, bits_ref[r], neg_inf))
        strict = strict + _row_sum(filled & (key > thr_key))
        tied = tied + _row_sum(filled & (key == thr_key))
    quota = k - strict
    ambiguous = full & (tied > quota)
    return thr_key, full, quota, ambiguous


def _ties(bits_ref, pos_ref, cut_ref, thr_key, quota, ambiguous, *, d, rows):
    neg_inf = jnp.int32(BITS_NEG_INF)

    # The quota-th smallest tied position, one bit at a time, on the 8-row
    # groups that hold an ambiguous row only (dcp_select.merge_threshold's
    # lowest_ties on those rows; other rows keep cut 0, unused).
    amb_i = ambiguous.astype(jnp.int32)
    for g in range(rows // SUB):
        lo = g * SUB
        amb_g = amb_i[lo : lo + SUB]
        quota_g = quota[lo : lo + SUB]
        thr_g = thr_key[lo : lo + SUB]

        @pl.when(jnp.max(amb_g) > 0)
        def _bisect(lo=lo, quota_g=quota_g, thr_g=thr_g):
            tied_masks = []
            for r in range(d):
                pos = pos_ref[r, pl.ds(lo, SUB), :]
                filled = pos != EMPTY
                key = _monotone(jnp.where(filled, bits_ref[r, pl.ds(lo, SUB), :], neg_inf))
                tied_masks.append((filled & (key == thr_g), pos))

            def bit(i, cut):
                trial = cut + jnp.left_shift(jnp.int32(1), 30 - i)
                below = jnp.zeros((SUB, 1), jnp.int32)
                for tmask, pos in tied_masks:
                    below = below + _row_sum(tmask & (pos < trial))
                return jnp.where(below < quota_g, trial, cut)

            cut_ref[pl.ds(lo, SUB), :] = jax.lax.fori_loop(
                0, 31, bit, jnp.zeros((SUB, 1), jnp.int32)
            )


def _keep(rank_ref, bits_ref, pos_ref, out_ref, cut_ref, thr_key, full, ambiguous, *, d, own):
    neg_inf = jnp.int32(BITS_NEG_INF)
    thr_pos = jnp.where(ambiguous, cut_ref[...], ALL_TIES)
    thr_pos = jnp.where(full, thr_pos, ALL_TIES)
    # keep_mask's threshold key: monotone(thr_val bits), thr_val bits =
    # monotone(thr_key) when full else -inf; monotone is an involution.
    thr = jnp.where(full, thr_key, _monotone(neg_inf))
    def kept(pos, bits):
        key = _monotone(bits)
        return (pos != EMPTY) & ((key > thr) | ((key == thr) & (pos <= thr_pos)))

    if own:
        # The rank's own list only (a dynamic leading-axis read).
        rank = rank_ref[0]
        pos = pos_ref[rank]
        out_ref[...] = jnp.where(kept(pos, bits_ref[rank]), pos, -1)
        return
    for r in range(d):
        pos = pos_ref[r]
        out_ref[r] = jnp.where(kept(pos, bits_ref[r]), pos, EMPTY)


def _block_rows(t):
    """Rows per grid step: whole 8-row tiles, at most 144 (one block at the
    B1 row shard / B3 rows; 288 -> 2 blocks)."""
    best = SUB
    for rows in range(SUB, min(t, 144) + 1, SUB):
        if t % rows == 0:
            best = rows
    return best


@functools.partial(jax.jit, static_argnames=("k", "own", "interpret"))
def fused_merge_keep(bits_pair, pos_pair, slots, picked, rank, *, k, own, interpret=False):
    """``bits_pair`` int32 [D, T, w] (f32 score bits of the D ranks' local
    lists), ``pos_pair`` int32 [D, T, w] (global group ids, ``EMPTY`` unfilled),
    ``slots`` / ``picked`` int32 [T, k] (the SparseCore top-k over the rows'
    ``D * w`` filled-masked bits: kept slots, ``-1`` padded, and their raw bits),
    ``rank`` int32 [1].

    ``own``: int32 [T, w], rank ``rank``'s list with dropped entries -1
    (``compressed_dcp_own_blocks``). Otherwise int32 [D, T, w], every list with
    dropped entries ``EMPTY`` (``compressed_dcp_blocks``' ``kept``)."""
    d, t, w = bits_pair.shape
    if t % SUB:
        pad = SUB - t % SUB
        bits_pair = jnp.pad(bits_pair, ((0, 0), (0, pad), (0, 0)))
        pos_pair = jnp.pad(pos_pair, ((0, 0), (0, pad), (0, 0)), constant_values=EMPTY)
        slots = jnp.pad(slots, ((0, pad), (0, 0)), constant_values=-1)
        picked = jnp.pad(picked, ((0, pad), (0, 0)))
    tp = bits_pair.shape[1]
    rows = _block_rows(tp)
    grid = (tp // rows,)
    lists = pl.BlockSpec((d, rows, w), lambda i, rank: (0, i, 0))
    sel = pl.BlockSpec((rows, k), lambda i, rank: (i, 0))
    if own:
        out_spec = pl.BlockSpec((rows, w), lambda i, rank: (i, 0))
        out_shape = jax.ShapeDtypeStruct((tp, w), jnp.int32)
    else:
        out_spec = pl.BlockSpec((d, rows, w), lambda i, rank: (0, i, 0))
        out_shape = jax.ShapeDtypeStruct((d, tp, w), jnp.int32)
    out = pl.pallas_call(
        functools.partial(_kernel, d=d, k=k, rows=rows, own=own),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=grid,
            in_specs=[lists, lists, sel, sel],
            out_specs=out_spec,
            scratch_shapes=[pltpu.VMEM((rows, 1), jnp.int32), pltpu.VMEM((rows, 1), jnp.int32)],
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=f"q38_idxx_dcp_fused_merge_d{d}_t{t}_w{w}_k{k}_{'own' if own else 'all'}",
    )(jnp.asarray(rank, jnp.int32).reshape(1), bits_pair, pos_pair, slots, picked)
    return out[:t] if own else out[:, :t]


def _gathered_rows(vals_pair, pos_pair):
    """merge_threshold's SparseCore operand: [T, D * w] raw bits, -inf unfilled."""
    d, t, w = vals_pair.shape
    vals = jnp.moveaxis(vals_pair, 0, 1).reshape(t, d * w)
    pos = jnp.moveaxis(pos_pair, 0, 1).reshape(t, d * w)
    return jnp.where(pos != EMPTY, jax.lax.bitcast_convert_type(vals, jnp.int32), BITS_NEG_INF)


def _sc_topk_scores(bits, k):
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

    return sparsecore_topk(bits, k, write_empty_rows=True, return_scores=True)


def fused_dcp_keep(vals_pair, pos_pair, k, rank=None, *, topk=None, interpret=False):
    """VLLM_TPU_QSA_DCP_FUSED_MERGE form of ``dcp_select.merge_threshold``
    (``from_scores``) + ``keep_mask``: the same SparseCore top-k over the
    pair's rows for the threshold value, then one kernel. ``rank`` given:
    ``compressed_dcp_own_blocks``' result (int32 [T, w], -1 dropped); else the
    kept ids of every list (int32 [D, T, w], ``EMPTY`` dropped)."""
    topk = topk or _sc_topk_scores
    slots, picked = topk(_gathered_rows(vals_pair, pos_pair), k)
    bits_pair = jax.lax.bitcast_convert_type(vals_pair, jnp.int32)
    own = rank is not None
    return fused_merge_keep(
        bits_pair, pos_pair.astype(jnp.int32), slots.astype(jnp.int32),
        picked.astype(jnp.int32), jnp.asarray(0 if rank is None else rank, jnp.int32).reshape(-1)[:1],
        k=k, own=own, interpret=interpret,
    )
