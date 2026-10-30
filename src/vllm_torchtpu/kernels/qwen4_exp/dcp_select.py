# SPDX-License-Identifier: Apache-2.0
"""Exact global QSA top-k from a DCP pair's owned local top-k lists.

Under ``VLLM_TPU_QSA_COMPRESSED_DCP`` each rank of a pair scores only the
compressed groups it owns and keeps a local top-``k``. Every global winner is
in its owner's local list (at most ``k - 1`` owned keys beat it there), so the
pair's ``2k`` candidates hold the global top-``k``. ``merge_threshold`` finds
the per-row boundary ``(value, position)`` of that top-``k`` and
``keep_mask`` keeps a rank's local entries on the winning side of it.

The order matches the replicated indexer bit for bit: the SparseCore top-k
compares raw f32 bits through a monotone key (so -0.0 ranks below +0.0) and
breaks boundary ties toward the lower group id. Empty slots carry position
``EMPTY`` (int32 max) and are never kept; a row with fewer than ``k`` valid
candidates keeps them all.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import BITS_NEG_INF

EMPTY = jnp.iinfo(jnp.int32).max
# Every position of a row is below this: the "keep every tie" boundary.
ALL_TIES = EMPTY - 1


def monotone_key(bits):
    """int32 key ordered like the f32 values of ``bits`` (raw-bit order)."""
    return jnp.bitwise_xor(bits, jnp.bitwise_and(jnp.right_shift(bits, 31), 0x7FFFFFFF))


def _sc_topk(bits, k):
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

    return sparsecore_topk(bits, k, write_empty_rows=True)


def merge_threshold(vals_pair, pos_pair, k, *, topk=None):
    """Boundary of the global top-``k`` over the pair's local lists.

    ``vals_pair`` f32 ``[D, T, k]`` and ``pos_pair`` int32 ``[D, T, k]`` (global
    group ids, ``EMPTY`` for an empty slot). Returns ``(thr_val f32 [T],
    thr_pos int32 [T])`` for :func:`keep_mask`. ``topk(bits, k)`` returns the
    slots of ``k`` largest raw-bit scores per row (``-1`` padded); it defaults
    to the SparseCore top-k and only fixes the threshold value, so its own tie
    order does not matter.
    """
    topk = topk or _sc_topk
    d, t, width = vals_pair.shape
    vals = jnp.moveaxis(vals_pair, 0, 1).reshape(t, d * width)
    pos = jnp.moveaxis(pos_pair, 0, 1).reshape(t, d * width)
    filled = pos != EMPTY
    bits = jnp.where(filled, jax.lax.bitcast_convert_type(vals, jnp.int32), BITS_NEG_INF)
    key = monotone_key(bits)
    slots = topk(bits, k)
    taken = slots >= 0
    big = jnp.iinfo(jnp.int32).max
    thr_key = jnp.min(
        jnp.where(taken, jnp.take_along_axis(key, jnp.maximum(slots, 0), axis=1), big),
        axis=1,
    )
    full = jnp.sum(taken, axis=1) >= k
    strict = filled & (key > thr_key[:, None])
    tied = filled & (key == thr_key[:, None])
    quota = k - jnp.sum(strict, axis=1)
    ambiguous = full & (jnp.sum(tied, axis=1) > quota)

    def lowest_ties(_):
        # The quota-th smallest tied position, one bit at a time.
        def bit(i, cut):
            trial = cut + jnp.left_shift(1, 30 - i)
            below = jnp.sum(tied & (pos < trial[:, None]), axis=1)
            return jnp.where(below < quota, trial, cut)

        return jax.lax.fori_loop(0, 31, bit, jnp.zeros_like(quota))

    cut = jax.lax.cond(
        jnp.any(ambiguous), lowest_ties, lambda _: jnp.zeros_like(quota), None
    )
    thr_pos = jnp.where(ambiguous, cut, ALL_TIES)
    # Fewer than k valid candidates: keep every valid one.
    thr_bits = jnp.where(full, monotone_key(thr_key), BITS_NEG_INF)
    thr_pos = jnp.where(full, thr_pos, ALL_TIES)
    return jax.lax.bitcast_convert_type(thr_bits, jnp.float32), thr_pos


def keep_mask(vals, pos, thr_val, thr_pos):
    """bool ``[T, k]``: which local entries are in the global top-``k``."""
    key = monotone_key(jax.lax.bitcast_convert_type(vals, jnp.int32))
    thr = monotone_key(jax.lax.bitcast_convert_type(thr_val, jnp.int32))[:, None]
    return (pos != EMPTY) & (
        (key > thr) | ((key == thr) & (pos <= thr_pos[:, None]))
    )
