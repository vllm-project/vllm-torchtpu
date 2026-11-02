# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_FUSED_DCP_BLOCKS: ``pooled_qsa.compressed_dcp_blocks`` as one
TensorCore Pallas kernel, bit-identical.

Per row tile of TR rows (rows on lanes, the 2*k candidates on sublanes):
  1. key = monotone f32-bit key; invalid (unfilled or <= -inf key) -> INT_MIN
  2. radix select (31+1 bit passes): kth = k-th largest key over valid
  3. strict/tied counts -> quota; ambiguous rows -> binary search on pos for
     the quota-th smallest tied pos (lower group id wins), only if any row
     of the tile is ambiguous
  4. masked = keep ? pos : EMPTY; bitonic sort along candidates; first k
Bit-exact with pooled_qsa.compressed_dcp_blocks (SC top-k validity: a key
<= KEY_NEG_INF is never taken).
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

EMPTY = int(np.iinfo(np.int32).max)
ALL_TIES = EMPTY - 1
INT_MIN = int(np.iinfo(np.int32).min)
BITS_NEG_INF = int(np.array(-np.inf, np.float32).view(np.int32))
KEY_NEG_INF = int(np.int32(BITS_NEG_INF) ^ np.int32(0x7FFFFFFF))
SUB = 8


def _mono(bits):
    return jnp.bitwise_xor(bits, jnp.bitwise_and(jnp.right_shift(bits, 31), 0x7FFFFFFF))


def _count(mask):
    # mask (R8, 8, L) bool -> (1, L) int32 counts over the candidate axes
    m = mask.astype(jnp.int32)
    return jnp.sum(jnp.sum(m, axis=0), axis=0, keepdims=True)


def _sort(x):
    """Ascending bitonic sort of x (R8, 8, L) int32 along the flattened (R8*8) axis."""
    r8, s8, lanes = x.shape
    n = r8 * s8
    sub = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    row = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    gidx = row * s8 + sub
    kk = 2
    while kk <= n:
        j = kk // 2
        while j >= 1:
            if j >= s8:
                jj = j // s8
                nb = r8 // (2 * jj)
                x5 = x.reshape(nb, 2, jj, s8, lanes)
                a, b = x5[:, 0], x5[:, 1]
                lo, hi = jnp.minimum(a, b), jnp.maximum(a, b)
                if kk == n:
                    na, nb_ = lo, hi
                else:
                    blk = jax.lax.broadcasted_iota(jnp.int32, a.shape, 0)
                    asc = ((blk * (2 * j)) & kk) == 0
                    na, nb_ = jnp.where(asc, lo, hi), jnp.where(asc, hi, lo)
                x = jnp.stack([na, nb_], axis=1).reshape(r8, s8, lanes)
            else:
                up = pltpu.roll(x, s8 - j, 1)  # x[i + j]
                dn = pltpu.roll(x, j, 1)  # x[i - j]
                is_lo = (sub & j) == 0
                p = jnp.where(is_lo, up, dn)
                asc = (gidx & kk) == 0
                take_min = jnp.bitwise_xor(is_lo.astype(jnp.int32), asc.astype(jnp.int32)) == 0
                x = jnp.where(take_min, jnp.minimum(x, p), jnp.maximum(x, p))
            j //= 2
        kk *= 2
    return x


def _kernel(vals_ref, pos_ref, out_ref, *, k):
    d, tr, w = vals_ref.shape
    parts_b, parts_p = [], []
    for r in range(d):
        parts_b.append(pltpu.bitcast(vals_ref[r], jnp.int32).T)  # (w, tr)
        parts_p.append(pos_ref[r].T)
    bits = jnp.concatenate(parts_b, axis=0).reshape(d * w // SUB, SUB, tr)
    pos = jnp.concatenate(parts_p, axis=0).reshape(d * w // SUB, SUB, tr)
    filled = pos != EMPTY
    key = _mono(bits)  # keep_mask's key (filled entries)
    valid = filled & (key > KEY_NEG_INF)
    skey = jnp.where(valid, key, INT_MIN)

    # 2. k-th largest skey: unsigned-space binary search from the MSB.
    def bit(i, ans):
        cand = ans | jnp.left_shift(jnp.int32(1), 31 - i)
        thr = jnp.bitwise_xor(cand, INT_MIN)
        c = _count(skey >= thr[None])
        return jnp.where(c >= k, cand, ans)

    ans = jax.lax.fori_loop(0, 32, bit, jnp.zeros((1, tr), jnp.int32))
    kth = jnp.bitwise_xor(ans, INT_MIN)  # (1, tr)
    nvalid = _count(valid)
    full = nvalid >= k
    strict = filled & (key > kth[None])
    tied = filled & (key == kth[None])
    quota = k - _count(strict)
    ambiguous = full & (_count(tied) > quota)

    def lowest_ties():
        q = quota

        def tbit(i, cut):
            trial = cut + jnp.left_shift(jnp.int32(1), 30 - i)
            below = _count(tied & (pos < trial[None]))
            return jnp.where(below < q, trial, cut)

        # bits above the largest tied position can never be taken: skip them (exact).
        maxp = jnp.max(jnp.where(tied, pos, 0))
        nb = sum((maxp >= (1 << b)).astype(jnp.int32) for b in range(31))
        return jax.lax.fori_loop(31 - nb, 31, tbit, jnp.zeros((1, tr), jnp.int32))

    cut = jax.lax.cond(
        jnp.any(ambiguous), lowest_ties, lambda: jnp.zeros((1, tr), jnp.int32)
    )
    thr_pos = jnp.where(ambiguous & full, cut, ALL_TIES)
    fullb = jnp.broadcast_to(full[None], key.shape)
    keep_full = (key > kth[None]) | ((key == kth[None]) & (pos <= thr_pos[None]))
    keep = filled & ((fullb & keep_full) | (jnp.logical_not(fullb) & (key >= KEY_NEG_INF)))
    masked = jnp.where(keep, pos, EMPTY)
    srt = _sort(masked)
    first = srt.reshape(d * w, tr)[:k]  # (k, tr)
    first = jnp.where(first == EMPTY, -1, first)
    out_ref[...] = first.T


@functools.partial(jax.jit, static_argnames=("block_topk", "tr", "interpret"))
def fused_dcp_blocks(vals_pair, pos_pair, block_topk, *, tr=128, interpret=False):
    d, t, w = vals_pair.shape
    tp = -(-t // tr) * tr
    if tp != t:
        vals_pair = jnp.pad(vals_pair, ((0, 0), (0, tp - t), (0, 0)))
        pos_pair = jnp.pad(pos_pair, ((0, 0), (0, tp - t), (0, 0)), constant_values=EMPTY)
    out = pl.pallas_call(
        functools.partial(_kernel, k=block_topk),
        grid=(tp // tr,),
        in_specs=[
            pl.BlockSpec((d, tr, w), lambda i: (0, i, 0)),
            pl.BlockSpec((d, tr, w), lambda i: (0, i, 0)),
        ],
        out_specs=pl.BlockSpec((tr, block_topk), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((tp, block_topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(vals_pair, pos_pair)
    return out[:t]
