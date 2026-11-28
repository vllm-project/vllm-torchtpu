# SPDX-License-Identifier: Apache-2.0
# Provenance: copied from src/vllm_torchtpu/kernels/qwen4_exp/qsa_union_plan.py
#   at 2b42ba7260191af5132bacd7c030065e5dbf8bd0.
#   Owned by models/qwen38_flash from this copy on.
"""Sort-free union plan for the D2 union prefill (prototype).

``qsa_union_prefill.union_plan`` builds each query group's union with
``jnp.sort`` and ranks selections with ``searchsorted``. On TPU an XLA sort is
a bitonic network over ``group * (K + 1)`` keys per group, and the
``searchsorted`` reads a ``[T, union_cap]`` repeat of the union.
``union_plan_nosort`` returns the same four arrays from bitsets instead:

1. the served Pallas selection bitset (``_selection_bitset``) with 128-row
   query blocks, one query per lane, tail groups included;
2. an OR over each group's lanes: the union bitset, ``[T // group, words]``;
3. per-word popcounts and their prefix sum: the rank of an id ``x`` is the
   number of union ids below it, ``before[x // 32] + popcount(word & below)``,
   which is ``searchsorted(union, x, side="left")`` without a search;
4. the union list: every non-empty word writes its index at its first slot
   (unique indices), a cumulative max spreads it over the word's slots, and
   each slot picks its bit with a five-step popcount bisection.

Every op is a cumsum, a cumulative max, elementwise bit arithmetic, a gather,
or a scatter with unique indices, so nothing lowers to a sort.

Semantic differences from ``union_plan`` (all outside the served contract):

- Ids at or above ``groups_cap`` are invalid, as in ``_selection_words``:
  they never enter the union and their ``vblocks`` entry is -1. A live row
  whose tail id (``length // ratio``) is at or above ``groups_cap`` gets
  ``tail_rank = usize``. ``union_plan`` keeps such ids.
- Each row's valid ``blocks`` ids must be distinct. The bitset kernel adds
  bits, so a repeated id carries into the next bit (the served block-sparse
  kernel's mask has the same precondition). The one repeat that does occur,
  a row that selects its own partial tail group, is handled here: that row's
  tail is dropped from the bitset because its selection already sets the bit.
- When a union exceeds ``union_cap`` the outputs follow ``union_plan``'s
  truncation (the ``union_cap`` smallest ids; ranks clamp at ``union_cap``)
  and ``usize`` is the full union size.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..kernels import (
    block_sparse_gqa_prefill as bsp,
)

_LANES = 128


def _kth_set_bit(word, k):
    """Position of the ``k``-th (0-based, from bit 0) set bit of each uint32 ``word``.

    Elementwise bisection: each step counts the set bits in the lower half of
    the current window and moves to the upper half when ``k`` is past them.
    Meaningful only where ``k < popcount(word)``.
    """
    pos = jnp.zeros(k.shape, jnp.int32)
    for width in (16, 8, 4, 2, 1):
        low = lax.population_count(
            (word >> pos.astype(jnp.uint32)) & jnp.uint32((1 << width) - 1)
        ).astype(jnp.int32)
        upper = k >= low
        k = jnp.where(upper, k - low, k)
        pos = jnp.where(upper, pos + width, pos)
    return pos


def _popcount(x):
    """Set bits of each int32 (SWAR; logical shifts, no population-count op)."""
    x = x - (lax.shift_right_logical(x, 1) & 0x55555555)
    x = (x & 0x33333333) + (lax.shift_right_logical(x, 2) & 0x33333333)
    x = (x + lax.shift_right_logical(x, 4)) & 0x0F0F0F0F
    x = x + lax.shift_right_logical(x, 8)
    x = x + lax.shift_right_logical(x, 16)
    return x & 0x3F


def _rank_kernel(first_ref, end_ref, query_ref, union_ref, before_ref, out_ref):
    """Rank of every query id among its row's group union, eight words a step.

    The bitset kernel's layout and loop: query ids on sublanes and rows on
    lanes; the union and prefix-count words on sublanes, each lane holding its
    row's group. A query in word ``w`` takes ``before[w] + popcount(union[w] &
    below(id))``; queries of -1 never match and stay 0. Each step compares
    every query of the block against eight word indices, so no gather is
    needed.
    """
    block = pl.program_id(0)
    rows, width = query_ref.shape[1:]
    out_ref[...] = jnp.zeros(out_ref.shape, jnp.int32)

    def words(index, _):
        base = pl.multiple_of(index * 8, 8)
        union = union_ref[0, pl.ds(base, 8), :]
        before = before_ref[0, pl.ds(base, 8), :]

        def chunk(step, _):
            start = pl.multiple_of(step * 8, 8)
            query = query_ref[0, pl.ds(start, 8), :]
            word = query >> 5
            below = (jnp.int32(1) << (query & 31)) - 1
            rank = out_ref[0, pl.ds(start, 8), :]
            for k in range(8):
                hit = word == base + k
                count = before[k : k + 1, :] + _popcount(union[k : k + 1, :] & below)
                rank = jnp.where(hit, count, rank)
            out_ref[0, pl.ds(start, 8), :] = rank
            return None

        lax.fori_loop(0, rows // 8, chunk, None)
        return None

    lax.fori_loop(first_ref[block], end_ref[block], words, None)


def _rank_pass(query, ubits, before, *, group):
    """int32[T, Q] ranks of ``query`` ids (-1 = skip) by ``_rank_kernel``: 128 rows per
    block.
    """
    tokens, width_q = query.shape
    ngroups, words = before.shape
    padded = -(-tokens // _LANES) * _LANES
    nblocks = padded // _LANES
    rows = -(-width_q // 8) * 8
    q = jnp.pad(query, ((0, padded - tokens), (0, rows - width_q)), constant_values=-1)
    q = q.reshape(nblocks, _LANES, rows).transpose(0, 2, 1)

    def per_lane(table):
        lanes = jnp.repeat(table, group, axis=0)
        lanes = jnp.pad(lanes, ((0, padded - lanes.shape[0]), (0, 0)))
        return lanes.reshape(nblocks, _LANES, words).transpose(0, 2, 1)

    hit_words = jnp.where(q >= 0, q >> 5, -1).reshape(nblocks, -1)
    lowest = jnp.min(jnp.where(hit_words >= 0, hit_words, words), axis=1)
    highest = jnp.max(hit_words, axis=1)
    first = jnp.minimum(lowest, words) // 8
    end = jnp.where(highest >= 0, highest // 8 + 1, 0)
    out = pl.pallas_call(
        _rank_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(nblocks,),
            in_specs=[
                pl.BlockSpec((1, rows, _LANES), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((1, words, _LANES), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((1, words, _LANES), lambda b, *_: (b, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, rows, _LANES), lambda b, *_: (b, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        out_shape=jax.ShapeDtypeStruct((nblocks, rows, _LANES), jnp.int32),
        name="qsa_union_rank",
    )(
        first,
        end,
        q,
        per_lane(lax.bitcast_convert_type(ubits, jnp.int32)),
        per_lane(before),
    )
    return out.transpose(0, 2, 1).reshape(padded, rows)[:tokens, :width_q]


@functools.partial(
    jax.jit,
    static_argnames=(
        "group",
        "union_cap",
        "groups_cap",
        "ratio",
        "shift",
        "rank_kernel",
    ),
)
def union_plan_nosort(
    blocks,
    lengths,
    *,
    group,
    union_cap,
    groups_cap,
    ratio=4,
    shift=0,
    rank_kernel=False,
):
    """``union_plan`` without sort, argsort, searchsorted or repeated-index scatters.

    blocks: int32[T, K] selected group ids (-1 padded); lengths: int32[T]
    (0 for padded rows); T % group == 0; valid ids are below ``groups_cap``.
    Returns ``union`` int32[T // group, union_cap] (ascending, -1 padded),
    ``vblocks`` int32[T, K], ``vlengths`` int32[T] and ``usize``
    int32[T // group], integer-equal to ``union_plan`` whenever every union
    fits in ``union_cap`` (see the module docstring for the exceptions).
    ``shift`` (test-only mutant) offsets every rank, as in ``union_plan``.
    ``rank_kernel``: rank the selections with the Pallas compare pass
    (``_rank_kernel``) instead of two flat XLA gathers of T * (K + 1) indices.
    """
    tokens, k = blocks.shape
    if tokens % group:
        raise ValueError("query groups must split the tokens")
    ngroups = tokens // group
    query_block = math.lcm(group, _LANES)
    padded = -(-tokens // query_block) * query_block
    words = -(-groups_cap // 32)
    words = -(-words // 8) * 8

    live = lengths > 0
    tail = lengths // ratio
    valid = (blocks >= 0) & (blocks < groups_cap) & live[:, None]
    sel = jnp.where(valid, blocks, -1)
    # The kernel adds bits: a row that also selected its own tail group would
    # add that bit twice, so its tail entry is dropped (length rounded down).
    tail_selected = jnp.any(sel == tail[:, None], axis=1)
    bit_lengths = jnp.where(tail_selected, lengths - lengths % ratio, lengths)
    bits = bsp._selection_bitset(
        jnp.pad(sel, ((0, padded - tokens), (0, 0)), constant_values=-1),
        jnp.pad(bit_lengths, (0, padded - tokens)),
        ratio=ratio,
        groups=groups_cap,
        words=words,
        query_block=query_block,
        width=query_block,
    )  # int32[padded // query_block, words, query_block], queries on lanes

    # Union bitset: OR over each group's ``group`` consecutive lanes.
    nblocks, segments = padded // query_block, query_block // group
    bits = lax.bitcast_convert_type(bits, jnp.uint32).reshape(
        nblocks, words, segments, group
    )
    ubits = lax.reduce(bits, np.uint32(0), lax.bitwise_or, (3,))
    ubits = ubits.transpose(0, 2, 1).reshape(nblocks * segments, words)[:ngroups]

    counts = lax.population_count(ubits).astype(jnp.int32)
    inclusive = jnp.cumsum(counts, axis=1, dtype=jnp.int32)
    before = inclusive - counts  # union ids in lower words
    usize = inclusive[:, -1]

    # Rank of every selection and of every row's tail id: union ids below it,
    # clamped at union_cap like a searchsorted over the truncated union.
    query = jnp.concatenate([sel, tail[:, None]], axis=1)
    inside = jnp.clip(query, 0, groups_cap - 1)
    row_group = jnp.arange(tokens, dtype=jnp.int32) // group
    flat = row_group[:, None] * words + inside // 32
    if rank_kernel:
        ranked = jnp.where(
            jnp.concatenate([valid, (tail < groups_cap)[:, None]], axis=1), query, -1
        )
        rank = _rank_pass(ranked, ubits, before, group=group)
    else:
        word = jnp.take(ubits.reshape(-1), flat, mode="clip")
        base = jnp.take(before.reshape(-1), flat, mode="clip")
        below = (jnp.uint32(1) << (inside % 32).astype(jnp.uint32)) - jnp.uint32(1)
        rank = base + lax.population_count(word & below).astype(jnp.int32)
    tail_rank = jnp.where(tail >= groups_cap, jnp.repeat(usize, group), rank[:, k])
    tail_rank = jnp.minimum(tail_rank, union_cap) + shift
    vblocks = jnp.where(valid, jnp.minimum(rank[:, :k], union_cap) + shift, -1)
    vlengths = jnp.where(live, tail_rank * ratio + lengths % ratio, 0).astype(jnp.int32)

    # Union list. Non-empty words start at distinct slots ``before``; the rest
    # (and starts past union_cap) go to distinct slots past union_cap.
    lanes = jnp.arange(words, dtype=jnp.int32)
    slots = jnp.where((counts > 0) & (before < union_cap), before, union_cap + lanes)
    rows = jnp.arange(ngroups, dtype=jnp.int32)[:, None]
    starts = (
        jnp.full((ngroups, union_cap + words), -1, jnp.int32)
        .at[rows, slots]
        .set(jnp.broadcast_to(lanes, (ngroups, words)), unique_indices=True)
    )
    slot_word = jnp.maximum(lax.cummax(starts[:, :union_cap], axis=1), 0)
    slot = jnp.arange(union_cap, dtype=jnp.int32)[None, :]
    slot_bits = jnp.take_along_axis(ubits, slot_word, axis=1, mode="clip")
    slot_base = jnp.take_along_axis(before, slot_word, axis=1, mode="clip")
    ids = slot_word * 32 + _kth_set_bit(slot_bits, slot - slot_base)
    union = jnp.where(slot < usize[:, None], ids, -1)
    return union, vblocks, vlengths, usize


def plan_selections(blocks, lengths, *, groups_cap, ratio=4, live=None, tail=None):
    """(sel, bit_lengths): the selections the bitset takes (ids below ``groups_cap`` of
    live rows, else -1) and the
    lengths it takes (a row's partial tail dropped where the row already selects its
    tail group, so no bit is added
    twice). ``live`` / ``tail``: the caller's ``lengths > 0`` / ``lengths // ratio``, if
    it has them (union_bits_plan
    passes its own, so its TensorCore plan lowers to the same program as before the
    SparseCore option)."""
    live = lengths > 0 if live is None else live
    tail = lengths // ratio if tail is None else tail
    valid = (blocks >= 0) & (blocks < groups_cap) & live[:, None]
    sel = jnp.where(valid, blocks, -1)
    tail_selected = jnp.any(sel == tail[:, None], axis=1)
    return sel, jnp.where(tail_selected, lengths - lengths % ratio, lengths)


@functools.partial(
    jax.jit, static_argnames=("group", "union_cap", "groups_cap", "ratio", "bitset")
)
def union_bits_plan(
    blocks, lengths, *, group, union_cap, groups_cap, ratio=4, bitset="tensorcore"
):
    """The plan the walking D2 kernel needs: no union list, no ranks, no virtual bitset.

    Returns ``bits`` int32[T // group, words, 128] (each row's selection bitset
    over real group ids, rows of a group on lanes 0..group-1, tail groups
    included), ``ubits`` int32[T // group, words] (the group's union),
    ``usize`` int32[T // group] and ``vlengths`` int32[T] (the virtual causal
    lengths of ``union_plan``: rank of the row's tail id times the ratio plus
    ``length % ratio``; one [T] gather pair). ``group`` must divide 128.
    Same id contract as ``union_plan_nosort``. ``bitset``: "tensorcore"
    (``bsp._selection_bitset``) or
    "sparsecore" (``qsa_union_sc_gather.union_sc_bits``, the same bits, off the
    TensorCore; TPU only).
    """
    tokens, k = blocks.shape
    if tokens % group or _LANES % group:
        raise ValueError("query groups must split the tokens and divide 128")
    ngroups = tokens // group
    padded = -(-tokens // _LANES) * _LANES
    words = -(-groups_cap // 32)
    words = -(-words // 8) * 8

    live = lengths > 0
    tail = lengths // ratio
    sel, bit_lengths = plan_selections(
        blocks, lengths, groups_cap=groups_cap, ratio=ratio, live=live, tail=tail
    )
    if bitset == "sparsecore":
        from ..kernels.qsa_union_sc_gather import (
            union_sc_bits,
        )

        bits = union_sc_bits(
            sel, bit_lengths, ratio=ratio, groups=groups_cap, words=words, group=group
        )
        ubits = lax.reduce(
            lax.bitcast_convert_type(bits[:, :, :group], jnp.uint32),
            np.uint32(0),
            lax.bitwise_or,
            (2,),
        )
    elif bitset == "tensorcore":
        packed = bsp._selection_bitset(
            jnp.pad(sel, ((0, padded - tokens), (0, 0)), constant_values=-1),
            jnp.pad(bit_lengths, (0, padded - tokens)),
            ratio=ratio,
            groups=groups_cap,
            words=words,
            query_block=_LANES,
            width=_LANES,
        )  # [padded // 128, words, 128]
        nblocks, segments = padded // _LANES, _LANES // group
        split = packed.reshape(nblocks, words, segments, group)
        ubits = lax.reduce(
            lax.bitcast_convert_type(split, jnp.uint32),
            np.uint32(0),
            lax.bitwise_or,
            (3,),
        )
        ubits = ubits.transpose(0, 2, 1).reshape(nblocks * segments, words)[:ngroups]
        bits = split.transpose(0, 2, 1, 3).reshape(nblocks * segments, words, group)[
            :ngroups
        ]
        bits = jnp.pad(bits, ((0, 0), (0, 0), (0, _LANES - group)))
    else:
        raise ValueError("bitset: 'tensorcore' or 'sparsecore'")

    counts = lax.population_count(ubits).astype(jnp.int32)
    inclusive = jnp.cumsum(counts, axis=1, dtype=jnp.int32)
    usize = inclusive[:, -1]
    before = inclusive - counts
    inside = jnp.clip(tail, 0, groups_cap - 1)
    flat = (jnp.arange(tokens, dtype=jnp.int32) // group) * words + inside // 32
    word = jnp.take(ubits.reshape(-1), flat, mode="clip")
    base = jnp.take(before.reshape(-1), flat, mode="clip")
    below = (jnp.uint32(1) << (inside % 32).astype(jnp.uint32)) - jnp.uint32(1)
    tail_rank = base + lax.population_count(word & below).astype(jnp.int32)
    tail_rank = jnp.where(tail >= groups_cap, jnp.repeat(usize, group), tail_rank)
    vlengths = jnp.where(
        live, jnp.minimum(tail_rank, union_cap) * ratio + lengths % ratio, 0
    )
    return (
        bits,
        lax.bitcast_convert_type(ubits, jnp.int32),
        usize,
        vlengths.astype(jnp.int32),
    )
