# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax Sparse Attention under decode context parallelism (DCP).

The ``dcp`` ranks that hold one replicated KV head split its context by
whole sparse blocks instead of each keeping all of it: with
``--cp-kv-cache-interleave-size`` equal to the block size, vLLM's logical
block ``j`` of a request covers ``dcp`` sparse blocks and rank ``r`` keeps
sparse block ``j * dcp + r`` in its page ``block_tables[j]``. So block ``b``
lives on rank ``b % dcp``, whole, and every block-level step stays local:

1. `msa_write_and_candidates_dcp`: each rank writes the K/V and index keys
   of its blocks' tokens, scores its blocks (the block max is over one
   block's keys, all on this rank) and keeps its local top-k candidates
   ``(score, block)`` under the selection's own order (forced blocks first,
   then score, ties to the lower block). The global top-k is a subset of the
   union of the ranks' local top-k, so exchanging ``dcp * topk`` candidates
   per token (an all-gather of ``[T, 2 * topk]`` f32, block ids as values)
   gives the exact non-DCP selection.
2. `msa_attend_dcp`: each rank merges the candidates into that selection,
   attends every query head of the KV head over the selected blocks it
   holds and returns the partial output with its log-sum-exp; the caller
   merges the ranks' partials (`attend_and_merge` runs both, or
   `ops.msa.merge_head_partials` merges over several KV heads).

A DCP group wider than a KV head's TP replicas (e.g. DCP 8 at TP 8 with 4
KV heads) spans ``G = dcp / replicas`` KV heads: every rank then stores all
``G`` heads' K/V rows (and the one shared index-key head) for its blocks,
and runs both steps per head; the caller all-gathers the new K/V and index
queries of the group's heads along with the query heads.
"""

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_fp4,
    msa_indexer,
    msa_select,
    msa_write,
)

_NEG_INF = float("-inf")
_LANES = 128
# Decode-sized batches (up to this many rows) pick in row blocks of a vreg's
# lanes, transposed, where that pads less than `msa_select.DEFAULT_BLOCK_ROWS`
# (c48 b36: 288 rows in 3 x 128, not 2 x 256: candidates 12.0 -> 3.4 us per
# layer, twins j-1b4fb146, j-ec9e7ff6; c48 b23's 184 rows in 2 x 128 were
# slower than in one block of 184, 4.6 -> 6.3 us at 136K context).
_DECODE_CANDIDATE_TOKENS = 1024

# Up to this many tokens (`msa_attention.SMALL_BATCH_TOKENS` is the non-DCP
# limit; every decode bucket) an 8-token verify group fills an attention tile
# alone: a 16-token tile holds two requests' disjoint page unions, so its
# groups score twice the rows over the same pages (AOT, 4-bit 16-page step:
# ~1,450 bundles at 8 tokens vs ~2,240 at 16). With the twins' selections
# dispersed as served (int4 index-scale fill), c48 b37 decode attention was
# 223 us per layer in 16-token tiles (j-52ca6e4b). Before that fill, the
# twins' tiles were nearly empty and 8-token ones lost on per-tile cost
# (c48 b39 108.4 -> 116.7 ms per step, j-82660a13 vs j-8a29ceb0).
DCP_SMALL_BATCH_TOKENS = 1024
# Pages per kernel step of those 8-token tiles: a tile's 128 query rows keep
# a 32-page score tile the size of a 16-token tile's 16-page one (AOT: 32
# pages ~3,000 bundles vs 2 x ~1,700 for two 16-page steps), and a tile pays
# a group step's fixed cost (~0.7 us) once per 32 pages, not 16 (c4 b3: -4.6
# us per call, c16 b6: -3.7, twins j-315ea1ce, j-c2aafd46). A step of one
# tile too, now that 8-bit pages enter the MXU as stored: c4 b1 decode
# attention 5.9 -> 5.3 us per call (twins j-e5b22c98, j-52a1417f; while
# pages were unpacked to bf16 it was +2.9 us, j-315ea1ce).
_SMALL_BATCH_PAGES_PER_STEP = 32


# Decode steps of up to this many tokens (a pair's 48 requests of 8 rows) run
# `_decode_plan_kernel` in the attention kernel, in place of the XLA work list
# (two sorts and ~20 small ops): anvil chain of one rank, 4-bit cache, region
# us at selection locality .97 / .5: 128 rows 42.3 -> 26.7 / 59.8 -> 49.4,
# 192 rows 51.9 -> 34.4 / 81.5 -> 73.5, 384 rows 74.4 -> 58.4 / 123.2 -> 120.8.
_PLAN_MAX_TOKENS = 384
# Page buffers a decode step's folded write may keep in VMEM (every run's
# pages, in and out): c48's 4-bit 128-row steps take ~5.6 MiB.
_FOLDED_WRITE_VMEM_BYTES = 16 << 20
# The f32 block scores `decode_candidates` keeps in VMEM (rows x this rank's
# pages): c48's 384-row steps at 256K tokens take 1.5 MiB.
_DECODE_SCORES_VMEM_BYTES = 16 << 20
# A pair's prefill-sized steps pass the attention their own and the peer's
# query heads as two arrays and swap back the peer's partials by a permute
# (`attend_and_merge`): no concatenated copy of the queries (a fusion of
# ~20 us per layer at 8K tokens in the c4 twin) and no all-to-all.
SPLIT_PAIR_QUERIES = True


def rank_array(rank: int | jax.Array) -> jax.Array:
    """The DCP rank as an ``int32[1]`` array.

    Callers pass it as a traced input (`ops.msa.get_msa_dcp_op` feeds a
    per-rank device buffer), so all ranks lower to one program: a Python
    ``int`` rank is baked into the HLO and gives every rank its own
    executable, compile and cache entry.
    """
    return jnp.asarray(rank, dtype=jnp.int32).reshape(1)


def owner_pages(
    block_tables_2d: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    valid: jax.Array,
    *,
    block_size: int,
    rank: int | jax.Array,
    dcp: int,
) -> jax.Array:
    """This rank's page of each token, dropped for tokens of other ranks."""
    block = pos // block_size
    page = block_tables_2d[seq_id, block // dcp]
    return jnp.where(valid & (block % dcp == rank), page, msa._DROPPED_PAGE)


def local_seq_lens(
    seq_lens: jax.Array, *, block_size: int, rank: int | jax.Array, dcp: int
) -> jax.Array:
    """A length whose page count is the number of blocks this rank holds:
    blocks ``rank, rank + dcp, ...`` below ``ceil(len / block_size)``."""
    blocks = -(-seq_lens // block_size)
    return jnp.maximum(blocks - rank + dcp - 1, 0) // dcp * block_size


def _candidates_kernel(
    rank_ref,
    scores_ref,
    pos_ref,
    score_out_ref,
    block_out_ref,
    *,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    dcp,
    col_bits,
):
    """`msa_select._kernel` over this rank's columns (column ``c`` is block
    ``c * dcp + rank``), also returning the picked scores, over the whole
    width. The rank is read from SMEM so that every rank runs the same
    program."""
    rank = rank_ref[0]
    rows, width = scores_ref.shape
    num_visible = pos_ref[...] // block_size + 1

    s = scores_ref[...]
    cols = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    blocks = cols * dcp + rank
    visible = blocks < num_visible
    s = jnp.where(visible, s, _NEG_INF)
    s = jnp.where(
        visible & (blocks >= num_visible - local_blocks),
        msa_select._FORCED_LOCAL_SCORE,
        s,
    )
    s = jnp.where(visible & (blocks < init_blocks), msa_select._FORCED_INIT_SCORE, s)
    # The scores the transposed kernel's keys keep, so both order alike.
    s = _key_scores(_score_keys(s, cols, col_bits), col_bits)[0]
    picked_scores, picked_blocks = [], []
    for _ in range(topk):
        best_score = jnp.max(s, axis=1, keepdims=True)
        best = jnp.min(jnp.where(s == best_score, cols, width), axis=1, keepdims=True)
        picked_scores.append(best_score)
        picked_blocks.append(jnp.where(best_score > _NEG_INF, best * dcp + rank, -1))
        s = jnp.where(cols == best, _NEG_INF, s)
    odd = (rank & 1) == 1  # picks in reverse (`local_candidates`)
    for out_ref, picked in (
        (score_out_ref, picked_scores),
        (block_out_ref, picked_blocks),
    ):
        out_ref[...] = jnp.where(
            odd,
            jnp.concatenate(picked[::-1], axis=1),
            jnp.concatenate(picked, axis=1),
        )


# `_candidates_kernel_t` picks on int32 keys that order the selection in one
# comparison: an order-preserving code of the score, shifted right one bit
# (keys span less than 2**31) with its low `_key_bits(num_cols)` bits holding
# the inverted column, so that the larger key is the higher score and, among
# equal kept scores, the lower column. Keys are distinct, so a pick is the
# largest key below the previous pick: ``max(key - (prev ^ INT_MIN))``, three
# VALU ops per vreg (an int32 max is a compare and a select) and no store
# (the f32 picks took five, a store and a second load: 56 -> 26 bundles per
# 32 vregs, AOT). The scores keep 11 mantissa
# bits (they come from fp8 / 4-bit keys and bf16 or e4m3 queries); every
# rank keys with the same bits, so a DCP group's lists merge exactly
# (`merge_pair_candidates`) on the scores the picks return. (The non-DCP
# `msa_select` orders full f32 scores: near-ties may resolve differently.)
_KEY_TOP = 1 << 30
_INT_MIN = -(1 << 31)
_PICK_ACCUMULATORS = 4


# Column bits of every key up to 2,048 columns (262,144 tokens of 128-token
# blocks on one rank), whatever the scores' padded width, so that the tile
# classes of a step and the token-order path keep the same scores.
_MIN_KEY_BITS = 11


def _key_bits(num_cols: int) -> int:
    return max(_MIN_KEY_BITS, (num_cols - 1).bit_length())


def _score_keys(s: jax.Array, cols: jax.Array, col_bits: int) -> jax.Array:
    """f32 scores ``s`` and their int32 column ids -> distinct int32 keys in
    ``[-2**30, 2**30)`` (see `_KEY_TOP`)."""
    low = (1 << col_bits) - 1
    code = pltpu.bitcast(s, jnp.int32)
    code = jnp.where(code < 0, code ^ 0x7FFFFFFF, code)
    return ((code >> 1) & ~low) | (low - cols)


def _key_scores(keys: jax.Array, col_bits: int) -> tuple[jax.Array, jax.Array]:
    """Keys -> (their kept f32 scores, -inf for -inf scores; columns)."""
    low = (1 << col_bits) - 1
    high = keys & ~low
    neg_inf_code = (-0x00800000 ^ 0x7FFFFFFF) >> 1  # -inf (0xFF800000): code >> 1
    code = high << 1
    bits = jnp.where(code < 0, code ^ 0x7FFFFFFF, code)
    score = jnp.where(
        high <= (neg_inf_code & ~low), _NEG_INF, pltpu.bitcast(bits, jnp.float32)
    )
    return score, low - (keys & low)


# `_candidates_kernel_t` builds keys `_KEY_CHUNK[pretransposed]` columns at a
# time; a pick scans them with `_PICK_ACCUMULATORS` running maxima (one per
# 8-sublane group), its loop unrolled `_PICK_UNROLL` times (AOT: 26 bundles
# per 32 vregs; 2 picks per scan in sorted pairs or 4 took 1.0-1.1 bundles
# per vreg and pick, as the insertions chain).
_KEY_CHUNK = {True: 32, False: _LANES}
_PICK_UNROLL = 4


def _candidates_kernel_t(
    rank_ref,
    scores_ref,
    pos_ref,
    score_out_ref,
    block_out_ref,
    key_ref,
    *,
    pretransposed=False,
    **kwargs,
):
    """`_candidates_kernel` with the row block transposed in VMEM: columns
    on sublanes, rows on lanes (``pos_ref [1, rows]``, outputs ``[topk,
    rows]``). Each pick's max reduces across vregs and sublanes on the VPU
    instead of across lanes on the XLU, which bound the row-major picks
    (~310 bundles per pick of a 256 x 128 block in the LLO), over packed keys
    (`_score_keys`). `pretransposed` scores already come as ``[columns,
    rows]``."""
    score_out_ref[...], block_out_ref[...] = _pick_t(
        rank_ref[0],
        scores_ref,
        pos_ref[...],
        key_ref,
        pretransposed=pretransposed,
        **kwargs,
    )


def _pick_t(
    rank,
    scores_ref,
    pos,
    key_ref,
    *,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    dcp,
    col_bits,
    real_cols,
    pretransposed=False,
    insert=False,
):
    """`_candidates_kernel_t`'s picks of one row block: ``(scores, blocks)``
    ``[topk, rows]`` for positions ``pos [1, rows]``.

    The block keys and scans only the columns its rows can see (the scores
    span max-model-len, mostly past every row's context): a loop bound read
    from its positions, one body for every width. With `insert` the keys
    are `_float_keys` (an f32 `key_ref`) picked in one pass
    (`_insert_picks`)."""
    num_cols = scores_ref.shape[0 if pretransposed else 1]
    rows = pos.shape[1]
    chunk = _KEY_CHUNK[pretransposed]
    num_visible = pos // block_size + 1  # [1, rows]
    # A rank's blocks below n are among its first ceil(n / dcp) columns.
    needed = (jnp.max(num_visible) + dcp - 1) // dcp
    chunks = jnp.clip((needed + chunk - 1) // chunk, 1, num_cols // chunk)

    def build(c, carry):
        start = pl.multiple_of(c * chunk, chunk)
        if pretransposed:
            s = scores_ref[pl.ds(start, chunk), :]
        else:
            s = scores_ref[:, pl.ds(start, chunk)].T
        cols = start + jax.lax.broadcasted_iota(jnp.int32, (chunk, rows), 0)
        keys = _column_keys(
            s,
            cols,
            num_visible,
            rank,
            dcp=dcp,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            col_bits=col_bits,
            real_cols=real_cols,
        )
        key_ref[pl.ds(start, chunk), :] = (
            _float_keys(keys, col_bits) if insert else keys
        )
        return carry

    jax.lax.fori_loop(0, chunks, build, 0)
    if not insert:
        picks = _scan_picks(key_ref, chunks * chunk, topk)
    elif chunk % (8 * _INSERT_UNROLL):
        raise NotImplementedError(f"{chunk}-column key chunks: no whole insert steps")
    else:
        keys = _insert_picks(key_ref, chunks * chunk, topk, col_bits)
        picks = [keys[i : i + 1] for i in range(topk)]
    return _picked(picks, rank, col_bits=col_bits, dcp=dcp)


def _scan_picks(key_ref, num_cols, topk) -> list[jax.Array]:
    """The `topk` largest keys of each column (row) of ``key_ref[:num_cols]``,
    ``[1, rows]`` each, best first; `num_cols` a traced multiple of 32.

    A pick is the largest key below the last (`_next_pick`): with
    `_PICK_ACCUMULATORS` running maxima, one per 8-sublane group, its loop
    unrolled `_PICK_UNROLL` times."""
    rows = key_ref.shape[1]
    sub = 8 * _PICK_ACCUMULATORS
    trips = num_cols // sub

    def scan(off, acc, v):
        x = key_ref[pl.ds(pl.multiple_of(v * sub, sub), sub), :]
        return tuple(
            jnp.maximum(a, x[8 * i : 8 * (i + 1)] - off) for i, a in enumerate(acc)
        )

    picks, prev = [], jnp.full((8, rows), _KEY_TOP, jnp.int32)
    for _ in range(topk):
        off = prev ^ _INT_MIN

        def unrolled(i, acc, off=off):
            for u in range(_PICK_UNROLL):
                acc = scan(off, acc, i * _PICK_UNROLL + u)
            return acc

        acc = (jnp.full((8, rows), _INT_MIN, jnp.int32),) * _PICK_ACCUMULATORS
        acc = jax.lax.fori_loop(0, trips // _PICK_UNROLL, unrolled, acc)
        acc = jax.lax.fori_loop(
            trips // _PICK_UNROLL * _PICK_UNROLL,
            trips,
            lambda v, acc, off=off: scan(off, acc, v),
            acc,
        )
        prev = _next_pick(acc, off)
        picks.append(prev[:1])
    return picks


def _float_keys(keys: jax.Array, col_bits: int) -> jax.Array:
    """Pick keys (`_score_keys`) as positive normal f32s of the same order:
    offset so that the -inf scores' keys start at the smallest normal, the
    keys of scores past ~2**126 (none: `msa_select`'s forced scores are
    2**99) saturated with their columns kept. f32 maxima and minima are one
    vector op each, where an int32 one is a compare and a select."""
    offset, top = _float_key_range(col_bits)
    low = (1 << col_bits) - 1
    keys = jnp.where(keys > top, (top & ~low) | (keys & low), keys)
    return pltpu.bitcast(keys + offset, jnp.float32)


def _float_key_range(col_bits: int) -> tuple[int, int]:
    """(offset, largest key) of `_float_keys`."""
    low = (1 << col_bits) - 1
    neg_inf_key = ((-0x00800000 ^ 0x7FFFFFFF) >> 1) & ~low
    offset = 0x00800000 - neg_inf_key
    return offset, 0x7F7FFFFF - offset


# `_insert_picks` inserts `_INSERT_UNROLL` key vregs per loop step into the
# lists of `_INSERT_ROWS` rows: two 128-row blocks' insertions in flight (AOT
# c48: 9.2 bundles per key vreg of 128 rows, 12.3 one block at a time; all
# 512 rows spill), and 8 keys amortize the loop's list copies.
_INSERT_ROWS = 256
_INSERT_UNROLL = 8


def _insert_picks(key_ref, num_cols, topk, col_bits) -> jax.Array:
    """`_scan_picks` of `_float_keys` ``key_ref [cols, rows]`` f32 (0 below
    every key), in one pass: each block of `_INSERT_ROWS` rows keeps, per
    sublane (columns of one residue mod 8), its `topk` largest keys in
    sorted vregs, a key entering by a max and a min per level (2 * topk - 1
    ops per key vreg against `_scan_picks`' 3 * topk), then the sublanes'
    lists merge.
    Returns the picks' int32 keys ``[topk, rows]``, best first; needs
    `topk` keys above 0 in every row."""
    offset, _ = _float_key_range(col_bits)
    rows = key_ref.shape[1]
    unroll = _INSERT_UNROLL
    blocks = []
    width = min(rows, _INSERT_ROWS)
    for b in range(rows // width):
        lanes = pl.ds(b * width, width)

        def insert(i, lists, lanes=lanes):
            lists = list(lists)
            for u in range(unroll):
                first = pl.multiple_of((i * unroll + u) * 8, 8)
                y = key_ref[pl.ds(first, 8), lanes]
                for j in range(topk - 1):
                    lists[j], y = jnp.maximum(lists[j], y), jnp.minimum(lists[j], y)
                lists[-1] = jnp.maximum(lists[-1], y)
            return tuple(lists)

        zeros = jnp.zeros((8, width), jnp.float32)
        lists = jax.lax.fori_loop(0, num_cols // (8 * unroll), insert, (zeros,) * topk)
        blocks.append(_merge_sublane_lists(list(lists)))
    keys = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=1)
    return pltpu.bitcast(keys, jnp.int32) - offset


def _merge_sublane_lists(lists) -> jax.Array:
    """Each sublane's descending lists ``topk x [8, rows]`` -> the `topk`
    largest of all ``[topk, rows]``, best first."""
    topk, picks = len(lists), []
    for i in range(topk):
        best = lists[0]
        for shift in (4, 2, 1):  # across sublanes: every sublane gets the max
            best = jnp.maximum(best, pltpu.roll(best, shift, 0))
        picks.append(best[:1])
        # Keys are distinct: one sublane holds the pick; pop its list.
        hit = lists[0] == best
        depth = topk - i
        lists[: depth - 1] = [
            jnp.where(hit, lists[j + 1], lists[j]) for j in range(depth - 1)
        ]
    return jnp.concatenate(picks)


def _merge_sort_network(n: int) -> list[tuple[int, int]]:
    """Batcher's odd-even merge sort of `n` (a power of 2) items: the
    compare-exchange pairs, in order (63 for 16 items)."""
    pairs, p = [], 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


def _compare_exchange(x, pairs):
    """Apply compare-exchange `pairs` to list `x`, larger first."""
    x = list(x)
    for a, b in pairs:
        x[a], x[b] = jnp.maximum(x[a], x[b]), jnp.minimum(x[a], x[b])
    return x


def _top_of_sorted(a, b):
    """The ``len(a)`` largest of descending lists `a` and `b`, descending:
    the maxima of `a` and reversed `b` (a bitonic list), then a bitonic
    sort."""
    n = len(a)
    x = [jnp.maximum(u, v) for u, v in zip(a, b[::-1])]
    d = n // 2
    while d:
        x = _compare_exchange(x, [(i, i + d) for i in range(n) if not i & d])
        d //= 2
    return x


# `_prefill_candidates_kernel` merges each sublane's lists across this many
# rows at a time: two 128-row blocks in flight.
_NETWORK_ROWS = 256


def _column_keys(
    s, cols, num_visible, rank, *, dcp, init_blocks, local_blocks, col_bits, real_cols
) -> jax.Array:
    """Pick keys (`_score_keys`) of scores ``s [columns, rows]`` of this rank's
    columns `cols` for rows that see `num_visible` blocks ``[1, rows]``:
    invisible blocks -inf, forced blocks `msa_select`'s forced scores."""
    blocks = cols * dcp + rank
    visible = (blocks < num_visible) & (cols < real_cols)
    s = jnp.where(visible, s, _NEG_INF)
    s = jnp.where(
        visible & (blocks >= num_visible - local_blocks),
        msa_select._FORCED_LOCAL_SCORE,
        s,
    )
    if init_blocks:
        s = jnp.where(
            visible & (blocks < init_blocks), msa_select._FORCED_INIT_SCORE, s
        )
    return _score_keys(s, cols, col_bits)


def _next_pick(acc, off) -> jax.Array:
    """The pick of a scan's running maxima `acc` (of ``key - off``, `_pick_t`)
    as an ``[8, rows]`` key, the same in every sublane. ``key - off`` wraps
    the keys at or above the last pick below every other, so this is the
    largest key below it."""
    best = functools.reduce(jnp.maximum, acc)
    for shift in (4, 2, 1):  # across sublanes: every sublane gets the max
        best = jnp.maximum(best, pltpu.roll(best, shift, 0))
    return best + off


def _picked(picks, rank, *, col_bits, dcp) -> tuple[jax.Array, jax.Array]:
    """``(scores, blocks)`` ``[topk, rows]`` of the picked keys (``[1, rows]``
    each, best first)."""
    # Odd ranks list their picks in reverse (`local_candidates`).
    keys = jnp.where(
        (rank & 1) == 1, jnp.concatenate(picks[::-1]), jnp.concatenate(picks)
    )
    score, col = _key_scores(keys, col_bits)
    return score, jnp.where(score > _NEG_INF, col * dcp + rank, -1)


# Row blocks of a multiple of this many rows (a vreg's lanes) pick
# transposed (`_candidates_kernel_t`, `_merge_kernel_t`); smaller decode
# blocks would fill few lanes of each vreg.
_TRANSPOSED_ROWS = 128


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "topk",
        "init_blocks",
        "local_blocks",
        "dcp",
        "block_rows",
        "tile_major",
        "interpret",
    ),
)
def local_candidates(
    scores: jax.Array,
    pos: jax.Array,
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: int | jax.Array,
    dcp: int,
    block_rows: int = msa_select.DEFAULT_BLOCK_ROWS,
    live_rows: jax.Array | None = None,
    tile_major: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """This rank's top-k ``(scores [T, topk] f32, blocks [T, topk] i32)``,
    in the selection's order (best first), reversed on ranks of odd index:
    a DCP pair's lists then meet as one bitonic sequence
    (`merge_pair_candidates`), and wider groups' merges take any order.

    With `live_rows` (a traced count), only the row blocks holding rows
    ``[0, live_rows)`` are picked; the other rows' results are unset.

    ``scores [T, 1, P]`` are this rank's block scores (column ``c`` is global
    block ``c * dcp + rank``), ``[P, T]`` transposed, or with `tile_major`
    ``[tiles, P, rows]`` (the large tiles' layout in `msa_indexer`, a row
    block per tile); forced blocks carry `msa_select`'s forced
    scores and missing picks are ``(-inf, -1)``.
    """
    pretransposed = tile_major or scores.ndim == 2
    if tile_major:
        num_tiles, num_cols, block_rows = scores.shape
        num_tokens = num_tiles * block_rows
    elif pretransposed:
        num_cols, num_tokens = scores.shape
    else:
        num_tokens, num_heads, num_cols = scores.shape
        if num_heads != 1:
            raise NotImplementedError(
                f"MSA block selection expects one index head per rank, got {num_heads}."
            )
    # A decode batch fills one block of its own (sublane-rounded) size, not
    # `block_rows` rows mostly of padding: the kernel's topk passes scan every
    # row of the block.
    if not tile_major:
        block_rows = min(block_rows, max(8, -(-num_tokens // 8) * 8))
    # 128-row transposed blocks (rows on the lanes) where they pad less, and for
    # a batch of at most 128 rows (decode; 2 x 128 for 184 rows was slower in
    # the c48 twin, `_DECODE_CANDIDATE_TOKENS`): they pick with
    # elementwise maxima, where a row-major block of a few rows chains three
    # cross-lane reductions per pick (7,389 cycles at 8 rows, AOT): 6.5 ->
    # 4.6 us per call at 8-40 rows, bit-identical (bench j-05ee4bfa). Not for
    # a class with `live_rows`: its unpadded scores must hold whole blocks of
    # the size it was planned for (below).
    rows_128 = -(-num_tokens // _TRANSPOSED_ROWS) * _TRANSPOSED_ROWS
    if (
        not tile_major
        and live_rows is None
        and num_tokens <= _DECODE_CANDIDATE_TOKENS
        and (
            num_tokens <= _TRANSPOSED_ROWS
            or rows_128 < -(-num_tokens // block_rows) * block_rows
        )
    ):
        block_rows = _TRANSPOSED_ROWS
    if pretransposed:
        block_rows = -(-block_rows // _TRANSPOSED_ROWS) * _TRANSPOSED_ROWS
    pad = -num_tokens % block_rows
    s = scores if pretransposed else scores[:, 0]
    if pad and (live_rows is None or num_tokens < block_rows):
        s = jnp.pad(s, ((0, 0), (0, pad)) if pretransposed else ((0, pad), (0, 0)))
    # Without `live_rows` every row block runs, so a last block past the
    # scores always reads past them: in the c48 decode twin (384 tokens, DCP 2,
    # 4-bit) that halted every chip at warmup (j-433e936a). A class's grid
    # stops at its live rows, so its scores keep no padded copy (~17 us per
    # layer for the c48 mixed step's small tiles, j-c22c589e); its last block
    # reads past them only when the live rows reach it. Padding rows pick from
    # garbage and are dropped, as the kernels reduce within a row only.
    p = jnp.pad(pos, (0, pad))[:, None]
    # Every rank keys with the bits of the unpadded width (`_MIN_KEY_BITS`).
    col_bits = _key_bits(num_cols)
    transposed = block_rows % _TRANSPOSED_ROWS == 0
    num_blocks = (num_tokens + pad) // block_rows
    scratch = []
    if transposed:
        # Whole key chunks (`_candidates_kernel_t`); the kernel masks the
        # padded columns.
        chunk = _KEY_CHUNK[pretransposed]
        kernel = functools.partial(
            _candidates_kernel_t, real_cols=num_cols, pretransposed=pretransposed
        )
        if num_cols % chunk:
            col_pad = -num_cols % chunk
            widths = [(0, col_pad), (0, 0)] if pretransposed else [(0, 0), (0, col_pad)]
            s = jnp.pad(s, [(0, 0)] * (s.ndim - 2) + widths, constant_values=_NEG_INF)
            num_cols += col_pad
        pos_spec = pl.BlockSpec((1, block_rows), lambda i: (0, i))
        p = p.T
        out_spec = pl.BlockSpec((topk, block_rows), lambda i: (0, i))
        out_dims = (topk, num_tokens + pad)
        scratch = [pltpu.VMEM((num_cols, block_rows), jnp.int32)]
    else:
        kernel = _candidates_kernel
        pos_spec = pl.BlockSpec((block_rows, 1), lambda i: (i, 0))
        out_spec = pl.BlockSpec((block_rows, topk), lambda i: (i, 0))
        out_dims = (num_tokens + pad, topk)
    picked_scores, picked_blocks = pl.pallas_call(
        functools.partial(
            kernel,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            dcp=dcp,
            col_bits=col_bits,
        ),
        grid=(
            num_blocks
            if live_rows is None
            else jnp.clip(-(-live_rows // block_rows), 1, num_blocks),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            (
                pl.BlockSpec((None, num_cols, block_rows), lambda i: (i, 0, 0))
                if tile_major
                else pl.BlockSpec((num_cols, block_rows), lambda i: (0, i))
                if pretransposed
                else pl.BlockSpec((block_rows, num_cols), lambda i: (i, 0))
            ),
            pos_spec,
        ],
        out_specs=[out_spec, out_spec],
        out_shape=[
            jax.ShapeDtypeStruct(out_dims, jnp.float32),
            jax.ShapeDtypeStruct(out_dims, jnp.int32),
        ],
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",) if live_rows is None else ("arbitrary",),
            vmem_limit_bytes=msa_select._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_dcp_candidates",
    )(rank_array(rank), s, p)
    if transposed:
        picked_scores, picked_blocks = picked_scores.T, picked_blocks.T
    return picked_scores[:num_tokens], picked_blocks[:num_tokens]


class DecodeWrite(NamedTuple):
    """A write `decode_candidates` folds in: `msa.write_caches`' targets,
    tokens' pages and slots, its run bound and which target is the index
    cache the step scores."""

    targets: list[msa_write.Rows | msa_write.SlotValues]
    pages: jax.Array
    slots: jax.Array
    max_runs: int
    index_target: int


def _decode_candidates_kernel(
    # Scalar prefetch
    work_ref,  # [items] tile << _TILE_SHIFT | page group (`msa_indexer.class_work`)
    tile_info_ref,  # [tiles] query row << _ROW_SHIFT | request
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    seq_pages_ref,  # [max_num_seqs] this rank's pages of each request
    cu_ref,  # [max_num_seqs + 1] query starts
    rank_ref,  # [1]
    *refs,
    pick,
    write,
):
    """`msa_indexer._kernel_v3` (small tiles) with its block maxima kept in
    VMEM, token-major, then this rank's picks (`_pick_t`) at the last step:
    the scores never round-trip through HBM, and no gather, padding or
    second kernel stands between the indexer and the selection. A tile's rows
    of other requests (aligned tile starts) are left to their own tiles.

    With `write` (`_run_ops` keyword arguments, ``num_inputs`` and
    ``index_target`` added) the step's cache write runs here too: every run's
    pages are copied in at the first step; a run is merged and copied back
    when the first tile of its request scores the group holding its index
    page, whose ring copy then takes the merged page, and the rest at the
    last step. Its copies' latency hides under the indexer's page stream.

    Refs after the scalars: run tables (``count``, page, lo, hi, tok0, then
    each request's first run and each run's index page among this rank's,
    with `write`); in:
    the tile's query rows ``[Bq, D]``, positions ``[1, R]``, the write's
    inputs and caches, or the index cache; out: ``[R, 2 * topk]``
    `pack_candidates` of each row (row = token), then the written caches;
    scratch: the ring and its semaphores, scores ``[R, P]`` f32, keys ``[P,
    128]`` f32 (`_float_keys`), then the run flags (SMEM) and the write's scratch."""
    it = iter(refs)
    if write is not None:
        num_runs_ref, *run_tables = (next(it) for _ in range(5))
        seq_runs_ref, run_page_idx_ref = next(it), next(it)
    q_ref, pos_ref = next(it), next(it)
    if write is None:
        cache_ref = next(it)
    else:
        write = dict(write)
        num_inputs, index_target = write.pop("num_inputs"), write.pop("index_target")
        num_caches = len(write["specs"])
        write_in = [next(it) for _ in range(num_inputs + num_caches)]
    out_ref = next(it)
    if write is not None:
        write_out = [next(it) for _ in range(num_caches)]
        cache_ref = write_out[index_target]
    buf, sems, scores_ref, key_ref = (next(it) for _ in range(4))

    step, last = pl.program_id(0), pl.num_programs(0) - 1
    entry = work_ref[step]
    info = tile_info_ref[entry >> msa_indexer._TILE_SHIFT]
    seq = info & msa_indexer._SEQ_MASK
    row0 = pl.multiple_of(info >> msa_indexer._ROW_SHIFT, msa_indexer._ROW_ALIGN)
    block_q = q_ref.shape[0]
    pages_per_step = buf.shape[1]
    group = entry & msa_indexer._GROUP_MASK
    rows = row0 + jax.lax.broadcasted_iota(jnp.int32, (block_q, 1), 0)
    own = (rows >= cu_ref[seq]) & (rows < cu_ref[seq + 1])

    def store(first, m, rows_major, pos):
        m = m if rows_major else m.T
        cols = pl.ds(
            pl.multiple_of(group * pages_per_step + first, m.shape[1]), m.shape[1]
        )
        window = scores_ref.at[pl.ds(row0, block_q), cols]
        window[...] = jnp.where(own, m, window[...])

    after_wait = None
    if write is not None:
        merged_ref = next(it)
        ops = msa_write._run_ops(run_tables, write_in + write_out + list(it), **write)
        num_runs = num_runs_ref[0]
        merged_index = ops.out_bufs[index_target].bitcast(jnp.uint32)

        def runs(fn):
            jax.lax.fori_loop(0, num_runs, lambda r, c: (fn(r), c)[1], 0)

        @pl.when(step == 0)
        def _():
            def start(r):
                merged_ref[r] = 0
                for c in ops.in_copies(r, r):
                    c.start()

            runs(start)

        def settle(r):
            @pl.when(merged_ref[r] == 0)
            def _():
                for c in ops.in_copies(r, r):
                    c.wait()
                ops.merge(r, r)
                for c in ops.out_copies(r, r):
                    c.start()
                merged_ref[r] = 1

        # Inlined into each of `_kernel_v3`'s per-slot bodies (3 copies of
        # the merge): a decode run's merge is a few hundred bundles.
        def after_wait(slot):
            first = group * pages_per_step

            def substitute(r, carry):
                page = run_page_idx_ref[r] - first

                @pl.when((page >= 0) & (page < pages_per_step))
                def _():
                    settle(r)
                    buf[slot, page] = merged_index[r, :, 0, :]

                return carry

            # This tile's request's runs only.
            jax.lax.fori_loop(seq_runs_ref[seq], seq_runs_ref[seq + 1], substitute, 0)

    msa_indexer._kernel_v3(
        work_ref,
        tile_info_ref,
        block_tables_ref,
        seq_pages_ref,
        q_ref,
        cache_ref,
        None,
        buf,
        sems,
        store=store,
        after_wait=after_wait,
    )

    @pl.when(step == last)
    def _():
        if write is not None:
            runs(settle)
        width = key_ref.shape[1]
        for b in range(out_ref.shape[0] // width):
            r = pl.ds(b * width, width)
            score, block = _pick_t(
                rank_ref[0],
                scores_ref.at[r],
                pos_ref[:, r],
                key_ref,
                **pick,
                insert=True,
            )
            out_ref[r, :] = jnp.concatenate([score, block.astype(jnp.float32)]).T
        if write is not None:

            def wait_out(r):
                for c in ops.out_copies(r, r):
                    c.wait()

            runs(wait_out)


def decodes_in_one_kernel(
    num_tokens: int,
    index_cache: jax.Array,
    max_num_seqs: int,
    pages_per_seq: int,
    index_heads: int = 1,
) -> bool:
    """Whether `decode_candidates` takes a step of `num_tokens` query rows
    (tokens times `index_heads`) over this index cache: a step no larger than
    every request's verify rows (`msa._DECODE_ROWS` per head) or than the
    indexer's small tiles, whose block
    scores fit `_DECODE_SCORES_VMEM_BYTES`, and pages of v3 (fp8 or bf16,
    4-bit with their scales folded in).

    A request of more rows than a small tile (a short prefill chunk in a
    decode-sized step, absent from the c48 trace's mix) is scored in small
    tiles too, with bf16 queries where `msa_indexer.block_scores` takes a
    large e4m3 tile: no conditional per layer (that conditional, ~5.8 us per
    layer at c48's 192-row steps, j-17330f6e, cost more than the kernel
    saved when it routed such requests, twin j-90ea0b66)."""
    _, rows_per_page, _, _ = index_cache.shape
    rows = max(
        msa.block_score_options()["block_q"],
        max_num_seqs * msa._DECODE_ROWS * index_heads,
    )
    # `decode_candidates`' score scratch (whole row blocks of its page groups)
    # and key scratch (up to `_INSERT_ROWS` columns).
    score_rows = -(-num_tokens // _TRANSPOSED_ROWS) * _TRANSPOSED_ROWS + _INSERT_ROWS
    score_cols = -(-pages_per_seq // _LANES) * _LANES + 2 * _LANES
    return (
        num_tokens <= rows and 4 * score_rows * score_cols <= _DECODE_SCORES_VMEM_BYTES
    ) and (
        not msa_fp4.is_fp4(index_cache)
        or msa_fp4.scales_in_page(rows_per_page, index_cache.dtype)
    )


def decode_candidates(
    index_q: jax.Array,
    index_cache: jax.Array,
    local_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    seq_id: jax.Array,
    pos: jax.Array,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: int | jax.Array,
    dcp: int,
    write: DecodeWrite | None = None,
    interpret: bool = False,
) -> jax.Array | tuple[list[jax.Array], jax.Array]:
    """`pack_candidates` of `local_candidates` over `msa._block_scores` (one
    index head) or `msa.partial_block_scores` (``G > 1``) for a decode-sized
    step (`msa_indexer.is_prefill_step` false) in one kernel
    (`_decode_candidates_kernel`): ``[T, G * 2 * topk]``, head-major per
    token, the same values. With `write` (one head) it first writes the
    step's rows as `msa.write_caches` would (the index target is
    `index_cache`) and returns ``(caches, candidates)``.

    ``index_q [T, G, D]`` (rotated for a 4-bit cache): the heads fold into
    each request's query rows in (token, head) order, as
    `msa.partial_block_scores` scores them. ``local_lens`` this rank's
    lengths (`local_seq_lens`), ``seq_id`` and ``pos [T]`` the tokens'
    requests and positions (`msa.token_positions`)."""
    num_tokens, heads, head_dim = index_q.shape
    _, rows_per_page, _, _ = index_cache.shape
    msa_indexer.check_square_pages(index_cache)
    if heads > 1:
        if write is not None:
            raise NotImplementedError("a folded write takes one index head")
        index_q = index_q.reshape(num_tokens * heads, 1, head_dim)
        cu_q_lens = heads * cu_q_lens
        pos = jnp.repeat(pos, heads)
    out_tokens, num_tokens = num_tokens, num_tokens * heads
    if not decodes_in_one_kernel(
        num_tokens,
        index_cache,
        local_lens.shape[0],
        block_tables.shape[0] // local_lens.shape[0],
        heads,
    ):
        raise NotImplementedError(
            f"no one-kernel decode of {num_tokens} rows over {index_cache.shape}"
        )
    max_num_seqs = local_lens.shape[0]
    options = msa.block_score_options(
        heads,
        msa._wide_decode_scores(index_q, index_cache, local_lens, block_tables, heads),
    )
    pages_per_step = options["small_pages_per_step"]
    transposed = msa_indexer._scores_transposed(1, rows_per_page, index_cache.dtype)
    tile_cap = options["transposed_tile_q"] if transposed else msa_indexer._SMALL_TILE_Q
    block_q = min(options["block_q"], tile_cap, -(-num_tokens // 8) * 8)
    seq_ids = jnp.arange(max_num_seqs, dtype=jnp.int32)
    block_size = msa.page_size(index_cache)
    work = msa_indexer.class_work(
        index_q,
        local_lens,
        block_tables,
        cu_q_lens,
        seq_ids < num_seqs,
        block_q=block_q,
        # Aligned tile starts add up to _ROW_ALIGN - 1 rows per request.
        max_tiles=(num_tokens + (msa_indexer._ROW_ALIGN - 1) * max_num_seqs) // block_q
        + max_num_seqs,
        pages_per_step=pages_per_step,
        page_size=msa_indexer._page_tokens(rows_per_page, index_cache.dtype),
    )
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    num_cols = -(-pages_per_seq // pages_per_step) * pages_per_step
    rows = -(-num_tokens // _TRANSPOSED_ROWS) * _TRANSPOSED_ROWS
    # Steps of a multiple of `_INSERT_ROWS` rows pick in blocks of that many,
    # two 128-row blocks' insertions in flight (`_INSERT_ROWS`).
    pick_rows = _INSERT_ROWS if rows % _INSERT_ROWS == 0 else _TRANSPOSED_ROWS
    # The last tile starts at most at the last token's aligned row and runs
    # block_q rows; the picks read the first `rows`.
    score_rows = max(rows, -(-num_tokens // 8) * 8 + block_q)
    pick = dict(
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        dcp=dcp,
        col_bits=_key_bits(num_cols),
        real_cols=num_cols,
    )
    shift, row_shift = msa_indexer._TILE_SHIFT, msa_indexer._ROW_SHIFT
    scalars = [
        work.work,
        work.tile_info,
        block_tables,
        work.seq_pages,
        cu_q_lens,
        rank_array(rank),
    ]
    q_spec = pl.BlockSpec(
        (pl.Element(block_q), pl.Element(head_dim)),
        lambda w, work, tile_info, *_: (
            pl.multiple_of(
                tile_info[work[w] >> shift] >> row_shift, msa_indexer._ROW_ALIGN
            ),
            0,
        ),
    )
    inputs = [work.q_rows, jnp.pad(pos, (0, rows - num_tokens))[None]]
    in_specs = [q_spec, pl.BlockSpec((1, rows), lambda w, *_: (0, 0))]
    out_shape = [jax.ShapeDtypeStruct((rows, 2 * topk), jnp.float32)]
    out_specs = [pl.BlockSpec((rows, 2 * topk), lambda w, *_: (0, 0))]
    scratch = [
        pltpu.VMEM(
            (msa_indexer._RING_SLOTS, pages_per_step, rows_per_page, head_dim),
            jnp.uint32,
        ),
        pltpu.SemaphoreType.DMA((msa_indexer._RING_SLOTS,)),
        pltpu.VMEM((score_rows, num_cols), jnp.float32),
        pltpu.VMEM((num_cols, pick_rows), jnp.float32),
    ]
    vmem_bytes = 4 * (
        msa_indexer._RING_SLOTS * pages_per_step * rows_per_page * head_dim
        + (score_rows + pick_rows) * num_cols
    )
    aliases, write_kwargs = {}, None
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    if write is None:
        inputs.append(index_cache)
        in_specs.append(any_spec)
    else:
        ops = msa_write.write_operands(
            write.targets,
            write.pages,
            write.slots,
            max_runs=write.max_runs,
            depth=write.max_runs,
        )
        if ops.caches[write.index_target] is not index_cache:
            raise ValueError("the write's index target must be the scored cache")
        num_runs, *run_tables = ops.runs
        run_tok0 = jnp.minimum(run_tables[-1], num_tokens - 1)
        run_seq = msa_indexer.lookup(seq_id, run_tok0)
        run_page_idx = msa_indexer.lookup(pos, run_tok0) // block_size // dcp
        # Runs follow the tokens, so each request's are consecutive: request
        # s has runs [seq_runs[s], seq_runs[s + 1]).
        live_runs = jnp.arange(write.max_runs) < num_runs
        seq_runs = jnp.sum(
            live_runs[None, :]
            & (run_seq[None, :] < jnp.arange(max_num_seqs + 1)[:, None]),
            axis=1,
            dtype=jnp.int32,
        )
        scalars += [num_runs[None], *run_tables, seq_runs, run_page_idx]
        first_cache = len(scalars) + len(inputs) + len(ops.inputs)
        inputs += ops.inputs + ops.caches
        in_specs += [any_spec] * (len(ops.inputs) + len(ops.caches))
        out_shape += [jax.ShapeDtypeStruct(c.shape, c.dtype) for c in ops.caches]
        out_specs += [any_spec] * len(ops.caches)
        aliases = {first_cache + i: 1 + i for i in range(len(ops.caches))}
        scratch += [pltpu.SMEM((write.max_runs,), jnp.int32), *ops.scratch]
        vmem_bytes += ops.vmem_bytes
        write_kwargs = dict(
            ops.kwargs, num_inputs=len(ops.inputs), index_target=write.index_target
        )
    out = pl.pallas_call(
        functools.partial(_decode_candidates_kernel, pick=pick, write=write_kwargs),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(jnp.maximum(work.group_ends[-1], 1),),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch,
        ),
        out_shape=out_shape,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=max(msa_select._VMEM_LIMIT_BYTES, vmem_bytes + (8 << 20)),
        ),
        interpret=interpret,
        name="minimax_m3_msa_dcp_decode_candidates",
    )(*scalars, *inputs)
    candidates = out[0][:num_tokens].reshape(out_tokens, -1)
    if write is None:
        return candidates
    return out[1:], candidates


def pack_candidates(scores: jax.Array, blocks: jax.Array) -> jax.Array:
    """``[T, 2 * topk]`` f32: scores, then the block ids as f32 values, so one
    all-gather moves both.

    The ids travel as values (exact below 2**24; a block id is below
    max_model_len / block_size), not bitcast: a bitcast id is a subnormal
    f32 (and -1 a NaN), which a TPU collective may flush to zero or
    canonicalize, and every id would decode as block 0.
    """
    return jnp.concatenate([scores, blocks.astype(jnp.float32)], axis=-1)


def merge_candidates(candidates: jax.Array, topk: int) -> jax.Array:
    """Global selection ``[T, topk]`` (``-1`` padded) from every rank's
    packed candidates ``[dcp, T, 2 * topk]``, in `msa_select.select_blocks`'s
    order: score descending, ties to the lower block."""
    dcp, num_tokens, _ = candidates.shape
    scores = candidates[..., :topk]
    blocks = candidates[..., topk:].astype(jnp.int32)
    s = scores.transpose(1, 0, 2).reshape(num_tokens, dcp * topk)
    b = blocks.transpose(1, 0, 2).reshape(num_tokens, dcp * topk)
    # The real candidates (score > -inf) are distinct blocks: a rank picks
    # distinct columns and ranks hold disjoint blocks. So each one's place in
    # the order is the number of real candidates ahead of it, and the merge
    # is two dense compare-reduce fusions instead of topk dependent steps.
    # Their [rows, dcp * topk, dcp * topk] operands stay fused into the
    # reductions (dcp * topk <= 128 lanes per row at the served shapes).
    real = s > _NEG_INF
    s_i, b_i = s[:, :, None], b[:, :, None]
    s_j, b_j = s[:, None, :], b[:, None, :]
    ahead = real[:, None, :] & ((s_j > s_i) | ((s_j == s_i) & (b_j < b_i)))
    place = jnp.where(real, jnp.sum(ahead, axis=2, dtype=jnp.int32), topk)
    slot = jnp.arange(topk, dtype=jnp.int32)[None, None, :]
    return jnp.max(jnp.where(place[:, :, None] == slot, b_i, -1), axis=1)


def _merge_kernel(c_ref, o_ref, *, topk: int):
    """`merge_candidates` of a block of tokens, every head, in VMEM: ``topk``
    steps of (best score, lowest block among its ties) over the ``dcp *
    topk`` candidates. ``c_ref [dcp, Bt, G * 2 * topk]``, ``o_ref [Bt, G *
    topk]``."""
    dcp, block_t, width = c_ref.shape
    lane = jax.lax.broadcasted_iota(jnp.int32, (block_t, topk), 1)
    no_block = jnp.iinfo(jnp.int32).max
    for g in range(width // (2 * topk)):
        base = g * 2 * topk
        s = jnp.concatenate(
            [c_ref[r, :, base : base + topk] for r in range(dcp)], axis=1
        )
        b = jnp.concatenate(
            [c_ref[r, :, base + topk : base + 2 * topk] for r in range(dcp)], axis=1
        ).astype(jnp.int32)
        out = jnp.full((block_t, topk), -1, jnp.int32)
        for k in range(topk):
            best_score = jnp.max(s, axis=1, keepdims=True)
            best = jnp.min(
                jnp.where(s == best_score, b, no_block), axis=1, keepdims=True
            )
            out = jnp.where(lane == k, jnp.where(best_score > _NEG_INF, best, -1), out)
            s = jnp.where(b == best, _NEG_INF, s)
        o_ref[:, g * topk : (g + 1) * topk] = out


def _merge_kernel_t(c_ref, o_ref, *, topk: int):
    """`_merge_kernel` with each rank's block transposed in VMEM: candidate
    slots on sublanes, tokens on lanes, so a head's ``dcp * topk``
    candidates are whole vreg rows of the ranks' blocks (no lane shuffles)
    and each pick reduces across vregs and sublanes on the VPU instead of
    across lanes on the XLU. ``c_ref [dcp, Bt, G * 2 * topk]``, ``o_ref
    [Bt, G * topk]``."""
    dcp, block_t, width = c_ref.shape
    ranks = [c_ref[r].T for r in range(dcp)]  # [width, Bt] each
    no_block = jnp.iinfo(jnp.int32).max
    out = []
    for g in range(width // (2 * topk)):
        base = g * 2 * topk
        s = jnp.concatenate([c[base : base + topk] for c in ranks], axis=0)
        b = jnp.concatenate(
            [c[base + topk : base + 2 * topk] for c in ranks], axis=0
        ).astype(jnp.int32)
        for _ in range(topk):
            best_score = jnp.max(s, axis=0, keepdims=True)  # [1, Bt]
            best = jnp.min(
                jnp.where(s == best_score, b, no_block), axis=0, keepdims=True
            )
            out.append(jnp.where(best_score > _NEG_INF, best, -1))
            s = jnp.where(b == best, _NEG_INF, s)
    o_ref[...] = jnp.concatenate(out, axis=0).T


# Merges of more than this many candidates per token (DCP 8: 128) run
# `_merge_kernel`: `merge_candidates`' [rows, n, n] ranking and the relayouts
# of the [dcp, rows, 2 * topk] candidates into rows cost ~3 ms per layer in the
# c64 mixed step (32768 rows; j-c1444339). Narrower ones keep the dense
# ranking, cheap at n = 32 (`msa_attend_dcp` merges a DCP pair by
# `merge_pair_candidates`).
_KERNEL_MERGE_MIN_CANDIDATES = 64
_MERGE_BLOCK_TOKENS = 256


def merge_head_candidates(
    candidates: jax.Array, topk: int, interpret: bool = False
) -> jax.Array:
    """`merge_candidates` per KV head: ``[dcp, T, G * 2 * topk]`` (each
    rank's `pack_candidates` of its ``G`` heads, head-major) -> selection
    ``[T, G, topk]``."""
    dcp, num_tokens, width = candidates.shape
    num_heads = width // (2 * topk)
    if dcp * topk < _KERNEL_MERGE_MIN_CANDIDATES:
        rows = candidates.reshape(dcp, num_tokens * num_heads, 2 * topk)
        return merge_candidates(rows, topk).reshape(num_tokens, num_heads, topk)
    block_t = min(_MERGE_BLOCK_TOKENS, -(-num_tokens // 8) * 8)
    pad = -num_tokens % block_t
    c = jnp.pad(candidates, ((0, 0), (0, pad), (0, 0)))
    kernel = _merge_kernel_t if block_t % _TRANSPOSED_ROWS == 0 else _merge_kernel
    out = pl.pallas_call(
        functools.partial(kernel, topk=topk),
        grid=((num_tokens + pad) // block_t,),
        in_specs=[pl.BlockSpec((dcp, block_t, width), lambda i: (0, i, 0))],
        out_specs=pl.BlockSpec((block_t, num_heads * topk), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens + pad, num_heads * topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge",
    )(c)
    return out[:num_tokens].reshape(num_tokens, num_heads, topk)


def merge_pair_candidates(candidates: jax.Array, topk: int) -> jax.Array:
    """`merge_head_candidates` of a DCP pair as a set: ``[2, T, G * 2 *
    topk]``, the pair's `pack_candidates` in either order -> ``[T, G,
    topk]`` holding the same blocks, in no particular order (``-1`` for
    missing picks).

    One rank's picks are sorted in the selection's order (score descending,
    ties to the lower block; missing picks ``(-inf, -1)`` last), the other's
    reversed (`local_candidates`), so the top ``topk`` of their union is the
    better of slot ``i`` of each (a bitonic merge's first step): one
    elementwise fusion instead of the dense ranking (~20 us per layer at 8K
    tokens, c4 twin j-2e5da13f). The attention reads a token's selection as
    a set (page unions and membership compares)."""
    picks = []
    for g in range(candidates.shape[2] // (2 * topk)):
        c = candidates[:, :, 2 * topk * g : 2 * topk * (g + 1)]
        s0, s1 = c[0, :, :topk], c[1, :, :topk]
        b0, b1 = c[0, :, topk:].astype(jnp.int32), c[1, :, topk:].astype(jnp.int32)
        first = (s0 > s1) | ((s0 == s1) & (b0 < b1))
        s = jnp.where(first, s0, s1)
        picks.append(jnp.where(s > _NEG_INF, jnp.where(first, b0, b1), -1))
    return jnp.stack(picks, axis=1)


# Tokens per grid step of `rotate_folded_queries`.
_FOLD_BLOCK_TOKENS = 512


def rotate_folded_queries(index_q: jax.Array, *, interpret: bool = False) -> jax.Array:
    """`msa_fp4.rotate_queries` of ``index_q [T, G, D]`` (bf16) as the rows
    `_class_candidates` folds the heads into, ``[T * G, D]`` in (token,
    head) order, in f32 (the indexer's query rows). As XLA ops the rotation
    came out head-major and the fold re-laid it out in 2-row tiles (c64
    mixed twin j-c5ebe0af: ~27 us per layer)."""
    num_tokens, heads, head_dim = index_q.shape
    if index_q.dtype != jnp.bfloat16:  # the 6-pass f32 rotation
        rotated = msa_fp4.rotate_queries(index_q).reshape(num_tokens * heads, -1)
        return rotated.astype(jnp.float32)
    block = min(num_tokens, _FOLD_BLOCK_TOKENS)

    def kernel(x_ref, h_ref, o_ref):
        rows = x_ref.shape[1]
        for g in range(heads):
            y = jnp.dot(x_ref[g], h_ref[...], preferred_element_type=jnp.float32)
            y = (y * (1.0 / head_dim)).astype(index_q.dtype)
            o_ref[pl.ds(g, rows, stride=heads), :] = y.astype(o_ref.dtype)

    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(num_tokens, block),),
        in_specs=[
            pl.BlockSpec((heads, block, head_dim), lambda i: (0, i, 0)),
            pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block * heads, head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens * heads, head_dim), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_rotate_folded_queries",
    )(
        index_q.transpose(1, 0, 2),
        jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16),
    )


def _prefill_candidates_kernel(
    # Scalar prefetch
    work_ref,  # [items] tile << _TILE_SHIFT | page group (`msa_indexer.class_work`)
    tile_info_ref,  # [tiles] query row << _ROW_SHIFT | request
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    seq_pages_ref,  # [max_num_seqs] this rank's pages of each request
    cu_ref,  # [max_num_seqs + 1] query rows' starts (heads folded)
    rank_ref,  # [1]
    tiles_ref,  # [2] large tiles, small tiles in use
    small_info_ref,  # [small tiles] query row << _ROW_SHIFT | request
    # In
    q_ref,  # [Bq, D] the tile's query rows
    pos_ref,  # [1, Bq] their positions
    cache_ref,  # [pages, S / p, p, D] index cache, left in HBM
    small_ref,  # [small tiles * Bs, 2 * topk] small-class candidates, in HBM
    # Out
    out_ref,  # [rows, 2 * topk] candidates of every query row, in HBM
    # Scratch
    buf,  # `msa_indexer._kernel_v3`'s ring and semaphores
    sems,
    keys_ref,  # [pages per step, Bq] f32 the step's pick keys (`_float_keys`)
    lists_ref,  # [topk, 8, Bq] f32 each sublane's best keys so far, descending
    pad_scores,  # [32, 128] f32 and int32: the picks of position 0
    pad_keys,
    rows_ref,  # [rows + Bq, 2 * topk] f32 the candidates being assembled
    small_buf,  # [small tiles * Bs, 2 * topk] f32
    copy_sems,  # DMA [2]: small-class candidates in, candidates out
    *key_cache,  # [P, S / 4, D] the request's pages' fp8 MXU keys, if cacheable
    pick,
    chunk,
):
    """`msa_indexer._kernel_v3` over the large (prefill-chunk) tiles with each
    tile's pick keys built from its block maxima as they are scored, in
    VMEM, and merged into each sublane's `topk` best by groups of `topk` key
    vregs (a merge network and `_top_of_sorted`) among the next chunks'
    matmuls; at the tile's last page group the sublanes' lists merge into
    its rows' picks: the scores never round-trip through HBM. (A tile's
    whole scan spread over the next tile's scoring bodies made them slower
    than the scan alone: they have no free vector registers.)

    The picks land at their query rows (rows of other requests, from
    aligned tile starts and past a request's end, are left to their own
    tiles); at the last step the small tiles' `pack_candidates` join them
    at theirs and rows past the last request take the picks of position 0,
    so the output is in query-row order with no gather. `chunk` is
    `_kernel_v3`'s pages per dot."""
    tile_shift, row_shift = msa_indexer._TILE_SHIFT, msa_indexer._ROW_SHIFT
    seq_mask = msa_indexer._SEQ_MASK
    topk, dcp, col_bits = pick["topk"], pick["dcp"], pick["col_bits"]
    step, last = pl.program_id(0), pl.num_programs(0) - 1
    entry = work_ref[step]
    tile = entry >> tile_shift
    group = entry & msa_indexer._GROUP_MASK
    pages_per_step = buf.shape[1]
    rank = rank_ref[0]
    small_copy = pltpu.make_async_copy(small_ref, small_buf, copy_sems.at[0])

    @pl.when(step == 0)
    def _():
        small_copy.start()

    num_visible = pos_ref[...] // pick["block_size"] + 1  # [1, Bq]

    group_cols = 8 * topk
    network = _merge_sort_network(topk)

    @pl.when(step == 0)
    def _():
        lists_ref[...] = jnp.zeros(lists_ref.shape, jnp.float32)

    def merge_keys(first, num_keys):
        """Merge the step's pick keys ``[first, first + num_keys)`` (at most
        a group of `topk` key vregs, the rest read as 0) into the lists."""
        for b in range(lists_ref.shape[2] // _LANES):
            lanes = pl.ds(b * _LANES, _LANES)
            zeros = jnp.zeros((8, _LANES), jnp.float32)
            keys = [
                keys_ref[pl.ds(first + 8 * i, 8), lanes] if 8 * i < num_keys else zeros
                for i in range(topk)
            ]
            lists = [lists_ref[i, :, lanes] for i in range(topk)]
            merged = _top_of_sorted(lists, _compare_exchange(keys, network))
            for i in range(topk):
                lists_ref[i, :, lanes] = merged[i]

    def store(first, m, rows_major, pos):
        """Pick keys of the chunk's maxima, merged into the lists by groups
        of `topk` key vregs among the next chunks' matmuls (at the body's
        `pos`: a body merges its own keys)."""
        m = m.T if rows_major else m
        col0 = group * pages_per_step + first
        cols = col0 + jax.lax.broadcasted_iota(jnp.int32, m.shape, 0)
        keys = _column_keys(
            m,
            cols,
            num_visible,
            rank,
            dcp=dcp,
            init_blocks=pick["init_blocks"],
            local_blocks=pick["local_blocks"],
            col_bits=col_bits,
            real_cols=pick["real_cols"],
        )
        keys_ref[pl.ds(pos, m.shape[0]), :] = _float_keys(keys, col_bits)
        end = pos + m.shape[0]
        if end % group_cols == 0:
            merge_keys(end - group_cols, group_cols)

    def after_body(scored):
        # Columns past the scored chunks are past the request's pages.
        end = scored * chunk
        if end % group_cols:
            merge_keys(end - end % group_cols, end % group_cols)

    def cached(w):
        # A request's tiles are consecutive: all but its first find its
        # pages' keys in `key_cache`.
        t = work_ref[w] >> tile_shift
        prev = tile_info_ref[jnp.maximum(t - 1, 0)]
        return (t > 0) & ((prev & seq_mask) == (tile_info_ref[t] & seq_mask))

    msa_indexer._kernel_v3(
        work_ref,
        tile_info_ref,
        block_tables_ref,
        seq_pages_ref,
        q_ref,
        cache_ref,
        None,
        buf,
        sems,
        store=store,
        after_body=after_body,
        slot_bodies=False,
        key_cache=key_cache[0] if key_cache else None,
        cached=cached,
    )

    def put_rows(info, picked, live=True):
        """Store ``picked [n, 2 * topk]`` at the query rows from ``info``'s,
        those of its request only."""
        seq = info & seq_mask
        row0 = pl.multiple_of(info >> row_shift, msa_indexer._ROW_ALIGN)
        rows = row0 + jax.lax.broadcasted_iota(jnp.int32, (picked.shape[0], 1), 0)
        own = live & (rows >= cu_ref[seq]) & (rows < cu_ref[seq + 1])
        window = rows_ref.at[pl.ds(row0, picked.shape[0])]
        window[...] = jnp.where(own, picked, window[...])

    def packed(score, block):
        return jnp.concatenate([score, block.astype(jnp.float32)]).T

    next_tile = work_ref[jnp.minimum(step + 1, last)] >> tile_shift

    @pl.when((step == last) | (next_tile != tile))
    def _():
        # The lists hold every scored column's key: those past the columns
        # its rows can see key below them all.
        width = min(lists_ref.shape[2], _NETWORK_ROWS)
        blocks = []
        for b in range(lists_ref.shape[2] // width):
            lanes = pl.ds(b * width, width)
            blocks.append(
                _merge_sublane_lists([lists_ref[i, :, lanes] for i in range(topk)])
            )
        lists_ref[...] = jnp.zeros(lists_ref.shape, jnp.float32)
        keys = pltpu.bitcast(jnp.concatenate(blocks, axis=1), jnp.int32)
        keys = keys - _float_key_range(col_bits)[0]
        info = tile_info_ref[tile]
        score, block = _picked(
            [keys[i : i + 1] for i in range(topk)], rank, col_bits=col_bits, dcp=dcp
        )
        put_rows(info, packed(score, block), tile < tiles_ref[0])

    @pl.when(step == last)
    def _():
        small_copy.wait()
        small_rows = small_buf.shape[0] // small_info_ref.shape[0]

        def merge(i, carry):
            first = pl.multiple_of(i * small_rows, small_rows)
            put_rows(small_info_ref[i], small_buf[pl.ds(first, small_rows)])
            return carry

        jax.lax.fori_loop(0, tiles_ref[1], merge, 0)
        # Rows past the last request: a padding token's position is 0, which
        # sees one block, scored -inf unless forced.
        pad_scores[...] = jnp.full(pad_scores.shape, _NEG_INF, jnp.float32)
        score, block = _pick_t(
            rank,
            pad_scores,
            jnp.zeros((1, pad_scores.shape[1]), jnp.int32),
            pad_keys,
            pretransposed=True,
            **pick,
        )
        pad_row = packed(score, block)[:8]
        end = cu_ref[cu_ref.shape[0] - 1]
        num_rows = out_ref.shape[0]

        def fill(i, carry):
            row0 = pl.multiple_of(i * 8, 8)
            rows = row0 + jax.lax.broadcasted_iota(jnp.int32, (8, 1), 0)
            window = rows_ref.at[pl.ds(row0, 8)]
            window[...] = jnp.where(rows >= end, pad_row, window[...])
            return carry

        jax.lax.fori_loop(end // 8, -(-num_rows // 8), fill, 0)
        out = pltpu.make_async_copy(
            rows_ref.at[pl.ds(0, num_rows)], out_ref, copy_sems.at[1]
        )
        out.start()
        out.wait()


# Pages per large-tile step of `_prefill_candidates_kernel`, its bodies one
# per count of scored chunks (`_kernel_v3` without slot bodies: 3 x 128-page
# bodies ran ~28K bundles, 3 x 256-page ones ~58K and 1.3-6x slower): half
# the grid steps of 128-page groups.
_PREFILL_PAGES_PER_STEP = 256
# VMEM `_prefill_candidates_kernel` may take (v7x: 64 MiB per core, with room
# for the compiler's own); a larger step picks its large tiles in their own
# kernel. c64's 8K-token mixed step (2 heads folded) takes ~17 MiB.
_PREFILL_VMEM_BYTES = 40 << 20


def _prefill_vmem_bytes(
    plan, index_cache, small_rows: int, num_rows: int, topk: int, key_cache: bool
) -> int:
    """VMEM of `_prefill_candidates`: the page ring, the request's fp8 keys
    (with `key_cache`), the step's pick keys and the lists, the query tile
    (double-buffered), the assembled output and the small class's picks,
    whose [rows, 2 * topk] f32 pad to 128 lanes."""
    _, rows_per_page, _, head_dim = index_cache.shape
    block_q = plan.large_block_q
    ring = msa_indexer._RING_SLOTS * plan.large_pages_per_step * rows_per_page
    page_size = msa_indexer._page_tokens(rows_per_page, index_cache.dtype)
    cache = plan.large_pages * page_size * head_dim if key_cache else 0
    return cache + 4 * (
        ring * head_dim
        + (plan.large_pages_per_step + 8 * topk) * block_q
        + 2 * block_q * head_dim
        + (num_rows + block_q + small_rows) * _LANES
    )


def _prefill_chunk(plan, index_cache) -> int:
    """`_kernel_v3`'s pages per dot over the large tiles."""
    page_size = msa_indexer._page_tokens(index_cache.shape[1], index_cache.dtype)
    return msa_indexer._v3_chunk(
        plan.large_pages_per_step, plan.large_block_q, page_size
    )


def _prefill_fits(
    plan, index_cache, small_rows: int, num_rows: int, topk: int, key_cache: bool
) -> bool:
    """Whether `_prefill_candidates` takes the step: tile-major large and
    token-major small tiles, pick keys in whole merge groups of `topk` key
    vregs (whole dots each, whole key vregs per dot) and VMEM within
    `_PREFILL_VMEM_BYTES`."""
    step_cols = 8 * topk
    chunk = _prefill_chunk(plan, index_cache)
    return (
        plan.large_tile_major
        and plan.token_major
        and topk & (topk - 1) == 0
        and plan.large_pages_per_step % step_cols == 0
        and step_cols % chunk == 0
        and chunk % 8 == 0
        and _prefill_vmem_bytes(
            plan, index_cache, small_rows, num_rows, topk, key_cache
        )
        <= _PREFILL_VMEM_BYTES
    )


def _prefill_candidates(
    plan: msa_indexer._ClassPlan,
    index_cache: jax.Array,
    block_tables: jax.Array,
    folded_cu: jax.Array,
    positions,
    small: msa_indexer.ClassScores,
    small_packed: jax.Array,
    num_rows: int,
    *,
    pick: dict,
    rank: jax.Array,
    key_cache: bool,
    interpret: bool,
) -> jax.Array:
    """`_prefill_candidates_kernel`: the packed candidates ``[num_rows, 2 *
    topk]`` of every (heads-folded) query row, the large class's picked in
    the indexer kernel (its keys in a `key_cache` across tiles if set), the
    small class's (`small_packed`, as `small` scored them) placed at their
    rows. `positions(tile_seq, tile_row0, block_q)` gives a class's column
    positions."""
    _, rows_per_page, _, head_dim = index_cache.shape
    work = plan.large_work(whole_later_tiles=key_cache)
    block_q, pages_per_step = plan.large_block_q, plan.large_pages_per_step
    num_cols = plan.large_pages
    pos = positions(work.tile_seq, work.tile_row0, block_q).reshape(1, -1)
    shift, row_shift = msa_indexer._TILE_SHIFT, msa_indexer._ROW_SHIFT
    small_info = (
        jnp.clip(small.tile_row0, 0, num_rows // 8 * 8) << row_shift
    ) | small.tile_seq
    scalars = [
        work.work,
        work.tile_info,
        block_tables,
        work.seq_pages,
        folded_cu,
        rank_array(rank),
        jnp.stack([work.num_tiles, small.num_tiles]).astype(jnp.int32),
        small_info,
    ]
    q_spec = pl.BlockSpec(
        (pl.Element(block_q), pl.Element(head_dim)),
        lambda w, work, tile_info, *_: (
            pl.multiple_of(
                tile_info[work[w] >> shift] >> row_shift, msa_indexer._ROW_ALIGN
            ),
            0,
        ),
    )
    pos_spec = pl.BlockSpec((1, block_q), lambda w, work, *_: (0, work[w] >> shift))
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    width = small_packed.shape[1]
    assembled_rows = num_rows + block_q
    ring = (msa_indexer._RING_SLOTS, pages_per_step, rows_per_page, head_dim)
    page_size = msa_indexer._page_tokens(rows_per_page, index_cache.dtype)
    chunk = _prefill_chunk(plan, index_cache)
    scratch = [
        pltpu.VMEM(ring, jnp.uint32),
        pltpu.SemaphoreType.DMA((msa_indexer._RING_SLOTS,)),
        pltpu.VMEM((pages_per_step, block_q), jnp.float32),
        pltpu.VMEM((pick["topk"], 8, block_q), jnp.float32),
        pltpu.VMEM((_KEY_CHUNK[True], _LANES), jnp.float32),
        pltpu.VMEM((_KEY_CHUNK[True], _LANES), jnp.int32),
        pltpu.VMEM((assembled_rows, width), jnp.float32),
        pltpu.VMEM(small_packed.shape, jnp.float32),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    if key_cache:
        scratch.append(pltpu.VMEM((num_cols, page_size // 4, head_dim), jnp.uint32))
    vmem_bytes = _prefill_vmem_bytes(
        plan, index_cache, small_packed.shape[0], num_rows, pick["topk"], key_cache
    )
    return pl.pallas_call(
        functools.partial(_prefill_candidates_kernel, pick=pick, chunk=chunk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(jnp.maximum(work.group_ends[-1], 1),),
            in_specs=[q_spec, pos_spec, any_spec, any_spec],
            out_specs=any_spec,
            scratch_shapes=scratch,
        ),
        out_shape=jax.ShapeDtypeStruct((num_rows, width), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=max(msa_select._VMEM_LIMIT_BYTES, vmem_bytes + (8 << 20)),
        ),
        interpret=interpret,
        name="minimax_m3_msa_dcp_prefill_candidates",
    )(*scalars, work.q_rows, pos, index_cache, small_packed)


def _class_candidates(
    folded: jax.Array,
    num_heads: int,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    local_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    seq_id: jax.Array,
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: jax.Array,
    dcp: int,
    interpret: bool,
    scale_cache: jax.Array | None,
    scale_row: int | None,
) -> jax.Array:
    """Packed candidates ``[T, G * 2 * topk]`` (`pack_candidates` per head,
    head-major) of a prefill-sized step from the ``G = num_heads`` heads'
    index queries ``folded [T * G, D]``: each `msa_indexer` tile class picks
    in the layout it scored in, and only the picks go back to token order.
    Gathering the classes' ``[P, columns]`` scores to ``[T, P]`` first (a
    concatenate, column gather and transpose, ~110 us per layer at c48)
    moved 32x the bytes, which the transposing pick kernel then turned back.
    With tile-major large tiles (kernel v3) the large class picks inside its
    indexer kernel, which also places every row's candidates
    (`_prefill_candidates`).

    ``G > 1`` heads are folded into each request's query rows in (token,
    head) order, as ``index_q [T, G, D]`` lays them out: the block scores are
    blind to the row order within a request, and a (head, token) order took
    a row gather of the queries (~45 us per layer at c64's 8K-token mixed
    steps, twin j-a202588d).
    """
    num_tokens = folded.shape[0] // num_heads
    rows = jnp.arange(num_tokens * num_heads, dtype=jnp.int32)
    rows = rows.reshape(num_tokens, num_heads).T  # [G, T]: head h of token t
    folded = folded[:, None]
    folded_cu = num_heads * cu_q_lens
    class_plan = functools.partial(
        msa_indexer.class_plan,
        folded,
        index_cache,
        local_lens,
        block_tables,
        folded_cu,
        num_seqs,
        scale_cache=scale_cache,
        scale_row=scale_row,
        interpret=interpret,
    )
    options = msa.block_score_options(num_heads)
    plan = class_plan(**options)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    max_num_seqs = q_lens.shape[0]
    seq_of_token = jnp.minimum(seq_id, max_num_seqs - 1)
    candidates = functools.partial(
        local_candidates,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=dcp,
        interpret=interpret,
    )

    def positions(tile_seq, tile_row0, block_q):
        # A column's query row sits at offset o of its request's folded rows,
        # token o // G of it: position = len - q_len + o // G. Rows of a tile
        # past its request (never gathered back) stay inside its positions,
        # which bound the candidate scan's width.
        per_tile = functools.partial(msa_indexer.lookup, index=tile_seq)
        row = tile_row0[:, None] + jnp.arange(block_q, dtype=jnp.int32)[None, :]
        offset = row - per_tile(folded_cu[:-1])[:, None]
        q_len = jnp.maximum(per_tile(q_lens), 1)[:, None]
        token = jnp.clip(offset // num_heads, 0, q_len - 1)
        return (per_tile(seq_lens)[:, None] - q_len + token).reshape(-1)

    def class_picks(scored, block_q, transposed):
        # A class's scores are padded to its worst-case tile count (the large
        # class: 66 tiles of 512 at 8K tokens for ~14 real ones); pick in
        # its live tiles only.
        s, b = candidates(
            scored.scores if transposed else scored.scores[:, None, :],
            positions(scored.tile_seq, scored.tile_row0, block_q),
            live_rows=scored.num_tiles * block_q,
            tile_major=scored.scores.ndim == 3,
        )
        return pack_candidates(s, b)

    small = plan.small()
    small_picks = class_picks(small, plan.small_block_q, not plan.token_major)
    prefill_plan = class_plan(**options | {"pages_per_step": _PREFILL_PAGES_PER_STEP})
    fits = functools.partial(
        _prefill_fits,
        prefill_plan,
        index_cache,
        small_picks.shape[0],
        num_tokens * num_heads,
        topk,
    )
    # Without the key cache if only that does not fit: it is as wide as the
    # block tables (max-model-len), whatever the step's context.
    key_cache = msa_indexer.keys_cacheable(
        plan.large_block_q, index_cache.shape[1], index_cache.dtype
    ) and fits(True)
    if key_cache or fits(False):
        packed = _prefill_candidates(
            prefill_plan,
            index_cache,
            block_tables,
            folded_cu,
            positions,
            small,
            small_picks,
            num_tokens * num_heads,
            pick=dict(
                block_size=block_size,
                topk=topk,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                dcp=dcp,
                col_bits=_key_bits(prefill_plan.large_pages),
                real_cols=prefill_plan.large_pages,
            ),
            rank=rank,
            key_cache=key_cache,
            interpret=interpret,
        )  # [T * G, 2 * topk], (token, head) rows
        return packed.reshape(num_tokens, -1)
    picks, cols = [small_picks], []
    large = plan.large()
    picks.append(class_picks(large, plan.large_block_q, True))
    for scored, block_q, picked in (
        (small, plan.small_block_q, picks[0]),
        (large, plan.large_block_q, picks[1]),
    ):
        # Each head's and token's column in this class (`_class_scores`).
        start = msa_indexer.lookup(scored.tile_start, seq_of_token)[None]
        off = rows - start
        col = (
            msa_indexer.lookup(scored.first_tile, seq_of_token)[None] + off // block_q
        ) * block_q
        cols.append(jnp.clip(col + off % block_q, 0, picked.shape[0] - 1))
    # A head's and token's picks: its large-class column if its request is
    # large, else its small-class one.
    col = jnp.where(
        msa_indexer.lookup(plan.is_large, seq_of_token)[None],
        picks[0].shape[0] + cols[1],
        cols[0],
    )
    packed = jnp.concatenate(picks, axis=0)[col.reshape(-1)]  # [G * T, 2 * topk]
    return (
        packed.reshape(num_heads, num_tokens, -1)
        .transpose(1, 0, 2)
        .reshape(num_tokens, -1)
    )


def msa_write_and_candidates_dcp(
    k: jax.Array | msa.QuantizedKV,
    v: jax.Array | None,
    index_q: jax.Array,
    index_k: jax.Array,
    kv_cache: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    rank: int | jax.Array,
    dcp: int,
    scale_cache: jax.Array | None = None,
    index_q_rotated: bool = False,
    after_write: Callable[[jax.Array, jax.Array | None], Any] | None = None,
    prefill: bool | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """First half of the DCP forward: write this rank's tokens, score its
    blocks and pick its candidates.

    `after_write(kv_cache, scale_cache)` (a step that writes its caches before
    the indexer) starts work on the written caches that is to run under the
    indexer (`msa_dcp_heads.start_first_round_exchange`); its result is
    returned last, ordered after the indexer.

    `prefill` overrides the choice of the prefill (tile class) candidate path
    by this call's rows (`msa_indexer.is_prefill_step`): one token half of a
    prefill-sized step takes the whole step's path (StepHalves), not the
    decode path's conditional, which waits for neighbouring collectives.

    With 4-bit caches (`msa_fp4`, one KV head per rank) pass `scale_cache`;
    it is returned updated after `index_cache`. `index_q` is then rotated
    here (`msa_fp4.rotate_queries`) unless `index_q_rotated`.

    Shapes as `msa.msa_forward`, except that a rank may hold ``G >= 1`` KV
    heads (DCP wider than a KV head's replicas): ``k``/``v``/``index_q
    [T, G, D]``, cache rows ``[K_0 | V_0 | ... | K_G-1 | V_G-1]``, and the
    single index-key head shared by all; with 4-bit caches `k` may come
    quantized (`msa.QuantizedKV`, `v` None). ``block_tables`` are vLLM's logical
    DCP blocks of ``dcp * block_size`` tokens. Returns ``(kv_cache,
    index_cache, candidates [T, G * 2 * topk] f32)``: `pack_candidates` per
    head, head-major.
    """
    num_tokens, num_heads, _ = index_q.shape
    rank = rank_array(rank)[0]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    pages = owner_pages(
        block_tables_2d,
        seq_id,
        pos,
        valid,
        block_size=block_size,
        rank=rank,
        dcp=dcp,
    )
    fp4 = scale_cache is not None
    scale_row = msa_fp4.index_scale_row(num_heads) if fp4 else None
    rotate = False
    if fp4:
        targets = msa.fp4_targets(
            kv_cache, index_cache, scale_cache, k, v, index_k, interpret=interpret
        )
        rotate = not index_q_rotated
    else:
        kv_rows = jnp.concatenate([k, v], axis=-1).reshape(num_tokens, -1)
        targets = [
            msa_write.Rows(kv_cache, kv_rows),
            msa_write.Rows(index_cache, index_k, dim_major=True),
        ]
    # A decode step of one KV head writes its caches inside the indexer kernel
    # (`decode_candidates`).
    max_runs = msa.write_runs_bound(num_tokens, max_num_seqs, block_size)
    # A folded write keeps every run's pages in VMEM, in and out.
    run_bytes = 2 * sum(t.cache[0].size * t.cache.dtype.itemsize for t in targets)
    if prefill is None:
        prefill = msa_indexer.is_prefill_step(num_heads * num_tokens, max_num_seqs)
    fold_write = (
        num_heads == 1
        and not prefill
        and decodes_in_one_kernel(
            num_tokens,
            index_cache,
            max_num_seqs,
            block_tables.shape[0] // max_num_seqs,
        )
        and max_runs * run_bytes <= _FOLDED_WRITE_VMEM_BYTES
    )
    if not fold_write:
        kv_cache, index_cache, *rest = msa.write_caches(
            targets, pages, pos, max_num_seqs=max_num_seqs, interpret=interpret
        )
        if fp4:
            (scale_cache,) = rest
    pending = ()
    if after_write is not None:
        if fold_write:
            raise ValueError("after_write needs the write ahead of the indexer")
        # The barriers keep the started work ahead of the indexer and its
        # completion behind it: XLA would otherwise schedule the two halves
        # of an async exchange back to back.
        index_q, started = jax.lax.optimization_barrier(
            (index_q, after_write(kv_cache, scale_cache))
        )
        pending = (started,)

    def done(*outs):
        if not pending:
            return outs
        outs, started = jax.lax.optimization_barrier((outs, pending[0]))
        return (*outs, started)

    local_lens = local_seq_lens(seq_lens, block_size=block_size, rank=rank, dcp=dcp)
    candidates = functools.partial(
        local_candidates,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=dcp,
        interpret=interpret,
    )
    if rotate and not prefill:
        index_q = msa_fp4.rotate_queries(index_q)
    if prefill:
        packed = _class_candidates(
            rotate_folded_queries(index_q, interpret=interpret)
            if rotate
            else index_q.reshape(num_tokens * num_heads, -1),
            num_heads,
            index_cache,
            seq_lens,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            seq_id,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            rank=rank,
            dcp=dcp,
            interpret=interpret,
            scale_cache=scale_cache,
            scale_row=scale_row,
        )
        if fp4:
            return done(kv_cache, index_cache, scale_cache, packed)
        return done(kv_cache, index_cache, packed)
    if fold_write:
        caches, packed = decode_candidates(
            index_q,
            index_cache,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            seq_id,
            pos,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            rank=rank,
            dcp=dcp,
            write=DecodeWrite(
                msa.cache_targets(targets),
                pages,
                pos % block_size,
                max_runs,
                index_target=1,
            ),
            interpret=interpret,
        )
        return (*caches, packed)
    if num_heads > 1 and decodes_in_one_kernel(
        num_tokens * num_heads,
        index_cache,
        max_num_seqs,
        block_tables.shape[0] // max_num_seqs,
        num_heads,
    ):
        # All heads' rows in one kernel, as one head's with its write
        # (`fold_write`); the write stays ahead (`after_write`).
        packed = decode_candidates(
            index_q,
            index_cache,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            seq_id,
            pos,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            rank=rank,
            dcp=dcp,
            interpret=interpret,
        )
        if fp4:
            return done(kv_cache, index_cache, scale_cache, packed)
        return done(kv_cache, index_cache, packed)
    if num_heads == 1:
        scores = msa._block_scores(
            index_q,
            index_cache,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            interpret,
            scale_cache,
            scale_row,
        )
    else:
        # All heads in one pass over this rank's index pages: [G, T, P],
        # head-major rows for the selection.
        scores = msa.partial_block_scores(
            index_q,
            index_cache,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            shards=1,
            interpret=interpret,
            scale_cache=scale_cache,
            scale_row=scale_row,
        ).reshape(num_heads * num_tokens, 1, -1)
    cand_scores, cand_blocks = candidates(scores, jnp.tile(pos, num_heads))
    packed = pack_candidates(cand_scores, cand_blocks)  # [G * T, 2 * topk]
    packed = packed.reshape(num_heads, num_tokens, -1).transpose(1, 0, 2)
    if fp4:
        return done(kv_cache, index_cache, scale_cache, packed.reshape(num_tokens, -1))
    return done(kv_cache, index_cache, packed.reshape(num_tokens, -1))


def msa_attend_dcp(
    q: jax.Array,
    kv_cache: jax.Array,
    candidates: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    sm_scale: float,
    topk: int,
    rank: int | jax.Array,
    dcp: int,
    scale_cache: jax.Array | None = None,
    q_head_major: bool = False,
    head_order: jax.Array | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array] | list:
    """Second half of the DCP forward: attend over the selected blocks this
    rank holds.

    With a 4-bit `kv_cache` pass its `scale_cache`; `q` must then be rotated
    (`msa_fp4.rotate_queries`) and the partial output is in the rotated
    basis (callers rotate their own queries before the gather and unrotate
    their own merged heads, a 1 / DCP share of the group's each).

    ``q [T, G * Hg, D]`` holds every query head of the group's ``G`` KV heads
    (KV head ``h`` owns query heads ``[h * Hg, (h + 1) * Hg)``), or ``[G * Hg,
    T, D]`` with `q_head_major`; ``candidates [dcp, T, G * 2 * topk]`` every
    rank's candidates. Returns
    the partial head-major, as the kernel writes it, ``(out [G * Hg, T, D],
    lse [G * Hg, T] f32)``: the LSE merge reduce-scatters heads along the
    leading axis (`ops.msa.merge_head_partials`), or `attend_and_merge`
    sends each rank its block (a pair: a permute of the peer's heads).
    ``q`` may be a pair of head-major arrays holding the heads in order.
    With `head_order` (``int32[G]``; 4-bit, several KV heads, decode-sized
    steps) one kernel call per KV head in that order instead, as thunks
    (`msa_attention_lse.msa_sparse_attention_lse_head_calls`).
    """
    num_tokens = jax.tree.leaves(q)[0].shape[1 if q_head_major else 0]
    rank = rank_array(rank)[0]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    pages_per_seq = block_tables_2d.shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    small = num_tokens <= DCP_SMALL_BATCH_TOKENS
    if dcp == 2:
        selected = merge_pair_candidates(candidates, topk)  # [T, G, topk]
        if not small:
            # One fusion: XLA sank the work list's [tiles, union] reshape into
            # its inputs as four relayout copies (~11 us per layer at 8K
            # tokens, c4 twin j-8f751cd8).
            selected = jax.lax.optimization_barrier(selected)
    else:
        selected = merge_head_candidates(candidates, topk, interpret)
    num_heads = selected.shape[1]
    own_block = pos // block_size
    local_key = jnp.where(
        valid & (own_block % dcp == rank),
        seq_id * pages_per_seq + own_block // dcp,
        -1,
    )
    # Page keys (seq * pages_per_seq + logical block) of the selected blocks
    # this rank holds, as `msa.sparse_attention`; [T, G, topk].
    mine = (selected >= 0) & (selected % dcp == rank) & valid[:, None, None]
    keys = jnp.where(mine, seq_id[:, None, None] * pages_per_seq + selected // dcp, -1)
    if small:
        block_q = msa_attention.SMALL_BATCH_BLOCK_Q
        pages_per_step = _SMALL_BATCH_PAGES_PER_STEP
    elif scale_cache is not None:
        block_q = msa_attention.PREFILL_BLOCK_Q_4BIT
        pages_per_step = msa_attention.PREFILL_PAGES_PER_STEP_4BIT
    else:
        block_q = msa_attention.PREFILL_BLOCK_Q
        pages_per_step = msa_attention.PREFILL_PAGES_PER_STEP
    common = dict(
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        q_head_major=q_head_major,
        interpret=interpret,
    )
    args = (local_key, pos % block_size + 1, block_tables.reshape(-1))
    if head_order is not None:
        if scale_cache is None or num_heads == 1 or q_head_major:
            raise ValueError("per-head calls serve token-major 4-bit groups")
        return msa_attention_lse.msa_sparse_attention_lse_head_calls(
            q,
            kv_cache,
            keys,
            *args,
            head_order,
            sm_scale=sm_scale,
            block_q=block_q,
            pages_per_step=pages_per_step,
            scale_cache=scale_cache,
            interpret=interpret,
        )
    if scale_cache is not None:
        if num_heads == 1:
            return msa_attention_lse.msa_sparse_attention_lse(
                q,
                kv_cache,
                keys[:, 0],
                *args,
                head=0,
                head_major=True,
                **common,
                **msa._fp4_attention_kwargs(scale_cache),
            )
        # KV head h's K / V scales are scale-page rows 2h / 2h + 1.
        return msa_attention_lse.msa_sparse_attention_lse_heads(
            q, kv_cache, keys, *args, head_major=True, **common, scale_cache=scale_cache
        )
    if num_heads == 1:
        return msa_attention_lse.msa_sparse_attention_lse(
            q, kv_cache, keys[:, 0], *args, head=0, head_major=True, **common
        )
    # One kernel for all the group's KV heads (the head is a kernel input).
    return msa_attention_lse.msa_sparse_attention_lse_heads(
        q, kv_cache, keys, *args, head_major=True, **common
    )


_PARTIALS_BLOCK_TOKENS = 256
# The pair merges stream their partials: shorter blocks pay a per-step DMA cost.
_PAIR_PARTIALS_BLOCK_TOKENS = 1024
# Their double-buffered blocks (two partials in, one output) stay within this
# at any head count.
_PAIR_BLOCK_VMEM_BYTES = 16 * 1024 * 1024


def _pair_block_tokens(
    num_tokens: int, num_heads: int, head_dim: int, itemsize: int, block_tokens: int
) -> int:
    """A pair merge's token block: `block_tokens`, fewer (whole multiples of
    `_PARTIALS_BLOCK_TOKENS`) when its buffers would pass
    `_PAIR_BLOCK_VMEM_BYTES`."""
    token_bytes = 2 * 3 * num_heads * head_dim * itemsize
    fit = _PAIR_BLOCK_VMEM_BYTES // (token_bytes * _PARTIALS_BLOCK_TOKENS)
    fit = max(1, fit) * _PARTIALS_BLOCK_TOKENS
    return max(1, min(num_tokens, block_tokens, fit))


def _merge_partials_kernel(lse_ref, parts_ref, *rest, rotated):
    """One token block of `merge_partials`: ``lse_ref [dcp, Bt, Hl]``,
    ``parts_ref [dcp, Hl, Bt, D]``, (``hadamard [D, D]``), ``o_ref [Bt,
    Hl * D]``."""
    o_ref = rest[-1]
    dcp, num_heads, _, head_dim = parts_ref.shape
    lse = lse_ref[...]
    # torch.logsumexp over the ranks; a token no rank attended (every lse
    # -inf) gets NaN weights, which the where below turns into zeros.
    peak = jnp.max(lse, axis=0)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    total = peak + jnp.log(jnp.sum(jnp.exp(lse - peak[None]), axis=0))
    weight = jnp.exp(lse - total[None])
    weight = jnp.where(weight > 0, weight, 0.0)  # NaN (no rank attended) -> 0
    for h in range(num_heads):
        merged = None
        for r in range(dcp):
            w = jnp.broadcast_to(weight[r, :, h : h + 1], parts_ref.shape[2:])
            # A plain product: a rank that attended nothing for a token has
            # a zero partial (`msa_sparse_attention_lse`), never NaN, so no
            # select guards it (a compare and a select per element, ~20% of
            # the kernel's bundles).
            term = parts_ref[r, h].astype(jnp.float32) * w
            merged = term if merged is None else merged + term
        merged = merged.astype(o_ref.dtype)
        if rotated:
            merged = jnp.dot(merged, rest[0][...], preferred_element_type=jnp.float32)
            merged = (merged * (1.0 / head_dim)).astype(o_ref.dtype)
        o_ref[:, h * head_dim : (h + 1) * head_dim] = merged


@functools.partial(jax.jit, static_argnames=("block_tokens", "interpret"))
def merge_partials(
    parts: jax.Array,
    lse: jax.Array,
    hadamard: jax.Array | None = None,
    *,
    block_tokens: int = _PARTIALS_BLOCK_TOKENS,
    interpret: bool = False,
) -> jax.Array:
    """The log-sum-exp merge of every DCP rank's partial attention output of
    this rank's heads, token-major: ``parts [dcp, Hl, T, D]`` (rank ``r``'s
    head-major partial), ``lse [dcp, T, Hl]`` f32 -> ``[T, Hl * D]``.

    Weights and sum in f32, rounded once; a rank's partial of a token it
    attended nothing for must be zero (as `msa_sparse_attention_lse` writes
    it), not NaN. A 4-bit cache's partials (in the rotated basis) are
    unrotated by its bf16 `hadamard` (``x @ H / D``, as
    `ops.msa._rotate_heads`). One pass that writes the layout the output
    projection reads: the reduce-scatter merge weighted the whole ``[dcp *
    Hl, T, D]`` partial and XLA relaid it out for the token-major output
    (~43 + 24 us per layer at 8K tokens, c4 twin j-db4e365d)."""
    dcp, num_heads, num_tokens, head_dim = parts.shape
    block_t = max(1, min(num_tokens, block_tokens))
    rotated = hadamard is not None
    return pl.pallas_call(
        functools.partial(_merge_partials_kernel, rotated=rotated),
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=[
            pl.BlockSpec((dcp, block_t, num_heads), lambda i: (0, i, 0)),
            pl.BlockSpec((dcp, num_heads, block_t, head_dim), lambda i: (0, 0, i, 0)),
            *(
                [pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0))]
                if rotated
                else []
            ),
        ],
        out_specs=pl.BlockSpec((block_t, num_heads * head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), parts.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge_partials",
    )(lse.astype(jnp.float32), parts, *([hadamard] if rotated else []))


def _merge_pair_kernel(lse_ref, parts_ref, *rest, rotated):
    """One token block of `merge_pair_partials`: ``lse_ref [2, Hl, Bt]``,
    ``parts_ref [2, Hl, Bt, D]``, (``hadamard [D, D]``), ``o_ref [Bt, Hl *
    D]``."""
    o_ref = rest[-1]
    if isinstance(parts_ref, tuple):  # own partial and the peer's, apart
        (lse0_ref, lse1_ref), (p0_ref, p1_ref) = lse_ref, parts_ref
        num_heads, _, head_dim = p1_ref.shape
        lse0, lse1 = lse0_ref[...], lse1_ref[...]
    else:
        _, num_heads, _, head_dim = parts_ref.shape
        lse0, lse1 = lse_ref[0], lse_ref[1]
        p0_ref, p1_ref = parts_ref.at[0], parts_ref.at[1]
    # `_merge_partials_kernel`'s weight of rank 0, lane-dense.
    peak = jnp.maximum(lse0, lse1)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    total = peak + jnp.log(jnp.exp(lse0 - peak) + jnp.exp(lse1 - peak))
    weight = jnp.exp(lse0 - total)
    weight = jnp.where(weight > 0, weight, 0.0)  # NaN -> 0
    for h in range(num_heads):
        p0 = p0_ref[h].astype(jnp.float32)
        p1 = p1_ref[h].astype(jnp.float32)
        # The head's weights across the lanes as the transpose of their
        # sublane broadcast: AOT 634 bundles per 256 tokens, against 777 for
        # lane broadcasts of a [Bt, Hl] transpose and 1,137 for
        # `merge_partials`' two per head (XLU-bound).
        w = jnp.broadcast_to(weight[h : h + 1], (head_dim, weight.shape[1])).T
        # w * p0 + (1 - w) * p1, one weight per head. A rank that attended
        # nothing has a zero partial, never NaN.
        merged = (p1 + w * (p0 - p1)).astype(o_ref.dtype)
        if rotated:
            merged = jnp.dot(merged, rest[0][...], preferred_element_type=jnp.float32)
            merged = (merged * (1.0 / head_dim)).astype(o_ref.dtype)
        o_ref[:, h * head_dim : (h + 1) * head_dim] = merged


@functools.partial(jax.jit, static_argnames=("block_tokens", "interpret"))
def merge_pair_partials(
    parts: jax.Array,
    lse: jax.Array,
    hadamard: jax.Array | None = None,
    *,
    block_tokens: int = _PAIR_PARTIALS_BLOCK_TOKENS,
    interpret: bool = False,
) -> jax.Array:
    """`merge_partials` of a DCP pair with head-major LSEs: ``parts [2, Hl,
    T, D]``, ``lse [2, Hl, T]`` f32 (as the partial's ``[2 * Hl, T]``
    reshapes, no transpose) -> ``[T, Hl * D]``.

    The weights as `merge_partials`'; the sum ``p1 + w0 * (p0 - p1)`` rounds
    differently from ``w0 * p0 + w1 * p1``, at bf16 level. `merge_partials`
    took 37-39 us per layer at 8K tokens (twins j-2e5da13f, j-d33c9244), and
    its token-major LSEs a transposing copy."""
    _, num_heads, num_tokens, head_dim = parts.shape
    block_t = _pair_block_tokens(
        num_tokens, num_heads, head_dim, parts.dtype.itemsize, block_tokens
    )
    rotated = hadamard is not None
    return pl.pallas_call(
        functools.partial(_merge_pair_kernel, rotated=rotated),
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=[
            pl.BlockSpec((2, num_heads, block_t), lambda i: (0, 0, i)),
            pl.BlockSpec((2, num_heads, block_t, head_dim), lambda i: (0, 0, i, 0)),
            *(
                [pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0))]
                if rotated
                else []
            ),
        ],
        out_specs=pl.BlockSpec((block_t, num_heads * head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), parts.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge_pair_partials",
    )(lse.astype(jnp.float32), parts, *([hadamard] if rotated else []))


def merge_split_partials(
    out: jax.Array,
    peer: jax.Array,
    lse: jax.Array,
    peer_lse: jax.Array,
    hadamard: jax.Array | None = None,
    *,
    block_tokens: int = _PAIR_PARTIALS_BLOCK_TOKENS,
    interpret: bool = False,
) -> jax.Array:
    """`merge_pair_partials` of this rank's partial ``out [2 Hl, T, D]``
    (its own heads first: their rows are read in place) and the peer's
    partial of those heads ``peer [Hl, T, D]``, with ``lse [2 Hl, T]`` and
    ``peer_lse [Hl, T]`` f32 -> ``[T, Hl * D]``: no stacking copy of the two."""
    num_heads, num_tokens, head_dim = peer.shape
    block_t = _pair_block_tokens(
        num_tokens, num_heads, head_dim, peer.dtype.itemsize, block_tokens
    )
    rotated = hadamard is not None

    def kernel(lse0_ref, lse1_ref, p0_ref, p1_ref, *rest):
        _merge_pair_kernel(
            (lse0_ref, lse1_ref), (p0_ref, p1_ref), *rest, rotated=rotated
        )

    heads = pl.BlockSpec((num_heads, block_t, head_dim), lambda i: (0, i, 0))
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=[
            pl.BlockSpec((num_heads, block_t), lambda i: (0, i)),
            pl.BlockSpec((num_heads, block_t), lambda i: (0, i)),
            heads,
            heads,
            *(
                [pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0))]
                if rotated
                else []
            ),
        ],
        out_specs=pl.BlockSpec((block_t, num_heads * head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), out.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge_split_partials",
    )(
        lse.astype(jnp.float32),
        peer_lse.astype(jnp.float32),
        out,
        peer,
        *([hadamard] if rotated else []),
    )


def _gather_heads(q: jax.Array, rank: jax.Array, axis: str, dcp: int) -> jax.Array:
    """The group's head-major query heads ``[dcp * Hl, T, D]`` from this
    rank's ``q [Hl, T, D]``. A pair swaps its halves by one collective
    permute: XLA's all-gather of the 16 MB c4 / c16 queries took 158 us per
    layer in the twin (j-094494a7), a permute plus the concatenation ~51 us
    on the same core pair in a standalone bench (j-a7bf7869; all-gather ~90)."""
    if dcp != 2:
        return jax.lax.all_gather(q, axis, axis=0, tiled=True)
    other = jax.lax.ppermute(q, axis, [(0, 1), (1, 0)])
    # [q, other, q] from row `rank * Hl` is the pair's heads in rank order.
    both = jnp.concatenate([q, other, q])
    return jax.lax.dynamic_slice_in_dim(both, rank * q.shape[0], 2 * q.shape[0])


def _decode_plan_kernel(
    rank_ref,
    cu_ref,
    seq_lens_ref,
    peer_ref,
    own_ref,
    keys_ref,
    local_key_ref,
    local_limit_ref,
    work_ref,
    *,
    topk,
    block_size,
    pages_per_seq,
    block_q,
    work,
):
    """A DCP pair's decode plan, in VMEM, from this rank's and its peer's
    packed candidates ``[T, 2 * topk]`` (the peer's as the 16-bit planes of
    `words_to_u16_planes`, ``[T, >= 4 * topk]``) and the requests' query
    starts and lengths (`cu_ref`, `seq_lens_ref` in SMEM): the merged
    selection (`merge_candidates`' dense ranking), this rank's page keys
    ``[T, topk]``, each token's local page and limit ``[T, 1]``
    (`msa.token_positions`) and the attention call's work list
    (`msa_attention_lse.store_work_list`). The same values as the XLA path
    of `msa_attend_dcp`, which costs a dozen small ops of fixed cost each;
    `msa_attention_lse.planned_attention` runs it before attending."""
    dcp = 2
    rank = rank_ref[0]
    num_tokens = own_ref.shape[0]
    iota = functools.partial(jax.lax.broadcasted_iota, jnp.int32)
    # Candidates down the sublanes, tokens across the lanes: candidate j's
    # score and block broadcast over sublanes for the compares.
    # The peer's planes transposed: its low and high halves are then rows
    # 4 * topk apart, which join without a lane shift.
    width = own_ref.shape[1]
    planes = peer_ref[...].astype(jnp.uint32).T  # [D, T]
    peer = planes[:width] | (planes[width : 2 * width] << 16)
    halves = []
    for c in (own_ref[...].T, jax.lax.bitcast_convert_type(peer, jnp.float32)):
        halves.append((c[:topk], c[topk : 2 * topk].astype(jnp.int32)))
    places = []
    for s_i, b_i in halves:
        ahead = jnp.zeros(s_i.shape, jnp.int32)
        for s_all, b_all in halves:
            for j in range(topk):
                s_j, b_j = s_all[j : j + 1], b_all[j : j + 1]
                before = (s_j > _NEG_INF) & ((s_j > s_i) | ((s_j == s_i) & (b_j < b_i)))
                ahead += before.astype(jnp.int32)
        places.append(jnp.where(s_i > _NEG_INF, ahead, topk))
    # Candidate i goes to slot place[i] (the real ones' places are distinct):
    # a select per candidate, no reduction.
    slot = iota((topk, num_tokens), 0)
    picked = jnp.full((topk, num_tokens), -1, jnp.int32)
    for (_, b_i), place in zip(halves, places):
        for i in range(topk):
            picked = jnp.where(slot == place[i : i + 1], b_i[i : i + 1], picked)
    # Token positions with tokens across the lanes, from the query starts
    # and lengths as scalars: compares and selects, no lane reductions.
    num_seqs = seq_lens_ref.shape[0]
    tok = iota((1, num_tokens), 1)
    seq_id = jnp.zeros((1, num_tokens), jnp.int32)
    for s in range(1, num_seqs + 1):
        seq_id += jnp.where(cu_ref[s] <= tok, 1, 0)
    seq_id = jnp.minimum(seq_id, num_seqs - 1)
    start = jnp.zeros_like(seq_id)
    end = jnp.zeros_like(seq_id)
    seq_len = jnp.zeros_like(seq_id)
    for s in range(num_seqs):
        here = seq_id == s
        start = jnp.where(here, cu_ref[s], start)
        end = jnp.where(here, cu_ref[s + 1], end)
        seq_len = jnp.where(here, seq_lens_ref[s], seq_len)
    valid = tok < cu_ref[num_seqs]
    pos = jnp.where(valid, seq_len - (end - start) + (tok - start), 0)
    own_block = pos // block_size
    base = seq_id * pages_per_seq
    local_key = jnp.where(
        valid & (own_block % dcp == rank), base + own_block // dcp, -1
    )
    mine = (picked >= 0) & (picked % dcp == rank) & valid
    keys = jnp.where(mine, base + picked // dcp, -1)  # [K, T]
    # One transpose turns all three into a token per row, the keys repeated
    # across the lanes (k, k + K, ...): token t's then sit at lanes
    # [t * K, (t + 1) * K) of its tile's row.
    rows = jnp.concatenate(
        [keys] * (_LANES // topk)
        + [local_key, pos % block_size + 1, jnp.zeros((6, num_tokens), jnp.int32)]
    )
    cols = rows.T  # [T, 128 + 8]
    selected = cols[:, :_LANES]
    keys_ref[...] = cols[:, :topk]
    local_key_ref[...] = cols[:, _LANES : _LANES + 1]
    local_limit_ref[...] = cols[:, _LANES + 1 : _LANES + 2]
    num_tiles = num_tokens // block_q
    token_of_lane = iota((1, _LANES), 1) // topk
    tile_rows = []
    for i in range(num_tiles):
        row = jnp.full((1, _LANES), -1, jnp.int32)
        for t in range(block_q):
            token = selected[i * block_q + t : i * block_q + t + 1, :]
            row = jnp.where(token_of_lane == t, token, row)
        tile_rows.append(row)
    msa_attention_lse.store_work_list(tile_rows, work_ref, **work)


def use_decode_plan(
    num_tokens: int, topk: int, num_heads: int, head_dim: int, num_keys: int
) -> bool:
    """Whether a DCP pair's decode step of ``num_tokens`` rows of
    `num_heads` query heads per rank attends through `_attend_pair_decode`
    (8-token tiles whose work list the kernel builds, in one grid step, a
    tile's lse in one 128-lane row of the permute buffer, block tables of
    `num_keys` entries)."""
    block_q = msa_attention.SMALL_BATCH_BLOCK_Q
    return (
        num_tokens % block_q == 0
        and num_tokens <= _PLAN_MAX_TOKENS
        and 2 * num_heads * block_q == _LANES
        and head_dim == _LANES
        and num_keys <= msa_attention_lse.MAX_PLAN_KEYS
        and _LANES % topk == 0
        and block_q * topk == _LANES
        and msa_attention_lse.fits_work_row(block_q, topk, _SMALL_BATCH_PAGES_PER_STEP)
    )


def _requests_in_step(rank_ref, cu_ref, seq_lens_ref):
    """The requests (block-table rows) a step's tokens belong to: those that
    start before the last query ends (``cu_ref`` in SMEM)."""
    num_seqs = seq_lens_ref.shape[0]
    end = cu_ref[num_seqs]
    return sum(jnp.where(cu_ref[s] < end, 1, 0) for s in range(num_seqs))


def _attend_pair_decode(
    q,
    swap,
    kv_cache,
    candidates,
    seq_lens,
    block_tables,
    cu_q_lens,
    rank,
    *,
    sm_scale,
    topk,
    scale_cache,
    interpret,
):
    """`msa_attend_dcp` of a pair's decode step over this rank's query heads
    `q` and the peer's heads and candidates in the permuted `swap` buffer
    (`swap_buffer`), its plan (`_decode_plan_kernel`) run by the attention
    kernel itself: `msa_attention_lse.planned_attention`'s ``(out, lse,
    back)``."""
    num_heads, num_tokens, _ = q.shape
    block_q = msa_attention.SMALL_BATCH_BLOCK_Q
    pages_per_step = _SMALL_BATCH_PAGES_PER_STEP
    sections, _, tiles_per_step, groups_per_tile = msa_attention_lse.call_work_spec(
        num_tokens, 2 * num_heads, q.dtype, topk, block_q, pages_per_step
    )
    plan = functools.partial(
        _decode_plan_kernel,
        topk=topk,
        block_size=msa.page_size(kv_cache),
        pages_per_seq=block_tables.shape[0] // seq_lens.shape[0],
        block_q=block_q,
        work=dict(
            pages_per_step=pages_per_step,
            groups_per_tile=groups_per_tile,
            tiles_per_step=tiles_per_step,
            sections=sections,
        ),
    )
    i32 = jnp.int32
    return msa_attention_lse.planned_attention(
        q,
        swap,
        kv_cache,
        block_tables.reshape(seq_lens.shape[0], -1),
        plan,
        (rank_array(rank), cu_q_lens.astype(i32), seq_lens.astype(i32)),
        (candidates,),
        rows_in_use=_requests_in_step,
        topk=topk,
        sm_scale=sm_scale,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=interpret,
        **(
            {}
            if scale_cache is None
            else dict(scale_cache=scale_cache, scale_rows=(0, 1))
        ),
    )


def _merge_pair_decode_kernel(out_ref, lse_ref, back_ref, *rest, rotated):
    """`_merge_partials_kernel` of a pair's decode step: this rank's partial
    of its heads (``out_ref [Hl, T, D]``, lse ``lse_ref [1, tiles, 1, 2 Hl *
    Bq]`` lanes ``[0, Hl * Bq)``) and the peer's, from ``back_ref [Hl + 1,
    T, D]`` (`planned_attention`'s `back`), (``hadamard [D, D]``), ``o_ref
    [T, Hl * D]``. The same arithmetic in the same order."""
    o_ref = rest[-1]
    num_heads, num_tokens, head_dim = out_ref.shape
    num_tiles = lse_ref.shape[1]
    block_q = num_tokens // num_tiles
    half = num_heads * block_q
    planes = back_ref[num_heads, : 2 * num_tiles].astype(jnp.uint32)
    peer = planes[:num_tiles] | (planes[num_tiles:] << 16)
    rows = -(-num_tiles // 8) * 8

    def columns(x):  # [tiles, 128] -> [128, tiles]: tile i's lanes down column i
        return jnp.pad(x, ((0, rows - num_tiles), (0, 0))).T[:, :num_tiles]

    # This rank's heads: lanes [0, half) of its own rows, [half, 2 half) of
    # the peer's (the peer's lanes of its own heads come second).
    lse = jnp.stack(
        [
            columns(lse_ref[0, :, 0, :])[:half],
            columns(pltpu.bitcast(peer, jnp.float32))[half:],
        ]
    )
    peak = jnp.max(lse, axis=0)
    peak = jnp.where(jnp.isinf(peak), 0.0, peak)
    total = peak + jnp.log(jnp.sum(jnp.exp(lse - peak[None]), axis=0))
    weight = jnp.exp(lse - total[None])
    weight = jnp.where(weight > 0, weight, 0.0)  # [2, half, tiles]
    parts = (out_ref, back_ref.bitcast(out_ref.dtype))
    for h in range(num_heads):
        tiles = []
        for i in range(num_tiles):
            merged = None
            for r in range(2):
                w = weight[r, h * block_q : (h + 1) * block_q, i : i + 1]
                w = jnp.broadcast_to(w, (block_q, head_dim))
                part = parts[r][h, i * block_q : (i + 1) * block_q, :]
                term = part.astype(jnp.float32) * w
                merged = term if merged is None else merged + term
            tiles.append(merged)
        merged = jnp.concatenate(tiles).astype(o_ref.dtype)
        if rotated:
            merged = jnp.dot(merged, rest[0][...], preferred_element_type=jnp.float32)
            merged = (merged * (1.0 / head_dim)).astype(o_ref.dtype)
        o_ref[:, h * head_dim : (h + 1) * head_dim] = merged


def merge_pair_decode(
    out: jax.Array,
    lse: jax.Array,
    back: jax.Array,
    hadamard: jax.Array | None = None,
    *,
    interpret: bool = False,
) -> jax.Array:
    """`merge_partials` of a pair's decode step from what the attention
    kernel wrote (`msa_attention_lse.planned_attention`): this rank's ``out
    [2 Hl, T, D]`` (its heads first) and ``lse``, and the peer's ``back``
    buffer as the permute delivered it -> ``[T, Hl * D]``. Reading the
    buffers as they are leaves no XLA op between the permute and the merge
    (c4 b1 twin: ~0.6 us per layer of 16-bit packing, LSE relayouts and
    copies around the permute, j-02cb6be8)."""
    num_heads = back.shape[0] - 1
    _, num_tokens, head_dim = out.shape
    rotated = hadamard is not None

    def whole(shape):
        return pl.BlockSpec(shape, lambda i: (0,) * len(shape))

    return pl.pallas_call(
        functools.partial(_merge_pair_decode_kernel, rotated=rotated),
        grid=(1,),
        in_specs=[
            whole((num_heads, num_tokens, head_dim)),
            whole(lse.shape),
            whole(back.shape),
            *([whole((head_dim, head_dim))] if rotated else []),
        ],
        out_specs=whole((num_tokens, num_heads * head_dim)),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), out.dtype),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge_partials",
    )(out, lse, back, *([hadamard] if rotated else []))


def swap_buffer(x: jax.Array, extra: jax.Array) -> jax.Array:
    """The 16-bit ``[H + 1, T, D]`` permute buffer of ``x [H, T, D]``
    (16-bit) and ``extra [T, n]`` (32-bit): `x`'s bits, then `extra`'s as
    one more head row (`words_to_u16_planes`; every permute start costs ~1
    us on the TensorCore at decode, c4 twin j-6bbe97e2). As 16-bit integers:
    float copies may canonicalize NaN or flush subnormal bit patterns."""
    _, num_tokens, head_dim = x.shape
    n = extra.shape[1]
    if x.dtype.itemsize != 2 or extra.dtype.itemsize != 4 or 2 * n > head_dim:
        raise ValueError(
            f"16-bit x and 32-bit extra of at most {head_dim // 2} columns, got "
            f"{x.dtype} {x.shape}, {extra.dtype} {extra.shape}"
        )
    bits = words_to_u16_planes(extra)[None]
    bits = jnp.pad(bits, ((0, 0), (0, 0), (0, head_dim - 2 * n)))
    return jnp.concatenate([jax.lax.bitcast_convert_type(x, jnp.uint16), bits])


def words_to_u16_planes(x: jax.Array) -> jax.Array:
    """32-bit ``x [..., n]`` as uint16 ``[..., 2n]``: the low halves in lanes
    ``[0, n)``, the high ones in ``[n, 2n)``. A bitcast to ``[..., n, 2]``
    interleaves them, and reading them back took two strided lane slices
    (~0.8 us each for a ``[192, 32]`` row in the c48 decode twin,
    j-1d070da1); planes come back by contiguous slices that fuse."""
    words = jax.lax.bitcast_convert_type(x, jnp.uint32)
    halves = [words & 0xFFFF, words >> 16]
    return jnp.concatenate([h.astype(jnp.uint16) for h in halves], axis=-1)


def u16_planes_to_words(planes: jax.Array, n: int, dtype) -> jax.Array:
    """Inverse of `words_to_u16_planes` for the first ``2n`` lanes of
    ``planes``: ``[..., n]`` of 32-bit `dtype`."""
    lo, hi = (planes[..., i * n : (i + 1) * n].astype(jnp.uint32) for i in (0, 1))
    return jax.lax.bitcast_convert_type(lo | (hi << 16), dtype)


def _swap(x: jax.Array, axis: str) -> jax.Array:
    """A DCP pair's `x` swapped by one collective permute."""
    return jax.lax.ppermute(x, axis, [(0, 1), (1, 0)])


def _swap_with_row(x: jax.Array, extra: jax.Array, axis: str):
    """A DCP pair's ``x [H, T, D]`` (16-bit) and ``extra [T, n]`` (32-bit)
    swapped by one collective permute of their `swap_buffer`."""
    got = _swap(swap_buffer(x, extra), axis)
    return (
        jax.lax.bitcast_convert_type(got[:-1], x.dtype),
        u16_planes_to_words(got[-1], extra.shape[1], extra.dtype),
    )


def attend_and_merge(
    rank: jax.Array,
    q: jax.Array,
    kv_cache: jax.Array,
    candidates: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    scale_cache: jax.Array | None = None,
    hadamard: jax.Array | None = None,
    *,
    axis: str,
    dcp: int,
    sm_scale: float,
    topk: int,
    interpret: bool = False,
) -> jax.Array:
    """The attention half of the DCP forward over one KV head, inside a
    ``shard_map`` over `axis` (the DCP group): this rank's head-major query
    heads ``q [Hl, T, D]`` (rotated for a 4-bit cache, then pass its
    `scale_cache` and `hadamard`) and packed ``candidates [T, 2 * topk]`` ->
    its heads' merged output ``[T, Hl * D]``.

    A prefill-sized step all-gathers the query heads and candidates, runs
    `msa_attend_dcp` over the group's heads, and sends each rank's head block
    of the partial and its LSEs to their owner by all-to-all for
    `merge_partials`. As XLA collectives in one program they move the
    head-major buffers as laid out; torch's all-to-all of them cost ~890 us
    per layer at 8K tokens (c4 twin, j-c60806bf). A pair (`SPLIT_PAIR_QUERIES`)
    swaps the queries by a permute, attends the two halves in place, and
    permutes back the peer's heads' partials (`merge_split_partials`). A
    decode-sized step of a
    pair swaps own and peer halves by permutes instead (`_swap_with_row`),
    and the shortest of them plan inside the attention kernel
    (`_attend_pair_decode`)."""
    num_heads, num_tokens, head_dim = q.shape
    rank = rank_array(rank)[0]
    # Decode-sized operands of a pair move by permutes (latency), and its
    # candidates' bits ride the query permute as one more 16-bit head row.
    small = dcp == 2 and num_tokens <= DCP_SMALL_BATCH_TOKENS and 4 * topk <= head_dim
    # (8-row LSE blocks of the merge; a 0-token step keeps the plain path.)
    pair = (
        dcp == 2
        and not small
        and SPLIT_PAIR_QUERIES
        and num_heads % 8 == 0
        and num_tokens > 0
    )
    if small and use_decode_plan(
        num_tokens, topk, num_heads, head_dim, block_tables.size
    ):
        # The attention reads the permuted buffer in place and writes the one
        # it permutes back, which the merge reads as delivered (the order-free
        # selection and the commuting merge as below).
        swap = _swap(swap_buffer(q, candidates), axis)
        out, lse, back = _attend_pair_decode(
            q,
            swap,
            kv_cache,
            candidates,
            seq_lens,
            block_tables,
            cu_q_lens,
            rank,
            sm_scale=sm_scale,
            topk=topk,
            scale_cache=scale_cache,
            interpret=interpret,
        )
        return merge_pair_decode(
            out, lse, _swap(back, axis), hadamard, interpret=interpret
        )
    if small:
        # Own heads and candidates first, the peer's second: every slice is
        # static (rank-indexed updates and slices of the permute buffers were
        # 13 extra ops per layer, ~0.6-1.3 us each in served layers 4-59,
        # c4 j-9986c92e). The selection is order-free (distinct blocks by
        # score) and the pair's two-term merge commutes, so the result is the
        # rank-ordered one bit for bit.
        # One permute carries the queries and the candidates.
        peer_q, peer_candidates = _swap_with_row(q, candidates, axis)
        q_all = jnp.concatenate([q, peer_q])
    elif pair:
        # Own heads first, the peer's second, as two arrays the attention
        # reads in place (no concatenated copy); the partials of the peer's
        # heads go back to it by one permute (the merge commutes).
        q_all = (q, jax.lax.ppermute(q, axis, [(0, 1), (1, 0)]))
    else:
        q_all = _gather_heads(q, rank, axis, dcp)
    if dcp == 2 and not small:
        # `merge_pair_candidates` reads the pair in either order: own
        # first, by one permute (XLA's all-gather of the [T, 2 * topk]
        # candidates blocked the TensorCore 15 us per layer at 8K tokens,
        # j-2e5da13f).
        peer_candidates = jax.lax.ppermute(candidates, axis, [(0, 1), (1, 0)])
    cand_all = (
        jnp.stack([candidates, peer_candidates])
        if dcp == 2
        else jax.lax.all_gather(candidates, axis, axis=0)
    )
    out, lse = msa_attend_dcp(
        q_all,
        kv_cache,
        cand_all,
        seq_lens,
        block_tables,
        cu_q_lens,
        sm_scale=sm_scale,
        topk=topk,
        rank=rank,
        dcp=dcp,
        scale_cache=scale_cache,
        q_head_major=True,
        interpret=interpret,
    )
    if small:
        # The peer's partial of this rank's heads (the back half of its own:
        # its heads came first) and their LSEs, in one permute.
        peer_out, peer_lse = _swap_with_row(out[num_heads:], lse[num_heads:].T, axis)
        parts = jnp.stack([out[:num_heads], peer_out])
        lses = jnp.stack([lse[:num_heads].T, peer_lse])
        return merge_partials(parts, lses, hadamard, interpret=interpret)
    if pair:
        swap = functools.partial(
            jax.lax.ppermute, axis_name=axis, perm=[(0, 1), (1, 0)]
        )
        return merge_split_partials(
            out,
            swap(out[num_heads:]),
            lse,
            swap(lse[num_heads:]),
            hadamard,
            interpret=interpret,
        )
    # Block r of each result came from rank r: its partial of this rank's heads.
    parts = out.reshape(dcp, num_heads, num_tokens, head_dim)
    lses = lse.reshape(dcp, num_heads, num_tokens)
    parts = jax.lax.all_to_all(parts, axis, 0, 0, tiled=True)
    lses = jax.lax.all_to_all(lses, axis, 0, 0, tiled=True)
    if dcp == 2:
        return merge_pair_partials(parts, lses, hadamard, interpret=interpret)
    return merge_partials(parts, lses.transpose(0, 2, 1), hadamard, interpret=interpret)
