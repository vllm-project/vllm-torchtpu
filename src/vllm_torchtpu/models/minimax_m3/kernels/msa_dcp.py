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
from typing import NamedTuple

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
# The narrowest column count `local_candidates` scans (a vreg's lanes), and
# the fewest tokens for which it picks a width per row block.
_MIN_CANDIDATE_WIDTH = 128
_MIN_CANDIDATE_TOKENS = 1024

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
# Pages per kernel step of those 8-token tiles in batches of more than
# `_WIDE_GROUP_MIN_TOKENS` tokens: a tile's 128 query rows keep a 32-page
# score tile the size of a 16-token tile's 16-page one (AOT: 32 pages ~3,000
# bundles vs 2 x ~1,700 for two 16-page steps). Batches of up to 128 tokens
# (c4 / c16 decode) keep 16-page groups: there the 32-page ones saved at most
# 1.4 us per layer (c16 b6, j-5eda23f9 vs j-c93c225d), and their other
# accumulation order re-rolled the twin's dummy-weight expert routing into a
# reproducibly worse MoE balance at c4 b1 p90-100 (+0.58 ms per step, every
# core's MSA ops flat, j-eb9a7c1a vs j-fbcd168a).
_SMALL_BATCH_PAGES_PER_STEP = 32
_WIDE_GROUP_MIN_TOKENS = 129


def _small_pages_per_step(num_tokens: int) -> int:
    """Pages per kernel step of a decode-sized DCP batch's 8-token tiles."""
    if num_tokens >= _WIDE_GROUP_MIN_TOKENS:
        return _SMALL_BATCH_PAGES_PER_STEP
    return msa_attention.DEFAULT_PAGES_PER_STEP


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
    widths_ref,
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
    widths,
):
    """`msa_select._kernel` over this rank's columns (column ``c`` is block
    ``c * dcp + rank``), also returning the picked scores. The rank is read
    from SMEM so that every rank runs the same program.

    Row block ``i`` scans only its first ``widths_ref[i]`` columns (one of
    the static `widths`, covering every block its rows see): the scores span
    the whole max-model-len width, mostly past every row's context (c4
    prefill twin, j-3c651507: ~240 of 1,024 columns visible)."""
    rank = rank_ref[0]
    width_now = widths_ref[pl.program_id(0)]
    num_visible = pos_ref[...] // block_size + 1

    def select(width):
        score_out_ref[...], block_out_ref[...] = _pick(
            scores_ref[:, :width],
            num_visible,
            rank,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            dcp=dcp,
        )

    for width in widths:
        pl.when(width_now == width)(functools.partial(select, width))


def _pick(s, num_visible, rank, *, topk, init_blocks, local_blocks, dcp):
    """The top-k ``(scores, blocks)`` ``[rows, topk]`` of row-major scores
    ``s [rows, width]`` of this rank's columns (column ``c`` is block ``c *
    dcp + rank``), each row seeing its first ``num_visible [rows, 1]`` blocks."""
    rows, width = s.shape
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
    picked_scores, picked_blocks = [], []
    for _ in range(topk):
        best_score = jnp.max(s, axis=1, keepdims=True)
        best = jnp.min(jnp.where(s == best_score, cols, width), axis=1, keepdims=True)
        picked_scores.append(best_score)
        picked_blocks.append(jnp.where(best_score > _NEG_INF, best * dcp + rank, -1))
        s = jnp.where(cols == best, _NEG_INF, s)
    return jnp.concatenate(picked_scores, axis=1), jnp.concatenate(
        picked_blocks, axis=1
    )


def _pick_t(st, num_visible, rank, *, topk, init_blocks, local_blocks, dcp):
    """`_pick` of transposed scores ``st [width, rows]`` (``num_visible [1,
    rows]``), returning ``[topk, rows]`` each: every max and first index
    reduces across vregs and sublanes on the VPU. Row-major, they reduce
    across lanes on the XLU, whose latency bounds the dependent picks
    (decode, 8 to 48 rows x 512 columns: ~7,500 cycles at any row count vs
    ~4,100 here, AOT)."""
    width, rows = st.shape
    cols = jax.lax.broadcasted_iota(jnp.int32, (width, rows), 0)
    blocks = cols * dcp + rank
    visible = blocks < num_visible
    st = jnp.where(visible, st, _NEG_INF)
    st = jnp.where(
        visible & (blocks >= num_visible - local_blocks),
        msa_select._FORCED_LOCAL_SCORE,
        st,
    )
    st = jnp.where(visible & (blocks < init_blocks), msa_select._FORCED_INIT_SCORE, st)
    # Column ids as f32 (exact below 2**24): a pick's first tie is one f32 min.
    ids = cols.astype(jnp.float32)
    picked_scores, picked_blocks = [], []
    for _ in range(topk):
        best_score = jnp.max(st, axis=0, keepdims=True)
        best = jnp.min(
            jnp.where(st == best_score, ids, float(width)), axis=0, keepdims=True
        )
        picked_scores.append(best_score)
        picked_blocks.append(
            jnp.where(best_score > _NEG_INF, best.astype(jnp.int32) * dcp + rank, -1)
        )
        st = jnp.where(ids == best, _NEG_INF, st)
    return jnp.concatenate(picked_scores, axis=0), jnp.concatenate(
        picked_blocks, axis=0
    )


def _ahead(xs, xi, ys, yi):
    """Whether candidate x ranks ahead of candidate y: score descending, ties
    to the lower column (`_pick`'s order)."""
    return (xs > ys) | ((xs == ys) & (xi < yi))


def _ahead_by_selects(xs, xi, ys, yi):
    """`_ahead` as one compare of selected values (scores, or the ids on a
    tie): vector-ALU work in place of `_ahead`'s two mask-logic ops."""
    tie = xs == ys
    return jnp.where(tie, yi, xs) > jnp.where(tie, xi, ys)


def _bitonic_sort(s, i, ascending):
    """Each lane's bitonic sequence of ``8 * len(s)`` distinct candidates
    (position ``p`` at sublane ``p % 8`` of vreg ``p // 8`` of the scores `s`
    and column ids `i`, lists of ``[8, L]``) in order: best first, or last if
    `ascending`."""
    s, i = list(s), list(i)
    n = 8 * len(s)
    sublane = jax.lax.broadcasted_iota(jnp.int32, s[0].shape, 0)
    d = n // 2
    while d >= 8:  # pairs in different vregs
        for lo in range(len(s)):
            hi = lo + d // 8
            if lo % (2 * d // 8) >= d // 8:
                continue
            keep = _ahead(s[lo], i[lo], s[hi], i[hi])
            keep = ~keep if ascending else keep
            s[lo], s[hi] = jnp.where(keep, s[lo], s[hi]), jnp.where(keep, s[hi], s[lo])
            i[lo], i[hi] = jnp.where(keep, i[lo], i[hi]), jnp.where(keep, i[hi], i[lo])
        d //= 2
    while d >= 1:  # pairs within a vreg: sublanes p and p + d
        lower = (sublane & d) == 0
        # The lower candidate stays iff it ranks ahead (behind, ascending); the
        # pair's candidates are distinct, so the upper one stays iff it does not.
        flip = lower if ascending else ~lower
        for v in range(len(s)):
            ps = jnp.where(lower, pltpu.roll(s[v], 8 - d, 0), pltpu.roll(s[v], d, 0))
            pi = jnp.where(lower, pltpu.roll(i[v], 8 - d, 0), pltpu.roll(i[v], d, 0))
            keep = _ahead(s[v], i[v], ps, pi) ^ flip
            s[v], i[v] = jnp.where(keep, s[v], ps), jnp.where(keep, i[v], pi)
        d //= 2
    return s, i


def _sort_pairs(n):
    """Batcher's odd-even merge sort of `n` places: its compare-exchanges
    ``(i, j)``, ``i < j``, in order."""
    pairs = []
    p = 1
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


class _Network:
    """Compare-exchanges of whole vregs of candidates ``(scores, ids)``, each
    vreg element on its own; None is a candidate behind every other.

    A compare-exchange is ~7 vector-ALU ops and `_ahead`'s 2 mask-logic ops;
    the AOT schedule packs up to 4 of the first and 1 of the second per bundle,
    so every `_SELECT_FORM_PERIOD`-th compare takes `_ahead_by_selects` (8 ALU
    ops, no mask op) to share the load."""

    def __init__(self):
        self.compares = 0

    def ahead(self, a, b):
        self.compares += 1
        if self.compares % _SELECT_FORM_PERIOD == 0:
            return _ahead_by_selects(*a, *b)
        return _ahead(*a, *b)

    def exchange(self, a, b, k=None):
        """``(ahead, behind)`` of `a` and `b` (`k`: whether `a` is ahead)."""
        if a is None or b is None:
            return (b, None) if a is None else (a, None)
        k = self.ahead(a, b) if k is None else k
        return (
            (jnp.where(k, a[0], b[0]), jnp.where(k, a[1], b[1])),
            (jnp.where(k, b[0], a[0]), jnp.where(k, b[1], a[1])),
        )

    def sort(self, xs, in_column_order=False):
        """`xs` best first. `in_column_order`: their ids ascend, so the first
        stage's neighbours tie to the first."""
        xs = list(xs)
        for n, (i, j) in enumerate(_sort_pairs(len(xs))):
            first_stage = n < len(xs) // 2
            k = xs[i][0] >= xs[j][0] if in_column_order and first_stage else None
            xs[i], xs[j] = self.exchange(xs[i], xs[j], k)
        return xs

    def merge(self, a, b, k):
        """The first `k` of the ordered lists `a` and `b` (disjoint), in order:
        the better of each place of `a` and of reversed `b` (a bitonic list
        holding the first k), then a bitonic merge."""
        n = 1 << (min(k, len(a) + len(b)) - 1).bit_length()
        a = a[:n] + [None] * (n - len(a[:n]))
        b = b[:n] + [None] * (n - len(b[:n]))
        c = [self.exchange(a[p], b[n - 1 - p])[0] for p in range(n)]
        d = n // 2
        while d >= 1:
            for p in range(n):
                if p & d == 0:
                    c[p], c[p + d] = self.exchange(c[p], c[p + d])
            d //= 2
        return [x for x in c[:k] if x is not None]


def _lane_lists(st, ids, topk):
    """Each lane's first `topk` candidates of the transposed scores ``st
    [8 * V, 128]`` at column ids `ids`, in order and reversed: two lists of
    ``topk / 8`` vregs each for the scores and the ids (place ``p`` at sublane
    ``p % 8`` of vreg ``p // 8``).

    Each of a vreg's slots (sublane, lane) orders its V candidates (the same
    slot of every vreg) by a sorting network of whole-vreg compare-exchanges,
    in chunks of `_SORT_CHUNK` merged; then the 8 slots of a lane merge in
    three levels, each against its lists rolled 4 / 2 / 1 sublanes."""
    net = _Network()
    slots = [
        (st[8 * v : 8 * v + 8], ids[8 * v : 8 * v + 8]) for v in range(st.shape[0] // 8)
    ]
    lists = []
    for c in range(0, len(slots), _SORT_CHUNK):
        chunk = net.sort(slots[c : c + _SORT_CHUNK], in_column_order=True)[:topk]
        lists = net.merge(lists, chunk, topk) if lists else chunk
    for d in (4, 2, 1):
        rolled = [(pltpu.roll(s, d, 0), pltpu.roll(i, d, 0)) for s, i in lists]
        lists = net.merge(lists, rolled, topk)
    sublane = jax.lax.broadcasted_iota(jnp.int32, st[:8].shape, 0)

    def gather(order):  # place p from sublane p % 8 of its own vreg
        out = []
        for v in range(topk // 8):
            s, i = order[8 * v]
            for j in range(1, 8):
                s = jnp.where(sublane == j, order[8 * v + j][0], s)
                i = jnp.where(sublane == j, order[8 * v + j][1], i)
            out.append((s, i))
        return [x for x, _ in out], [y for _, y in out]

    return gather(lists), gather(lists[::-1])


def _mask_columns(s, ids, num_visible, rank, *, init_blocks, local_blocks, dcp):
    """`_pick`'s masking of the scores `s` at column ids `ids` (f32), by
    thresholds on the column: block ``c * dcp + rank`` is below ``n`` iff
    ``c`` is below ``ceil((n - rank) / dcp)``."""

    def first_column_at(n):
        return (-((rank - n) // dcp)).astype(jnp.float32)

    s = jnp.where(
        ids >= first_column_at(num_visible - local_blocks),
        msa_select._FORCED_LOCAL_SCORE,
        s,
    )
    if init_blocks:
        s = jnp.where(
            ids < first_column_at(jnp.int32(init_blocks)),
            msa_select._FORCED_INIT_SCORE,
            s,
        )
    return jnp.where(ids < first_column_at(num_visible), s, _NEG_INF)


def _pick_t_folded(
    s, num_visible, rank, *, folds, topk, init_blocks, local_blocks, dcp
):
    """`pack_candidates` of `_pick` of row-major scores ``s [rows, width]``
    (``num_visible [rows, 1]``, ``folds * rows <= 128``, ``width % (8 *
    folds) == 0``, ``width / folds >= topk``, `folds` and ``topk / 8`` powers
    of two), by sorting networks: compare-exchanges and selects of whole
    vregs, no dependent chain of reductions.

    The scores are transposed with column block ``f`` of ``width / folds``
    on lanes ``[f * rows, (f + 1) * rows)`` (the lanes transposed rows leave
    idle), so that each lane holds one block of a row on its sublanes and
    vregs. `_lane_lists` orders each lane's candidates; then each block's
    list is rolled onto block 0's lanes (all rolls at once, one latency) and
    the blocks merge in a tree: the better of each place of a list and of its
    pair's reversed list (the pair's top-k, bitonic), sorted. The real
    candidates are distinct (column ids), so the result is `_pick`'s."""
    rows, width = s.shape
    part = width // folds
    spare = _LANES - folds * rows
    st = jnp.concatenate(
        [s[:, f * part : (f + 1) * part] for f in range(folds)]
        + ([jnp.full((spare, part), _NEG_INF, s.dtype)] if spare else []),
        axis=0,
    ).T  # [part, 128]
    # Masked transposed: the rows' limits come over the lanes by a transpose
    # that overlaps the scores', not by a lane broadcast before it.
    num_visible = jnp.concatenate(
        [num_visible] * folds + ([jnp.zeros((spare, 1), jnp.int32)] if spare else []),
        axis=0,
    ).T  # [1, 128]
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, _LANES), 1)
    ids = (
        jax.lax.broadcasted_iota(jnp.int32, st.shape, 0) + lane // rows * part
    ).astype(jnp.float32)
    st = _mask_columns(
        st,
        ids,
        num_visible,
        rank,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        dcp=dcp,
    )
    best_first, reversed_ = _lane_lists(st, ids, topk)

    def on_block0(f):  # even blocks best first, odd ones reversed
        return [
            [pltpu.roll(x, _LANES - f * rows, 1) if f else x for x in xs]
            for xs in (best_first if f % 2 == 0 else reversed_)
        ]

    level = [on_block0(f) for f in range(folds)]
    while len(level) > 1:
        merged = []
        for j in range(0, len(level), 2):
            (a_s, a_i), (b_s, b_i) = level[j], level[j + 1]
            keep = [_ahead(*a, *b) for a, b in zip(zip(a_s, a_i), zip(b_s, b_i))]
            top_s = [jnp.where(k, a, b) for k, a, b in zip(keep, a_s, b_s)]
            top_i = [jnp.where(k, a, b) for k, a, b in zip(keep, a_i, b_i)]
            # The second of the next level's pairs reversed.
            merged.append(_bitonic_sort(top_s, top_i, ascending=j // 2 % 2 == 1))
        level = merged
    out_s, out_i = level[0]
    rank = jnp.asarray(rank).astype(jnp.float32)
    out_b = [
        jnp.where(x > _NEG_INF, i * dcp + rank, -1.0) for x, i in zip(out_s, out_i)
    ]
    # Both halves of the packed rows in one transpose.
    return jnp.concatenate(out_s + out_b, axis=0).T[:rows]


def _pick_folds(rows: int, width: int, topk: int) -> int:
    """Column blocks `_pick_t_folded` lays side by side (1: none)."""
    return next(
        (
            f
            for f in (8, 4, 2)
            if f * rows <= _LANES
            and width % (8 * f) == 0
            and width // f >= topk
            and (f < 8 or width >= _EIGHT_FOLD_WIDTH)
        ),
        1,
    )


class _PackedPick(NamedTuple):
    """The top-k of token rows at their positions, packed (`pack_candidates`):
    the `msa_indexer.RowSelect` of a decode step's local candidates."""

    block_size: int
    topk: int
    init_blocks: int
    local_blocks: int
    dcp: int

    def __call__(self, s, pos, rank):
        options = dict(
            topk=self.topk,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            dcp=self.dcp,
        )
        num_visible = pos // self.block_size + 1
        rows, width = s.shape
        vregs = self.topk // 8
        # The bitonic merges sort lists of whole vregs, a power of two of them.
        folds = 1
        if self.topk % 8 == 0 and vregs & (vregs - 1) == 0:
            folds = _pick_folds(rows, width, self.topk)
        if folds > 1 and width <= _FOLDED_PICK_WIDTH:
            return _pick_t_folded(s, num_visible, rank, folds=folds, **options)
        if width > _TRANSPOSED_PICK_WIDTH:
            return pack_candidates(*_pick(s, num_visible, rank, **options))
        scores, blocks = _pick_t(s.T, num_visible.T, rank, **options)
        return pack_candidates(scores.T, blocks.T)


# A decode pick scans this many columns at most transposed, unfolded
# (`_pick_t`, whose cost grows with the width: ~7,800 cycles at 1,024 against
# the row-major picks' ~7,600 at any width, AOT), and folded (the served
# 1,024-entry rows); its widths step by `_PICK_WIDTH_STEP`, one unrolled body
# each.
_TRANSPOSED_PICK_WIDTH = 768
_FOLDED_PICK_WIDTH = 1024
_LANES = 128
# Eight column blocks (if their rows fit the lanes) from this width: AOT at 8
# rows, 512 / 768 / 1,024 columns: 980 / 1,182 / 1,233 bundles at four
# blocks, 1,061 / 1,126 / 1,078 at eight.
_EIGHT_FOLD_WIDTH = 768
# AOT at 8 / 48 rows x 1,024 columns, every compare by selects / every 4th /
# none: 1,108 / 1,723, 1,073 / 1,664, 1,125 / 1,802 bundles.
_SELECT_FORM_PERIOD = 4
# Candidates a slot sorts at once before merging into its list (8 or 32: within
# +-2% at 8 to 48 rows, AOT).
_SORT_CHUNK = 16
_PICK_WIDTH_STEP = 256


def _candidates_kernel_t(
    rank_ref,
    widths_ref,
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
    widths,
    pretransposed=False,
):
    """`_candidates_kernel` with the row block transposed in VMEM: columns
    on sublanes, rows on lanes (``pos_ref [1, rows]``, outputs ``[topk,
    rows]``). Each pick's max and first index then reduce across vregs and
    sublanes on the VPU instead of across lanes on the XLU, which bound the
    row-major picks (~310 bundles per pick of a 256 x 128 block in the LLO).
    `pretransposed` scores already come as ``[columns, rows]``."""
    rank = rank_ref[0]
    width_now = widths_ref[pl.program_id(0)]
    rows = pos_ref.shape[1]
    num_visible = pos_ref[...] // block_size + 1  # [1, rows]

    def select(width):
        s = scores_ref[:width, :] if pretransposed else scores_ref[:, :width].T
        cols = jax.lax.broadcasted_iota(jnp.int32, (width, rows), 0)
        blocks = cols * dcp + rank
        visible = blocks < num_visible
        s = jnp.where(visible, s, _NEG_INF)
        s = jnp.where(
            visible & (blocks >= num_visible - local_blocks),
            msa_select._FORCED_LOCAL_SCORE,
            s,
        )
        s = jnp.where(
            visible & (blocks < init_blocks), msa_select._FORCED_INIT_SCORE, s
        )
        # Column ids as f32 (exact below 2**24): the first of a pick's ties is
        # one f32 min, where an int32 min takes a compare and a select.
        col_ids = cols.astype(jnp.float32)

        def pick(k, s):
            best_score = jnp.max(s, axis=0, keepdims=True)
            best = jnp.min(
                jnp.where(s == best_score, col_ids, float(width)),
                axis=0,
                keepdims=True,
            )
            score_out_ref[pl.ds(k, 1), :] = best_score
            block_out_ref[pl.ds(k, 1), :] = jnp.where(
                best_score > _NEG_INF, best.astype(jnp.int32) * dcp + rank, -1
            )
            return jnp.where(col_ids == best, _NEG_INF, s)

        if width <= _UNROLLED_WIDTH:
            for k in range(topk):
                s = pick(k, s)
        else:
            jax.lax.fori_loop(0, topk, pick, s, unroll=_PICK_UNROLL)

    for width in widths:
        pl.when(width_now == width)(functools.partial(select, width))


# Row copies `_row_copy_kernel` issues (and waits for) per loop step.
_COPY_UNROLL = 8


def _row_copy_kernel(
    rows_ref, rank_ref, widths_ref, scores_ref, pos_ref, *refs, kernel
):
    """`kernel` over this row block's token rows of the token-major scores
    (`msa_indexer.TokenRows`): one DMA per row, at ``rows_ref`` (SMEM), into
    the VMEM block that `kernel` scans."""
    score_out_ref, block_out_ref, block_ref, sem = refs
    block_rows = block_ref.shape[0]
    first = pl.program_id(0) * block_rows

    def copy(r, src):
        return pltpu.make_async_copy(
            scores_ref.at[pl.ds(src, 1)], block_ref.at[pl.ds(r, 1)], sem
        )

    def start(r, carry):
        copy(r, rows_ref[first + r]).start()
        return carry

    def wait(r, carry):
        copy(r, 0).wait()  # every copy is one row of the same size
        return carry

    jax.lax.fori_loop(0, block_rows, start, 0, unroll=_COPY_UNROLL)
    jax.lax.fori_loop(0, block_rows, wait, 0, unroll=_COPY_UNROLL)
    kernel(rank_ref, widths_ref, block_ref, pos_ref, score_out_ref, block_out_ref)


# Scan widths step by this many pretransposed columns up to
# `_FINE_WIDTH_LIMIT`, then by 128.
_FINE_WIDTH_STEP = 32
_FINE_WIDTH_LIMIT = 256
# Up to this width a block's picks unroll; past it they loop two at a time:
# unrolled bodies of every wider width would multiply the kernel's code, and
# the loop costs ~15% more per column there (AOT bundles: 384 columns 4,468
# looped, 1,024 columns 12,541 looped vs 10,876 unrolled).
_UNROLLED_WIDTH = 512
_PICK_UNROLL = 2
# Past this width the scan widths step by half of it (bounding the
# kernel's bodies: 20 at 4,096 columns).
_COARSE_WIDTH_FROM = 1024

# Row blocks of a multiple of this many rows (a vreg's lanes) pick
# transposed (`_candidates_kernel_t`, `_merge_kernel_t`); smaller decode
# blocks would fill few lanes of each vreg.
_TRANSPOSED_ROWS = 128


def _scan_widths(num_cols: int, pretransposed: bool) -> list[int]:
    """`local_candidates`' scan widths up to `num_cols`. A pick costs per
    scanned column, so widths step finely: by `_FINE_WIDTH_STEP` columns
    (sublanes of the pretransposed scores) up to `_FINE_WIDTH_LIMIT`, then by
    a vreg's 128 lanes (powers of two cost ~25% more scan on average: c64's
    146-column blocks scanned 256)."""
    step = _FINE_WIDTH_STEP if pretransposed else _MIN_CANDIDATE_WIDTH
    widths = list(range(step, min(num_cols, _FINE_WIDTH_LIMIT) + 1, step))
    # Then steps of 128 up to _COARSE_WIDTH_FROM, and of half of that
    # past it: one kernel body per width.
    for lo, hi, step in (
        (_MIN_CANDIDATE_WIDTH, _COARSE_WIDTH_FROM, _MIN_CANDIDATE_WIDTH),
        (_COARSE_WIDTH_FROM, num_cols, _COARSE_WIDTH_FROM // 2),
    ):
        first = (widths[-1] if widths else 0) + step
        widths += range(max(first, lo), min(hi, num_cols) + 1, step)
    if not widths or widths[-1] != num_cols:
        widths.append(num_cols)
    return widths


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "topk",
        "init_blocks",
        "local_blocks",
        "dcp",
        "block_rows",
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
    rows: jax.Array | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """This rank's top-k ``(scores [T, topk] f32, blocks [T, topk] i32)``.

    With `live_rows` (a traced count), only the row blocks holding rows
    ``[0, live_rows)`` are picked; the other rows' results are unset.

    ``scores [T, 1, P]`` are this rank's block scores (column ``c`` is global
    block ``c * dcp + rank``), or ``[P, T]`` transposed (the large tiles'
    layout in `msa_indexer`); forced blocks carry `msa_select`'s forced
    scores and missing picks are ``(-inf, -1)``. With `rows` (``[T]``),
    ``scores [R, P]`` are `msa_indexer.TokenRows` scores, token ``t``'s in row
    ``rows[t]``, which the kernel copies itself: XLA's row gather cost
    ~0.7 us per layer at decode (b1-b6 twins, j-05384dca).
    """
    by_row = rows is not None
    pretransposed = scores.ndim == 2 and not by_row
    if by_row:
        num_tokens, num_cols = rows.shape[0], scores.shape[1]
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
    block_rows = min(block_rows, max(8, -(-num_tokens // 8) * 8))
    if pretransposed:
        block_rows = -(-block_rows // _TRANSPOSED_ROWS) * _TRANSPOSED_ROWS
    pad = -num_tokens % block_rows
    s = scores if pretransposed or by_row else scores[:, 0]
    if by_row:
        rows = jnp.pad(rows, (0, pad))  # padding rows copy row 0
    elif pad and (live_rows is None or num_tokens < block_rows):
        s = jnp.pad(s, ((0, 0), (0, pad)) if pretransposed else ((0, pad), (0, 0)))
    # Without `live_rows` every row block runs, so a last block past the
    # scores always reads past them: in the c48 decode twin (384 tokens, DCP 2,
    # 4-bit) that halted every chip at warmup (j-433e936a). A class's grid
    # stops at its live rows, so its scores keep no padded copy (~17 us per
    # layer for the c48 mixed step's small tiles, j-c22c589e); its last block
    # reads past them only when the live rows reach it. Padding rows pick from
    # garbage and are dropped, as the kernels reduce within a row only.
    p = jnp.pad(pos, (0, pad))[:, None]
    # Static scan widths (`_scan_widths`, one kernel body each); each row
    # block takes the narrowest one holding every block its rows see (a
    # rank's blocks below n are among its first ceil(n / dcp) columns). A
    # decode batch scans the full width: its scan is a few us, and picking a
    # width would cost more.
    widths = (
        _scan_widths(num_cols, pretransposed)
        if num_tokens > _MIN_CANDIDATE_TOKENS
        else [num_cols]
    )
    num_blocks = (num_tokens + pad) // block_rows
    if len(widths) == 1:
        row_widths = jnp.full((num_blocks,), num_cols, jnp.int32)
    else:
        last_pos = jnp.max(p.reshape(num_blocks, block_rows), axis=1)
        needed = -(-(last_pos // block_size + 1) // dcp)
        w = jnp.asarray(widths, jnp.int32)
        row_widths = jnp.min(jnp.where(w >= needed[:, None], w, num_cols), axis=1)
    transposed = block_rows % _TRANSPOSED_ROWS == 0
    if transposed:
        kernel = functools.partial(_candidates_kernel_t, pretransposed=pretransposed)
        pos_spec = pl.BlockSpec((1, block_rows), lambda i: (0, i))
        p = p.T
        out_spec = pl.BlockSpec((topk, block_rows), lambda i: (0, i))
        out_dims = (topk, num_tokens + pad)
    else:
        kernel = _candidates_kernel
        pos_spec = pl.BlockSpec((block_rows, 1), lambda i: (i, 0))
        out_spec = pl.BlockSpec((block_rows, topk), lambda i: (i, 0))
        out_dims = (num_tokens + pad, topk)
    kernel = functools.partial(
        kernel,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        dcp=dcp,
        widths=tuple(widths),
    )
    if by_row:
        kernel = functools.partial(_row_copy_kernel, kernel=kernel)
        scores_spec = pl.BlockSpec(memory_space=pl.ANY)
    elif pretransposed:
        scores_spec = pl.BlockSpec((num_cols, block_rows), lambda i: (0, i))
    else:
        scores_spec = pl.BlockSpec((block_rows, num_cols), lambda i: (i, 0))
    picked_scores, picked_blocks = pl.pallas_call(
        kernel,
        grid=(
            num_blocks
            if live_rows is None
            else jnp.clip(-(-live_rows // block_rows), 1, num_blocks),
        ),
        in_specs=[
            *([pl.BlockSpec(memory_space=pltpu.SMEM)] if by_row else []),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            scores_spec,
            pos_spec,
        ],
        out_specs=[out_spec, out_spec],
        scratch_shapes=(
            [
                pltpu.VMEM((block_rows, num_cols), jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ]
            if by_row
            else []
        ),
        out_shape=[
            jax.ShapeDtypeStruct(out_dims, jnp.float32),
            jax.ShapeDtypeStruct(out_dims, jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",) if live_rows is None else ("arbitrary",),
            vmem_limit_bytes=msa_select._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_dcp_candidates",
    )(*([rows] if by_row else []), rank_array(rank), row_widths, s, p)
    if transposed:
        picked_scores, picked_blocks = picked_scores.T, picked_blocks.T
    return picked_scores[:num_tokens], picked_blocks[:num_tokens]


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
    # Each rank's slice side by side, not a transpose of the stack: the
    # concatenation fuses into the ranking, the transpose was relaid out.
    s = jnp.concatenate([c[:, :topk] for c in candidates], axis=1)
    b = jnp.concatenate([c[:, topk:] for c in candidates], axis=1).astype(jnp.int32)
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
# c64 mixed step (32768 rows; j-c1444339). Narrower merges (DCP 2, as c4/c16/c48
# serve) keep the dense ranking, cheap at n = 32.
_KERNEL_MERGE_MIN_CANDIDATES = 64
_MERGE_BLOCK_TOKENS = 256


def kept_own_candidates(
    candidates: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array]:
    """This rank's blocks ``[T, topk]`` of one KV head's packed candidates
    ``[ranks, T, 2 * topk]`` (this rank's first) and which of them the global
    top-k keeps: those that fewer than topk real candidates of all ranks
    rank ahead of (score descending, ties to the lower block), as
    `merge_candidates` orders them. The same pages as its selection's
    blocks of this rank, in candidate order: an attention that only tests
    a page's membership needs neither the other ranks' blocks nor the
    order."""
    s = jnp.concatenate([c[:, :topk] for c in candidates], axis=1)
    b = jnp.concatenate([c[:, topk:] for c in candidates], axis=1).astype(jnp.int32)
    s_i, b_i = s[:, :topk, None], b[:, :topk, None]
    s_j, b_j = s[:, None, :], b[:, None, :]
    ahead = (s_j > _NEG_INF) & ((s_j > s_i) | ((s_j == s_i) & (b_j < b_i)))
    place = jnp.sum(ahead, axis=2, dtype=jnp.int32)
    return b[:, :topk], (s[:, :topk] > _NEG_INF) & (place < topk)


# Tokens per tile of `pair_work`: `msa_attention.SMALL_BATCH_BLOCK_Q` decode rows,
# whose `_TILE_KEYS` page keys fill one vreg's lanes.
_TILE_ROWS = msa_attention.SMALL_BATCH_BLOCK_Q
_TILE_KEYS = 128
# Page keys `pair_work` ranks in f32, exactly.
_F32_EXACT_KEYS = 1 << 24
# Decode buckets up to this many tokens build their work list in `pair_work`,
# which unrolls its tiles; larger batches keep XLA's, which handles their
# tiles side by side.
_PAIR_WORK_MAX_TOKENS = 64
# Words of a `msa_attention.pack_table16` table per vreg (`_union_pages`).
_TABLE_VREG_WORDS = 8 * 128
# `_union_pages` gathers from every vreg of the packed table: the served 16 x
# 1,024-entry table is 8 of them.
_PAIR_WORK_MAX_TABLE_VREGS = 16
# lcm(1 .. _TILE_ROWS): a key's share of its distinct value when it occurs
# k <= _TILE_ROWS times in a tile (at most once per row) is this / k, exactly.
_SHARES = 840


def _kept_keys_t(own, peer, base):
    """`kept_own_candidates`' page keys, transposed: ``own``, ``peer [T, 2 *
    topk]`` packed candidates, ``base [T, 1]`` (-1 for a padding token) ->
    ``[topk, T]`` keys, -1 where not kept. Candidates down sublanes: a
    candidate's row broadcasts along sublanes, where a lane broadcast per
    candidate was a cross-lane permute each (64 per tile, the longest chain
    of the kernel in the AOT schedule)."""
    topk = own.shape[1] // 2
    own_t, peer_t = own.T, peer.T
    s_i, b_i = own_t[:topk], own_t[topk:]
    place = jnp.zeros(s_i.shape, jnp.int32)
    for cand in (own_t, peer_t):
        for j in range(topk):
            s_j = jnp.broadcast_to(cand[j : j + 1], s_i.shape)
            b_j = jnp.broadcast_to(cand[topk + j : topk + j + 1], s_i.shape)
            ahead = (s_j > _NEG_INF) & ((s_j > s_i) | ((s_j == s_i) & (b_j < b_i)))
            place += ahead.astype(jnp.int32)
    base_t = base.T
    kept = (s_i > _NEG_INF) & (place < topk) & (base_t >= 0)
    # A kept block is real (>= 0); the pair path has dcp == 2.
    return jnp.where(kept, base_t + (b_i.astype(jnp.int32) >> 1), -1)


def _tile_union(keys_t, tile):
    """Tile `tile`'s distinct page keys, ascending, -1 after (as
    `msa_attention._tile_unions` and `_union_slots` order them), ``[1, 128]``,
    and their count on every lane, from the kept keys ``keys_t [topk, T]``
    (`_kept_keys_t`; the tile's are columns ``[8 * tile, 8 * tile + 8)``).

    Each key's slot is the number of distinct keys below it, counted as the
    sum of every smaller key's share of its value (`_SHARES` over its
    multiplicity), so no compare against earlier lanes and no second
    pass is needed. The keys are exact f32 values (below `_F32_EXACT_KEYS`),
    every sum an integer below 2**24: the reductions run exactly in f32."""
    topk, num_tokens = keys_t.shape
    rows, lanes = _TILE_ROWS, _TILE_KEYS
    if num_tokens < lanes:
        keys_t = jnp.concatenate(
            [keys_t, jnp.full((topk, lanes - num_tokens), -1, jnp.int32)], axis=1
        )
    if tile:  # the tile's columns to lanes [0, 8)
        keys_t = pltpu.roll(keys_t, lanes - rows * tile, 1)
    column = jax.lax.broadcasted_iota(jnp.int32, keys_t.shape, 1)
    wide = jnp.where(column < rows, keys_t, -1)
    # Key (k, t) to lane k * 8 + t: down one lane vector, and across the lanes
    # of one sublane vector.
    spread = pltpu.roll(wide, 0, 1, stride=rows, stride_axis=0)
    by_lane = jnp.max(spread, axis=0, keepdims=True).astype(jnp.float32)
    across = jnp.broadcast_to(by_lane, (lanes, lanes))
    down = across.T
    count = jnp.sum(jnp.where(down == across, 1.0, 0.0), axis=0, keepdims=True)
    share = jnp.zeros_like(count)
    for k in range(1, rows + 1):
        share = jnp.where(count == k, float(_SHARES // k), share)
    share = jnp.where(by_lane >= 0, share, 0.0)  # [1, L]
    below = jnp.sum(
        jnp.where(across < down, jnp.broadcast_to(share, (lanes, lanes)), 0.0),
        axis=1,
        keepdims=True,
    )  # [L, 1]: the distinct keys below each key, times _SHARES
    slot = _SHARES * jax.lax.broadcasted_iota(jnp.int32, (lanes, lanes), 1)
    union = jnp.max(
        jnp.where((down >= 0) & (below == slot.astype(jnp.float32)), down, -1.0),
        axis=0,
        keepdims=True,
    )
    distinct = jnp.round(jnp.sum(share, axis=1, keepdims=True) * (1.0 / _SHARES))
    return union.astype(jnp.int32), jnp.broadcast_to(distinct, (1, lanes)).astype(
        jnp.int32
    )


def _union_pages(union, table_ref):
    """The physical pages ``[1, 128]`` of a tile's union of page keys (-1 for
    none) from the `msa_attention.pack_table16` table in VMEM, ``table_ref
    [V, 8, 128]`` words (key ``k`` is the low or high half of word ``k >>
    1``): one lane gather per table vreg, all on the same lanes, then the
    key's vreg and sublane picked by selects. Vector work only: an SMEM input
    would cost the kernel a copy of ~0.5 us before it starts (j-718491f0)."""
    word = jnp.maximum(union, 0) >> 1
    lane = jnp.broadcast_to(word & (_LANES - 1), (8, _LANES))
    vreg = jnp.broadcast_to(word >> 10, (8, _LANES))
    picked = jnp.zeros((8, _LANES), jnp.int32)
    for v in range(table_ref.shape[0]):
        entries = jnp.take_along_axis(
            table_ref[v], lane, axis=1, mode="promise_in_bounds"
        )  # [8, 128]: sublane s holds word v * 1024 + s * 128 + lane
        picked = jnp.where(vreg == v, entries, picked)
    sublane = jax.lax.broadcasted_iota(jnp.int32, (8, _LANES), 0)
    picked = jnp.where(sublane == ((word >> 7) & 7), picked, 0)
    words = jnp.sum(picked, axis=0, keepdims=True)  # one nonzero term per lane
    pages = (words >> ((union & 1) << 4)) & 0xFFFF
    return jnp.where(union >= 0, pages, -1)


def _pair_work_kernel(own_ref, peer_ref, base_ref, *refs, single, groups, pages, heads):
    """`pair_work` in VMEM: this rank's kept candidates' page keys
    (`_kept_keys_t`), then each tile's page union (`_tile_union`) and, with
    `pages`, its physical pages (`_union_pages`). With `heads`, the pair's
    query heads go to their rank-ordered places in HBM by two copies that run
    under the ranking."""
    table_ref = None
    if pages:
        table_ref, *refs = refs
    if heads:
        rank_ref, q_ref, q_peer_ref, keys_ref, work_ref, q_all_ref, sems = refs
        num_heads = q_ref.shape[0]
        rank = rank_ref[0]
        copies = [
            pltpu.make_async_copy(
                src, q_all_ref.at[pl.ds(block * num_heads, num_heads)], sems.at[i]
            )
            for i, (src, block) in enumerate(((q_ref, rank), (q_peer_ref, 1 - rank)))
        ]
        for copy in copies:
            copy.start()
    else:
        keys_ref, work_ref = refs
    num_tokens = keys_ref.shape[0]
    tiles = num_tokens // _TILE_ROWS
    union_len = _TILE_KEYS
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, union_len), 1)
    keys_t = _kept_keys_t(own_ref[...], peer_ref[...], base_ref[...])
    keys_ref[...] = keys_t.T
    # With a table, the unions' physical pages follow `_packed_work`'s row.
    pages_at = tiles * union_len + (2 * groups + 2 if single else tiles)
    # Unrolled: the tiles' chains are independent and short of work each.
    counts = jnp.zeros_like(lane)
    for t in range(tiles):
        union, count = _tile_union(keys_t, t)
        work_ref[:, t * union_len : (t + 1) * union_len] = union
        if table_ref is not None:
            at = pages_at + t * union_len
            work_ref[:, at : at + union_len] = _union_pages(union, table_ref)
        counts = jnp.where(lane == t, count, counts)
    if not single:  # `_tile_page_lists`' tile counts after the unions
        work_ref[:, tiles * union_len : pages_at] = counts[:, :tiles]
    else:
        _single_tile_words(work_ref, lane, count, groups)
    if heads:
        for copy in copies:
            copy.wait()


def _single_tile_words(work_ref, lane, count, groups):
    """One tile's `msa_attention._single_tile_groups` meta words and counts,
    and the grid's group bounds (`msa_attention_lse._packed_work`), after its
    union in `work_ref`: lanes [0, G) meta words, [G, 2G) counts, 2G and
    2G + 1 the bounds."""
    union_len = _TILE_KEYS
    pages_per_step = union_len // groups
    g = lane % groups
    group_count = jnp.clip(count - g * pages_per_step, 0, pages_per_step)
    # Groups holding a page, or the first: pages_per_step is a power of two.
    num_live = jnp.maximum(
        (count + pages_per_step - 1) >> (pages_per_step.bit_length() - 1), 1
    )
    meta = (
        (g << msa_attention._ROW_SHIFT)
        | jnp.where(g == 0, msa_attention._FIRST_BIT, 0)
        | jnp.where(g == num_live - 1, msa_attention._LAST_BIT, 0)
    )
    row = jnp.where(lane < groups, meta, group_count)
    row = jnp.where(lane < 2 * groups, row, jnp.where(lane == 2 * groups, 0, num_live))
    work_ref[:, union_len : union_len + 2 * groups + 2] = row[:, : 2 * groups + 2]


def _takes_pair_work(own_first, num_heads, dcp, num_tokens, topk, num_keys):
    """Whether `msa_attend_dcp` builds its work list in `pair_work`: a pair's
    decode step attending its kept own candidates, within the kernel's
    envelope (a tile's `_TILE_ROWS` rows of topk keys fill `_TILE_KEYS`
    lanes, at most `_PAIR_WORK_MAX_TOKENS` tokens, keys exact in f32)."""
    return (
        own_first
        and num_heads == 1
        and dcp == 2
        and 0 < num_tokens <= _PAIR_WORK_MAX_TOKENS
        and topk * _TILE_ROWS == _TILE_KEYS
        and num_keys <= _F32_EXACT_KEYS
    )


def _pair_work_pages(page_table: jax.Array | None) -> bool:
    """Whether `pair_work` also looks up its unions' physical pages, given
    the `msa_attention.pack_table16` table (None if its ids need more than
    16 bits): the attention then copies no page table into SMEM (~17 ns per
    KiB, ~0.55 us per call for the served 32 KiB, j-718491f0), and the
    lookup gathers from every vreg of the table (`_union_pages`)."""
    return (
        page_table is not None
        and -(-page_table.shape[0] // _TABLE_VREG_WORDS) <= _PAIR_WORK_MAX_TABLE_VREGS
    )


def pair_work(
    own: jax.Array,
    peer: jax.Array,
    key_base: jax.Array,
    *,
    pages_per_step: int,
    page_table: jax.Array | None = None,
    heads: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """A DCP pair's decode attention inputs from both ranks' packed candidates
    (``own`` this rank's, ``peer`` the other's, ``[T, 2 * topk]``) in one
    kernel: the page keys of this rank's kept candidates ``[T, topk]``
    (`kept_own_candidates`; ``key_base [T, 1]`` is a token's ``seq_id *
    pages_per_seq``, or -1 for a padding token) and the attention's work list
    row (`msa_attention_lse._packed_work` of those keys), followed with
    `page_table` (`msa_attention.pack_table16`'s words, `_pair_work_pages`)
    by the unions' physical pages so that the attention reads no page table.
    XLA built both in
    ~20 dependent small ops between the candidate exchange and the attention
    (~2 us per layer in the served b8 decode step, j-f07a4a0e).

    With ``heads = (rank, q, q_peer)`` (this rank's and the other rank's
    head-major query heads ``[Hl, T, D]``) it also returns the pair's heads
    in rank order ``[2 * Hl, T, D]``, which XLA otherwise concatenates in an
    op of its own before the attention."""
    num_tokens, width = own.shape
    topk = width // 2
    if topk * _TILE_ROWS != _TILE_KEYS:
        raise ValueError(
            f"pair_work tiles {_TILE_ROWS} rows of {_TILE_KEYS // _TILE_ROWS} keys"
        )
    pad = -num_tokens % _TILE_ROWS
    if num_tokens + pad > _TILE_KEYS:
        raise ValueError(f"pair_work takes up to {_TILE_KEYS} tokens, got {num_tokens}")
    if pad:
        own = jnp.pad(own, ((0, pad), (0, 0)), constant_values=_NEG_INF)
        peer = jnp.pad(peer, ((0, pad), (0, 0)), constant_values=_NEG_INF)
        key_base = jnp.pad(key_base, ((0, pad), (0, 0)), constant_values=-1)
    tiles = (num_tokens + pad) // _TILE_ROWS
    groups = msa_attention._groups_per_tile(_TILE_ROWS, topk, pages_per_step)
    if groups * pages_per_step != _TILE_KEYS:
        raise ValueError(
            f"{pages_per_step} pages per step do not tile {_TILE_KEYS} keys"
        )
    single = tiles == 1
    # `_packed_work`'s row: [union | meta words | counts | step bounds] for one
    # tile, [unions | tile counts] for several (`in_kernel`); then the pages.
    pages = page_table is not None
    width = (2 * groups + 2 if single else tiles) + (1 + pages) * tiles * _TILE_KEYS
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    in_specs = [vmem] * 3
    out_specs = [vmem] * 2
    out_shape = [
        jax.ShapeDtypeStruct((num_tokens + pad, topk), jnp.int32),
        jax.ShapeDtypeStruct((1, width), jnp.int32),
    ]
    scratch_shapes = []
    operands = [own, peer, key_base]
    if pages:
        if not _pair_work_pages(page_table):
            raise ValueError(
                f"pair_work looks pages up in at most {_PAIR_WORK_MAX_TABLE_VREGS} "
                f"vregs of a packed table, got {page_table.shape}"
            )
        vregs = -(-page_table.shape[0] // _TABLE_VREG_WORDS)
        table = jnp.pad(
            page_table, (0, vregs * _TABLE_VREG_WORDS - page_table.shape[0])
        )
        in_specs.append(vmem)
        operands.append(table.reshape(vregs, 8, _LANES))
    if heads is not None:
        rank, q, q_peer = heads
        hbm = pl.BlockSpec(memory_space=pl.ANY)
        in_specs += [pl.BlockSpec(memory_space=pltpu.SMEM), hbm, hbm]
        out_specs.append(hbm)
        out_shape.append(jax.ShapeDtypeStruct((2 * q.shape[0], *q.shape[1:]), q.dtype))
        scratch_shapes.append(pltpu.SemaphoreType.DMA((2,)))
        operands += [rank_array(rank), q, q_peer]
    keys, work, *q_all = pl.pallas_call(
        functools.partial(
            _pair_work_kernel,
            single=single,
            groups=groups,
            pages=pages,
            heads=heads is not None,
        ),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        interpret=interpret,
        name="minimax_m3_msa_dcp_pair_work",
    )(*operands)
    return keys[:num_tokens], work, *q_all


def merge_head_candidates(
    candidates: jax.Array, topk: int, interpret: bool = False
) -> jax.Array:
    """`merge_candidates` per KV head: ``[dcp, T, G * 2 * topk]`` (each
    rank's `pack_candidates` of its ``G`` heads, head-major, the ranks in any
    order) -> selection ``[T, G, topk]``. The real candidates are distinct
    blocks, so both merges order them by (score, block) alone."""
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


def _class_candidates(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    local_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    seq_id: jax.Array,
    candidates,
    *,
    interpret: bool,
    scale_cache: jax.Array | None,
    scale_row: int | None,
) -> jax.Array:
    """Packed candidates ``[T, G * 2 * topk]`` (`pack_candidates` per head,
    head-major) of a prefill-sized step: each `msa_indexer` tile class picks
    in the layout it scored in, and only the picks go back to token order.
    Gathering the classes' ``[P, columns]`` scores to ``[T, P]`` first (a
    concatenate, column gather and transpose, ~110 us per layer at c48)
    moved 32x the bytes, which the transposing pick kernel then turned back.

    ``index_q [T, G, D]``: ``G > 1`` heads are folded into each request's
    query rows (`msa.fold_index_heads`), as `msa.partial_block_scores` does.
    """
    num_tokens, num_heads, _ = index_q.shape
    if num_heads == 1:
        folded = index_q
        rows = jnp.arange(num_tokens, dtype=jnp.int32)[None]
    else:
        folded, rows = msa.fold_index_heads(index_q, local_lens, cu_q_lens)
        folded = folded[:, None]
    folded_cu = num_heads * cu_q_lens
    plan = msa_indexer.class_plan(
        folded,
        index_cache,
        local_lens,
        block_tables,
        folded_cu,
        num_seqs,
        **msa.block_score_options(num_heads),
        scale_cache=scale_cache,
        scale_row=scale_row,
        interpret=interpret,
    )
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    max_num_seqs = q_lens.shape[0]
    seq_of_token = jnp.minimum(seq_id, max_num_seqs - 1)
    picks, cols = [], []
    for scored, block_q, transposed in (
        (plan.small(), plan.small_block_q, not plan.token_major),
        (plan.large(), plan.large_block_q, True),
    ):
        # A column's query row sits at offset o of its request's folded rows,
        # token o mod q_len of it: position = len - q_len + o mod q_len.
        per_tile = functools.partial(msa_indexer.lookup, index=scored.tile_seq)
        row = scored.tile_row0[:, None] + jnp.arange(block_q, dtype=jnp.int32)[None, :]
        offset = row - per_tile(folded_cu[:-1])[:, None]
        q_len = jnp.maximum(per_tile(q_lens), 1)[:, None]
        col_pos = (per_tile(seq_lens)[:, None] - q_len + offset % q_len).reshape(-1)
        # A class's scores are padded to its worst-case tile count (the large
        # class: 66 tiles of 512 at 8K tokens for ~14 real ones); pick in
        # its live tiles only.
        s, b = candidates(
            scored.scores if transposed else scored.scores[:, None, :],
            col_pos,
            live_rows=scored.num_tiles * block_q,
        )
        picks.append(pack_candidates(s, b))
        # Each head's and token's column in this class (`_class_scores`).
        start = msa_indexer.lookup(scored.tile_start, seq_of_token)[None]
        off = rows - start
        col = (
            msa_indexer.lookup(scored.first_tile, seq_of_token)[None] + off // block_q
        ) * block_q
        cols.append(jnp.clip(col + off % block_q, 0, picks[-1].shape[0] - 1))
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


def _written_pages(writes, seq_id, pos, max_num_seqs, block_size, dcp):
    """Per request, the page its run writes, as the indexer reads its pages
    (index ``j`` of its block-table row holds this rank's block ``j * dcp +
    rank``, `owner_pages`), or -1; then the run's index (0 if none):
    ``[2 * max_num_seqs]``. A decode request (at most a block of tokens)
    writes at most one page per rank."""
    num_runs, _, _, _, tok0 = writes.scalars
    max_runs, num_tokens = tok0.shape[0], seq_id.shape[0]
    tok0 = jnp.minimum(tok0, num_tokens - 1)
    run = jnp.arange(max_runs, dtype=jnp.int32)
    seq = jnp.where(run < num_runs[0], msa_indexer.lookup(seq_id, tok0), -1)
    page = msa_indexer.lookup(pos, tok0) // block_size // dcp
    mine = seq[None, :] == jnp.arange(max_num_seqs, dtype=jnp.int32)[:, None]
    return jnp.concatenate(
        [
            jnp.max(jnp.where(mine, page[None, :], -1), axis=1),
            jnp.max(jnp.where(mine, run[None, :], 0), axis=1),
        ]
    )


def msa_write_and_candidates_dcp(
    k: jax.Array,
    v: jax.Array,
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
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """First half of the DCP forward: write this rank's tokens, score its
    blocks and pick its candidates.

    With 4-bit caches (`msa_fp4`, one KV head per rank) pass `scale_cache`;
    it is returned updated after `index_cache`.

    Shapes as `msa.msa_forward`, except that a rank may hold ``G >= 1`` KV
    heads (DCP wider than a KV head's replicas): ``k``/``v``/``index_q
    [T, G, D]``, cache rows ``[K_0 | V_0 | ... | K_G-1 | V_G-1]``, and the
    single index-key head shared by all. ``block_tables`` are vLLM's logical
    DCP blocks of ``dcp * block_size`` tokens. Returns ``(kv_cache,
    index_cache, candidates [T, G * 2 * topk] f32)``: `pack_candidates` per
    head, head-major.
    """
    num_tokens, num_heads, _ = k.shape
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
    local_lens = local_seq_lens(seq_lens, block_size=block_size, rank=rank, dcp=dcp)
    decode = not msa_indexer.is_prefill_step(num_heads * num_tokens, max_num_seqs)
    if num_heads == 1 and not fp4 and decode:
        plan = msa_indexer.class_plan(
            index_q,
            index_cache,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            **msa.block_score_options(),
            interpret=interpret,
        )
        # A request's tokens (at most a tile class's block_q when it
        # selects) then span at most one block per rank: one written page each
        # (`_written_pages`).
        if plan.selects and block_size >= msa_indexer.DEFAULT_BLOCK_Q:
            # A decode step: one kernel writes, scores and picks.
            kv_rows = jnp.concatenate([k, v], axis=-1).reshape(num_tokens, -1)
            writes = msa.run_writes(
                [
                    msa_write.Rows(kv_cache, kv_rows),
                    msa_write.Rows(index_cache, index_k),
                ],
                pages,
                pos,
                max_num_seqs=max_num_seqs,
            )
            pick = _PackedPick(block_size, topk, init_blocks, local_blocks, dcp)
            selected = plan.small(
                select=msa_indexer.RowSelect(pick, 2 * topk, _PICK_WIDTH_STEP),
                select_inputs=(pos, rank),
                writes=(
                    writes,
                    _written_pages(writes, seq_id, pos, max_num_seqs, block_size, dcp),
                ),
            )
            return (*selected.caches, selected.rows)
    if fp4:
        kv_cache, index_cache, scale_cache = msa.write_fp4(
            kv_cache,
            index_cache,
            scale_cache,
            k,
            v,
            index_k,
            pages,
            pos,
            max_num_seqs=max_num_seqs,
            interpret=interpret,
        )
        index_q = msa_fp4.rotate_queries(index_q)
    else:
        kv_rows = jnp.concatenate([k, v], axis=-1).reshape(num_tokens, -1)
        kv_cache, index_cache = msa.write_caches(
            [msa_write.Rows(kv_cache, kv_rows), msa_write.Rows(index_cache, index_k)],
            pages,
            pos,
            max_num_seqs=max_num_seqs,
            interpret=interpret,
        )
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
    if msa_indexer.is_prefill_step(num_heads * num_tokens, max_num_seqs):
        packed = _class_candidates(
            index_q,
            index_cache,
            seq_lens,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            seq_id,
            candidates,
            interpret=interpret,
            scale_cache=scale_cache,
            scale_row=scale_row,
        )
        if fp4:
            return kv_cache, index_cache, scale_cache, packed
        return kv_cache, index_cache, packed
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
            token_rows=True,
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
    if isinstance(scores, msa_indexer.TokenRows):
        cand_scores, cand_blocks = candidates(scores.scores, pos, rows=scores.rows)
    else:
        cand_scores, cand_blocks = candidates(scores, jnp.tile(pos, num_heads))
    packed = pack_candidates(cand_scores, cand_blocks)  # [G * T, 2 * topk]
    packed = packed.reshape(num_heads, num_tokens, -1).transpose(1, 0, 2)
    if fp4:
        return kv_cache, index_cache, scale_cache, packed.reshape(num_tokens, -1)
    return kv_cache, index_cache, packed.reshape(num_tokens, -1)


def msa_attend_dcp(
    q: jax.Array | tuple[jax.Array, jax.Array],
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
    own_first: bool = False,
    pair_packs: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
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
    leading axis (`ops.msa.merge_head_partials`), or the merge's all-to-all
    sends each rank its block (`attend_and_merge`). With `own_first`
    (``candidates[0]`` is this rank's) and one KV head, the page keys come
    from this rank's kept candidates (`kept_own_candidates`). With
    `pair_packs` (a pair, one KV head) the results are this rank's and the
    other rank's packs ``(own, peer) [Hl + 1, T, D]`` uint16 for `merge_pair`
    (`msa_attention_lse.msa_sparse_attention_lse`). A pair (``dcp == 2``,
    `q_head_major`) may pass ``q = (own, peer)``, this rank's heads and the
    other rank's, ``[Hl, T, D]`` each: they are placed in rank order by the
    decode work-list kernel (`pair_work`) or else concatenated.
    """
    halves = isinstance(q, tuple)
    if halves and not (q_head_major and dcp == 2):
        raise ValueError("query halves need a head-major pair")
    num_tokens = (q[0] if halves else q).shape[1 if q_head_major else 0]
    rank = rank_array(rank)[0]
    if pair_packs and not (dcp == 2 and q_head_major):
        raise ValueError("pair packs need a head-major pair")
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    pages_per_seq = block_tables_2d.shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    num_heads = candidates.shape[2] // (2 * topk)
    own_block = pos // block_size
    local_key = jnp.where(
        valid & (own_block % dcp == rank),
        seq_id * pages_per_seq + own_block // dcp,
        -1,
    )
    # Page keys (seq * pages_per_seq + logical block) of the selected blocks
    # this rank holds, as `msa.sparse_attention`; [T, G, topk].
    work = None
    page_table = msa_attention.pack_table16(block_tables.reshape(-1), kv_cache.shape[0])
    work_pages = _pair_work_pages(page_table)
    if _takes_pair_work(
        own_first, num_heads, dcp, num_tokens, topk, max_num_seqs * pages_per_seq
    ):
        # One kernel: kept page keys and the attention's work list (with its
        # physical pages, and the pair's query heads in rank order).
        key_base = jnp.where(valid, seq_id * pages_per_seq, -1)[:, None]
        keys, work, *placed = pair_work(
            candidates[0],
            candidates[1],
            key_base,
            pages_per_step=_small_pages_per_step(num_tokens),
            page_table=page_table if work_pages else None,
            heads=(rank, *q) if halves else None,
            interpret=interpret,
        )
        keys = keys[:, None]
        if halves:
            (q,) = placed
    else:
        if halves:
            q = _pair_heads(*q, rank)
        if own_first and num_heads == 1:
            blocks, kept = kept_own_candidates(candidates, topk)
            mine, selected = (kept & valid[:, None])[:, None], blocks[:, None]
        else:
            selected = merge_head_candidates(candidates, topk, interpret)
            mine = (selected >= 0) & (selected % dcp == rank) & valid[:, None, None]
    if work is None:
        keys = jnp.where(
            mine, seq_id[:, None, None] * pages_per_seq + selected // dcp, -1
        )
    small = num_tokens <= DCP_SMALL_BATCH_TOKENS
    if small:
        block_q = msa_attention.SMALL_BATCH_BLOCK_Q
        pages_per_step = _small_pages_per_step(num_tokens)
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
    work_arg = {} if work is None else {"work": work, "work_pages": work_pages}
    if pair_packs and num_heads != 1:
        raise ValueError(f"pair packs need one KV head, got {num_heads}")
    pair_arg = {"pair_rank": rank_array(rank)} if pair_packs else {}
    args = (local_key, pos % block_size + 1, block_tables.reshape(-1))
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
                **work_arg,
                **pair_arg,
                **msa._fp4_attention_kwargs(scale_cache),
            )
        # KV head h's K / V scales are scale-page rows 2h / 2h + 1.
        return msa_attention_lse.msa_sparse_attention_lse_heads(
            q, kv_cache, keys, *args, head_major=True, **common, scale_cache=scale_cache
        )
    if num_heads == 1:
        return msa_attention_lse.msa_sparse_attention_lse(
            q,
            kv_cache,
            keys[:, 0],
            *args,
            head=0,
            head_major=True,
            **common,
            **work_arg,
            **pair_arg,
        )
    # One kernel for all the group's KV heads (the head is a kernel input).
    return msa_attention_lse.msa_sparse_attention_lse_heads(
        q, kv_cache, keys, *args, head_major=True, **common
    )


_PARTIALS_BLOCK_TOKENS = 256


def _merge_partials_kernel(lse_ref, parts_ref, *rest, rotated):
    """One token block of `merge_partials`: ``lse_ref [dcp, Bt, Hl]``,
    ``parts_ref [dcp, Hl, Bt, D]``, (``hadamard [D, D]``), ``o_ref [Bt,
    Hl * D]``."""
    dcp = parts_ref.shape[0]
    _merge_rows(
        lse_ref[...],
        lambda r, h: parts_ref[r, h],
        dcp,
        rest[-1],
        rest[0] if rotated else None,
    )


def _merge_rows(lse, part, dcp, o_ref, hadamard_ref):
    """The LSE merge of ``dcp`` partials: ``lse [dcp, Bt, Hl]`` f32, ``part(r,
    h)`` rank ``r``'s ``[Bt, D]`` partial of head ``h`` -> ``o_ref [Bt, Hl *
    D]`` (unrotated by `hadamard_ref` if given)."""
    num_heads = lse.shape[2]
    head_dim = o_ref.shape[1] // num_heads
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
            p = part(r, h)
            w = jnp.broadcast_to(weight[r, :, h : h + 1], p.shape)
            # A plain product: a rank that attended nothing for a token has
            # a zero partial (`msa_sparse_attention_lse`), never NaN, so no
            # select guards it (a compare and a select per element, ~20% of
            # the kernel's bundles).
            term = p.astype(jnp.float32) * w
            merged = term if merged is None else merged + term
        merged = merged.astype(o_ref.dtype)
        if hadamard_ref is not None:
            merged = jnp.dot(
                merged, hadamard_ref[...], preferred_element_type=jnp.float32
            )
            merged = (merged * (1.0 / head_dim)).astype(o_ref.dtype)
        o_ref[:, h * head_dim : (h + 1) * head_dim] = merged


def _merge_pair_kernel(rank_ref, own_ref, peer_ref, *rest, rotated):
    """One token block of `merge_pair`: this rank's and the other rank's packs
    of this rank's heads, ``own_ref`` and ``peer_ref [Hl + 1, Bt, D]``
    uint16, (``hadamard [D, D]``), ``o_ref [Bt, Hl * D]``. The terms add in
    rank order, as `merge_partials` adds them."""
    o_ref = rest[-1]
    num_heads = own_ref.shape[0] - 1
    first = rank_ref[0] == 0  # this rank's terms come first

    def lses(ref):  # [Bt, Hl] f32 from the pack's LSE halves row
        bits = ref[num_heads].astype(jnp.int32)  # [Bt, D]
        return pltpu.bitcast(
            (bits[:, :num_heads] << 16) | bits[:, num_heads : 2 * num_heads],
            jnp.float32,
        )

    own_lse, peer_lse = lses(own_ref), lses(peer_ref)
    lse = jnp.stack(
        [jnp.where(first, own_lse, peer_lse), jnp.where(first, peer_lse, own_lse)]
    )

    def part(r, h):
        own = pltpu.bitcast(own_ref[h], o_ref.dtype)
        peer = pltpu.bitcast(peer_ref[h], o_ref.dtype)
        return jnp.where(first == (r == 0), own, peer)

    _merge_rows(lse, part, 2, o_ref, rest[0] if rotated else None)


@functools.partial(jax.jit, static_argnames=("dtype", "block_tokens", "interpret"))
def merge_pair(
    own: jax.Array,
    peer: jax.Array,
    rank: jax.Array,
    hadamard: jax.Array | None = None,
    *,
    dtype: jnp.dtype,
    block_tokens: int = _PARTIALS_BLOCK_TOKENS,
    interpret: bool = False,
) -> jax.Array:
    """`merge_partials` of a DCP pair, token-major ``[T, Hl * D]`` `dtype`:
    this rank's pack of its heads' partial and the other rank's
    (`msa_attend_dcp`'s `pair_packs`), ``own`` and ``peer [Hl + 1, T, D]``
    uint16 each: a head block's `dtype` partial bits, then a row of each
    token's ``Hl`` LSEs (f32 bits), high halves then low halves. As 16-bit
    integers, the permute moves them as bits: float copies may canonicalize
    NaN or flush subnormal patterns (the LSE halves are arbitrary bits)."""
    heads_and_lse, num_tokens, head_dim = own.shape
    num_heads = heads_and_lse - 1
    if own.dtype != jnp.uint16 or 2 * num_heads > head_dim:
        raise ValueError(
            f"uint16 packs with 2 * heads <= head_dim, got {own.dtype} {own.shape}"
        )
    block_t = max(1, min(num_tokens, block_tokens))
    rotated = hadamard is not None
    pack_spec = pl.BlockSpec((heads_and_lse, block_t, head_dim), lambda i, r: (0, i, 0))
    return pl.pallas_call(
        functools.partial(_merge_pair_kernel, rotated=rotated),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(pl.cdiv(num_tokens, block_t),),
            in_specs=[
                pack_spec,
                pack_spec,
                *(
                    [pl.BlockSpec((head_dim, head_dim), lambda i, r: (0, 0))]
                    if rotated
                    else []
                ),
            ],
            out_specs=pl.BlockSpec(
                (block_t, num_heads * head_dim), lambda i, r: (i, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens, num_heads * head_dim), dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge_pair",
    )(rank_array(rank), own, peer, *([hadamard] if rotated else []))


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


def _pair_heads(own: jax.Array, peer: jax.Array, rank: jax.Array) -> jax.Array:
    """A pair's head-major query heads in rank order ``[2 * Hl, T, D]`` from
    this rank's and the other rank's ``[Hl, T, D]``."""
    # [own, peer, own] from row `rank * Hl` is the pair's heads in rank order.
    both = jnp.concatenate([own, peer, own])
    return jax.lax.dynamic_slice_in_dim(both, rank * own.shape[0], 2 * own.shape[0])


def _gather_heads(q: jax.Array, rank: jax.Array, axis: str, dcp: int):
    """The group's head-major query heads ``[dcp * Hl, T, D]`` from this
    rank's ``q [Hl, T, D]``; a pair's as ``(q, the other rank's q)`` for
    `msa_attend_dcp` to place. A pair swaps its halves by one collective
    permute: XLA's all-gather of the 16 MB c4 / c16 queries took 158 us per
    layer in the twin (j-094494a7), a permute plus the concatenation ~51 us
    on the same core pair in a standalone bench (j-a7bf7869; all-gather ~90)."""
    if dcp != 2:
        return jax.lax.all_gather(q, axis, axis=0, tiled=True)
    return q, jax.lax.ppermute(q, axis, [(0, 1), (1, 0)])


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

    The query heads and candidates are all-gathered, `msa_attend_dcp` runs
    over the group's heads, and each rank's head block of the partial and
    its LSEs go to their owner by all-to-all for `merge_partials`; a
    decode-sized pair's by one permute of the other rank's block for
    `merge_pair`. As XLA collectives in one program they move the head-major
    buffers as laid out; torch's all-to-all of them cost ~890 us per layer at
    8K tokens (c4 twin, j-c60806bf)."""
    num_heads, num_tokens, head_dim = q.shape
    rank = rank_array(rank)[0]
    q_group = _gather_heads(q, rank, axis, dcp)
    # Decode-sized operands of a pair move by permutes (latency); a prefill
    # step's partials by XLA's all-to-all, which writes them in place.
    small = dcp == 2 and num_tokens <= DCP_SMALL_BATCH_TOKENS
    if small:
        # A pair's candidates by one permute: an XLA all-gather blocks the
        # TensorCore for its latency (3.2 us per layer at c4 b3, j-c71e5160).
        # No rank-ordered copy (zeros, rank-indexed updates, relayouts): the
        # selection does not depend on the ranks' order, and this rank's own
        # candidates come first (`msa_attend_dcp`'s `own_first`).
        assert dcp == 2, "the swap below pairs two ranks"
        received = jax.lax.ppermute(candidates, axis, [(0, 1), (1, 0)])
        cand_all = jnp.stack([candidates, received])
    else:
        cand_all = jax.lax.all_gather(candidates, axis, axis=0)
    partial = msa_attend_dcp(
        q_group,
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
        own_first=small,
        pair_packs=small,
        interpret=interpret,
    )
    if small:
        # The attention wrote the other rank's pack as the permute sends it;
        # ours stays where it was written.
        own, peer = partial
        peer = jax.lax.ppermute(peer, axis, [(0, 1), (1, 0)])
        return merge_pair(own, peer, rank, hadamard, dtype=q.dtype, interpret=interpret)
    # Block r of each result came from rank r: its partial of this rank's heads.
    out, lse = partial
    parts = out.reshape(dcp, num_heads, num_tokens, head_dim)
    lses = lse.reshape(dcp, num_heads, num_tokens).transpose(0, 2, 1)
    parts = jax.lax.all_to_all(parts, axis, 0, 0, tiled=True)
    lses = jax.lax.all_to_all(lses, axis, 0, 0, tiled=True)
    return merge_partials(parts, lses, hadamard, interpret=interpret)
