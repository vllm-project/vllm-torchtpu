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
# The narrowest column count `local_candidates` scans (a vreg's lanes), and
# the fewest rows for which it picks a width per row block: below it a pick
# is latency-bound whatever its width (c4 b1 3.4 us at 1,024 columns), above
# it per-column work (c48 b36 decode: 288 rows x 1,024 columns, 12 us per
# layer, of which ~370 columns are visible).
_MIN_CANDIDATE_WIDTH = 128
_MIN_CANDIDATE_TOKENS = 128
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


# Decode steps of up to this many tokens run `_decode_plan_kernel`: its dense
# per-tile unions cost per tile, and at 16 tiles the XLA path was cheaper
# (standalone bench j-8228a157: 8 tokens 4.0 vs 5.2 us, 48 tokens 4.8 vs 6.6,
# 128 tokens 9.0 vs 7.9).
_PLAN_MAX_TOKENS = 64
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
    prefill twin, j-3c651507: ~240 of 1,024 columns visible). `local_candidates`
    now passes this row-major kernel one full width (its blocks are not
    transposed); the per-block widths remain for the transposed kernel's
    callers and the tests."""
    rank = rank_ref[0]
    width_now = widths_ref[pl.program_id(0)]
    rows = scores_ref.shape[0]
    num_visible = pos_ref[...] // block_size + 1

    def select(width):
        s = scores_ref[:, :width]
        cols = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
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
        picked_scores, picked_blocks = [], []
        for _ in range(topk):
            best_score = jnp.max(s, axis=1, keepdims=True)
            best = jnp.min(
                jnp.where(s == best_score, cols, width), axis=1, keepdims=True
            )
            picked_scores.append(best_score)
            picked_blocks.append(
                jnp.where(best_score > _NEG_INF, best * dcp + rank, -1)
            )
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

    for width in widths:
        pl.when(width_now == width)(functools.partial(select, width))


def _picks_run(width: int, topk: int) -> bool:
    """Whether a block `width` columns wide picks by `_picks_running`."""
    return (
        width >= _RUNNING_PICK_WIDTH
        and width % (8 * _RUNNING_PICK_ACCUMULATORS) == 0
        and topk % 2 == 0
    )


def _picks_reduced(s, cols, width, rank, topk, dcp, score_out_ref, block_out_ref):
    """`_candidates_kernel_t`'s picks of a narrow block ``s [width, rows]``:
    per pick a max, the first column holding it, and its removal."""
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
        _store_pick(k, best_score, best, rank, topk, dcp, score_out_ref, block_out_ref)
        return jnp.where(col_ids == best, _NEG_INF, s)

    if width <= _UNROLLED_WIDTH:
        for k in range(topk):
            s = pick(k, s)
    else:
        jax.lax.fori_loop(0, topk, pick, s, unroll=_PICK_UNROLL)


def _store_pick(k, best_score, best, rank, topk, dcp, score_out_ref, block_out_ref):
    """Pick `k` (score and f32 column ``[1, rows]``) into slot `k`, or
    ``topk - 1 - k`` on odd ranks (`local_candidates`)."""
    slot = jnp.where((rank & 1) == 1, topk - 1 - k, k)
    score_out_ref[pl.ds(slot, 1), :] = best_score
    block_out_ref[pl.ds(slot, 1), :] = jnp.where(
        best_score > _NEG_INF, best.astype(jnp.int32) * dcp + rank, -1
    )


def _picks_running(
    s, width, rank, topk, dcp, score_out_ref, block_out_ref, s_ref, t_ref, col_ref
):
    """`_picks_reduced` of a wide block in one pass per pick over its vreg
    rows: each pass removes the previous pick and keeps a running max and
    the first column holding it per sublane, 5 VALU ops per vreg (a compare
    and a select for the removal, a compare, a max and a select), where the
    max, first-column and removal passes take 6. `col_ref [>= width, rows]`
    holds the column ids; the passes read the block from one of `s_ref` /
    `t_ref` and write it to the other, which lets loads run ahead of the
    stores."""
    rows = s.shape[1]
    s_ref[:width, :] = s
    col_ref[:width, :] = jax.lax.broadcasted_iota(jnp.int32, (width, rows), 0).astype(
        jnp.float32
    )
    sub = _RUNNING_PICK_ACCUMULATORS * 8

    def scan(src, dst, prev):
        def body(v, acc):
            at = pl.ds(pl.multiple_of(v * sub, sub), sub)
            cs = col_ref[at, :]
            xs = jnp.where(
                cs == jnp.tile(prev, (_RUNNING_PICK_ACCUMULATORS, 1)),
                _NEG_INF,
                src[at, :],
            )
            dst[at, :] = xs
            new = []
            for a, (m, idx) in enumerate(acc):
                x, c = xs[8 * a : 8 * (a + 1)], cs[8 * a : 8 * (a + 1)]
                new.append((jnp.maximum(m, x), jnp.where(x > m, c, idx)))
            return tuple(new)

        init = (
            (
                jnp.full((8, rows), _NEG_INF, jnp.float32),
                jnp.full((8, rows), float(width), jnp.float32),
            ),
        ) * _RUNNING_PICK_ACCUMULATORS
        acc = jax.lax.fori_loop(
            0, width // sub, body, init, unroll=_RUNNING_PICK_UNROLL
        )
        m = jnp.concatenate([a[0] for a in acc], axis=0)  # [sub, rows]
        idx = jnp.concatenate([a[1] for a in acc], axis=0)
        best_score = jnp.max(m, axis=0, keepdims=True)
        best = jnp.min(
            jnp.where(m == best_score, idx, float(width)), axis=0, keepdims=True
        )
        return best_score, best

    def two_picks(i, prev):
        for k, (src, dst) in enumerate(((s_ref, t_ref), (t_ref, s_ref))):
            best_score, best = scan(src, dst, prev)
            _store_pick(
                2 * i + k,
                best_score,
                best,
                rank,
                topk,
                dcp,
                score_out_ref,
                block_out_ref,
            )
            prev = jnp.broadcast_to(best, (8, rows))
        return prev

    jax.lax.fori_loop(0, topk // 2, two_picks, jnp.full((8, rows), -1.0, jnp.float32))


def _candidates_kernel_t(
    rank_ref,
    widths_ref,
    scores_ref,
    pos_ref,
    score_out_ref,
    block_out_ref,
    *scratch,
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
        if not _picks_run(width, topk):
            _picks_reduced(
                s, cols, width, rank, topk, dcp, score_out_ref, block_out_ref
            )
        else:
            _picks_running(
                s, width, rank, topk, dcp, score_out_ref, block_out_ref, *scratch
            )

    for width in widths:
        pl.when(width_now == width)(functools.partial(select, width))


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
# Blocks at least this wide pick by `_picks_running` (whose columns must be
# whole groups of `_RUNNING_PICK_ACCUMULATORS` vregs), over
# `_RUNNING_PICK_ACCUMULATORS` independent running maxima, its vreg-row loop
# unrolled `_RUNNING_PICK_UNROLL` times. AOT per 256-row block: 1,024 columns
# ~7.6k bundles (11.7k looped reduced picks). Each pass adds ~40 bundles of
# reductions: from 128 columns with 2 accumulators (AOT ~1.2k vs 1.46k) the c4
# twin's narrow blocks took 45 us per layer against 40 (j-2f091da4).
_RUNNING_PICK_WIDTH = 256
_RUNNING_PICK_ACCUMULATORS = 4
_RUNNING_PICK_UNROLL = 4
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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """This rank's top-k ``(scores [T, topk] f32, blocks [T, topk] i32)``,
    in the selection's order (best first), reversed on ranks of odd index:
    a DCP pair's lists then meet as one bitonic sequence
    (`merge_pair_candidates`), and wider groups' merges take any order.

    With `live_rows` (a traced count), only the row blocks holding rows
    ``[0, live_rows)`` are picked; the other rows' results are unset.

    ``scores [T, 1, P]`` are this rank's block scores (column ``c`` is global
    block ``c * dcp + rank``), or ``[P, T]`` transposed (the large tiles'
    layout in `msa_indexer`); forced blocks carry `msa_select`'s forced
    scores and missing picks are ``(-inf, -1)``.
    """
    pretransposed = scores.ndim == 2
    if pretransposed:
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
        live_rows is None
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
    # Static scan widths (`_scan_widths`, one kernel body each); each row
    # block takes the narrowest one holding every block its rows see (a
    # rank's blocks below n are among its first ceil(n / dcp) columns). Only
    # transposed blocks: row-major ones scan the full width (c48 b23's one
    # block of 184 rows took 4.6 us per layer at 1,024 columns, 18 us with
    # widths, twins j-9662bca6, j-7cf6dd33).
    transposed = block_rows % _TRANSPOSED_ROWS == 0
    widths = (
        _scan_widths(num_cols, pretransposed)
        if transposed and num_tokens >= _MIN_CANDIDATE_TOKENS
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
    scratch = []
    if transposed:
        kernel = functools.partial(_candidates_kernel_t, pretransposed=pretransposed)
        pos_spec = pl.BlockSpec((1, block_rows), lambda i: (0, i))
        p = p.T
        out_spec = pl.BlockSpec((topk, block_rows), lambda i: (0, i))
        out_dims = (topk, num_tokens + pad)
        # `_picks_running`'s block (twice) and column ids, if any width runs.
        running = [w for w in widths if _picks_run(w, topk)]
        if running:
            scratch = [pltpu.VMEM((max(running), block_rows), jnp.float32)] * 3
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
            widths=tuple(widths),
        ),
        grid=(
            num_blocks
            if live_rows is None
            else jnp.clip(-(-live_rows // block_rows), 1, num_blocks),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            (
                pl.BlockSpec((num_cols, block_rows), lambda i: (0, i))
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
    )(rank_array(rank), row_widths, s, p)
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
    query rows in (token, head) order, as laid out: the block scores are
    blind to the row order within a request, and a (head, token) order took
    a row gather of the queries (~45 us per layer at c64's 8K-token mixed
    steps, twin j-a202588d).
    """
    num_tokens, num_heads, head_dim = index_q.shape
    rows = jnp.arange(num_tokens * num_heads, dtype=jnp.int32)
    rows = rows.reshape(num_tokens, num_heads).T  # [G, T]: head h of token t
    folded = index_q.reshape(num_tokens * num_heads, 1, head_dim)
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
        # token o // G of it: position = len - q_len + o // G. Rows of a tile
        # past its request (never gathered back) stay inside its positions,
        # which bound the candidate scan's width.
        per_tile = functools.partial(msa_indexer.lookup, index=scored.tile_seq)
        row = scored.tile_row0[:, None] + jnp.arange(block_q, dtype=jnp.int32)[None, :]
        offset = row - per_tile(folded_cu[:-1])[:, None]
        q_len = jnp.maximum(per_tile(q_lens), 1)[:, None]
        token = jnp.clip(offset // num_heads, 0, q_len - 1)
        col_pos = (per_tile(seq_lens)[:, None] - q_len + token).reshape(-1)
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
        return kv_cache, index_cache, scale_cache, packed.reshape(num_tokens, -1)
    return kv_cache, index_cache, packed.reshape(num_tokens, -1)


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
    leading axis (`ops.msa.merge_head_partials`), or `attend_and_merge`
    sends each rank its block (a pair: a permute of the peer's heads).
    ``q`` may be a pair of head-major arrays holding the heads in order.
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
    block_tokens: int = _PARTIALS_BLOCK_TOKENS,
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
    block_t = max(1, min(num_tokens, block_tokens))
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
    block_tokens: int = _PARTIALS_BLOCK_TOKENS,
    interpret: bool = False,
) -> jax.Array:
    """`merge_pair_partials` of this rank's partial ``out [2 Hl, T, D]``
    (its own heads first: their rows are read in place) and the peer's
    partial of those heads ``peer [Hl, T, D]``, with ``lse [2 Hl, T]`` and
    ``peer_lse [Hl, T]`` f32 -> ``[T, Hl * D]``: no stacking copy of the two."""
    num_heads, num_tokens, head_dim = peer.shape
    block_t = max(1, min(num_tokens, block_tokens))
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
    own_ref,
    peer_ref,
    seq_lens_ref,
    cu_ref,
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
    packed candidates ``[T, 2 * topk]``: the merged selection
    (`merge_candidates`' dense ranking), this rank's page keys ``[T, topk]``,
    each token's local page and limit ``[T, 1]`` (`msa.token_positions`) and
    the attention call's work list (`msa_attention_lse.store_work_list`).
    The same values as the XLA path of `msa_attend_dcp`, which costs a dozen
    small ops of fixed cost each; `msa_attention_lse.planned_attention` runs
    it in its first grid step."""
    dcp = 2
    rank = rank_ref[0]
    num_tokens = own_ref.shape[0]
    iota = functools.partial(jax.lax.broadcasted_iota, jnp.int32)
    # Candidates down the sublanes, tokens across the lanes: candidate j's
    # score and block broadcast over sublanes for the compares.
    halves = []
    for ref in (own_ref, peer_ref):
        c = ref[...].T  # [2 * topk, T]
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
    # Back to a token per row, slot k at lanes k, k + K, ...: token t's keys
    # then sit at lanes [t * K, (t + 1) * K) of its tile row.
    selected = jnp.concatenate([picked.T] * (_LANES // topk), axis=1)  # [T, 128]

    # Token positions from the requests' query starts and lengths.
    num_seqs = seq_lens_ref.shape[1]
    tok = iota((num_tokens, 1), 0)
    cu = cu_ref[...]  # [1, num_seqs + 1]
    req = iota((1, num_seqs + 1), 1)
    seq_id = jnp.sum(jnp.where((req >= 1) & (cu <= tok), 1, 0), axis=1, keepdims=True)
    seq_id = jnp.minimum(seq_id, num_seqs - 1)

    def lookup(table, index):
        cols = iota((1, table.shape[1]), 1)
        return jnp.sum(jnp.where(cols == index, table, 0), axis=1, keepdims=True)

    start = lookup(cu, seq_id)
    q_len = lookup(cu, seq_id + 1) - start
    valid = tok < cu[:, num_seqs : num_seqs + 1]
    pos = lookup(seq_lens_ref[...], seq_id) - q_len + (tok - start)
    pos = jnp.where(valid, pos, 0)
    own_block = pos // block_size
    base = seq_id * pages_per_seq
    local_key_ref[...] = jnp.where(
        valid & (own_block % dcp == rank), base + own_block // dcp, -1
    )
    local_limit_ref[...] = pos % block_size + 1
    mine = (selected >= 0) & (selected % dcp == rank) & valid
    keys = jnp.where(mine, base + selected // dcp, -1)
    keys_ref[...] = keys[:, :topk]

    # Each tile's keys as one 128-lane row: token t's at lanes [t * K, (t + 1) * K).
    num_tiles = num_tokens // block_q
    rows = -(-num_tiles // 8) * 8
    tile_rows = jnp.full((rows, _LANES), -1, jnp.int32)
    tile_row_id = iota((rows, _LANES), 0)
    token_of_lane = iota((1, _LANES), 1) // topk
    for i in range(num_tiles):
        row = jnp.full((1, _LANES), -1, jnp.int32)
        for t in range(block_q):
            token = keys[i * block_q + t : i * block_q + t + 1, :]
            row = jnp.where(token_of_lane == t, token, row)
        tile_rows = jnp.where(tile_row_id == i, row, tile_rows)
    msa_attention_lse.store_work_list(tile_rows, work_ref, num_tiles=num_tiles, **work)


def use_decode_plan(num_tokens: int, topk: int) -> bool:
    """Whether a DCP pair's decode step of ``num_tokens`` rows attends through
    `_attend_pair_decode` (8-token tiles whose work list the kernel builds)."""
    block_q = msa_attention.SMALL_BATCH_BLOCK_Q
    return (
        num_tokens % block_q == 0
        and num_tokens <= _PLAN_MAX_TOKENS
        and _LANES % topk == 0
        and block_q * topk == _LANES
        and msa_attention_lse.fits_work_row(block_q, topk, _SMALL_BATCH_PAGES_PER_STEP)
    )


def _attend_pair_decode(
    q_all,
    kv_cache,
    own,
    peer,
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
    """`msa_attend_dcp` of a pair's decode step, its plan
    (`_decode_plan_kernel`) run by the attention kernel itself
    (`msa_attention_lse.planned_attention`)."""
    num_q_heads, num_tokens, _ = q_all.shape
    block_q = msa_attention.SMALL_BATCH_BLOCK_Q
    pages_per_step = _SMALL_BATCH_PAGES_PER_STEP
    sections, _, tiles_per_step, groups_per_tile = msa_attention_lse.call_work_spec(
        num_tokens, num_q_heads, q_all.dtype, topk, block_q, pages_per_step
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
        q_all,
        kv_cache,
        block_tables.reshape(-1),
        plan,
        rank_array(rank),
        (own, peer, seq_lens.astype(i32)[None], cu_q_lens.astype(i32)[None]),
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


def _swap_with_row(x: jax.Array, extra: jax.Array, axis: str):
    """A DCP pair's ``x [H, T, D]`` (16-bit) and ``extra [T, n]`` (32-bit)
    swapped by one collective permute: the bits of `extra` ride as one more
    uint16 head row (every permute start costs ~1 us on the TensorCore at
    decode, c4 twin j-6bbe97e2). As 16-bit integers: float copies may
    canonicalize NaN or flush subnormal bit patterns."""
    _, num_tokens, head_dim = x.shape
    width = 2 * extra.shape[1]
    if x.dtype.itemsize != 2 or extra.dtype.itemsize != 4 or width > head_dim:
        raise ValueError(
            f"16-bit x and 32-bit extra of at most {head_dim // 2} columns, got "
            f"{x.dtype} {x.shape}, {extra.dtype} {extra.shape}"
        )
    bits = jax.lax.bitcast_convert_type(extra, jnp.uint16).reshape(1, num_tokens, width)
    bits = jnp.pad(bits, ((0, 0), (0, 0), (0, head_dim - width)))
    send = jnp.concatenate([jax.lax.bitcast_convert_type(x, jnp.uint16), bits])
    got = jax.lax.ppermute(send, axis, [(0, 1), (1, 0)])
    peer_extra = got[-1, :, :width].reshape(num_tokens, width // 2, 2)
    return (
        jax.lax.bitcast_convert_type(got[:-1], x.dtype),
        jax.lax.bitcast_convert_type(peer_extra, extra.dtype),
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
    if small and use_decode_plan(num_tokens, topk):
        out, lse = _attend_pair_decode(
            q_all,
            kv_cache,
            candidates,
            peer_candidates,
            seq_lens,
            block_tables,
            cu_q_lens,
            rank,
            sm_scale=sm_scale,
            topk=topk,
            scale_cache=scale_cache,
            interpret=interpret,
        )
    else:
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
