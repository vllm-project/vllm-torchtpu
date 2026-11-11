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
   per token (an all-gather of ``[2 * topk, T]`` int32, `to_slots`) gives
   the exact non-DCP selection.
2. `msa_attend_dcp`: each rank merges the candidates into that selection,
   attends every query head of the KV head over the selected blocks it
   holds and returns the partial output with its log-sum-exp; the caller
   merges the ranks' partials (`ops.msa.merge_head_partials`).

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
)

_NEG_INF = float("-inf")
# The narrowest column count `local_candidates` scans (a vreg's lanes), and
# the fewest tokens for which it picks a width per row block.
_MIN_CANDIDATE_WIDTH = 128
_MIN_CANDIDATE_TOKENS = 1024

# Up to this many tokens (`msa_attention.SMALL_BATCH_TOKENS` is the non-DCP
# limit) an 8-token verify group fills an attention tile alone
# (no padding rows, half the page union of a 16-token tile). Larger decode
# batches keep 16-token tiles: at 48-64 tiles of 8 the per-tile work list and
# kernel steps cost more than the smaller unions save (c48 b39 decode twin
# 108.4 -> 116.7 ms per step, c64 DCP8 b57 147.8 -> 191.1 ms; j-82660a13 vs
# j-8a29ceb0, j-c0fd0e40 vs j-11d1908c).
DCP_SMALL_BATCH_TOKENS = 128


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
        score_out_ref[...] = jnp.concatenate(picked_scores, axis=1)
        block_out_ref[...] = jnp.concatenate(picked_blocks, axis=1)

    for width in widths:
        pl.when(width_now == width)(functools.partial(select, width))


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
    fine_widths=False,
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
        if fine_widths:
            _pick_looped(s, cols, width, score_out_ref, block_out_ref, topk, dcp, rank)
            return
        picked_scores, picked_blocks = [], []
        for _ in range(topk):
            best_score = jnp.max(s, axis=0, keepdims=True)
            best = jnp.min(
                jnp.where(s == best_score, cols, width), axis=0, keepdims=True
            )
            picked_scores.append(best_score)
            picked_blocks.append(
                jnp.where(best_score > _NEG_INF, best * dcp + rank, -1)
            )
            s = jnp.where(cols == best, _NEG_INF, s)
        score_out_ref[...] = jnp.concatenate(picked_scores, axis=0)
        block_out_ref[...] = jnp.concatenate(picked_blocks, axis=0)

    for width in widths:
        pl.when(width_now == width)(functools.partial(select, width))


def _pick_looped(s, cols, width, score_out_ref, block_out_ref, topk, dcp, rank):
    """`_candidates_kernel_t`'s picks for `FINE_SCAN_WIDTHS` (r6): each pick
    stores its row as it goes, ties resolve on f32 column ids (exact below
    2**24: one f32 min instead of an int32 compare and select), and blocks
    wider than `_UNROLLED_WIDTH` loop over the picks instead of unrolling
    them. Same picks as the unrolled loop."""
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


# r6 (opt-in, `FINE_SCAN_WIDTHS`): scan widths step by this many
# pretransposed columns up to `_FINE_WIDTH_LIMIT`, then by 128.
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
# Set from TPU_MINIMAX_M3_DCP_FINE_SCAN_WIDTHS by the attention layer.
FINE_SCAN_WIDTHS = False


def _scan_widths(num_cols: int, pretransposed: bool) -> list[int]:
    """`local_candidates`' scan widths up to `num_cols` with
    `FINE_SCAN_WIDTHS`. A pick costs per scanned column, so widths step
    finely: by `_FINE_WIDTH_STEP` columns (sublanes of the pretransposed
    scores) up to `_FINE_WIDTH_LIMIT`, then by a vreg's 128 lanes (powers of
    two cost ~25% more scan on average: c64's 146-column blocks scanned
    256)."""
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
        "live_select",
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
    live_select: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """This rank's top-k ``(scores [T, topk] f32, blocks [T, topk] i32)``.

    With `live_rows` (a traced count), only the row blocks holding rows
    ``[0, live_rows)`` are picked; the other rows' results are unset.

    With `live_select`, a batch that would scan the full width (at most
    `_MIN_CANDIDATE_TOKENS` rows) copies in and scans only the columns its
    row block sees (`_live_candidates_kernel`), with the same picks.

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
    if pretransposed:
        block_rows = -(-block_rows // _TRANSPOSED_ROWS) * _TRANSPOSED_ROWS
    pad = -num_tokens % block_rows
    if pretransposed:
        s = jnp.pad(scores, ((0, 0), (0, pad)))
    else:
        s = jnp.pad(scores[:, 0], ((0, pad), (0, 0)))
    p = jnp.pad(pos, (0, pad))[:, None]
    # Static scan widths: 128 x powers of two below the full width, and the
    # full width (one kernel body each); each row block takes the narrowest
    # one holding every block its rows see (a rank's blocks below n are among
    # its first ceil(n / dcp) columns). A decode batch scans the full width:
    # its scan is a few us, and picking a width would cost more.
    fine_widths = FINE_SCAN_WIDTHS
    if fine_widths and num_tokens > _MIN_CANDIDATE_TOKENS:
        widths = _scan_widths(num_cols, pretransposed)
    else:
        widths = [num_cols]
        while widths[0] > _MIN_CANDIDATE_WIDTH and num_tokens > _MIN_CANDIDATE_TOKENS:
            widths.insert(
                0, max(_MIN_CANDIDATE_WIDTH, 1 << (widths[0] - 1).bit_length() - 1)
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
    if live_select and len(widths) == 1 and _live_fits(num_cols, pretransposed):
        picked_scores, picked_blocks = _live_candidates(
            s,
            p.T if transposed else p,
            rank_array(rank),
            num_blocks=num_blocks,
            block_rows=block_rows,
            transposed=transposed,
            pretransposed=pretransposed,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            dcp=dcp,
            live_rows=live_rows,
            interpret=interpret,
        )
        if transposed:
            picked_scores, picked_blocks = picked_scores.T, picked_blocks.T
        return picked_scores[:num_tokens], picked_blocks[:num_tokens]
    if transposed:
        kernel = functools.partial(
            _candidates_kernel_t, pretransposed=pretransposed, fine_widths=fine_widths
        )
        pos_spec = pl.BlockSpec((1, block_rows), lambda i: (0, i))
        p = p.T
        out_spec = pl.BlockSpec((topk, block_rows), lambda i: (0, i))
        out_dims = (topk, num_tokens + pad)
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


# Set from `--additional-config '{"minimax_msa_live_select": true}'` at layer
# init (`attention.live_select_enabled`); read at trace time. Off by default.
LIVE_SELECT_ENABLED = False

# Columns a `_live_candidates_kernel` loop step scans when they sit on
# sublanes (8 vregs of a 128-row block), and columns per copy when the
# scores keep them on lanes.
_LIVE_CHUNK = 64
_LANES = 128
_NO_COL = jnp.iinfo(jnp.int32).max


def _live_fits(num_cols: int, pretransposed: bool) -> bool:
    """Whether `_live_candidates` takes scores of `num_cols` columns: whole
    vregs per copy (8 sublanes or 128 lanes)."""
    return num_cols % (8 if pretransposed else _LANES) == 0


def _live_columns(
    pos: jax.Array, rank: jax.Array, *, block_size: int, dcp: int, num_cols: int
) -> jax.Array:
    """Columns any row of each row block of ``pos [blocks, rows]`` sees: the
    rank's blocks below ``n`` are its first ``ceil((n - rank) / dcp)``
    columns, so this is the local page count (`local_seq_lens`) of the
    block's longest request."""
    num_visible = jnp.max(pos, axis=1) // block_size + 1
    return jnp.clip(-(-(num_visible - rank) // dcp), 0, num_cols)


def _live_chunks(need, cols: int):
    """Loop steps of `cols` columns that cover the first `need` columns."""
    return (need + cols - 1) // cols


def _live_candidates_kernel(
    # Scalar prefetch
    counts_ref,  # [2] rank, grid steps
    need_ref,  # [row blocks] columns any row of the block sees
    # In
    scores_hbm,  # [T, P] (or [P, T] pretransposed) scores in HBM
    pos_ref,  # [rows, 1] (or [1, rows] transposed)
    # Out
    score_out_ref,  # [rows, topk] (or [topk, rows] transposed)
    block_out_ref,
    # Scratch
    buf_ref,  # [2, rows, P] (or [2, P, rows] pretransposed): two blocks
    sems,  # DMA semaphores, one per buffer
    masked_ref,  # [rows, P] (or [P, rows] transposed): masked scores
    *,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    dcp,
    col_axis,
    src_axis,
    copy_cols,
    scan_cols,
):
    """`_candidates_kernel` (``col_axis`` 1) or `_candidates_kernel_t`
    (``col_axis`` 0) over the first ``need_ref[i]`` columns of row block
    ``i`` only, with the same picks: a column past every row's last visible
    block is ``-inf`` for all of them, so it only ever ties a pick that is
    already a miss ``(-inf, -1)``.

    The block's live columns are copied in ``copy_cols`` at a time (the next
    block's while this one picks). Each pick scans ``scan_cols`` at a time
    and keeps a running max and its first column per vreg slot (strict ``>``
    over columns in increasing order keeps the lowest of equal scores), then
    reduces the slots once; the previous pick is cleared on the same scan."""
    i = pl.program_id(0)
    rank, steps = counts_ref[0], counts_ref[1]
    slot = i % 2
    num_visible = pos_ref[...] // block_size + 1
    rows = num_visible.shape[1 - col_axis]
    unit = 8 if col_axis == 0 else _LANES
    slot_shape = (unit, rows) if col_axis == 0 else (rows, unit)
    slot_col = jax.lax.broadcasted_iota(jnp.int32, slot_shape, col_axis)

    def num_chunks(block, cols):
        return _live_chunks(need_ref[block], cols)

    def copy_block(block, buf, op):
        def body(c, carry):
            cols = pl.ds(pl.multiple_of(c * copy_cols, copy_cols), copy_cols)
            block_rows = pl.ds(pl.multiple_of(block * rows, rows), rows)
            src, dst = (
                (scores_hbm.at[cols, block_rows], buf_ref.at[buf, cols])
                if src_axis == 0
                else (scores_hbm.at[block_rows, cols], buf_ref.at[buf, :, cols])
            )
            getattr(pltpu.make_async_copy(src, dst, sems.at[buf]), op)()
            return carry

        jax.lax.fori_loop(0, num_chunks(block, copy_cols), body, 0)

    @pl.when(i == 0)
    def _first():
        copy_block(0, 0, "start")

    @pl.when(i + 1 < steps)
    def _next():
        copy_block(i + 1, 1 - slot, "start")

    copy_block(i, slot, "wait")

    def chunk_cols(base, n):
        shape = (n, rows) if col_axis == 0 else (rows, n)
        return jax.lax.broadcasted_iota(jnp.int32, shape, col_axis) + base

    def masked_at(base, n):
        cols = pl.ds(pl.multiple_of(base, n), n)
        return masked_ref.at[cols] if col_axis == 0 else masked_ref.at[:, cols]

    def accumulate(best, s, base):
        m, col = best
        for k in range(s.shape[col_axis] // unit):
            v = jax.lax.slice_in_dim(s, k * unit, (k + 1) * unit, axis=col_axis)
            gt = v > m
            m = jnp.where(gt, v, m)
            col = jnp.where(gt, slot_col + (base + k * unit), col)
        return m, col

    def first_scan(c, best):
        base = c * copy_cols
        cols = pl.ds(pl.multiple_of(base, copy_cols), copy_cols)
        s = buf_ref[slot, cols] if src_axis == 0 else buf_ref[slot, :, cols]
        if src_axis != col_axis:
            s = s.T
        blocks = chunk_cols(base, copy_cols) * dcp + rank
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
        masked_at(base, copy_cols)[...] = s
        return accumulate(best, s, base)

    def later_scan(picked):
        def body(c, best):
            base = c * scan_cols
            ref = masked_at(base, scan_cols)
            s = jnp.where(chunk_cols(base, scan_cols) == picked, _NEG_INF, ref[...])
            ref[...] = s
            return accumulate(best, s, base)

        return body

    init = (
        jnp.full(slot_shape, _NEG_INF, jnp.float32),
        jnp.full(slot_shape, _NO_COL, jnp.int32),
    )
    scan, trips = first_scan, num_chunks(i, copy_cols)
    picked_scores, picked_blocks = [], []
    for _ in range(topk):
        m, col = jax.lax.fori_loop(0, trips, scan, init)
        best_score = jnp.max(m, axis=col_axis, keepdims=True)
        best = jnp.min(
            jnp.where(m == best_score, col, _NO_COL), axis=col_axis, keepdims=True
        )
        picked_scores.append(best_score)
        picked_blocks.append(jnp.where(best_score > _NEG_INF, best * dcp + rank, -1))
        scan, trips = later_scan(best), num_chunks(i, scan_cols)
    score_out_ref[...] = jnp.concatenate(picked_scores, axis=col_axis)
    block_out_ref[...] = jnp.concatenate(picked_blocks, axis=col_axis)


def _live_candidates(
    s: jax.Array,
    p: jax.Array,
    rank: jax.Array,
    *,
    num_blocks: int,
    block_rows: int,
    transposed: bool,
    pretransposed: bool,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    dcp: int,
    live_rows: jax.Array | None,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    """`local_candidates`' pick of padded scores ``s`` (``[T, P]``, or
    ``[P, T]`` pretransposed) and positions ``p`` (``[T, 1]``, or ``[1, T]``
    when `transposed`) with `_live_candidates_kernel`: picks ``[T, topk]``
    (``[topk, T]`` when `transposed`)."""
    num_cols = s.shape[0] if pretransposed else s.shape[1]
    col_axis = 0 if transposed else 1
    src_axis = 0 if pretransposed else 1
    if src_axis == 0:
        copy_cols = _select_chunk(num_cols)
    else:
        copy_cols = _LANES
    scan_cols = min(_LIVE_CHUNK, copy_cols) if col_axis == 0 else _LANES
    need = _live_columns(
        p.reshape(num_blocks, block_rows),
        rank[0],
        block_size=block_size,
        dcp=dcp,
        num_cols=num_cols,
    )
    steps = (
        num_blocks
        if live_rows is None
        else jnp.clip(-(-live_rows // block_rows), 1, num_blocks).astype(jnp.int32)
    )
    if transposed:
        pos_spec = pl.BlockSpec((1, block_rows), lambda i, *_: (0, i))
        out_spec = pl.BlockSpec((topk, block_rows), lambda i, *_: (0, i))
        out_dims = (topk, num_blocks * block_rows)
    else:
        pos_spec = pl.BlockSpec((block_rows, 1), lambda i, *_: (i, 0))
        out_spec = pl.BlockSpec((block_rows, topk), lambda i, *_: (i, 0))
        out_dims = (num_blocks * block_rows, topk)
    buf_block = (num_cols, block_rows) if src_axis == 0 else (block_rows, num_cols)
    masked_block = (num_cols, block_rows) if col_axis == 0 else (block_rows, num_cols)
    kernel = functools.partial(
        _live_candidates_kernel,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        dcp=dcp,
        col_axis=col_axis,
        src_axis=src_axis,
        copy_cols=copy_cols,
        scan_cols=scan_cols,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(steps,),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY), pos_spec],
            out_specs=[out_spec, out_spec],
            scratch_shapes=[
                pltpu.VMEM((2, *buf_block), s.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM(masked_block, s.dtype),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct(out_dims, jnp.float32),
            jax.ShapeDtypeStruct(out_dims, jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            # The next block's copy starts in this block's step.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=msa_select._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_dcp_live_candidates",
    )(
        jnp.stack([rank[0], jnp.asarray(steps, jnp.int32)]),
        need.astype(jnp.int32),
        s,
        p,
    )


def pack_candidates(scores: jax.Array, blocks: jax.Array) -> jax.Array:
    """``[T, 2 * topk]`` f32: scores, then the block ids as f32 values, so one
    all-gather moves both.

    The ids travel as values (exact below 2**24; a block id is below
    max_model_len / block_size), not bitcast: a bitcast id is a subnormal
    f32 (and -1 a NaN), which a TPU collective may flush to zero or
    canonicalize, and every id would decode as block 0.
    """
    return jnp.concatenate([scores, blocks.astype(jnp.float32)], axis=-1)


def _slot_bits(picks: jax.Array, topk: int) -> jax.Array:
    """Packed candidates as columns, ``[2 * topk, N]`` f32 (`pack_candidates`
    transposed) -> int32: the scores' bit patterns, then the block ids."""
    scores = jax.lax.bitcast_convert_type(picks[:topk], jnp.int32)
    return jnp.concatenate([scores, picks[topk:].astype(jnp.int32)], axis=0)


def to_slots(packed: jax.Array, topk: int) -> jax.Array:
    """`pack_candidates` rows of ``G`` heads, ``[T, G * 2 * topk]`` (head-major
    per token) -> the all-gather layout ``[2 * topk, T * G]`` int32.

    Column ``t * G + h`` holds head ``h`` of token ``t``: rows ``[0, topk)``
    the scores' f32 bit patterns, rows ``[topk, 2 * topk)`` the block ids
    (``-1``: none). The collective moves int32, so nothing canonicalizes the
    bits, and with the ``T * G`` columns on lanes the payload is not padded:
    ``[T, 2 * topk]`` rows pad to 128 lanes, 2-4x the bytes, and the
    all-gather's time follows the padded bytes (DCP4, 8192 tokens: f32 [T,
    32] and [T, 64] both ~145-150 us, j-1597224e).
    """
    rows = packed.reshape(-1, 2 * topk)
    return _slot_bits(rows.T, topk)


def merge_candidates(candidates: jax.Array, topk: int) -> jax.Array:
    """Global selection ``[T, topk]`` (``-1`` padded) from every rank's
    packed candidates ``[dcp, T, 2 * topk]``, in `msa_select.select_blocks`'s
    order: score descending, ties to the lower block."""
    dcp, num_tokens, _ = candidates.shape
    scores = candidates[..., :topk]
    blocks = candidates[..., topk:].astype(jnp.int32)
    s = scores.transpose(0, 2, 1).reshape(dcp * topk, num_tokens)
    b = blocks.transpose(0, 2, 1).reshape(dcp * topk, num_tokens)
    return _rank_merge(s, b, topk).T


def _rank_merge(s: jax.Array, b: jax.Array, topk: int) -> jax.Array:
    """`merge_candidates` of candidates ``(s, b) [n, N]`` (candidate ``i`` of
    column ``c``) -> ``[topk, N]``."""
    # The real candidates (score > -inf) are distinct blocks: a rank picks
    # distinct columns and ranks hold disjoint blocks. So each one's place in
    # the order is the number of real candidates ahead of it, and the merge
    # is two dense compare-reduce fusions instead of topk dependent steps.
    # Their [n, n, N] operands stay fused into the reductions.
    real = s > _NEG_INF
    s_i, b_i = s[:, None, :], b[:, None, :]
    s_j, b_j = s[None, :, :], b[None, :, :]
    ahead = real[None, :, :] & ((s_j > s_i) | ((s_j == s_i) & (b_j < b_i)))
    place = jnp.where(real, jnp.sum(ahead, axis=1, dtype=jnp.int32), topk)
    slot = jnp.arange(topk, dtype=jnp.int32)[None, :, None]
    return jnp.max(jnp.where(place[:, None, :] == slot, b_i, -1), axis=0)


def _merge_kernel_t(c_ref, o_ref, *, topk: int, sub: int):
    """`merge_candidates` of a block of columns, in VMEM, `sub` columns at a
    time: ``topk`` steps of (best score, lowest block among its ties) over
    the ``dcp * topk`` candidates. Candidate slots on sublanes and columns on
    lanes, so a column's candidates are whole vreg rows of the ranks' blocks
    (no lane shuffles) and each pick reduces across vregs and sublanes on
    the VPU instead of across lanes on the XLU. ``c_ref [dcp, 2 * topk,
    Bc]`` (`to_slots` of each rank), ``o_ref [Bc, topk]``."""
    dcp, _, block_c = c_ref.shape
    no_block = jnp.iinfo(jnp.int32).max

    def body(i, carry):
        cols = pl.ds(pl.multiple_of(i * sub, sub), sub)
        s = jnp.concatenate(
            [
                jax.lax.bitcast_convert_type(c_ref[r, :topk, cols], jnp.float32)
                for r in range(dcp)
            ],
            axis=0,
        )
        b = jnp.concatenate([c_ref[r, topk:, cols] for r in range(dcp)], axis=0)
        out = []
        for _ in range(topk):
            best_score = jnp.max(s, axis=0, keepdims=True)  # [1, sub]
            best = jnp.min(
                jnp.where(s == best_score, b, no_block), axis=0, keepdims=True
            )
            out.append(jnp.where(best_score > _NEG_INF, best, -1))
            s = jnp.where(b == best, _NEG_INF, s)
        o_ref[cols, :] = jnp.concatenate(out, axis=0).T
        return carry

    jax.lax.fori_loop(0, block_c // sub, body, 0)


def _sublane_partner(x: jax.Array, d: int) -> jax.Array:
    """``x[i ^ d]`` over the 8 sublanes of ``x [8, N]`` (``d`` in 1, 2, 4)."""
    row = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    return jnp.where(row & d, pltpu.roll(x, d, 0), pltpu.roll(x, 8 - d, 0))


def _beats(sa, ba, sb, bb):
    """Candidate a ahead of b in the selection order: higher score, ties to
    the lower block."""
    return (sa > sb) | ((sa == sb) & (ba < bb))


def _merge_sorted(a, b):
    """Top ``8 m`` of two candidate lists, each sorted (score descending,
    ties to the lower block) as ``m`` groups ``(s, b) [8, N]`` of 8 slots
    (slot ``8 j + i`` is sublane ``i`` of group ``j``), sorted the same way:
    a bitonic half-cleaner against ``b`` reversed, then a bitonic sort."""
    m = len(a)

    def reverse(x):
        for d in (4, 2, 1):
            x = _sublane_partner(x, d)
        return x

    rb = [(reverse(s), reverse(k)) for s, k in b[::-1]]
    out = []
    for (sa, ka), (sb, kb) in zip(a, rb):
        take = _beats(sa, ka, sb, kb)
        out.append((jnp.where(take, sa, sb), jnp.where(take, ka, kb)))
    # Bitonic clean-up: slot i against slot i + d, i with bit d clear keeps
    # the leader.
    d = 4 * m
    while d >= 8:
        g = d // 8
        for j in range(m):
            if j & g:
                continue
            (s0, k0), (s1, k1) = out[j], out[j + g]
            take = _beats(s0, k0, s1, k1)
            out[j] = (jnp.where(take, s0, s1), jnp.where(take, k0, k1))
            out[j + g] = (jnp.where(take, s1, s0), jnp.where(take, k1, k0))
        d //= 2
    while d >= 1:
        for j in range(m):
            s, k = out[j]
            ps, pk = _sublane_partner(s, d), _sublane_partner(k, d)
            row = jax.lax.broadcasted_iota(jnp.int32, s.shape, 0)
            take = _beats(s, k, ps, pk) == ((row & d) == 0)
            out[j] = (jnp.where(take, s, ps), jnp.where(take, k, pk))
        d //= 2
    return out


def _merge_network_kernel(c_ref, o_ref, *, topk: int, sub: int):
    """`_merge_kernel_t` for per-rank lists that come sorted (score
    descending, ties to the lower block; every candidate kernel picks in
    that order): a tree of `_merge_sorted` over the ranks instead of
    ``topk`` max-extractions over all ``dcp * topk`` candidates, ~3x fewer
    vector ops at DCP 4, and no cross-sublane reduction."""
    dcp, _, block_c = c_ref.shape

    def body(i, carry):
        cols = pl.ds(pl.multiple_of(i * sub, sub), sub)
        lists = []
        for r in range(dcp):
            s = jax.lax.bitcast_convert_type(c_ref[r, :topk, cols], jnp.float32)
            b = c_ref[r, topk:, cols]
            lists.append([(s[j : j + 8], b[j : j + 8]) for j in range(0, topk, 8)])
        while len(lists) > 1:
            merged = [
                _merge_sorted(lists[k], lists[k + 1])
                for k in range(0, len(lists) - 1, 2)
            ]
            lists = merged + lists[len(lists) - len(lists) % 2 :]
        s = jnp.concatenate([g[0] for g in lists[0]], axis=0)
        b = jnp.concatenate([g[1] for g in lists[0]], axis=0)
        o_ref[cols, :] = jnp.where(s > _NEG_INF, b, -1).T
        return carry

    jax.lax.fori_loop(0, block_c // sub, body, 0)


def _network_merge(topk: int) -> bool:
    """Whether `_merge_network_kernel` merges lists of `topk`: whole groups
    of 8 slots, a power of two of them."""
    groups = topk // 8
    return topk % 8 == 0 and groups & (groups - 1) == 0


# Merges of more than this many candidates per column (DCP 8: 128) run
# `_merge_kernel_t`: `merge_candidates`' [n, n, columns] ranking costs ~3 ms
# per layer in the c64 mixed step (32768 columns; j-c1444339). Narrower
# merges keep the dense ranking, cheap at n = 32. A `_network_merge` topk
# runs `_merge_network_kernel` at every DCP size (one merge at DCP 2).
_KERNEL_MERGE_MIN_CANDIDATES = 64
# Columns per grid step, and per loop step inside it (as many as the
# previous 256-token blocks of G heads held, with a head's live values).
_MERGE_BLOCK_COLUMNS = 1024
# DCP sizes the network merge serves (below the min, the dense ranking).
_NETWORK_MERGE_MIN_DCP = 2
_NETWORK_MERGE_MAX_DCP = 8
_MERGE_SUB_COLUMNS = 256


def merge_head_candidates(
    candidates: jax.Array, topk: int, num_heads: int = 1, interpret: bool = False
) -> jax.Array:
    """`merge_candidates` per KV head: every rank's `to_slots` candidates,
    ``[dcp, 2 * topk, T * G]`` int32 -> selection ``[T, G, topk]``."""
    dcp, _, num_cols = candidates.shape
    network = _network_merge(topk) and dcp <= _NETWORK_MERGE_MAX_DCP
    if dcp * topk < _KERNEL_MERGE_MIN_CANDIDATES and not (
        network and dcp >= _NETWORK_MERGE_MIN_DCP
    ):
        s = jax.lax.bitcast_convert_type(candidates[:, :topk], jnp.float32)
        b = candidates[:, topk:]
        merged = _rank_merge(
            s.reshape(dcp * topk, num_cols), b.reshape(dcp * topk, num_cols), topk
        )
        return merged.T.reshape(-1, num_heads, topk)
    lanes = -(-num_cols // _TRANSPOSED_ROWS) * _TRANSPOSED_ROWS
    # The network holds every rank's list live: 128 columns at a time.
    sub = min(_TRANSPOSED_ROWS if network else _MERGE_SUB_COLUMNS, lanes)
    block_c = min(_MERGE_BLOCK_COLUMNS, -(-lanes // sub) * sub)
    pad = -num_cols % block_c
    c = jnp.pad(candidates, ((0, 0), (0, 0), (0, pad)))
    out = pl.pallas_call(
        functools.partial(
            _merge_network_kernel if network else _merge_kernel_t, topk=topk, sub=sub
        ),
        grid=((num_cols + pad) // block_c,),
        in_specs=[pl.BlockSpec((dcp, 2 * topk, block_c), lambda i: (0, 0, i))],
        out_specs=pl.BlockSpec((block_c, topk), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_cols + pad, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_dcp_merge",
    )(c)
    return out[:num_cols].reshape(-1, num_heads, topk)


def _per_tile(tile_seq: jax.Array, table: jax.Array) -> jax.Array:
    """``table[tile_seq]`` for a per-request table, by a one-hot over the
    requests (a gather from a small table lowers to a compare chain)."""
    onehot = tile_seq[:, None] == jnp.arange(table.shape[0], dtype=jnp.int32)
    return jnp.sum(jnp.where(onehot, table[None, :], 0), axis=1)


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
        per_tile = functools.partial(_per_tile, scored.tile_seq)
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
        start = scored.tile_start[seq_of_token][None]
        off = rows - start
        col = (scored.first_tile[seq_of_token][None] + off // block_q) * block_q
        cols.append(jnp.clip(col + off % block_q, 0, picks[-1].shape[0] - 1))
    # A head's and token's picks: its large-class column if its request is
    # large, else its small-class one.
    col = jnp.where(
        plan.is_large[seq_of_token][None], picks[0].shape[0] + cols[1], cols[0]
    )
    packed = jnp.concatenate(picks, axis=0)[col.reshape(-1)]  # [G * T, 2 * topk]
    return (
        packed.reshape(num_heads, num_tokens, -1)
        .transpose(1, 0, 2)
        .reshape(num_tokens, -1)
    )


# Steps of at least this many tokens (prefill buckets) take
# `_tile_candidates`: index heads folded token-major (a free reshape, no
# `msa.fold_index_heads` scatter), and both tile classes' picks written
# straight into the all-gather's slots by `_tile_select_kernel`. Decode steps keep `_class_candidates` /
# the row path.
TILE_SELECT_MIN_TOKENS = 1024


# Set from `--additional-config '{"minimax_msa_tile_select": true}'` at layer
# init (`attention.tile_select_enabled`); read at trace time. Off by default.
TILE_SELECT_ENABLED = False


def tile_select(num_tokens: int, num_heads: int = 1) -> bool:
    """Whether a step of `num_tokens` tokens and `num_heads` KV heads per
    rank picks its candidates with `_tile_candidates`."""
    return (
        TILE_SELECT_ENABLED
        and num_tokens >= TILE_SELECT_MIN_TOKENS
        and num_heads & (num_heads - 1) == 0
    )


_NO_ROW = jnp.iinfo(jnp.int32).max
# The bit pattern of a -inf f32 score (a missing pick).
_NEG_INF_BITS = -(1 << 23)


def _select_chunk(num_pages: int) -> int:
    """Pages per loop step of `_tile_select_kernel`: whole vregs (8 sublanes)
    that divide the class's padded page count."""
    for chunk in (64, 32, 16, 8):
        if num_pages % chunk == 0:
            return chunk
    raise ValueError(f"padded pages {num_pages} are not a multiple of 8")


def _place(out_ref, values, src0, dst, n):
    """Copy lanes ``[src0, src0 + n)`` of ``values [rows, W]`` (``W`` a
    multiple of 128, ``src0 + n <= W``) to columns ``[dst, dst + n)`` of the
    VMEM ``out_ref [rows, cols]`` and leave every other column as it is.
    Requests start at any column and a store's lane offset must be
    128-aligned, so the values are rotated into the aligned window of ``W +
    128`` columns that holds ``[dst, dst + n)`` and selected against its
    current contents."""
    rows, width = values.shape
    window = width + 128
    base = jnp.minimum(dst // 128 * 128, out_ref.shape[1] - window)
    base = pl.multiple_of(base, 128)
    off = dst - base
    ext = jnp.concatenate([values, jnp.zeros((rows, 128), values.dtype)], axis=1)
    moved = pltpu.roll(ext, (off - src0 + window) % window, 1)
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, window), 1)
    keep = (lane >= off) & (lane < off + n)
    out_ref[:, pl.ds(base, window)] = jnp.where(
        keep, moved, out_ref[:, pl.ds(base, window)]
    )


def _tile_rows(row0, block_q, seq_row0_ref, seq):
    """The folded rows ``[lo, hi)`` of request `seq` in the tile of
    `block_q` rows from `row0`: a tile may start below its request's first
    row (`msa_indexer` aligns tile starts) and end past its last."""
    lo = jnp.maximum(row0, seq_row0_ref[seq])
    return lo, jnp.maximum(jnp.minimum(row0 + block_q, seq_row0_ref[seq + 1]), lo)


def _tile_select_kernel(
    # Scalar prefetch
    tile_seq_ref,  # [tiles] request of each large tile
    tile_row0_ref,  # [tiles] first (folded) query row of each large tile
    small_seq_ref,  # [small tiles] request of each small tile
    small_row0_ref,  # [small tiles] first folded query row of each small tile
    seq_row0_ref,  # [max_num_seqs + 1] first folded query row of each request
    first_pos_ref,  # [max_num_seqs] position of each request's first token
    seq_pages_ref,  # [max_num_seqs] this rank's pages of each request
    counts_ref,  # [3] rank, live large tiles, live small tiles
    # In
    scores_hbm,  # [P, tiles * Bq] transposed block scores of the large class
    small_ref,  # [2 * topk, small columns] the small class's slots
    # Out
    out_ref,  # [2 * topk, columns] slots of every folded row (VMEM resident)
    # Scratch
    scores_ref,  # [2, P, Bq] two tiles' live pages
    sems,  # DMA semaphores, one per buffer
    masked_ref,  # [P, Bq]
    *,
    heads_shift,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    dcp,
    chunk,
    small_q,
    small_width,
):
    """The step's slots ``[2 * topk, T * G]`` (`to_slots`; column = folded
    row ``t * G + h``), written in place: at the first step every column is
    set to none and the small class's picks (`local_candidates`, tile-major
    columns) are moved to their rows' columns; each step then runs
    `_candidates_kernel_t` for every query row (lane) of one large tile and
    writes its rows' columns. Only the request's own pages on this rank are
    scanned, `chunk` at a time; each pick keeps a per-sublane running max and
    its first page, then reduces the 8 sublanes once. A tile's scores are
    copied in for its live pages only (the class pads every tile to the
    longest request's pages), the next tile's while this one picks; its
    picks stay in vregs and are written once."""
    tile = pl.program_id(0)
    rank = counts_ref[0]
    _, num_pages, block_q = scores_ref.shape
    slot = tile % 2

    def live_chunks(t):
        pages = jnp.minimum(seq_pages_ref[tile_seq_ref[t]], num_pages)
        return (pages + chunk - 1) // chunk

    def copy_tile(t, buf, op):
        def body(c, carry):
            rows = pl.ds(pl.multiple_of(c * chunk, chunk), chunk)
            copy = pltpu.make_async_copy(
                scores_hbm.at[rows, pl.ds(pl.multiple_of(t * block_q, block_q), block_q)],
                scores_ref.at[buf, rows],
                sems.at[buf],
            )
            getattr(copy, op)()
            return carry

        jax.lax.fori_loop(0, live_chunks(t), body, 0)

    @pl.when((tile == 0) & (counts_ref[1] > 0))
    def _first():
        copy_tile(0, 0, "start")

    @pl.when(tile + 1 < counts_ref[1])
    def _next():
        copy_tile(tile + 1, 1 - slot, "start")

    @pl.when(tile == 0)
    def _init():
        none = jnp.full((topk, out_ref.shape[1]), _NEG_INF_BITS, jnp.int32)
        out_ref[:topk, :] = none
        out_ref[topk:, :] = jnp.full_like(none, -1)

        def place_small(t, carry):
            row0, seq = small_row0_ref[t], small_seq_ref[t]
            lo, hi = _tile_rows(row0, small_q, seq_row0_ref, seq)
            src = t * small_q + lo - row0
            group = pl.multiple_of(t * small_q // 128 * 128, 128)
            values = small_ref[:, pl.ds(group, small_width)]
            _place(out_ref, values, src - group, lo, hi - lo)
            return carry

        jax.lax.fori_loop(0, counts_ref[2], place_small, 0)

    @pl.when(tile < counts_ref[1])
    def _select():
        seq = tile_seq_ref[tile]
        row0 = tile_row0_ref[tile]
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, block_q), 1)
        # Folded rows are token-major: row = token * heads + head.
        tok = (row0 + lane - seq_row0_ref[seq]) >> heads_shift
        num_visible = (first_pos_ref[seq] + tok) // block_size + 1
        num_chunks = live_chunks(tile)
        sub = jax.lax.broadcasted_iota(jnp.int32, (8, block_q), 0)
        copy_tile(tile, slot, "wait")

        def accumulate(best, s, base):
            # Strict > over pages visited in increasing order keeps, per
            # sublane slot, the lowest page among equal scores.
            m, page = best
            for k in range(chunk // 8):
                v = s[k * 8 : (k + 1) * 8]
                gt = v > m
                m = jnp.where(gt, v, m)
                page = jnp.where(gt, sub + (base + k * 8), page)
            return m, page

        def first_pass(c, best):
            base = pl.multiple_of(c * chunk, chunk)
            s = scores_ref[slot, pl.ds(base, chunk), :]
            pages = jax.lax.broadcasted_iota(jnp.int32, (chunk, block_q), 0) + base
            blocks = pages * dcp + rank
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
            masked_ref[pl.ds(base, chunk), :] = s
            return accumulate(best, s, base)

        def later_pass(picked):
            def body(c, best):
                base = pl.multiple_of(c * chunk, chunk)
                s = masked_ref[pl.ds(base, chunk), :]
                pages = jax.lax.broadcasted_iota(jnp.int32, (chunk, block_q), 0) + base
                s = jnp.where(pages == picked, _NEG_INF, s)
                masked_ref[pl.ds(base, chunk), :] = s
                return accumulate(best, s, base)

            return body

        init = (
            jnp.full((8, block_q), _NEG_INF, jnp.float32),
            jnp.full((8, block_q), _NO_ROW, jnp.int32),
        )
        body = first_pass
        # Pick i is sublane i % 8 of group i // 8.
        groups = -(-topk // 8)
        score_bits = [jnp.full((8, block_q), _NEG_INF_BITS, jnp.int32)] * groups
        picked_blocks = [jnp.full((8, block_q), -1, jnp.int32)] * groups
        for i in range(topk):
            m, page = jax.lax.fori_loop(0, num_chunks, body, init)
            best = jnp.max(m, axis=0, keepdims=True)
            picked = jnp.min(jnp.where(m == best, page, _NO_ROW), axis=0, keepdims=True)
            g, here = i // 8, sub == i % 8
            score_bits[g] = jnp.where(
                here, jax.lax.bitcast_convert_type(best, jnp.int32), score_bits[g]
            )
            picked_blocks[g] = jnp.where(
                here, jnp.where(best > _NEG_INF, picked * dcp + rank, -1), picked_blocks[g]
            )
            body = later_pass(picked)
        picks = jnp.concatenate(
            [
                jnp.concatenate(score_bits, axis=0)[:topk],
                jnp.concatenate(picked_blocks, axis=0)[:topk],
            ],
            axis=0,
        )
        lo, hi = _tile_rows(row0, block_q, seq_row0_ref, seq)
        _place(out_ref, picks, lo - row0, lo, hi - lo)


def _tile_select(
    large: "msa_indexer.ClassScores",
    small: "msa_indexer.ClassScores",
    small_slots: jax.Array,
    seq_row0: jax.Array,
    first_pos: jax.Array,
    seq_pages: jax.Array,
    rank: jax.Array,
    *,
    num_rows: int,
    large_q: int,
    small_q: int,
    heads: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    dcp: int,
    interpret: bool,
) -> jax.Array:
    """`_tile_select_kernel` over the live tiles of the large (transposed)
    class ``large.scores [P, tiles * large_q]``: the slots ``[2 * topk,
    num_rows]`` of every folded row, the small class's ``small_slots [2 *
    topk, small columns]`` (column ``t * small_q + i``: row ``i`` of small
    tile ``t``) moved into place."""
    num_pages = large.scores.shape[0]
    # Each small tile's columns sit in one aligned group of `small_width`.
    small_width = -(-small_q // 128) * 128
    if (128 % small_q) if small_q < 128 else (small_q % 128):
        raise ValueError(f"small tiles of {small_q} rows do not tile 128 lanes")
    # Lane-aligned and wide enough for one placement window.
    cols = max(-(-num_rows // 128) * 128, large_q + 128)
    small_slots = jnp.pad(small_slots, ((0, 0), (0, -small_slots.shape[1] % 128)))
    kernel = functools.partial(
        _tile_select_kernel,
        heads_shift=heads.bit_length() - 1,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        dcp=dcp,
        chunk=_select_chunk(num_pages),
        small_q=small_q,
        small_width=small_width,
    )
    counts = jnp.stack([rank.reshape(()), large.num_tiles, small.num_tiles]).astype(
        jnp.int32
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=8,
            # Live tiles only: the class pads to its worst-case tile count.
            grid=(jnp.maximum(large.num_tiles, 1),),
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(small_slots.shape, lambda t, *_: (0, 0)),
            ],
            out_specs=pl.BlockSpec((2 * topk, cols), lambda t, *_: (0, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, num_pages, large_q), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM((num_pages, large_q), jnp.float32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((2 * topk, cols), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=msa_select._VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="minimax_m3_msa_tile_select",
    )(
        large.tile_seq,
        jnp.clip(large.tile_row0, 0, None),
        small.tile_seq,
        jnp.clip(small.tile_row0, 0, None),
        seq_row0,
        first_pos,
        seq_pages,
        counts,
        large.scores,
        small_slots,
    )
    return out if cols == num_rows else out[:, :num_rows]


def _tile_candidates(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    local_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    rank: jax.Array,
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    dcp: int,
    interpret: bool,
    scale_cache: jax.Array | None,
    scale_row: int | None,
) -> jax.Array:
    """`to_slots` of `_class_candidates` of a prefill step, bit for bit on
    every real token, with less glue: the ``G`` index heads fold token-major
    (row ``t * G + h``, a reshape; `msa_indexer` scores any row order of a
    request's rows alike), so folded row ``t * G + h`` is slot column ``t *
    G + h``. The small class picks with `local_candidates`; one
    `_tile_select_kernel` call picks the large class per tile and writes
    both classes' picks straight into the ``[2 * topk, T * G]`` slots (no
    packed buffer, transpose or column gather). Padding tokens get no
    candidate."""
    num_tokens, num_heads, head_dim = index_q.shape
    num_rows = num_tokens * num_heads
    folded_cu = num_heads * cu_q_lens
    plan = msa_indexer.class_plan(
        index_q.reshape(num_rows, 1, head_dim),
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
    heads_shift = num_heads.bit_length() - 1
    first_pos = seq_lens - (cu_q_lens[1:] - cu_q_lens[:-1])
    seq_row0 = folded_cu[:-1]
    # Small class (decode rows of the step): `local_candidates` in the layout
    # it scored in, as `_class_candidates`.
    small, block_q = plan.small(), plan.small_block_q
    per_tile = functools.partial(_per_tile, small.tile_seq)
    row = small.tile_row0[:, None] + jnp.arange(block_q, dtype=jnp.int32)[None, :]
    col_pos = per_tile(first_pos)[:, None] + (
        (row - per_tile(seq_row0)[:, None]) >> heads_shift
    )
    s, b = local_candidates(
        small.scores[:, None, :] if plan.token_major else small.scores,
        col_pos.reshape(-1),
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        rank=rank,
        dcp=dcp,
        live_rows=small.num_tiles * block_q,
        interpret=interpret,
    )
    small_slots = jnp.concatenate(
        [jax.lax.bitcast_convert_type(s, jnp.int32).T, b.T], axis=0
    )
    page_size = msa_indexer._page_tokens(index_cache.shape[1], index_cache.dtype)
    return _tile_select(
        plan.large(),
        small,
        small_slots,
        folded_cu,
        first_pos,
        -(-local_lens // page_size),
        rank_array(rank),
        num_rows=num_rows,
        large_q=plan.large_block_q,
        small_q=block_q,
        heads=num_heads,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        dcp=dcp,
        interpret=interpret,
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
    index_cache, candidates [2 * topk, T * G] int32)``: `to_slots`.
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
        kv_cache, index_cache = msa.write_kv_and_index(
            kv_cache,
            index_cache,
            jnp.concatenate([k, v], axis=-1).reshape(num_tokens, -1),
            index_k,
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
        live_select=LIVE_SELECT_ENABLED,
        interpret=interpret,
    )
    if tile_select(num_tokens, num_heads):
        packed = _tile_candidates(
            index_q,
            index_cache,
            seq_lens,
            local_lens,
            block_tables,
            cu_q_lens,
            distribution[2],
            rank,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            dcp=dcp,
            interpret=interpret,
            scale_cache=scale_cache,
            scale_row=scale_row,
        )
        if fp4:
            return kv_cache, index_cache, scale_cache, packed
        return kv_cache, index_cache, packed
    if msa_indexer.is_prefill_step(num_heads * num_tokens, max_num_seqs):
        packed = to_slots(_class_candidates(
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
        ), topk)
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
    packed = to_slots(packed.reshape(num_heads, num_tokens, -1).transpose(1, 0, 2), topk)
    if fp4:
        return kv_cache, index_cache, scale_cache, packed
    return kv_cache, index_cache, packed


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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Second half of the DCP forward: attend over the selected blocks this
    rank holds.

    With a 4-bit `kv_cache` pass its `scale_cache`; `q` must then be rotated
    (`msa_fp4.rotate_queries`) and the partial output is in the rotated
    basis (callers rotate their own queries before the gather and unrotate
    their own merged heads, a 1 / DCP share of the group's each).

    ``q [T, G * Hg, D]`` holds every query head of the group's ``G`` KV heads
    (KV head ``h`` owns query heads ``[h * Hg, (h + 1) * Hg)``);
    ``candidates [dcp, 2 * topk, T * G]`` every rank's `to_slots`. Returns
    the partial head-major, as the kernel writes it, ``(out [G * Hg, T, D],
    lse [G * Hg, T] f32)``: the LSE merge reduce-scatters heads along the
    leading axis (`ops.msa.merge_head_partials`).
    """
    num_tokens = q.shape[0]
    rank = rank_array(rank)[0]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    pages_per_seq = block_tables_2d.shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    selected = merge_head_candidates(
        candidates, topk, candidates.shape[2] // num_tokens, interpret
    )  # [T, G, topk]
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
    common = dict(
        sm_scale=sm_scale,
        block_q=(
            msa_attention.SMALL_BATCH_BLOCK_Q
            if num_tokens <= DCP_SMALL_BATCH_TOKENS
            else msa_attention.DEFAULT_BLOCK_Q
        ),
        pages_per_step=msa_attention.DEFAULT_PAGES_PER_STEP,
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
