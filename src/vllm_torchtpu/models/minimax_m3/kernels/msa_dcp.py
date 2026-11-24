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

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import tight_vmem
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
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """This rank's top-k ``(scores [T, topk] f32, blocks [T, topk] i32)``.

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
    picked_scores, picked_blocks = tight_vmem.pallas_call(
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
    if envs.TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT:
        # Same selection with the candidates on the reduced (minor) axis:
        # [rows, topk, dcp * topk] -> max over lanes. XLA lays the
        # [rows, dcp * topk, topk] form out badly at some row counts (32
        # rows: 3.5 us vs 0.4 us at 24); the values are identical.
        slot = jnp.arange(topk, dtype=jnp.int32)[None, :, None]
        return jnp.max(jnp.where(place[:, None, :] == slot, b[:, None, :], -1), axis=2)
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
    block_size: int | None = None,
    rank: jax.Array | int = 0,
    dcp: int = 1,
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
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    options = msa.block_score_options(num_heads)
    opt = msa_indexer.prefill_opt()

    def make_plan(large_bq, pages_per_step, max_requests):
        def tile_lens(tile_seq, tile_row0):
            """This rank's length (`local_seq_lens`) up to each large tile's
            last visible token: its rows are tokens of one request, head-major."""
            per_tile = functools.partial(msa_indexer.lookup, index=tile_seq)
            fc = per_tile(folded_cu[:-1])
            q_len = jnp.maximum(per_tile(q_lens), 1)
            first = jnp.maximum(tile_row0 - fc, 0)
            last = jnp.minimum(tile_row0 + large_bq, fc + num_heads * q_len) - 1 - fc
            last = jnp.maximum(last, first)
            top = jnp.where(first // q_len == last // q_len, last % q_len, q_len - 1)
            visible = per_tile(seq_lens) - q_len + top + 1
            return local_seq_lens(visible, block_size=block_size, rank=rank, dcp=dcp)

        return msa_indexer.class_plan(
            folded,
            index_cache,
            local_lens,
            block_tables,
            folded_cu,
            num_seqs,
            **dict(options, large_block_q=large_bq, pages_per_step=pages_per_step),
            scale_cache=scale_cache,
            scale_row=scale_row,
            interpret=interpret,
            fp8_q=msa_indexer.fp8_queries_enabled(),
            large_tile_lens_fn=tile_lens if block_size is not None else None,
            large_max_requests=max_requests,
        )

    plan = make_plan(options["large_block_q"], options["pages_per_step"], 0)
    max_num_seqs = q_lens.shape[0]
    seq_of_token = jnp.minimum(seq_id, max_num_seqs - 1)

    def class_picks(scored, block_q, transposed):
        """A class's packed picks ``[columns, 2 * topk]`` and each head's and
        token's column in them ``[G, T]``."""
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
        picks = pack_candidates(s, b)
        # Each head's and token's column in this class (`_class_scores`).
        start = msa_indexer.lookup(scored.tile_start, seq_of_token)[None]
        off = rows - start
        col = (
            msa_indexer.lookup(scored.first_tile, seq_of_token)[None] + off // block_q
        ) * block_q
        return picks, jnp.clip(col + off % block_q, 0, picks.shape[0] - 1)

    small_picks, small_col = class_picks(
        plan.small(), plan.small_block_q, not plan.token_major
    )
    is_large_tok = msa_indexer.lookup(plan.is_large, seq_of_token)[None]
    if opt.cond:
        # Taller tiles (and their pages per step) when at most `opt.cond`
        # requests are large; the default tiles otherwise. Two branches only:
        # a wider switch stops the layers sharing the attention code.
        fast = make_plan(
            opt.block_q or 2 * msa_indexer.DEFAULT_LARGE_BLOCK_Q,
            opt.pps or options["pages_per_step"],
            opt.cond,
        )

        # The branches return the picks (padded to the default branch's
        # columns) and each row's column; the gather back to token order
        # stays outside, as without the option: inside, the conditional
        # waits for the gather (offloaded to the SparseCore), +60-75 us per
        # layer at 8K tokens (bench j-84225507).
        rows_base = plan.large().scores.shape[1]

        def large_picks(p):
            picks, col = class_picks(p.large(), p.large_block_q, True)
            return jnp.pad(picks, ((0, rows_base - picks.shape[0]), (0, 0))), col

        large_p, large_col = jax.lax.cond(
            jnp.sum(plan.is_large.astype(jnp.int32)) <= opt.cond,
            lambda: large_picks(fast),
            lambda: large_picks(plan),
        )
        col = jnp.where(is_large_tok, small_picks.shape[0] + large_col, small_col)
        packed = jnp.concatenate([small_picks, large_p], axis=0)[col.reshape(-1)]
    else:
        large_picks, large_col = class_picks(plan.large(), plan.large_block_q, True)
        # A head's and token's picks: its large-class column if its request
        # is large, else its small-class one.
        col = jnp.where(is_large_tok, small_picks.shape[0] + large_col, small_col)
        packed = jnp.concatenate([small_picks, large_picks], axis=0)[
            col.reshape(-1)
        ]  # [G * T, 2 * topk]
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
            block_size=block_size,
            rank=rank,
            dcp=dcp,
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


def _fp8_opt() -> str:
    """TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT (read when traced)."""
    from vllm_torchtpu import envs

    return envs.TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT


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
    leading axis (`ops.msa.merge_head_partials`), or the merge's all-to-all
    sends each rank its block (`attend_and_merge`).
    """
    num_tokens = q.shape[1 if q_head_major else 0]
    rank = rank_array(rank)[0]
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    pages_per_seq = block_tables_2d.shape[1]
    seq_id, pos, valid = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    selected = merge_head_candidates(candidates, topk, interpret)  # [T, G, topk]
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
    small = num_tokens <= DCP_SMALL_BATCH_TOKENS
    decode_opt = {}
    if small:
        block_q = msa_attention.SMALL_BATCH_BLOCK_Q
        pages_per_step = (
            _SMALL_BATCH_PAGES_PER_STEP
            if num_tokens >= _WIDE_GROUP_MIN_TOKENS
            else msa_attention.DEFAULT_PAGES_PER_STEP
        )
        decode_opt = (
            msa_attention_lse.decode_opt()
            if num_tokens >= _WIDE_GROUP_MIN_TOKENS
            else {}
        )
        pages_per_step = decode_opt.get("pages_per_step", pages_per_step)
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
    if small and "step_widths" in decode_opt:
        common["step_widths"] = decode_opt["step_widths"]
    if num_tokens >= _MIN_CANDIDATE_TOKENS:  # prefill-sized steps only
        common["step_widths"] = msa_attention_lse.PREFILL_STEP_WIDTHS
        common["row_split"] = msa_attention_lse.PREFILL_ROW_SPLIT
        if msa_attention_lse.PREFILL_BLOCK_Q:
            common["block_q"] = msa_attention_lse.PREFILL_BLOCK_Q
        if msa_attention_lse.PREFILL_PAGES_PER_STEP:
            common["pages_per_step"] = msa_attention_lse.PREFILL_PAGES_PER_STEP
        if scale_cache is not None:
            common["page_cache_slots"] = msa_attention_lse.PREFILL_PAGE_CACHE_SLOTS
            common["page_cache_bf16"] = msa_attention_lse.PREFILL_PAGE_CACHE_BF16
        elif _fp8_opt():
            common["prefill_precision"] = _fp8_opt()
    args = (local_key, pos % block_size + 1, block_tables.reshape(-1))
    adaptive = (
        msa_attention_lse.adaptive_prefill()
        if scale_cache is not None
        and num_tokens >= _MIN_CANDIDATE_TOKENS
        and common.get("page_cache_slots")
        else None
    )
    if adaptive is not None:
        # Two kernel configurations in the program; each call runs one, by
        # how wide its selections are (the other's outputs are not computed).
        threshold, wide_kw = adaptive
        msa_attention_lse._proof(
            msa_attention_lse.ADAPTIVE_ENV,
            f"MSA prefill tiles per call: > {threshold} pages per "
            f"{common['block_q']}-token tile -> block_q {wide_kw['block_q']}, "
            f"tile pages {wide_kw['page_cache_tile_pages']}"
            + (
                f", pages per step {wide_kw['pages_per_step']}"
                if "pages_per_step" in wide_kw
                else ""
            )
            + (
                f", widths {wide_kw['step_widths']}"
                if "step_widths" in wide_kw
                else ""
            ),
        )
        wide = dict(common, **wide_kw)

        def attend(kw):
            def run(operands):
                q_, kv_, keys_, sc_, *args_ = operands
                if num_heads == 1:
                    return msa_attention_lse.msa_sparse_attention_lse(
                        q_, kv_, keys_[:, 0], *args_, head=0, head_major=True,
                        **kw, **msa._fp4_attention_kwargs(sc_),
                    )
                return msa_attention_lse.msa_sparse_attention_lse_heads(
                    q_, kv_, keys_, *args_, head_major=True, **kw, scale_cache=sc_
                )

            return run

        return jax.lax.cond(
            msa_attention_lse.wide_selection(keys, common["block_q"], threshold),
            attend(wide),
            attend(common),
            (q, kv_cache, keys, scale_cache, *args),
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


def gather_by_permute(x: jax.Array, rank: jax.Array, axis: str, dcp: int) -> jax.Array:
    """Every rank's `x` in rank order, ``[dcp, *x.shape]``, by ``dcp - 1``
    collective permutes (rank ``r`` gets rank ``r + s``'s `x` from shift
    ``s``). For decode-sized operands of a DCP pair (two cores of one
    chip): each XLA all-gather there blocks the TensorCore for its latency
    (3.2 us per layer at c4 b3, j-c71e5160). Across chips (DCP 8) the
    permutes' starts cost more than the all-gather (c64 twin j-f92f5540)."""
    out = jnp.zeros((dcp, *x.shape), x.dtype)
    out = jax.lax.dynamic_update_index_in_dim(out, x, rank, 0)
    for shift in range(1, dcp):
        perm = [(src, (src - shift) % dcp) for src in range(dcp)]
        received = jax.lax.ppermute(x, axis, perm)
        out = jax.lax.dynamic_update_index_in_dim(
            out, received, (rank + shift) % dcp, 0
        )
    return out


def all_to_all_by_permute(
    x: jax.Array, rank: jax.Array, axis: str, dcp: int
) -> jax.Array:
    """`jax.lax.all_to_all` of ``x [dcp, ...]`` along its leading axis (block
    ``d`` goes to rank ``d``; block ``s`` of the result came from rank
    ``s``), by `gather_by_permute`'s permutes. For decode-sized blocks: each
    XLA all-to-all costs its latency on the TensorCore (~1.3 us per layer at
    c4 decode, j-c71e5160)."""
    out = jnp.zeros_like(x)
    own = jax.lax.dynamic_index_in_dim(x, rank, 0, False)
    out = jax.lax.dynamic_update_index_in_dim(out, own, rank, 0)
    for shift in range(1, dcp):
        perm = [(src, (src - shift) % dcp) for src in range(dcp)]
        send = jax.lax.dynamic_index_in_dim(x, (rank - shift) % dcp, 0, False)
        received = jax.lax.ppermute(send, axis, perm)
        out = jax.lax.dynamic_update_index_in_dim(
            out, received, (rank + shift) % dcp, 0
        )
    return out


def _exchange_partials(
    parts: jax.Array, lses: jax.Array, rank: jax.Array, axis: str, dcp: int
) -> tuple[jax.Array, jax.Array]:
    """`all_to_all_by_permute` of ``parts [dcp, Hl, T, D]`` and ``lses [dcp,
    T, Hl]`` f32 in one set of permutes: the LSEs' bits ride as one more uint16
    head row (a permute copies bytes; every permute start costs ~1 us on the
    TensorCore at decode, c4 twin j-6bbe97e2)."""
    num_ranks, num_heads, num_tokens, head_dim = parts.shape
    if parts.dtype.itemsize != 2 or 2 * num_heads > head_dim:
        raise ValueError(
            f"16-bit partials with 2 * heads <= head_dim, got {parts.dtype} "
            f"{parts.shape}"
        )
    # As 16-bit integers: float copies may canonicalize NaN or flush
    # subnormal bit patterns (the LSE halves are arbitrary bits).
    words = jax.lax.bitcast_convert_type(parts, jnp.uint16)
    bits = jax.lax.bitcast_convert_type(lses, jnp.uint16)  # [dcp, T, Hl, 2]
    bits = bits.reshape(num_ranks, 1, num_tokens, 2 * num_heads)
    bits = jnp.pad(bits, ((0, 0), (0, 0), (0, 0), (0, head_dim - 2 * num_heads)))
    both = all_to_all_by_permute(
        jnp.concatenate([words, bits], axis=1), rank, axis, dcp
    )
    bits = both[:, num_heads, :, : 2 * num_heads].reshape(
        num_ranks, num_tokens, num_heads, 2
    )
    return (
        jax.lax.bitcast_convert_type(both[:, :num_heads], parts.dtype),
        jax.lax.bitcast_convert_type(bits, lses.dtype),
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
    its LSEs go to their owner by all-to-all for `merge_partials`. As XLA
    collectives in one program they move the head-major buffers as laid out;
    torch's all-to-all of them cost ~890 us per layer at 8K tokens (c4 twin,
    j-c60806bf)."""
    num_heads, num_tokens, head_dim = q.shape
    rank = rank_array(rank)[0]
    q_all = _gather_heads(q, rank, axis, dcp)
    # Decode-sized operands of a pair move by permutes (latency); a prefill
    # step's partials by XLA's all-to-all, which writes them in place.
    small = dcp == 2 and num_tokens <= DCP_SMALL_BATCH_TOKENS
    if small:
        cand_all = gather_by_permute(candidates, rank, axis, dcp)
    else:
        cand_all = jax.lax.all_gather(candidates, axis, axis=0)
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
    # Block r of each result came from rank r: its partial of this rank's heads.
    parts = out.reshape(dcp, num_heads, num_tokens, head_dim)
    lses = lse.reshape(dcp, num_heads, num_tokens).transpose(0, 2, 1)
    if small:
        parts, lses = _exchange_partials(parts, lses, rank, axis, dcp)
    else:
        parts = jax.lax.all_to_all(parts, axis, 0, 0, tiled=True)
        lses = jax.lax.all_to_all(lses, axis, 0, 0, tiled=True)
    return merge_partials(parts, lses, hadamard, interpret=interpret)
