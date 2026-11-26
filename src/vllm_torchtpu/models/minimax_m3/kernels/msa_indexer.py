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
"""Pallas lightning-indexer block scores for MiniMax Sparse Attention.

For every query token and every KV page of its request, the score of the
page is the max over its tokens of ``index_q . index_k`` (no scale: only the
order matters). Each request's tokens are cut into query tiles of
`block_q` rows; a grid step scores one tile against `pages_per_step` pages
read straight from the paged index cache, so a request's index keys are read
once per tile. The cache packs consecutive tokens into 32-bit words
(`packed_page`); each packing slot is scored separately, which a max does not
mind.

Long (prefill) requests' scores are produced transposed,
``[pages, tiles * block_q]``, so each output block is lane-aligned, and
gathered back to ``[tokens, pages]``; the short (decode) requests' 128-page
groups are lane-aligned either way and come out token-major.
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4
from vllm_torchtpu.models.minimax_m3.kernels.packed_page import (
    check_page_dtype,
    unpack_page,
)

DEFAULT_BLOCK_Q = 128
DEFAULT_LARGE_BLOCK_Q = 512
# Pages per grid step of the large (prefill) tiles: a step's fixed cost
# against its 16-page dots (bench, us per call, 32 / 64 / 128 pages: c64
# mixed 372 / 357 / 345, c48 281 / 266 / 262, c4 123 / 105 / 95, c16 161 /
# 142 / 135; bit-identical). The kernel grows with it (AOT c48: 6.6k, 18.6k,
# 41.8k bundles; one straight-line body per ring slot and chunk count).
DEFAULT_PAGES_PER_STEP = 128
# The block_q class (decode, MTP verify) runs the v3 DMA ring; 128 pages per
# step keep more bytes in flight (j-dba77583: 32 req x 8 tokens at 128K,
# 1.95 -> 1.66 ms).
DEFAULT_SMALL_PAGES_PER_STEP = 128
DEFAULT_KERNEL = "v3"
# Rows per token-major query tile of the block_q class (see `block_scores`).
_SMALL_TILE_Q = 64
# Folded 4-bit and fp8 pages are scored transposed (`_chunk_maxima_t`), whose
# cost grows with the tile's rows: by default 32 holds a decode request of 4
# folded index heads x 8 (MTP) tokens; callers with one index head per rank
# pass `transposed_tile_q=8` (`msa.block_score_options`).
_SMALL_TILE_Q_TRANSPOSED = 32
# Large prefill tiles: v3's DMA ring, as the small class. v2's 32 BlockSpec
# page inputs cost ~1,900 scalar bundles per 32-page step beside ~2,500 of
# vector work (AOT); v3 is bit-identical and -17..-24% per indexer call at
# the c48 / c64 DCP 2 / c4 mixed-step shapes (bench j-5eb4a611).
DEFAULT_LARGE_KERNEL = "v3"
_TILE_SHIFT = 16
# tile_info = query row of the tile << _ROW_SHIFT | request of the tile
_ROW_SHIFT = 10
_SEQ_MASK = (1 << _ROW_SHIFT) - 1
# f32 query rows are read at element offsets that are multiples of the sublane tile.
_ROW_ALIGN = 8
_GROUP_MASK = (1 << _TILE_SHIFT) - 1


def _kernel_v1(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    # In
    q_ref,  # [Bq, D]
    *refs,  # G x [1, S / p, p, D] index-cache pages, then the output
):
    pages_per_step = len(refs) - 1
    page_refs, o_ref = refs[:pages_per_step], refs[pages_per_step]
    q = q_ref[...].astype(jnp.bfloat16)
    maxima = []
    for page_ref in page_refs:
        page_max = None
        for keys in unpack_page(page_ref):
            s = jax.lax.dot_general(
                keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            m = jnp.max(s, axis=0, keepdims=True)
            page_max = m if page_max is None else jnp.maximum(page_max, m)
        maxima.append(page_max)
    o_ref[...] = jnp.concatenate(maxima, axis=0)


# Scores per dot in `_kernel_v2`: bounds the f32 ``[pages * S, block_q]``
# product kept in VMEM (2 MiB).
_MAX_SCORE_ELEMS = 512 * 1024


def _page_keys(page_ref) -> jax.Array:
    """``[1, S / p, p, D]`` page -> bf16 ``[S, D]`` keys in packing order.

    One width-changing bitcast of the packed 32-bit words: the token order
    inside the page changes, which a max over the page does not see.
    """
    words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
    return _words_keys(words, page_ref.dtype)


# Folded 4-bit pages: scale the keys (``[S, D]`` bf16, before the dot) rather
# than the f32 scores (``[S, Bq]``, after it): fewer elements per page, the
# same per-key scale (rounded to bf16 with the key, far finer than fp8 keys).
SCALE_KEYS = True


def _folded_keys(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A 4-bit page with its scales folded in (`msa_fp4.scales_in_page`):
    ``[S / 8 + 1, D]`` words -> (unscaled bf16 keys ``[S, D]``, the per-slot
    scales of the last word row as a ``[1, S]`` f32 lane vector)."""
    keys = pltpu.bitcast(words[:-1], jnp.int4).astype(jnp.bfloat16)
    return keys, pltpu.bitcast(words[-1:], jnp.float32)


def _scale_rows_to_column(rows: list[jax.Array]) -> jax.Array:
    """The group's ``[1, S]`` slot-scale rows -> one ``[n * S, 1]`` column in
    the score rows' (page-major) order, non-finite read as 0. One transpose
    of the stacked ``[n, S]`` rows per dot group: a ``[1, S] -> [S, 1]``
    transpose per page made the 4-bit indexer 50% slower than fp8."""
    stacked = rows[0] if len(rows) == 1 else jnp.concatenate(rows, axis=0)  # [n, S]
    stacked = jnp.where(jnp.isfinite(stacked), stacked, 0.0)
    cols = stacked.T  # [S, n]
    return jnp.concatenate([cols[:, g : g + 1] for g in range(len(rows))], axis=0)


def _words_keys(words: jax.Array, dtype) -> jax.Array:
    """Packed ``[S / p, D]`` words -> bf16 ``[S, D]`` keys. A 4-bit (uint8)
    cache's int4 values come out unscaled, in slot order (`msa_fp4`)."""
    if dtype == jnp.uint8:
        return pltpu.bitcast(words, jnp.int4).astype(jnp.bfloat16)
    return pltpu.bitcast(words, dtype).astype(jnp.bfloat16)


def _scale_column(scale_page: jax.Array, row: int) -> jax.Array:
    """``[8, 128]`` scale page -> ``[128, 1]`` scales of its slots for scale-page
    row `row`. An unwritten slot's scale may be any bits; non-finite reads as 0."""
    col = scale_page.T[:, row : row + 1]
    return jnp.where(jnp.isfinite(col), col, 0.0)


def _kernel_v2(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    # In
    q_ref,  # [Bq, D]
    *refs,  # G x [1, S / p, p, D] index-cache pages, [G x [1, 8, 128] scale pages], out
    scale_row: int | None = None,
):
    """`_kernel_v1` with the pages' keys unpacked by one bitcast each and
    stacked, so each dot keeps the query tile latched across many pages.

    With `scale_row` (4-bit caches) each page is followed, after all pages,
    by its scale page, and a key's score is scaled before the block max."""
    num_inputs = len(refs) - 1
    pages_per_step = num_inputs if scale_row is None else num_inputs // 2
    page_refs, o_ref = refs[:pages_per_step], refs[num_inputs]
    scale_refs = refs[pages_per_step:num_inputs]
    q = q_ref[...].astype(jnp.bfloat16)
    block_q = q.shape[0]
    page_size = _page_tokens(page_refs[0].shape[1], page_refs[0].dtype)
    pages_per_dot = max(
        1, min(pages_per_step, _MAX_SCORE_ELEMS // (page_size * block_q))
    )
    maxima = []
    folded = msa_fp4.scales_in_page(page_refs[0].shape[1], page_refs[0].dtype)
    for first in range(0, pages_per_step, pages_per_dot):
        group = page_refs[first : first + pages_per_dot]
        if folded:
            pairs = [_folded_keys(r.bitcast(jnp.uint32)[0, :, 0, :]) for r in group]
            keys = [k for k, _ in pairs]
        else:
            keys = [_page_keys(r) for r in group]
        keys = keys[0] if len(keys) == 1 else jnp.concatenate(keys, axis=0)
        col = _scale_rows_to_column([c for _, c in pairs]) if folded else None
        if folded and SCALE_KEYS:
            keys = keys * col.astype(jnp.bfloat16)
        s = jax.lax.dot_general(
            keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        if folded and not SCALE_KEYS:
            s = s * col
        elif scale_row is not None:
            cols = [
                _scale_column(r[0], scale_row)
                for r in scale_refs[first : first + pages_per_dot]
            ]
            s = s * (cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=0))
        maxima.append(jnp.max(s.reshape(len(group), page_size, block_q), axis=1))
    o_ref[...] = maxima[0] if len(maxima) == 1 else jnp.concatenate(maxima, axis=0)


def _page_tokens(rows_per_page: int, dtype) -> int:
    """Tokens per page: 4-bit (uint8) caches hold 8 per 32-bit word (a folded
    page's extra word row holds scales, not tokens)."""
    if dtype == jnp.uint8:
        return msa_fp4.data_rows(rows_per_page) * msa_fp4.TOKENS_PER_WORD
    return rows_per_page * (4 // jnp.dtype(dtype).itemsize)


# Pages per dot in `_kernel_v3`: 2 x 64 pages, paired along the contraction
# (as `_v3_chunk` bounds it; prefill tiles stay at 16). Decode calls ran
# faster in larger dots, standalone kernel (bench j-3fb5ff45): c64 b46
# 82.0 -> 77.7 us, c48 b22 69.9 -> 68.2, c16 b5 18.9 -> 18.5, c4 b1 flat,
# against 32-page dots.
_PAGES_PER_CHUNK = 128
# `_kernel_v3` issues the ahead group's descriptors during its first chunks:
# all of a 128- or 256-page group's 32-page chunks (2 chunks of 128-page
# groups: standalone decode calls, int4 folded 23 x 450 pages 101 -> 92 us,
# fp8 5 x 450 pages 25.5 -> 24.1, j-0a884760; of 256-page ones 4 -> 8: 87.7
# -> 77.9 and 22.7 -> 21.5, j-83e37332). An earlier sweep found all chunks
# slower (j-2e0b290e), before the copies took one scalar address each.
# Measured with 32-page chunks; with `_PAGES_PER_CHUNK` 128 a 128-page group
# is one chunk and a 256-page group two, which then carry all of it (the
# 128-page dots measured faster with this spreading collapsed, j-3fb5ff45).
_DMA_ISSUE_CHUNKS = 8
# Partial last chunks (pages) the transposed tiles score in `_kernel_v3`, so
# that a 256-page group with 129-208 live pages skips its last 48 pages and
# one with 209-224 its last 32. Each size adds a body per ring slot, paid in
# code and a branch per step (with one body per slot and count, (64, 96) was
# slower than (96,) at every decode shape, bench j-3bcdf4d5; with the slot
# dispatched first, b46 decode at 196 pages per request: (96,) 63.7, (80, 96)
# 60.6, (64, 80, 96) 60.7 us; ragged 100-256 pages 61.6, 60.8, 60.5 us;
# anvil sweeps 20261126-175749-14ef, 20261126-180005-0367).
_TAIL_PAGES = (80, 96)
assert all(t % 2 == 0 for t in _TAIL_PAGES), "tails pair their pages"
# Page groups in `_kernel_v3`'s DMA ring.
_RING_SLOTS = 3
# `_kernel_v3`'s block-table windows start at multiples of this many entries
# (the lane width), as HBM -> SMEM copies of a 1-D table need.
_TABLE_ALIGN = 128
# Block-table windows in SMEM: the ring's groups and the windows copied
# ahead of them.
_TABLE_SLOTS = _RING_SLOTS + 1
# Folded 4-bit and fp8 tiles of up to this many rows score transposed
# (`_chunk_maxima_t`, maxima across lanes); larger ones score token-major
# (maxima across sublanes), a folded page with its slot scales folded into
# the keys.
_TRANSPOSED_MAX_ROWS = 64


def _scores_transposed(block_q: int, rows_per_page: int, dtype) -> bool:
    """Whether `_kernel_v3` scores `block_q`-row tiles of these pages with
    `_chunk_maxima_t`."""
    fp8 = dtype == jnp.float8_e4m3fn and FP8_MXU_KEYS
    folded = msa_fp4.scales_in_page(rows_per_page, dtype)
    return (folded or fp8) and block_q <= _TRANSPOSED_MAX_ROWS


def _paired_query(q: jax.Array) -> jax.Array:
    """``[Bq, D]`` -> ``[[q, 0], [0, q]]`` of ``[2 Bq, 2 D]``."""
    zeros = jnp.zeros_like(q)
    return jnp.concatenate(
        [
            jnp.concatenate([q, zeros], axis=1),
            jnp.concatenate([zeros, q], axis=1),
        ],
        axis=0,
    )


def _chunk_maxima(q_pair, page_keys, num_pages) -> jax.Array:
    """``[num_pages, Bq]`` block maxima of `num_pages` pages against a query
    tile given as `_paired_query`.

    Pages ``j`` and ``h + j`` (``h = num_pages / 2``) share one MXU row,
    ``[k_j | k_h+j] @ q_pair^T``: both are scored with one contraction of
    ``2 D``, the MXU depth at D = 128. `page_keys(j)` is page ``j``'s bf16
    ``[S, D]`` keys.
    """
    half = num_pages // 2
    block_q = q_pair.shape[0] // 2
    keys = [
        jnp.concatenate([page_keys(j), page_keys(half + j)], axis=1)
        for j in range(half)
    ]
    page_size = keys[0].shape[0]
    s = jax.lax.dot_general(
        jnp.concatenate(keys, axis=0),
        q_pair,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [half * S, 2 Bq]
    m = jnp.max(s.reshape(half, page_size, 2 * block_q), axis=1)
    return jnp.concatenate([m[:, :block_q], m[:, block_q:]], axis=0)


# Keys enter the MXU as fp8 in the transposed path: fp8 caches as stored and
# 4-bit keys converted exactly (|k| <= 8), half the bytes of bf16 keys per
# transposed weight push, and no fp8 -> bf16 unpack (AOT, 128 pages at 8 rows:
# fp8 2,171 -> 819 bundles, 4-bit 1,323 -> 1,008). Products and their f32
# sums are those of bf16 keys.
FP8_MXU_KEYS = True
# The large tiles (prefill chunks) score fp8 queries against fp8 keys: the
# fp8 MXU rate, twice bf16's, on a body the bf16 x fp8 form left MXU-bound
# (AOT 87 %). The queries round to e4m3 for the block scores only (the top-k
# selection's input), and so do 4-bit keys times their scales: scaling the
# f32 scores instead took a product per score and spilled the MXU results
# (AOT c48 large-class kernel 6,619 -> 5,300 bundles; bench c64 mixed 372 ->
# 318 us per call, c48 281 -> 236).
FP8_MXU_QUERIES = True


def _transposed_keys(buf, p, live, dtype, folded, masked):
    """Page `p` of `buf` as ``[S, D]`` MXU keys: 4-bit keys unscaled, fp8 keys
    of a page past the request's end (`masked`) zeroed, as stale VMEM may hold
    NaN, which would reach the paired page's scores through the zero half of
    the paired query."""
    rows = buf.shape[1]
    key_dtype = jnp.float8_e4m3fn if FP8_MXU_KEYS else jnp.bfloat16
    if folded:
        k = pltpu.bitcast(buf[p, : rows - 1], jnp.int4)
    else:
        w = buf[p]
        k = pltpu.bitcast(jnp.where(p < live, w, 0) if masked else w, dtype)
    return k if k.dtype == key_dtype else k.astype(key_dtype)


def _chunk_scores_t(q_pair, buf, first, chunk, live, issue, *, dtype, folded):
    """The scores of `_chunk_maxima_t`, ``[2 Bq, chunk / 2 * S]`` (page ``j``
    of the chunk in rows ``[0, Bq)``, columns ``[j S, (j + 1) S)``, page
    ``chunk / 2 + j`` in rows ``[Bq, 2 Bq)``), and with folded pages their
    slot scales (``[1, S]`` each, unwritten slots 0) in page order, else
    None."""
    rows = buf.shape[1]
    half = chunk // 2
    keys, scale_rows = [], [[], []]
    for j in range(half):
        issue(j)
        pair = []
        for side, p in enumerate((first + j, first + half + j)):
            # Only the right page of a pair can be past the end while its
            # partner is not.
            pair.append(_transposed_keys(buf, p, live, dtype, folded, side == 1))
            if folded:
                s = pltpu.bitcast(buf[p, rows - 1 :], jnp.float32)  # [1, S]
                # Unwritten slots: scale 0. A page past the request's end
                # scales only its own column, which callers mask.
                scale_rows[side].append(jnp.where(jnp.isfinite(s), s, 0.0))
        keys.append(jnp.concatenate(pair, axis=1))
    s = jax.lax.dot_general(
        q_pair,
        jnp.concatenate(keys, axis=0),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [2 Bq, half * S]
    return s, (scale_rows[0] + scale_rows[1] if folded else None)


def _page_maxima(page_scores, scale, chunk):
    """``[Bq, chunk]``: page ``p``'s block maximum on lane ``p``, from its
    ``[Bq, S]`` `page_scores(p)` times its ``[1, S]`` slot scales `scale(p)`
    (if given). The maxima go to their lanes by selects, in a balanced tree: a
    reshape of the scores for one max took ~6 relayout ops per page, and a
    chain of selects through every page serialized them (bench j-08540f0e)."""
    maxima = []
    for p in range(chunk):
        page = page_scores(p)
        if scale is not None:
            page = page * scale(p)
        maxima.append(jnp.max(page, axis=1, keepdims=True))  # [Bq, 1]
    block_q = maxima[0].shape[0]
    lane = jax.lax.broadcasted_iota(jnp.int32, (block_q, chunk), 1)

    def merge(lo, hi):
        """The maxima of pages ``[lo, hi)``, right on those lanes."""
        if hi - lo == 1:
            return jnp.broadcast_to(maxima[lo], (block_q, chunk))
        mid = (lo + hi) // 2
        return jnp.where(lane < mid, merge(lo, mid), merge(mid, hi))

    return merge(0, chunk)


def _chunk_maxima_t(
    q_pair, buf, first, chunk, live, issue, *, dtype, folded
) -> jax.Array:
    """``[Bq, chunk]`` block maxima of the pages ``buf[first : first + chunk]``
    with the scores computed transposed, ``[2 Bq, tokens]``: the MXU result
    is ``2 Bq`` rows rather than ``chunk * S / 2``. A folded 4-bit page's
    per-token scales are a lane vector (the page's last word row), applied by
    a sublane broadcast. `issue(j)` runs among the keys of page pair ``j``.
    Columns of pages past the request's end hold garbage (callers mask them).
    """
    s, scales = _chunk_scores_t(
        q_pair, buf, first, chunk, live, issue, dtype=dtype, folded=folded
    )
    half = chunk // 2
    block_q = s.shape[0] // 2
    page_size = s.shape[1] // half

    def page_scores(p):
        side, j = divmod(p, half)
        return s[
            side * block_q : (side + 1) * block_q, j * page_size : (j + 1) * page_size
        ]

    return _page_maxima(
        page_scores, None if scales is None else (lambda p: scales[p]), chunk
    )


def _v3_chunk(pages_per_step: int, block_q: int, page_size: int) -> int:
    """Pages per dot of `_kernel_v3`: the f32 ``[chunk / 2 * S, 2 Bq]``
    product within twice `_MAX_SCORE_ELEMS`."""
    chunk = min(_PAGES_PER_CHUNK, pages_per_step)
    while chunk > 2 and chunk * page_size * block_q > 2 * _MAX_SCORE_ELEMS:
        chunk //= 2
    return chunk


def _kernel_v3(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    seq_pages_ref,  # [max_num_seqs] pages of each request
    # In
    q_ref,  # [Bq, D]
    cache_ref,  # [pages, S / p, p, D] index cache, left in HBM
    block_tables_ref,  # [max_num_seqs * pages_per_seq (+ pad)], left in HBM
    # Out
    o_ref,  # [G, Bq], or [Bq, G] if `transposed`
    # Scratch
    buf,  # [R, G, S / p, D] ring of R groups of pages, as 32-bit words
    sems,  # DMA semaphores [R], one per ring slot
    tables,  # SMEM [W * table_len]: block-table windows of W work items
    table_sems,  # DMA semaphores [W], one per window
    # With `defer`:
    #   out_buf [2, G / 128, Bq, 128] f32: two steps' maxima of their other chunks
    #   raw_buf [chunk, Bq, S] f32: a step's last chunk's scores, page-major
    #   meta SMEM [1]: their chunk's lane tile
    #   raw_scale [chunk, S] f32: their slot scales (folded pages only)
    *deferred,
    pages_per_seq,
    transposed=False,
    defer=False,
):
    """`_kernel_v2` with the pages fetched by a hand-written DMA ring of
    `_RING_SLOTS` groups: one scalar address per page and no copy for pages
    past the request's end (their scores are never read). A step issues the
    group `_RING_SLOTS - 1` steps ahead: with one group ahead, each step
    waited out its pages' DMA latency (~1.3 us per step at 64 or 128 pages,
    bench j-9956cd34).

    Each ring slot has straight-line bodies (static buffers, so the
    scheduler sees the ahead group's copies do not touch the pages being
    scored), and the ahead group's descriptors are issued among this
    group's matmuls. A wait between chunks would drain the MXU pipeline, so
    all of the group's pages are waited for first.

    A folded 4-bit page (`msa_fp4.scales_in_page`) is ``S / 8`` word rows of
    int4 keys and one row of f32 slot scales (`_chunk_maxima_t`).

    The block table stays in HBM: each work item's window of it is copied
    into SMEM a step before its pages are issued, so the call does not start
    with a copy of the whole table into SMEM.

    With `defer` (transposed tiles), a step's last chunk leaves its scores
    in VMEM, and the next step takes their maxima among its own matmuls and
    writes the finished block: step ``s`` writes work item ``s - 1``'s
    output block, and one more grid step writes the last item's. A step no
    longer ends on its last chunk's lane maxima.
    """
    slots, pages_per_step, rows, head_dim = buf.shape
    table_slots = table_sems.shape[0]
    table_len = tables.shape[0] // table_slots
    ahead = slots - 1
    block_q = q_ref.shape[0]
    folded = msa_fp4.scales_in_page(rows, cache_ref.dtype)
    page_size = _page_tokens(rows, cache_ref.dtype)
    chunk = _v3_chunk(pages_per_step, block_q, page_size)
    step, num_steps = pl.program_id(0), pl.num_programs(0)
    # Work items: the grid's steps, but for the extra step of `defer`.
    num_items = num_steps - 1 if defer else num_steps
    if defer:
        out_buf, raw_buf, meta, *raw_scale = deferred
        raw_scale = raw_scale[0] if folded else None
    # The pages as 32-bit words: a dense VMEM layout, so each key vreg is one
    # load (a packed [S / p, p, D] page tiles VMEM in (p, D) tiles).
    words = cache_ref.bitcast(jnp.uint32).reshape(cache_ref.shape[0], rows, head_dim)

    def group(w):
        """(first block-table entry, live pages) of work item `w`."""
        entry = work_ref[w]
        seq = tile_seq_ref[entry >> _TILE_SHIFT] & _SEQ_MASK
        first = (entry & _GROUP_MASK) * pages_per_step
        return seq * pages_per_seq + first, jnp.maximum(seq_pages_ref[seq] - first, 1)

    def table_start(base):
        """The first block-table entry of the window holding `base`."""
        return base // _TABLE_ALIGN * _TABLE_ALIGN

    def table_copy(k):
        """The copy of work item `k`'s block-table window into its slot."""
        base, _ = group(jnp.minimum(k, num_items - 1))
        slot = k % table_slots
        return pltpu.make_async_copy(
            block_tables_ref.at[
                pl.ds(pl.multiple_of(table_start(base), _TABLE_ALIGN), table_len)
            ],
            tables.at[pl.ds(pl.multiple_of(slot * table_len, _TABLE_ALIGN), table_len)],
            table_sems.at[slot],
        )

    def copy(base, g, slot, table_slot):
        col = g
        if pages_per_seq % pages_per_step:
            # Keep the last group's block-table reads in its request's row.
            first = base % pages_per_seq
            col = jnp.minimum(first + g, pages_per_seq - 1) - first
        if table_len > pages_per_step:  # windows start below unaligned groups
            col = col + base - table_start(base)
        return pltpu.make_async_copy(
            words.at[tables[table_slot * table_len + col]],
            buf.at[slot, g],
            sems.at[slot],
        )

    def wait_pages(live, slot):
        """Wait for the slot's `live` page copies: the slot's semaphore counts
        their bytes, so one wait per set bit of `live` waits for them all
        (a wait per page cost two serial bundles each)."""
        live = jnp.minimum(live, pages_per_step)
        n = 1
        while n <= pages_per_step:
            pl.when((live & n) != 0)(fake_copy(n, slot).wait)
            n *= 2

    def fake_copy(n, slot):
        """A descriptor of `n` pages' bytes on the slot's semaphore, to wait on."""
        return pltpu.make_async_copy(
            words.at[pl.ds(0, n)], buf.at[slot, pl.ds(0, n)], sems.at[slot]
        )

    def start(item, pages, slot):
        """Start the copies of `item`'s (`ahead_group`) `pages` that are below
        its live pages, alternate pages on the second DMA queue (priority 1)."""
        base, live, table_slot = item
        for g in pages:
            pl.when(g < live)(
                functools.partial(copy(base, g, slot, table_slot).start, priority=g % 2)
            )

    def ahead_group(k):
        """(first entry, live pages, table slot) of work item `k`; no live
        pages past the grid."""
        k_base, k_live = group(jnp.minimum(k, num_items - 1))
        return k_base, jnp.where(k < num_items, k_live, 0), k % table_slots

    lead = table_slots - ahead - 1

    @pl.when(step == 0)
    def _():
        for k in range(ahead + lead):
            pl.when(k < num_items)(table_copy(k).start)
        for k in range(ahead):
            pl.when(k < num_items)(table_copy(k).wait)
            start(ahead_group(k), range(pages_per_step), k)

    # The ahead group's table window is needed for its copies below; the
    # one `lead` steps later starts now.
    pl.when(step + ahead < num_items)(table_copy(step + ahead).wait)
    pl.when(step + ahead + lead < num_items)(
        lambda: table_copy(step + ahead + lead).start()
    )
    base, live = group(jnp.minimum(step, num_items - 1))
    next_item = ahead_group(step + ahead)
    q_pair = _paired_query(q_ref[...].astype(jnp.bfloat16))
    # Prefill-chunk tiles only (`FP8_MXU_QUERIES`); decode-sized tiles keep bf16.
    fp8_queries = FP8_MXU_QUERIES and block_q > DEFAULT_BLOCK_Q
    # Saturate: e4m3 has no inf, and a cast past its 448 range reads NaN.
    q_pair_fp8 = (
        jnp.clip(q_pair, -448, 448).astype(jnp.float8_e4m3fn) if fp8_queries else None
    )
    num_chunks = pages_per_step // chunk
    scores_transposed = _scores_transposed(block_q, rows, cache_ref.dtype)
    # The ahead group's descriptors, spread over the first chunks' page pairs
    # (in one chunk they crowd its scalar slots: slower, bench j-9956cd34;
    # `_DMA_ISSUE_CHUNKS` says how many).
    issue_chunks = min(_DMA_ISSUE_CHUNKS, num_chunks)
    per_pair = -(-pages_per_step // (issue_chunks * (chunk // 2)))

    if defer:
        lane_tiles, lanes = out_buf.shape[1], out_buf.shape[-1]
        parity = step % 2

        def write_previous():
            """Work item ``step - 1``'s output block (`o_ref`): the maxima of
            its last chunk, left in `raw_buf`, and of its other chunks."""
            scale = (lambda p: raw_scale[p : p + 1]) if folded else None
            last = _page_maxima(lambda p: raw_buf[p], scale, chunk)
            for t in range(lane_tiles):
                o_ref[:, t * lanes : (t + 1) * lanes] = jnp.where(
                    meta[0] == t, last, out_buf[1 - parity, t]
                )

    def score_chunk(slot, first, share, size=chunk, last=False):
        """Score pages ``[first, first + size)``, issuing the ahead group's
        descriptors `share` among the first page pairs (`start`); a partial
        chunk (transposed tiles only) issues the rest of its share after."""

        def issue(j):
            if j < chunk // 2:
                start(
                    next_item,
                    share[j * per_pair : (j + 1) * per_pair],
                    (slot + ahead) % slots,
                )

        if scores_transposed:
            args = (q_pair, buf.at[slot], first, size, live, issue)
            kwargs = dict(dtype=cache_ref.dtype, folded=folded)
            if defer and last:
                s, scales = _chunk_scores_t(*args, **kwargs)
                half, page_size = size // 2, s.shape[1] // (size // 2)
                for p in range(size):
                    side, j = divmod(p, half)
                    raw_buf[p] = s[
                        side * block_q : (side + 1) * block_q,
                        j * page_size : (j + 1) * page_size,
                    ]
                if folded:
                    raw_scale[:size] = jnp.concatenate(scales)
                meta[0] = first // lanes
            elif defer:  # a full chunk, one lane tile
                out_buf[parity, first // lanes] = _chunk_maxima_t(*args, **kwargs)
            elif transposed:
                o_ref[:, pl.ds(first, size)] = _chunk_maxima_t(*args, **kwargs)
            else:
                o_ref[pl.ds(first, size), :] = _chunk_maxima_t(*args, **kwargs).T
            if size < chunk:
                rest = share[size // 2 * per_pair :]
                start(next_item, rest, (slot + ahead) % slots)
            return

        if folded:
            # Each page's slot scales as a column of one transpose for the
            # chunk; stale VMEM past the request's end and unwritten slots
            # get scale 0.
            scale_rows = jnp.concatenate(
                [
                    pltpu.bitcast(buf[slot, first + j, rows - 1 :], jnp.float32)
                    for j in range(chunk)
                ],
                axis=0,
            )  # [chunk, S]
            page = first + jax.lax.broadcasted_iota(jnp.int32, scale_rows.shape, 0)
            scale_cols = jnp.where(
                jnp.isfinite(scale_rows) & (page < live), scale_rows, 0.0
            ).T.astype(jnp.bfloat16)  # [S, chunk]

        fp8_mxu = fp8_queries and (folded or cache_ref.dtype == jnp.float8_e4m3fn)

        def page_keys(j):
            issue(j)
            if folded:
                k = pltpu.bitcast(buf[slot, first + j, : rows - 1], jnp.int4)
                k = k.astype(jnp.bfloat16) * scale_cols[:, j : j + 1]
                if fp8_mxu:  # saturate: e4m3 has no inf (a cast past 448 is NaN)
                    return jnp.clip(k, -448, 448).astype(jnp.float8_e4m3fn)
                return k
            # A page past the request's end was never copied: its stale words
            # may be NaN or inf, which would reach its pair's score through
            # the zero half of q_pair.
            w = jnp.where(first + j < live, buf[slot, first + j], 0)
            k = pltpu.bitcast(w, cache_ref.dtype)
            return k if fp8_mxu else k.astype(jnp.bfloat16)

        if fp8_mxu:
            m = _chunk_maxima(q_pair_fp8, page_keys, chunk)
        else:
            m = _chunk_maxima(q_pair, page_keys, chunk)
        if transposed:
            o_ref[:, pl.ds(first, chunk)] = m.T
        else:
            o_ref[pl.ds(first, chunk), :] = m

    share = per_pair * (chunk // 2)

    def body(slot, scored):
        """Score the group's first `scored` pages (whole chunks, but for a
        partial last chunk of transposed tiles); the rest are past the
        request's last page (their scores are never read) and only issue
        their share of the ahead group's descriptors."""
        wait_pages(live, slot)
        if defer:  # among this step's matmuls
            write_previous()
        for c in range(num_chunks):
            pages = range(
                min(c * share, pages_per_step), min((c + 1) * share, pages_per_step)
            )
            size = min(chunk, scored - c * chunk)
            if size > 0:
                score_chunk(slot, c * chunk, pages, size, (c + 1) * chunk >= scored)
            else:
                start(next_item, pages, (slot + ahead) % slots)

    # One straight-line body per slot and count of scored pages, so that a
    # chunk's matmuls overlap the previous chunk's maxima.
    counts = sorted({num_chunks} | {2**k for k in range(1, 8) if 2**k < num_chunks})
    scored = [n * chunk for n in counts]
    if scores_transposed:
        # Transposed (decode) tiles also stop after one chunk or at a partial
        # last chunk: a step scores up to the next of these page counts
        # (bench j-7a44afac, b43 twin shape: 70.1 -> 66.1 us per call).
        tails = {s - chunk + t for s in scored for t in _TAIL_PAGES if t < chunk}
        scored = sorted({chunk} | set(scored) | {s for s in tails if s > chunk})
    group_live = jnp.minimum(live, pages_per_step)

    def slot_bodies(slot):
        for i, n in enumerate(scored):
            lo = scored[i - 1] if i else 0
            pl.when((group_live > lo) & (group_live <= n))(
                functools.partial(body, slot, n)
            )

    # Slot first, then count: a step skips the other slots' bodies with one
    # branch each.
    for slot in range(slots):
        pl.when((step % slots == slot) & (step < num_items))(
            functools.partial(slot_bodies, slot)
        )
    if defer:
        pl.when(step == num_items)(write_previous)


_KERNELS = {"v1": _kernel_v1, "v2": _kernel_v2}


def lookup(table: jax.Array, index: jax.Array) -> jax.Array:
    """``table[index]`` for a small per-request table and in-range `index`
    (an out-of-range one reads 0), by a one-hot over its entries: XLA lowers
    a gather from a small table to a compare chain over one scalar operand
    per entry, ~0.6 us each (a [T] lookup into 64 requests' table: ~72 us,
    c64 decode twin j-0a66fa2e)."""
    onehot = index[..., None] == jnp.arange(table.shape[0], dtype=index.dtype)
    if table.dtype == jnp.bool_:
        return jnp.any(onehot & table, axis=-1)
    return jnp.sum(jnp.where(onehot, table, 0), axis=-1, dtype=table.dtype)


def _tiles(cu_q_lens: jax.Array, seq_in_class: jax.Array, block_q: int, max_tiles: int):
    """Query tiles of one tile class.

    A request's tiles start at its first token rounded down to a multiple of
    `_ROW_ALIGN` (the kernel reads query rows at aligned element offsets), so
    a tile may begin with up to `_ROW_ALIGN - 1` rows of the previous request
    and end with rows of the next; those rows' scores are never read.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    start = (cu_q_lens[:-1] // _ROW_ALIGN) * _ROW_ALIGN
    q_lens = jnp.where(seq_in_class, cu_q_lens[1:] - start, 0)
    tiles_per_seq = -(-q_lens // block_q)
    tile_ends = jnp.cumsum(tiles_per_seq)
    tile = jnp.arange(max_tiles, dtype=jnp.int32)
    tile_seq = jnp.minimum(
        jnp.searchsorted(tile_ends, tile, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    first_tile = tile_ends - tiles_per_seq
    tile_row0 = (
        lookup(start, tile_seq) + (tile - lookup(first_tile, tile_seq)) * block_q
    )
    return tile_seq, tile_row0, first_tile, start, tile_ends[-1]


class ClassScores(NamedTuple):
    """One tile class's scores (`_class_scores`) and where its rows sit."""

    # [P, tiles * Bq]; [tiles * Bq, P] if token-major; [tiles, P, Bq] if tile-major
    scores: jax.Array
    col: jax.Array  # [T] each query row's column
    tile_seq: jax.Array  # [tiles] request of each tile
    tile_row0: jax.Array  # [tiles] query row of each tile's first column
    num_tiles: jax.Array  # tiles in use
    first_tile: jax.Array  # [max_num_seqs] each request's first tile
    tile_start: jax.Array  # [max_num_seqs] query row of its first tile


def _class_scores(
    index_q,
    index_cache,
    seq_lens,
    block_tables,
    cu_q_lens,
    seq_in_class,
    *,
    block_q,
    max_tiles,
    pages_per_step,
    kernel,
    interpret,
    token_major=False,
    tile_major=False,
    scale_cache=None,
    scale_row=None,
    q_pad=None,
):
    """Transposed scores ``[P, tiles * block_q]`` of one tile class, and the
    column of each token (`ClassScores`); `token_major` (kernel v3) returns
    them as ``[tiles * block_q, P]`` instead, so a token's scores are one row,
    and `tile_major` (kernel v3) as ``[tiles, P, block_q]``, so a tile's
    scores are contiguous.
    The query rows are padded by ``q_pad >= block_q`` rows (default
    `block_q`): classes sharing it share one padded copy."""
    num_tokens, _, head_dim = index_q.shape
    _, rows_per_page, packing, _ = index_cache.shape
    page_size = _page_tokens(rows_per_page, index_cache.dtype)
    fp4 = scale_row is not None
    if fp4 and kernel == "v1":
        raise NotImplementedError("4-bit index caches need kernel v2 or v3")
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    num_groups_per_seq = -(-pages_per_seq // pages_per_step)
    padded_pages = num_groups_per_seq * pages_per_step

    tile_seq, tile_row0, first_tile, tile_start, num_tiles = _tiles(
        cu_q_lens, seq_in_class, block_q, max_tiles
    )
    # The kernel reads each tile's query rows straight from the token-ordered
    # queries at an element offset (a gather of [tiles * block_q, D] cost
    # ~0.5 ms per layer on an 8K prefill, twin j-52e2df18). Rows past a
    # request's end read the next request's queries; their columns are never
    # gathered back. f32 so that any row offset is a whole 32-bit row.
    if max_num_seqs > _SEQ_MASK + 1:
        raise NotImplementedError(
            f"at most {_SEQ_MASK + 1} requests, got {max_num_seqs}"
        )
    q_pad = block_q if q_pad is None else q_pad
    q_rows = jnp.pad(index_q[:, 0].astype(jnp.float32), ((0, q_pad), (0, 0)))
    tile_info = (
        jnp.clip(tile_row0, 0, num_tokens // _ROW_ALIGN * _ROW_ALIGN) << _ROW_SHIFT
    ) | tile_seq

    # Work list: every page group of every tile's request.
    seq_pages = -(-seq_lens // page_size)
    groups_per_tile = jnp.where(
        jnp.arange(max_tiles) < num_tiles,
        jnp.maximum(-(-lookup(seq_pages, tile_seq) // pages_per_step), 1),
        0,
    )
    group_ends = jnp.cumsum(groups_per_tile)
    w = jnp.arange(max_tiles * num_groups_per_seq, dtype=jnp.int32)
    work_tile = jnp.minimum(
        jnp.searchsorted(group_ends, w, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_tiles - 1,
    )
    # A one-hot lookup: gathers from these max_tiles-entry tables lowered to
    # compare/select chains over max_tiles HBM scalar operands each, ~0.6 us
    # per operand (20 us for this fusion in the decode twin, j-b312f26a).
    in_tile = work_tile[:, None] == jnp.arange(max_tiles, dtype=jnp.int32)[None, :]
    work_group = w - jnp.sum(
        jnp.where(in_tile, (group_ends - groups_per_tile)[None, :], 0), axis=1
    )
    work = (work_tile << _TILE_SHIFT) | jnp.clip(work_group, 0, _GROUP_MASK)

    def page_block(g):
        def index_map(w, work, tile_info, block_tables, *_):
            entry = work[w]
            seq = tile_info[entry >> _TILE_SHIFT] & _SEQ_MASK
            col = jnp.minimum(
                (entry & _GROUP_MASK) * pages_per_step + g, pages_per_seq - 1
            )
            return (block_tables[seq * pages_per_seq + col], 0, 0, 0)

        return pl.BlockSpec((1, rows_per_page, packing, head_dim), index_map)

    def scale_block(g):
        page_map = page_block(g).index_map
        return pl.BlockSpec(
            (1, scale_cache.shape[1], msa_fp4.SCALE_LANES),
            lambda *args: page_map(*args)[:3],
        )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        # One step even with no tile in the class; its scores are never read.
        grid=(jnp.maximum(group_ends[-1], 1),),
        in_specs=[
            pl.BlockSpec(
                (pl.Element(block_q), pl.Element(head_dim)),
                lambda w, work, tile_info, *_: (
                    pl.multiple_of(
                        tile_info[work[w] >> _TILE_SHIFT] >> _ROW_SHIFT, _ROW_ALIGN
                    ),
                    0,
                ),
            ),
            *(page_block(g) for g in range(pages_per_step)),
            *(scale_block(g) for g in range(pages_per_step) if fp4),
        ],
        out_specs=pl.BlockSpec(
            (pages_per_step, block_q),
            lambda w, work, *_: (work[w] & _GROUP_MASK, work[w] >> _TILE_SHIFT),
        ),
    )
    out_shape = jax.ShapeDtypeStruct((padded_pages, max_tiles * block_q), jnp.float32)
    out_specs = grid_spec.out_specs
    if token_major:
        if kernel != "v3":
            raise NotImplementedError(
                f"token-major scores need kernel v3, got {kernel}"
            )
        out_shape = jax.ShapeDtypeStruct(out_shape.shape[::-1], jnp.float32)
        out_specs = pl.BlockSpec(
            (block_q, pages_per_step),
            lambda w, work, *_: (work[w] >> _TILE_SHIFT, work[w] & _GROUP_MASK),
        )
    elif tile_major:
        if kernel != "v3":
            raise NotImplementedError(f"tile-major scores need kernel v3, got {kernel}")
        out_shape = jax.ShapeDtypeStruct(
            (max_tiles, padded_pages, block_q), jnp.float32
        )
        out_specs = pl.BlockSpec(
            (None, pages_per_step, block_q),
            lambda w, work, *_: (work[w] >> _TILE_SHIFT, work[w] & _GROUP_MASK, 0),
        )
    params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",), disable_bounds_checks=kernel == "v3"
    )
    if kernel == "v3":
        if fp4:
            raise NotImplementedError(
                "kernel v3 reads 4-bit pages with folded scales only"
            )
        chunk = _v3_chunk(pages_per_step, block_q, page_size)
        if pages_per_step % chunk or chunk % 2:
            raise NotImplementedError(
                f"kernel v3 pairs pages in chunks of up to {_PAGES_PER_CHUNK}: "
                f"pages_per_step must be an even divisor or a multiple of it, "
                f"got {pages_per_step}"
            )
        # Block-table windows of a group's entries, whole alignment units:
        # an unaligned group's starts below it, so it takes one more, and the
        # table is padded for the last request's.
        table_len = -(-pages_per_step // _TABLE_ALIGN) * _TABLE_ALIGN
        if (pages_per_seq % _TABLE_ALIGN) or (pages_per_step % _TABLE_ALIGN):
            table_len += _TABLE_ALIGN
        last_base = (max_num_seqs - 1) * pages_per_seq + padded_pages - pages_per_step
        table_end = last_base // _TABLE_ALIGN * _TABLE_ALIGN + table_len
        tables = jnp.pad(block_tables, (0, max(0, table_end - block_tables.shape[0])))
        # Transposed token-major tiles defer each step's last chunk's maxima
        # (`_kernel_v3`), whole 128-lane chunks only.
        defer = (
            token_major
            and _scores_transposed(block_q, rows_per_page, index_cache.dtype)
            and chunk == 128
        )
        grid, q_spec, deferred = grid_spec.grid, grid_spec.in_specs[0], []
        if defer:
            # One more step, and the output block of the item a step back.
            grid = (grid[0] + 1,)
            last_work = work.shape[0] - 1
            q_spec = pl.BlockSpec(
                q_spec.block_shape,
                lambda w, work, tile_info, *_: (
                    pl.multiple_of(
                        tile_info[work[jnp.minimum(w, last_work)] >> _TILE_SHIFT]
                        >> _ROW_SHIFT,
                        _ROW_ALIGN,
                    ),
                    0,
                ),
            )
            # Steps 0 and 1 map to the same block, so step 0's write (of no
            # chunk) is never written back: step 1 overwrites the block.
            out_specs = pl.BlockSpec(
                (block_q, pages_per_step),
                lambda w, work, *_: (
                    work[jnp.maximum(w - 1, 0)] >> _TILE_SHIFT,
                    work[jnp.maximum(w - 1, 0)] & _GROUP_MASK,
                ),
            )
            deferred = [
                pltpu.VMEM((2, pages_per_step // 128, block_q, 128), jnp.float32),
                pltpu.VMEM((chunk, block_q, page_size), jnp.float32),
                pltpu.SMEM((1,), jnp.int32),
            ]
            if msa_fp4.scales_in_page(rows_per_page, index_cache.dtype):
                deferred.append(pltpu.VMEM((chunk, head_dim), jnp.float32))
        scores_t = pl.pallas_call(
            functools.partial(
                _kernel_v3,
                pages_per_seq=pages_per_seq,
                transposed=token_major,
                defer=defer,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                grid=grid,
                in_specs=[
                    q_spec,
                    pl.BlockSpec(memory_space=pl.ANY),
                    pl.BlockSpec(memory_space=pl.ANY),
                ],
                out_specs=out_specs,
                scratch_shapes=[
                    pltpu.VMEM(
                        (_RING_SLOTS, pages_per_step, rows_per_page, head_dim),
                        jnp.uint32,
                    ),
                    pltpu.SemaphoreType.DMA((_RING_SLOTS,)),
                    pltpu.SMEM((_TABLE_SLOTS * table_len,), jnp.int32),
                    pltpu.SemaphoreType.DMA((_TABLE_SLOTS,)),
                    *deferred,
                ],
            ),
            out_shape=out_shape,
            compiler_params=params,
            interpret=interpret,
            name="minimax_m3_msa_block_scores",
        )(work, tile_info, seq_pages, q_rows, index_cache, tables)
    else:
        kernel_fn = _KERNELS[kernel]
        if fp4:
            kernel_fn = functools.partial(kernel_fn, scale_row=scale_row)
        scores_t = pl.pallas_call(
            kernel_fn,
            grid_spec=grid_spec,
            out_shape=out_shape,
            compiler_params=params,
            interpret=interpret,
            name="minimax_m3_msa_block_scores",
        )(
            work,
            tile_info,
            block_tables,
            q_rows,
            *([index_cache] * pages_per_step),
            *([scale_cache] * (pages_per_step if fp4 else 0)),
        )

    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    col = _token_columns(
        tok, cu_q_lens, first_tile, tile_start, block_q, max_tiles * block_q
    )
    return ClassScores(
        scores_t, col, tile_seq, tile_row0, num_tiles, first_tile, tile_start
    )


def _token_columns(rows, cu_q_lens, first_tile, tile_start, block_q, num_cols):
    """The score column of each query row in `rows`: row t of request r sits
    in r's tiles at its offset."""
    seq = jnp.minimum(
        jnp.searchsorted(
            cu_q_lens[1:], rows, side="right", method="compare_all"
        ).astype(jnp.int32),
        cu_q_lens.shape[0] - 2,
    )
    offset = rows - lookup(tile_start, seq)
    col = (lookup(first_tile, seq) + offset // block_q) * block_q + offset % block_q
    return jnp.clip(col, 0, num_cols - 1)


class _ClassPlan(NamedTuple):
    """`block_scores`' two tile classes. `small()` / `large()` score them
    (`ClassScores` each); `token_major` is the small class's layout."""

    small: Callable
    large: Callable
    small_block_q: int
    large_block_q: int
    large_pages: int
    max_large: int
    is_large: jax.Array
    token_major: bool


def is_prefill_step(num_tokens: int, max_num_seqs: int) -> bool:
    """At least `max_num_seqs` full small tiles of rows (decode requests of
    1 + 7 MTP tokens fill an eighth of a tile per folded head): `block_scores`
    scores both tile classes unconditionally. The bound is inclusive: a
    2048-token step of two folded heads at 64 requests (4,096 rows) is a
    mixed step, and the token-order path gathered all its block scores back
    to token order (~76 us per layer, twin j-8b2aa6c0)."""
    return num_tokens >= max_num_seqs * _SMALL_TILE_Q


def class_plan(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    block_q: int = DEFAULT_BLOCK_Q,
    large_block_q: int = DEFAULT_LARGE_BLOCK_Q,
    pages_per_step: int = DEFAULT_PAGES_PER_STEP,
    small_pages_per_step: int = DEFAULT_SMALL_PAGES_PER_STEP,
    kernel: str = DEFAULT_KERNEL,
    large_kernel: str = DEFAULT_LARGE_KERNEL,
    scale_cache: jax.Array | None = None,
    scale_row: int | None = None,
    transposed_tile_q: int = _SMALL_TILE_Q_TRANSPOSED,
    interpret: bool = False,
) -> _ClassPlan:
    """`block_scores`' tile classes, unscored (arguments as `block_scores`)."""
    num_tokens, num_heads, _ = index_q.shape
    if num_heads != 1:
        raise NotImplementedError(
            f"MSA indexer scores one index head per rank, got {num_heads}."
        )
    if msa_fp4.is_fp4(index_cache) and msa_fp4.scales_in_page(
        index_cache.shape[1], index_cache.dtype
    ):
        scale_cache = scale_row = None  # the pages carry their own scales
    elif msa_fp4.is_fp4(index_cache):
        if scale_cache is None or scale_row is None:
            raise ValueError("a 4-bit index cache needs scale_cache and scale_row")
    else:
        check_page_dtype(index_cache.shape[2], index_cache.dtype)
        scale_row = None
    max_num_seqs = seq_lens.shape[0]
    seq_ids = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    active = seq_ids < num_seqs
    is_large = active & (q_lens > block_q)
    score = functools.partial(
        _class_scores,
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_row=scale_row,
        # One padded f32 copy of the queries for both classes (~32 us per
        # layer each at DCP 8, c64 twin j-0a66fa2e).
        q_pad=max(block_q, large_block_q),
    )
    # Large requests have more than block_q tokens each.
    max_large = min(max_num_seqs, num_tokens // (block_q + 1))
    # Kernel v3 reads 4-bit pages with folded scales only.
    unfolded_fp4 = scale_row is not None
    small_kernel = "v2" if kernel == "v3" and unfolded_fp4 else kernel
    large_kernel = "v2" if large_kernel == "v3" and unfolded_fp4 else large_kernel
    # Token-major scores make the gather back to token order a row gather; a
    # column gather first relayouts the whole padded [P, tiles * block_q] f32
    # block (36 us per layer in the c48 decode twin, j-a281ffb0). Needs
    # lane-aligned [tile, pages_per_step] blocks.
    token_major = small_kernel == "v3" and small_pages_per_step % 128 == 0
    # Token-major tiles of up to _SMALL_TILE_Q rows: decode requests have 8
    # (MTP) to 32 (folded index heads) rows, and a tile of up to 64 rows costs
    # the same as one of 8 (paired scores are 2 x tile lanes). A request of up
    # to block_q rows takes several tiles; none holds more than the batch.
    transposed = scale_row is None and _scores_transposed(
        1, index_cache.shape[1], index_cache.dtype
    )
    tile = (
        min(
            block_q,
            transposed_tile_q if transposed else _SMALL_TILE_Q,
            -(-num_tokens // _ROW_ALIGN) * _ROW_ALIGN,
        )
        if token_major
        else block_q
    )
    # Aligned tile starts add up to _ROW_ALIGN - 1 rows per request.
    tiles_per_request = -(-(block_q + _ROW_ALIGN - 1) // tile)
    small = functools.partial(
        score,
        active & ~is_large,
        block_q=tile,
        max_tiles=min(
            tiles_per_request * max_num_seqs,
            (num_tokens + (_ROW_ALIGN - 1) * max_num_seqs) // tile + max_num_seqs,
        ),
        pages_per_step=small_pages_per_step,
        kernel=small_kernel,
        token_major=token_major,
    )
    large = functools.partial(
        score,
        is_large,
        block_q=large_block_q,
        # +1 per request for the partial last tile, plus the up-to-7 rows
        # each aligned start adds.
        max_tiles=num_tokens // large_block_q
        + max_large
        + -(-(_ROW_ALIGN - 1) * max_large // large_block_q)
        + 1,
        pages_per_step=pages_per_step,
        kernel=large_kernel,
        # A tile's scores contiguous: its picks read one run, where the
        # transposed [P, tiles * block_q] layout's block was P runs of one
        # tile row (2 KB each), DMA-bound (c64 mixed candidates bench).
        tile_major=large_kernel == "v3",
    )
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    return _ClassPlan(
        small=small,
        large=large,
        small_block_q=tile,
        large_block_q=large_block_q,
        large_pages=-(-pages_per_seq // pages_per_step) * pages_per_step,
        max_large=max_large,
        is_large=is_large,
        token_major=token_major,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_q",
        "large_block_q",
        "pages_per_step",
        "small_pages_per_step",
        "kernel",
        "large_kernel",
        "scale_row",
        "transposed_tile_q",
        "interpret",
    ),
)
def block_scores(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    block_q: int = DEFAULT_BLOCK_Q,
    large_block_q: int = DEFAULT_LARGE_BLOCK_Q,
    pages_per_step: int = DEFAULT_PAGES_PER_STEP,
    small_pages_per_step: int = DEFAULT_SMALL_PAGES_PER_STEP,
    kernel: str = DEFAULT_KERNEL,
    large_kernel: str = DEFAULT_LARGE_KERNEL,
    scale_cache: jax.Array | None = None,
    scale_row: int | None = None,
    transposed_tile_q: int = _SMALL_TILE_Q_TRANSPOSED,
    row_index: jax.Array | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Block-max index scores ``[T, 1, P]`` (``P`` padded to the page group),
    or with `row_index` ``[R, 1, P]``: row ``i`` is query row
    ``row_index[i]``'s, picked by the gather that collects the tile classes
    (gathering the result again was another SparseCore offload and a
    relayout, ~15 us per layer in the c64 decode twin, j-9a9fccc2).

    Requests of at most `block_q` query tokens (decode, short chunks) are
    scored in `block_q`-row tiles; longer prefill chunks in `large_block_q`-row
    tiles, so a request's index keys are read once per large tile rather than
    once per `block_q` rows.

    Args:
      index_q: ``[T, 1, D]`` index queries (one index head per rank).
      index_cache: ``[pages, S / p, p, D]`` key-only cache (bf16 or fp8).
      seq_lens, block_tables, cu_q_lens, num_seqs: ragged batch metadata.
      kernel: ``"v2"`` (bitcast unpack, query latched across pages), ``"v3"``
        (v2 with a hand-written DMA ring for the pages) or the original
        ``"v1"``, kept as the reference arm for benchmarks. A 4-bit cache
        without folded scales runs v2 where v3 is asked for.
      large_kernel: the same choice (and fallback) for the `large_block_q` tiles.
      pages_per_step / small_pages_per_step: pages per grid step of the large
        and the block_q tile classes.
      scale_cache, scale_row: for a 4-bit (uint8) index cache without folded
        scales (`msa_fp4.scales_in_page`),
        the ``[pages, 8, 128]`` f32 scale cache and the row of the index-key
        scales. `index_q` must then be rotated (`msa_fp4.rotate_queries`).
    Pages past a token's own page hold scores of keys it must not see, and
    pages past its request's last page are not written; callers mask both.
    """
    plan = class_plan(
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        num_seqs,
        block_q=block_q,
        large_block_q=large_block_q,
        pages_per_step=pages_per_step,
        small_pages_per_step=small_pages_per_step,
        kernel=kernel,
        large_kernel=large_kernel,
        scale_cache=scale_cache,
        scale_row=scale_row,
        transposed_tile_q=transposed_tile_q,
        interpret=interpret,
    )
    num_tokens = index_q.shape[0]
    max_num_seqs = seq_lens.shape[0]
    max_large, is_large, token_major = plan.max_large, plan.is_large, plan.token_major
    scored = plan.small()
    small = scored.scores
    # The classes may pad pages to different group multiples; pad to the wider.
    pages = max(small.shape[1 if token_major else 0], plan.large_pages)
    query_rows = (
        jnp.arange(num_tokens, dtype=jnp.int32) if row_index is None else row_index
    )

    def columns(scored, block_q, num_cols):
        if row_index is None:
            return scored.col
        return _token_columns(
            row_index,
            cu_q_lens,
            scored.first_tile,
            scored.tile_start,
            block_q,
            num_cols,
        )

    small_col = columns(
        scored, plan.small_block_q, small.shape[0 if token_major else 1]
    )

    def small_only():
        rows = small[small_col] if token_major else small[:, small_col].T
        return jnp.pad(rows, ((0, 0), (0, pages - rows.shape[1])))

    if max_large == 0:
        return small_only()[:, None, :]

    def with_large():
        large_scored = plan.large()
        large = large_scored.scores
        if large.ndim == 3:  # tile-major
            large = large.transpose(1, 0, 2).reshape(large.shape[1], -1)
        large_col = columns(large_scored, plan.large_block_q, large.shape[1])
        small_t = small.T if token_major else small
        seq = jnp.minimum(
            jnp.searchsorted(
                cu_q_lens[1:], query_rows, side="right", method="compare_all"
            ).astype(jnp.int32),
            max_num_seqs - 1,
        )
        # One gather over both classes' columns (XLA fuses the concatenate
        # into it) instead of a transposing gather per class and a select.
        col = jnp.where(lookup(is_large, seq), small_t.shape[1] + large_col, small_col)
        small_t, large = (
            jnp.pad(x, ((0, pages - x.shape[0]), (0, 0))) for x in (small_t, large)
        )
        return jnp.concatenate([small_t, large], axis=1)[:, col].T

    if is_prefill_step(num_tokens, max_num_seqs):
        # More rows than a batch of decode requests: a prefill step. A
        # conditional would wait for the gather (offloaded to the SparseCore)
        # and its neighbours' collectives at its end (c4 mixed twin +50 us
        # per layer, j-08c5e333).
        return with_large()[:, None, :]
    # Decode steps have no large request: skip its kernel call and gather.
    return jax.lax.cond(jnp.any(is_large), with_large, small_only)[:, None, :]
