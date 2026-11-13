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
DEFAULT_PAGES_PER_STEP = 32
# Pages per grid step the MSA forward gives the large (prefill) class
# (`msa.block_score_options`): a positive multiple of 32. Wider steps amortize
# the per-step cost over more pages; the scores are bit-identical (each page
# maximum is one dot, whatever the grid). Set from key
# minimax_indexer_large_pages_per_step by the attention layer.
LARGE_PAGES_PER_STEP = DEFAULT_PAGES_PER_STEP
# The block_q class (decode, MTP verify) runs the v3 DMA ring; 128 pages per
# step keep more bytes in flight (j-dba77583: 32 req x 8 tokens at 128K,
# 1.95 -> 1.66 ms). The large prefill class keeps LARGE_PAGES_PER_STEP.
DEFAULT_SMALL_PAGES_PER_STEP = 128
DEFAULT_KERNEL = "v3"
# Rows per token-major query tile of the block_q class (see `block_scores`).
_SMALL_TILE_Q = 64
# Folded 4-bit pages are scored transposed (`_folded_chunk_maxima_t`), whose
# cost grows with the tile's rows: by default 32 holds a decode request of 4
# folded index heads x 8 (MTP) tokens; callers with one index head per rank
# pass `folded_tile_q=8` (`msa.block_score_options`).
_SMALL_TILE_Q_FOLDED = 32
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


# Pages per dot in `_kernel_v3`: 2 x 16 pages, paired along the contraction.
_PAGES_PER_CHUNK = 32
# `_kernel_v3` issues the next group's descriptors during its first chunks.
_DMA_ISSUE_CHUNKS = 2
# Folded 4-bit tiles of up to this many rows score transposed
# (`_folded_chunk_maxima_t`, maxima across lanes); larger ones score
# token-major with the slot scales folded into the keys (maxima across
# sublanes, as the other caches).
_FOLDED_TRANSPOSED_MAX_ROWS = 64
# r6 (opt-in): fp8 index caches also score transposed (`_fp8_chunk_maxima_t`)
# for tiles of up to `_FOLDED_TRANSPOSED_MAX_ROWS` rows, their keys fed to
# the MXU as fp8 as stored (no fp8 -> bf16 unpack; AOT, 128 pages at 8
# rows: 2,171 -> 819 bundles). Set from key minimax_indexer_fp8_transposed
# by the attention layer.
FP8_TRANSPOSED = False
# r6 (opt-in): `_kernel_v3`'s DMA ring holds 3 page groups (issuing 2 steps
# ahead; with one group ahead each step waited out its pages' DMA latency,
# ~1.3 us per step at 64 or 128 pages), alternate pages go to the second DMA
# queue, and a group's pages are waited for with one wait per set bit of its
# page count. Set from key minimax_indexer_dma_ring3 by the attention
# layer.
DMA_RING3 = False
_RING3_SLOTS = 3


def _fp8_transposed(block_q: int, dtype) -> bool:
    """Whether `_kernel_v3` scores fp8-cache tiles of `block_q` rows with
    `_fp8_chunk_maxima_t` (`FP8_TRANSPOSED`)."""
    return (
        FP8_TRANSPOSED
        and dtype == jnp.float8_e4m3fn
        and block_q <= _FOLDED_TRANSPOSED_MAX_ROWS
    )
# Transposed folded 4-bit scoring feeds the MXU the int4 keys as fp8 rather
# than bf16: int4 values are exact in e4m3 and bf16 x fp8 is native on tpu7x,
# so the scores are unchanged, and the cheaper unpack makes the c64 decode
# indexer 9-17% faster (DCP8 / DCP4, bit-identical scores, j-fd386dc1).
# Set with --additional-config '{"msa_index_int4_fp8": true}'
# (`set_int4_fp8_keys`), before the model's first trace.
_int4_fp8_keys = False


def set_int4_fp8_keys(enabled: bool) -> None:
    global _int4_fp8_keys
    _int4_fp8_keys = bool(enabled)


def int4_fp8_keys() -> bool:
    return _int4_fp8_keys


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


def _folded_chunk_maxima_t(q_pair, buf, first, chunk, live, issue) -> jax.Array:
    """``[Bq, chunk]`` block maxima of the folded 4-bit pages
    ``buf[first : first + chunk]`` with the scores computed transposed,
    ``[2 Bq, tokens]``: the per-token scales are a lane vector (the page's
    last word row), applied by a sublane broadcast, and the MXU result is
    ``2 Bq`` rows rather than ``chunk * S / 2``. `issue(j)` runs among the
    keys of page pair ``j``.
    """
    rows = buf.shape[1]
    half = chunk // 2
    block_q = q_pair.shape[0] // 2
    keys, scale_rows = [], [[], []]
    for j in range(half):
        issue(j)
        pair = []
        for side, p in enumerate((first + j, first + half + j)):
            k = pltpu.bitcast(buf[p, : rows - 1], jnp.int4).astype(
                jnp.float8_e4m3fn if _int4_fp8_keys else jnp.bfloat16
            )
            pair.append(k)
            s = pltpu.bitcast(buf[p, rows - 1 :], jnp.float32)  # [1, S]
            # Stale VMEM past the request's end, unwritten slots: scale 0.
            scale_rows[side].append(jnp.where(jnp.isfinite(s) & (p < live), s, 0.0))
        keys.append(jnp.concatenate(pair, axis=1))
    page_size = keys[0].shape[0]
    s = jax.lax.dot_general(
        q_pair,
        jnp.concatenate(keys, axis=0),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [2 Bq, half * S]
    left, right = (jnp.concatenate(r, axis=1) for r in scale_rows)
    row = jax.lax.broadcasted_iota(jnp.int32, s.shape, 0)
    s = s * jnp.where(row < block_q, left, right)
    m = jnp.max(s.reshape(2 * block_q, half, page_size), axis=2)  # [2 Bq, half]
    return jnp.concatenate([m[:block_q], m[block_q:]], axis=1)


def _fp8_chunk_maxima_t(q_pair, buf, first, chunk, live, issue, dtype) -> jax.Array:
    """`_folded_chunk_maxima_t` for fp8 pages (`FP8_TRANSPOSED`, r6): the keys
    enter the MXU as fp8 as stored; a page past the request's end is zeroed
    (stale VMEM may hold NaN, which would reach the paired page's scores
    through the zero half of the paired query). Only the right page of a
    pair can be past the end while its partner is not."""
    half = chunk // 2
    block_q = q_pair.shape[0] // 2
    keys = []
    for j in range(half):
        issue(j)
        pair = []
        for side, p in enumerate((first + j, first + half + j)):
            w = buf[p]
            k = pltpu.bitcast(jnp.where(p < live, w, 0) if side == 1 else w, dtype)
            pair.append(k)
        keys.append(jnp.concatenate(pair, axis=1))
    page_size = keys[0].shape[0]
    s = jax.lax.dot_general(
        q_pair,
        jnp.concatenate(keys, axis=0),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [2 Bq, half * S]
    m = jnp.max(s.reshape(2 * block_q, half, page_size), axis=2)  # [2 Bq, half]
    return jnp.concatenate([m[:block_q], m[block_q:]], axis=1)


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
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    seq_pages_ref,  # [max_num_seqs] pages of each request
    # In
    q_ref,  # [Bq, D]
    cache_ref,  # [pages, S / p, p, D] index cache, left in HBM
    # Out
    o_ref,  # [G, Bq], or [Bq, G] if `transposed`
    # Scratch
    buf,  # [2, G, S / p, D] pages of even and odd steps, as 32-bit words
    sems,  # DMA semaphores [2], one per ring slot
    *,
    transposed=False,
):
    """`_kernel_v2` with the pages fetched by a hand-written double-buffered
    DMA ring: one scalar address per page and no copy for pages past the
    request's end (their scores are never read).

    Each ring slot has straight-line bodies (static buffers, so the
    scheduler sees the next group's copies do not touch the pages being
    scored), and the next group's descriptors are issued among this group's
    matmuls. A wait between chunks would drain the MXU pipeline, so all of
    the group's pages are waited for first.

    A folded 4-bit page (`msa_fp4.scales_in_page`) is ``S / 8`` word rows of
    int4 keys and one row of f32 slot scales (`_folded_chunk_maxima_t`).
    """
    slots, pages_per_step, rows, head_dim = buf.shape
    # Groups issued ahead of the running one: 1 (v3's double buffer), or 2
    # with `DMA_RING3` (the ring's size decides; see `_class_scores`).
    ring3 = slots > 2
    ahead = slots - 1
    block_q = q_ref.shape[0]
    folded = msa_fp4.scales_in_page(rows, cache_ref.dtype)
    page_size = _page_tokens(rows, cache_ref.dtype)
    chunk = _v3_chunk(pages_per_step, block_q, page_size)
    pages_per_seq = block_tables_ref.shape[0] // seq_pages_ref.shape[0]
    step, num_steps = pl.program_id(0), pl.num_programs(0)
    # The pages as 32-bit words: a dense VMEM layout, so each key vreg is one
    # load (a packed [S / p, p, D] page tiles VMEM in (p, D) tiles).
    words = cache_ref.bitcast(jnp.uint32).reshape(cache_ref.shape[0], rows, head_dim)

    def group(w):
        """(first block-table entry, live pages) of work item `w`."""
        entry = work_ref[w]
        seq = tile_seq_ref[entry >> _TILE_SHIFT] & _SEQ_MASK
        first = (entry & _GROUP_MASK) * pages_per_step
        return seq * pages_per_seq + first, jnp.maximum(seq_pages_ref[seq] - first, 1)

    def copy(base, g, slot):
        col = g
        if pages_per_seq % pages_per_step:
            # Keep the last group's block-table reads in its request's row.
            first = base % pages_per_seq
            col = jnp.minimum(first + g, pages_per_seq - 1) - first
        return pltpu.make_async_copy(
            words.at[block_tables_ref[base + col]],
            buf.at[slot, g],
            sems.at[slot],
        )

    def start(base, live, pages, slot):
        for g in pages:
            if ring3:
                # Alternate pages on the second DMA queue (priority 1).
                pl.when(g < live)(
                    functools.partial(copy(base, g, slot).start, priority=g % 2)
                )
            else:
                pl.when(g < live)(copy(base, g, slot).start)

    def ahead_group(k):
        """(first entry, live pages) of work item `k`; none past the grid."""
        k_base, k_live = group(jnp.minimum(k, num_steps - 1))
        return k_base, jnp.where(k < num_steps, k_live, 0)

    def wait_pages(live, slot):
        """`DMA_RING3`: wait for the slot's `live` page copies with one wait
        per set bit of `live` (the slot's semaphore counts their bytes; a
        wait per page cost two serial bundles each)."""
        live = jnp.minimum(live, pages_per_step)
        n = 1
        while n <= pages_per_step:
            fake = pltpu.make_async_copy(
                words.at[pl.ds(0, n)], buf.at[slot, pl.ds(0, n)], sems.at[slot]
            )
            pl.when((live & n) != 0)(fake.wait)
            n *= 2

    @pl.when(step == 0)
    def _():
        for k in range(ahead):
            start(*ahead_group(k), range(pages_per_step), k)

    base, live = group(step)
    next_base, next_live = ahead_group(step + ahead)
    q_pair = _paired_query(q_ref[...].astype(jnp.bfloat16))
    num_chunks = pages_per_step // chunk
    # The next group's descriptors, spread over the first chunks' page pairs:
    # issued late in the step, a descriptor's DMA latency shows at the next
    # step's wait.
    issue_chunks = min(_DMA_ISSUE_CHUNKS, num_chunks)
    per_pair = -(-pages_per_step // (issue_chunks * (chunk // 2)))

    def score_chunk(slot, first, share):
        """Score pages ``[first, first + chunk)``, issuing the next group's
        descriptors `share` among the first page pairs."""

        def issue(j):
            if j < chunk // 2:
                start(
                    next_base,
                    next_live,
                    share[j * per_pair : (j + 1) * per_pair],
                    (slot + ahead) % slots,
                )

        if _fp8_transposed(block_q, cache_ref.dtype):
            m = _fp8_chunk_maxima_t(
                q_pair, buf.at[slot], first, chunk, live, issue, cache_ref.dtype
            )  # [Bq, chunk]
            if transposed:
                o_ref[:, pl.ds(first, chunk)] = m
            else:
                o_ref[pl.ds(first, chunk), :] = m.T
            return

        if folded and block_q <= _FOLDED_TRANSPOSED_MAX_ROWS:
            m = _folded_chunk_maxima_t(
                q_pair, buf.at[slot], first, chunk, live, issue
            )  # [Bq, chunk]
            if transposed:
                o_ref[:, pl.ds(first, chunk)] = m
            else:
                o_ref[pl.ds(first, chunk), :] = m.T
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

        def page_keys(j):
            issue(j)
            if folded:
                k = pltpu.bitcast(buf[slot, first + j, : rows - 1], jnp.int4)
                return k.astype(jnp.bfloat16) * scale_cols[:, j : j + 1]
            # A page past the request's end was never copied: its stale words
            # may be NaN or inf, which would reach its pair's score through
            # the zero half of q_pair.
            w = jnp.where(first + j < live, buf[slot, first + j], 0)
            return pltpu.bitcast(w, cache_ref.dtype).astype(jnp.bfloat16)

        m = _chunk_maxima(q_pair, page_keys, chunk)
        if transposed:
            o_ref[:, pl.ds(first, chunk)] = m.T
        else:
            o_ref[pl.ds(first, chunk), :] = m

    share = per_pair * (chunk // 2)

    def body(slot, scored):
        """Score the group's first `scored` chunks; the rest are past the
        request's last page (their scores are never read) and only issue
        their share of the next group's descriptors."""
        if ring3:
            wait_pages(live, slot)
        else:
            for g in range(pages_per_step):
                pl.when(g < live)(copy(base, g, slot).wait)
        for c in range(num_chunks):
            pages = range(
                min(c * share, pages_per_step), min((c + 1) * share, pages_per_step)
            )
            if c < scored:
                score_chunk(slot, c * chunk, pages)
            else:
                start(next_base, next_live, pages, (slot + ahead) % slots)

    # One straight-line body per slot and count of scored chunks, so that a
    # chunk's matmuls overlap the previous chunk's maxima.
    counts = sorted({num_chunks} | {2**k for k in range(1, 8) if 2**k < num_chunks})
    live_chunks = jnp.minimum(-(-live // chunk), num_chunks)
    for slot in range(slots):
        for i, n in enumerate(counts):
            lo = counts[i - 1] if i else 0
            pl.when((step % slots == slot) & (live_chunks > lo) & (live_chunks <= n))(
                functools.partial(body, slot, n)
            )


_KERNELS = {"v1": _kernel_v1, "v2": _kernel_v2}

# r6 (opt-in): per-request table lookups by a one-hot instead of a gather
# (`lookup`). Set from key minimax_msa_onehot_lookup by the attention
# layer.
ONEHOT_LOOKUP = False

# Opt-in (key minimax_indexer_hoist_metadata): `block_scores` steps that pick
# their large class under a `lax.cond` (decode-sized buckets with a prefill
# chunk) compute the large class's tiles, work list and columns and every
# row's gather column outside the cond, by one-hot lookups. Outside a cond
# the metadata is the same expression of the step's inputs in every layer and
# XLA CSEs it to one computation per step; inside, each layer's branch
# recomputed it (c48 512-token mixed step: 77 us per layer, j-f5a8745c).
# Same scores.
HOIST_METADATA = False
# Opt-in (key minimax_indexer_row_gather): with token-major small scores, the
# step's rows come from a row gather of the small class and a column gather
# of the large class only, selected per row, instead of transposing the
# small class into one concatenated column gather (a [P, columns] f32 copy
# and pad per layer). Same scores.
ROW_GATHER = False


def lookup(table: jax.Array, index: jax.Array, onehot: bool | None = None) -> jax.Array:
    """``table[index]`` for a small per-request table and in-range `index`.

    With `onehot` (default `ONEHOT_LOOKUP`), by a one-hot over its entries
    (an out-of-range index reads 0): XLA lowers a gather from a small table
    to a compare chain over one scalar operand per entry, ~0.6 us each (a [T]
    lookup into 64 requests' table: ~72 us, c64 decode twin j-0a66fa2e).
    Otherwise the gather."""
    if not (ONEHOT_LOOKUP if onehot is None else onehot):
        return table[index]
    onehot = index[..., None] == jnp.arange(table.shape[0], dtype=index.dtype)
    if table.dtype == jnp.bool_:
        return jnp.any(onehot & table, axis=-1)
    return jnp.sum(jnp.where(onehot, table, 0), axis=-1, dtype=table.dtype)


def _tiles(
    cu_q_lens: jax.Array,
    seq_in_class: jax.Array,
    block_q: int,
    max_tiles: int,
    onehot: bool | None = None,
):
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
        lookup(start, tile_seq, onehot)
        + (tile - lookup(first_tile, tile_seq, onehot)) * block_q
    )
    return tile_seq, tile_row0, first_tile, start, tile_ends[-1]


class ClassScores(NamedTuple):
    """One tile class's scores (`_class_scores`) and where its rows sit."""

    scores: jax.Array  # [P, tiles * Bq], or [tiles * Bq, P] if token-major
    col: jax.Array  # [T] each query row's column
    tile_seq: jax.Array  # [tiles] request of each tile
    tile_row0: jax.Array  # [tiles] query row of each tile's first column
    num_tiles: jax.Array  # tiles in use
    first_tile: jax.Array  # [max_num_seqs] each request's first tile
    tile_start: jax.Array  # [max_num_seqs] query row of its first tile


def _query_rows(index_q, pad):
    """Token-ordered f32 queries with `pad` rows after the last token, which
    the tile at the end may read past (`_class_scores`)."""
    return jnp.pad(index_q[:, 0].astype(jnp.float32), ((0, pad), (0, 0)))


class ClassMeta(NamedTuple):
    """One tile class's metadata (`_class_meta`): a function of the step's
    ragged batch (lengths, query layout) only, the same in every layer."""

    tile_seq: jax.Array  # [tiles] request of each tile
    tile_row0: jax.Array  # [tiles] query row of each tile's first column
    first_tile: jax.Array  # [max_num_seqs] each request's first tile
    tile_start: jax.Array  # [max_num_seqs] query row of its first tile
    num_tiles: jax.Array  # tiles in use
    tile_info: jax.Array  # [tiles] row << _ROW_SHIFT | request
    work: jax.Array  # [tiles * groups] tile << _TILE_SHIFT | page group
    num_work: jax.Array  # work entries in use
    seq_pages: jax.Array  # [max_num_seqs] pages of each request
    col: jax.Array  # [T] each query row's column


def _class_meta(
    num_tokens,
    page_size,
    seq_lens,
    block_tables,
    cu_q_lens,
    seq_in_class,
    *,
    block_q,
    max_tiles,
    pages_per_step,
    onehot=None,
):
    """`_class_scores`' tiles, work list and token columns (`ClassMeta`).
    `onehot` picks `lookup`'s form (default `ONEHOT_LOOKUP`); the values are
    the same either way."""
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    num_groups_per_seq = -(-pages_per_seq // pages_per_step)

    tile_seq, tile_row0, first_tile, tile_start, num_tiles = _tiles(
        cu_q_lens, seq_in_class, block_q, max_tiles, onehot
    )
    tile_info = (
        jnp.clip(tile_row0, 0, num_tokens // _ROW_ALIGN * _ROW_ALIGN) << _ROW_SHIFT
    ) | tile_seq

    # Work list: every page group of every tile's request.
    seq_pages = -(-seq_lens // page_size)
    groups_per_tile = jnp.where(
        jnp.arange(max_tiles) < num_tiles,
        jnp.maximum(-(-lookup(seq_pages, tile_seq, onehot) // pages_per_step), 1),
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

    # Back to token order: token t of request r sits in r's tiles at its offset.
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], tok, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    offset = tok - lookup(tile_start, seq, onehot)
    col = (
        lookup(first_tile, seq, onehot) + offset // block_q
    ) * block_q + offset % block_q
    col = jnp.clip(col, 0, max_tiles * block_q - 1)
    return ClassMeta(
        tile_seq,
        tile_row0,
        first_tile,
        tile_start,
        num_tiles,
        tile_info,
        work,
        group_ends[-1],
        seq_pages,
        col,
    )


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
    scale_cache=None,
    scale_row=None,
    q_rows=None,
    meta=None,
):
    """Transposed scores ``[P, tiles * block_q]`` of one tile class, and the
    column of each token (`ClassScores`); `token_major` (kernel v3) returns
    them as ``[tiles * block_q, P]`` instead, so a token's scores are one row.
    `q_rows` (`_query_rows`, padded by at least block_q) is shared by both
    classes. `meta` (`_class_meta` of the same arguments), when given, is
    used instead of computing it here."""
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

    # The kernel reads each tile's query rows straight from the token-ordered
    # queries at an element offset (a gather of [tiles * block_q, D] cost
    # ~0.5 ms per layer on an 8K prefill, twin j-52e2df18). Rows past a
    # request's end read the next request's queries; their columns are never
    # gathered back. f32 so that any row offset is a whole 32-bit row.
    if max_num_seqs > _SEQ_MASK + 1:
        raise NotImplementedError(
            f"at most {_SEQ_MASK + 1} requests, got {max_num_seqs}"
        )
    if q_rows is None:
        q_rows = _query_rows(index_q, block_q)
    assert q_rows.shape[0] >= num_tokens + block_q, (q_rows.shape, block_q)
    if meta is None:
        meta = _class_meta(
            num_tokens,
            page_size,
            seq_lens,
            block_tables,
            cu_q_lens,
            seq_in_class,
            block_q=block_q,
            max_tiles=max_tiles,
            pages_per_step=pages_per_step,
        )
    tile_info, work, seq_pages = meta.tile_info, meta.work, meta.seq_pages

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
        grid=(jnp.maximum(meta.num_work, 1),),
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
        ring_slots = _RING3_SLOTS if DMA_RING3 else 2
        scores_t = pl.pallas_call(
            functools.partial(_kernel_v3, transposed=token_major),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                grid=grid_spec.grid,
                in_specs=[grid_spec.in_specs[0], pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=out_specs,
                scratch_shapes=[
                    pltpu.VMEM(
                        (ring_slots, pages_per_step, rows_per_page, head_dim),
                        jnp.uint32,
                    ),
                    pltpu.SemaphoreType.DMA((ring_slots,)),
                ],
            ),
            out_shape=out_shape,
            compiler_params=params,
            interpret=interpret,
            name="minimax_m3_msa_block_scores",
        )(work, tile_info, block_tables, seq_pages, q_rows, index_cache)
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

    return ClassScores(
        scores_t,
        meta.col,
        meta.tile_seq,
        meta.tile_row0,
        meta.num_tiles,
        meta.first_tile,
        meta.tile_start,
    )


class _ClassPlan(NamedTuple):
    """`block_scores`' two tile classes. `small()` / `large()` score them
    (`ClassScores` each; ``meta=`` takes the class's precomputed `ClassMeta`);
    `large_meta()` is the large class's `ClassMeta` alone (``onehot=`` as
    `_class_meta`); `token_major` is the small class's layout."""

    small: Callable
    large: Callable
    large_meta: Callable
    small_block_q: int
    large_block_q: int
    large_pages: int
    max_large: int
    is_large: jax.Array
    token_major: bool


def is_prefill_step(num_tokens: int, max_num_seqs: int) -> bool:
    """More rows than a batch of decode requests: `block_scores` scores both
    tile classes unconditionally."""
    return num_tokens > max_num_seqs * _SMALL_TILE_Q


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
    folded_tile_q: int = _SMALL_TILE_Q_FOLDED,
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
    folded = msa_fp4.is_fp4(index_cache) and scale_row is None
    # With `FP8_TRANSPOSED`, fp8 tiles take the transposed tile size too.
    folded = folded or (scale_row is None and _fp8_transposed(1, index_cache.dtype))
    tile = (
        min(
            block_q,
            folded_tile_q if folded else _SMALL_TILE_Q,
            -(-num_tokens // _ROW_ALIGN) * _ROW_ALIGN,
        )
        if token_major
        else block_q
    )
    # Aligned tile starts add up to _ROW_ALIGN - 1 rows per request.
    tiles_per_request = -(-(block_q + _ROW_ALIGN - 1) // tile)
    # A prefill scores both classes: one padded copy of the queries for both
    # (it paid the f32 cast and pad of [num_tokens, D] once per class). A
    # decode step runs the large class only under a cond, so it keeps the
    # small class's own short pad.
    q_rows = (
        _query_rows(index_q, max(tile, large_block_q))
        if is_prefill_step(num_tokens, max_num_seqs)
        else None
    )
    small = functools.partial(
        score,
        active & ~is_large,
        q_rows=q_rows,
        block_q=tile,
        max_tiles=min(
            tiles_per_request * max_num_seqs,
            (num_tokens + (_ROW_ALIGN - 1) * max_num_seqs) // tile + max_num_seqs,
        ),
        pages_per_step=small_pages_per_step,
        kernel=small_kernel,
        token_major=token_major,
    )
    large_tiles = dict(
        block_q=large_block_q,
        # +1 per request for the partial last tile, plus the up-to-7 rows
        # each aligned start adds.
        max_tiles=num_tokens // large_block_q
        + max_large
        + -(-(_ROW_ALIGN - 1) * max_large // large_block_q)
        + 1,
        pages_per_step=pages_per_step,
    )
    large = functools.partial(
        score, is_large, q_rows=q_rows, kernel=large_kernel, **large_tiles
    )
    large_meta = functools.partial(
        _class_meta,
        num_tokens,
        _page_tokens(index_cache.shape[1], index_cache.dtype),
        seq_lens,
        block_tables,
        cu_q_lens,
        is_large,
        **large_tiles,
    )
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    return _ClassPlan(
        small=small,
        large=large,
        large_meta=large_meta,
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
        "folded_tile_q",
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
    folded_tile_q: int = _SMALL_TILE_Q_FOLDED,
    interpret: bool = False,
) -> jax.Array:
    """Block-max index scores ``[T, 1, P]`` (``P`` padded to the page group).

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
        folded_tile_q=folded_tile_q,
        interpret=interpret,
    )
    num_tokens = index_q.shape[0]
    max_num_seqs = seq_lens.shape[0]
    max_large, is_large, token_major = plan.max_large, plan.is_large, plan.token_major
    small, small_col, *_ = plan.small()
    # The classes may pad pages to different group multiples; pad to the wider.
    pages = max(small.shape[1 if token_major else 0], plan.large_pages)

    def small_only():
        rows = small[small_col] if token_major else small[:, small_col].T
        return jnp.pad(rows, ((0, 0), (0, pages - rows.shape[1])))

    if max_large == 0:
        return small_only()[:, None, :]

    # Columns of the small class's [P, columns] layout (`with_large`).
    small_cols = small.shape[0 if token_major else 1]
    row_gather = ROW_GATHER and token_major

    def row_columns(large_col, onehot=None):
        """Each row's large-class flag and its column in the concatenated
        ``[P, small | large columns]`` layout."""
        tok = jnp.arange(num_tokens, dtype=jnp.int32)
        seq = jnp.minimum(
            jnp.searchsorted(
                cu_q_lens[1:], tok, side="right", method="compare_all"
            ).astype(jnp.int32),
            max_num_seqs - 1,
        )
        row_large = lookup(is_large, seq, onehot)
        return row_large, jnp.where(row_large, small_cols + large_col, small_col)

    def gather_rows(large, large_col, row_large, col):
        if row_gather:
            # Token-major small scores: a row gather (as `small_only`), and a
            # column gather of the large class alone; no [P, columns] copy.
            rows_small = small[small_col]
            rows_large = large[:, large_col].T
            rows_small, rows_large = (
                jnp.pad(x, ((0, 0), (0, pages - x.shape[1])))
                for x in (rows_small, rows_large)
            )
            return jnp.where(row_large[:, None], rows_large, rows_small)
        # One gather over both classes' columns (XLA fuses the concatenate
        # into it) instead of a transposing gather per class and a select.
        small_t = small.T if token_major else small
        small_t, large = (
            jnp.pad(x, ((0, pages - x.shape[0]), (0, 0))) for x in (small_t, large)
        )
        return jnp.concatenate([small_t, large], axis=1)[:, col].T

    def with_large():
        large, large_col, *_ = plan.large()
        return gather_rows(large, large_col, *row_columns(large_col))

    if is_prefill_step(num_tokens, max_num_seqs):
        # More rows than a batch of decode requests: a prefill step. A
        # conditional would wait for the gather (offloaded to the SparseCore)
        # and its neighbours' collectives at its end (c4 mixed twin +50 us
        # per layer, j-08c5e333).
        return with_large()[:, None, :]
    large_branch = with_large
    if HOIST_METADATA:
        # The large class's metadata and the rows' columns outside the cond:
        # the same in every layer, so XLA computes them once per step (a
        # branch is opaque to CSE). One-hot lookups keep the step's one copy
        # cheap where no request is large.
        large_meta = plan.large_meta(onehot=True)
        row_large, col = row_columns(large_meta.col, onehot=True)

        def large_branch():
            large = plan.large(meta=large_meta).scores
            return gather_rows(large, large_meta.col, row_large, col)

    # Decode steps have no large request: skip its kernel call and gather.
    return jax.lax.cond(jnp.any(is_large), large_branch, small_only)[:, None, :]
