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

Scores are produced transposed, ``[pages, tiles * block_q]``, so each output
block is lane-aligned, and gathered back to ``[tokens, pages]``.
"""

import functools

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
# The block_q class (decode, MTP verify) runs the v3 DMA ring; 128 pages per
# step keep more bytes in flight (j-dba77583: 32 req x 8 tokens at 128K,
# 1.95 -> 1.66 ms). The large prefill class keeps 32.
DEFAULT_SMALL_PAGES_PER_STEP = 128
DEFAULT_KERNEL = "v3"
# Large prefill tiles: v3's DMA ring is slower there on fp8 (1.79 vs 1.51 ms).
DEFAULT_LARGE_KERNEL = "v2"
_TILE_SHIFT = 16
# tile_info = query row of the tile << _ROW_SHIFT | request of the tile
_ROW_SHIFT = 10
_SEQ_MASK = (1 << _ROW_SHIFT) - 1
# f32 query rows are read at element offsets that are multiples of the sublane tile.
_ROW_ALIGN = 8
_GROUP_MASK = (1 << _TILE_SHIFT) - 1
# Packed tile class (`block_scores` packed_rows): requests of at most
# PACKED_Q_LEN query tokens per index head (decode / MTP verify: 1 + 7 DSpark
# drafts), all heads folded. Only 8-row tiles (16 pages per dot) are used:
# with 4 heads (32-row tiles, 4 pages per dot) packing did not beat the
# 128-row tile (TPU bench j-5f208314, DCP8 64 requests 702 vs 697 us).
PACKED_Q_LEN = 8
PACKED_MAX_ROWS = 8


def packed_rows(num_heads: int) -> int | None:
    """`block_scores` packed_rows for `num_heads` folded index heads, or None
    (packing off)."""
    rows = max(8, num_heads * PACKED_Q_LEN)
    rows = 1 << (rows - 1).bit_length()
    return rows if rows <= PACKED_MAX_ROWS else None


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
# v3 page DMAs: "group" issues them per DMA_GROUP pages (one branch per
# group; inside a live group every page is fetched, those past the request's
# end as a copy of its last page) and waits without re-reading the block
# table; "v29" (one branch and one block-table read per page for both the
# start and the wait) is the benchmark reference. DCP8 64-request decode:
# 706 -> 635 us, same scores (j-d702723e).
DMA_MODE = "group"
DMA_GROUP = 32
# Pages per dot of a packed step (`_scores_v3` pack > 1).
_PACKED_PAGES_PER_DOT = 32


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


def _scale_rows_to_column(rows: list[jax.Array], live=None) -> jax.Array:
    """The group's ``[1, S]`` slot-scale rows -> one ``[n * S, 1]`` column in
    the score rows' (page-major) order, non-finite read as 0. One transpose
    of the stacked ``[n, S]`` rows per dot group: a ``[1, S] -> [S, 1]``
    transpose per page made the 4-bit indexer 50% slower than fp8."""
    stacked = rows[0] if len(rows) == 1 else jnp.concatenate(rows, axis=0)  # [n, S]
    stacked = jnp.where(jnp.isfinite(stacked), stacked, 0.0)
    if live is not None:
        # Packed dots (`_scores_v3`): a page's scaled keys meet the other
        # pages' zero query blocks, so no key may be inf (a scale that rounds
        # to bf16 inf: 0 * inf = NaN) and pages past the request's end (not
        # fetched: any bits) are zeroed. Only scores nobody reads change.
        page = jax.lax.broadcasted_iota(jnp.int32, stacked.shape, 0)
        rounded = stacked.astype(jnp.bfloat16).astype(jnp.float32)
        keep = jnp.isfinite(rounded) & (page < live)
        stacked = jnp.where(keep, stacked, 0.0)
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


def _scores_v3(
    q,
    page_refs,
    o_ref,
    scale_refs=None,
    scale_row=None,
    live=None,
    pack=1,
    dma_group=None,
):
    """`_kernel_v2`'s scoring over pages already in VMEM. With `live` (the
    step's pages up to the request's last), a dot group of pages all past it
    is not scored: its scores are never read, and it is stored as -inf.

    With ``pack > 1``, `q` is the block-diagonal ``[128, pack * D]`` query
    matrix of `_block_diagonal`: `pack` pages share one dot row (their keys
    side by side along the contraction), so the ``[tokens, 128]`` product
    carries `pack` pages x ``128 / pack`` query rows per lane row instead of
    one page x 128 rows, and a step pops and max-reduces `pack` times fewer
    score vregs. Each lane only meets its own page's keys (the other
    blocks are exact zeros), so its score is the same dot. The output is
    ``[pages / pack, 128]``, lane ``b * 128 / pack + r`` = page ``b`` of the
    row's `pack`, query row ``r``."""
    block_q = q.shape[0]
    pages_per_step = len(page_refs)
    rows = page_refs[0].shape[0]
    page_size = _page_tokens(rows, page_refs[0].dtype)
    if pack > 1:
        # The product is [pages / pack * S, 128]; dead groups stay as fine
        # as the unpacked kernel's.
        pages_per_dot = min(pages_per_step, max(pack, _PACKED_PAGES_PER_DOT))
    else:
        pages_per_dot = max(
            1, min(pages_per_step, _MAX_SCORE_ELEMS // (page_size * block_q))
        )
    folded = msa_fp4.scales_in_page(rows, page_refs[0].dtype)
    groups = range(0, pages_per_step, pages_per_dot)
    if pack > 1 and (
        pages_per_step % pack
        or pages_per_dot % pack
        or scale_row is not None
        or (folded and not SCALE_KEYS)
    ):
        raise NotImplementedError("packed scoring needs whole packs, folded scales")

    def group_scores(first):
        group = page_refs[first : first + pages_per_dot]
        keys, cols = [], []
        for r in group:
            words = r.bitcast(jnp.uint32)[:, 0, :]
            if folded:
                k, c = _folded_keys(words)
                keys.append(k)
                cols.append(c)
            else:
                keys.append(_words_keys(words, r.dtype))
        # Grouped DMAs (`dma_group` pages per group) fetch every page of a
        # dot group that lies inside one DMA group.
        dead_fetched = dma_group is not None and dma_group % pages_per_dot == 0
        zero_dead = pack > 1 and live is not None and not dead_fetched
        if zero_dead and not folded:
            # Pages past the request's end are not fetched: their VMEM may
            # hold NaN bit patterns (fp8 / bf16), which a zero query block
            # would spread to the pack's live pages. 4-bit keys are always
            # finite (and their scales are sanitised).
            keys = [
                jnp.where(first + g < live, k, jnp.zeros_like(k))
                for g, k in enumerate(keys)
            ]
        keys = keys[0] if len(keys) == 1 else jnp.concatenate(keys, axis=0)
        packed_live = live - first if pack > 1 else None
        col = _scale_rows_to_column(cols, packed_live) if folded else None
        if folded and SCALE_KEYS:
            keys = keys * col.astype(jnp.bfloat16)
        n = len(group)
        if pack > 1:
            # [n * S, D] -> [n / pack * S, pack * D]: page j * pack + b's keys
            # in rows [j * S, (j + 1) * S), lanes [b * D, (b + 1) * D).
            keys = jnp.concatenate(
                [
                    jnp.concatenate(
                        [
                            keys[
                                (j * pack + b) * page_size : (j * pack + b + 1)
                                * page_size
                            ]
                            for b in range(pack)
                        ],
                        axis=1,
                    )
                    for j in range(n // pack)
                ],
                axis=0,
            )
        s = jax.lax.dot_general(
            keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        if pack > 1:
            return jnp.max(s.reshape(n // pack, page_size, block_q), axis=1)
        if folded and not SCALE_KEYS:
            s = s * col
        elif scale_row is not None:
            cols = [
                _scale_column(r[...], scale_row)
                for r in scale_refs[first : first + pages_per_dot]
            ]
            s = s * (cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=0))
        return jnp.max(s.reshape(n, page_size, block_q), axis=1)

    if live is None:
        maxima = [group_scores(first) for first in groups]
        o_ref[...] = maxima[0] if len(maxima) == 1 else jnp.concatenate(maxima, axis=0)
        return
    for first in groups:
        size = min(pages_per_dot, pages_per_step - first) // pack
        rows_out = pl.ds(first // pack, size)

        def store(first=first, rows_out=rows_out):
            o_ref[rows_out, :] = group_scores(first)

        if first == 0:  # every step has a live page
            store()
            continue
        pl.when(first < live)(store)

        @pl.when(first >= live)
        def _(rows_out=rows_out, size=size):
            o_ref[rows_out, :] = jnp.full((size, block_q), -jnp.inf, jnp.float32)


def _block_diagonal(q: jax.Array, pack: int) -> jax.Array:
    """``[R, D]`` f32 query rows -> bf16 ``[pack * R, pack * D]`` with `q` in
    the diagonal blocks and exact zeros elsewhere (built in f32: an
    ``R = 8``-row bf16 block is half a packed sublane tile)."""
    rows, head_dim = q.shape
    zeros = lambda n: jnp.zeros((rows, n * head_dim), jnp.float32)  # noqa: E731
    blocks = []
    for b in range(pack):
        parts = (
            ([zeros(b)] if b else [])
            + [q]
            + ([zeros(pack - 1 - b)] if b < pack - 1 else [])
        )
        blocks.append(parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1))
    return jnp.concatenate(blocks, axis=0).astype(jnp.bfloat16)


def _per_group(fn, pages_per_step: int, live) -> None:
    """``fn(g0)`` for every DMA group of a step that holds a live page (the
    first always does)."""
    for g0 in range(0, pages_per_step, DMA_GROUP):
        if g0 == 0:
            fn(g0)
        else:
            pl.when(g0 < live)(functools.partial(fn, g0))


def _kernel_v3(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    seq_pages_ref,  # [max_num_seqs] pages of each request
    # In
    q_ref,  # [Bq, D]
    cache_ref,  # [pages, S / p, p, D] index cache, left in HBM
    *refs,
    scale_row: int | None = None,
    skip_dead: bool = True,
    pack: int = 1,
    dma_mode: str = "v29",
):
    """`_kernel_v2` with the pages fetched by a hand-written double-buffered
    DMA ring: one scalar address per page and no copy for pages past the
    request's end (their scores are never read). With `skip_dead`, dot groups
    of such pages are not scored either (`_scores_v3`): a request's last
    group of 128 pages is mostly empty (34% of the slots at 64-request decode,
    j-5cc99100).

    refs: without `scale_row`, ``o_ref [G, Bq]``, then scratch ``buf0``,
    ``buf1`` (``[G, S / p, p, D]`` pages of even / odd steps) and DMA
    semaphores ``[2, G]``. With `scale_row` (4-bit caches) the scale cache
    ``[pages, 8, 128]`` (in HBM) comes first, and the scratch adds scale
    buffers ``[G, 8, 128]`` for even / odd steps and their semaphores.
    """
    if scale_row is None:
        o_ref, buf0, buf1, sems = refs
        scale_ref = sbuf0 = sbuf1 = ssems = None
    else:
        scale_ref, o_ref, buf0, buf1, sems, sbuf0, sbuf1, ssems = refs
    pages_per_step = buf0.shape[0]
    pages_per_seq = block_tables_ref.shape[0] // seq_pages_ref.shape[0]
    step, num_steps = pl.program_id(0), pl.num_programs(0)

    def live_pages(w):
        """Request, first page and live pages (at least 1) of work item `w`."""
        entry = work_ref[w]
        seq = tile_seq_ref[entry >> _TILE_SHIFT] & _SEQ_MASK
        first = (entry & _GROUP_MASK) * pages_per_step
        return seq, first, jnp.maximum(seq_pages_ref[seq] - first, 1)

    def copies(w, buf, slot, sbuf=None):
        seq, first, live = live_pages(w)
        base = seq * pages_per_seq + first
        out = []
        for g in range(pages_per_step):
            col = jnp.minimum(first + g, pages_per_seq - 1) - first
            page = block_tables_ref[base + col]
            out.append(
                (
                    g < live,
                    pltpu.make_async_copy(
                        cache_ref.at[page], buf.at[g], sems.at[slot, g]
                    ),
                )
            )
            if sbuf is not None:
                out.append(
                    (
                        g < live,
                        pltpu.make_async_copy(
                            scale_ref.at[page], sbuf.at[g], ssems.at[slot, g]
                        ),
                    )
                )
        return out

    bufs = ((buf0, sbuf0), (buf1, sbuf1))

    def start(w, slot):
        if dma_mode == "v29":
            for live, copy in copies(w, bufs[slot][0], slot, bufs[slot][1]):
                pl.when(live)(copy.start)
            return
        seq, first, live = live_pages(w)
        base = seq * pages_per_seq + first
        buf, sbuf = bufs[slot]

        def group(g0):
            for g in range(g0, min(g0 + DMA_GROUP, pages_per_step)):
                page = block_tables_ref[base + jnp.minimum(g, live - 1)]
                pltpu.make_async_copy(
                    cache_ref.at[page], buf.at[g], sems.at[slot, g]
                ).start()
                if sbuf is not None:
                    pltpu.make_async_copy(
                        scale_ref.at[page], sbuf.at[g], ssems.at[slot, g]
                    ).start()

        _per_group(group, pages_per_step, live)

    def wait(w, slot):
        if dma_mode == "v29":
            for live, copy in copies(w, bufs[slot][0], slot, bufs[slot][1]):
                pl.when(live)(copy.wait)
            return
        # A wait needs only the destination and the semaphore.
        live = live_pages(w)[2]
        buf, sbuf = bufs[slot]

        def group(g0):
            for g in range(g0, min(g0 + DMA_GROUP, pages_per_step)):
                pltpu.make_async_copy(
                    cache_ref.at[0], buf.at[g], sems.at[slot, g]
                ).wait()
                if sbuf is not None:
                    pltpu.make_async_copy(
                        scale_ref.at[0], sbuf.at[g], ssems.at[slot, g]
                    ).wait()

        _per_group(group, pages_per_step, live)

    @pl.when(step == 0)
    def _():
        start(step, 0)

    even = step % 2 == 0

    @pl.when(step + 1 < num_steps)
    def _():
        @pl.when(even)
        def _():
            start(step + 1, 1)

        @pl.when(jnp.logical_not(even))
        def _():
            start(step + 1, 0)

    if pack > 1:  # q_ref holds one request's 128 / pack rows
        q = _block_diagonal(q_ref[...].astype(jnp.float32), pack)
    else:
        q = q_ref[...].astype(jnp.bfloat16)

    def score(slot):
        buf, sbuf = bufs[slot]
        wait(step, slot)
        _scores_v3(
            q,
            [buf.at[g] for g in range(pages_per_step)],
            o_ref,
            None if sbuf is None else [sbuf.at[g] for g in range(pages_per_step)],
            scale_row,
            live_pages(step)[2] if skip_dead or pack > 1 else None,
            pack,
            DMA_GROUP if dma_mode == "group" else None,
        )

    pl.when(even)(lambda: score(0))
    pl.when(jnp.logical_not(even))(lambda: score(1))


_KERNELS = {"v1": _kernel_v1, "v2": _kernel_v2}


def _tiles(
    cu_q_lens: jax.Array,
    seq_in_class: jax.Array,
    block_q: int,
    max_tiles: int,
    align: int = _ROW_ALIGN,
):
    """Query tiles of one tile class.

    A request's tiles start at its first token rounded down to a multiple of
    `_ROW_ALIGN` (the kernel reads query rows at aligned element offsets), so
    a tile may begin with up to `_ROW_ALIGN - 1` rows of the previous request
    and end with rows of the next; those rows' scores are never read.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    start = (cu_q_lens[:-1] // align) * align
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
    tile_row0 = start[tile_seq] + (tile - first_tile[tile_seq]) * block_q
    return tile_seq, tile_row0, first_tile, start, tile_ends[-1]


def _packed_query_rows(tile_row0: jax.Array, rows: int) -> jax.Array:
    """``[tiles, rows]`` query rows of each packed tile."""
    return tile_row0[:, None] + jnp.arange(rows, dtype=jnp.int32)


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
    scale_cache=None,
    scale_row=None,
    skip_dead=True,
    pack=1,
):
    """Transposed scores ``[P, tiles * block_q]`` of one tile class, and the
    column of each token.

    ``pack > 1`` (v3, ``block_q = 128 / pack``): `pack` pages share each dot
    (`_scores_v3`). A tile is exactly one request's rows (gathered, no
    aligned start), so every request of the class fits one tile."""
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

    if pack > 1 and (kernel != "v3" or fp4 or block_q * pack != 128):
        raise NotImplementedError("packed tiles: v3, folded scales, 128 lanes")
    tile_seq, tile_row0, first_tile, tile_start, num_tiles = _tiles(
        cu_q_lens, seq_in_class, block_q, max_tiles, align=1 if pack > 1 else _ROW_ALIGN
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
    if pack > 1:
        # [tiles * block_q, D]: tile i holds rows tile_row0[i] + [0, block_q)
        # (rows past the request's end are other requests' queries; their
        # lanes are never gathered back).
        rows = _packed_query_rows(tile_row0, block_q)
        q_rows = index_q[jnp.clip(rows, 0, num_tokens - 1).reshape(-1), 0].astype(
            jnp.float32
        )
        tile_info = tile_seq
    else:
        q_rows = jnp.pad(index_q[:, 0].astype(jnp.float32), ((0, block_q), (0, 0)))
        tile_info = (
            jnp.clip(tile_row0, 0, num_tokens // _ROW_ALIGN * _ROW_ALIGN) << _ROW_SHIFT
        ) | tile_seq

    # Work list: every page group of every tile's request.
    seq_pages = -(-seq_lens // page_size)
    groups_per_tile = jnp.where(
        jnp.arange(max_tiles) < num_tiles,
        jnp.maximum(-(-seq_pages[tile_seq] // pages_per_step), 1),
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
    work_group = w - (group_ends[work_tile] - groups_per_tile[work_tile])
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

    if pack > 1:
        q_spec = pl.BlockSpec(
            (block_q, head_dim), lambda w, work, *_: (work[w] >> _TILE_SHIFT, 0)
        )
        lanes = block_q * pack
    else:
        q_spec = pl.BlockSpec(
            (pl.Element(block_q), pl.Element(head_dim)),
            lambda w, work, tile_info, *_: (
                pl.multiple_of(
                    tile_info[work[w] >> _TILE_SHIFT] >> _ROW_SHIFT, _ROW_ALIGN
                ),
                0,
            ),
        )
        lanes = block_q
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        # One step even with no tile in the class; its scores are never read.
        grid=(jnp.maximum(group_ends[-1], 1),),
        in_specs=[
            q_spec,
            *(page_block(g) for g in range(pages_per_step)),
            *(scale_block(g) for g in range(pages_per_step) if fp4),
        ],
        out_specs=pl.BlockSpec(
            (pages_per_step // pack, lanes),
            lambda w, work, *_: (work[w] & _GROUP_MASK, work[w] >> _TILE_SHIFT),
        ),
    )
    out_shape = jax.ShapeDtypeStruct(
        (padded_pages // pack, max_tiles * lanes), jnp.float32
    )
    params = pltpu.CompilerParams(dimension_semantics=("arbitrary",))
    if kernel == "v3":
        page_buf = pltpu.VMEM(
            (pages_per_step, rows_per_page, packing, head_dim), index_cache.dtype
        )
        scratch = [page_buf, page_buf, pltpu.SemaphoreType.DMA((2, pages_per_step))]
        hbm = [pl.BlockSpec(memory_space=pl.ANY)]
        operands = [index_cache]
        if fp4:
            scale_buf = pltpu.VMEM(
                (pages_per_step, scale_cache.shape[1], msa_fp4.SCALE_LANES), jnp.float32
            )
            scratch += [
                scale_buf,
                scale_buf,
                pltpu.SemaphoreType.DMA((2, pages_per_step)),
            ]
            hbm.append(pl.BlockSpec(memory_space=pl.ANY))
            operands.append(scale_cache)
        scores_t = pl.pallas_call(
            functools.partial(
                _kernel_v3,
                scale_row=scale_row,
                skip_dead=skip_dead,
                pack=pack,
                dma_mode=DMA_MODE,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                grid=grid_spec.grid,
                in_specs=[grid_spec.in_specs[0], *hbm],
                out_specs=grid_spec.out_specs,
                scratch_shapes=scratch,
            ),
            out_shape=out_shape,
            compiler_params=params,
            interpret=interpret,
            name="minimax_m3_msa_block_scores",
        )(work, tile_info, block_tables, seq_pages, q_rows, *operands)
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

    if pack > 1:
        # [P / pack, tiles, pack, block_q] -> the [P, tiles * block_q] layout.
        scores_t = (
            scores_t.reshape(padded_pages // pack, max_tiles, pack, block_q)
            .transpose(0, 2, 1, 3)
            .reshape(padded_pages, max_tiles * block_q)
        )
    # Back to token order: token t of request r sits in r's tiles at its offset.
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], tok, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    offset = tok - tile_start[seq]
    col = (first_tile[seq] + offset // block_q) * block_q + offset % block_q
    col = jnp.clip(col, 0, max_tiles * block_q - 1)
    return scores_t, col


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
        "skip_dead",
        "packed_rows",
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
    skip_dead: bool = True,
    packed_rows: int | None = None,
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
        ``"v1"``, kept as the reference arm for benchmarks.
      large_kernel: the same choice for the `large_block_q` tiles.
      pages_per_step / small_pages_per_step: pages per grid step of the large
        and the block_q tile classes.
      scale_cache, scale_row: for a 4-bit (uint8) index cache without folded
        scales (`msa_fp4.scales_in_page`),
        the ``[pages, 8, 128]`` f32 scale cache and the row of the index-key
        scales. `index_q` must then be rotated (`msa_fp4.rotate_queries`).
      skip_dead: v3 skips scoring page groups past a request's last page,
        and the large tile class runs only when some request is large
        (``False``: the reference arm for benchmarks; same scores on every
        page a token can see).
      packed_rows: requests of at most this many query rows (8, 16, 32 or
        64; decode and MTP verify with all index heads folded) get their own
        tile class of `packed_rows`-row tiles in which ``128 / packed_rows``
        pages share each dot (`_scores_v3` ``pack``): the same scores with
        that many times fewer score vregs per page. Needs `skip_dead` and a
        bf16 / fp8 / folded 4-bit cache; ``None`` or 128: off.
    Pages past a token's own page hold scores of keys it must not see, and
    pages past its request's last page are not written; callers mask both.
    """
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
        skip_dead=skip_dead,
    )
    # Aligned tile starts can push a request of <= block_q tokens into a
    # second tile.
    small_tiles = min(2 * max_num_seqs, num_tokens + max_num_seqs)
    # Large requests have more than block_q tokens each.
    max_large = min(max_num_seqs, num_tokens // (block_q + 1))
    if packed_rows is not None and packed_rows not in (8, 16, 32, 64):
        raise ValueError(f"packed_rows must be 8, 16, 32 or 64, got {packed_rows}")
    packed = (
        packed_rows is not None
        and packed_rows < block_q
        and skip_dead
        and scale_row is None
        and kernel == "v3"
        and small_pages_per_step % (128 // packed_rows) == 0
        and (SCALE_KEYS or not msa_fp4.is_fp4(index_cache))
    )

    def unpacked(_):
        small, small_col = score(
            active & ~is_large,
            block_q=block_q,
            max_tiles=small_tiles,
            pages_per_step=small_pages_per_step,
            kernel=kernel,
        )
        if max_large == 0:
            return small[:, small_col].T[:, None, :]

        def with_large(small):
            large, large_col = score(
                is_large,
                block_q=large_block_q,
                max_tiles=_large_tiles(num_tokens, large_block_q, max_large),
                pages_per_step=pages_per_step,
                kernel=large_kernel,
            )
            tok = jnp.arange(num_tokens, dtype=jnp.int32)
            seq = jnp.minimum(
                jnp.searchsorted(
                    cu_q_lens[1:], tok, side="right", method="compare_all"
                ).astype(jnp.int32),
                max_num_seqs - 1,
            )
            # One gather over both classes' columns (XLA fuses the concatenate
            # into it) instead of a transposing gather per class and a select.
            col = jnp.where(is_large[seq], small.shape[1] + large_col, small_col)
            # The classes may pad pages to different group multiples; pad to the
            # wider.
            pages = max(small.shape[0], large.shape[0])
            small, large = (
                jnp.pad(x, ((0, pages - x.shape[0]), (0, 0))) for x in (small, large)
            )
            return jnp.concatenate([small, large], axis=1)[:, col].T[:, None, :]

        if not skip_dead:
            return with_large(small)
        pages_per_seq = block_tables.shape[0] // max_num_seqs
        pages = max(
            small.shape[0], -(-pages_per_seq // pages_per_step) * pages_per_step
        )

        def small_only(small):
            # No large request: every token's column is in the small class, so
            # this is the gather above without the (empty) large class.
            small = jnp.pad(small, ((0, pages - small.shape[0]), (0, 0)))
            return small[:, small_col].T[:, None, :]

        # Decode batches have no large request; the large class's 1-step launch,
        # its work list and the concatenate cost ~20 us per layer at 64-request
        # decode (j-5cc99100).
        return jax.lax.cond(jnp.any(is_large), with_large, small_only, small)

    if not packed:
        return unpacked(None)
    # Decode / verify steps (every request packable): one packed class.
    # Any other step runs the classes above unchanged.
    is_packed = active & (q_lens <= packed_rows)
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    steps = (small_pages_per_step,) + ((pages_per_step,) if max_large else ())
    pages = max(-(-pages_per_seq // step) * step for step in steps)

    def packed_only(_):
        scores_t, col = score(
            is_packed,
            block_q=packed_rows,
            max_tiles=min(max_num_seqs, num_tokens),
            pages_per_step=small_pages_per_step,
            kernel="v3",
            pack=128 // packed_rows,
        )
        scores_t = jnp.pad(scores_t, ((0, pages - scores_t.shape[0]), (0, 0)))
        return scores_t[:, col].T[:, None, :]

    return jax.lax.cond(jnp.all(is_packed | ~active), packed_only, unpacked, None)


def _large_tiles(num_tokens: int, large_block_q: int, max_large: int) -> int:
    # +1 per request for the partial last tile, plus the up-to-7 rows each
    # aligned start adds.
    return (
        num_tokens // large_block_q
        + max_large
        + -(-(_ROW_ALIGN - 1) * max_large // large_block_q)
        + 1
    )
