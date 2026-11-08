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
"""In-place row writes into the packed MSA caches.

An XLA scatter into the packed layout ``[pages, S / p, p, W]`` cannot index
the packed axis in place: XLA relayouts the whole cache to ``[rows, W]`` and
back around it, two full-cache copies per cache per layer. This kernel
instead rewrites only the pages that receive rows. A step of a token run
(consecutive tokens of one request landing in one page) reads the page as
32-bit words, merges the run's rows into their packing slots, and writes the
page back over the aliased cache.

The kernel fetches each run's rows from the batch's rows itself
(`_kernel_chunked`): the 8-slot chunks covering the run, copied into a
double-buffered page of rows; nothing is staged per run. Staging a page of rows per run with an XLA gather
first (`write_rows_staged`, kept as the reference) gathered every one of
`max_runs` pages, live or not: 110 us per layer for the index keys and 130
us for the K/V rows at 64-request decode (j-5cc99100), most of it a 1-D
gather of the folded scales.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

_UINT_BY_BITS = {16: jnp.uint16, 8: jnp.uint8}


def _kernel(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run (padding runs repeat the last one)
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    # In
    new_ref,  # [1, p, S / p, W] uint32: slot r * p + j of the run's page at [j, r]
    page_ref,  # [1, S / p, p, W] the page before the write
    # Out
    o_ref,  # [1, S / p, p, W] aliased with the cache
    *refs,  # with folded scales: [1, 1, W] uint32 scale bits, then o_ref
    bits: int,
    scale_lanes: int = 0,
):
    scale_ref = None
    if refs:  # folded scales: the scale input precedes the output
        scale_ref, o_ref = o_ref, refs[0]
    run = pl.program_id(0)
    active = run < num_runs_ref[0]

    @pl.when(active)
    def _():
        rows, _, width = page_ref.shape[1:]
        packing = 32 // bits
        words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
        lo, hi = run_lo_ref[run], run_hi_ref[run]
        row = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 0)
        for j in range(packing):
            slot = row * packing + j
            take = (slot >= lo) & (slot < hi)
            keep = jnp.uint32(~(((1 << bits) - 1) << (bits * j)) & 0xFFFFFFFF)
            merged = (words & keep) | (new_ref[0, j] << jnp.uint32(bits * j))
            words = jnp.where(take, merged, words)
        if scale_ref is not None:  # last word row: scale j of slot s at lane j * S + s
            page_size = (rows - 1) * packing
            lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
            slot = lane % page_size
            take = (
                (row == rows - 1) & (slot >= lo) & (slot < hi)
                & (lane < scale_lanes * page_size)
            )
            words = jnp.where(take, scale_ref[0], words)
        o_ref.bitcast(jnp.uint32)[0, :, 0, :] = words

    # With no run at all the single output page must still be written back
    # unchanged. Later padding steps revisit the last run's page and leave
    # its (already written) output block alone.
    @pl.when(jnp.logical_and(jnp.logical_not(active), run == 0))
    def _():
        o_ref[...] = page_ref[...]


def _merge_scales(words_row, scale_row, lo, hi, page_size, scale_lanes):
    """Folded scales: the page's last word row ``[1, W]`` with lane
    ``j * S + s`` (``j < L``, ``s`` in ``[lo, hi)``) taken from `scale_row`."""
    lane = jax.lax.broadcasted_iota(jnp.int32, words_row.shape, 1)
    slot = lane % page_size
    take = (slot >= lo) & (slot < hi) & (lane < scale_lanes * page_size)
    return jnp.where(take, scale_row, words_row)


_LANES = 128

# Slots per DMA of `_kernel_chunked`: a decode run of up to 8 tokens moves one
# or two chunks rather than a page of rows.
_CHUNK = 8


def _kernel_chunked(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run (padding runs repeat the last one)
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    run_src_ref,  # [R] row of `rows_hbm` that slot 0 of the run's page maps to
    # In
    rows_hbm,  # [W / 128, T + 2C, 4, 128] uint8: each token's slot words by lane tile, in HBM
    page_ref,  # [1, S / p, p, W] the page before the write
    *refs,
    bits: int,
    scale_lanes: int = 0,
):
    """`_kernel` that fetches each run's rows itself: the `_CHUNK`-slot chunks
    covering the run's slots, copied from the batch's rows into a page of rows
    ``[W / 128, S, 4, 128]`` (lane tile, slot order), double-buffered across
    runs. Lane tiles lead so that each strided slot load reads a 128-lane
    memref (Mosaic's strided load needs a 128-lane base).

    refs: without folded scales ``o_ref``, then scratch ``buf0``, ``buf1`` and
    DMA semaphores ``[2, 2]``. With folded scales the scale lanes
    ``[1, T + 2C, 4, 128]`` (in HBM, u32 lane ``j`` = scale ``j``) come first
    and the scratch adds their ``[1, S, 4, 128]`` buffers ``sbuf0``, ``sbuf1``.
    """
    if scale_lanes:
        scale_hbm, o_ref, buf0, buf1, sbuf0, sbuf1, sems = refs
    else:
        o_ref, buf0, buf1, sems = refs
        scale_hbm = sbuf0 = sbuf1 = None
    run = pl.program_id(0)
    num_runs = num_runs_ref[0]
    active = run < num_runs
    bufs = ((buf0, sbuf0), (buf1, sbuf1))

    def for_chunks(r, slot, method):
        buf, sbuf = bufs[slot]
        lo, hi, src = run_lo_ref[r], run_hi_ref[r], run_src_ref[r]

        def body(b, carry):
            first = b * _CHUNK
            copies = [
                pltpu.make_async_copy(
                    rows_hbm.at[:, pl.ds(src + first, _CHUNK)],
                    buf.at[:, pl.ds(first, _CHUNK)],
                    sems.at[slot, 0],
                )
            ]
            if sbuf is not None:
                copies.append(
                    pltpu.make_async_copy(
                        scale_hbm.at[:, pl.ds(src + first, _CHUNK)],
                        sbuf.at[:, pl.ds(first, _CHUNK)],
                        sems.at[slot, 1],
                    )
                )
            for copy in copies:
                getattr(copy, method)()
            return carry

        jax.lax.fori_loop(lo // _CHUNK, (hi + _CHUNK - 1) // _CHUNK, body, 0)

    even = run % 2 == 0

    @pl.when(jnp.logical_and(active, run == 0))
    def _():
        for_chunks(run, 0, "start")

    @pl.when(run + 1 < num_runs)
    def _():
        pl.when(even)(lambda: for_chunks(run + 1, 1, "start"))
        pl.when(jnp.logical_not(even))(lambda: for_chunks(run + 1, 0, "start"))

    def merge(slot):
        buf, sbuf = bufs[slot]
        for_chunks(run, slot, "wait")
        rows, _, width = page_ref.shape[1:]
        packing = 32 // bits
        data_rows = rows - 1 if scale_lanes else rows
        page_size = data_rows * packing
        lo, hi = run_lo_ref[run], run_hi_ref[run]
        page = page_ref.bitcast(jnp.uint32)
        out = o_ref.bitcast(jnp.uint32)
        row = jax.lax.broadcasted_iota(jnp.int32, (data_rows, _LANES), 0)
        for c in range(width // _LANES):
            lanes = pl.ds(c * _LANES, _LANES)
            new = buf.at[c].bitcast(jnp.uint32).reshape(page_size, _LANES)
            words = page[0, :data_rows, 0, lanes]
            for j in range(packing):
                # Slot r * p + j of word row r is row r * p + j of the buffer.
                slot = row * packing + j
                take = (slot >= lo) & (slot < hi)
                keep = jnp.uint32(~(((1 << bits) - 1) << (bits * j)) & 0xFFFFFFFF)
                new_j = new[pl.ds(j, data_rows, stride=packing), :]
                merged = (words & keep) | (new_j << jnp.uint32(bits * j))
                words = jnp.where(take, merged, words)
            out[0, :data_rows, 0, lanes] = words
        if scale_lanes:
            # [S, 128] lanes (scale j at lane j) -> [128, S]: row j holds
            # scale j of every slot. Moved as f32 bits, no arithmetic.
            by_slot = sbuf.at[0].bitcast(jnp.uint32).reshape(page_size, _LANES)
            by_scale = pltpu.bitcast(by_slot[...], jnp.float32).T
            by_scale = pltpu.bitcast(by_scale, jnp.uint32)
            parts = [by_scale[j : j + 1] for j in range(scale_lanes)]
            if scale_lanes * page_size < width:
                parts.append(
                    jnp.zeros((1, width - scale_lanes * page_size), jnp.uint32)
                )
            scale_row = parts[0] if len(parts) == 1 else jnp.concatenate(parts, 1)
            out[0, data_rows:, 0, :] = _merge_scales(
                page[0, data_rows:, 0, :], scale_row, lo, hi, page_size, scale_lanes
            )

    @pl.when(active)
    def _():
        pl.when(even)(lambda: merge(0))
        pl.when(jnp.logical_not(even))(lambda: merge(1))

    @pl.when(jnp.logical_and(jnp.logical_not(active), run == 0))
    def _():
        o_ref[...] = page_ref[...]


def _runs(pages: jax.Array, slots: jax.Array, num_pages: int, max_runs: int):
    """Token runs: maximal spans of consecutive tokens writing one page."""
    num_tokens = pages.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = (pages >= 0) & (pages < num_pages)
    prev_page = jnp.concatenate([jnp.full((1,), -1, pages.dtype), pages[:-1]])
    prev_slot = jnp.concatenate([jnp.full((1,), -2, slots.dtype), slots[:-1]])
    start = valid & ((pages != prev_page) | (slots != prev_slot + 1))
    run_id = jnp.cumsum(start.astype(jnp.int32)) - 1
    num_runs = jnp.minimum(jnp.sum(start.astype(jnp.int32)), max_runs)
    at_start = jnp.where(start, run_id, max_runs)
    run_page = jnp.zeros(max_runs, jnp.int32).at[at_start].set(pages, mode="drop")
    run_lo = jnp.zeros(max_runs, jnp.int32).at[at_start].set(slots, mode="drop")
    run_tok0 = jnp.zeros(max_runs, jnp.int32).at[at_start].set(tok, mode="drop")
    count = (
        jnp.zeros(max_runs, jnp.int32)
        .at[jnp.where(valid, run_id, max_runs)]
        .add(1, mode="drop")
    )
    last = run_page[jnp.maximum(num_runs - 1, 0)]
    padding = jnp.arange(max_runs) >= num_runs
    run_page = jnp.where(padding, last, run_page)
    return num_runs, run_page, run_lo, run_lo + count, run_tok0


@functools.partial(jax.jit, static_argnames=("max_runs", "bits", "interpret"))
def write_rows_staged(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    bits: int | None = None,
    scales: jax.Array | None = None,
    interpret: bool = False,
) -> jax.Array:
    """`write_rows` with each run's page of rows staged by an XLA gather
    first. Kept as the reference for `write_rows`.

    Tokens with a page outside the cache are dropped. Consecutive tokens of a
    run must take consecutive slots of one page, as a request's new tokens do;
    a run boundary is wherever that breaks. `max_runs` bounds the number of
    runs (see `max_write_runs`).

    `bits` per slot defaults to the cache dtype's width. With ``bits=4`` (the
    uint8 4-bit caches of `msa_fp4`) `rows` are uint8 values below 16, and
    each 32-bit word holds 8 slots. `scales` (f32 ``[T]`` or ``[T, L]``) go to
    the last word row of a page with folded scales (`msa_fp4.scales_in_page`):
    scale ``j`` of slot ``s`` at lane ``j * S + s`` (``L * S <= W``).
    """
    num_pages, rows_per_page, word_bytes, width = cache.shape
    if bits is None:
        bits = 32 // word_bytes
    packing = 32 // bits
    folded = scales is not None
    if folded and not (
        bits == 4 and msa_fp4.scales_in_page(rows_per_page, cache.dtype)
    ):
        raise ValueError(
            f"scales need a 4-bit cache with a scale row, got {cache.shape}"
        )
    data_rows = rows_per_page - 1 if folded else rows_per_page
    page_size = data_rows * packing
    num_tokens = rows.shape[0]
    num_runs, run_page, run_lo, run_hi, run_tok0 = _runs(
        pages, slots, num_pages, max_runs
    )
    # Each run's rows laid out by slot ([run, j, r] = slot r * p + j), so the
    # kernel merges whole packing slots without strided loads.
    if bits == 4:
        word_bits = rows.astype(jnp.uint32) & 0xF
    else:
        word_bits = jax.lax.bitcast_convert_type(rows, _UINT_BY_BITS[bits]).astype(
            jnp.uint32
        )
    src = run_tok0[:, None] + jnp.arange(page_size, dtype=jnp.int32) - run_lo[:, None]
    src = jnp.clip(src, 0, num_tokens - 1)
    new = word_bits[src]
    new = new.reshape(max_runs, data_rows, packing, width).transpose(0, 2, 1, 3)
    extra = []
    scale_lanes = 0
    if folded:
        new = jnp.pad(new, ((0, 0), (0, 0), (0, rows_per_page - data_rows), (0, 0)))
        scales2 = scales.reshape(num_tokens, -1)
        scale_lanes = scales2.shape[1]
        if scale_lanes * page_size > width:
            raise ValueError(f"{scale_lanes} scales per slot do not fit {width} lanes")
        bits_by_slot = jax.lax.bitcast_convert_type(
            scales2.astype(jnp.float32), jnp.uint32
        )[src]  # [runs, S, L]
        bits_by_slot = bits_by_slot.transpose(0, 2, 1).reshape(max_runs, -1)
        extra = [
            jnp.pad(bits_by_slot, ((0, 0), (0, width - scale_lanes * page_size)))[
                :, None, :
            ]
        ]

    page_block = pl.BlockSpec(
        (1, rows_per_page, word_bytes, width),
        lambda r, n, run_page, *_: (run_page[r], 0, 0, 0),
    )
    return pl.pallas_call(
        functools.partial(_kernel, bits=bits, scale_lanes=scale_lanes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(max_runs,),
            in_specs=[
                pl.BlockSpec(
                    (1, packing, rows_per_page, width), lambda r, *_: (r, 0, 0, 0)
                ),
                page_block,
                *(
                    [pl.BlockSpec((1, 1, width), lambda r, *_: (r, 0, 0))]
                    if folded
                    else []
                ),
            ],
            out_specs=page_block,
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        # Operand 5 = the cache (after 4 scalar-prefetch operands and `new`).
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_write_rows",
    )(num_runs[None], run_page, run_lo, run_hi, new, cache, *extra)


def _word_bits(rows: jax.Array, bits: int) -> jax.Array:
    """``rows [T, W]`` -> uint32 ``[T, W]``, one slot value per word."""
    if bits == 4:
        return rows.astype(jnp.uint32) & 0xF
    return jax.lax.bitcast_convert_type(rows, _UINT_BY_BITS[bits]).astype(jnp.uint32)


def _token_words(words: jax.Array) -> jax.Array:
    """uint32 ``[T, W]`` -> uint8 ``[W / 128, T + 2C, 4, 128]``: each token's
    words by 128-lane tile in the cache's byte layout, so a token is an
    untiled index (any row offset can be copied), with `_CHUNK` zero rows
    before and after."""
    num_tokens, width = words.shape
    words = jnp.pad(words, ((_CHUNK, _CHUNK), (0, 0)))
    words = words.reshape(num_tokens + 2 * _CHUNK, width // _LANES, _LANES)
    return jax.lax.bitcast_convert_type(words, jnp.uint8).transpose(1, 0, 3, 2)


@functools.partial(jax.jit, static_argnames=("max_runs", "bits", "interpret"))
def write_rows(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    bits: int | None = None,
    scales: jax.Array | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Write ``rows [T, W]`` (cache dtype) at ``(pages[t], slots[t])``.

    Tokens with a page outside the cache are dropped. Consecutive tokens of a
    run must take consecutive slots of one page, as a request's new tokens do;
    a run boundary is wherever that breaks. `max_runs` bounds the number of
    runs (see `max_write_runs`).

    `bits` per slot defaults to the cache dtype's width. With ``bits=4`` (the
    uint8 4-bit caches of `msa_fp4`) `rows` are uint8 values below 16, and
    each 32-bit word holds 8 slots. `scales` (f32 ``[T]`` or ``[T, L]``) go to
    the last word row of a page with folded scales (`msa_fp4.scales_in_page`):
    scale ``j`` of slot ``s`` at lane ``j * S + s`` (``L * S <= W``).
    """
    num_pages, rows_per_page, word_bytes, width = cache.shape
    if bits is None:
        bits = 32 // word_bytes
    packing = 32 // bits
    folded = scales is not None
    if folded and not (
        bits == 4 and msa_fp4.scales_in_page(rows_per_page, cache.dtype)
    ):
        raise ValueError(
            f"scales need a 4-bit cache with a scale row, got {cache.shape}"
        )
    data_rows = rows_per_page - 1 if folded else rows_per_page
    page_size = data_rows * packing
    if page_size % _CHUNK or width % _LANES:
        raise NotImplementedError(f"pages of {page_size} slots, {width} lanes")
    num_runs, run_page, run_lo, run_hi, run_tok0 = _runs(
        pages, slots, num_pages, max_runs
    )
    # Slot 0 of a run's page maps to token tok0 - lo, row + _CHUNK of the
    # padded rows; the chunks covering [lo, hi) stay inside the padding.
    run_src = run_tok0 - run_lo + _CHUNK
    token_words = _token_words(_word_bits(rows, bits))
    scratch = [pltpu.VMEM((width // _LANES, page_size, 4, _LANES), jnp.uint8)] * 2
    extra = []
    scale_lanes = 0
    if folded:
        scales2 = scales.reshape(rows.shape[0], -1)
        scale_lanes = scales2.shape[1]
        if scale_lanes * page_size > width:
            raise ValueError(f"{scale_lanes} scales per slot do not fit {width} lanes")
        if page_size % msa_fp4.SCALE_LANES or scale_lanes > msa_fp4.SCALE_LANES:
            raise NotImplementedError(
                f"folded scales of {page_size}-slot pages, {scale_lanes} per slot"
            )
        scale_bits = jax.lax.bitcast_convert_type(
            scales2.astype(jnp.float32), jnp.uint32
        )
        scale_bits = jnp.pad(
            scale_bits, ((0, 0), (0, msa_fp4.SCALE_LANES - scale_lanes))
        )
        extra = [_token_words(scale_bits)]
        scratch += [pltpu.VMEM((1, page_size, 4, _LANES), jnp.uint8)] * 2
    scratch.append(pltpu.SemaphoreType.DMA((2, 2)))

    page_block = pl.BlockSpec(
        (1, rows_per_page, word_bytes, width),
        lambda r, n, run_page, *_: (run_page[r], 0, 0, 0),
    )
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        functools.partial(_kernel_chunked, bits=bits, scale_lanes=scale_lanes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(max_runs,),
            in_specs=[hbm, page_block, *([hbm] if folded else [])],
            out_specs=page_block,
            scratch_shapes=scratch,
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        # Operand 6 = the cache (after 5 scalar-prefetch operands and the rows).
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_write_rows",
    )(num_runs[None], run_page, run_lo, run_hi, run_src, token_words, cache, *extra)


def max_write_runs(num_tokens: int, max_num_seqs: int, page_size: int) -> int:
    """Runs of a batch: a request's q tokens span at most
    ``(q - 1) // page_size + 2`` pages."""
    return max(1, min(num_tokens, num_tokens // page_size + 2 * max_num_seqs))
