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
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

_UINT_BY_BITS = {16: jnp.uint16, 8: jnp.uint8}


def _strided_rows(ref, start, count: int, stride: int, folds: int) -> jax.Array:
    """Rows ``start, start + stride, ...`` (`count` of them) of a 32-bit
    ``[R, 128 * folds]`` array held as ``[R * folds, 128]`` in `ref` (row
    ``r``'s lanes ``[128 i, 128 (i + 1))`` at ref row ``r * folds + i``):
    Mosaic's strided loads need 128-lane rows, as in `ragged_paged_attention` v3."""
    return jnp.concatenate(
        [
            ref[pl.ds(start * folds + i, count, stride=stride * folds)]
            for i in range(folds)
        ],
        axis=1,
    )


def _kernel(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run (padding runs repeat the last one)
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    row0_ref,  # [R] row of the run page's slot 0 in `new_ref`'s array
    # In
    new_ref,  # [(S + 8) * W / 128, 128] uint32: the run page's slot s at row
    # row0 % 8 + s of the [S + 8, W] rows (`_strided_rows`)
    page_ref,  # [1, S / p, p, W] the page before the write
    # Out
    o_ref,  # [1, S / p, p, W] aliased with the cache
    *refs,  # with folded scales: [S + 8, lanes] uint32 scale bits of slot s
    # on every lane of row row0 % 8 + s, then o_ref
    bits: int,
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
        data_rows = rows - 1 if scale_ref is not None else rows
        off = row0_ref[run] % _ROW_TILE
        for j in range(packing):
            slot = row * packing + j
            take = (slot >= lo) & (slot < hi)
            keep = jnp.uint32(~(((1 << bits) - 1) << (bits * j)) & 0xFFFFFFFF)
            # Slots j, p + j, ...: rows of the run's page at stride p.
            new = _strided_rows(new_ref, off + j, data_rows, packing, width // 128)
            if data_rows < rows:
                new = jnp.pad(new, ((0, rows - data_rows), (0, 0)))
            merged = (words & keep) | (new << jnp.uint32(bits * j))
            words = jnp.where(take, merged, words)
        if scale_ref is not None:  # last word row: one f32 scale per slot (lane)
            lanes = scale_ref.shape[1]
            # [1, lanes]: slot s's bits at lane s.
            by_slot = scale_ref[pl.ds(off, lanes), :].T[:1]
            if width > lanes:
                by_slot = jnp.pad(by_slot, ((0, 0), (0, width - lanes)))
            lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
            take = (row == rows - 1) & (lane >= lo) & (lane < hi)
            words = jnp.where(take, by_slot[:, :width], words)
        o_ref.bitcast(jnp.uint32)[0, :, 0, :] = words

    # With no run at all the single output page must still be written back
    # unchanged. Later padding steps revisit the last run's page and leave
    # its (already written) output block alone.
    @pl.when(jnp.logical_and(jnp.logical_not(active), run == 0))
    def _():
        o_ref[...] = page_ref[...]


def _kernel_by_token(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run (padding runs repeat the last one)
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    run_tok0_ref,  # [R] first token of each run
    # In
    rows_ref,  # [T, W] uint32: every token's row, one value per word
    page_ref,  # [1, S / p, p, W] the page before the write
    # Out
    o_ref,  # [1, S / p, p, W] aliased with the cache
    *refs,  # with folded scales: [T, W] uint32 scale bits per token, then o_ref
    bits: int,
):
    """`_kernel` that merges a run's tokens one by one from the batch's rows,
    so no per-run page of rows is gathered first."""
    scale_ref = None
    if refs:  # folded scales: the scale input precedes the output
        scale_ref, o_ref = o_ref, refs[0]
    run = pl.program_id(0)
    active = run < num_runs_ref[0]

    @pl.when(active)
    def _():
        rows, _, width = page_ref.shape[1:]
        packing = 32 // bits
        lo, tok0 = run_lo_ref[run], run_tok0_ref[run]
        row = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)

        def put(i, words):
            slot = lo + i
            shift = (slot % packing * bits).astype(jnp.uint32)
            keep = ~(jnp.uint32((1 << bits) - 1) << shift)
            new = rows_ref[pl.ds(tok0 + i, 1), :]
            words = jnp.where(
                row == slot // packing, (words & keep) | (new << shift), words
            )
            if scale_ref is not None:  # last word row: the scale at lane = slot
                take = (row == rows - 1) & (lane == slot)
                words = jnp.where(take, scale_ref[pl.ds(tok0 + i, 1), :], words)
            return words

        words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
        words = jax.lax.fori_loop(0, run_hi_ref[run] - lo, put, words)
        o_ref.bitcast(jnp.uint32)[0, :, 0, :] = words

    @pl.when(jnp.logical_and(jnp.logical_not(active), run == 0))
    def _():
        o_ref[...] = page_ref[...]


# Steps of at most this many tokens (decode, MTP verify, small prefill chunks)
# write token by token, rows and 4-bit scale pages alike
# (`_kernel_by_token`): gathering max_runs x page_size rows per layer cost
# 6 us per layer for 8 tokens in the b1 decode twin (j-3847708e), whatever
# their width, and 0.12-0.17 ms per layer for the 312-456-token decode
# batches of c48 / c64 (j-fd793d3b, j-7df0161c). A run then costs one merge
# per token instead of one per page.
_BY_TOKEN_MAX_TOKENS = 512
# Lanes of the per-row scale bits `_kernel` reads (one f32 tile).
_SCALE_BLOCK_LANES = 128
# Rows of a 32-bit sublane tile: `_kernel`'s row blocks start at a multiple.
_ROW_TILE = 8


def _kernel_slot_scales(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run (padding runs repeat the last one)
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    run_tok0_ref,  # [R] first token of each run
    # In
    values_ref,  # [T, rows, S] f32: every token's values, broadcast over lanes
    page_ref,  # [1, rows, S] the scale page before the write
    # Out
    o_ref,  # [1, rows, S] aliased with the scale cache
    *,
    num_values: int,
):
    """Per-slot values (lane = slot) of a run's tokens, one by one, into rows
    ``[0, num_values)`` of its scale page."""
    run = pl.program_id(0)
    active = run < num_runs_ref[0]

    @pl.when(active)
    def _():
        shape = page_ref.shape[1:]
        lo, tok0 = run_lo_ref[run], run_tok0_ref[run]
        row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

        def put(i, page):
            take = (lane == lo + i) & (row < num_values)
            return jnp.where(take, values_ref[tok0 + i], page)

        o_ref[0] = jax.lax.fori_loop(0, run_hi_ref[run] - lo, put, page_ref[0])

    @pl.when(jnp.logical_and(jnp.logical_not(active), run == 0))
    def _():
        o_ref[...] = page_ref[...]


def _kernel_slot_scales_by_page(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R] page of each run (padding runs repeat the last one)
    run_lo_ref,  # [R] first slot of each run within its page
    run_hi_ref,  # [R] one past the last slot
    row0_ref,  # [R] row of the run page's slot 0 in `values_ref`'s array
    # In
    values_ref,  # [S + 8, 128] f32: slot s's values in lanes [0, n) of row row0 % 8 + s
    page_ref,  # [1, rows, S] the scale page before the write
    # Out
    o_ref,  # [1, rows, S] aliased with the scale cache
    *,
    num_values: int,
):
    """A run's per-slot values (lane = slot) into rows ``[0, num_values)`` of
    its scale page, all at once: the run's rows, transposed."""
    run = pl.program_id(0)
    active = run < num_runs_ref[0]

    @pl.when(active)
    def _():
        rows, lanes = page_ref.shape[1:]
        off = row0_ref[run] % _ROW_TILE
        by_slot = values_ref[pl.ds(off, lanes), :].T[:rows]  # [rows, S]
        row = jax.lax.broadcasted_iota(jnp.int32, (rows, lanes), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (rows, lanes), 1)
        take = (row < num_values) & (lane >= run_lo_ref[run]) & (lane < run_hi_ref[run])
        o_ref[0] = jnp.where(take, by_slot, page_ref[0])

    @pl.when(jnp.logical_and(jnp.logical_not(active), run == 0))
    def _():
        o_ref[...] = page_ref[...]


def _write_slot_scales_by_page(
    scale_cache, values, num_runs, run_page, run_lo, run_hi, run_tok0, *, interpret
):
    """`write_slot_scales` of a large batch: each run reads its page's worth
    of value rows straight from the batch (as `write_rows`)."""
    num_tokens, num_values = values.shape
    _, rows, lanes = scale_cache.shape
    tile = _ROW_TILE
    padded = jnp.pad(
        values.astype(jnp.float32),
        ((lanes, lanes + tile), (0, _SCALE_BLOCK_LANES - num_values)),
    )
    row0 = run_tok0 - run_lo + lanes
    page_block = pl.BlockSpec(
        (1, rows, lanes), lambda r, n, run_page, *_: (run_page[r], 0, 0)
    )
    return pl.pallas_call(
        functools.partial(_kernel_slot_scales_by_page, num_values=num_values),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(run_page.shape[0],),
            in_specs=[
                pl.BlockSpec(
                    (pl.Element(lanes + tile), pl.Element(_SCALE_BLOCK_LANES)),
                    lambda r, n, run_page, lo, hi, row0: (
                        pl.multiple_of(row0[r] // tile * tile, tile),
                        0,
                    ),
                ),
                page_block,
            ],
            out_specs=page_block,
        ),
        out_shape=jax.ShapeDtypeStruct(scale_cache.shape, scale_cache.dtype),
        # Operand 6 = the scale cache (after 5 scalar-prefetch operands and values).
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_write_slot_scales",
    )(num_runs[None], run_page, run_lo, run_hi, row0, padded, scale_cache)


@functools.partial(jax.jit, static_argnames=("max_runs", "interpret"))
def write_slot_scales(
    scale_cache: jax.Array,
    values: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    interpret: bool = False,
) -> jax.Array:
    """Write f32 ``values [T, n]`` into rows ``[0, n)`` of the ``[pages, rows,
    S]`` scale cache at ``(pages[t], lane slots[t])``; runs as `write_rows`.

    Small batches go token by token through an aliased kernel: XLA's scatter
    cost 0.1 ms per layer for a c48 decode batch (j-fd793d3b); larger ones
    page by page (`_write_slot_scales_by_page`): XLA's scatter there passed
    over the whole scale cache, 0.11-0.18 ms per layer at 8K tokens (c48 /
    c64 twins, j-b0a2e4d7, j-f1b20471).
    """
    num_tokens, num_values = values.shape
    num_pages, rows, lanes = scale_cache.shape
    num_runs, run_page, run_lo, run_hi, run_tok0 = _runs(
        pages, slots, num_pages, max_runs
    )
    if num_tokens > _BY_TOKEN_MAX_TOKENS:
        return _write_slot_scales_by_page(
            scale_cache,
            values,
            num_runs,
            run_page,
            run_lo,
            run_hi,
            run_tok0,
            interpret=interpret,
        )
    padded = jnp.pad(values.astype(jnp.float32), ((0, 0), (0, rows - num_values)))
    broadcast = jnp.broadcast_to(padded[:, :, None], (num_tokens, rows, lanes))
    page_block = pl.BlockSpec(
        (1, rows, lanes), lambda r, n, run_page, *_: (run_page[r], 0, 0)
    )
    return pl.pallas_call(
        functools.partial(_kernel_slot_scales, num_values=num_values),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(max_runs,),
            in_specs=[
                pl.BlockSpec((num_tokens, rows, lanes), lambda r, *_: (0, 0, 0)),
                page_block,
            ],
            out_specs=page_block,
        ),
        out_shape=jax.ShapeDtypeStruct(scale_cache.shape, scale_cache.dtype),
        # Operand 6 = the scale cache (after 5 scalar-prefetch operands and values).
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_write_slot_scales",
    )(num_runs[None], run_page, run_lo, run_hi, run_tok0, broadcast, scale_cache)


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
    each 32-bit word holds 8 slots. `scales` (f32 ``[T]``) go to the last word
    row of a page with folded scales (`msa_fp4.scales_in_page`), lane = slot.
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
    if bits == 4:
        word_bits = rows.astype(jnp.uint32) & 0xF
    else:
        word_bits = jax.lax.bitcast_convert_type(rows, _UINT_BY_BITS[bits]).astype(
            jnp.uint32
        )
    scale_bits = None
    if folded:
        scale_bits = jax.lax.bitcast_convert_type(
            scales.astype(jnp.float32), jnp.uint32
        )
    page_block = pl.BlockSpec(
        (1, rows_per_page, word_bytes, width),
        lambda r, n, run_page, *_: (run_page[r], 0, 0, 0),
    )
    if num_tokens <= _BY_TOKEN_MAX_TOKENS:
        tokens_block = pl.BlockSpec((num_tokens, width), lambda r, *_: (0, 0))
        extra = []
        if folded:  # each token's scale bits on every lane
            extra = [jnp.broadcast_to(scale_bits[:, None], (num_tokens, width))]
        return pl.pallas_call(
            functools.partial(_kernel_by_token, bits=bits),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                grid=(max_runs,),
                in_specs=[tokens_block, page_block, *([tokens_block] * len(extra))],
                out_specs=page_block,
            ),
            out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            # Operand 6 = the cache (after 5 scalar-prefetch operands and rows).
            input_output_aliases={6: 0},
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
            interpret=interpret,
            name="minimax_m3_msa_write_rows",
        )(num_runs[None], run_page, run_lo, run_hi, run_tok0, word_bits, cache, *extra)
    # Each run reads its page's worth of rows straight from the batch, padded
    # by a page each side so that slot s of the run's page is row
    # tok0 - lo + s + S: no [runs, S, W] gather of rows (and of the scale
    # bits) is built first (XLA gathers of 0.06-0.16 ms per write at 8K
    # tokens, the 4-bit scale-bit element gather 0.1 ms, c48 twin j-b0a2e4d7).
    # Blocks start at the row tile below; the kernel reads from row0 % tile.
    tile = _ROW_TILE
    folds = width // 128
    padded = jnp.pad(word_bits, ((page_size, page_size + tile), (0, 0)))
    padded = padded.reshape(-1, 128)  # row r's 128-lane blocks consecutive
    row0 = run_tok0 - run_lo + page_size

    def block(lanes, folds=1):
        return pl.BlockSpec(
            (pl.Element((page_size + tile) * folds), pl.Element(lanes)),
            lambda r, n, run_page, lo, hi, row0: (
                pl.multiple_of(row0[r] // tile * tile * folds, tile),
                0,
            ),
        )

    extra, extra_specs = [], []
    if folded:  # each token's scale bits on every lane, read as the rows
        # The kernel reads a lane tile of rows from within the page's block.
        assert page_size >= _SCALE_BLOCK_LANES, page_size
        extra = [
            jnp.broadcast_to(
                jnp.pad(scale_bits, (page_size, page_size + tile))[:, None],
                (num_tokens + 2 * page_size + tile, _SCALE_BLOCK_LANES),
            )
        ]
        extra_specs = [block(_SCALE_BLOCK_LANES)]
    rows_block = block(128, folds)
    return pl.pallas_call(
        functools.partial(_kernel, bits=bits),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(max_runs,),
            in_specs=[rows_block, page_block, *extra_specs],
            out_specs=page_block,
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        # Operand 6 = the cache (after 5 scalar-prefetch operands and the rows).
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_write_rows",
    )(num_runs[None], run_page, run_lo, run_hi, row0, padded, cache, *extra)


def max_write_runs(num_tokens: int, max_num_seqs: int, page_size: int) -> int:
    """Runs of a batch: a request's q tokens span at most
    ``(q - 1) // page_size + 2`` pages."""
    return max(1, min(num_tokens, num_tokens // page_size + 2 * max_num_seqs))
