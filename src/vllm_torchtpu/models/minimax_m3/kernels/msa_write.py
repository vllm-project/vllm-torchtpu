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

A small batch (decode) instead moves only the rows that change: each
window of `WINDOW_ROWS` packed rows that receives tokens is copied from the
cache, merged token by token and copied back, and nothing is staged.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_UINT_BY_BITS = {16: jnp.uint16, 8: jnp.uint8}

# Packed rows per window of the row write: one sublane tile of 32-bit words.
WINDOW_ROWS = 8
# Batches of at most this many tokens take the row write, larger ones the
# page write: the row write merges token by token, the page write merges a
# whole page per packing slot.
ROW_WRITE_MAX_TOKENS = 256


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
):
    run = pl.program_id(0)
    active = run < num_runs_ref[0]

    @pl.when(active)
    def _():
        rows, packing, width = page_ref.shape[1:]
        bits = 32 // packing
        words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
        lo, hi = run_lo_ref[run], run_hi_ref[run]
        row = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 0)
        for j in range(packing):
            slot = row * packing + j
            take = (slot >= lo) & (slot < hi)
            keep = jnp.uint32(~(((1 << bits) - 1) << (bits * j)) & 0xFFFFFFFF)
            merged = (words & keep) | (new_ref[0, j] << jnp.uint32(bits * j))
            words = jnp.where(take, merged, words)
        o_ref.bitcast(jnp.uint32)[0, :, 0, :] = words

    # With no run at all the single output page must still be written back
    # unchanged. Later padding steps revisit the last run's page and leave
    # its (already written) output block alone.
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


@functools.partial(jax.jit, static_argnames=("max_num_seqs", "interpret"))
def write_rows(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_num_seqs: int,
    interpret: bool = False,
) -> jax.Array:
    """Write ``rows [T, W]`` (cache dtype) at ``(pages[t], slots[t])``.

    Tokens with a page outside the cache are dropped. Consecutive tokens of a
    run must take consecutive slots of one page, as a request's new tokens do;
    a run boundary is wherever that breaks. `max_num_seqs` bounds the
    requests in the batch, each one run of consecutive slots across pages;
    dropped tokens must not split a request.
    """
    rows_per_page, packing = cache.shape[1:3]
    num_tokens = rows.shape[0]
    bits = 32 // packing
    word_bits = jax.lax.bitcast_convert_type(rows, _UINT_BY_BITS[bits]).astype(
        jnp.uint32
    )
    if num_tokens <= ROW_WRITE_MAX_TOKENS:
        span = math.gcd(WINDOW_ROWS, rows_per_page) * packing
        max_windows = max_write_runs(num_tokens, max_num_seqs, span)
        return _write_windows(cache, word_bits, pages, slots, max_windows, interpret)
    max_runs = max_write_runs(num_tokens, max_num_seqs, rows_per_page * packing)
    return _write_pages(cache, word_bits, pages, slots, max_runs, interpret)


def _write_pages(cache, word_bits, pages, slots, max_runs, interpret):
    num_pages, rows_per_page, packing, width = cache.shape
    page_size = rows_per_page * packing
    num_tokens = word_bits.shape[0]
    num_runs, run_page, run_lo, run_hi, run_tok0 = _runs(
        pages, slots, num_pages, max_runs
    )
    # Each run's rows laid out by slot ([run, j, r] = slot r * p + j), so the
    # kernel merges whole packing slots without strided loads.
    src = run_tok0[:, None] + jnp.arange(page_size, dtype=jnp.int32) - run_lo[:, None]
    new = word_bits[jnp.clip(src, 0, num_tokens - 1)]
    new = new.reshape(max_runs, rows_per_page, packing, width).transpose(0, 2, 1, 3)

    page_block = pl.BlockSpec(
        (1, rows_per_page, packing, width),
        lambda r, n, run_page, *_: (run_page[r], 0, 0, 0),
    )
    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(max_runs,),
            in_specs=[
                pl.BlockSpec(
                    (1, packing, rows_per_page, width), lambda r, *_: (r, 0, 0, 0)
                ),
                page_block,
            ],
            out_specs=page_block,
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        # Operand 5 = the cache (after 4 scalar-prefetch operands and `new`).
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_write_rows",
    )(num_runs[None], run_page, run_lo, run_hi, new, cache)


def _window_kernel(
    # Scalar prefetch
    num_windows_ref,  # [1]
    window_page_ref,  # [N] page of each window
    window_row_ref,  # [N] first packed row of each window
    window_tok0_ref,  # [N] first token of each window
    window_end_ref,  # [N] one past its last token
    token_row_ref,  # [T] packed row of each token within its window
    token_shift_ref,  # [T] bit offset of each token within its word
    # In
    new_ref,  # [T, W] uint32: each token's row in the low bits
    cache_ref,  # [pages, S / p, p, W] left in HBM
    # Out
    o_ref,  # aliased with cache_ref
    # Scratch
    buf,  # [N, WINDOW_ROWS, p, W] the windows
    sems,  # DMA semaphores [2, N]
):
    num_windows = num_windows_ref[0]
    rows, packing = buf.shape[1:3]
    bits = 32 // packing

    def window(ref, w):
        return ref.at[window_page_ref[w], pl.ds(window_row_ref[w], rows)]

    def fetch(w):
        return pltpu.make_async_copy(window(cache_ref, w), buf.at[w], sems.at[0, w])

    def store(w):
        return pltpu.make_async_copy(buf.at[w], window(o_ref, w), sems.at[1, w])

    # Windows are disjoint, so every fetch can be in flight before any store.
    @pl.loop(0, num_windows)
    def _(w):
        fetch(w).start()

    words = buf.bitcast(jnp.uint32)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, (rows, words.shape[-1]), 0)

    @pl.loop(0, num_windows)
    def _(w):
        fetch(w).wait()

        def merge(t, window):
            shift = token_shift_ref[t].astype(jnp.uint32)
            keep = ~(jnp.uint32((1 << bits) - 1) << shift)
            merged = (window & keep) | (new_ref[pl.ds(t, 1), :] << shift)
            return jnp.where(row_ids == token_row_ref[t], merged, window)

        words[w, :, 0, :] = jax.lax.fori_loop(
            window_tok0_ref[w], window_end_ref[w], merge, words[w, :, 0, :]
        )
        store(w).start()

    @pl.loop(0, num_windows)
    def _(w):
        store(w).wait()


def _windows(
    pages: jax.Array, rows: jax.Array, num_pages: int, size: int, max_windows: int
):
    """Windows to rewrite: maximal spans of consecutive tokens whose packed
    `rows` fall in one window of `size` rows of one page."""
    num_tokens = pages.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = (pages >= 0) & (pages < num_pages)
    win = rows // size

    # Token t continues the window of token t - 1.
    cont = valid[1:] & valid[:-1] & (pages[1:] == pages[:-1]) & (win[1:] == win[:-1])
    start = valid & ~jnp.concatenate([jnp.zeros(1, bool), cont])
    end = valid & ~jnp.concatenate([cont, jnp.zeros(1, bool)])
    window_id = jnp.cumsum(start.astype(jnp.int32)) - 1
    num_windows = jnp.minimum(jnp.sum(start.astype(jnp.int32)), max_windows)
    at_start = jnp.where(start, window_id, max_windows)
    at_end = jnp.where(end, window_id, max_windows)

    def per_window(at, values):
        return jnp.zeros(max_windows, jnp.int32).at[at].set(values, mode="drop")

    return (
        num_windows,
        per_window(at_start, pages),
        per_window(at_start, win * size),
        per_window(at_start, tok),
        per_window(at_end, tok + 1),
    )


def _write_windows(cache, word_bits, pages, slots, max_windows, interpret):
    num_pages, rows_per_page, packing, width = cache.shape
    rows = math.gcd(WINDOW_ROWS, rows_per_page)
    token_row = slots // packing
    num_windows, window_page, window_row, window_tok0, window_end = _windows(
        pages, token_row, num_pages, rows, max_windows
    )
    return pl.pallas_call(
        _window_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(word_bits.shape, lambda i, *_: (0, 0)),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[
                pltpu.VMEM((max_windows, rows, packing, width), cache.dtype),
                pltpu.SemaphoreType.DMA((2, max_windows)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        # Operand 8 = the cache (after 7 scalar-prefetch operands and the rows).
        input_output_aliases={8: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_write_rows",
    )(
        num_windows[None],
        window_page,
        window_row,
        window_tok0,
        window_end,
        token_row % rows,
        slots % packing * (32 // packing),
        word_bits,
        cache,
    )


def max_write_runs(num_tokens: int, max_num_seqs: int, span: int) -> int:
    """Spans of `span` slots a batch writes: a request's q tokens touch at
    most ``(q - 1) // span + 2`` of them."""
    return max(1, min(num_tokens, num_tokens // span + 2 * max_num_seqs))
