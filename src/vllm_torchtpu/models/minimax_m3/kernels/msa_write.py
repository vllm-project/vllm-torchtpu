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
    *refs,  # with folded scales: [1, 1, W] uint32 scale bits by slot, then o_ref
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
        for j in range(packing):
            slot = row * packing + j
            take = (slot >= lo) & (slot < hi)
            keep = jnp.uint32(~(((1 << bits) - 1) << (bits * j)) & 0xFFFFFFFF)
            merged = (words & keep) | (new_ref[0, j] << jnp.uint32(bits * j))
            words = jnp.where(take, merged, words)
        if scale_ref is not None:  # last word row: one f32 scale per slot (lane)
            lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
            take = (row == rows - 1) & (lane >= lo) & (lane < hi)
            words = jnp.where(take, scale_ref[0], words)
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
    if folded:
        new = jnp.pad(new, ((0, 0), (0, 0), (0, rows_per_page - data_rows), (0, 0)))
        bits_by_slot = jax.lax.bitcast_convert_type(
            scales.astype(jnp.float32), jnp.uint32
        )[src]  # [runs, S]
        extra = [jnp.pad(bits_by_slot, ((0, 0), (0, width - page_size)))[:, None, :]]

    page_block = pl.BlockSpec(
        (1, rows_per_page, word_bytes, width),
        lambda r, n, run_page, *_: (run_page[r], 0, 0, 0),
    )
    return pl.pallas_call(
        functools.partial(_kernel, bits=bits),
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


def max_write_runs(num_tokens: int, max_num_seqs: int, page_size: int) -> int:
    """Runs of a batch: a request's q tokens span at most
    ``(q - 1) // page_size + 2`` pages."""
    return max(1, min(num_tokens, num_tokens // page_size + 2 * max_num_seqs))
