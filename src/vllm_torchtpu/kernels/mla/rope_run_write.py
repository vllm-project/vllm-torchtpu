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
"""Rope/TENSORCORE KV cache write as a few contiguous DMAs per sequence.

`update_sparse_mla_kv_cache` uses this for buckets of at least
`kv_cache_utils.ROPE_RUN_MIN_TOKENS` tokens; it is byte-identical to the XLA
scatter `kv_cache_utils._scatter_rows` it replaces there.

Why. The rope/TENSORCORE pool is `(num_pages, slabs_per_page, kv_packing,
lane)` uint8. `_scatter_rows` writes it with one XLA scatter that has one
update per row, `cache.at[page, slot // kv_packing, slot % kv_packing]`. On
TPU v7x that costs about 100 us for one 512-row prompt, although a prefill's
new rows are consecutive positions in their pages.

What. A slab holds `kv_packing` consecutive positions of one page. A sequence's
new rows cover consecutive positions, so they touch consecutive slabs, and
within one page those slabs are consecutive in memory. Three steps:

1. `rope_run_plan` builds slab-ordered values: entry `j` is one whole slab.
   Entries are aligned to positions, not to token indices -- a sequence's first
   new row can sit anywhere inside a slab, so cutting the token array into
   `kv_packing`-row groups would mix rows of different slabs. A sequence's
   entries are consecutive and in slab order. Only its head and tail slab can
   be partly covered; the rows it does not own there keep the pool's old
   bytes, so only `2 * num_seqs` old slabs are read from the pool.
2. It also derives a run table, one run per (sequence, page touched):
   `(src_entry, dst_page, dst_slab, length)`. A run never crosses a page, so
   the kernel addresses the pool in its native 4D shape and nothing reshapes a
   pool-sized array at the XLA level (see `partial_kv_update` for why that
   matters).
3. `_run_write_kernel` copies each run with async DMAs straight from the slab
   values in HBM into the aliased pool in HBM.

Distinct sequences write disjoint slabs (each owns the pages it writes), so
the DMAs never race.

Run bound. A sequence with `q` new rows starting at position `f` touches
`(f + q - 1) // P - f // P + 1 <= (q - 1) // P + 2` pages, and
`sum_s q_s // P <= num_tokens // P`, so `num_tokens // P + 2 * num_seqs` runs
always suffice. Runs past the last live one have length 0, so both per-run DMA
loops run zero times for them: nothing is issued or written.

Chunking. A DMA descriptor has a static shape; only its starts are dynamic. A
run of `L >= C` slabs is covered by `ceil(L / C)` chunks of `C` slabs; the last
chunk is shifted back to end exactly at the run end, so it overlaps its
predecessor and rewrites those slabs with the same bytes. A run shorter than
`C` would need a chunk that spills past the run end into slabs that must keep
their content, so it is copied slab by slab.

`CHUNK_SLABS = 16`: a larger `C` saves DMAs only on long runs and costs up to
`C - 1` single-slab DMAs on every run just below `C` (a prefill segment's
partial head or tail page).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils

CHUNK_SLABS = 16

RunTable = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _segment_of(ends: jax.Array, idx: jax.Array) -> jax.Array:
    """Index of the segment (given by its cumulative `ends`) each `idx` is in.

    A broadcast compare rather than `searchsorted`; out-of-range `idx` clamp to
    the last segment and are masked by the caller.
    """
    seg = jnp.sum(ends[None, :] <= idx[:, None], axis=1, dtype=jnp.int32)
    return jnp.minimum(seg, ends.shape[0] - 1)


def rope_run_plan(
    pool: jax.Array,
    token_bytes: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
) -> tuple[jax.Array, RunTable]:
    """Slab values and run table for `_run_write_kernel`.

    Args:
      pool: Rope/TENSORCORE cache, `(num_pages, slabs_per_page, kv_packing,
        lane)` uint8. Only head and tail slabs are read from it.
      token_bytes: This step's rows, `[num_tokens, lane]` uint8.
      seq_lens, block_tables, query_start_loc: As for
        `update_sparse_mla_kv_cache`.

    Returns:
      `slab_value` `[n_entries, kv_packing, lane]` and the run table
      `(run_src, run_page, run_slab, run_len)`, each `[n_runs]` int32.
    """
    _, slabs_per_page, kv_packing, _ = pool.shape
    page_size = slabs_per_page * kv_packing
    num_tokens = token_bytes.shape[0]
    num_seqs = seq_lens.shape[0]
    pages = block_tables.reshape(num_seqs, -1)
    max_pages = pages.shape[1]
    seq_lens = seq_lens.astype(jnp.int32)
    query_start_loc = query_start_loc.astype(jnp.int32)

    q_len = query_start_loc[1 : num_seqs + 1] - query_start_loc[:num_seqs]
    first = seq_lens - q_len  # position of each sequence's first new row
    last = seq_lens - 1
    live_seq = q_len > 0

    # Slab entries.
    n_slabs = jnp.where(live_seq, last // kv_packing - first // kv_packing + 1, 0)
    ends = jnp.cumsum(n_slabs)
    starts = ends - n_slabs
    n_entries = num_tokens // kv_packing + 2 * num_seqs
    j = jnp.arange(n_entries, dtype=jnp.int32)
    seq = _segment_of(ends, j)
    live = j < ends[-1]
    slab_pos = (first[seq] // kv_packing + (j - starts[seq])) * kv_packing
    row_pos = slab_pos[:, None] + jnp.arange(kv_packing, dtype=jnp.int32)[None]
    mine = (
        live[:, None]
        & (row_pos >= first[seq][:, None])
        & (row_pos <= last[seq][:, None])
    )
    tok = jnp.clip(
        query_start_loc[seq][:, None] + (row_pos - first[seq][:, None]),
        0,
        num_tokens - 1,
    )
    new = token_bytes[tok]  # [n_entries, kv_packing, lane]

    # Old bytes of each sequence's head and tail slab, the only slabs a
    # sequence can cover partly. A dead sequence reads slab 0 of page 0; any
    # in-bounds slab would do, as its value is never selected.
    def slab_of(pos):
        ordinal = jnp.clip(pos // page_size, 0, max_pages - 1)
        page = jnp.where(live_seq, pages[jnp.arange(num_seqs), ordinal], 0)
        slab = jnp.where(live_seq, (pos % page_size) // kv_packing, 0)
        return page, slab

    old_head = pool[slab_of(first)]  # [num_seqs, kv_packing, lane]
    old_tail = pool[slab_of(last)]
    is_head = j == starts[seq]
    old = jnp.where(is_head[:, None, None], old_head[seq], old_tail[seq])
    slab_value = jnp.where(mine[..., None], new, old)

    # Run table: one run per (sequence, page touched).
    n_runs = num_tokens // page_size + 2 * num_seqs
    n_pages = jnp.where(live_seq, last // page_size - first // page_size + 1, 0)
    r_ends = jnp.cumsum(n_pages)
    r_starts = r_ends - n_pages
    r = jnp.arange(n_runs, dtype=jnp.int32)
    rseq = _segment_of(r_ends, r)
    rlive = r < r_ends[-1]
    seq_first, seq_last = first[rseq], last[rseq]
    ordinal = seq_first // page_size + (r - r_starts[rseq])
    lo = jnp.maximum(seq_first, ordinal * page_size)
    hi = jnp.minimum(seq_last, ordinal * page_size + page_size - 1)
    run_src = starts[rseq] + lo // kv_packing - seq_first // kv_packing
    run_page = pages[rseq, jnp.clip(ordinal, 0, max_pages - 1)]
    run_slab = (lo % page_size) // kv_packing
    run_len = hi // kv_packing - lo // kv_packing + 1

    def run_field(x: jax.Array) -> jax.Array:
        return jnp.where(rlive, x, 0).astype(jnp.int32)

    runs = (
        run_field(run_src),
        run_field(run_page),
        run_field(run_slab),
        run_field(run_len),
    )
    return slab_value, runs


def _run_write_kernel(
    run_src,
    run_page,
    run_slab,
    run_len,
    vals_hbm,
    _pool_in,
    pool_hbm,
    sems,
    *,
    chunk: int,
):
    """Copy every run `vals[src:src + len] -> pool[page, slab:slab + len]`.

    All DMAs are started first and waited for at the end, so they overlap.
    Chunk DMAs signal `sems[0]`, single-slab DMAs `sems[1]`; a wait only needs
    the byte count, so each uses a descriptor shaped like the copies it
    retires. `_pool_in` is never read; it exists only so `input_output_aliases`
    has something to alias.
    """

    def chunk_copy(src, page, slab):
        return pltpu.make_async_copy(
            vals_hbm.at[pl.ds(src, chunk)],
            pool_hbm.at[page, pl.ds(slab, chunk)],
            sems.at[0],
        )

    def slab_copy(src, page, slab):
        return pltpu.make_async_copy(
            vals_hbm.at[pl.ds(src, 1)],
            pool_hbm.at[page, pl.ds(slab, 1)],
            sems.at[1],
        )

    def issue(r, counts):
        n_chunk, n_slab = counts
        length = run_len[r]
        src, page, slab = run_src[r], run_page[r], run_slab[r]
        long_run = length >= chunk
        # Both operands are non-negative, so truncating `lax.div` equals floor
        # division without the sign fix-up `//` lowers to.
        chunks_here = jnp.where(long_run, lax.div(length + chunk - 1, chunk), 0)

        def start_chunk(k, carry):
            # The last chunk ends exactly at the run end.
            off = jnp.minimum(k * chunk, length - chunk)
            chunk_copy(src + off, page, slab + off).start()
            return carry

        lax.fori_loop(0, chunks_here, start_chunk, 0)
        slabs_here = jnp.where(long_run, 0, length)

        def start_slab(k, carry):
            slab_copy(src + k, page, slab + k).start()
            return carry

        lax.fori_loop(0, slabs_here, start_slab, 0)
        return n_chunk + chunks_here, n_slab + slabs_here

    zero = jnp.int32(0)
    n_chunk, n_slab = lax.fori_loop(0, run_len.shape[0], issue, (zero, zero))

    def wait_chunk(_, carry):
        chunk_copy(0, 0, 0).wait()
        return carry

    def wait_slab(_, carry):
        slab_copy(0, 0, 0).wait()
        return carry

    lax.fori_loop(0, n_chunk, wait_chunk, 0)
    lax.fori_loop(0, n_slab, wait_slab, 0)


def _write_runs(
    pool: jax.Array,
    slab_value: jax.Array,
    runs: RunTable,
    *,
    interpret: bool | pltpu.InterpretParams,
) -> jax.Array:
    """Apply a run table to `pool` in place (aliased)."""
    assert pool.dtype == slab_value.dtype, (pool.dtype, slab_value.dtype)
    assert slab_value.shape[1:] == pool.shape[2:], (slab_value.shape, pool.shape)
    # A chunk lies inside one page and inside the slab values.
    chunk = min(CHUNK_SLABS, slab_value.shape[0], pool.shape[1])
    return pl.pallas_call(
        functools.partial(_run_write_kernel, chunk=chunk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(runs),
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),  # slab values, HBM
                pl.BlockSpec(memory_space=pl.ANY),  # pool, aliased
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[pltpu.SemaphoreType.DMA((2,))],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={len(runs) + 1: 0},
        interpret=interpret,
    )(*runs, slab_value, pool)


def write_rope_runs(
    pool: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    spec: kv_cache_utils.SparseMLAKVCacheSpec,
    *,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Rope/TENSORCORE cache write, byte-identical to `_scatter_rows`.

    Args:
      pool: Rope cache, shaped and typed by `spec`.
      k_pe: Decoupled RoPE keys (`[num_tokens, rope_dim]`).
      seq_lens, block_tables, query_start_loc: As for
        `update_sparse_mla_kv_cache`.
      spec: Layout descriptor the rope cache was allocated from.
      interpret: Forwarded to `pl.pallas_call`, for CPU tests.

    Returns:
      The updated rope cache.
    """
    assert spec.cache_type is kv_cache_utils.KVCacheType.ROPE, spec
    assert spec.layout is kv_cache_utils.KVCacheLayout.TENSORCORE, spec
    assert pool.shape == spec.shape, (pool.shape, spec.shape)
    token_bytes = kv_cache_utils.as_token_bytes(k_pe, spec.token_bytes)
    slab_value, runs = rope_run_plan(
        pool, token_bytes, seq_lens, block_tables, query_start_loc
    )
    return _write_runs(pool, slab_value, runs, interpret=interpret)
