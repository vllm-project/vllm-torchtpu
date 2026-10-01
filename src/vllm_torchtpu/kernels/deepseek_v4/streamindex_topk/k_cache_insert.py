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
"""Pallas insert of a step's indexer K rows into the packed u8 K cache.

Equivalent to the XLA scatter

    cache_kv = cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k)

on a cache of shape ``u8[num_pages, page_size // 4, 4, lkv_dim]``, but touches
only the 4-token groups ``(page, slot // 4)`` this step writes. XLA lowers that
scatter with a relayout of the whole cache to ``[num_pages * page_size,
lkv_dim]`` and back, which dominates at prefill sizes.

* The cache is an HBM ref (``memory_space=pl.ANY``) aliased input -> output,
  so untouched bytes are never read or written.
* One ``[4, lkv_dim]`` group is whole memory tiles of the cache's native
  layout; traced indices appear only on the untiled leading dims.
* A full group (all 4 slots written by this step) is one VMEM -> HBM DMA of
  the staged rows. A partial group (a sequence's head or tail) is read
  HBM -> VMEM, row-masked merged, and written back.
* Padding tokens (``>= cu_q_lens[-1]``) and ``q_len == 0`` sequences produce
  no group.

The group list is built with jnp ops at the static bound
``G = T // 4 + 2 * num_seqs`` (a sequence of ``q`` tokens starting at slot
offset ``o`` in [0, 3] spans ``ceil((o + q) / 4) <= q / 4 + 1.5`` groups) and
scalar-prefetched. Every started DMA is waited before the kernel returns.

Upstream API references (tokamax @851e948
``_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py`` = T; jax @ce57049
``tests/pallas/tpu_pallas_test.py`` = J):
  * aliased HBM cache written by ``make_async_copy`` at traced page ids:
    T:1370-1395, T:1429-1500 (``_update_kv_cache``); the alias index counts
    the scalar-prefetch operands: T:2485-2510, megablox/gmm.py:504-540.
  * the same ``[pages, page_size // packing, packing, align(dim, 128)]`` u8
    cache layout: T:69-86.
  * ``PrefetchScalarGridSpec`` + ``pl.ANY`` in/out + VMEM / DMA-semaphore
    scratch: T:2368-2487, ragged_paged_attention/kernel.py:832-887.
  * many same-size DMAs on one semaphore, waited by a rebuilt descriptor:
    T:570-577, T:1520-1535, ragged_paged_attention/kernel.py:33-70.
  * ``pl.ANY`` in/out with aliasing: J:3532-3558
    (``test_simple_tile_aligned_dynamic_size_dma``).
  * uint8 ``jnp.where`` in-kernel: J:3214-3240 (``test_bool_select_operation``).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

KV_PACKING = 4

# Smallest static token count (``k.shape[0]``) at which the kernel beats the
# XLA scatter at every measured larger bucket; below it callers keep the
# scatter. See `insert_indexer_k_cache_auto`.
KERNEL_MIN_TOKENS = 32

# Meta rows, each of length G.
_PAGE, _GROUP, _LO, _HI, _PRANK, _FULL = range(6)


def insert_indexer_k_cache_ref(
    cache_kv: jax.Array,
    k: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
) -> jax.Array:
    """XLA scatter insert. ``k`` is ``u8[T, lkv_dim]`` (already lane-padded)."""
    _, block_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    block_size = block_size_per_kv_packing * kv_packing
    tok = jnp.arange(k.shape[0], dtype=jnp.int32)
    seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32)
    q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
    pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
    page = jnp.where(
        tok < cu_q_lens[-1],
        page_indices.reshape(seq_lens.shape[0], -1)[seq_id, pos // block_size],
        jnp.int32(2**30),
    )
    slot = pos % block_size
    return cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k)


def _build_groups(k, seq_lens, page_indices, cu_q_lens, *, block_size):
    """Returns (meta i32[6 * G], counts i32[2], staging u8[G, 4, D], G).

    Meta rows: page, group within page, rows [lo, hi) written, partial rank
    (-1 for full groups), full flag. Active groups occupy [0, n_active).
    """
    num_tokens, _ = k.shape
    num_seqs = seq_lens.shape[0]
    kp = KV_PACKING
    num_groups = num_tokens // kp + 2 * num_seqs
    q_len = cu_q_lens[1:] - cu_q_lens[:-1]
    p1 = seq_lens
    p0 = p1 - q_len
    ng = jnp.where(q_len > 0, (p1 + kp - 1) // kp - p0 // kp, 0).astype(jnp.int32)
    cum = jnp.cumsum(ng)
    off = cum - ng
    n_active = cum[-1]
    gi = jnp.arange(num_groups, dtype=jnp.int32)
    seq = jnp.minimum(jnp.searchsorted(cum, gi, side="right"), num_seqs - 1)
    seq = seq.astype(jnp.int32)
    active = gi < n_active
    # Absolute position of the group's slot 0.
    gstart = (p0[seq] // kp + gi - off[seq]) * kp
    pages = page_indices.reshape(num_seqs, -1)
    page = pages[seq, jnp.clip(gstart // block_size, 0, pages.shape[1] - 1)]
    grp = (gstart % block_size) // kp
    lo = jnp.clip(p0[seq] - gstart, 0, kp)
    hi = jnp.clip(p1[seq] - gstart, 0, kp)
    full = active & (lo == 0) & (hi == kp)
    partial = active & ~full
    prank = jnp.where(partial, jnp.cumsum(partial.astype(jnp.int32)) - 1, -1)
    n_partial = jnp.sum(partial.astype(jnp.int32))
    # Token feeding row j of group i; rows outside [lo, hi) are masked/unused.
    j = jnp.arange(kp, dtype=jnp.int32)
    tok = cu_q_lens[seq][:, None] + (gstart[:, None] + j[None, :] - p0[seq][:, None])
    tok = jnp.clip(tok, 0, num_tokens - 1)
    staging = jnp.take(k, tok.reshape(-1), axis=0).reshape(num_groups, kp, -1)
    meta = jnp.concatenate([page, grp, lo, hi, prank, full.astype(jnp.int32)])
    counts = jnp.stack([n_active, n_partial]).astype(jnp.int32)
    return meta.astype(jnp.int32), counts, staging, num_groups


def _insert_kernel(
    meta_ref, cnt_ref, stg_ref, cache_in_ref, cache_ref, tmp_ref, sem_ref, *, G
):
    del cache_in_ref  # Aliased to cache_ref; all accesses go through the output.
    n_active = cnt_ref[0]
    n_partial = cnt_ref[1]

    def fld(row, i):
        return meta_ref[row * G + i]

    # 1. Start every partial group's HBM -> VMEM read.
    def start_reads(i, c):
        @pl.when(fld(_FULL, i) == 0)
        def _():
            pltpu.make_async_copy(
                cache_ref.at[fld(_PAGE, i), fld(_GROUP, i)],
                tmp_ref.at[fld(_PRANK, i)],
                sem_ref.at[0],
            ).start()

        return c

    jax.lax.fori_loop(0, n_active, start_reads, 0)

    # 2. Wait them; all group DMAs are the same size.
    wait_read = pltpu.make_async_copy(cache_ref.at[0, 0], tmp_ref.at[0], sem_ref.at[0])

    def wait_reads(i, c):
        wait_read.wait()
        return c

    jax.lax.fori_loop(0, n_partial, wait_reads, 0)

    # 3. Merge partial groups, start every write.
    def start_writes(i, c):
        page, g, lo, hi, pr, full = (fld(row, i) for row in range(6))

        @pl.when(full == 1)
        def _():
            pltpu.make_async_copy(
                stg_ref.at[i], cache_ref.at[page, g], sem_ref.at[1]
            ).start()

        @pl.when(full == 0)
        def _():
            old = tmp_ref[pr]
            new = stg_ref[i]
            row = jax.lax.broadcasted_iota(jnp.int32, old.shape, 0)
            tmp_ref[pr] = jnp.where((row >= lo) & (row < hi), new, old)
            pltpu.make_async_copy(
                tmp_ref.at[pr], cache_ref.at[page, g], sem_ref.at[1]
            ).start()

        return c

    jax.lax.fori_loop(0, n_active, start_writes, 0)

    # 4. Wait every write.
    wait_write = pltpu.make_async_copy(stg_ref.at[0], cache_ref.at[0, 0], sem_ref.at[1])

    def wait_writes(i, c):
        wait_write.wait()
        return c

    jax.lax.fori_loop(0, n_active, wait_writes, 0)


def insert_indexer_k_cache(
    cache_kv: jax.Array,
    k: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Pallas equivalent of `insert_indexer_k_cache_ref`.

    Args:
      cache_kv: u8[num_pages, page_size // 4, 4, lkv_dim]; aliased in -> out,
        so donate it at the jit boundary to avoid a defensive copy.
      k: u8[T, lkv_dim], this step's rows, already padded to lkv_dim.
      seq_lens: i32[num_seqs], sequence length including this step's tokens.
      page_indices: i32[num_seqs * pages_per_seq].
      cu_q_lens: i32[num_seqs + 1]; tokens at or past cu_q_lens[-1] are
        padding and are not written.

    Sequences must not share a page that this step writes.
    """
    _, bpk, kp, lkv_dim = cache_kv.shape
    if kp != KV_PACKING:
        raise ValueError(f"expected kv_packing={KV_PACKING}, got {cache_kv.shape}")
    if k.shape[-1] != lkv_dim:
        raise ValueError(f"k width {k.shape[-1]} != cache lkv_dim {lkv_dim}")
    meta, counts, staging, num_groups = _build_groups(
        k, seq_lens, page_indices, cu_q_lens, block_size=bpk * kp
    )
    max_partial = max(2 * seq_lens.shape[0], 1)  # <= 2 partial groups per seq
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(1,),
        in_specs=[
            pl.BlockSpec((num_groups, kp, lkv_dim), lambda i, *_: (0, 0, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=[
            pltpu.VMEM((max_partial, kp, lkv_dim), cache_kv.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    return pl.pallas_call(
        functools.partial(_insert_kernel, G=num_groups),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(cache_kv.shape, cache_kv.dtype),
        # Operand indices count the scalar-prefetch operands:
        # meta=0, counts=1, staging=2, cache=3.
        input_output_aliases={3: 0},
        interpret=interpret,
        name="insert_indexer_k_cache",
    )(meta, counts, staging, cache_kv)


def insert_indexer_k_cache_auto(
    cache_kv: jax.Array,
    k: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
) -> jax.Array:
    """Kernel when the static token count is >= `KERNEL_MIN_TOKENS`, else the
    XLA scatter (cheaper for decode-sized buckets)."""
    fn = (
        insert_indexer_k_cache
        if k.shape[0] >= KERNEL_MIN_TOKENS
        else insert_indexer_k_cache_ref
    )
    return fn(cache_kv, k, seq_lens, page_indices, cu_q_lens)
