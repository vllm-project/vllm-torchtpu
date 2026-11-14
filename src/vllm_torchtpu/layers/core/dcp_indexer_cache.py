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
"""DCP indexer K cache writer, shared by the plain DCP and PCP manager paths."""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.mla.kv_cache_utils import OOB_PAGE, _row_positions


def scatter_indexer_k_rows(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    rank: jax.Array,
    *,
    cp_size: int,
    interleave_size: int,
) -> jax.Array:
    """Write the indexer K rows this DCP rank owns (inside a DCP `shard_map`)."""
    _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
    page_size = block_size_per_kv_packing * kv_packing
    # The latent cache is guarded on its own page size in
    # `update_sparse_mla_kv_cache_dcp`; this cache is a separate
    # allocation, so it needs the check on its own number.
    if page_size % interleave_size:
        raise ValueError(
            f"interleave_size={interleave_size} must divide the indexer "
            f"K cache page_size={page_size}. Otherwise an interleave "
            "chunk straddles a page boundary and the local index "
            "`streamindex_topk_dcp` derives from a global position stops "
            "agreeing with the (page, slot) written here."
        )
    k_packed = jnp.pad(k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1])))

    # Same notion of "where does this row belong" as the latent-cache
    # writers, so the two cannot drift apart.
    _, seq_id, pos, in_batch = _row_positions(k_packed.shape[0], seq_lens, cu_q_lens)

    virtual_page_size = jnp.int32(page_size * cp_size)
    cycle = jnp.int32(cp_size * interleave_size)
    interleave = jnp.int32(interleave_size)
    offset = pos % virtual_page_size
    owner = (offset % cycle) // interleave
    slot = (offset // cycle) * interleave + (offset % interleave)

    # Virtual ordinal -> physical block in this rank's own shard, exactly
    # as `streamindex_topk_dcp` resolves it on the read side.
    local_page_indices = jnp.mod(page_indices, jnp.int32(cache_kv.shape[0])).reshape(
        seq_lens.shape[0], -1
    )
    # A padded token's `pos` is meaningless and can run off the end of the
    # page table; clip so the gather stays in bounds and let `valid` drop
    # the row, the same way the sparse MLA writer does.
    max_pages_per_seq = local_page_indices.shape[1]
    virtual_page = jnp.clip(pos // virtual_page_size, 0, max_pages_per_seq - 1)

    valid = jnp.logical_and(in_batch, owner == rank)
    page = jnp.where(
        valid, local_page_indices[seq_id, virtual_page], jnp.int32(OOB_PAGE)
    )
    return cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)
