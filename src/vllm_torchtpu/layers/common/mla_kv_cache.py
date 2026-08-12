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
"""Paged KV cache write helpers for the MLA attention interfaces."""

import jax
import jax.numpy as jnp


def update_sparse_mla_kv_cache(kv_cache: jax.Array, kv_c_normed: jax.Array,
                               k_pe: jax.Array, seq_lens: jax.Array,
                               block_tables: jax.Array,
                               query_start_loc: jax.Array) -> jax.Array:
    """Scatter this step's new MLA latents into the paged fp8 KV cache.

    Args:
      kv_cache: Paged MLA latent cache (`[num_blocks, block_size // kv_packing, kv_packing, padded_row]`).
      kv_c_normed: This step's compressed KV latents (`[num_tokens, lkv_dim]`).
      k_pe: Decoupled RoPE keys (`[num_tokens, rope_dim]`).
      seq_lens: Per-sequence total KV length including the tokens being inserted in this step (`[num_seqs]`).
      block_tables: Flattened per-sequence-padded page table (`[num_seqs * pages_per_seq]`).
      query_start_loc: Cumulative new-token counts (`[num_seqs + 1]`).

    Returns:
      The updated cache, same shape/dtype as `kv_cache`.
    """
    num_tokens = kv_c_normed.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id = jnp.searchsorted(query_start_loc[1:], tok,
                              side="right").astype(jnp.int32)
    q_len = query_start_loc[seq_id + 1] - query_start_loc[seq_id]
    local = tok - query_start_loc[seq_id]
    pos = seq_lens[seq_id] - q_len + local
    valid = tok < query_start_loc[-1]

    num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(num_seqs, -1)

    # page index for padded tokens, XLA drops out-of-bounds scatter writes.
    OOB_PAGE = jnp.int32(2**30)
    kv_packing = kv_cache.shape[2]
    page_size = kv_cache.shape[1] * kv_packing
    page = jnp.where(valid, block_tables_2d[seq_id, pos // page_size],
                     OOB_PAGE)
    slot = pos % page_size

    new_kv_cache_rows = jnp.concatenate([kv_c_normed, k_pe], axis=-1)
    return kv_cache.at[page, slot // kv_packing,
                       slot % kv_packing, :new_kv_cache_rows.shape[-1]].set(
                           new_kv_cache_rows)
