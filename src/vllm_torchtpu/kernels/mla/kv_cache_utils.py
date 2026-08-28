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
import dataclasses
import enum

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.mla.sparse import dsa_gather
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel


class KVCacheLayout(enum.Enum):
    """Memory layout of a paged KV cache."""
    # [num_pages, page_size, head_dim] OR
    # [num_pages, page_size, head_dim // TILE_LANE_BYTES, TILE_LANE_BYTES]
    SPARSECORE = "sparsecore"
    # [num_pages, page_size // kv_packing, kv_packing, head_dim]
    TENSORCORE = "tensorcore"


class KVCacheType(enum.Enum):
    """The type of a paged KV cache."""
    NOPE = "nope"
    ROPE = "rope"


@dataclasses.dataclass(frozen=True)
class SparseMLAKVCacheSpec:
    cache_type: KVCacheType
    layout: KVCacheLayout
    num_pages: int
    page_size: int
    head_dim: int
    kv_packing: int

    @classmethod
    def create(cls, cache_type: KVCacheType, layout: KVCacheLayout,
               num_pages: int, page_size: int, head_dim: int,
               kv_packing: int) -> "SparseMLAKVCacheSpec":
        num_pages, packed_page_size, kv_packing, head_dim = (
            sparse_mla_kernel.get_kv_cache_shape(num_pages, page_size,
                                                 head_dim, None, kv_packing))
        return cls(cache_type=cache_type,
                   layout=layout,
                   num_pages=num_pages,
                   page_size=packed_page_size * kv_packing,
                   head_dim=head_dim,
                   kv_packing=kv_packing)

    @property
    def shape(self) -> tuple[int, ...]:
        # Tensorcore memory layout: Packs 'kv_packing' tokens into 32-bit words.
        # The physical memory layout is tiled.
        # Sparsecore memory layout: each token gets its own addressable row, the
        # physical memory layout is flat and contiguous.
        if self.layout is KVCacheLayout.TENSORCORE:
            return (self.num_pages, self.page_size // self.kv_packing,
                    self.kv_packing, self.head_dim)
        if self.layout is KVCacheLayout.SPARSECORE:
            if self.cache_type is KVCacheType.NOPE:
                lane_bytes = dsa_gather.TILE_LANE_BYTES
                return (self.num_pages, self.page_size,
                        self.head_dim // lane_bytes, lane_bytes)
            if self.cache_type is KVCacheType.ROPE:
                return (self.num_pages, self.page_size, self.head_dim)
            raise ValueError(f"unsupported cache type {self.cache_type}")
        raise ValueError(f"unsupported cache layout {self.layout}")


def _scatter_rows(cache: jax.Array, spec: SparseMLAKVCacheSpec,
                  values: jax.Array, page: jax.Array,
                  slot: jax.Array) -> jax.Array:
    """Write each token's `values` into the (`page`, `slot`) it lands in."""
    values_u8 = jax.lax.bitcast_convert_type(values, jnp.uint8)
    pad = spec.head_dim - values_u8.shape[-1]
    if pad:
        values_u8 = jnp.concatenate(
            [values_u8,
             jnp.zeros((values_u8.shape[0], pad), jnp.uint8)],
            axis=-1)

    if spec.layout is KVCacheLayout.TENSORCORE:
        return cache.at[page, slot // spec.kv_packing,
                        slot % spec.kv_packing].set(values_u8)
    return cache.at[page, slot].set(
        values_u8.reshape(values_u8.shape[0], *cache.shape[2:]))


def update_sparse_mla_kv_cache(
        kv_cache_nope: jax.Array, kv_cache_rope: jax.Array,
        kv_c_normed: jax.Array, k_pe: jax.Array, seq_lens: jax.Array,
        block_tables: jax.Array, query_start_loc: jax.Array, *,
        nope_spec: SparseMLAKVCacheSpec,
        rope_spec: SparseMLAKVCacheSpec) -> tuple[jax.Array, jax.Array]:
    """Scatter this step's new MLA latents into the split sparse-MLA cache.

    Args:
      kv_cache_nope: uint8 nope cache, shaped by `nope_spec`.
      kv_cache_rope: uint8 rope cache, shaped by `rope_spec`.
      kv_c_normed: This step's compressed KV latents (`[num_tokens, lkv_dim]`).
      k_pe: Decoupled RoPE keys (`[num_tokens, rope_dim]`).
      seq_lens: Per-sequence total KV length including the tokens being inserted in this step (`[num_seqs]`).
      block_tables: Flattened per-sequence-padded page table (`[num_seqs * pages_per_seq]`).
      query_start_loc: Cumulative new-token counts (`[num_seqs + 1]`).
      nope_spec: Layout descriptor the nope cache was allocated from.
      rope_spec: Layout descriptor the rope cache was allocated from.

    Returns:
      The updated (nope, rope) kv caches, same shapes/dtypes as the inputs.
    """
    assert kv_c_normed.dtype == jnp.float8_e4m3fn, (
        "sparse MLA kernel requires --kv-cache-dtype fp8 (got "
        f"{kv_c_normed.dtype})")
    assert (kv_cache_nope.dtype == jnp.uint8
            and kv_cache_rope.dtype == jnp.uint8)
    assert kv_cache_nope.shape == nope_spec.shape, (
        f"nope cache {kv_cache_nope.shape} does not match its spec "
        f"{nope_spec.shape}")
    assert kv_cache_rope.shape == rope_spec.shape, (
        f"rope cache {kv_cache_rope.shape} does not match its spec "
        f"{rope_spec.shape}")

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
    page_size = nope_spec.page_size
    page = jnp.where(valid, block_tables_2d[seq_id, pos // page_size],
                     OOB_PAGE)
    slot = pos % page_size

    return (_scatter_rows(kv_cache_nope, nope_spec, kv_c_normed, page, slot),
            _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot))
