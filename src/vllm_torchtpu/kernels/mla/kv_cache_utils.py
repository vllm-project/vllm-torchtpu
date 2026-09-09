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
import torch
from torch_tpu._internal.pallas import pallas

from vllm_torchtpu.kernels.mla.sparse import dsa_gather
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

_WORD_BYTES = 4


class KVCacheLayout(enum.Enum):
    """Physical HBM layout of a paged KV cache.

    TENSORCORE: uint8, 2D-tiled. 4 sublanes fold into 32-bit words via hardware
    TC tiling.
    SPARSECORE: uint32, 1D-flat. The writer explicitly bit-packs `_WORD_BYTES`
    byte bands into 32-bit words (_pack4) for native SC DMA.
    """
    SPARSECORE = "sparsecore"
    TENSORCORE = "tensorcore"


class KVCacheType(enum.Enum):
    """The type of a paged KV cache."""
    NOPE = "nope"
    ROPE = "rope"


_LAYOUT_DTYPES = {
    KVCacheLayout.TENSORCORE: torch.uint8,
    KVCacheLayout.SPARSECORE: torch.uint32,
}


@dataclasses.dataclass(frozen=True)
class SparseMLAKVCacheSpec:
    cache_type: KVCacheType
    layout: KVCacheLayout
    num_pages: int
    page_size: int
    head_dim: int
    kv_packing: int

    @property
    def dtype(self) -> torch.dtype:
        return _LAYOUT_DTYPES[self.layout]

    @property
    def jax_dtype(self) -> jnp.dtype:
        return pallas.TORCH_TO_JAX_DTYPE_MAP[self.dtype]

    @classmethod
    def create(cls, cache_type: KVCacheType, layout: KVCacheLayout,
               num_pages: int, page_size: int, head_dim: int,
               kv_packing: int) -> "SparseMLAKVCacheSpec":
        num_pages, packed_page_size, kv_packing, head_dim = (
            sparse_mla_kernel.get_kv_cache_shape(num_pages, page_size,
                                                 head_dim, None, kv_packing))
        page_size = packed_page_size * kv_packing

        if layout is KVCacheLayout.SPARSECORE:
            assert head_dim % _WORD_BYTES == 0, (
                f"head_dim {head_dim} must be a multiple of {_WORD_BYTES}")
            if cache_type is KVCacheType.ROPE:
                assert page_size % _WORD_BYTES == 0, (
                    f"page_size {page_size} must be a multiple of "
                    f"{_WORD_BYTES}")

        return cls(cache_type=cache_type,
                   layout=layout,
                   num_pages=num_pages,
                   page_size=page_size,
                   head_dim=head_dim,
                   kv_packing=kv_packing)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.cache_type is KVCacheType.NOPE:
            if self.layout is KVCacheLayout.TENSORCORE:
                return (self.num_pages, self.page_size,
                        self.head_dim // dsa_gather.TILE_LANE_BYTES,
                        dsa_gather.TILE_LANE_BYTES)
            return (self.num_pages, self.page_size,
                    self.head_dim // _WORD_BYTES)
        if self.cache_type is KVCacheType.ROPE:
            if self.layout is KVCacheLayout.TENSORCORE:
                return (self.num_pages, self.page_size // self.kv_packing,
                        self.kv_packing, self.head_dim)
            return (self.num_pages, self.page_size // _WORD_BYTES,
                    self.head_dim)
        raise ValueError(f"unsupported cache type {self.cache_type}")


def _pack4(values_u8: jax.Array) -> jax.Array:
    """Pack four uint8 bands `[..., 4, n]` into uint32 words `[..., n]`."""
    bands = values_u8.astype(jnp.uint32)
    return (bands[..., 0, :] | (bands[..., 1, :] << 8)
            | (bands[..., 2, :] << 16) | (bands[..., 3, :] << 24))


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
    num_tokens = values_u8.shape[0]

    if spec.layout is KVCacheLayout.TENSORCORE:
        if spec.cache_type is KVCacheType.NOPE:
            return cache.at[page, slot].set(
                values_u8.reshape(num_tokens, *cache.shape[2:]))
        return cache.at[page, slot // spec.kv_packing,
                        slot % spec.kv_packing].set(values_u8)
    elif spec.layout is KVCacheLayout.SPARSECORE:
        packed = _pack4(values_u8.reshape(num_tokens, _WORD_BYTES, -1))
        rows = cache.reshape(spec.num_pages, spec.page_size, -1)
        return rows.at[page, slot].set(packed).reshape(cache.shape)

    raise ValueError(f"unsupported layout {spec.layout}")


def update_sparse_mla_kv_cache(
        kv_cache_nope: jax.Array, kv_cache_rope: jax.Array,
        kv_c_normed: jax.Array, k_pe: jax.Array, seq_lens: jax.Array,
        block_tables: jax.Array, query_start_loc: jax.Array, *,
        nope_spec: SparseMLAKVCacheSpec,
        rope_spec: SparseMLAKVCacheSpec) -> tuple[jax.Array, jax.Array]:
    """Scatter this step's new MLA latents into the split sparse-MLA cache.

    Args:
      kv_cache_nope: Nope cache, shaped and typed by `nope_spec`.
      kv_cache_rope: Rope cache, shaped and typed by `rope_spec`.
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
    assert kv_cache_nope.dtype == nope_spec.jax_dtype, (
        f"nope cache {kv_cache_nope.dtype} does not match its spec "
        f"{nope_spec.jax_dtype}")
    assert kv_cache_rope.dtype == rope_spec.jax_dtype, (
        f"rope cache {kv_cache_rope.dtype} does not match its spec "
        f"{rope_spec.jax_dtype}")
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
    assert nope_spec.page_size == rope_spec.page_size, (
        f"nope page size {nope_spec.page_size} != rope page size "
        f"{rope_spec.page_size}")
    page_size = nope_spec.page_size
    page = jnp.where(valid, block_tables_2d[seq_id, pos // page_size],
                     OOB_PAGE)
    slot = pos % page_size

    return (_scatter_rows(kv_cache_nope, nope_spec, kv_c_normed, page, slot),
            _scatter_rows(kv_cache_rope, rope_spec, k_pe, page, slot))
