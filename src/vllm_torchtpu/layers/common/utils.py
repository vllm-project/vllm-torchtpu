# Copyright 2025 Google LLC
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

import jax
import jax.numpy as jnp


def reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array,
                                             split_sizes: list[int],
                                             n_shards: int, dim: int):
    """
    Reorder a replicated concatenated tensor such that when sharded on multiple chips, each shard is a concatenation of the shards of the individual tensors.
    For example, let the concatenated_tensor be:
        AAAAAAAAAAAABBBBBBBBCCCC
            12 As     8 Bs  4 Cs
    and let the split_sizes = [12, 8, 4] and n_shards = 4.
    The output is:
        AAABBCAAABBCAAABBCAAABBC
    In other words, it reorders the input tensor into 4 segements, with each segment corresponding to a shard and being AAABBC.
    Args:
        concatenated_tensor: the tensor, concatenated on the dimension specified by `dim`.
        split_sizes: each individual tensor's size on the dimension specified by `dim`.
        n_shards: num of shards.
        dim: the dimension on which the concatenated_tensor is concatenated.
    """
    # Split the concatenated tensor into individual tensors.
    if dim < 0:
        dim += concatenated_tensor.ndim
    split_tensors = []
    start_offset = 0
    old_shape = concatenated_tensor.shape
    # New shape ensures each split_tensor[i] maps to a tensor in ith shards
    new_shape = old_shape[:dim] + (n_shards, -1) + old_shape[dim + 1:]
    for split_size in split_sizes:
        split_tensor = jax.lax.slice_in_dim(concatenated_tensor,
                                            start_offset,
                                            start_offset + split_size,
                                            axis=dim)
        split_tensors.append(split_tensor.reshape(new_shape))
        start_offset += split_size
    # While maintaining 0th dim as a shard dim, we concatenate along 1th dim to
    # to create concatenated tnensor where 0th dim maps to shard dim.
    reordered_tensor = jnp.concatenate(split_tensors, axis=dim + 1)
    return reordered_tensor.reshape(old_shape)


def inverse_reorder_for_sharding(reordered_tensor: jax.Array,
                                 split_sizes: list[int], n_shards: int,
                                 dim: int):
    """Inverse of reorder_concatenated_tensor_for_sharding."""
    if dim < 0:
        dim += reordered_tensor.ndim

    old_shape = reordered_tensor.shape
    shard_split_sizes = []
    for split_size in split_sizes:
        assert split_size % n_shards == 0
        shard_split_sizes.append(split_size // n_shards)

    shard_size = sum(shard_split_sizes)
    new_shape = old_shape[:dim] + (n_shards, shard_size) + old_shape[dim + 1:]
    reshaped = reordered_tensor.reshape(new_shape)

    split_tensors = []
    start_offset = 0
    for shard_split_size in shard_split_sizes:
        split_tensor = jax.lax.slice_in_dim(reshaped,
                                            start_offset,
                                            start_offset + shard_split_size,
                                            axis=dim + 1)
        split_shape = (old_shape[:dim] + (n_shards * shard_split_size, ) +
                       old_shape[dim + 1:])
        split_tensors.append(split_tensor.reshape(split_shape))
        start_offset += shard_split_size

    return jnp.concatenate(split_tensors, axis=dim)


def slice_sharded_tensor_for_concatenation(sharded_tensor: jax.Array,
                                           split_sizes: list[int],
                                           n_shards: int):
    """
    Slice the input tensor which is sharded on multiple chips (on the last dim) into individual tensors with the same sharding.
    For example, let the sharded_tensor be:
        AAABBC | AAABBC | AAABBC | AAABBC
        Shard0   Shard1   Shard2   Shard3
    and let the split_sizes = [12, 8, 4] and n_shards = 4.
    The output is a list of 3 tensors:
         AAA   |  AAA   |  AAA   |  AAA
          BB   |   BB   |   BB   |   BB
           C   |    C   |    C   |    C
        Shard0   Shard1   Shard2   Shard3
    In other words, each individual tensor is a slice of the input tensor with the same sharding.
    Args:
        sharded_tensor: the input tensor, sharded on the last dim.
        split_sizes: each individual tensor's size on the last dim.
        n_shards: num of shards.
    """
    new_shape = sharded_tensor.shape[:-1] + (n_shards, -1)
    # New shape ensures each sharded_tensor[:, i] maps to a tensor in ith shards
    sharded_tensor = sharded_tensor.reshape(new_shape)

    split_tensors = []
    start_offset = 0
    for split_size in split_sizes:
        assert split_size % n_shards == 0
        sz = split_size // n_shards  # size of this split tensor per shard
        end_offset = start_offset + sz
        # Because we are slicing over last dim, sharding dim remains intact.
        # Therefore, splitting happens locally.
        split_tensor = sharded_tensor[..., start_offset:end_offset]
        split_tensors.append(split_tensor.reshape(new_shape[:-2] + (-1, )))
        start_offset = end_offset

    return split_tensors


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


def repack_sparse_mla_cache(
        kv_cache: jax.Array, lkv_dim: int, rope_dim: int,
        dequant_scale: float) -> tuple[jax.Array, jax.Array]:
    """Repack the fp8 cache (pages, page_size // kv_packing, kv_packing, padded_concat_head_dim)
    into the tiled uint8 pair csa_gather requires; see the layout contract
    constants (TILE_SUBROWS, TILE_LANE_BYTES) in csa_gather.py.
    """
    from vllm_torchtpu.kernels.mla.sparse import csa_gather

    assert kv_cache.dtype == jnp.float8_e4m3fn, (
        "sparse MLA kernel requires --kv-cache-dtype fp8 (got "
        f"{kv_cache.dtype}); the nope conversion is a pure bitcast.")

    tile_subrows = csa_gather.TILE_SUBROWS
    lane_bytes = csa_gather.TILE_LANE_BYTES

    # `csa_gather` moves one (4, 128) uint8 tile of nope and one 128-byte lane
    # row of rope per token.
    assert (
        lkv_dim == tile_subrows * lane_bytes and rope_dim * 2 == lane_bytes
    ), ("csa_gather used in the sparse MLA kernel needs the fp8 nope head "
        f"dimension to be {tile_subrows * lane_bytes} and the bf16 rope head "
        f"dimension to be {lane_bytes // 2}, got {lkv_dim}+{rope_dim}")

    pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1] * kv_cache.shape[2]
    flat = kv_cache.reshape(pages, page_size, kv_cache.shape[-1])

    nope_u8 = jax.lax.bitcast_convert_type(
        flat[..., :lkv_dim],
        jnp.uint8).reshape(pages, page_size, lkv_dim // lane_bytes, lane_bytes)

    # TODO(gxd3): if GLM 5.2 keeps rope in fp8 like nope, both sides get cheaper:
    # this repack becomes a plain bitcast, and csa_gather's `process_rope` drops
    # the per-element shift/mask/or that rebuilds (1, 128) uint8 into (1, 64)
    # bf16, leaving the SparseCore to just move bytes.
    rope_bf16 = flat[..., lkv_dim:lkv_dim + rope_dim].astype(jnp.bfloat16)
    if dequant_scale != 1.0:
        rope_bf16 = rope_bf16 * jnp.bfloat16(dequant_scale)
    rope_u16 = jax.lax.bitcast_convert_type(rope_bf16, jnp.uint16)
    rope_u8 = jnp.concatenate([(rope_u16 >> 8).astype(jnp.uint8),
                               (rope_u16 & 0xFF).astype(jnp.uint8)],
                              axis=-1).reshape(pages,
                                               page_size // tile_subrows,
                                               tile_subrows, lane_bytes)
    return nope_u8, rope_u8
