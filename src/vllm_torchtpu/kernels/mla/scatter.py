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
"""SparseCore Pallas scatter of MLA latents into the paged KV cache."""

import jax

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter_utils


def _pack(values: jax.Array, spec: kv_cache_utils.SparseMLAKVCacheSpec) -> jax.Array:
    """Converts `values` to the rows that the SparseCore writer copies for `spec`.

    Args:
      values: `[num_tokens, unpadded head_dim]` fp8 latents.
      spec: layout descriptor of the destination cache.

    Returns:
      uint8 `[num_tokens, token_bytes]` rows for the TensorCore layout. uint32
      word rows for the SparseCore layout; with `TPU_KV_PACK_FUSED_PAD=1` they
      are already padded to `scatter_utils.padded_num_tokens(num_tokens)`.
    """
    if spec.layout is kv_cache_utils.KVCacheLayout.TENSORCORE:
        return kv_cache_utils.as_token_bytes(values, spec.token_bytes)
    if envs.TPU_KV_PACK_FUSED_PAD:
        # The SparseCore writer reads a whole number of descriptor blocks, so
        # a 16-token decode step is padded to 32 rows. Pad here, inside the
        # pack, so that one op produces the padded packed rows; scatter_rows
        # then pads only `dst_rows`.
        return kv_cache_utils.pack_tokens(
            values,
            spec.token_bytes,
            num_rows=scatter_utils.padded_num_tokens(values.shape[0]),
        )
    return kv_cache_utils.pack_tokens(values, spec.token_bytes)


def scatter(
    cache: jax.Array,
    values: jax.Array,
    spec: kv_cache_utils.SparseMLAKVCacheSpec,
    seq_lens: jax.Array | None = None,
    block_tables: jax.Array | None = None,
    query_start_loc: jax.Array | None = None,
    dst_rows: jax.Array | None = None,
) -> jax.Array:
    """Packs `values` for `spec`'s layout and scatters one row per token.

    Args:
      cache: paged KV cache in HBM, allocated as `spec.shape` / `spec.jax_dtype`.
      values: [num_tokens, unpadded head_dim] fp8 latents -- `kv_c_normed` for a
        NOPE cache, `k_pe` for a ROPE one.
      spec: layout descriptor the cache was allocated from.
      seq_lens: per-sequence KV length including this step's tokens.
      block_tables: flattened per-sequence-padded page table.
      query_start_loc: cumulative new-token counts, [num_seqs + 1].
      dst_rows: precomputed destination row per token; derived from the three
        arrays above when None.

    Returns:
      The updated cache, same shape and dtype as `cache`.
    """
    assert cache.shape == spec.shape and cache.dtype == spec.jax_dtype, (
        f"cache {cache.shape} {cache.dtype} does not match spec"
        f" {spec.shape} {spec.jax_dtype}"
    )
    if dst_rows is None:
        assert (
            seq_lens is not None
            and block_tables is not None
            and query_start_loc is not None
        ), "pass dst_rows or the addressing arrays"
        dst_rows = kv_cache_utils.get_dst_rows(
            num_tokens=values.shape[0],
            seq_lens=seq_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            page_size=spec.page_size,
        )
    src = _pack(values, spec)
    return scatter_utils.scatter_rows(
        cache,
        src,
        dst_rows,
        name=f"scatter_{spec.cache_type.value}_{spec.layout.value}_cache",
    )


def scatter_pair(
    nope_cache: jax.Array,
    nope_values: jax.Array,
    nope_spec: kv_cache_utils.SparseMLAKVCacheSpec,
    rope_cache: jax.Array,
    rope_values: jax.Array,
    rope_spec: kv_cache_utils.SparseMLAKVCacheSpec,
    dst_rows: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Writes the nope and rope latents of each token with one SparseCore program.

    The result is bit-identical to two `scatter` calls with the same
    `dst_rows`. Each value is packed exactly as `scatter` packs it, including
    the `TPU_KV_PACK_FUSED_PAD` padding, and `scatter_utils.scatter_rows_pair`
    then writes both caches in one kernel.

    Args:
      nope_cache: nope cache in HBM, allocated as `nope_spec`.
      nope_values: `[num_tokens, lkv_dim]` fp8 `kv_c_normed`.
      nope_spec: nope layout descriptor; must be the SparseCore layout.
      rope_cache: rope cache in HBM, allocated as `rope_spec`.
      rope_values: `[num_tokens, rope_dim]` fp8 `k_pe`.
      rope_spec: rope layout descriptor; must be the SparseCore layout.
      dst_rows: `[num_tokens]` destination row per token, shared by both
        caches (`kv_cache_utils.get_dst_rows`).

    Returns:
      The updated `(nope_cache, rope_cache)`.
    """
    for cache, spec in ((nope_cache, nope_spec), (rope_cache, rope_spec)):
        assert spec.layout is kv_cache_utils.KVCacheLayout.SPARSECORE, (
            f"scatter_pair needs SparseCore layouts, got {spec.layout}"
        )
        assert cache.shape == spec.shape and cache.dtype == spec.jax_dtype, (
            f"cache {cache.shape} {cache.dtype} does not match spec"
            f" {spec.shape} {spec.jax_dtype}"
        )
    return scatter_utils.scatter_rows_pair(
        nope_cache,
        _pack(nope_values, nope_spec),
        rope_cache,
        _pack(rope_values, rope_spec),
        dst_rows,
        name="scatter_nope_rope_sparsecore_cache",
    )
