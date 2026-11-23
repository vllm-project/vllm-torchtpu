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

from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter_utils


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
    if spec.layout is kv_cache_utils.KVCacheLayout.TENSORCORE:
        src = kv_cache_utils.as_token_bytes(values, spec.token_bytes)
    else:
        src = kv_cache_utils.pack_tokens(values, spec.token_bytes)
    return scatter_utils.scatter_rows(
        cache,
        src,
        dst_rows,
        name=f"scatter_{spec.cache_type.value}_{spec.layout.value}_cache",
    )


def scatter_pair(
    nope_cache: jax.Array,
    kv_c_normed: jax.Array,
    nope_spec: kv_cache_utils.SparseMLAKVCacheSpec,
    rope_cache: jax.Array,
    k_pe: jax.Array,
    rope_spec: kv_cache_utils.SparseMLAKVCacheSpec,
    dst_rows: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Packs both latents and writes them with one SparseCore program.

    Gives the same caches as `scatter` on the nope cache followed by `scatter`
    on the rope cache, with the same `dst_rows`. Both specs must use the
    SPARSECORE layout, because the pair writer supports only the uint32 cache.

    Args:
      nope_cache: nope KV cache, allocated as `nope_spec.shape` /
        `nope_spec.jax_dtype`.
      kv_c_normed: [num_tokens, unpadded nope head_dim] fp8 latents.
      nope_spec: layout descriptor the nope cache was allocated from.
      rope_cache: rope KV cache, allocated as `rope_spec.shape` /
        `rope_spec.jax_dtype`.
      k_pe: [num_tokens, unpadded rope head_dim] fp8 latents.
      rope_spec: layout descriptor the rope cache was allocated from.
      dst_rows: [num_tokens] destination row per token, the same in both
        caches; a negative or out-of-range row drops the token.

    Returns:
      The updated (nope_cache, rope_cache), same shapes and dtypes as the
      inputs.
    """
    for cache, spec in ((nope_cache, nope_spec), (rope_cache, rope_spec)):
        assert cache.shape == spec.shape and cache.dtype == spec.jax_dtype, (
            f"cache {cache.shape} {cache.dtype} does not match spec"
            f" {spec.shape} {spec.jax_dtype}"
        )
        assert spec.layout is kv_cache_utils.KVCacheLayout.SPARSECORE, (
            f"the pair writer needs SPARSECORE caches, got {spec.layout}"
        )
    return scatter_utils.scatter_rows_pair(
        nope_cache,
        kv_cache_utils.pack_tokens(kv_c_normed, nope_spec.token_bytes),
        rope_cache,
        kv_cache_utils.pack_tokens(k_pe, rope_spec.token_bytes),
        dst_rows,
        name="scatter_nope_rope_sparsecore_cache",
    )
