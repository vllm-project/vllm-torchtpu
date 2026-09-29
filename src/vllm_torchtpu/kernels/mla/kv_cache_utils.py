# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.mla.kv_cache_utils`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.mla.kv_cache_utils import (
    OOB_PAGE,
    SKIP_ROW,
    TILE_LANE_BYTES,
    WORD_BYTES,
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
    _check_inputs,
    _row_positions,
    _scatter_rows,
    align_to,
    as_token_bytes,
    get_dst_rows,
    get_dtype_bitwidth,
    get_dtype_packing,
    get_kv_cache_shape,
    get_page_and_slot,
    pack_tokens,
    update_sparse_mla_kv_cache,
    update_sparse_mla_kv_cache_dcp,
    update_sparse_mla_kv_cache_dcp_jax,
    update_sparse_mla_kv_cache_jax,
)

__all__ = [
    "KVCacheLayout",
    "KVCacheType",
    "OOB_PAGE",
    "SKIP_ROW",
    "SparseMLAKVCacheSpec",
    "TILE_LANE_BYTES",
    "WORD_BYTES",
    "_check_inputs",
    "_row_positions",
    "_scatter_rows",
    "align_to",
    "as_token_bytes",
    "get_dst_rows",
    "get_dtype_bitwidth",
    "get_dtype_packing",
    "get_kv_cache_shape",
    "get_page_and_slot",
    "pack_tokens",
    "update_sparse_mla_kv_cache",
    "update_sparse_mla_kv_cache_dcp",
    "update_sparse_mla_kv_cache_dcp_jax",
    "update_sparse_mla_kv_cache_jax",
]
