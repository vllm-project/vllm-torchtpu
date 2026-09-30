# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Top-level attention interface functions for hardware-accelerated TPU attention.

Re-exports core primitives from `llm_tpu_core.layers.attention.interface`.
"""

from vllm_torchtpu.llm_tpu_core.layers.attention.interface import (
    MAX_ALLOWED_PAGE_INDICES_N,
    AttentionMetadata,
    apply_splash,
    attention,
    attention_bundled,
    get_kv_cache_shape,
    get_kv_cache_shape_hd64,
    mla_attention,
    mla_ragged_paged_attention,
    paged_attention_with_guarded_smem,
    ragged_paged_attention,
    ragged_paged_attention_batched,
    ragged_paged_attention_hd64,
    sharded_flash_attention,
    sharded_paged_attention,
    sharded_ragged_paged_attention,
    sharded_splash_attention,
    sparse_mla_attention,
    sparse_mla_attention_dcp,
    sparse_mla_ragged_paged_attention,
    update_cache,
)

__all__ = [
    "MAX_ALLOWED_PAGE_INDICES_N",
    "AttentionMetadata",
    "apply_splash",
    "attention",
    "attention_bundled",
    "get_kv_cache_shape",
    "get_kv_cache_shape_hd64",
    "mla_attention",
    "mla_ragged_paged_attention",
    "paged_attention_with_guarded_smem",
    "ragged_paged_attention",
    "ragged_paged_attention_batched",
    "ragged_paged_attention_hd64",
    "sharded_flash_attention",
    "sharded_paged_attention",
    "sharded_ragged_paged_attention",
    "sharded_splash_attention",
    "sparse_mla_attention",
    "sparse_mla_attention_dcp",
    "sparse_mla_ragged_paged_attention",
    "update_cache",
]
