# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""KV cache materializer wrapper for vllm_torchtpu.

Re-exports core materialization logic from `llm_tpu_core.kv_cache.kv_cache_materializer`
for seamless backward compatibility.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kv_cache.kv_cache_materializer import (
    LayerKVCache,
    MaterializedKVCache,
    allocate_raw_kv_cache_tensors,
    build_kernel_block_size_by_group_id,
    can_share_attention_cache,
    format_kv_cache_layout_summary,
    layer_to_pool_index,
    make_attention_cache_tensor,
    materialize_kv_cache_tensors,
    materialize_shared_attention_cache,
    resolve_block_major_pool_layout,
)

__all__ = [
    "LayerKVCache",
    "MaterializedKVCache",
    "allocate_raw_kv_cache_tensors",
    "build_kernel_block_size_by_group_id",
    "can_share_attention_cache",
    "format_kv_cache_layout_summary",
    "layer_to_pool_index",
    "make_attention_cache_tensor",
    "materialize_kv_cache_tensors",
    "materialize_shared_attention_cache",
    "resolve_block_major_pool_layout",
]
