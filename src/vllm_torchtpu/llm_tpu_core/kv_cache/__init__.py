# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""KV cache geometry, packing, and materialization logic for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kv_cache.block_major_pool import (
    BlockMajorPoolLayout,
    flat_kernel_ids,
    flat_manager_ids,
    is_block_major_layout_applicable,
    kernel_view,
)
from vllm_torchtpu.llm_tpu_core.kv_cache.kv_cache_materializer import (
    LayerKVCache,
    MaterializedKVCache,
    format_kv_cache_layout_summary,
    layer_to_pool_index,
    materialize_kv_cache_tensors,
    resolve_block_major_pool_layout,
)
from vllm_torchtpu.llm_tpu_core.kv_cache.specs import (
    AttentionGroup,
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

__all__ = [
    "AttentionGroup",
    "AttentionSpec",
    "BlockMajorPoolLayout",
    "EncoderOnlyAttentionSpec",
    "KVCacheConfig",
    "KVCacheSpec",
    "LayerKVCache",
    "MLAAttentionSpec",
    "MambaSpec",
    "MaterializedKVCache",
    "UniformTypeKVCacheSpecs",
    "flat_kernel_ids",
    "flat_manager_ids",
    "format_kv_cache_layout_summary",
    "is_block_major_layout_applicable",
    "kernel_view",
    "layer_to_pool_index",
    "materialize_kv_cache_tensors",
    "resolve_block_major_pool_layout",
]
