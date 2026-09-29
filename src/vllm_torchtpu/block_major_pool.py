# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Block-major unified KV pool layout helpers.

Re-exports core primitives from `llm_tpu_core.kv_cache.block_major_pool`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_torchtpu import envs
from vllm_torchtpu.llm_tpu_core.kv_cache.block_major_pool import (
    BlockMajorPoolLayout,
    flat_kernel_ids,
    flat_manager_ids,
    is_block_major_layout_applicable,
    kernel_view,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def unified_block_major_enabled(vllm_config: VllmConfig) -> bool:
    """Returns whether unified hybrid pools are merged into one block-major pool.

    Resolved purely from `VllmConfig` so the scheduler process, runner buffer
    allocation, and layer kernel construction remain consistent. Enabled only
    for hybrid models on the unified KV layout when `VLLM_TPU_BLOCK_MAJOR_KV`
    is set; dense models retain per-layer caches and the dense block-major bundle.
    """
    if not envs.VLLM_TPU_BLOCK_MAJOR_KV:
        return False
    from vllm_torchtpu.platforms.tpu_block_size_utils import (
        unified_kv_layout_enabled,
    )

    return is_block_major_layout_applicable(
        block_major_flag=True,
        is_unified_kv_layout=unified_kv_layout_enabled(vllm_config),
        is_hybrid=vllm_config.model_config.is_hybrid,
    )


__all__ = [
    "BlockMajorPoolLayout",
    "flat_kernel_ids",
    "flat_manager_ids",
    "kernel_view",
    "unified_block_major_enabled",
]
