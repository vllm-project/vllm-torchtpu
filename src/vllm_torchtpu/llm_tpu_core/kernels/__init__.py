# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Pallas kernels and hardware acceleration primitives for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.kernels import pool_adapters
from vllm_torchtpu.llm_tpu_core.kernels.block_sparse import (
    compute_sparse_block_offsets,
)
from vllm_torchtpu.llm_tpu_core.kernels.paged_attention import (
    compute_paged_attention_shape,
)
from vllm_torchtpu.llm_tpu_core.kernels.partial_kv_update import (
    pallas_scatter_slots,
)
from vllm_torchtpu.llm_tpu_core.kernels.router_topk import (
    router_topk,
)
from vllm_torchtpu.llm_tpu_core.kernels.rpa import (
    AttentionScope,
    BlockSizes,
    KVLayout,
    ModelConfigs,
    RpaCase,
    RpaConfigs,
    ServingConfigs,
)

__all__ = [
    "AttentionScope",
    "BlockSizes",
    "KVLayout",
    "ModelConfigs",
    "RpaCase",
    "RpaConfigs",
    "ServingConfigs",
    "compute_paged_attention_shape",
    "compute_sparse_block_offsets",
    "pallas_scatter_slots",
    "pool_adapters",
    "router_topk",
]
