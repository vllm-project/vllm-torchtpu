# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for multi-layer KV scatter.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.kv_transfer.kv_scatter`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.kv_transfer.kv_scatter import (
    multi_layer_scatter_into,
    prepare_scatter_args,
    scatter_available,
    smoke_test_multi_layer_scatter,
)

__all__ = [
    "multi_layer_scatter_into",
    "prepare_scatter_args",
    "scatter_available",
    "smoke_test_multi_layer_scatter",
]
