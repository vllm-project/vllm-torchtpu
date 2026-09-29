# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for tensor sharding utils.

Re-exports core primitives from `vllm_torchtpu.llm_tpu_core.layers.utils`.
"""

from vllm_torchtpu.llm_tpu_core.layers.utils import (
    inverse_reorder_for_sharding,
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation,
)

__all__ = [
    "inverse_reorder_for_sharding",
    "reorder_concatenated_tensor_for_sharding",
    "slice_sharded_tensor_for_concatenation",
]
