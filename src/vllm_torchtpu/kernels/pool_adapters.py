# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.pool_adapters`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.pool_adapters import (
    copy_blocks,
    gather_region,
    is_seq_along_lane_pool,
    scatter_region,
    v3_state_source,
)

__all__ = [
    "copy_blocks",
    "gather_region",
    "is_seq_along_lane_pool",
    "scatter_region",
    "v3_state_source",
]
