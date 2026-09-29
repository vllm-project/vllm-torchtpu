# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.partial_kv_update`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.partial_kv_update import (
    pallas_index_copy_,
    pallas_scatter_slots,
)

__all__ = [
    "pallas_index_copy_",
    "pallas_scatter_slots",
]
