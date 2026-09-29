# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.kimi_k3.kda`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.kimi_k3.kda import (
    kda_step,
    ragged_kda,
)

__all__ = [
    "kda_step",
    "ragged_kda",
]
