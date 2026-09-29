# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.wrapper`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.wrapper import (
    fused_conv1d_gdn,
    inner_kernel,
    outer_kernel,
)

__all__ = [
    "fused_conv1d_gdn",
    "inner_kernel",
    "outer_kernel",
]
