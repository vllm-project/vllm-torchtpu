# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for DeepSeek V4 o_projection.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.o_projection`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.o_projection import (
    DEFAULT_VMEM_LIMIT_BYTES,
    LANE,
    SUBLANE,
    CosSinRef,
    _cos_sin_body,
    _cos_sin_kernel,
    _kernel,
    _largest_divisor,
    _rope_heads,
    fused_reverse_rope_wo_a_projection,
    gather_cos_sin,
    wo_a_projection,
)

__all__ = [
    "DEFAULT_VMEM_LIMIT_BYTES",
    "LANE",
    "SUBLANE",
    "CosSinRef",
    "_cos_sin_body",
    "_cos_sin_kernel",
    "_kernel",
    "_largest_divisor",
    "_rope_heads",
    "fused_reverse_rope_wo_a_projection",
    "gather_cos_sin",
    "wo_a_projection",
]
