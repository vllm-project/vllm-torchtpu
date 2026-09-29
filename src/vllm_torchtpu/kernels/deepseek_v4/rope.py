# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper for DeepSeek V4 rope kernel.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.rope`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.rope import (
    DEFAULT_VMEM_LIMIT_BYTES,
    LANE,
    CosSinRef,
    _apply_rope_2lane_body,
    _apply_rope_2lane_pipeline,
    _cos_sin_lanes,
    _rotate_gptj,
    apply_rope_2lane,
)

__all__ = [
    "DEFAULT_VMEM_LIMIT_BYTES",
    "LANE",
    "CosSinRef",
    "_apply_rope_2lane_body",
    "_apply_rope_2lane_pipeline",
    "_cos_sin_lanes",
    "_rotate_gptj",
    "apply_rope_2lane",
]
