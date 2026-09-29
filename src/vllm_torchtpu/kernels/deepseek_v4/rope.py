# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.rope`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.deepseek_v4.rope import (
    DEFAULT_VMEM_LIMIT_BYTES,
    LANE,
    CosSinRef,
    qnorm_rope,
    rope,
    rope_quant,
)

__all__ = [
    "CosSinRef",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "LANE",
    "qnorm_rope",
    "rope",
    "rope_quant",
]
