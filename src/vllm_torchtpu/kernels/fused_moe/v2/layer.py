# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.fused_moe.v2.layer`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.fused_moe.v2.layer import (
    AGQ_MIN_LOCAL_TOKENS,
    SCORING_FNS,
    fused_ep_moe_v2,
)

__all__ = [
    "AGQ_MIN_LOCAL_TOKENS",
    "SCORING_FNS",
    "fused_ep_moe_v2",
]
