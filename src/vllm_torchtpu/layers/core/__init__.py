# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Core layers package for vllm_torchtpu.

Re-exports core primitives from `llm_tpu_core.layers`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.layers import (
    NEG,
    MoERouter,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TopKResult,
    apply_rotary_pos_emb,
    rowmax_topk,
)

__all__ = [
    "NEG",
    "MoERouter",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TopKResult",
    "apply_rotary_pos_emb",
    "rowmax_topk",
]
