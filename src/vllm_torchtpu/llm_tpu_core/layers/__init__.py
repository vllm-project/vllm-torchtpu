# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware-adapted neural network layers and operators for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.layers.moe import (
    NEG,
    MoERouter,
    TopKResult,
    rowmax_topk,
)
from vllm_torchtpu.llm_tpu_core.layers.rms_norm import (
    RMSNorm,
)
from vllm_torchtpu.llm_tpu_core.layers.rotary import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_gptj,
    rotate_half,
    rotate_neox,
)
from vllm_torchtpu.llm_tpu_core.layers.swiglu import (
    SwiGLU,
)

__all__ = [
    "NEG",
    "MoERouter",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TopKResult",
    "apply_rotary_pos_emb",
    "rotate_gptj",
    "rotate_half",
    "rotate_neox",
    "rowmax_topk",
]
