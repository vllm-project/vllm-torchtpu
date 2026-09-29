# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware-adapted layers for vllm_torchtpu.

Re-exports core primitives from `llm_tpu_core.layers`.
"""

from __future__ import annotations

# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` only affects layers built after the import, so
# this module must be imported before any model is constructed.
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.llm_tpu_core.layers import (
    NEG,
    MoERouter,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TopKResult,
    apply_rotary_pos_emb,
    rotate_gptj,
    rotate_half,
    rotate_neox,
    rowmax_topk,
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers


def register_layers() -> None:
    register_models()
    register_tokenizers()


__all__ = [
    "NEG",
    "MoERouter",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TopKResult",
    "apply_rotary_pos_emb",
    "attention",
    "linear",
    "register_layers",
    "rotate_gptj",
    "rotate_half",
    "rotate_neox",
    "rowmax_topk",
]
