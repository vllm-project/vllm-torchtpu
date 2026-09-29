# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Hardware-adapted neural network layers and operators for TPU.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.layers import attention
from vllm_torchtpu.llm_tpu_core.layers.fused_moe_gmm import (
    _counting_sort_positions,
    _select_ragged_gather_reduce,
    fused_moe_func,
    gmm_wrapper,
    moe_gmm,
    prepare_routed_gmm_inputs,
    quantize_to_native_fp4_kmajor,
    ragged_gather_reduce,
    requant_unpack_kmajor,
    resolve_onehot_permute_threshold,
    unpack_fp4_to_e2m1,
)
from vllm_torchtpu.llm_tpu_core.layers.mamba_state_copy_op import (
    copy_mamba_state_blocks,
    copy_mamba_state_rows,
)
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
    "attention",
    "_counting_sort_positions",
    "_select_ragged_gather_reduce",
    "apply_rotary_pos_emb",
    "copy_mamba_state_blocks",
    "copy_mamba_state_rows",
    "fused_moe_func",
    "gmm_wrapper",
    "moe_gmm",
    "prepare_routed_gmm_inputs",
    "quantize_to_native_fp4_kmajor",
    "ragged_gather_reduce",
    "requant_unpack_kmajor",
    "resolve_onehot_permute_threshold",
    "rotate_gptj",
    "rotate_half",
    "rotate_neox",
    "rowmax_topk",
    "unpack_fp4_to_e2m1",
]
