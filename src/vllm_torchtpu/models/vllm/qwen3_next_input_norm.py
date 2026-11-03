# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Move Qwen3-Next / Qwen3.5 input RMSNorms into the FP8 input projections.

A decoder layer's input norm feeds nothing but its attention's input
projections. When those run the FP8 linear method, they can normalize their
own input inside the matmul kernel (``INPUT_RMS_NORM_ATTR``), and the norm
then only adds the residual, so the normalized activation is never written to
HBM.
"""

import torch
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
)

from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)
from vllm_torchtpu.layers.adapter.layernorm import TpuGemmaRMSNorm
from vllm_torchtpu.layers.adapter.linear_common import INPUT_RMS_NORM_ATTR
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _runs_fp8(proj: torch.nn.Module | None) -> bool:
    return isinstance(getattr(proj, "quant_method", None), VllmFp8LinearMethodTPU)


def _fold(layer: Qwen3NextDecoderLayer, norm: TpuGemmaRMSNorm) -> bool:
    # Exact attention types: a subclass may read `hidden_states` beyond the
    # projections that normalize it.
    if layer.layer_type == "full_attention":
        attn = layer.self_attn
        if type(attn) is not Qwen3NextAttention or not _runs_fp8(attn.qkv_proj):
            return False
        # Not registered as a submodule: the weight keeps its one name.
        object.__setattr__(attn.qkv_proj, INPUT_RMS_NORM_ATTR, norm)
        return True
    if layer.layer_type == "linear_attention":
        attn = layer.linear_attn
        if type(attn) is not VllmGatedDeltaNetAttention:
            return False
        return _runs_fp8(getattr(attn, "in_proj_qkvz", None)) and (
            attn.fold_input_norm(norm)
        )
    return False


def fold_input_norms(model: torch.nn.Module) -> int:
    """Fold every eligible decoder-layer input norm of `model` into its
    attention's input projections; returns how many were folded. Call before
    the first forward."""
    folded = 0
    for layer in model.modules():
        if not isinstance(layer, Qwen3NextDecoderLayer):
            continue
        norm = layer.input_layernorm
        if not isinstance(norm, TpuGemmaRMSNorm) or not norm.can_defer():
            continue
        if _fold(layer, norm):
            norm.defer_to_consumers()
            folded += 1
    if folded:
        logger.info("Folded %d input RMSNorms into FP8 input projections.", folded)
    return folded
