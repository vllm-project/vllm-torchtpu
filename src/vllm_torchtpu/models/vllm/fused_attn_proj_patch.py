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
"""TPU_FUSED_ATTN_PROJ: runs decoder layers through the fused_attn_proj ops.

Patches Qwen3-MoE's, Qwen3-Next / Qwen3.5's and Gemma 4's forwards to call
the fused ops; qwen3_5_patch's attention/MoE overlap forward calls them too.
TPUModelRunner.load_model builds the ops with arm(). Layers under
TPU_MOE_SEQUENCE_PARALLEL stay unfused.
"""

from typing import Any

import torch
from vllm.model_executor.models import gemma4, qwen3_moe, qwen3_next

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import fused_attn_proj
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_qwen3_moe_forward = qwen3_moe.Qwen3MoeDecoderLayer.forward
_qwen3_next_forward = qwen3_next.Qwen3NextDecoderLayer.forward
_gemma4_attention_forward = gemma4.Gemma4Attention.forward


def _patched_qwen3_moe_forward(self, positions, hidden_states, residual):
    fused = fused_attn_proj.attention(
        self.self_attn, hidden_states, residual, positions
    )
    if fused is None:
        return _qwen3_moe_forward(self, positions, hidden_states, residual)
    hidden_states, residual = self.post_attention_layernorm(*fused)
    return self.mlp(hidden_states), residual


def _patched_qwen3_next_forward(self, hidden_states, residual, positions, **kwargs):
    """Qwen3-Next and Qwen3.5 (also patched by qwen3_5_patch's overlap path)."""
    attn = self.linear_attn if self.layer_type == "linear_attention" else self.self_attn
    fused = fused_attn_proj.attention(attn, hidden_states, residual, positions)
    if fused is None:
        return _qwen3_next_forward(self, hidden_states, residual, positions, **kwargs)
    hidden_states, residual = self.post_attention_layernorm(*fused)
    return self.mlp(hidden_states), residual


class _NormInAttention(torch.nn.Module):
    """A fused Gemma 4 layer's input_layernorm: the attention op applies it.

    Keeps the norm's weight under its name, so its state dict is unchanged.
    """

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.weight = norm.weight
        self.unfused = (norm,)  # A tuple, so it stays out of the module tree.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _patched_gemma4_attention_forward(self, positions, hidden_states, **kwargs: Any):
    norm = getattr(self, "_norm_in_attention", None)
    if norm is None:
        return _gemma4_attention_forward(self, positions, hidden_states, **kwargs)
    fused = None
    if all(v is None for v in kwargs.values()):  # Not with multimodal masks.
        fused = fused_attn_proj.attention(self, hidden_states, None, positions)
    if fused is not None:
        return fused[0]
    unfused = norm.unfused[0](hidden_states)
    return _gemma4_attention_forward(self, positions, unfused, **kwargs)


def arm(model: torch.nn.Module, hf_config: Any, *, page_size: int) -> int:
    """Builds the fused ops of model's eligible layers; returns how many.

    TPUModelRunner.load_model calls it after the attention kernels are
    initialized. Safe to call again (e.g. after an in-place weight update): it
    undoes the previous Gemma 4 norm swaps and rebuilds every op.
    """
    gemma_layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, gemma4.Gemma4DecoderLayer)
    ]
    for layer in gemma_layers:
        if isinstance(layer.input_layernorm, _NormInAttention):
            layer.input_layernorm = layer.input_layernorm.unfused[0]
            layer.self_attn.__dict__.pop("_norm_in_attention", None)
    fused = fused_attn_proj.prepare_model(model, hf_config, page_size=page_size)
    for layer in gemma_layers:
        if fused_attn_proj.is_fused(layer.self_attn):
            layer.input_layernorm = _NormInAttention(layer.input_layernorm)
            # object.__setattr__: not a submodule of self_attn.
            object.__setattr__(
                layer.self_attn, "_norm_in_attention", layer.input_layernorm
            )
    return fused


def maybe_patch_fused_attn_proj(*args: object, **kwargs: object) -> None:
    if not envs.TPU_FUSED_ATTN_PROJ:
        return
    qwen3_moe.Qwen3MoeDecoderLayer.forward = _patched_qwen3_moe_forward
    if not envs.TPU_MOE_SEQUENCE_PARALLEL:  # Else qwen3_next_patch's forward.
        qwen3_next.Qwen3NextDecoderLayer.forward = _patched_qwen3_next_forward
    gemma4.Gemma4Attention.forward = _patched_gemma4_attention_forward
    logger.info("Applied TPU patch: fused_attn_proj (TPU_FUSED_ATTN_PROJ).")
