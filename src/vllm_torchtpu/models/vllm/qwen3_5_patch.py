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
"""Qwen3.5 MoE: overlap the attention TP all-reduce with MoE routing.

Upstream all-reduces the attention output, then every rank runs the
post-attention norm, the router gate, and the experts on all tokens. Here a
layer reduce-scatters ``attn_partial + residual / tp``, runs the norm and the
router gate on its token shard, all-gathers the normed tokens and the logits,
and adds its attention partial sum to its MoE partial sum so the MoE
all-reduce also completes the attention reduction. The layer returns the
pre-attention residual, so the next fused add+norm sees ``moe + attn +
residual``, the upstream stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.fp8_all_gather import fp8_all_gather
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_init = Qwen3_5DecoderLayer.__init__

_MIN_PER_CHIP_SIZE_MB = 8
# gmm_v2 quantizes its lhs with one scale per 512 channels. Matching that
# granularity keeps an outlier from clipping a whole 4096-wide row.
_FP8_SUBCHANNEL_SIZE = 512


def _overlap_min_tokens(hidden_size: int, tp_size: int, element_size: int) -> int:
    return (_MIN_PER_CHIP_SIZE_MB * 1024 * 1024 * tp_size) // (
        hidden_size * element_size
    )


def _configure_layer(layer: Qwen3_5DecoderLayer) -> None:
    mlp: Any = layer.mlp
    if mlp.tp_size <= 1:
        raise ValueError("TPU_ATTN_MOE_OVERLAP requires TP > 1")
    if layer.use_attn_reduce_scatter_for_moe:
        raise ValueError("TPU_ATTN_MOE_OVERLAP does not support sequence-parallel MoE")
    if mlp.replicate_shared_expert:
        # A replicated shared expert is added on every rank before the MoE
        # all-reduce; folding the attention rows into that all-reduce would
        # sum it tp times.
        raise ValueError(
            "TPU_ATTN_MOE_OVERLAP does not support replicate_shared_expert"
        )
    proj = (
        layer.linear_attn.out_proj
        if layer.layer_type == "linear_attention"
        else layer.self_attn.o_proj
    )
    proj.reduce_results = False
    # MoERunner recomputes router_logits from its own gate when one is set.
    experts: Any = mlp.experts
    experts.gate = None
    experts.moe_config.skip_final_all_reduce = True


def _patched_init(
    self: Qwen3_5DecoderLayer,
    vllm_config: Any,
    layer_type: str,
    prefix: str = "",
) -> None:
    _orig_init(self, vllm_config, layer_type, prefix=prefix)
    _configure_layer(self)


def _overlap_forward(
    self: Qwen3_5DecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_tokens = positions.shape[-1]
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if self.layer_type == "linear_attention":
        attn = self.linear_attn(hidden_states=hidden_states)
    else:
        attn = self.self_attn(hidden_states=hidden_states, positions=positions)
    # out_proj / o_proj run with reduce_results=False: per-rank partial sum.
    if self.layer_scale:
        attn = attn * (self.attn_layer_scale.to(attn.dtype)[0] + 1)

    mlp: Any = self.mlp
    tp_size: int = mlp.tp_size
    min_tokens = _overlap_min_tokens(attn.shape[-1], tp_size, attn.element_size())
    # Only the backbone warmup gives each side of this branch its own trace.
    # The MTP drafter compiles outside warmup, so it keeps the all-reduce path.
    overlap = (
        not torch.compiler.is_compiling() or shape_variants.aot_cache_tag() is not None
    ) and num_tokens >= min_tokens
    if overlap:
        x_loc = tensor_model_parallel_reduce_scatter(attn + residual / tp_size, 0)
        h_loc = self.post_attention_layernorm(x_loc)
        logits_loc, _ = mlp.gate(h_loc)
        logits = tensor_model_parallel_all_gather(logits_loc, 0).narrow(
            0, 0, num_tokens
        )
        if envs.TPU_ATTN_MOE_OVERLAP_FP8_ALL_GATHER:
            h = fp8_all_gather(
                h_loc,
                lambda t: tensor_model_parallel_all_gather(t, 0),
                block_size=_FP8_SUBCHANNEL_SIZE,
            ).narrow(0, 0, num_tokens)
        else:
            h = tensor_model_parallel_all_gather(h_loc, 0).narrow(0, 0, num_tokens)
    else:
        h, residual = self.post_attention_layernorm(
            tensor_model_parallel_all_reduce(attn), residual
        )
        logits, _ = mlp.gate(h)
    out = mlp.experts(hidden_states=h, router_logits=logits)
    if self.layer_scale:
        out = out * (self.ffn_layer_scale.to(out.dtype)[0] + 1)
    if overlap:
        out = out + attn
    return tensor_model_parallel_all_reduce(out), residual


def maybe_patch_qwen3_5_attn_moe_overlap(
    model_config: ModelConfig | None,
) -> bool | None:
    """Returns False until a Qwen3.5 MoE model config is available."""
    if not envs.TPU_ATTN_MOE_OVERLAP:
        return None
    if (
        model_config is None
        or model_config.hf_text_config.model_type != "qwen3_5_moe_text"
    ):
        return False
    if envs.USE_MOE_FUSED_EP_KERNEL:
        raise ValueError("TPU_ATTN_MOE_OVERLAP requires USE_MOE_FUSED_EP_KERNEL=False")
    if envs.TPU_MOE_SEQUENCE_PARALLEL:
        raise ValueError(
            "TPU_ATTN_MOE_OVERLAP requires TPU_MOE_SEQUENCE_PARALLEL=False"
        )
    Qwen3_5DecoderLayer.__init__ = _patched_init
    Qwen3_5DecoderLayer.forward = _overlap_forward
    logger.info("Applied TPU patch: Qwen3.5 MoE attention-to-MoE collective overlap.")
    return None
