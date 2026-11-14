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
"""Reflection Beam text model for TPU.

Follows the layer structure and parameter names of upstream
vllm/models/beam/nvidia/model.py, which runs only on CUDA.
Its Triton kernels are replaced by the PyTorch ops in ops.py and its fp32
router by router.BeamRouter; attention, linear layers, MoE and the LM head
are the stock vLLM layers. Left out: the fused K/V cache insert, the
low-latency packed projection, the fused all-reduce boundary, pipeline
parallelism, EAGLE-3 aux hidden states, and the separate QKV and gate
projections that upstream uses when TP exceeds the KV head count.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.beam import BeamConfig
from vllm.v1.attention.backend import AttentionType

from vllm_torchtpu.models.vllm.beam import ops
from vllm_torchtpu.models.vllm.beam.router import BeamRouter


def _beam_config(vllm_config: VllmConfig) -> BeamConfig:
    return cast(BeamConfig, vllm_config.model_config.hf_config)


def _depth_scaled_coeffs(layer_id: int, is_attention: bool) -> ops.ResidualScales:
    """Depth-scaled residual coefficients for one sublayer, as upstream's.

    ``num_adds`` counts residual adds up to and including this one, with a
    1-based layer index ``L = layer_id + 1``: ``2L`` for the attention add,
    ``2L + 1`` for the MLP add.
    """
    one_based = layer_id + 1
    num_adds = 2 * one_based if is_attention else 2 * one_based + 1
    denom = math.sqrt(num_adds)
    return ops.ResidualScales(
        branch=1.0 / denom, residual=math.sqrt(num_adds - 1) / denom
    )


class RMSNormWeight(nn.Module):
    """Holds an RMSNorm weight, as upstream's ``RMSNorm``; ``ops`` applies it."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class BeamAttention(nn.Module):
    """GQA with per-head Q/K RMSNorm, RoPE or SSMax, and a sigmoid output gate.

    Sliding-window layers use RoPE; the NoPE full-attention layers use SSMax.
    The attention output is scaled by ``sigmoid(gate + bias)`` before ``o_proj``.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _beam_config(vllm_config)
        layer_id = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        # TPUBeamConfig checks that TP divides the KV head count, so the
        # merged projection gives every rank whole KV heads.
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        self.head_dim = config.head_dim or config.hidden_size // total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        total_q_size = total_num_heads * self.head_dim
        total_kv_size = total_num_kv_heads * self.head_dim
        quant_config = vllm_config.quant_config

        self.qkv_gate_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [total_q_size, total_kv_size, total_kv_size, total_q_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_gate_proj",
        )
        self.o_proj = RowParallelLinear(
            total_q_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.q_norm = RMSNormWeight(self.head_dim)
        self.k_norm = RMSNormWeight(self.head_dim)
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads))
        set_weight_attrs(self.gate_bias, {"weight_loader": sharded_weight_loader(0)})

        is_sliding = config.layer_types[layer_id] == "sliding_attention"
        # Beam rotates in fp32. The cache covers max_model_len positions, not
        # the 1M max_position_embeddings (512 MB in fp32).
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=min(
                    config.max_position_embeddings,
                    vllm_config.model_config.max_model_len,
                ),
                is_neox_style=True,
                rope_parameters=config.rope_parameters,
                dtype=torch.float32,
            )
            if is_sliding
            else None
        )
        # Backends interpret `sliding_window` as `(q - k) <= sliding_window - 1`.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window if is_sliding else None,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        packed, _ = self.qkv_gate_proj(x)
        q, k, v, gate = packed.split(
            [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
        )
        q, k = ops.beam_qk_norm_rope(
            q,
            k,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            eps=self.rms_norm_eps,
            positions=positions,
            rotary_emb=self.rotary_emb,
        )
        out = ops.beam_output_gate(self.attn(q, k, v), gate, self.gate_bias)
        out, _ = self.o_proj(out)
        return out


class BeamMLP(nn.Module):
    """SwiGLU MLP for the ``dense`` entries of ``mlp_layer_types``."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _beam_config(vllm_config)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(ops.beam_silu_mul(gate_up))
        return out


class BeamSparseMoeBlock(nn.Module):
    """Grouped sigmoid ``noaux_tc`` MoE block."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _beam_config(vllm_config)
        self.gate = BeamRouter(config.hidden_size, config.num_experts)
        e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.gate.register_parameter("e_score_correction_bias", e_score_correction_bias)
        self.experts = FusedMoEFactory(
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=vllm_config.quant_config,
            use_grouped_topk=(
                config.n_group is not None and config.topk_group is not None
            ),
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=self.gate.out_dtype,
            prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Given a gate, the runner computes the router logits from x itself.
        return self.experts(hidden_states=x, router_logits=x)


class BeamDecoderLayer(nn.Module):
    """Sandwich-norm block that also returns the next layer's normalized input.

    Post-norm, depth-scaled residual add and the next pre-norm form one
    elementwise chain, as in upstream's fused boundary kernel.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _beam_config(vllm_config)
        layer_id = extract_layer_index(prefix)
        self.rms_norm_eps = config.rms_norm_eps
        self.self_attn = BeamAttention(vllm_config, f"{prefix}.self_attn")
        self.mlp: nn.Module = (
            BeamMLP(vllm_config, f"{prefix}.mlp")
            if config.mlp_layer_types[layer_id] == "dense"
            else BeamSparseMoeBlock(vllm_config, f"{prefix}.mlp")
        )
        self.input_layernorm = RMSNormWeight(config.hidden_size)
        self.pre_feedforward_layernorm = RMSNormWeight(config.hidden_size)
        self.post_attention_layernorm = RMSNormWeight(config.hidden_size)
        self.post_feedforward_layernorm = RMSNormWeight(config.hidden_size)
        self.attn_scales = _depth_scaled_coeffs(layer_id, is_attention=True)
        self.mlp_scales = _depth_scaled_coeffs(layer_id, is_attention=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        normalized_states: torch.Tensor,
        next_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.self_attn(positions, normalized_states)
        hidden_states, x = ops.beam_norm_residual_norm(
            x,
            hidden_states,
            post_weight=self.post_attention_layernorm.weight,
            next_weight=self.pre_feedforward_layernorm.weight,
            eps=self.rms_norm_eps,
            scales=self.attn_scales,
        )
        hidden_states, normalized_states = ops.beam_norm_residual_norm(
            self.mlp(x),
            hidden_states,
            post_weight=self.post_feedforward_layernorm.weight,
            next_weight=next_norm_weight,
            eps=self.rms_norm_eps,
            scales=self.mlp_scales,
        )
        return hidden_states, normalized_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class BeamModel(nn.Module):
    """Beam decoder stack; returns the final-normed hidden states."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _beam_config(vllm_config)
        # Names the compile cache directory (see
        # _patch_vllm_compile_prefix_isolation); without it the plugin falls
        # back to an order-dependent counter and warns.
        self.prefix = prefix
        # As upstream's BeamModel: Beam drives SWA per layer off `layer_types`,
        # passing an explicit `per_layer_sliding_window` (None on the
        # full-attention layers). Without this, those layers would pick up
        # `cache_config.sliding_window` as a fallback in `Attention.__init__`
        # and silently get a window.
        vllm_config.cache_config.sliding_window = None
        self.rms_norm_eps = config.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            BeamDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNormWeight(config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        layers = cast(Sequence[BeamDecoderLayer], self.layers)
        normalized_states = ops.beam_rms_norm(
            hidden_states, layers[0].input_layernorm.weight, self.rms_norm_eps
        )
        next_norm_weights = [layer.input_layernorm.weight for layer in layers[1:]]
        next_norm_weights.append(self.norm.weight)
        for layer, next_norm_weight in zip(layers, next_norm_weights):
            hidden_states, normalized_states = layer(
                positions, hidden_states, normalized_states, next_norm_weight
            )
        # The last layer applied the final norm.
        return normalized_states


class BeamForCausalLM(nn.Module, SupportsQuant):
    """Beam causal LM on TPU.

    Not MixtureOfExperts (EPLB is rejected), SupportsPP or SupportsEagle3.
    """

    # Checkpoint shards of the merged layers (after the mapper), for matching
    # quantization schemes and ignore lists.
    packed_modules_mapping = {
        "qkv_gate_proj": ["q_proj", "k_proj", "v_proj", "output_gate_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # Upstream's BeamModel.get_weights_mapper for the merged QKV + gate path.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            # Renamed so it cannot be matched as a `gate_up_proj` shard by
            # `packed_modules_mapping` consumers such as LoRA.
            ".self_attn.gate_proj": ".self_attn.output_gate_proj",
        },
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".self_attn.q_proj": (".self_attn.qkv_gate_proj", 0),
            ".self_attn.k_proj": (".self_attn.qkv_gate_proj", 1),
            ".self_attn.v_proj": (".self_attn.qkv_gate_proj", 2),
            ".self_attn.output_gate_proj": (".self_attn.qkv_gate_proj", 3),
            # `.mlp.experts.N.gate_proj` does not contain `.mlp.gate_proj`, so
            # expert weights fall through to `MoERunner.load_weights` (which is
            # also what keeps them correct under EP and redundant experts).
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        },
    )
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _beam_config(vllm_config)
        self.model = BeamModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # In the model dtype; an fp32 head is set through `head_dtype` instead.
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.final_logit_softcapping = config.final_logit_softcapping

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert intermediate_tensors is None, "TPUBeamConfig rejects PP"
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.final_logit_softcapping is not None:
            logits = ops.beam_softcap_logits(logits, self.final_logit_softcapping)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
