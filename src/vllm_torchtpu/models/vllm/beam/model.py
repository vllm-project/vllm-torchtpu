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
from collections.abc import Iterable
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
from vllm.transformers_utils.configs.beam import BeamConfig
from vllm.v1.attention.backend import AttentionType

from vllm_torchtpu.models.vllm.beam import ops
from vllm_torchtpu.models.vllm.beam.router import BeamRouter


def _depth_scaled_coeffs(layer_id: int, is_attention: bool) -> tuple[float, float]:
    """Depth-scaled residual coefficients ``(x_scale, r_scale)`` for one sublayer.

    ``num_adds`` counts residual adds up to and including this one, with a
    1-based layer index ``L = layer_id + 1``: ``2L`` for the attention add,
    ``2L + 1`` for the MLP add.
    """
    one_based = layer_id + 1
    num_adds = 2 * one_based if is_attention else 2 * one_based + 1
    denom = math.sqrt(num_adds)
    return 1.0 / denom, math.sqrt(num_adds - 1) / denom


class Norm(nn.Module):
    """RMSNorm weight; the math lives in ops.beam_rms_norm and
    ops.beam_norm_residual_norm."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class BeamAttention(nn.Module):
    """GQA with per-head Q/K RMSNorm, then RoPE (sliding-window layers) or
    SSMax (NoPE full-attention layers). The attention output is scaled by
    ``sigmoid(gate + bias)`` before ``o_proj``.
    """

    def __init__(self, config: BeamConfig, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        layer_id = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        # BeamForCausalLM checks that TP divides the KV head count, so the
        # merged projection gives every rank whole KV heads.
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        self.head_dim = config.head_dim or config.hidden_size // total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        quant_config = vllm_config.quant_config

        self.qkv_gate_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [
                total_num_heads * self.head_dim,
                total_num_kv_heads * self.head_dim,
                total_num_kv_heads * self.head_dim,
                total_num_heads * self.head_dim,
            ],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_gate_proj",
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        # The output gate has a separate bias per head.
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads))
        set_weight_attrs(self.gate_bias, {"weight_loader": sharded_weight_loader(0)})

        is_sliding = config.layer_types[layer_id] == "sliding_attention"
        self.disable_rope = not is_sliding
        # Beam rotates in fp32.
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
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
            self.head_dim**-0.5,
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
        rope = self.rotary_emb
        q, k = ops.beam_qk_norm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rms_norm_eps,
            positions,
            rope.cos_sin_cache if rope is not None else None,
            self.head_dim,
            rope.rotary_dim if rope is not None else 0,
            rope.is_neox_style if rope is not None else True,
            ssmax=self.disable_rope,
        )
        out = ops.beam_output_gate(
            self.attn(q, k, v),
            gate,
            self.gate_bias,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        out, _ = self.o_proj(out)
        return out


class BeamMLP(nn.Module):
    """SwiGLU MLP for the ``dense`` entries of ``mlp_layer_types``."""

    def __init__(self, config: BeamConfig, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )
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
        # Beam rounds once after the fp32 SiLU and gate/up multiplication.
        out, _ = self.down_proj(ops.beam_silu_mul(gate_up))
        return out


class BeamSparseMoeBlock(nn.Module):
    """Grouped sigmoid ``noaux_tc`` MoE block."""

    def __init__(self, config: BeamConfig, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.gate = BeamRouter(config.hidden_size, config.num_experts)
        e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32)
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
        # The runner computes the router logits with the gate passed above.
        return self.experts(hidden_states=x, router_logits=x)


class BeamDecoderLayer(nn.Module):
    """Sandwich-norm block that also returns the next layer's normalized input.

    Post-norm, depth-scaled residual add and the next pre-norm form one
    elementwise chain, as in upstream's fused boundary kernel.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        layer_id = extract_layer_index(prefix)
        self.rms_norm_eps = config.rms_norm_eps
        self.self_attn = BeamAttention(config, vllm_config, f"{prefix}.self_attn")
        self.mlp: nn.Module = (
            BeamMLP(config, vllm_config, f"{prefix}.mlp")
            if config.mlp_layer_types[layer_id] == "dense"
            else BeamSparseMoeBlock(config, vllm_config, f"{prefix}.mlp")
        )
        self.input_layernorm = Norm(config.hidden_size)
        self.pre_feedforward_layernorm = Norm(config.hidden_size)
        self.post_attention_layernorm = Norm(config.hidden_size)
        self.post_feedforward_layernorm = Norm(config.hidden_size)
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
            self.post_attention_layernorm.weight,
            self.rms_norm_eps,
            self.pre_feedforward_layernorm.weight,
            self.rms_norm_eps,
            self.attn_scales,
        )
        hidden_states, normalized_states = ops.beam_norm_residual_norm(
            self.mlp(x),
            hidden_states,
            self.post_feedforward_layernorm.weight,
            self.rms_norm_eps,
            next_norm_weight,
            self.rms_norm_eps,
            self.mlp_scales,
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
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        # Beam drives SWA per layer off `layer_types`, passing an explicit
        # `per_layer_sliding_window` (None on the full-attention layers).
        # Without this, those layers would pick up `cache_config.sliding_window`
        # as a fallback in `Attention.__init__` and silently get a window.
        if vllm_config.cache_config is not None:
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
        self.norm = Norm(config.hidden_size)

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
        normalized_states = ops.beam_rms_norm(
            hidden_states, self.layers[0].input_layernorm.weight, self.rms_norm_eps
        )
        next_norm_weights = [layer.input_layernorm.weight for layer in self.layers[1:]]
        next_norm_weights.append(self.norm.weight)
        for layer, next_norm_weight in zip(self.layers, next_norm_weights):
            hidden_states, normalized_states = layer(
                positions, hidden_states, normalized_states, next_norm_weight
            )
        # The last layer applied the final norm.
        return normalized_states


class BeamForCausalLM(nn.Module, SupportsQuant):
    """Beam causal LM on TPU.

    Not MixtureOfExperts (EPLB is rejected), SupportsPP or SupportsEagle3.
    """

    # The checkpoint shards of the merged layers, as named after the mapper,
    # for quantization scheme and ignore matching.
    packed_modules_mapping = {
        "qkv_gate_proj": ["q_proj", "k_proj", "v_proj", "output_gate_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # Upstream's mapping for the merged QKV + output-gate projection.
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
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            # Expert parallelism is validated; expert rebalancing is not.
            raise NotImplementedError(
                "Beam has not been validated with EPLB; run without --enable-eplb."
            )
        if parallel_config.use_sequence_parallel_moe:
            # Data-parallel attention with expert-parallel MoE routes tokens
            # through the sequence-parallel path, which is not yet validated.
            raise NotImplementedError(
                "Beam has not been validated with sequence-parallel MoE "
                "(data parallelism with expert parallelism)."
            )
        if config.tie_word_embeddings:
            raise NotImplementedError("Beam requires tie_word_embeddings=False.")
        # The merged QKV + output-gate projection shards whole KV heads.
        tp_size = parallel_config.tensor_parallel_size
        num_heads, num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        if num_kv_heads % tp_size or num_heads % tp_size:
            valid = [
                n
                for n in range(1, num_kv_heads + 1)
                if not num_kv_heads % n and not num_heads % n
            ]
            raise NotImplementedError(
                f"Beam on TPU: tensor_parallel_size must divide "
                f"num_key_value_heads ({num_kv_heads}): use one of {valid}, got "
                f"{tp_size}. Scale further with --data-parallel-size and "
                "--enable-expert-parallel."
            )

        self.model = BeamModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Untied LM head, in the model dtype. An fp32 head is available the
        # standard way, via `head_dtype` on the config (see
        # `LogitsProcessor._apply_head`), rather than by pinning the parameter
        # dtype here.
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        # Apply the softcap in fp32 while keeping the head in the model dtype.
        self.final_logit_softcapping = config.final_logit_softcapping

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise NotImplementedError("Beam on TPU does not support PP")
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
