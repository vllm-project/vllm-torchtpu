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
"""Beam text model for TPU.

Follows upstream vllm/models/beam/nvidia/model.py: sliding-window attention
layers with RoPE and full-attention layers without it (SSMax query scaling),
per-head Q/K RMSNorm, a sigmoid attention output gate, sandwich norms with
depth-scaled residuals, dense SwiGLU or grouped sigmoid ``noaux_tc`` MoE, and
a softcapped LM head. Upstream runs these as Triton and CuTe DSL kernels; here
they are torch ops with the same FP32 arithmetic, rounding to the activation
dtype at the same points. On TPU, vLLM's ``RMSNorm`` already applies the
weight in FP32 before rounding (``layers/adapter/ir_ops.py``).
"""

import math
from collections.abc import Iterable
from typing import cast

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.beam import BeamConfig
from vllm.v1.attention.backend import AttentionType


def _router_logits_jax(hidden_states: jax.Array, weight: jax.Array) -> jax.Array:
    """``hidden_states @ weight.T`` in FP32; TPU's default rounds FP32 operands
    to BF16."""
    return jax.lax.dot_general(
        hidden_states.astype(jnp.float32),
        weight,
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


def _fake_router_logits(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0], weight.shape[0]), dtype=torch.float32
    )


# Registered at import: the router runs inside the traced forward, where Dynamo
# cannot trace an op registration.
_router_logits = pallas.jax_op("pallas::beam_router_logits", _router_logits_jax)
_router_logits.register_fake(_fake_router_logits)


def _use_merged_qkv_gate(config: BeamConfig, tp_size: int) -> bool:
    return config.num_key_value_heads >= tp_size


def _weights_mapper(merged_qkv_gate: bool) -> WeightsMapper:
    """Checkpoint names to stacked parameters; experts load in the MoE runner."""
    if merged_qkv_gate:
        attention = {
            f".self_attn.{name}": (".self_attn.qkv_gate_proj", shard)
            for shard, name in enumerate(
                ("q_proj", "k_proj", "v_proj", "output_gate_proj")
            )
        }
    else:
        attention = {
            f".self_attn.{shard}_proj": (".self_attn.qkv_proj", shard)
            for shard in ("q", "k", "v")
        }
    return WeightsMapper(
        # Renamed so the attention gate is not taken for a gate_up_proj shard.
        orig_to_new_substr={".self_attn.gate_proj": ".self_attn.output_gate_proj"},
        orig_to_new_stacked={
            **attention,
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        },
    )


def _depth_scaled_coeffs(layer_id: int, is_attention: bool) -> tuple[float, float]:
    """Residual coefficients ``(x_scale, residual_scale)`` of one sublayer.

    ``num_adds`` counts the residual adds up to and including this one, with
    the 1-based layer index ``L = layer_id + 1``: ``2L`` for the attention add,
    ``2L + 1`` for the MLP add.
    """
    one_based = layer_id + 1
    num_adds = 2 * one_based if is_attention else 2 * one_based + 1
    denominator = math.sqrt(num_adds)
    return 1.0 / denominator, math.sqrt(num_adds - 1) / denominator


def _scaled_residual_add(
    x: torch.Tensor, residual: torch.Tensor, scales: tuple[float, float]
) -> torch.Tensor:
    x_scale, residual_scale = scales
    dtype = x.dtype
    x = (x.float() * x_scale).to(dtype)
    residual = (residual.to(dtype).float() * residual_scale).to(dtype)
    return (x.float() + residual.float()).to(dtype)


def _ssmax(q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """SSMax scaling of NoPE queries ``[tokens, heads, head_dim]``."""
    # (1 + (position + 1) / 2048) ** 0.1
    scale = torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048.0))
    return (q.float() * scale[:, None, None]).to(q.dtype)


def _rotate_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    x_fp32 = x.float()
    half = rotary_dim // 2
    x1 = x_fp32[..., :half]
    x2 = x_fp32[..., half:rotary_dim]
    rotated = torch.cat(
        (x1 * cos - x2 * sin, x2 * cos + x1 * sin, x_fp32[..., rotary_dim:]),
        dim=-1,
    )
    return rotated.to(x.dtype)


class BeamAttention(nn.Module):
    """GQA with per-head Q/K RMSNorm, then RoPE (sliding-window layers) or SSMax
    query scaling (full-attention layers); the output is scaled by
    ``sigmoid(gate + gate_bias)`` before ``o_proj``.

    The normalized Q/K round to the activation dtype before the FP32 position
    transform, and RoPE reads the FP32 cos/sin cache unrounded.
    """

    def __init__(
        self,
        config: BeamConfig,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_id = extract_layer_index(prefix)
        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        assert total_num_heads % tp_size == 0
        if total_num_kv_heads >= tp_size:
            assert total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % total_num_kv_heads == 0
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or hidden_size // total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.merged_qkv_gate = _use_merged_qkv_gate(config, tp_size)
        if self.merged_qkv_gate:
            self.qkv_gate_proj = MergedColumnParallelLinear(
                hidden_size,
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
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                total_num_heads,
                total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.output_gate_proj = ColumnParallelLinear(
                hidden_size,
                total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.output_gate_proj",
            )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads), requires_grad=False)
        set_weight_attrs(self.gate_bias, {"weight_loader": sharded_weight_loader(0)})

        is_sliding = config.layer_types[layer_id] == "sliding_attention"
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
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window if is_sliding else None,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.merged_qkv_gate:
            packed, _ = self.qkv_gate_proj(hidden_states)
            q, k, v, gate = packed.split(
                [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            gate, _ = self.output_gate_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        rotary_emb = self.rotary_emb
        if rotary_emb is None:
            q = _ssmax(q, positions)
        else:
            cos_sin = rotary_emb.cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.unsqueeze(-2).chunk(2, dim=-1)
            q = _rotate_neox(q, cos, sin, rotary_emb.rotary_dim)
            k = _rotate_neox(k, cos, sin, rotary_emb.rotary_dim)

        attn_output = self.attn(q.flatten(-2), k.flatten(-2), v)
        output, _ = self.o_proj(self._apply_output_gate(attn_output, gate))
        return output

    def _apply_output_gate(
        self, attn_output: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """``attn_output * sigmoid(gate + gate_bias)``, rounding after the bias
        add and after the sigmoid."""
        heads = (self.num_heads, self.head_dim)
        dtype = gate.dtype
        gate = gate.unflatten(-1, heads).float() + self.gate_bias.float()[:, None]
        gate = torch.sigmoid(gate.to(dtype).float()).to(dtype)
        output = attn_output.unflatten(-1, heads).float() * gate.float()
        return output.to(attn_output.dtype).flatten(-2)


class BeamMLP(nn.Module):
    """SwiGLU MLP of the dense layers; SiLU-and-mul in FP32, rounding the
    product once."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.float().chunk(2, dim=-1)
        activation = gate / (1.0 + torch.exp(-gate)) * up
        output, _ = self.down_proj(activation.to(gate_up.dtype))
        return output


class BeamRouter(nn.Module):
    """FP32 router weight and ``noaux_tc`` score correction bias."""

    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _router_logits(hidden_states, self.weight)


class BeamSparseMoeBlock(nn.Module):
    """Grouped sigmoid ``noaux_tc`` MoE, routed scale applied to the output."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        self.gate = BeamRouter(config.hidden_size, config.num_experts)
        self.experts = FusedMoEFactory(
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
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=torch.float32,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.experts(
            hidden_states=hidden_states, router_logits=self.gate(hidden_states)
        )


class BeamDecoderLayer(nn.Module):
    """Sandwich-norm block: each sublayer's normalized output is added to the
    depth-scaled residual."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.self_attn = BeamAttention(
            config,
            vllm_config.cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp: BeamMLP | BeamSparseMoeBlock
        if config.mlp_layer_types[layer_id] == "dense":
            self.mlp = BeamMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = BeamSparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.attn_scales = _depth_scaled_coeffs(layer_id, is_attention=True)
        self.mlp_scales = _depth_scaled_coeffs(layer_id, is_attention=False)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        x = self.self_attn(positions, self.input_layernorm(hidden_states))
        hidden_states = _scaled_residual_add(
            self.post_attention_layernorm(x), hidden_states, self.attn_scales
        )
        x = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return _scaled_residual_add(
            self.post_feedforward_layernorm(x), hidden_states, self.mlp_scales
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class BeamModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        self.config = config
        # Names the torch.compile cache so it persists across restarts.
        self.prefix = prefix
        self.hf_to_vllm_mapper = _weights_mapper(
            _use_merged_qkv_gate(config, get_tensor_model_parallel_world_size())
        )
        # Windows come per layer from layer_types; a global window would also
        # reach the full-attention layers through Attention's fallback.
        vllm_config.cache_config.sliding_window = None

        self.embed_tokens: VocabParallelEmbedding | PPMissingLayer
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BeamDecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm: RMSNorm | PPMissingLayer = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            hidden_states = inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, None
        )
        for index in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[index](positions, hidden_states)
            self._maybe_add_hidden_state(
                aux_hidden_states, index + 1, hidden_states, None
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class BeamForCausalLM(nn.Module, SupportsPP, SupportsQuant, SupportsEagle3):
    hf_to_vllm_mapper = _weights_mapper(merged_qkv_gate=False)
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_gate_proj": ["q_proj", "k_proj", "v_proj", "output_gate_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = cast(BeamConfig, vllm_config.model_config.hf_config)
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError(
                "Beam has not been validated with EPLB; run without --enable-eplb."
            )
        if config.tie_word_embeddings:
            raise NotImplementedError("Beam requires tie_word_embeddings=False.")
        self.config = config
        self.quant_config = vllm_config.quant_config

        self.model = BeamModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        cap = self.config.final_logit_softcapping
        if logits is None or cap is None:
            return logits
        return (torch.tanh(logits.float() * (1.0 / cap)) * cap).to(logits.dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
