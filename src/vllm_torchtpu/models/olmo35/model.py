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
"""OLMo 3.5 (Ai2's hybrid KDA / attention / latent-MoE model) for TPU.

Follows MaxText's `decoder_block: olmoe3`, the training reference for the
checkpoints (branch `agagik-olmo35`, pinned):
https://github.com/AI-Hypercomputer/maxtext/blob/37d1e72aa3c8a31a1ce1570b539cea83fafe0104/src/maxtext/models/olmoe3.py

Per layer:

    h   = x + attn_out_norm(mixer(attn_in_norm(x)))
    out = h + ffn_out_norm(shared_ffn(y) + latent_up(moe(latent_down(y))))
          where y = ffn_in_norm(h)

The mixer is full attention on every `full_attention_interval`-th layer and
Kimi Delta Attention elsewhere; the first `first_k_dense_replace` layers have a
wide dense SwiGLU and no routed experts. The token embedding is scaled by
sqrt(hidden_size) and RMS-normalized before the first layer.

The KDA, latent-MoE and MLP layers are the shared ones in `layers/adapter/`
(also used by Kimi K3): KDA runs the fused conv1d + GDN v3 Pallas kernel with
OLMo's value heads twice as wide as its key heads and beta in [0, 2), and the
routed experts go through vLLM's TPU fused-MoE backend with the latent
projections as its routed input / output transforms.

Checkpoint names (HF safetensors), per layer `model.layers.{i}.`:

    attn_in_norm / attn_out_norm / ffn_in_norm / ffn_out_norm .weight
    KDA layers, mixer.:
        q_proj / k_proj / v_proj / b_proj .weight       -> fused_qkvb_proj
        f_a_proj / g_a_proj .weight                     -> fused_fa_ga_proj
        f_b_proj.weight, g_b_proj.weight / .bias
        q_conv1d / k_conv1d / v_conv1d .weight          [channels, 1, kernel]
        A_log, dt_bias, o_norm.weight, o_proj.weight
    attention layers, mixer.:
        q_proj / k_proj / v_proj .weight                -> qkv_proj
        g_proj.weight, o_proj.weight, q_norm, k_norm    [heads, head_dim]
        ssmax_scale                                     [heads], when used
    dense layers (i < first_k_dense_replace), mlp.:
        gate_proj / up_proj / down_proj .weight
    MoE layers, moe.:
        gate.weight
        latent_down.weight                              -> routed_expert_down_proj
        latent_up.weight                                -> routed_expert_up_proj
        shared_experts.{gate_proj,up_proj,down_proj}.weight
        experts.{e}.{gate_proj,up_proj,down_proj}.weight

plus `model.embed_tokens.weight`, `model.embedding_norm.weight`,
`model.norm.weight` and `lm_head.weight`. Ai2 has not published the release
format; the loader will follow it once it is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_torchtpu.layers.adapter.gated_mlp import GatedMLP
from vllm_torchtpu.layers.adapter.kda_attention import (
    KdaAttention,
    kda_state_dtype,
    kda_state_shape,
)
from vllm_torchtpu.layers.adapter.latent_moe import LatentMoE

from .config import Olmo35Config


def _config(vllm_config: VllmConfig) -> Olmo35Config:
    config = vllm_config.model_config.hf_text_config
    if not isinstance(config, Olmo35Config):
        raise TypeError(
            f"OLMo 3.5 needs the TPU Olmo35Config; got {type(config).__qualname__}. "
            "Another class may own model_type 'olmo35' (see the warning from "
            "register_olmo35_config)."
        )
    return config


def _local_kda_heads(config: Olmo35Config, tp_size: int) -> int:
    if config.linear_num_heads % tp_size:
        raise ValueError(
            f"TP ({tp_size}) must divide the OLMo 3.5 KDA heads "
            f"({config.linear_num_heads})"
        )
    return config.linear_num_heads // tp_size


def olmo35_kda(vllm_config: VllmConfig, prefix: str) -> KdaAttention:
    """OLMo 3.5's KDA mixer: the shared layer with value heads twice the key
    head size, low-rank gates of the value head size, a biased output gate
    and beta in [0, 2).

    TPUOlmo35Config rejects the unified pool and speculative decoding when
    vLLM builds the config, before any layer is constructed.
    """
    config = _config(vllm_config)
    return KdaAttention(
        vllm_config,
        prefix,
        hidden_size=config.hidden_size,
        num_heads=config.linear_num_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
        conv_kernel_size=config.linear_conv_kernel_dim,
        gate_rank=config.linear_value_head_dim,
        norm_eps=config.linear_norm_eps,
        gate_bias=True,
        beta_scale=config.linear_beta_scale,
    )


def _head_sharded_loader(total_heads: int):
    """Load a per-head [heads, ...] tensor onto this rank's heads.

    Replicates heads round-robin the way QKVParallelLinear does for KV heads
    when TP exceeds the head count.
    """

    def loader(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        local = parameter.shape[0]
        if total_heads >= tp_size:
            start = rank * local
        else:
            start = rank // (tp_size // total_heads)
        loaded_weight = loaded_weight.reshape(total_heads, *parameter.shape[1:])
        default_weight_loader(parameter, loaded_weight.narrow(0, start, local))

    return loader


class Olmo35Attention(nn.Module):
    """GQA full attention: NoPE, per-head QK RMSNorm, scalable softmax and a
    sigmoid output gate.

    MaxText scales the query by head_dim^-0.5 * log(position + 1) * gain[head]
    after the QK norm (scalable softmax) and runs the softmax unscaled, so the
    whole factor is folded into the query here and the attention scale is 1.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _config(vllm_config)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5
        self.use_scalable_softmax = config.use_scalable_softmax
        quant_config = vllm_config.quant_config

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.g_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Independent RMSNorm gains per head (MaxText qk_norm_per_head).
        self.q_norm = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.k_norm = nn.Parameter(torch.ones(self.num_kv_heads, self.head_dim))
        set_weight_attrs(
            self.q_norm, {"weight_loader": _head_sharded_loader(self.total_num_heads)}
        )
        set_weight_attrs(
            self.k_norm,
            {"weight_loader": _head_sharded_loader(self.total_num_kv_heads)},
        )
        # Per-head scalable-softmax gain; only checkpoints that use scalable
        # softmax carry it.
        if self.use_scalable_softmax:
            self.ssmax_scale = nn.Parameter(
                torch.ones(self.num_heads, dtype=torch.float32)
            )
            set_weight_attrs(
                self.ssmax_scale,
                {"weight_loader": _head_sharded_loader(self.total_num_heads)},
            )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _head_rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        x_f32 = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_f32 * weight.float()

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate, _ = self.g_proj(hidden_states)

        q = self._head_rms_norm(
            q.reshape(num_tokens, self.num_heads, self.head_dim), self.q_norm
        )
        k = self._head_rms_norm(
            k.reshape(num_tokens, self.num_kv_heads, self.head_dim), self.k_norm
        ).to(dtype)
        q_scale = torch.full_like(positions, self.scaling, dtype=torch.float32)
        if self.use_scalable_softmax:
            q_scale = q_scale * torch.log(positions.float() + 1.0)
            q = q * (q_scale[:, None] * self.ssmax_scale[None, :])[..., None]
        else:
            q = q * q_scale[:, None, None]
        q = q.to(dtype)

        output = self.attn(
            q.reshape(num_tokens, self.q_size),
            k.reshape(num_tokens, self.kv_size),
            v.contiguous(),
        )
        output = output * torch.sigmoid(gate)
        output, _ = self.o_proj(output)
        return output


def olmo35_moe(
    config: Olmo35Config, quant_config: QuantizationConfig | None, prefix: str
) -> LatentMoE:
    """Shared SwiGLU expert plus routed experts in a latent space.

    The router scores the full-width input; the experts see
    `latent_down(x)` and their weighted sum goes through `latent_up`.
    Routing is softmax over all experts, top-k, renormalized and scaled by
    `routed_scaling_factor` (top_k); the scale is folded into `latent_up`'s
    weight at load time, so the fused-MoE router runs unscaled.
    """
    return LatentMoE(
        hidden_size=config.hidden_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        intermediate_size=config.moe_intermediate_size,
        hidden_act=config.hidden_act,
        quant_config=quant_config,
        prefix=prefix,
        shared_experts=GatedMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
        ),
        scoring_func="softmax",
        renormalize=True,
        routed_scaling_factor=1.0,
        use_score_correction_bias=False,
        latent_size=config.moe_latent_size,
    )


class Olmo35DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _config(vllm_config)
        self.layer_idx = extract_layer_index(prefix)
        self.is_full_attention = config.layer_types[self.layer_idx] == "full_attention"
        if self.is_full_attention:
            self.mixer: nn.Module = Olmo35Attention(vllm_config, f"{prefix}.mixer")
        else:
            self.mixer = olmo35_kda(vllm_config, f"{prefix}.mixer")

        self.is_moe = self.layer_idx >= config.first_k_dense_replace
        if self.is_moe:
            self.moe = olmo35_moe(
                config, vllm_config.quant_config, prefix=f"{prefix}.moe"
            )
        else:
            self.mlp = GatedMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        eps = config.rms_norm_eps
        self.attn_in_norm = RMSNorm(config.hidden_size, eps)
        self.attn_out_norm = RMSNorm(config.hidden_size, eps)
        self.ffn_in_norm = RMSNorm(config.hidden_size, eps)
        self.ffn_out_norm = RMSNorm(config.hidden_size, eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        mixer_out = self.mixer(positions, self.attn_in_norm(hidden_states))
        hidden_states = hidden_states + self.attn_out_norm(mixer_out)
        ffn_in = self.ffn_in_norm(hidden_states)
        ffn_out = self.moe(ffn_in) if self.is_moe else self.mlp(ffn_in)
        return hidden_states + self.ffn_out_norm(ffn_out)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class Olmo35Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embedding_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                Olmo35DecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)
        hidden_states = self.embedding_norm(hidden_states * self.embed_scale)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Olmo35ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """OLMo 3.5 text model for TPU."""

    # (checkpoint name, parameter name, shard id). A KDA layer and an
    # attention layer both have mixer.q_proj / k_proj / v_proj, so each entry
    # applies only where its target parameter exists.
    stacked_params_mapping = (
        (".mixer.q_proj.", ".mixer.fused_qkvb_proj.", 0),
        (".mixer.k_proj.", ".mixer.fused_qkvb_proj.", 1),
        (".mixer.v_proj.", ".mixer.fused_qkvb_proj.", 2),
        (".mixer.b_proj.", ".mixer.fused_qkvb_proj.", 3),
        (".mixer.f_a_proj.", ".mixer.fused_fa_ga_proj.", 0),
        (".mixer.g_a_proj.", ".mixer.fused_fa_ga_proj.", 1),
        (".mixer.q_proj.", ".mixer.qkv_proj.", "q"),
        (".mixer.k_proj.", ".mixer.qkv_proj.", "k"),
        (".mixer.v_proj.", ".mixer.qkv_proj.", "v"),
        (".gate_proj.", ".gate_up_proj.", 0),
        (".up_proj.", ".gate_up_proj.", 1),
    )

    # (checkpoint name, parameter name) of the latent-MoE projections.
    latent_params_mapping = (
        (".moe.latent_down.", ".moe.routed_expert_down_proj."),
        (".moe.latent_up.", ".moe.routed_expert_up_proj."),
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = _config(vllm_config)
        self.model = Olmo35Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: object | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("OLMo 3.5 on TPU does not support PP")
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        for module in self.modules():
            if isinstance(module, LatentMoE):
                module.initialize_dummy_router()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return kda_state_dtype(vllm_config)

    @classmethod
    # The TPU KDA conv state is 3-D ([kernel - 1, 1, conv_dim], as for
    # KimiLinearForCausalLM); IsHybrid annotates a 2-D conv shape.
    # pyrefly: ignore[bad-override]
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        config = _config(vllm_config)
        return kda_state_shape(
            num_heads=_local_kda_heads(
                config, vllm_config.parallel_config.tensor_parallel_size
            ),
            head_k_dim=config.linear_key_head_dim,
            head_v_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        routed_scale = self.config.routed_scaling_factor

        def _ordinary_weights():
            for name, loaded_weight in weights:
                if ".moe.latent_up." in name and routed_scale != 1.0:
                    # See olmo35_moe: the top-k rescale is linear, so it is
                    # folded into the latent up-projection.
                    loaded_weight = loaded_weight * routed_scale
                for checkpoint_name, param_name in self.latent_params_mapping:
                    name = name.replace(checkpoint_name, param_name)
                if ".moe.experts." in name:
                    for (
                        param_name,
                        weight_name,
                        expert_id,
                        expert_shard_id,
                    ) in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        # FusedMoE attaches weight_loader to its parameters.
                        parameter = cast(Any, params_dict[name])
                        parameter.weight_loader(
                            parameter,
                            loaded_weight,
                            name,
                            expert_id=expert_id,
                            shard_id=expert_shard_id,
                        )
                        loaded.add(name)
                        break
                    else:
                        # Not a gate / up / down expert weight: hand it to
                        # AutoWeightsLoader, which rejects unknown names. The
                        # mapping covers every expert id; FusedMoE's loader
                        # itself skips experts that are not local under EP.
                        yield name, loaded_weight
                    continue
                for weight_name, param_name, shard_id in self.stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    candidate = name.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    # Merged linear layers attach weight_loader to their weights.
                    parameter = cast(Any, params_dict[candidate])
                    parameter.weight_loader(parameter, loaded_weight, shard_id)
                    loaded.add(candidate)
                    break
                else:
                    yield name, loaded_weight

        loader = AutoWeightsLoader(self)
        return loaded | loader.load_weights(_ordinary_weights())
