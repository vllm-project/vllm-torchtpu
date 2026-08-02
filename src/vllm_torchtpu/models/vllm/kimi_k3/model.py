# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only Kimi K3 and Kimi-Linear model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import \
    fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc, MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              maybe_prefix)
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from .attention import KimiDeltaAttention, MultiHeadLatentAttention
from .layers import AttentionResidual, KimiMLP
from .moe import KimiMoE


class KimiDecoderLayer(nn.Module):

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        if config.is_kda_layer(self.layer_idx):
            self.self_attn = KimiDeltaAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = MultiHeadLatentAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )

        is_moe = (config.num_experts is not None
                  and self.layer_idx >= config.first_k_dense_replace
                  and self.layer_idx % config.moe_layer_freq == 0)
        self.is_moe = is_moe
        if is_moe:
            self.block_sparse_moe = KimiMoE(
                config,
                vllm_config.quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = KimiMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
                situ_beta=getattr(config, "activation_situ_beta", None),
                situ_linear_beta=getattr(config, "activation_situ_linear_beta",
                                         None),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

        self.attn_res_block_size = getattr(config, "attn_res_block_size", None)
        if self.attn_res_block_size is not None:
            self.self_attention_res = AttentionResidual(
                config.hidden_size,
                config.rms_norm_eps,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res = AttentionResidual(
                config.hidden_size,
                config.rms_norm_eps,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residuals: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.attn_res_block_size is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions, hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            if self.is_moe:
                hidden_states = self.block_sparse_moe(hidden_states)
            else:
                hidden_states = self.mlp(hidden_states)
            return residual + hidden_states, None

        assert block_residuals is not None
        prefix_sum = hidden_states
        if block_residuals.shape[-2] > 0:
            hidden_states = self.self_attention_res(prefix_sum,
                                                    block_residuals)
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residuals = torch.cat(
                (block_residuals, prefix_sum.unsqueeze(-2)), dim=-2)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        hidden_states = self.mlp_res(prefix_sum, block_residuals)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return prefix_sum + hidden_states, block_residuals


@support_torch_compile(dynamic_arg_dims={
    "input_ids": 0,
    "positions": 0,
    "inputs_embeds": 0,
})
class KimiModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: KimiLinearConfig = vllm_config.model_config.hf_text_config
        self.attn_res_block_size = getattr(config, "attn_res_block_size", None)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList([
            KimiDecoderLayer(
                config,
                vllm_config,
                prefix=f"{prefix}.layers.{layer_idx}",
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.attn_res_block_size is not None:
            self.output_attn_res = AttentionResidual(
                config.hidden_size,
                config.rms_norm_eps,
                prefix=f"{prefix}.output_attn_res_proj",
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (inputs_embeds if inputs_embeds is not None else
                         self.embed_input_ids(input_ids))
        block_residuals = (hidden_states.new_empty(
            hidden_states.shape[0],
            0,
            hidden_states.shape[1],
        ) if self.attn_res_block_size is not None else None)
        assert hidden_states is not None

        for layer in self.layers:
            hidden_states, block_residuals = layer(
                positions,
                hidden_states,
                block_residuals,
            )

        if block_residuals is not None:
            hidden_states = self.output_attn_res(hidden_states,
                                                 block_residuals)
        return self.norm(hidden_states)


class KimiLinearForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """Kimi text model for TPU."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "model.layers.",
            "language_model.": "",
        },
        orig_to_new_substr={
            ".self_attention_res_norm.": ".self_attention_res.norm.",
            ".self_attention_res_proj.": ".self_attention_res.proj.",
            ".mlp_res_norm.": ".mlp_res.norm.",
            ".mlp_res_proj.": ".mlp_res.proj.",
            ".output_attn_res_norm.": ".output_attn_res.norm.",
            ".output_attn_res_proj.": ".output_attn_res.proj.",
        },
        orig_to_new_stacked={
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        },
    )
    fused_mla_mapper = WeightsMapper(orig_to_new_stacked={
        ".q_a_proj.": (".fused_qkv_a_proj.", 0),
        ".kv_a_proj_with_mqa.": (".fused_qkv_a_proj.", 1),
    }, )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: KimiLinearConfig = vllm_config.model_config.hf_text_config
        self.model = KimiModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
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
            raise ValueError("The TPU Kimi model does not support PP")
        return self.model(
            input_ids,
            positions,
            inputs_embeds,
        )

    def compute_logits(self,
                       hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config: KimiLinearConfig = vllm_config.model_config.hf_text_config
        kda_config = config.linear_attn_config
        assert kda_config is not None
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            kda_config["num_heads"],
            kda_config["head_dim"],
            conv_kernel_size=kda_config["short_conv_kernel_size"],
            num_spec=0,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls, ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        expert_params_mapping = (fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts,
        ) if self.config.num_experts is not None else [])
        params_dict = dict(self.named_parameters())
        experts_use_weight = not any(
            name.endswith("w13_weight_packed") for name in params_dict)
        ordinary_weights: list[tuple[str, torch.Tensor]] = []
        loaded_experts: set[str] = set()
        mapper = self.hf_to_vllm_mapper
        if self.config.q_lora_rank is not None:
            mapper |= self.fused_mla_mapper

        for name, loaded_weight in mapper.apply(weights):
            if name.startswith(("vision_tower.", "mm_projector.")):
                continue
            if experts_use_weight and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")

            for (param_name, weight_name, expert_id,
                 expert_shard_id) in expert_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                parameter = params_dict[name]
                parameter.weight_loader(
                    parameter,
                    loaded_weight,
                    name,
                    expert_id=expert_id,
                    shard_id=expert_shard_id,
                )
                loaded_experts.add(name)
                break
            else:
                ordinary_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loaded_experts | loader.load_weights(ordinary_weights)
