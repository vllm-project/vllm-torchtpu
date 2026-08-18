# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu.layers.vllm.moe_hierarchical import (
    hierarchical_moe_parallel_config, hierarchical_split_or_none)

from .layers import KimiMLP


class KimiRoutedOutputTransform(nn.Module):

    def __init__(
        self,
        norm: RMSNorm | None,
        up_proj: ReplicatedLinear,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        return hidden_states


class KimiMoE(nn.Module):
    """Standard or latent Kimi MoE backed by the fused TPU GMM custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.num_experts is not None
        assert config.num_experts_per_token is not None
        assert config.moe_intermediate_size is not None

        hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config_or_none()
        self.use_ep = (vllm_config is not None
                       and vllm_config.parallel_config.enable_expert_parallel)
        routed_expert_hidden_size = getattr(config,
                                            "routed_expert_hidden_size", None)
        expert_hidden_size = (routed_expert_hidden_size
                              if routed_expert_hidden_size is not None else
                              hidden_size)

        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts))

        self.routed_expert_down_proj = (ReplicatedLinear(
            hidden_size,
            expert_hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.routed_expert_down_proj",
        ) if routed_expert_hidden_size is not None else None)
        self.routed_expert_norm = (
            RMSNorm(expert_hidden_size, config.rms_norm_eps)
            if self.routed_expert_down_proj is not None
            and getattr(config, "latent_moe_use_norm", False) else None)
        self.routed_expert_up_proj = (ReplicatedLinear(
            expert_hidden_size,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.routed_expert_up_proj",
        ) if self.routed_expert_down_proj is not None else None)
        routed_output_transform = (KimiRoutedOutputTransform(
            self.routed_expert_norm, self.routed_expert_up_proj)
                                   if self.routed_expert_up_proj is not None
                                   else None)

        self.shared_experts = (KimiMLP(
            hidden_size,
            config.moe_intermediate_size * config.num_shared_experts,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
            situ_beta=getattr(config, "activation_situ_beta", None),
            situ_linear_beta=getattr(config, "activation_situ_linear_beta",
                                     None),
        ) if config.num_shared_experts > 0 else None)

        padded_intermediate_size = config.moe_intermediate_size
        min_per_partition = getattr(config,
                                    "min_moe_intermediate_per_partition", 128)
        if self.use_ep:
            split = hierarchical_split_or_none()
            moe_tp_size = split[2] if split is not None else 1
        else:
            moe_tp_size = self.tp_size
        if (moe_tp_size > 1 and padded_intermediate_size // moe_tp_size
                < min_per_partition):
            padded_intermediate_size = min_per_partition * moe_tp_size

        situ_beta = (getattr(config, "activation_situ_beta", None)
                     if config.hidden_act == "situ" else None)
        situ_linear_beta = (getattr(config, "activation_situ_linear_beta",
                                    None)
                            if config.hidden_act == "situ" else None)
        # Under TPU_MOE_HIERARCHICAL_EP this builds the routed experts
        # with expert parallelism between chips and tensor parallelism
        # within one chip, instead of vLLM's flat all-experts-per-device EP.
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                hidden_size=expert_hidden_size,
                intermediate_size=padded_intermediate_size,
                activation=config.hidden_act,
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
                renormalize=config.moe_renormalize,
                quant_config=quant_config,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.moe_router_activation_func,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=config.routed_scaling_factor,
                routed_input_transform=self.routed_expert_down_proj,
                routed_output_transform=routed_output_transform,
            )
        if padded_intermediate_size != config.moe_intermediate_size:
            routed_experts = self.experts.routed_experts
            w13_weight = getattr(routed_experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = routed_experts.w13_weight_packed
            w2_weight = getattr(routed_experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = routed_experts.w2_weight_packed
            w13_weight.data.zero_()
            w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                config.moe_intermediate_size // moe_tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states,
                              router_logits=router_logits)
        return output.view(num_tokens, hidden_size)
