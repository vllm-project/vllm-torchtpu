# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

import torch
from torch import nn
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

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
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_before_norm = self.norm is not None and self.tp_size > 1
        if reduced_before_norm:
            # Routed experts shard their intermediate dimension across TP
            # ranks.  K3 normalizes the summed latent output, so reducing only
            # after this nonlinear transform (the generic FusedMoE order)
            # produces a different result.  FusedMoE still performs its final
            # reduction to combine the shared-expert partials; divide this
            # replicated routed result below to compensate for that reduction.
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        if reduced_before_norm:
            hidden_states = hidden_states / self.tp_size
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
                                    "min_moe_intermediate_per_partition", 256)
        if (self.tp_size > 1 and padded_intermediate_size // self.tp_size
                < min_per_partition):
            padded_intermediate_size = min_per_partition * self.tp_size

        situ_beta = (getattr(config, "activation_situ_beta", None)
                     if config.hidden_act == "situ" else None)
        situ_linear_beta = (getattr(config, "activation_situ_linear_beta",
                                    None)
                            if config.hidden_act == "situ" else None)
        # Current vLLM represents activations with an enum that does not yet
        # include Kimi-K3's SITU.  SITU has the same gated weight layout as
        # SiLU, so construct the generic MoE with SiLU and restore the TPU
        # kernel's richer activation descriptor below.
        moe_activation = ("silu" if config.hidden_act == "situ" else
                          config.hidden_act)
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=expert_hidden_size,
            intermediate_size=padded_intermediate_size,
            activation=moe_activation,
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
        # SITU parameters used to be accepted directly by FusedMoE. Current
        # vLLM keeps activation-specific settings on the runner's MoE config.
        self.experts.moe_config.activation_situ_beta = situ_beta
        self.experts.moe_config.activation_situ_linear_beta = situ_linear_beta
        if config.hidden_act == "situ":
            self.experts.routed_experts.activation = "situ"
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
                config.moe_intermediate_size // self.tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states,
                              router_logits=router_logits)
        return output.view(num_tokens, hidden_size)
