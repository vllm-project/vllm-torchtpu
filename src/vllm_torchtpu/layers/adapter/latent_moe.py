# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Standard or latent MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

import zlib
from typing import cast

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import make_latent_projection
from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)


class RoutedOutputTransform(nn.Module):
    def __init__(
        self,
        norm: RMSNorm | None,
        # Replicated or intra-chip sharded; both return ``(output, bias)``.
        up_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        return hidden_states


class LatentMoE(nn.Module):
    """Routed experts, optionally in a latent space, plus shared experts.

    The router scores the full-width input. With ``latent_size`` the routed
    experts see ``routed_expert_down_proj(x)`` and their reduced output goes
    through the optional ``routed_expert_norm`` and ``routed_expert_up_proj``
    (vLLM's routed input and output transforms).

    Args:
        intermediate_size: Routed expert width. Below
            ``min_intermediate_per_partition`` per MoE TP shard it is
            zero-padded up to that.
        shared_experts: Added to the routed output by the fused-MoE runner.
        latent_norm_eps: Adds ``routed_expert_norm`` with this epsilon when
            ``latent_size`` is set.
        use_score_correction_bias: Adds the router's
            ``gate.e_score_correction_bias``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None,
        prefix: str,
        shared_experts: nn.Module | None,
        scoring_func: str,
        renormalize: bool,
        routed_scaling_factor: float,
        use_score_correction_bias: bool,
        latent_size: int | None = None,
        latent_norm_eps: float | None = None,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        situ_beta: float | None = None,
        situ_linear_beta: float | None = None,
        min_intermediate_per_partition: int = 128,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config_or_none()
        self.use_ep = (
            vllm_config is not None
            and vllm_config.parallel_config.enable_expert_parallel
        )
        expert_hidden_size = latent_size if latent_size is not None else hidden_size

        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.use_score_correction_bias = use_score_correction_bias
        if use_score_correction_bias:
            self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))

        # Replicated by default; sharded across a chip's cores under
        # TPU_LATENT_PROJ_INTRA_CHIP_TP. These two are a latent model's
        # largest unquantized weights and are read in full by every rank each
        # step.
        self.routed_expert_down_proj = (
            make_latent_projection(
                hidden_size,
                expert_hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            if latent_size is not None
            else None
        )
        self.routed_expert_norm = (
            RMSNorm(expert_hidden_size, latent_norm_eps)
            if self.routed_expert_down_proj is not None and latent_norm_eps is not None
            else None
        )
        self.routed_expert_up_proj = (
            make_latent_projection(
                expert_hidden_size,
                hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            if self.routed_expert_down_proj is not None
            else None
        )
        routed_output_transform = (
            RoutedOutputTransform(self.routed_expert_norm, self.routed_expert_up_proj)
            if self.routed_expert_up_proj is not None
            else None
        )
        self.shared_experts = shared_experts

        padded_intermediate_size = intermediate_size
        if self.use_ep:
            split = hierarchical_split_or_none()
            moe_tp_size = split[2] if split is not None else 1
        else:
            moe_tp_size = self.tp_size
        if (
            moe_tp_size > 1
            and padded_intermediate_size // moe_tp_size < min_intermediate_per_partition
        ):
            padded_intermediate_size = min_intermediate_per_partition * moe_tp_size

        is_situ = hidden_act == "situ"
        # Under TPU_MOE_HIERARCHICAL_EP this builds the routed experts
        # with expert parallelism between chips and tensor parallelism
        # within one chip, instead of vLLM's flat all-experts-per-device EP.
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=expert_hidden_size,
                intermediate_size=padded_intermediate_size,
                activation=hidden_act,
                activation_situ_beta=situ_beta if is_situ else None,
                activation_situ_linear_beta=situ_linear_beta if is_situ else None,
                renormalize=renormalize,
                quant_config=quant_config,
                use_grouped_topk=use_grouped_topk,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=scoring_func,
                e_score_correction_bias=(
                    cast(torch.Tensor, self.gate.e_score_correction_bias)
                    if use_score_correction_bias
                    else None
                ),
                routed_scaling_factor=routed_scaling_factor,
                routed_input_transform=self.routed_expert_down_proj,
                routed_output_transform=routed_output_transform,
            )
        if padded_intermediate_size != intermediate_size:
            routed_experts = self.experts.routed_experts
            w13_weight = getattr(routed_experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = cast(torch.Tensor, routed_experts.w13_weight_packed)
            w2_weight = getattr(routed_experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = cast(torch.Tensor, routed_experts.w2_weight_packed)
            w13_weight.data.zero_()
            w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                intermediate_size // moe_tp_size
            )

    def initialize_dummy_router(self) -> None:
        """Use fan-in random weights and fixed, distinct random layer biases.

        The generic dummy loader's +/-0.001 weights tie every router logit,
        and make the correction bias dominate sigmoid routing scores. Match
        the prototype's unit logit variance for normalized inputs without
        imposing any route counts.
        """
        config = get_current_vllm_config_or_none()
        if config is None or str(config.load_config.load_format).lower() != "dummy":
            return
        generator = torch.Generator(device="cpu").manual_seed(
            1234 + zlib.crc32(self.prefix.encode())
        )
        updates: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.use_score_correction_bias:
            bias_param = cast(torch.Tensor, self.gate.e_score_correction_bias)
            bias = (
                torch.randn(
                    bias_param.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.02
            )
            updates.append((bias_param, bias))
        weight_param = cast(torch.Tensor, self.gate.weight)
        weight = torch.randn(
            weight_param.shape, generator=generator, dtype=torch.float32
        )
        # The TPU dense loader transposes weights to [input, output]. Use
        # the logical fan-in rather than depending on the stored layout.
        weight *= self.gate.input_size**-0.5
        updates.append((weight_param, weight))
        with torch.no_grad():
            for param, value in updates:
                param.copy_(value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return output.view(num_tokens, hidden_size)
