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
"""Kimi MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.latent_moe import LatentMoE

from .collective_ops import FusedPrefillCollectives
from .layers import KimiMLP

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
        VllmCompressedTensorsW4ANMxfp4MoEMethod,
    )


class KimiMoE(LatentMoE):
    """Standard or latent Kimi MoE, with Kimi K3's sequence-parallel prefill."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        assert config.num_experts is not None
        assert config.num_experts_per_token is not None
        assert config.moe_intermediate_size is not None

        prefill_group = None
        shared_group = None
        if envs.TPU_K3_SP_PREFILL:
            prefill_group = FusedPrefillCollectives(prefix)
            shared_group = get_intra_chip_group()
            if shared_group is None or shared_group.world_size != 2:
                raise ValueError("K3 SP prefill needs a physical two-core chip group")

        shared_experts = (
            KimiMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                tp_group=shared_group,
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
            if config.num_shared_experts > 0
            else None
        )
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            intermediate_size=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
            shared_experts=shared_experts,
            scoring_func=config.moe_router_activation_func,
            renormalize=config.moe_renormalize,
            routed_scaling_factor=config.routed_scaling_factor,
            use_score_correction_bias=True,
            latent_size=config.routed_expert_hidden_size,
            latent_norm_eps=(
                config.rms_norm_eps if config.latent_moe_use_norm else None
            ),
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            situ_beta=config.activation_situ_beta,
            situ_linear_beta=config.activation_situ_linear_beta,
            min_intermediate_per_partition=getattr(
                config, "min_moe_intermediate_per_partition", 128
            ),
        )
        self.prefill_group = prefill_group
        self.shared_group = shared_group
        if self.shared_group is not None and not hasattr(
            self.experts.routed_experts.quant_method, "apply_with_routing"
        ):
            raise ValueError("K3 SP prefill requires the native MXFP4 MoE backend")

    def forward(
        self, hidden_states: torch.Tensor, sequence_parallel: bool = False
    ) -> torch.Tensor:
        if sequence_parallel or self.shared_group is not None:
            return self._forward_expert_parallel(hidden_states, sequence_parallel)
        return super().forward(hidden_states)

    def _forward_expert_parallel(
        self, hidden_states: torch.Tensor, sequence_parallel: bool
    ) -> torch.Tensor:
        """Run EP experts with sharded or replicated tokens, then add shared experts."""
        group = self.prefill_group
        assert group is not None
        router_logits, _ = self.gate(hidden_states)
        latent = hidden_states
        if self.routed_expert_down_proj is not None:
            latent, _ = self.routed_expert_down_proj(latent)
        routed = self.experts.routed_experts
        if sequence_parallel:
            method = cast(
                "VllmCompressedTensorsW4ANMxfp4MoEMethod", routed.quant_method
            )
            weights, ids = moe_routing.route(
                routed,
                latent,
                router_logits,
                topk_weights_float32=method._tpu_topk_weights_float32,
            )
            latent, weights, ids = group.gather_moe_inputs(
                latent, weights, ids, num_experts=self.gate.output_size
            )
            output = method.apply_with_routing(routed, latent, weights, ids)
        else:
            output = routed.quant_method.apply_monolithic(routed, latent, router_logits)
        output = (
            group.reduce_scatter(output, dim=0)
            if sequence_parallel
            else group.all_reduce(output)
        )
        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm(output)
        if self.routed_expert_up_proj is not None:
            output, _ = self.routed_expert_up_proj(output)
        if self.shared_experts is not None:
            output = output + self.shared_experts(
                hidden_states, sequence_parallel=sequence_parallel
            )
        return output
