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
"""MiniMax M3 feed-forward blocks."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedConfig
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
)
from vllm_torchtpu.models.minimax_m3.expert_placement import ExpertPlacement

# The checkpoint config may omit swiglu_beta; SwiGLU-OAI's up-bias is 1.
_DEFAULT_SWIGLU_BETA = 1.0


def swiglu_params(config: PreTrainedConfig) -> tuple[float, float, float]:
    """``(limit, alpha, beta)`` of the SwiGLU-OAI activation."""
    if config.hidden_act != "swigluoai":
        raise ValueError(f"Unsupported activation {config.hidden_act!r}")
    return (
        config.swiglu_limit,
        config.swiglu_alpha,
        getattr(config, "swiglu_beta", _DEFAULT_SWIGLU_BETA),
    )


class MiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP: ``g * sigmoid(alpha * g) * (u + beta)``."""

    def __init__(
        self,
        config: PreTrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        limit, alpha, beta = swiglu_params(config)
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=limit, alpha=alpha, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class MiniMaxM3MoE(nn.Module):
    """Sigmoid-routed experts plus one always-on shared expert.

    The routing bias only picks the experts; the picked sigmoid scores are
    renormalized and scaled by ``routed_scaling_factor``.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        num_experts = config.num_local_experts
        self.gate = GateLinear(
            config.hidden_size,
            num_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        self.shared_experts = MiniMaxM3MLP(
            config,
            config.shared_intermediate_size * config.n_shared_experts,
            quant_config,
            prefix=f"{prefix}.shared_experts",
            reduce_results=False,
        )
        limit, alpha, beta = swiglu_params(config)
        # Under TPU_MOE_HIERARCHICAL_EP each chip's two cores split the
        # intermediate columns of the chip's experts instead of owning half
        # the experts each: the same bytes per core, but a decode step's
        # touched experts spread over pairs of cores, so the core that sets
        # the post-MoE all-reduce streams fewer of them.
        placement = ExpertPlacement(num_experts)
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                # Replica slots (expert_placement.py) are extra physical experts.
                num_experts=placement.num_physical_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                scoring_func=config.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias,
                renormalize=True,
                # w13 is [all gates; all ups], not gpt-oss's interleaved layout.
                activation="swigluoai_uninterleave",
                swiglu_limit=limit,
                swiglu_alpha=alpha,
                swiglu_beta=beta,
                routed_scaling_factor=config.routed_scaling_factor,
                router_logits_dtype=torch.float32,
                shared_experts=self.shared_experts,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
            )
        table = placement.replica_table(prefix)
        if table is not None:
            self._arm_replicas(placement, table, prefix)

    def _arm_replicas(
        self, placement: ExpertPlacement, table: list[int], prefix: str
    ) -> None:
        """Hand the layer's replica table to the routed experts' op: the dense
        fp4 decode kernel picks the copies, other steps run the home copies."""
        from vllm_torchtpu.models.minimax_m3.quant import MiniMaxM3Nvfp4MoEMethod

        routed = self.experts.routed_experts
        if not isinstance(routed.quant_method, MiniMaxM3Nvfp4MoEMethod):
            raise ValueError(
                f"Expert replicas need NVFP4 experts; {prefix} runs "
                f"{type(routed.quant_method).__name__}."
            )
        if routed.moe_config.ep_size != placement.groups:
            raise ValueError(
                f"The expert placement's replicas need {placement.groups} EP ranks "
                f"(TPU_MOE_HIERARCHICAL_EP); {prefix} has "
                f"{routed.moe_config.ep_size}."
            )
        # Every layer's table has the same length: unused rows name slot -1.
        rows = placement.groups * placement.replicas_per_group
        table = table + [-1, 0, 0, 0] * (rows - len(table) // 4)
        routed.register_buffer(
            "_expert_replicas",
            torch.tensor(table, dtype=torch.int32),
            persistent=False,
        )
        routed.expert_replica_groups = placement.groups

    def carries_router_weight(self) -> bool:
        """Whether `forward` with a ``gate_weight`` copy computes what
        `self.gate` does: GateLinear's plain unquantized linear in the
        weight's dtype, which it runs when no specialized router GEMM tier
        applies (always on TPU)."""
        gate = self.gate
        tiers = (
            "allow_ll_bf16_gemm",
            "allow_rocm_bf16x3_router_gemm",
            "allow_fp32_router_gemm",
            "allow_bf16x3_router_gemm",
            "allow_cublas_router_gemm",
        )
        return (
            type(gate.quant_method) is UnquantizedLinearMethod
            and gate.bias is None
            and gate.out_dtype == gate.weight.dtype
            and not any(getattr(gate, tier, False) for tier in tiers)
        )

    def forward(
        self, hidden_states: torch.Tensor, gate_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``gate_weight``: a copy of the router weight placed for this call
        (an all-reduce carried it into VMEM), else None."""
        if gate_weight is None:
            router_logits, _ = self.gate(hidden_states)
        else:
            # GateLinear's path when `carries_router_weight`.
            router_logits = F.linear(hidden_states.to(gate_weight.dtype), gate_weight)
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)
