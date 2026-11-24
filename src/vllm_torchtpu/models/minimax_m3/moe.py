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
from torch import nn
from transformers import PreTrainedConfig
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

import vllm_torchtpu.envs as envs

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

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """``x`` may come as e4m3 rows with per-row f32 scales (an SP step's
        e4m3 all-gather), which gate_up_proj contracts as e4m3."""
        x, _ = self.down_proj(self.activation(x))
        return x

    def activation(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """``act_fn(gate_up_proj(x))``: the down projection's input."""
        if isinstance(x, tuple):
            e4m3_apply = getattr(self.gate_up_proj.quant_method, "apply_e4m3_rows", None)
            if e4m3_apply is None:
                gate_up, _ = self.gate_up_proj((x[0].float() * x[1]).to(torch.bfloat16))
            else:
                # gate_up_proj is bias-free and keeps the linear's forward = apply.
                gate_up = e4m3_apply(self.gate_up_proj, *x)
        else:
            gate_up, _ = self.gate_up_proj(x)
        return self.act_fn(gate_up)


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
        self.experts = FusedMoEFactory(
            num_experts=num_experts,
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

    def forward(
        self, hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """``hidden_states`` may come as e4m3 rows with per-row f32 scales (an
        SP step's e4m3 all-gather): the router and the experts (shared expert
        included) take them dequantized to bf16."""
        if isinstance(hidden_states, tuple):
            q8, row_scale = hidden_states
            hidden_states = (q8.float() * row_scale).to(torch.bfloat16)
        router_logits, _ = self.gate(hidden_states)
        if self._shared_in_combine():
            return self._forward_shared_in_combine(hidden_states, router_logits)
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)

    def _shared_in_combine(self) -> bool:
        """TPU_MINIMAX_M3_SHARED_IN_COMBINE on a step whose layer reduces the
        MoE output itself (SP steps: the runner would return shared + routed
        unreduced), with the plain GMM path, routing weights carrying the
        routed scale, and an e4m3 per-channel shared down_proj."""
        if not envs.TPU_MINIMAX_M3_SHARED_IN_COMBINE:
            return False
        runner = self.experts
        if not runner.moe_config.skip_final_all_reduce:
            return False
        down = self.shared_experts.down_proj
        qm = runner.routed_experts.quant_method
        return (
            getattr(qm, "shared_in_combine_supported", None) is not None
            and qm.shared_in_combine_supported()
            and runner.routed_scaling_factor == 1.0
            and down.weight.dtype == torch.float8_e4m3fn
            and getattr(down, "weight_scale", None) is not None
            and down.weight_scale.dim() == 1
            and getattr(down, "bias", None) is None
        )

    def _forward_shared_in_combine(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """The runner's unreduced shared + routed sum, with the shared
        down_proj run inside the TensorCore combine (same roundings)."""
        down = self.shared_experts.down_proj
        act = self.shared_experts.activation(hidden_states)
        routed = self.experts.routed_experts
        return routed.quant_method.apply_monolithic_shared(
            routed,
            hidden_states,
            router_logits,
            act,
            down.weight,
            down.weight_scale,
        )
