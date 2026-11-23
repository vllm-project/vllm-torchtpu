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
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)
from vllm_torchtpu.models.minimax_m3.quant import MXFP8_DEQUANTIZE_ATTR

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


def _refuse_hierarchical_runai_load() -> None:
    """MiniMax-M3.1 served with TP-sharded experts through the Run:ai
    streaming loader produced generations that never terminated: with the
    EP weight filter, half the ranks fetched no expert tensors; with the full
    checkpoint stream, the expert stager ended with parameters incomplete.
    The cause is not found. The default loader serves the same split
    correctly, so the split refuses runai_streamer rather than load silently
    wrong weights."""
    if hierarchical_split_or_none() is None:
        return
    load_format = get_current_vllm_config().load_config.load_format
    if str(getattr(load_format, "value", load_format)) == "runai_streamer":
        raise ValueError(
            "TPU_MOE_HIERARCHICAL_EP is not supported with --load-format "
            "runai_streamer for MiniMax M3; load with the default loader or "
            "unset TPU_MOE_HIERARCHICAL_EP."
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
        # bf16: XLA hides gate_up's weight copy behind attention and fuses
        # down_proj with the expert combine; measured faster than e4m3 here.
        for linear in (self.shared_experts.gate_up_proj, self.shared_experts.down_proj):
            setattr(linear, MXFP8_DEQUANTIZE_ATTR, True)
        limit, alpha, beta = swiglu_params(config)
        # Under TPU_MOE_HIERARCHICAL_EP this builds the routed experts with
        # expert parallelism between chips and tensor parallelism within one
        # chip; otherwise it is vLLM's flat expert parallelism.
        _refuse_hierarchical_runai_load()
        with hierarchical_moe_parallel_config():
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)
