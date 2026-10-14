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
"""GLM-5.3-Flash mixture-of-experts layer."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

from vllm_torchtpu.models.vllm.glm5_next.layers import Glm5NextMLP


class Glm5NextMoE(nn.Module):
    """288 routed experts + 1 shared expert, DeepSeek-style noaux_tc routing.

    ``n_group``/``topk_group`` are both 1 in the released config, so the group
    mask is a no-op and this reduces to plain sigmoid + bias-corrected top-k.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.routed_scaling_factor = config.routed_scaling_factor

        tp_size = get_tensor_model_parallel_world_size()
        if self.n_routed_experts % tp_size:
            raise ValueError(
                f"n_routed_experts={self.n_routed_experts} must divide "
                f"TP size {tp_size}"
            )

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # noaux_tc: the router adds a learned per-expert bias to the *choice*
        # scores only; the returned weights are the unbiased sigmoid scores.
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
        # The DeepSeek-V4 MoE carries a hash-routing table here; GLM has none,
        # but the shared expert plumbing below still reads the attribute.
        self.gate.tid2eid = None

        intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = Glm5NextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            swiglu_limit=self.swiglu_limit,
            quant_config=quant_config,
            # The fused MoE reduces routed and shared outputs together.
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            apply_routed_scale_to_output=True,
            # The decoder layer reduces: all-reduce, or reduce-scatter in a
            # sequence-parallel step (see `sp.py`).
            reduce_results=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        org_shape = hidden_states.shape
        # FusedMoEFactory gives the runner our gate, so it computes the logits.
        out = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        return out.view(org_shape)
