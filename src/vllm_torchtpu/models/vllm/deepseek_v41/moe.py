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
"""DeepSeek-V4.1 MoE layer with non-SP expert-parallel router gate slicing."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.distributed.parallel_state import get_ep_group, get_tp_group

from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig


class DeepseekV41MoE(DeepseekV4MoE):
    """TPU-adapted DeepSeek-V4.1 MoE block with non-SP EP gate input slicing.

    Data lifecycle:
        - Created: Shallow ``VllmConfig`` / ``ModelConfig`` / ``hf_config`` copies
          with ``num_hash_layers = 0`` during ``__init__`` (never mutates caller
          config in place); sliced ``router_logits`` and output activation view in
          ``forward``.
        - Used: ``hidden_states`` and optional ``input_ids`` (read-only).
        - Changed: Detaches ``self.experts.gate = None`` on the constructed
          instance so routing runs explicitly on rank-sliced tokens in ``forward``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "layers.0.ffn",
        use_sequence_parallel: bool = False,
        reduce_results: bool = True,
    ) -> None:
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("V4.1 TPU MoE does not support EPLB")
        if vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS:
            raise NotImplementedError("V4.1 TPU requires the native V4 MoE backend")

        # Pass num_hash_layers=0 via shallow config copies so the shared
        # vllm_config.model_config.hf_config is never mutated in place.
        hf_cfg = copy.copy(vllm_config.model_config.hf_config)
        hf_cfg.num_hash_layers = 0
        model_cfg = copy.copy(vllm_config.model_config)
        model_cfg.hf_config = hf_cfg
        moe_vllm_cfg = copy.copy(vllm_config)
        moe_vllm_cfg.model_config = model_cfg

        super().__init__(vllm_config=moe_vllm_cfg, prefix=prefix)
        self.use_sequence_parallel = use_sequence_parallel
        self.reduce_results = reduce_results
        # V4.1 evaluates self.gate on rank-sliced tokens in forward() before
        # invoking self.experts, so detach the gate from the parent MoERunner.
        self.experts.gate = None

    def _maybe_slice_gate_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Slice ``hidden_states`` to the local EP/TP rank shard when monolithic EP.

        Args:
            hidden_states: Input hidden states ``[num_tokens, hidden_size]`` (Used).

        Returns:
            Rank-local ``hidden_states`` slice or full ``hidden_states`` (Created/Used).

        Raises:
            ValueError: If ``num_tokens`` is not divisible by the EP/TP group size.
        """
        moe_config = self.experts.moe_config
        routed_experts = self.experts.routed_experts
        quant_method = routed_experts.quant_method
        group = get_tp_group() if moe_config.dp_size > 1 else get_ep_group()
        num_tokens = hidden_states.shape[0]
        world_size = group.world_size
        use_ep = moe_config.moe_parallel_config.use_ep and quant_method.is_monolithic
        has_custom_routing = (
            quant_method.supports_internal_mk
            or routed_experts.custom_routing_function is not None
        )
        if (
            self.use_sequence_parallel
            or moe_config.pcp_size > 1
            or not use_ep
            or has_custom_routing
            or world_size <= 1
        ):
            return hidden_states
        if num_tokens % world_size != 0:
            raise ValueError(
                f"Tokens ({num_tokens}) must be divisible by group size ({world_size})"
            )
        chunk_tokens = num_tokens // world_size
        start = group.rank_in_group * chunk_tokens
        return hidden_states[start : start + chunk_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate router gate on rank-sliced tokens and run fused experts.

        Args:
            hidden_states: Input token activations ``[num_tokens, hidden_size]`` (Used).
            input_ids: Optional token IDs ``[num_tokens]`` forwarded to experts (Used).

        Returns:
            MoE output tensor of shape ``hidden_states.shape`` (Created).
        """
        gate_input = self._maybe_slice_gate_input(hidden_states)
        router_logits, _ = self.gate(gate_input)
        expert_output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        return expert_output.view(hidden_states.shape)
