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
"""PCP manager

`PcpManager` delegates to a stateful `PcpShardingPlan` (`sharding_plan.py`),
selecting `HeadTailPcpShardingPlan` on sparse MLA models and
`InterleavedPcpShardingPlan` on non-MLA models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import get_pcp_group
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutDescriptor
from vllm_torchtpu.pcp_sharding_plan import PcpShardingPlanName
from vllm_torchtpu.platforms.pcp_validation import (
    effective_step_token_budget,
    validate_pcp_manager_config,
)
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import HeadTailPcpShardingPlan
from vllm_torchtpu.runner.pcp.interleaved_sharding_plan import (
    InterleavedPcpShardingPlan,
)
from vllm_torchtpu.runner.pcp.sharding_plan import (
    PcpShardingPlan,
    assert_host_token_buffer_capacity,
)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata


def _layer_supports_pcp_manager(layer: Any, plan: PcpShardingPlanName) -> bool:
    if plan in getattr(layer, "supported_pcp_sharding_plans", ()):
        return True
    impl = getattr(layer, "impl", None)
    return impl is not None and plan in getattr(
        impl, "supported_pcp_sharding_plans", ()
    )


def validate_pcp_manager_layers(
    layers: dict[str, Any],
    plan: PcpShardingPlanName,
) -> None:
    """Rejects attention layers unsupported by the active PCP sharding plan."""
    unsupported = sorted(
        {
            type(layer).__name__
            for layer in layers.values()
            if not _layer_supports_pcp_manager(layer, plan)
        }
    )
    if unsupported:
        raise NotImplementedError(
            "USE_TPU_PCP_MANAGER=1 does not support these attention layers: "
            f"{', '.join(unsupported)}."
        )


class PcpManager:
    """PCP manager owned by `TPUModelRunner`. The runner reads the prepared
    step's state from `sharding_plan`."""

    uses_selected_logits_hidden_states: bool = True

    def __init__(self, sharding_plan: PcpShardingPlan) -> None:
        self.sharding_plan = sharding_plan

    @property
    def plan_name(self) -> PcpShardingPlanName:
        return self.sharding_plan.plan_name

    @property
    def uses_pcp_metadata(self) -> bool:
        return self.sharding_plan.uses_pcp_metadata

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> PcpManager:
        plan = validate_pcp_manager_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        group = get_pcp_group()
        pcp_rank = 0 if group is None else int(group.rank_in_group)
        _, step_token_budget = effective_step_token_budget(vllm_config.scheduler_config)
        if plan is PcpShardingPlanName.HEADTAIL:
            return cls(
                HeadTailPcpShardingPlan(
                    pcp_size=parallel_config.prefill_context_parallel_size,
                    pcp_rank=pcp_rank,
                    dcp_size=parallel_config.decode_context_parallel_size,
                    page_size=vllm_config.cache_config.block_size,
                    workspace_tokens=envs.TPU_PCP_KV_WORKSPACE_TOKENS,
                    max_num_batched_tokens=step_token_budget,
                    max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                )
            )
        return cls(
            InterleavedPcpShardingPlan(
                pcp_size=parallel_config.prefill_context_parallel_size,
                pcp_rank=pcp_rank,
                interleave_size=parallel_config.cp_kv_cache_interleave_size,
                max_num_batched_tokens=step_token_budget,
            )
        )

    @property
    def local_reqs_per_req(self) -> int:
        return self.sharding_plan.local_reqs_per_req

    @property
    def max_local_num_tokens(self) -> int:
        return self.sharding_plan.max_local_num_tokens

    @property
    def sequence_layout_descriptor(self) -> SequenceLayoutDescriptor:
        return self.sharding_plan.sequence_layout_descriptor

    @property
    def requires_backend_preinit(self) -> bool:
        return self.sharding_plan.requires_backend_preinit

    @property
    def backend_preinit_world_size(self) -> int:
        return self.sharding_plan.pcp_size

    @property
    def local_compile_token_multiple(self) -> int:
        return self.sharding_plan.local_compile_token_multiple

    def num_reqs_in_chunk(self, q_lens: Any, seq_lens: Any, capacity: int) -> int:
        return self.sharding_plan.num_reqs_in_chunk(q_lens, seq_lens, capacity)

    def configure_kv_cache(self, kv_cache_config: Any) -> None:
        self.sharding_plan.configure_kv_cache(kv_cache_config)

    def padded_num_tokens(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
    ) -> int:
        return self.sharding_plan.padded_num_tokens(
            runner=runner, scheduler_output=scheduler_output
        )

    def reserve_host_token_capacity(
        self, runner: Any, required_num_tokens: int
    ) -> None:
        assert_host_token_buffer_capacity(runner, required_num_tokens)

    def prepare_real(self, **kwargs: Any) -> None:
        self.sharding_plan.prepare_real(**kwargs)

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        self.sharding_plan.prepare_dummy(
            num_tokens=num_tokens, num_reqs=num_reqs, max_num_tokens=max_num_tokens
        )

    def build_metadata(self, **kwargs: Any) -> PcpMetadata | None:
        return self.sharding_plan.build_metadata(**kwargs)

    def maybe_select_logits_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.sharding_plan.maybe_select_logits_hidden_states(hidden_states)

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.sharding_plan.finalize_hidden_states(hidden_states)
