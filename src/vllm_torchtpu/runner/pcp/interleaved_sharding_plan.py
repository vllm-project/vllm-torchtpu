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
"""InterleavedPcpShardingPlan

Deals each step's request-major tokens round-robin across PCP ranks in chunks
of `cp_kv_cache_interleave_size` tokens for streamed PCP attention and GDN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_torchtpu.distributed.pcp import all_gather_equal_tokens
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
)
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutEligibility,
    PcpSequenceLayoutPlanner,
)
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutPlan
from vllm_torchtpu.pcp_sharding_plan import PcpShardingPlanName
from vllm_torchtpu.platforms.pcp_validation import (
    interleaved_max_local_tokens,
    validate_interleaved_pcp_params,
)
from vllm_torchtpu.runner.pcp.sharding_plan import (
    PcpShardingPlan,
    assert_host_token_buffer_capacity,
)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata


class InterleavedPcpShardingPlan(PcpShardingPlan):
    """Interleaved round-robin PCP token sharding plan (`dcp_size == 1`)."""

    plan_name = PcpShardingPlanName.INTERLEAVED
    sequence_layout_descriptor = PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR
    requires_backend_preinit = True
    local_compile_token_multiple = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE

    def __init__(
        self,
        *,
        pcp_size: int,
        pcp_rank: int,
        interleave_size: int,
        max_num_batched_tokens: int,
    ) -> None:
        validate_interleaved_pcp_params(pcp_size, 1, interleave_size)
        if not 0 <= pcp_rank < pcp_size:
            raise ValueError(f"pcp_rank must be in [0, {pcp_size}), got {pcp_rank}.")
        super().__init__(pcp_size=pcp_size, pcp_rank=pcp_rank, dcp_size=1)
        self.interleave_size = int(interleave_size)
        self._planner = PcpSequenceLayoutPlanner(
            PcpSequenceLayoutEligibility(
                pcp_size=self.pcp_size,
                interleave_size=self.interleave_size,
                dcp_size=1,
            ),
            pcp_rank=self.pcp_rank,
            reserve_host_buffers=assert_host_token_buffer_capacity,
        )
        self._max_local_num_tokens = interleaved_max_local_tokens(
            max_num_batched_tokens, self.pcp_size, self.interleave_size
        )

    @property
    def max_local_num_tokens(self) -> int:
        return self._max_local_num_tokens

    def padded_num_tokens(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
    ) -> int:
        num_reqs = runner.input_batch.num_reqs
        req_ids = runner.input_batch.req_ids[:num_reqs]
        q_lens = np.array(
            [int(scheduler_output.num_scheduled_tokens[req_id]) for req_id in req_ids],
            dtype=np.int32,
        )
        if q_lens.size == 0 or int(q_lens.sum()) <= 0:
            return 0
        local_padded = self._planner.padded_num_tokens(
            runner=runner,
            scheduler_output=scheduler_output,
            start_index=0,
            num_scheduled_tokens_per_req=q_lens,
        )
        return local_padded * self.pcp_size

    def _apply_layout_plan(
        self,
        plan: SequenceLayoutPlan,
        *,
        local_num_reqs: int,
    ) -> None:
        self.sequence_layout_plan = plan
        self.token_slice = plan.token_slice
        self.global_num_tokens = plan.global_num_tokens
        self.global_padded_num_tokens = plan.global_padded_num_tokens
        self.local_num_tokens = plan.local_num_tokens
        self.local_padded_num_tokens = plan.local_padded_num_tokens
        self.local_num_reqs = int(local_num_reqs)
        self.logits_indices_cpu = plan.logits_indices_cpu

    def prepare_real(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_scheduled_tokens_per_req: Any,
        total_num_scheduled_tokens: int,
        use_max_model_len: bool,
        target_num_reqs: int,
        padded_num_reqs: int,
    ) -> None:
        self._reset_step_state()
        if num_reqs <= 0 or total_num_scheduled_tokens <= 0:
            return
        q_lens = np.asarray(num_scheduled_tokens_per_req[:num_reqs], dtype=np.int32)
        if int(q_lens.sum()) != int(total_num_scheduled_tokens):
            raise ValueError(
                "total_num_scheduled_tokens must match "
                "sum(num_scheduled_tokens_per_req): "
                f"got {total_num_scheduled_tokens} vs {int(q_lens.sum())}."
            )
        encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if encoder_inputs:
            req_ids = runner.input_batch.req_ids[start_index : start_index + num_reqs]
            if any(
                req_id is not None and bool(encoder_inputs.get(req_id))
                for req_id in req_ids
            ):
                raise NotImplementedError(
                    "Interleaved PCP manager does not support multimodal "
                    "encoder inputs."
                )
        plan = self._planner.prepare_real(
            runner=runner,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=q_lens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            use_max_model_len=use_max_model_len,
            target_num_reqs=target_num_reqs,
            padded_num_reqs=padded_num_reqs,
        )
        self._apply_layout_plan(plan, local_num_reqs=num_reqs)
        if (
            plan.logits_local_indices_cpu is not None
            and plan.logits_owner_mask_cpu is not None
        ):
            self._stage_logits_selection(
                runner.device,
                plan.logits_local_indices_cpu,
                plan.logits_owner_mask_cpu,
            )

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        del max_num_tokens
        self._reset_step_state()
        plan = self._planner.prepare_dummy(
            num_tokens=num_tokens, num_reqs=num_reqs, kv_cache_initialized=False
        )
        self._apply_layout_plan(plan, local_num_reqs=num_reqs)

    def build_metadata(
        self,
        *,
        runner: Any,
        block_table_obj: Any,
        target_num_reqs: int,
        target_num_blocks: int,
        split_block_tables: torch.Tensor | None = None,
    ) -> PcpMetadata | None:
        del runner, block_table_obj, target_num_reqs
        del target_num_blocks, split_block_tables
        return None

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return all_gather_equal_tokens(hidden_states, dim=0)
