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
"""`InterleavedPcpShardingPlan`: round-robin token sharding, the token layout
of the legacy PCP streaming path (`USE_TPU_PCP_MANAGER=0`).

The step's request-major tokens form one flat stream; rank `r` holds the
`interleave_size`-token chunks `r, r + P, r + 2P, ...` of it. The attention
metadata stays global request-major, and the kernels (`batched_rpa_new`
`sharding_mode="interleaved"`, the GDN PCP path) derive their rank-local
schedules from it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_torchtpu.distributed.pcp import all_gather_equal_tokens
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    apply_pcp_rank_major_token_order,
    pcp_local_token_counts,
)
from vllm_torchtpu.runner.pcp.sharding_plan import (
    PcpShardingPlan,
    grow_host_token_buffers,
    round_up_to_bucket,
)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata


class InterleavedPcpShardingPlan(PcpShardingPlan):
    """Rank `r` holds chunks `r, r + P, ...` of `interleave_size` tokens of the
    step's flat request-major token stream."""

    def __init__(
        self,
        pcp_size: int,
        pcp_rank: int,
        dcp_size: int,
        interleave_size: int,
        max_num_batched_tokens: int,
    ) -> None:
        if int(interleave_size) <= 0:
            raise ValueError(
                "The interleaved PCP plan needs cp_kv_cache_interleave_size > 0, "
                f"got {interleave_size}."
            )
        super().__init__(pcp_size, pcp_rank, dcp_size)
        self.interleave_size = int(interleave_size)
        cycle = self.pcp_size * self.interleave_size
        self._max_local_num_tokens = (
            -(-int(max_num_batched_tokens) // cycle) * self.interleave_size
        )

    @property
    def local_reqs_per_req(self) -> int:
        return 1

    @property
    def max_local_num_tokens(self) -> int:
        return self._max_local_num_tokens

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
        del scheduler_output, use_max_model_len, target_num_reqs
        self._reset_step_state()
        num_tokens = int(total_num_scheduled_tokens)
        q_lens = np.asarray(num_scheduled_tokens_per_req[:num_reqs], dtype=np.int64)
        rows = slice(start_index, start_index + num_reqs)
        num_computed = runner.input_batch.num_computed_tokens_cpu[rows].astype(np.int64)
        num_prompt = np.asarray(
            runner.input_batch.num_prompt_tokens[rows], dtype=np.int64
        )
        if np.any(q_lens <= 0):
            raise NotImplementedError(
                "The interleaved PCP plan needs scheduled tokens for every request."
            )
        if np.any((num_computed < num_prompt) & (num_computed + q_lens > num_prompt)):
            raise NotImplementedError(
                "The interleaved PCP plan does not support a request whose "
                "scheduled tokens cross its prompt/decode boundary."
            )

        # Ownership follows the step's flat token stream (`query_start_loc[:-1]`),
        # as the kernels assume.
        owner_starts = np.zeros(num_reqs, dtype=np.int64)
        owner_starts[1:] = np.cumsum(q_lens[:-1])
        rank_counts = pcp_local_token_counts(
            q_lens,
            self.pcp_size,
            self.interleave_size,
            token_owner_start_offsets_per_req=owner_starts,
        )
        local_padded = round_up_to_bucket(
            runner.num_tokens_paddings, max(1, int(rank_counts.max(initial=0)))
        )
        dp_target_bucket = getattr(runner, "_dp_target_bucket", None)
        if dp_target_bucket is not None and dp_target_bucket > local_padded:
            local_padded = int(dp_target_bucket)

        # Pack every rank's rows rank-major; this rank reads its own slice.
        num_packed = local_padded * self.pcp_size
        grow_host_token_buffers(runner, num_packed)
        mrope = (
            runner.mrope_positions.cpu[:, :num_packed].numpy()
            if getattr(runner, "uses_mrope", False)
            else None
        )
        request_major_to_packed = apply_pcp_rank_major_token_order(
            runner.input_ids_cpu[:num_packed].numpy(),
            runner.positions_np[:num_packed],
            q_lens,
            self.pcp_size,
            self.interleave_size,
            num_packed,
            mrope,
            token_owner_start_offsets_per_req=owner_starts,
        )

        local_start = self.pcp_rank * local_padded
        local_stop = local_start + local_padded
        is_local = (request_major_to_packed >= local_start) & (
            request_major_to_packed < local_stop
        )
        self.request_major_to_local = np.where(
            is_local, request_major_to_packed - local_start, -1
        ).astype(np.int32)

        # Sampling rows: each request's last token, contributed by its owner.
        last_packed = request_major_to_packed[np.cumsum(q_lens) - 1]
        owner_mask = np.zeros(padded_num_reqs, dtype=bool)
        owner_mask[:num_reqs] = (last_packed >= local_start) & (
            last_packed < local_stop
        )
        logits_local = np.zeros(padded_num_reqs, dtype=np.int32)
        logits_local[:num_reqs] = np.where(
            owner_mask[:num_reqs], last_packed - local_start, 0
        )
        logits_indices = np.full(padded_num_reqs, -1, dtype=np.int32)
        logits_indices[:num_reqs] = last_packed

        self.token_slice = slice(local_start, local_stop)
        self.global_num_tokens = num_tokens
        self.global_padded_num_tokens = num_packed
        self.local_num_tokens = int(rank_counts[self.pcp_rank])
        self.local_padded_num_tokens = local_padded
        self.local_num_reqs = int(num_reqs)
        self.logits_indices_cpu = torch.from_numpy(logits_indices)
        self.logits_local_indices_cpu = torch.from_numpy(logits_local)
        self.logits_owner_mask_cpu = torch.from_numpy(owner_mask)

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        del max_num_tokens
        self._reset_step_state()
        num_tokens = int(num_tokens)
        self.token_slice = slice(0, num_tokens)
        self.global_num_tokens = num_tokens * self.pcp_size
        self.global_padded_num_tokens = num_tokens * self.pcp_size
        self.local_num_tokens = num_tokens
        self.local_padded_num_tokens = num_tokens
        self.local_num_reqs = int(num_reqs)

    def build_metadata(
        self,
        *,
        runner: Any,
        block_table_obj: Any,
        target_num_reqs: int,
        target_num_blocks: int,
        split_block_tables: torch.Tensor | None = None,
    ) -> PcpMetadata | None:
        del runner, block_table_obj, target_num_reqs, target_num_blocks
        del split_block_tables
        return None

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Every rank gets the rank-major packed rows of the whole step.
        return all_gather_equal_tokens(hidden_states, dim=0)
