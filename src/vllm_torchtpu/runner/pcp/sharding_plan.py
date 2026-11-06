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
"""`PcpShardingPlan`: how the PCP manager shards a step's tokens over the PCP
ranks and restores them after the forward (`prepare -> forward -> restore`).
Token rows are packed rank-major (`[pcp_size * num_local_tokens_padded]`).
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_torchtpu.distributed.pcp import all_reduce_sum
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    SequenceLayoutDescriptor,
    SequenceLayoutPlan,
)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata


def round_up_to_bucket(paddings: list[int] | tuple[int, ...], value: int) -> int:
    index = bisect.bisect_left(paddings, value)
    if index >= len(paddings):
        raise ValueError(
            "PCP local token count exceeds the runner's largest compile bucket: "
            f"required_tokens={value} "
            f"max_compile_bucket={paddings[-1] if paddings else None}."
        )
    return int(paddings[index])


class PcpShardingPlan(ABC):
    """How a step's tokens are sharded over PCP ranks and restored after.

    `prepare_real` / `prepare_dummy` overwrite the step state below; the
    runner reads it until the next prepare.
    """

    local_reqs_per_req: int = 1
    sequence_layout_descriptor: SequenceLayoutDescriptor = (
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR
    )
    requires_backend_preinit: bool = False
    local_compile_token_multiple: int = 1

    def __init__(self, pcp_size: int, pcp_rank: int, dcp_size: int) -> None:
        self.pcp_size = int(pcp_size)
        self.pcp_rank = int(pcp_rank)
        self.dcp_size = int(dcp_size)
        self.page_size = 0
        self._reset_step_state()

    def _reset_step_state(self) -> None:
        self.sequence_layout_plan: SequenceLayoutPlan | None = None
        self._keepalive: list[torch.Tensor] = []
        # This rank's rows `token_slice` of the rank-major packed tokens.
        self.token_slice = slice(0, 0)
        self.global_num_tokens = 0
        self.global_padded_num_tokens = 0
        self.local_num_tokens = 0
        self.local_padded_num_tokens = 0
        # Requests of the attention metadata, and their persistent-batch rows
        # (None: the runner's request order).
        self.local_num_reqs = 0
        self.local_block_rows: np.ndarray | None = None
        # This rank's local row of each request-major token, -1 if another
        # rank holds it (None: `token_slice` maps it).
        self.request_major_to_local: np.ndarray | None = None
        # Global request-major sampling rows; each request's last token is
        # local row `logits_local_indices[i]` on the ranks in
        # `logits_owner_mask`. Both are on the device, uploaded in
        # `prepare_real` before the runner's input staging fence.
        self.logits_indices_cpu: torch.Tensor | None = None
        self.logits_local_indices: torch.Tensor | None = None
        self.logits_owner_mask: torch.Tensor | None = None
        # No split request: every rank holds every sampling row.
        self.logits_all_local = False

    @property
    @abstractmethod
    def max_local_num_tokens(self) -> int: ...

    def set_page_size(self, page_size: int) -> None:
        self.page_size = int(page_size)

    def configure_kv_cache(
        self, kv_cache_config: Any, kernel_block_size: int | None = None
    ) -> None:
        del kv_cache_config, kernel_block_size

    @abstractmethod
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
        """Repack the runner's host buffers for this rank and plan the step."""

    @abstractmethod
    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        """Plan a dummy/warmup step of `num_tokens` local tokens, in the shapes
        a real step of the runner's `max_num_tokens` bucket set uses."""

    @abstractmethod
    def build_metadata(
        self,
        *,
        runner: Any,
        block_table_obj: Any,
        target_num_reqs: int,
        target_num_blocks: int,
        split_block_tables: torch.Tensor | None = None,
    ) -> PcpMetadata | None: ...

    def num_reqs_in_chunk(
        self, is_prefilling: Any, seq_lens: Any, capacity: int
    ) -> int:
        """Requests that fit `capacity` local requests, given whether each is
        prefilling (a prefill can become `local_reqs_per_req` of them) and its
        context length after this step (`seq_lens`)."""
        del seq_lens
        if self.local_reqs_per_req == 1:
            return min(len(is_prefilling), max(capacity, 1))
        # TODO(weiyulin): Remove once the head and tail chunks share one
        # page table row.
        num_local = 0
        for i, prefilling in enumerate(is_prefilling):
            num_local += self.local_reqs_per_req if prefilling else 1
            if num_local > capacity:
                return max(i, 1)
        return len(is_prefilling)

    def request_distribution_counts(
        self, chunk_num_decode: int, num_reqs: int
    ) -> tuple[int, int]:
        """`(decode_end, num_reqs)` of the main attention metadata."""
        return chunk_num_decode, num_reqs

    def local_index_for_request_major_token(
        self, request_major_index: int
    ) -> int | None:
        if self.sequence_layout_plan is not None:
            return self.sequence_layout_plan.local_index_for_request_major_token(
                request_major_index
            )
        if self.request_major_to_local is not None:
            local = int(self.request_major_to_local[int(request_major_index)])
            return None if local < 0 else local
        start, stop = self.token_slice.start, self.token_slice.stop
        if not start <= request_major_index < stop:
            return None
        return request_major_index - start

    def maybe_select_logits_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """The sampling rows of every request, or None to sample from
        `finalize_hidden_states`."""
        if self.logits_local_indices is None or self.logits_owner_mask is None:
            return None
        selected = hidden_states.index_select(0, self.logits_local_indices)
        if self.logits_all_local:
            return selected
        # The owner of each request's last token contributes it, the others
        # zeros, summed over PCP.
        mask = self.logits_owner_mask
        selected = torch.where(mask.unsqueeze(1), selected, torch.zeros_like(selected))
        return all_reduce_sum(selected)

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states
