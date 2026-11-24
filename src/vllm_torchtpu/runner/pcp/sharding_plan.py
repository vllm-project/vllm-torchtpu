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
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_torchtpu.distributed.pcp import all_reduce_sum

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


def ensure_host_token_buffer_capacity(runner: Any, required_num_tokens: int) -> None:
    """Grow CPU-only token staging buffers for rank-major PCP token packing."""
    required_num_tokens = int(required_num_tokens)
    if required_num_tokens <= 0:
        return

    if len(runner.arange_np) < required_num_tokens:
        runner.arange_np = np.arange(required_num_tokens, dtype=np.int64)

    current_num_tokens = int(runner.input_ids_cpu.shape[0])
    if current_num_tokens >= required_num_tokens:
        return

    attrs = ["input_ids_cpu", "positions_cpu"]
    if runner.supports_mm_inputs and hasattr(runner, "is_mm_embed_cpu"):
        attrs.append("is_mm_embed_cpu")
    for attr in attrs:
        buf = getattr(runner, attr)
        cur = int(buf.shape[0])
        if cur < required_num_tokens:
            grown = torch.zeros(
                required_num_tokens,
                dtype=buf.dtype,
                device=buf.device,
                pin_memory=buf.is_pinned(),
            )
            grown[:cur] = buf
            setattr(runner, attr, grown)
            if attr == "positions_cpu":
                runner.positions_np = grown.numpy()

    if runner.uses_mrope:
        old_mrope = runner.mrope_positions
        mrope_cols = int(old_mrope.cpu.shape[1])
        if mrope_cols < required_num_tokens + 1:
            new_mrope = runner._make_buffer(
                3, required_num_tokens + 1, dtype=old_mrope.cpu.dtype
            )
            new_mrope.cpu[:, :mrope_cols] = old_mrope.cpu[:, :mrope_cols]
            runner.mrope_positions = new_mrope


@dataclass(frozen=True)
class PcpDraftLayout:
    """Immutable per-chunk draft token localization and reduction snapshot."""

    local_to_request_major: np.ndarray
    padded_request_to_packed: np.ndarray
    local_start: int
    local_end: int

    def localize_token_tensor_and_gather_indices(
        self,
        request_major_token_tensor: torch.Tensor,
        request_major_gather_indices: torch.Tensor,
        *,
        num_valid_gathers: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Localize request-major tokens and request-aligned gather indices."""
        num_gathers = request_major_gather_indices.shape[0]
        request_major_gather_indices = torch.nn.functional.pad(
            request_major_gather_indices[:num_valid_gathers],
            (0, num_gathers - num_valid_gathers),
            value=-1,
        )

        source_indices = torch.tensor(
            self.local_to_request_major,
            dtype=torch.long,
            device=request_major_token_tensor.device,
        )
        source_valid = source_indices.ge(0)
        safe_source_indices = source_indices.clamp_min(0)
        local_tokens = torch.index_select(
            request_major_token_tensor, 0, safe_source_indices
        )
        source_mask_shape = (source_valid.shape[0],) + (1,) * (local_tokens.ndim - 1)
        local_tokens = torch.where(
            source_valid.reshape(source_mask_shape),
            local_tokens,
            torch.zeros_like(local_tokens),
        )

        request_to_packed_tensor = torch.tensor(
            self.padded_request_to_packed,
            dtype=torch.long,
            device=request_major_gather_indices.device,
        )
        request_gathers = request_major_gather_indices.to(torch.long)
        gather_valid = request_gathers.ge(0)
        safe_request_gathers = request_gathers.clamp_min(0)
        packed_gathers = torch.index_select(
            request_to_packed_tensor, 0, safe_request_gathers.reshape(-1)
        ).reshape(request_gathers.shape)
        owner_mask = (
            gather_valid
            & packed_gathers.ge(self.local_start)
            & packed_gathers.lt(self.local_end)
        )
        local_gathers = torch.where(
            owner_mask,
            packed_gathers - self.local_start,
            torch.full_like(packed_gathers, -1),
        )
        return local_tokens, local_gathers.to(request_major_gather_indices.dtype)

    def aggregate_request_aligned_tensor(
        self,
        local_tensor: torch.Tensor,
        owner_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Zero non-owner rows and sum request-aligned rows across PCP ranks."""
        masked = torch.where(
            owner_mask.unsqueeze(-1), local_tensor, torch.zeros_like(local_tensor)
        )
        return all_reduce_sum(masked)


class PcpShardingPlanName(StrEnum):
    HEADTAIL = "headtail"
    INTERLEAVED = "interleaved"


class PcpShardingPlan(ABC):
    """How a step's tokens are sharded over PCP ranks and restored after.

    `prepare_real` / `prepare_dummy` overwrite the step state below; the
    runner reads it until the next prepare.
    """

    plan_name: PcpShardingPlanName
    local_reqs_per_req: int = 1
    requires_backend_preinit: bool = False
    local_compile_token_multiple: int = 1
    uses_pcp_metadata: bool = False
    uses_pcp_streaming: bool = False

    def __init__(self, pcp_size: int, pcp_rank: int, dcp_size: int) -> None:
        self.pcp_size = int(pcp_size)
        self.pcp_rank = int(pcp_rank)
        self.dcp_size = int(dcp_size)
        self.page_size = 0
        self._reset_step_state()

    def _reset_step_state(self) -> None:
        self._keepalive: list[torch.Tensor] = []
        # This rank's rows `token_slice` of the rank-major packed tokens.
        self.token_slice = slice(0, 0)
        self.global_num_tokens = 0
        self.global_padded_num_tokens = 0
        self.local_num_tokens = 0
        self.local_padded_num_tokens = 0
        # Requests of the attention metadata.
        self.local_num_reqs = 0
        # Request-major token index for each local padded row (-1 for padding).
        self.local_to_request_major_token_indices: np.ndarray | None = None
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
    def padded_num_tokens(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_scheduled_tokens_per_req: Any,
    ) -> int:
        """Return `local_padded_num_tokens` for this chunk without staging."""

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

    def draft_layout(self) -> PcpDraftLayout | None:
        return None

    def split_mask(self, q_lens: Any) -> np.ndarray:
        """Which requests are split over the PCP ranks: those scheduling more
        than one token. The rest (decodes) are replicated on every rank."""
        # TODO(weiyulin): Spec decode verify windows also schedule several
        # tokens; tell them apart here once PCP x DCP supports spec decode.
        return np.asarray(q_lens) > 1

    def num_reqs_in_chunk(self, q_lens: Any, seq_lens: Any, capacity: int) -> int:
        """Requests that fit `capacity` local requests, given each one's
        scheduled tokens (`q_lens`; a split request can become
        `local_reqs_per_req` of them) and its context length after this step
        (`seq_lens`)."""
        del seq_lens
        if self.local_reqs_per_req == 1:
            return min(len(q_lens), max(capacity, 1))
        # TODO(weiyulin): Remove once the head and tail chunks share one
        # page table row.
        num_local = 0
        is_split = self.split_mask(q_lens)
        for i, split in enumerate(is_split):
            num_local += self.local_reqs_per_req if split else 1
            if num_local > capacity:
                return max(i, 1)
        return len(is_split)

    def request_distribution_counts(
        self, chunk_num_decode: int, num_reqs: int
    ) -> tuple[int, int]:
        """`(decode_end, num_reqs)` of the main attention metadata."""
        return chunk_num_decode, num_reqs

    def local_index_for_request_major_token(
        self, request_major_index: int
    ) -> int | None:
        idx = int(request_major_index)
        if self.request_major_to_local is not None:
            if not 0 <= idx < self.request_major_to_local.shape[0]:
                return None
            local = int(self.request_major_to_local[idx])
            return None if local < 0 else local
        start, stop = self.token_slice.start, self.token_slice.stop
        if not start <= idx < stop:
            return None
        return idx - start

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
