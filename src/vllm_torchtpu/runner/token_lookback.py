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
"""Event-fenced 4-slot host staging ring for per-request token lookback."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_torchtpu.runner.bounded_replay import BoundedReplayTracker

# Avoid single-slot H2D overwrite races under V1 async_scheduling
# (max_concurrent_batches=2): 4 ring slots give 2x headroom plus event fences.
LOOKBACK_RING_DEPTH: int = 4
LOOKBACK_PAD_TOKEN_ID: int = -1


class TokenLookbackRing:
    """4-slot ``torch.tpu.Event``-fenced host staging ring for lookback tokens.

    Each slot holds an ``int32[max_num_reqs, depth]`` host staging buffer paired
    with its most recent ``torch.tpu.Event`` fence. Buffers and fences are
    allocated together so their lengths stay invariant.
    """

    def __init__(self, buffers: tuple[Any, ...]) -> None:
        if not buffers:
            raise ValueError("TokenLookbackRing requires at least one staging buffer")
        self._buffers: tuple[Any, ...] = buffers
        self._fences: list[Any] = [None] * len(buffers)
        self._idx: int = 0

    @classmethod
    def create(
        cls,
        make_buffer: Callable[..., Any],
        max_num_reqs: int,
        depth: int,
        *,
        ring_depth: int = LOOKBACK_RING_DEPTH,
    ) -> TokenLookbackRing:
        buffers = tuple(
            make_buffer(max_num_reqs, depth, dtype=torch.int32)
            for _ in range(ring_depth)
        )
        return cls(buffers)

    def prepare(
        self,
        input_batch: Any,
        num_reqs: int,
        *,
        start_index: int = 0,
        async_scheduling: bool,
        device_type: str,
        replay_tracker: BoundedReplayTracker | None = None,
    ) -> torch.Tensor:
        """Stage per-request trailing token windows and upload to device."""
        if async_scheduling:
            slot_idx = self._idx % len(self._buffers)
            prior_fence = self._fences[slot_idx]
            if prior_fence is not None:
                prior_fence.synchronize()
                self._fences[slot_idx] = None
            staging_buf = self._buffers[slot_idx]
            self._idx += 1
        else:
            slot_idx = 0
            staging_buf = self._buffers[0]

        staging_buf.np.fill(LOOKBACK_PAD_TOKEN_ID)
        if num_reqs > 0:
            depth = staging_buf.np.shape[1]
            end_idx = start_index + num_reqs
            starts = input_batch.num_computed_tokens_cpu[start_index:end_idx, None]
            lookback_positions = starts - np.arange(1, depth + 1)
            limits = input_batch.num_prompt_tokens[start_index:end_idx, None]
            if replay_tracker:
                chunk_req_ids = input_batch.req_ids[start_index:end_idx]
                limits = limits.copy()
                for row, req_id in enumerate(chunk_req_ids):
                    batch_row = start_index + row
                    if replay_tracker.get(req_id) == int(starts[row, 0]):
                        limits[row, 0] = int(input_batch.num_tokens_no_spec[batch_row])
            valid_mask = (lookback_positions >= 0) & (lookback_positions < limits)
            batch_rows = np.arange(start_index, end_idx)[:, None]
            gathered_token_ids = input_batch.token_ids_cpu[
                batch_rows, np.clip(lookback_positions, 0, None)
            ]
            staging_buf.np[:num_reqs] = np.where(
                valid_mask, gathered_token_ids, LOOKBACK_PAD_TOKEN_ID
            )

        lookback_device_tensor = staging_buf.copy_to_gpu()
        if async_scheduling and device_type == "tpu":
            # pyrefly: ignore[missing-attribute]
            fence = torch.tpu.Event()
            fence.record()
            self._fences[slot_idx] = fence
        return lookback_device_tensor

    def prepare_for_chunk(
        self,
        input_batch: Any,
        chunk_span: tuple[int, int] | None,
        *,
        async_scheduling: bool,
        device_type: str,
        replay_tracker: BoundedReplayTracker | None = None,
    ) -> torch.Tensor:
        """Resolve ``chunk_span`` against ``input_batch.num_reqs`` and stage."""
        start_idx = 0 if chunk_span is None else chunk_span[0]
        chunk_reqs = (
            input_batch.num_reqs
            if chunk_span is None
            else min(chunk_span[1], max(0, input_batch.num_reqs - start_idx))
        )
        return self.prepare(
            input_batch,
            chunk_reqs,
            start_index=start_idx,
            async_scheduling=async_scheduling,
            device_type=device_type,
            replay_tracker=replay_tracker,
        )
