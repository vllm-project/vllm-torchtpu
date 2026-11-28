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
"""Per-request bounded-replay start tracking and host schedule snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_torchtpu.layers.core.attention_metadata import HostScheduleSnapshot

if TYPE_CHECKING:
    import numpy as np
    from vllm.v1.core.sched.output import SchedulerOutput


def prefix_replay_tokens(kv_cache_config: Any) -> int:
    """Return the maximum ``prefix_replay_tokens`` across KV cache groups."""
    if not isinstance(kv_cache_config, KVCacheConfig):
        return 0
    return max(
        (
            group.kv_cache_spec.prefix_replay_tokens
            for group in kv_cache_config.kv_cache_groups
        ),
        default=0,
    )


class BoundedReplayTracker:
    """Tracks per-request SWA bounded-replay start positions by request ID.

    ``replay_starts`` maps ``req_id -> token position`` (0-based) where bounded
    replay begins, mirroring upstream ``NewRequestData.replay_start`` on warm
    admission and ``num_computed_tokens`` on preemption resume.
    """

    def __init__(self) -> None:
        # req_id -> 0-based token position where bounded replay starts.
        self.replay_starts: dict[str, int] = {}

    def __bool__(self) -> bool:
        return bool(self.replay_starts)

    def get(self, req_id: str, default: int = 0) -> int:
        return self.replay_starts.get(req_id, default)

    def discard(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self.replay_starts.pop(req_id, None)

    def update(
        self,
        scheduler_output: SchedulerOutput,
        kv_cache_config: Any,
    ) -> None:
        if not prefix_replay_tokens(kv_cache_config):
            return
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.replay_start > 0:
                self.replay_starts[new_req.req_id] = new_req.replay_start
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            if req_id in cached_reqs.resumed_req_ids:
                computed = cached_reqs.num_computed_tokens[idx]
                if computed > 0:
                    self.replay_starts[req_id] = computed

    def build_replay_starts(
        self,
        req_ids: Sequence[str],
        start_index: int,
        num_reqs: int,
        target_num_reqs: int,
        kv_cache_config: Any,
    ) -> tuple[int, ...]:
        if not prefix_replay_tokens(kv_cache_config):
            return (0,) * target_num_reqs
        active = tuple(
            self.replay_starts.get(req_id, 0)
            for req_id in req_ids[start_index : start_index + num_reqs]
        )
        return active + (0,) * (target_num_reqs - num_reqs)


def build_host_schedule_snapshot(
    *,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    num_prompt_tokens: np.ndarray,
    req_ids: Sequence[str],
    start_index: int,
    num_reqs: int,
    target_num_reqs: int,
    replay_tracker: BoundedReplayTracker | None,
    kv_cache_config: Any,
) -> HostScheduleSnapshot:
    """Build a host schedule snapshot for a ``SequenceLayoutKind.ALL`` chunk."""
    pad_zeros = (0,) * (target_num_reqs - num_reqs)
    replay_starts = (
        replay_tracker.build_replay_starts(
            req_ids, start_index, num_reqs, target_num_reqs, kv_cache_config
        )
        if replay_tracker is not None
        else (0,) * target_num_reqs
    )
    return HostScheduleSnapshot(
        query_start_loc=tuple(
            int(v) for v in query_start_loc_np[: target_num_reqs + 1]
        ),
        seq_lens=tuple(int(v) for v in seq_lens_np[:num_reqs]) + pad_zeros,
        prompt_lens=(
            tuple(
                int(v) for v in num_prompt_tokens[start_index : start_index + num_reqs]
            )
            + pad_zeros
        ),
        replay_starts=replay_starts,
    )
