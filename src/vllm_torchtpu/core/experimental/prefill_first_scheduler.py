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
"""Solo whole-prompt prefill scheduler for long prompts.

When ``VLLM_TPU_SOLO_PREFILL_OVER_TOKENS=N > 0`` is set, steps take one of two
shapes:

* **Solo step**: one waiting request with more than ``N`` tokens to compute
  (none computed yet and no encoder inputs), scheduled whole and alone without
  reading the prefix cache while running requests sit that step out. Admitted
  only when free KV blocks (including the watermark while other requests run)
  and ``max_num_seqs`` have room for the whole prompt.
* **Regular step**: schedules at most ``N`` tokens and holds back waiting
  requests from the first solo-eligible prompt onward so shorter prompts do not
  bypass a long prompt waiting in FCFS order.

Prompts longer than a single step's budget (``max_num_scheduled_tokens`` and
``max_num_batched_tokens`` minus draft slots) cannot run solo and fall back to
regular chunked prefill.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_CLS = (
    "vllm_torchtpu.core.experimental.prefill_first_scheduler.PrefillFirstScheduler"
)


class PrefillFirstScheduler(AsyncScheduler):
    """Schedules prompts exceeding ``VLLM_TPU_SOLO_PREFILL_OVER_TOKENS`` in
    dedicated whole-prompt solo steps while capping regular steps to that
    threshold.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.solo_over_tokens = envs.VLLM_TPU_SOLO_PREFILL_OVER_TOKENS
        if self.solo_over_tokens <= 0:
            raise ValueError(
                "PrefillFirstScheduler requires VLLM_TPU_SOLO_PREFILL_OVER_TOKENS > 0."
            )
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        self.max_solo_tokens = min(
            self.max_num_scheduled_tokens,
            self.scheduler_config.max_num_batched_tokens - draft_slots,
        )
        self._check_solo_config()
        self._skip_solo = False
        logger.info(
            "PrefillFirstScheduler active (solo_over_tokens=%d, max_solo_tokens=%d)",
            self.solo_over_tokens,
            self.max_solo_tokens,
        )

    def _check_solo_config(self) -> None:
        if self.solo_over_tokens >= self.max_solo_tokens:
            raise ValueError(
                f"VLLM_TPU_SOLO_PREFILL_OVER_TOKENS ({self.solo_over_tokens}) must "
                f"be below the step budget ({self.max_solo_tokens}); raise "
                "--max-num-batched-tokens to at least the longest solo prompt."
            )
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < self.max_solo_tokens:
            raise ValueError(
                f"long_prefill_token_threshold ({threshold}) would split solo "
                f"prompts of up to {self.max_solo_tokens} tokens; set "
                "--long-prefill-token-threshold 0 or at least the step budget."
            )
        if self.connector is not None or self.need_mamba_block_aligned_split:
            raise ValueError(
                "Solo prefill supports neither KV connectors nor Mamba "
                "block-aligned splits."
            )
        if self.max_model_len > self.max_solo_tokens:
            logger.warning(
                "PrefillFirstScheduler: requests with more than %d tokens are "
                "prefilled in regular chunks of at most %d tokens.",
                self.max_solo_tokens,
                self.solo_over_tokens,
            )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        request = None
        if not throttle_prefills and not self._skip_solo:
            request = self._solo_candidate()
        self._skip_solo = False
        if request is not None:
            return self._schedule_solo(request)
        return self._schedule_regular(throttle_prefills)

    def _takes_solo_step(self, request: Request) -> bool:
        return (
            request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            and request.num_computed_tokens == 0
            and self.solo_over_tokens < request.num_tokens <= self.max_solo_tokens
            and not request.has_encoder_inputs
            and not (request.num_stale_output_tokens and not request.drop_stale_output)
        )

    def _solo_candidate(self) -> Request | None:
        if self._pause_state != PauseState.UNPAUSED or any(
            not self._is_blocked_waiting_status(r.status)
            for r in self.kv_holding_waiting
        ):
            return None
        num_active = len(self.running) + self.num_waiting_for_streaming_input
        if num_active >= min(self.max_num_active_reqs, self.max_num_running_reqs):
            return None
        request = next((r for r in self.waiting if self._takes_solo_step(r)), None)
        if request is None or not self._fits(request, request.num_tokens):
            return None
        return request

    def _fits(self, request: Request, num_tokens: int) -> bool:
        needed = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(num_tokens + self.num_lookahead_tokens, self.max_model_len),
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )
        if self.running:
            needed += self.kv_cache_manager.watermark_blocks
        reserved = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
        )
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return free - reserved >= needed

    def _schedule_solo(self, request: Request) -> SchedulerOutput:
        others = [r for r in self.waiting if r is not request]
        skip_reading_prefix_cache = request.skip_reading_prefix_cache
        request.skip_reading_prefix_cache = True
        try:
            with self._hidden_waiting(others), self._hidden_running(self.running[:]):
                output = super().schedule()
        finally:
            request.skip_reading_prefix_cache = skip_reading_prefix_cache
        scheduled = output.num_scheduled_tokens
        if not scheduled:
            logger.warning(
                "PrefillFirstScheduler: solo request %s was not admitted; "
                "falling back to a regular step next.",
                request.request_id,
            )
            self._skip_solo = True
        elif scheduled != {request.request_id: request.num_tokens}:
            raise RuntimeError(
                f"PrefillFirstScheduler scheduled {scheduled} for solo request "
                f"{request.request_id} of {request.num_tokens} tokens; a solo "
                "step must hold that whole prompt alone."
            )
        return output

    def _schedule_regular(self, throttle_prefills: bool) -> SchedulerOutput:
        held = self._held_from_regular()
        budget = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = min(budget, self.solo_over_tokens)
        try:
            with self._hidden_waiting(held):
                return super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = budget

    def _held_from_regular(self) -> list[Request]:
        waiting = list(self.waiting)
        for i, r in enumerate(waiting):
            if self._takes_solo_step(r):
                return waiting[i:]
        return []

    @contextlib.contextmanager
    def _hidden_waiting(self, hidden: list[Request]) -> Iterator[None]:
        if not hidden:
            yield
            return
        position = {r.request_id: i for i, r in enumerate(self.waiting)}
        self.waiting.remove_requests(hidden)
        try:
            yield
        finally:
            current = list(self.waiting)
            ahead = [r for r in current if r.request_id not in position]
            rest = sorted(
                [r for r in current if r.request_id in position] + hidden,
                key=lambda r: position[r.request_id],
            )
            self.waiting.remove_requests(current)
            for r in ahead + rest:
                self.waiting.add_request(r)

    @contextlib.contextmanager
    def _hidden_running(self, hidden: list[Request]) -> Iterator[None]:
        if not hidden:
            yield
            return
        order = self.running[:]
        hidden_ids = {r.request_id for r in hidden}
        self.running[:] = [r for r in order if r.request_id not in hidden_ids]
        self.max_num_running_reqs -= len(hidden)
        self.max_num_active_reqs -= len(hidden)
        try:
            yield
        finally:
            self.max_num_running_reqs += len(hidden)
            self.max_num_active_reqs += len(hidden)
            visible_ids = {r.request_id for r in self.running}
            order_ids = {r.request_id for r in order}
            self.running[:] = [
                r
                for r in order
                if r.request_id in hidden_ids or r.request_id in visible_ids
            ] + [r for r in self.running if r.request_id not in order_ids]

    def _make_cached_request_data(self, *args, **kwargs) -> CachedRequestData:
        data = super()._make_cached_request_data(*args, **kwargs)
        if self.connector is not None:
            return data
        # A running request that sat out a solo step is marked resumed on the
        # next regular step; GPUModelRunner._update_states only consumes the
        # trailing num_output_tokens from all_token_ids.
        for req_id, num_output_tokens in zip(data.req_ids, data.num_output_tokens):
            token_ids = data.all_token_ids.get(req_id)
            if token_ids is not None:
                data.all_token_ids[req_id] = token_ids[
                    len(token_ids) - num_output_tokens :
                ]
        return data
