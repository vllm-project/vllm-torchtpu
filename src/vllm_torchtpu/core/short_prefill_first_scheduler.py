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
"""Short-prompt-first scheduling beside a running long prefill.

vLLM schedules running requests before waiting ones, and a long prompt in
mid-prefill counts as running. Each of its chunks takes the whole token
budget, so a short prompt that arrives behind it gets no tokens until that
prefill ends.

ShortPrefillFirstScheduler moves waiting requests with at most
TPU_SHORT_PREFILL_FIRST_TOKENS uncached tokens to the head of the waiting
queue, in arrival order, and caps the running prefill chunks of the step at
the budget those requests leave. The budget and every request's tokens stay
the same; only the order of prompt chunks changes.

Enable with:
    TPU_SHORT_PREFILL_FIRST_TOKENS=2048
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs

SCHEDULER_CLS = (
    "vllm_torchtpu.core.short_prefill_first_scheduler.ShortPrefillFirstScheduler"
)

_ADMITTABLE = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


class ShortPrefillFirstScheduler(AsyncScheduler):
    """Serve short waiting prompts before running prefill chunks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.short_prompt_tokens = envs.TPU_SHORT_PREFILL_FIRST_TOKENS
        self._running_prefill_budget: int | None = None
        self._uncached_by_request: dict[str, int | None] = {}

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        reserved = self._promote_short_waiting()
        if reserved:
            self._running_prefill_budget = max(
                self.max_num_scheduled_tokens - self._decode_tokens() - reserved, 0
            )
        try:
            return super().schedule(throttle_prefills)
        finally:
            self._running_prefill_budget = None

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if (
            self._running_prefill_budget is None
            or request.status != RequestStatus.RUNNING
            or not request.is_prefill_chunk
        ):
            return super()._reserve_prefill_lookahead(
                request, num_computed_tokens, num_new_tokens
            )
        num_new_tokens = super()._reserve_prefill_lookahead(
            request,
            num_computed_tokens,
            min(num_new_tokens, self._running_prefill_budget),
        )
        self._running_prefill_budget -= num_new_tokens
        return num_new_tokens

    def _promote_short_waiting(self) -> int:
        """Move the short waiting requests that fit this step to the head of
        the waiting queue, in arrival order, and return their uncached tokens."""
        if (
            self.short_prompt_tokens <= 0
            or self.policy != SchedulingPolicy.FCFS
            or self._pause_state != PauseState.UNPAUSED
            or not self.waiting
            or not any(r.is_prefill_chunk for r in self.running)
        ):
            return 0
        self._forget_requests_that_left_waiting()
        free_slots = (
            self.max_num_running_reqs
            - len(self.running)
            - self.num_waiting_for_streaming_input
        )
        free_tokens = self.max_num_scheduled_tokens - self._decode_tokens()
        short: list[Request] = []
        reserved = 0
        for request in self.waiting:
            if len(short) >= free_slots or reserved >= free_tokens:
                break
            uncached = self._uncached_tokens(request)
            if (
                uncached is not None
                and uncached <= self.short_prompt_tokens
                and reserved + uncached <= free_tokens
            ):
                short.append(request)
                reserved += uncached
        if not short:
            return 0
        waiting = self.waiting
        assert isinstance(waiting, FCFSRequestQueue)
        waiting.remove_requests(short)
        for request in reversed(short):
            waiting.prepend_request(request)
        return reserved

    def _forget_requests_that_left_waiting(self) -> None:
        waiting = {request.request_id for request in self.waiting}
        for request_id in self._uncached_by_request.keys() - waiting:
            del self._uncached_by_request[request_id]

    def _uncached_tokens(self, request: Request) -> int | None:
        """Prompt tokens the request still has to prefill, or None when the
        waiting loop would not admit it as a new prefill. Looked up once per
        stay in the waiting queue."""
        if request.status not in _ADMITTABLE or request.num_computed_tokens > 0:
            return None
        if request.request_id not in self._uncached_by_request:
            self._uncached_by_request[request.request_id] = self._look_up_uncached(
                request
            )
        return self._uncached_by_request[request.request_id]

    def _look_up_uncached(self, request: Request) -> int:
        kvm = self.kv_cache_manager
        if not kvm.prefix_cache_lookup_enabled(request):
            return request.num_tokens
        _, cached, _ = kvm.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return request.num_tokens - cached

    def _decode_tokens(self) -> int:
        return sum(
            max(
                r.num_tokens_with_spec
                + r.num_output_placeholders
                - r.num_computed_tokens,
                0,
            )
            for r in self.running
            if not r.is_prefill_chunk
        )
