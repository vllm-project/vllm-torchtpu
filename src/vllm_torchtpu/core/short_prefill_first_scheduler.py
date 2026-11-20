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
the same; only the order of prompt chunks changes. PrefixAwareScheduler
serves short prompts the same way with prefix caching; core/short_prompts_first.py
says when to use which.

Enable with:
    TPU_SHORT_PREFILL_FIRST_TOKENS=2048
"""

from collections.abc import Iterator

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import Request

from vllm_torchtpu import envs
from vllm_torchtpu.core.short_prompts_first import (
    ADMITTABLE_STATUSES,
    ShortPromptsFirstScheduler,
    fit_short_prompts,
)

SCHEDULER_CLS = (
    "vllm_torchtpu.core.short_prefill_first_scheduler.ShortPrefillFirstScheduler"
)


class ShortPrefillFirstScheduler(ShortPromptsFirstScheduler):
    """Serve short waiting prompts before running prefill chunks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.short_prompt_tokens = envs.TPU_SHORT_PREFILL_FIRST_TOKENS
        self._skip_promotion = False

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        promoted: list[Request] = []
        if not self._skip_promotion:
            promoted = self._promote_short_waiting()
        reserved = sum(self._uncached_by_request[r.request_id] for r in promoted)
        if reserved:
            self._running_prefill_budget = max(
                self.max_num_scheduled_tokens - self._decode_tokens() - reserved, 0
            )
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self._running_prefill_budget = None
        # A reservation that no promoted request used (none got its KV blocks)
        # would cap the running prefill again next step, and a prefill capped
        # at 0 never reaches allocate_slots, so nothing is preempted or freed:
        # every step would schedule nothing. The next step reserves nothing.
        self._skip_promotion = bool(reserved) and not any(
            r.request_id in output.num_scheduled_tokens for r in promoted
        )
        return output

    def _promote_short_waiting(self) -> list[Request]:
        """Move the short waiting requests that fit this step to the head of
        the waiting queue, in arrival order, and return them."""
        if (
            self.short_prompt_tokens <= 0
            or self.policy != SchedulingPolicy.FCFS
            or self._pause_state != PauseState.UNPAUSED
            or not self.waiting
            or not any(r.is_prefill_chunk for r in self.running)
        ):
            return []
        self._forget_requests_that_left_waiting()
        _, short = fit_short_prompts(
            self._short_waiting(),
            reserved=0,
            free_tokens=self.max_num_scheduled_tokens - self._decode_tokens(),
            free_slots=self._free_sequence_slots(),
        )
        if not short:
            return []
        waiting = self.waiting
        assert isinstance(waiting, FCFSRequestQueue)
        waiting.remove_requests(short)
        for request in reversed(short):
            waiting.prepend_request(request)
        return short

    def _short_waiting(self) -> Iterator[tuple[Request, int]]:
        """The short waiting requests in arrival order, with their uncached
        tokens, each looked up when it is drawn."""
        for request in self.waiting:
            uncached = self._uncached_tokens(request)
            if uncached is not None and uncached <= self.short_prompt_tokens:
                yield request, uncached

    def _uncached_tokens(self, request: Request) -> int | None:
        """Prompt tokens the request still has to prefill, or None when the
        waiting loop would not admit it as a new prefill. Looked up once per
        stay in the waiting queue."""
        if request.status not in ADMITTABLE_STATUSES or request.num_computed_tokens > 0:
            return None
        return self._uncached_once(request)
