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
"""Parts shared by the schedulers that serve short prompts beside a running
long prefill.

vLLM schedules running requests before waiting ones, and a long prompt in
mid-prefill counts as running, so its chunks fill the step and a short prompt
that arrives behind it waits until that prefill ends. A scheduler built on
ShortPromptsFirstScheduler keeps part of the step's budget for short waiting
prompts and caps the chunks of running prefills at the budget left.

Two schedulers use it:
- ShortPrefillFirstScheduler (TPU_SHORT_PREFILL_FIRST_TOKENS) is the
  standalone one: it promotes short prompts in arrival order and caps every
  running prefill. Use it when nothing else picks the scheduler.
- PrefixAwareScheduler with short prompts first (TPU_PREFIX_AWARE_SHORT_PROMPTS)
  ranks short prompts by their uncached tokens, as its admission does, and
  keeps the oldest long prefill's full chunk. Use it with prefix caching,
  where PrefixAwareScheduler is the scheduler.
"""

from collections.abc import Iterable

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus

# Statuses of a waiting request the waiting loop can admit as a new prefill.
ADMITTABLE_STATUSES = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


def fit_short_prompts[T](
    candidates: Iterable[tuple[T, int]],
    reserved: int,
    free_tokens: int,
    free_slots: int,
) -> tuple[int, list[T]]:
    """Take candidates in order while a sequence slot and tokens are free,
    skipping each one that does not fit.

    A candidate is drawn only while room is left, so a lazy ``candidates``
    looks up no prompt it does not need.

    Args:
        candidates: (prompt, tokens) pairs in the order to take them.
        reserved: tokens already reserved this step.
        free_tokens: tokens the step can reserve in all.
        free_slots: sequence slots free for new prompts.

    Returns:
        The tokens reserved, including ``reserved``, and the prompts taken.
    """
    taken: list[T] = []
    remaining = iter(candidates)
    while len(taken) < free_slots and reserved < free_tokens:
        try:
            prompt, tokens = next(remaining)
        except StopIteration:
            break
        if reserved + tokens <= free_tokens:
            taken.append(prompt)
            reserved += tokens
    return reserved, taken


class ShortPromptsFirstScheduler(AsyncScheduler):
    """The running-prefill cap and the lookups both schedulers share."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._running_prefill_budget: int | None = None
        self._uncached_by_request: dict[str, int] = {}

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        budget = self._running_prefill_budget if self._is_capped(request) else None
        if budget is not None:
            num_new_tokens = min(num_new_tokens, budget)
        num_new_tokens = self._lookahead_within_cap(
            request, num_computed_tokens, num_new_tokens
        )
        if budget is not None:
            self._running_prefill_budget = budget - num_new_tokens
        return num_new_tokens

    def _is_capped(self, request: Request) -> bool:
        """Whether the running-prefill budget caps this request's chunk."""
        return (
            self._running_prefill_budget is not None
            and request.status == RequestStatus.RUNNING
            and request.is_prefill_chunk
        )

    def _lookahead_within_cap(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        return super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )

    def _uncached_once(self, request: Request) -> int:
        """The request's uncached prompt tokens, looked up once per stay in
        the waiting queue."""
        if request.request_id not in self._uncached_by_request:
            self._uncached_by_request[request.request_id] = self._look_up_uncached(
                request
            )
        return self._uncached_by_request[request.request_id]

    def _look_up_uncached(self, request: Request) -> int:
        return request.num_tokens - self._cached_prefix_tokens(request)

    def _cached_prefix_tokens(self, request: Request) -> int:
        """Prompt tokens of a request that has computed none in the prefix
        cache, as its admission would find them."""
        kvm = self.kv_cache_manager
        if not kvm.prefix_cache_lookup_enabled(request):
            return 0
        _, cached, _ = kvm.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return cached

    def _forget_requests_that_left_waiting(self) -> None:
        waiting = {request.request_id for request in self.waiting}
        for request_id in self._uncached_by_request.keys() - waiting:
            del self._uncached_by_request[request_id]

    def _free_sequence_slots(self) -> int:
        return (
            self.max_num_running_reqs
            - len(self.running)
            - self.num_waiting_for_streaming_input
        )

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
