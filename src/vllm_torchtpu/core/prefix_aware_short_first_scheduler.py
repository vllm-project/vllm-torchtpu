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
"""Prefix-aware admission with short prompts served ahead of running prefills.

PrefixAwareScheduler orders the waiting queue by uncached prompt tokens, but
vLLM schedules running requests first, and with a long-prefill threshold of
half the step budget two long prefills in progress take the whole step:
nothing waiting is admitted until one of them ends.
ShortPrefillFirstScheduler reserves budget ahead of running prefill chunks
for waiting prompts with few uncached tokens.

PrefixAwareShortFirstScheduler does both. A prompt is short when its
prefix-aware cold count (prompt tokens minus the HBM hit and the offload-tier
chunks right past it) is at most TPU_SHORT_PREFILL_FIRST_TOKENS. The short
prompts that fit the step go to the head of the waiting queue in prefix-aware
order (preempted first, then fewest cold tokens, then arrival), and the
running prefill chunks of the step are capped at the budget they leave. A
prompt back from an offload-tier load waits in skipped_waiting with its
loaded tokens computed; it is short when the tokens left are, and the waiting
loop serves that queue first, so it is reserved for ahead of the waiting
queue. The step budget and every request's tokens stay the same; only the
order of prompt chunks changes.

A long prompt still holds half the step per chunk for as long as it runs, so
two of them leave nothing to the prompts that are neither short nor long.
TPU_PREFILL_MAX_LONG_RUNNING=N (0, the default, is off) limits admission: a
request is long when its uncached tokens left (the cold count for a new one)
exceed TPU_PREFILL_LONG_TOKENS, and while N running prefills are long no
further long request is admitted; the requests behind it are. A held request
stays where it is in its queue. With fewer than N long prefills running, the
best-ranked long requests in the prefix-aware window are admitted as before.
Running requests are never capped or preempted by the limit, and a running
prefill stops counting once its tokens left are at or under the threshold.

With TPU_PREFILL_STEP_FILL=1, a step in which no waiting request can be
admitted (the queues are empty, or what they hold is held by the limit, not
ready, or has no free sequence slot) caps the running prefill chunks at the
step budget instead of the long-prefill threshold, in running order, so a
lone long prefill takes the whole step instead of half of it.

Enable with --scheduler-cls set to SCHEDULER_CLS and
TPU_SHORT_PREFILL_FIRST_TOKENS at half of --max-num-batched-tokens.
"""

from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.core.short_prefill_first_scheduler import (
    ShortPrefillFirstScheduler,
)

logger = init_logger("vllm." + __name__)

SCHEDULER_CLS = (
    "vllm_torchtpu.core.prefix_aware_short_first_scheduler."
    "PrefixAwareShortFirstScheduler"
)

_ADMITTABLE = (RequestStatus.WAITING, RequestStatus.PREEMPTED)

_LOG_HELD_EVERY = 1000


class PrefixAwareShortFirstScheduler(ShortPrefillFirstScheduler, PrefixAwareScheduler):
    """Admit fewest-uncached prompts first and serve the short ones ahead of
    running prefill chunks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._promoted_from_waiting: list[Request] = []
        self.max_long_prefills = envs.TPU_PREFILL_MAX_LONG_RUNNING
        self.long_prefill_tokens = envs.TPU_PREFILL_LONG_TOKENS
        self._held_ids: set[str] = set()
        self._held_steps = 0
        self._held_requests = 0
        self.step_fill = envs.TPU_PREFILL_STEP_FILL
        spec = self.vllm_config.speculative_config
        self._draft_slots = (
            spec.max_num_new_slots_for_drafting if spec is not None else 0
        )
        # (scheduled-token budget, input-token budget) left in a filling step.
        self._fill_left: tuple[int, int] | None = None
        logger.info(
            "PrefixAwareShortFirstScheduler ACTIVE (short prompts: at most %d "
            "uncached tokens; long prefills running at once: %s; step fill: %s)",
            self.short_prompt_tokens,
            (
                f"at most {self.max_long_prefills} over "
                f"{self.long_prefill_tokens} uncached tokens"
                if self.max_long_prefills > 0
                else "no limit"
            ),
            "on" if self.step_fill else "off",
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._promoted_from_waiting = []
        held = self._hold_long_requests()
        self._fill_left = (
            (
                self.max_num_scheduled_tokens,
                self.scheduler_config.max_num_batched_tokens,
            )
            if self.step_fill and not self._admittable_waiting()
            else None
        )
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self._fill_left = None
            self._restore_held_requests(held)
        # A promoted prompt admitted to an async KV load schedules no tokens,
        # but it has left the waiting queue and is not reserved for again
        # until the load ends, so the step-skip that guards against a
        # reservation nobody can use is not needed after it.
        if self._skip_promotion and any(
            r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            for r in self._promoted_from_waiting
        ):
            self._skip_promotion = False
        return output

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        fill = self._fill_left
        if (
            fill is not None
            and request.status == RequestStatus.RUNNING
            and request.is_prefill_chunk
            and not request.has_encoder_inputs
            and not self.need_mamba_block_aligned_split
        ):
            left = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - num_computed_tokens
            )
            room = min(
                fill[0],
                fill[1] - self._draft_slots,
                self.max_model_len
                - num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            num_new_tokens = max(num_new_tokens, min(left, room))
        num_new_tokens = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if fill is not None and num_new_tokens > 0:
            self._fill_left = (
                fill[0] - num_new_tokens,
                fill[1] - num_new_tokens - self._draft_slots,
            )
        return num_new_tokens

    def _admittable_waiting(self) -> bool:
        """Whether the waiting loop could admit a request this step: a free
        sequence slot and a waiting request that is not held and is ready.
        Called after _hold_long_requests has taken the held ones out."""
        free_slots = (
            self.max_num_running_reqs
            - len(self.running)
            - self.num_waiting_for_streaming_input
        )
        if free_slots <= 0 or self._pause_state != PauseState.UNPAUSED:
            return False
        if any(r.status in _ADMITTABLE for r in self.waiting):
            return True
        return any(self._skipped_uncached(r) is not None for r in self.skipped_waiting)

    def _promote_short_waiting(self) -> list[Request]:
        """Return the short prompts the step reserves budget for: the ready
        ones in skipped_waiting, then those of the waiting queue, which move
        to its head in prefix-aware order. ShortPrefillFirstScheduler.schedule
        sums their _uncached_by_request entries."""
        if (
            self.short_prompt_tokens <= 0
            or self.policy != SchedulingPolicy.FCFS
            or self._pause_state != PauseState.UNPAUSED
            or not (self.waiting or self.skipped_waiting)
            or not any(r.is_prefill_chunk for r in self.running)
        ):
            return []
        self._forget_requests_that_left_waiting()
        free_slots = (
            self.max_num_running_reqs
            - len(self.running)
            - self.num_waiting_for_streaming_input
        )
        free_tokens = self.max_num_scheduled_tokens - self._decode_tokens()
        short_skipped, reserved = self._pick_short(
            self.skipped_waiting, self._skipped_uncached, free_slots, free_tokens
        )
        short, _ = self._pick_short(
            self.waiting,
            self._uncached_tokens,
            free_slots - len(short_skipped),
            free_tokens - reserved,
        )
        if short:
            waiting = self.waiting
            assert isinstance(waiting, FCFSRequestQueue)
            waiting.remove_requests(short)
            for request in reversed(short):
                waiting.prepend_request(request)
        self._promoted_from_waiting = short
        return short_skipped + short

    def _pick_short(
        self,
        queue: Iterable[Request],
        uncached_of: Callable[[Request], int | None],
        free_slots: int,
        free_tokens: int,
    ) -> tuple[list[Request], int]:
        """Short requests of ``queue`` that fit the slots and tokens, in
        prefix-aware order, and the tokens they reserve."""
        candidates = []
        for request in queue:
            uncached = uncached_of(request)
            if uncached is not None and uncached <= self.short_prompt_tokens:
                rank = (request.num_preemptions == 0, uncached, request.arrival_time)
                candidates.append((rank, request))
        candidates.sort(key=lambda c: c[0])
        short: list[Request] = []
        reserved = 0
        for (_, uncached, _), request in candidates:
            if len(short) >= free_slots or reserved >= free_tokens:
                break
            if reserved + uncached <= free_tokens:
                short.append(request)
                reserved += uncached
        return short, reserved

    def _skipped_uncached(self, request: Request) -> int | None:
        """Prompt tokens a skipped_waiting request still has to prefill, or
        None when the waiting loop would not admit it this step."""
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            if request.request_id not in self.finished_recving_kv_req_ids:
                return None
        elif request.status not in _ADMITTABLE:
            return None
        if request.num_computed_tokens == 0:
            return self._uncached_tokens(request)
        uncached = max(request.num_tokens - request.num_computed_tokens, 0)
        self._uncached_by_request[request.request_id] = uncached
        return uncached

    def _hold_long_requests(
        self,
    ) -> list[tuple[FCFSRequestQueue, list[Request], list[Request]]]:
        """Take the long requests the step must not admit out of the waiting
        queues and return, per queue, its order before and what was taken.

        A request taken out cannot be admitted; the waiting loop goes on to
        the requests behind it. Running requests are not touched."""
        if self.max_long_prefills <= 0 or self.policy != SchedulingPolicy.FCFS:
            return []
        threshold = self.long_prefill_tokens
        running_long = sum(
            1
            for r in self.running
            if r.is_prefill_chunk and r.num_tokens - r.num_computed_tokens > threshold
        )
        allowed = max(self.max_long_prefills - running_long, 0)
        held_by_queue = []
        # The waiting loop serves skipped_waiting first.
        for queue, uncached_of in (
            (self.skipped_waiting, self._skipped_uncached),
            (self.waiting, self._uncached_tokens),
        ):
            assert isinstance(queue, FCFSRequestQueue)
            order = list(queue)
            longs = []
            for position, request in enumerate(order):
                uncached = uncached_of(request)
                if uncached is not None and uncached > threshold:
                    rank = (
                        request.num_preemptions == 0,
                        uncached,
                        request.arrival_time,
                    )
                    longs.append((position >= self.WINDOW, rank, request))
            if not longs:
                continue
            # The best-ranked long requests in the prefix-aware window go on
            # as without the limit, up to the free long slots.
            longs.sort(key=lambda c: (c[0], c[1]))
            admitted = {id(r) for in_window, _, r in longs[:allowed] if not in_window}
            allowed -= len(admitted)
            held = [r for _, _, r in longs if id(r) not in admitted]
            if held:
                queue.remove_requests(held)
                held_by_queue.append((queue, order, held))
                self._held_ids.update(r.request_id for r in held)
        if held_by_queue:
            self._held_steps += 1
            self._held_requests += sum(len(h) for _, _, h in held_by_queue)
            if self._held_steps % _LOG_HELD_EVERY == 0:
                logger.info(
                    "long-prefill limit: %d steps held %d requests in all "
                    "(this step %d, running long %d)",
                    self._held_steps,
                    self._held_requests,
                    len(self._held_ids),
                    running_long,
                )
        return held_by_queue

    def _restore_held_requests(
        self, held_by_queue: list[tuple[FCFSRequestQueue, list[Request], list[Request]]]
    ) -> None:
        """Put each held request back where it was: before the first request
        that followed it in its queue and is still there."""
        for queue, order, held in held_by_queue:
            position = {id(r): i for i, r in enumerate(order)}
            held_ids = {id(r) for r in held}
            pending = [r for r in order if id(r) in held_ids]
            restored: list[Request] = []
            k = 0
            for request in queue:
                i = position.get(id(request))
                if i is not None:
                    while k < len(pending) and position[id(pending[k])] < i:
                        restored.append(pending[k])
                        k += 1
                restored.append(request)
            restored.extend(pending[k:])
            queue.clear()
            queue.extend(restored)
        self._held_ids.clear()

    def _forget_requests_that_left_waiting(self) -> None:
        queued = {r.request_id for r in self.waiting}
        queued.update(r.request_id for r in self.skipped_waiting)
        queued.update(self._held_ids)
        for request_id in self._uncached_by_request.keys() - queued:
            del self._uncached_by_request[request_id]

    def _look_up_uncached(self, request: Request) -> int:
        return max(request.num_tokens - self._prefix_warmth(request), 0)
