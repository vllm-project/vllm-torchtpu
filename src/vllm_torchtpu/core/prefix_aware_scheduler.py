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
"""Prefix-aware admission scheduler for prefix-caching workloads.

Standard vLLM admits ready requests in FCFS order, which is oblivious to
whether a request's prompt prefix is already cached in HBM. In multi-turn
agentic workloads, admitting a long uncached prompt at queue[0] blocks ready
follow-up requests whose prefixes are already cached, and under high HBM
utilization can evict those cached prefixes before they are reused.

PrefixAwareScheduler inspects the ready queue within a lookahead window and
prioritizes requests with the fewest uncached prompt tokens. It modifies only
the admission ordering; memory management, eviction, and preemption policies
remain untouched.

vLLM looks up the prefix cache only when it admits a request. Two requests
with the same long prompt admitted a step apart therefore compute the same
chunks side by side: each step the first one caches its chunk, and the second,
which may not read a block hashed in the same step, computes the same
positions again. With VLLM_TPU_FOLLOW_SHARED_PREFIX set, a
running prefill whose whole next chunk is already in the prefix cache gets no
tokens that step and its budget goes to others. Once the cached prefix stops
growing during a step, the request is preempted and admitted again, so it
takes the cached tokens through the normal admission path, which skips blocks
hashed in the same step. A cached block holds the same KV the request would
compute, so every token stays the same.

A long prompt in mid-prefill counts as running, and vLLM schedules running
requests before waiting ones, so its chunks fill the step and a short waiting
prompt gets no tokens until that prefill ends. With
TPU_PREFIX_AWARE_SHORT_PROMPTS=first, each step first reserves budget for
short waiting prompts (in the same fewest-uncached-first order, while they
fit the step and a free sequence slot) and for running prefills of such
prompts, and caps the chunks of running longer prefills at the budget left.
The step budget, the chunk limit and every request's tokens stay the same;
only the order of prompt chunks changes.

A prompt is short when its uncached tokens fit in step // chunk full steps
(_short_prompt_limit says what this bounds and how it was set). The oldest
running long prefill keeps its full chunk, so a lone long prompt is never
slowed down and short prompts take tokens only from later long prefills;
capping every long prefill keeps long prompts in flight longer, and short
prompts then meet more of them. ShortPrefillFirstScheduler is the standalone
form of this (core/short_prompts_first.py says when to use which).

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
    VLLM_TPU_FOLLOW_SHARED_PREFIX=1 (optional)
    TPU_PREFIX_AWARE_SHORT_PROMPTS=first (optional)
"""

import enum
import time
from dataclasses import dataclass
from itertools import islice
from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.block_pool_marks import BLOCKS_HASHED_THIS_STEP
from vllm_torchtpu.core.short_prompts_first import (
    ADMITTABLE_STATUSES,
    ShortPromptsFirstScheduler,
    fit_short_prompts,
)

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


class ShortPrompts(enum.StrEnum):
    """TPU_PREFIX_AWARE_SHORT_PROMPTS: how PrefixAwareScheduler serves short
    prompts beside running long prefills."""

    OFF = "off"
    FIRST = "first"


@dataclass
class _Follower:
    """A running prefill that waits while another request caches its chunks."""

    cached_at_step_start: int | None = None
    cache_grew: bool = True


@dataclass
class _Rejoin:
    """A follower preempted to take its cached prefix at admission."""

    computed_tokens: int
    ahead: frozenset[str]
    hit_tokens: int = 0
    followed_steps: int = 0


class PrefixAwareScheduler(ShortPromptsFirstScheduler):
    """Admit ready requests prioritizing fewest uncached prompt tokens."""

    #: Lookahead window of ready requests to evaluate for cached prefix length.
    WINDOW = 64

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0

        self.follow_shared_prefix = envs.VLLM_TPU_FOLLOW_SHARED_PREFIX
        self._followers: dict[str, _Follower] = {}
        self._followed_steps: dict[str, int] = {}
        self._rejoins: dict[str, _Rejoin] = {}
        self._num_followed_steps = 0
        self._num_rejoins = 0
        self._num_followed_tokens = 0

        short_prompts = ShortPrompts(envs.TPU_PREFIX_AWARE_SHORT_PROMPTS)
        self.short_prompt_tokens = (
            None if short_prompts == ShortPrompts.OFF else self._short_prompt_limit()
        )
        self._uncached_admitted: dict[str, int] = {}

        if (
            self.policy != SchedulingPolicy.FCFS
            and self.short_prompt_tokens is not None
        ):
            raise NotImplementedError(
                "Short prompts first needs the FCFS scheduling policy; use "
                "--scheduling-policy fcfs or TPU_PREFIX_AWARE_SHORT_PROMPTS=off."
            )

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, follow shared prefix: %s, "
            "short prompts first: %s)",
            self.WINDOW,
            "on" if self.follow_shared_prefix else "off",
            "off"
            if self.short_prompt_tokens is None
            else f"at most {self.short_prompt_tokens} uncached tokens",
        )

    def _short_prompt_limit(self) -> int:
        """Uncached tokens up to which a waiting prompt is short: step // chunk
        full steps of tokens (8,192 at a 4,096-token step and 2,048-token
        chunks).

        A short prompt computes one chunk per step like any prompt, and in
        each of its steps it takes up to one chunk from the long prefills
        after the oldest. One short prompt holds them back by at most
        (step // chunk) ** 2 steps; short prompts that keep arriving can hold
        them back longer. The limit trades that delay against the short
        prompts' first-token time. It scales with the step and chunk sizes;
        the factor is tuned, on an agent workload of 1k to over 100k-token
        prompts at 32 concurrent requests, where 8,192 tokens gave a lower
        TTFT p90 than 4,096 or 16,384. Re-tune it when the prompt-length mix
        changes. The oldest long prefill keeps its chunk and short prompts
        take tokens from the others, so a step must hold 2 chunks or more.
        """
        step = self.max_num_scheduled_tokens
        chunk = self.scheduler_config.long_prefill_token_threshold
        if not 0 < chunk <= step // 2:
            raise NotImplementedError(
                "TPU_PREFIX_AWARE_SHORT_PROMPTS needs a step of 2 chunks or "
                "more; set --long-prefill-token-threshold between 1 and half "
                f"of the {step}-token step."
            )
        return step // chunk * step

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._forget_blocks_hashed_last_step()
        if self.follow_shared_prefix:
            self._update_followers()
        if self.short_prompt_tokens is not None:
            output = self._schedule_short_prompts_first(throttle_prefills)
            self._note_admitted_prompts(output)
        else:
            output = super().schedule(throttle_prefills)
        if self.follow_shared_prefix:
            self._restore_rejoined_places()
        return output

    def _forget_blocks_hashed_last_step(self) -> None:
        """Let this step's lookups hit the blocks the last step hashed.

        The forward of the last step writes them before this step's forward
        runs. Scheduler.schedule forgets them the same way, but the
        short-prompt lookups run before it and would count them as misses.
        """
        pool_state = self.kv_cache_manager.block_pool.__dict__
        hashed_last_step = pool_state.get(BLOCKS_HASHED_THIS_STEP)
        if hashed_last_step:
            hashed_last_step.clear()

    def _schedule_short_prompts_first(self, throttle_prefills: bool) -> SchedulerOutput:
        reserved = self._reserve_for_short_prompts()
        if reserved:
            self._running_prefill_budget = max(
                self.max_num_scheduled_tokens
                - self._decode_tokens()
                - reserved
                - self._full_chunk_tokens(),
                0,
            )
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self._running_prefill_budget = None
        return output

    def _lookahead_within_cap(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        num_new_tokens = super()._lookahead_within_cap(
            request, num_computed_tokens, num_new_tokens
        )
        if (
            self.follow_shared_prefix
            and num_new_tokens > 0
            and self._chunk_is_cached(request, num_computed_tokens + num_new_tokens)
        ):
            return 0
        return num_new_tokens

    def _is_capped(self, request: Request) -> bool:
        """Whether short prompts cap this running long prefill's chunk."""
        return (
            super()._is_capped(request)
            and not self._is_short_prompt(request)
            and not self._keeps_full_chunk(request)
        )

    def _keeps_full_chunk(self, request: Request) -> bool:
        longs = (
            r
            for r in self.running
            if r.is_prefill_chunk and not self._is_short_prompt(r)
        )
        return request is next(longs, None)

    def _full_chunk_tokens(self) -> int:
        return sum(
            self._chunk(r.num_tokens - r.num_computed_tokens)
            for r in self.running
            if r.is_prefill_chunk
            and not self._is_short_prompt(r)
            and self._keeps_full_chunk(r)
        )

    def _reserve_for_short_prompts(self) -> int:
        """Tokens this step keeps for short prompts' chunks while a longer
        prefill is running, or 0 when no running prefill is longer. The oldest
        running long prefill keeps its full chunk, so short prompts take budget
        only from the long prefills admitted after it."""
        if self._pause_state != PauseState.UNPAUSED:
            return 0
        prefills = [r for r in self.running if r.is_prefill_chunk]
        if all(self._is_short_prompt(r) for r in prefills):
            return 0
        free_tokens = (
            self.max_num_scheduled_tokens
            - self._decode_tokens()
            - self._full_chunk_tokens()
        )
        if free_tokens <= 0:
            return 0
        reserved = sum(
            self._chunk(r.num_tokens - r.num_computed_tokens)
            for r in prefills
            if self._is_short_prompt(r)
        )
        reserved, _ = fit_short_prompts(
            ((None, chunk) for chunk in self._short_waiting_chunks()),
            reserved,
            free_tokens,
            self._free_sequence_slots(),
        )
        return min(reserved, free_tokens)

    def _short_waiting_chunks(self) -> list[int]:
        """First-step chunks of the short waiting prompts, in admission order."""
        limit = self.short_prompt_tokens
        assert limit is not None
        ranked = []
        for request in islice(self.waiting, self.WINDOW):
            uncached = self._uncached_waiting_tokens(request)
            if uncached is not None and 0 < uncached <= limit:
                ranked.append((self._rank_key(request, uncached), uncached))
        ranked.sort(key=lambda item: item[0])
        return [self._chunk(uncached) for _, uncached in ranked]

    def _uncached_waiting_tokens(self, request: Request) -> int | None:
        """Prompt tokens a waiting request still has to prefill, or None when
        the waiting loop would not schedule it now. A first lookup is kept for
        the request's stay in the waiting queue."""
        if request.status not in ADMITTABLE_STATUSES:
            return None
        if request.num_computed_tokens > 0:
            return request.num_tokens - request.num_computed_tokens
        return self._uncached_once(request)

    def _is_short_prompt(self, request: Request) -> bool:
        """Whether this running prefill had at most short_prompt_tokens
        uncached tokens when it was first admitted."""
        uncached = self._uncached_admitted.get(request.request_id)
        assert uncached is not None, f"{request.request_id} was never admitted"
        assert self.short_prompt_tokens is not None
        return uncached <= self.short_prompt_tokens

    def _chunk(self, tokens: int) -> int:
        return min(tokens, self.scheduler_config.long_prefill_token_threshold)

    def _note_admitted_prompts(self, output: SchedulerOutput) -> None:
        """Record each newly admitted prompt's uncached tokens. A preempted
        request keeps its entry, so it is resumed (with the V2 model runner,
        among the new requests) as short or long as it was admitted."""
        for new in output.scheduled_new_reqs:
            request = self.requests[new.req_id]
            self._uncached_admitted.setdefault(
                new.req_id, request.num_prompt_tokens - new.num_computed_tokens
            )
            self._uncached_by_request.pop(new.req_id, None)
        self._forget_requests_that_left()

    def _forget_requests_that_left(self) -> None:
        for request_id in self._uncached_admitted.keys() - self.requests.keys():
            del self._uncached_admitted[request_id]
        self._forget_requests_that_left_waiting()

    def _get_local_prefix_cache_hit(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        blocks, num_local, boundary, diverged = super()._get_local_prefix_cache_hit(
            request
        )
        rejoin = self._rejoins.get(request.request_id)
        if rejoin is not None:
            rejoin.hit_tokens = num_local
        return blocks, num_local, boundary, diverged

    def _cached_prompt_tokens(self, request: Request) -> int:
        """Prompt tokens of ``request`` in the prefix cache, counting the
        blocks hashed in this step, whose KV a step in flight writes."""
        pool_state = self.kv_cache_manager.block_pool.__dict__
        hashed_this_step = pool_state.pop(BLOCKS_HASHED_THIS_STEP, None)
        try:
            _, num_tokens, _ = self.kv_cache_manager.coordinator.find_longest_cache_hit(
                request.block_hashes, max(request.num_tokens - 1, 0)
            )
        finally:
            if hashed_this_step is not None:
                pool_state[BLOCKS_HASHED_THIS_STEP] = hashed_this_step
        return int(num_tokens)

    def _chunk_is_cached(self, request: Request, chunk_end: int) -> bool:
        """Whether every token of this running prefill's next chunk is cached.

        Such a request waits this step as a follower, and records whether the
        cached prefix grew since the step started, that is whether another
        request is still caching the same prompt this step.
        """
        request_id = request.request_id
        cached = (
            self._cached_prompt_tokens(request)
            if self._is_prompt_chunk(request, chunk_end)
            else 0
        )
        if cached < chunk_end:
            self._followers.pop(request_id, None)
            self._followed_steps.pop(request_id, None)
            return False
        follower = self._followers.setdefault(request_id, _Follower())
        if follower.cached_at_step_start is not None:
            follower.cache_grew = cached > follower.cached_at_step_start
        self._followed_steps[request_id] = self._followed_steps.get(request_id, 0) + 1
        self._num_followed_steps += 1
        return True

    def _is_prompt_chunk(self, request: Request, chunk_end: int) -> bool:
        return (
            request.status == RequestStatus.RUNNING
            and request.num_output_tokens == 0
            and request.num_output_placeholders == 0
            and chunk_end < request.num_tokens
            and self.kv_cache_manager.prefix_cache_lookup_enabled(request)
        )

    def _update_followers(self) -> None:
        """Preempt each follower whose cached prefix stopped growing in the
        last step, so its admission takes the cached tokens."""
        for request_id, rejoin in list(self._rejoins.items()):
            request = self.requests.get(request_id)
            if request is None or request.status not in (
                RequestStatus.PREEMPTED,
                RequestStatus.WAITING_FOR_REMOTE_KVS,
            ):
                del self._rejoins[request_id]
                if request is not None:
                    self._log_rejoined(request, rejoin)

        if self._pause_state != PauseState.UNPAUSED:
            return
        now = time.monotonic()
        for request_id, follower in list(self._followers.items()):
            request = self.requests.get(request_id)
            if request is None or not self._is_prompt_chunk(
                request, request.num_computed_tokens
            ):
                del self._followers[request_id]
                self._followed_steps.pop(request_id, None)
                continue
            cached = self._cached_prompt_tokens(request)
            if follower.cache_grew:
                follower.cached_at_step_start = cached
                follower.cache_grew = False
                continue
            del self._followers[request_id]
            followed_steps = self._followed_steps.pop(request_id, 0)
            computed_tokens = request.num_computed_tokens
            if cached <= computed_tokens:
                continue
            place = self.running.index(request)
            ahead = frozenset(r.request_id for r in self.running[:place])
            del self.running[place]
            self._preempt_request(
                request, now, drop_stale_output=self.requires_kv_delivery
            )
            self._rejoins[request_id] = _Rejoin(
                computed_tokens=computed_tokens,
                ahead=ahead,
                followed_steps=followed_steps,
            )

    def _restore_rejoined_places(self) -> None:
        """Put each follower admitted again back behind the running requests
        that were ahead of it, so rejoining does not cost it its turn."""
        for rejoin_id, rejoin in self._rejoins.items():
            request = self.requests.get(rejoin_id)
            if request is None or request.status != RequestStatus.RUNNING:
                continue
            self.running.remove(request)
            place = max(
                (
                    i + 1
                    for i, r in enumerate(self.running)
                    if r.request_id in rejoin.ahead
                ),
                default=0,
            )
            self.running.insert(place, request)

    def _log_rejoined(self, request: Request, rejoin: _Rejoin) -> None:
        if request.status != RequestStatus.RUNNING:
            return
        taken = max(rejoin.hit_tokens - rejoin.computed_tokens, 0)
        self._num_rejoins += 1
        self._num_followed_tokens += taken
        logger.info(
            "follow shared prefix: %s took %d cached tokens past its own %d "
            "after %d followed steps (total: %d rejoins, %d followed steps, "
            "%d cached tokens taken)",
            request.request_id,
            taken,
            rejoin.computed_tokens,
            rejoin.followed_steps,
            self._num_rejoins,
            self._num_followed_steps,
            self._num_followed_tokens,
        )

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of ``request`` resident in the KV cache: those it has
        computed, or else its cached prefix."""
        if not self.kv_cache_manager.prefix_cache_lookup_enabled(request):
            return 0
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens
        return self._cached_prefix_tokens(request)

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth). Lower sort_key is admitted first."""
        warmth = self._prefix_warmth(request)
        return self._rank_key(request, request.num_tokens - warmth), warmth

    @staticmethod
    def _rank_key(request: Request, uncached: float) -> tuple[int, float, float]:
        """Admission order: preempted requests first, then fewest uncached
        tokens, then arrival."""
        tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        return (tier, float(uncached), request.arrival_time)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        queue = super()._select_waiting_queue_for_scheduling()
        if queue is not None:
            self._promote_warmest(queue)
        return queue

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
        if self.policy != SchedulingPolicy.FCFS or len(queue) < 2:
            return

        window = list(islice(queue, self.WINDOW))
        head = window[0]
        scored = [(self._rank_and_warmth(r), r) for r in window]
        (best_rank, best_warmth), best = min(scored, key=lambda t: t[0][0])

        if best is not head and best_rank < scored[0][0][0]:
            # Only FCFS reaches here (see the early return above).
            cast(FCFSRequestQueue, queue).remove(best)
            queue.prepend_request(best)

        self._paw_n += 1
        self._paw_cold_tokens += max(best.num_tokens - best_warmth, 0)
        self._paw_prompt_tokens += best.num_tokens

        if self._paw_n % _LOG_EVERY == 0:
            logger.info(
                "warmest-first: n=%d mean_cold_tokens=%.0f "
                "cold_frac=%.3f queue_depth=%d",
                self._paw_n,
                self._paw_cold_tokens / self._paw_n,
                self._paw_cold_tokens / max(self._paw_prompt_tokens, 1),
                len(queue),
            )
