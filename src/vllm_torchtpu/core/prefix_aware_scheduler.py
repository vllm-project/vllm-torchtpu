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

With a CPU offload tier behind the HBM prefix cache (an offloading KV
connector), a session the HBM cache evicted is still cached: its prompt loads
from the tier asynchronously and needs little prefill compute. Ranked on HBM
residency alone it looks fully cold, so under a deep queue it waits behind
every warm request (an AgentX GLM-5.2 P/D replay measured ~33 s mean, ~75 s at
>= 15 queued, versus 1.8-5.8 s for warm requests). TPU_PREFIX_AWARE_TIER_WARMTH=1
also counts the whole chunks the tier holds right past the HBM hit as warm, so
such a request is ranked by the compute it actually needs.

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

A waiting prompt whose first uncached block is a block a running prefill has
not cached yet would compute that shared prefix a second time. With
TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX set, such a prompt, short or long, waits
until the running prefill has cached the block, and with short prompts first
that prefill keeps its full chunk.

With TPU_PREFIX_AWARE_SHORTS_TAKE_OLDEST_CHUNK set, the oldest running long
prefill keeps its full chunk only while no short prompt waits or is in
prefill. Short prompts that run one after another behind each other then
share the step instead of each getting only the tokens the oldest long
prefill leaves: short prompts' chunks first, then every running long prefill
in admission order. A prefill a held prompt waits for still keeps its full
chunk. With no short prompt in the system, a lone long prompt is not slowed
down.

With TPU_PREFIX_AWARE_SHORT_PROMPTS=first-short-step, a step that starts a
short waiting prompt beside a running long prefill holds fewer tokens: short
prompts' chunks first, then running long prefills in admission order; in such
a step neither the oldest long prefill nor a prefill a held prompt waits for
keeps its full chunk, so a lone long prompt is slowed down too. A step's time
grows with its tokens, so the smaller step reaches the short prompt's first
token sooner. Its size is the smallest compiled step size
(compilation_config.compile_sizes) that holds one full chunk
(long_prefill_token_threshold) and is smaller than the full step, so a short
prompt's first chunk fits and the step pads to no larger size. The chunk limit
and every request's tokens stay the same; only the order and grouping of
prompt chunks change.

vLLM's admission counts as cached both the prompt's prefix in HBM and the
chunks of a KV connector's tier that follow it, which it loads instead of
computing. The waiting queue counts only the HBM prefix, so a prompt whose
earlier turns left HBM for the tier ranks as uncached and waits behind every
shorter prompt, though admission finds most of it cached. With
TPU_PREFIX_AWARE_RANK_WITH_TIER set, the queue's count (the order of
admission and which prompts are short) also takes the tier's chunks after the
HBM prefix, read once per stay in the queue without changing the tier. Only
the order of admission changes; every request computes the same tokens.

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
    VLLM_TPU_FOLLOW_SHARED_PREFIX=1 (optional)
    TPU_PREFIX_AWARE_SHORT_PROMPTS=first or first-short-step (optional)
    TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX=1 (optional)
    TPU_PREFIX_AWARE_SHORTS_TAKE_OLDEST_CHUNK=1 (optional, with short prompts)
    TPU_PREFIX_AWARE_RANK_WITH_TIER=1 (optional, with an offloading connector)
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
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.block_pool_marks import BLOCKS_HASHED_THIS_STEP
from vllm_torchtpu.core.short_prompts_first import (
    ADMITTABLE_STATUSES,
    ShortPromptsFirstScheduler,
    fit_short_prompts,
)
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000
# Shortest time between two logs of the follow-shared-prefix totals.
_FOLLOW_STATS_INTERVAL_S = 30.0

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


class ShortPrompts(enum.StrEnum):
    """TPU_PREFIX_AWARE_SHORT_PROMPTS: how PrefixAwareScheduler serves short
    prompts beside running long prefills."""

    OFF = "off"
    FIRST = "first"
    FIRST_SHORT_STEP = "first-short-step"


@dataclass
class _Follower:
    """A running prefill that waits while another request caches its chunks."""

    cached_at_step_start: int | None = None


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
    #: Seconds a request's offload-tier probe stays valid while its HBM hit
    #: does not change.
    TIER_PROBE_TTL_S = 1.0
    #: Requests whose HBM-uncached part is shorter than this many tier chunks
    #: are not probed: they already rank as warm.
    TIER_PROBE_MIN_CHUNKS = 2
    #: Maximum tier chunks probed per request (~256K tokens at 4096 tokens/chunk),
    #: bounding per-step store lookups on very long prompts.
    TIER_PROBE_MAX_CHUNKS = 64

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        self._init_tier_warmth(envs.TPU_PREFIX_AWARE_TIER_WARMTH)

        self.follow_shared_prefix = envs.VLLM_TPU_FOLLOW_SHARED_PREFIX
        self._followers: dict[str, _Follower] = {}
        self._followed_steps: dict[str, int] = {}
        self._rejoins: dict[str, _Rejoin] = {}
        self._num_followed_steps = 0
        self._num_rejoins = 0
        self._num_followed_tokens = 0
        self._follow_stats_logged_at: float | None = None

        short_prompts = ShortPrompts(envs.TPU_PREFIX_AWARE_SHORT_PROMPTS)
        self.short_prompt_tokens = (
            None if short_prompts == ShortPrompts.OFF else self._short_prompt_limit()
        )
        self._uncached_admitted: dict[str, int] = {}
        self.hold_shared_prefix = envs.TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX
        self.short_step_tokens = (
            self._smallest_step_below_the_full_step()
            if short_prompts == ShortPrompts.FIRST_SHORT_STEP
            else None
        )
        self._short_step = False
        self._skip_short_prompts = False
        self._carriers: set[str] = set()
        self._held_last_step: set[str] = set()
        self.shorts_take_oldest_chunk = envs.TPU_PREFIX_AWARE_SHORTS_TAKE_OLDEST_CHUNK
        self._short_prompts_present = False

        if self.shorts_take_oldest_chunk and self.short_prompt_tokens is None:
            raise NotImplementedError(
                "TPU_PREFIX_AWARE_SHORTS_TAKE_OLDEST_CHUNK needs short prompts "
                "first; set TPU_PREFIX_AWARE_SHORT_PROMPTS or unset it."
            )

        if self.policy != SchedulingPolicy.FCFS and (
            self.short_prompt_tokens is not None or self.hold_shared_prefix
        ):
            raise NotImplementedError(
                "Short prompts first and the shared-prefix hold need the FCFS "
                "scheduling policy; use --scheduling-policy fcfs, or turn off "
                "TPU_PREFIX_AWARE_SHORT_PROMPTS and "
                "TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX."
            )
        if self._tier is not None and self._tier_warmth:
            raise NotImplementedError(
                "TPU_PREFIX_AWARE_RANK_WITH_TIER already counts the tier in the "
                "window's rank; unset TPU_PREFIX_AWARE_TIER_WARMTH."
            )
        if self.hold_shared_prefix and not self.kv_cache_manager.enable_caching:
            raise NotImplementedError(
                "TPU_PREFIX_AWARE_HOLD_SHARED_PREFIX needs prefix caching; use "
                "--enable-prefix-caching or unset it."
            )

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, tier_warmth=%s, "
            "follow shared prefix: %s, short prompts first: %s; "
            "hold shared prefix: %s; short step: %s; "
            "short prompts take the oldest long chunk: %s; rank with tier: %s)",
            self.WINDOW,
            self._tier_warmth,
            "on" if self.follow_shared_prefix else "off",
            "off"
            if self.short_prompt_tokens is None
            else f"at most {self.short_prompt_tokens} uncached tokens",
            "on" if self.hold_shared_prefix else "off",
            "off"
            if self.short_step_tokens is None
            else f"{self.short_step_tokens} tokens",
            "on" if self.shorts_take_oldest_chunk else "off",
            "on" if self._tier is not None else "off",
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

    def _smallest_step_below_the_full_step(self) -> int:
        """The smallest compiled step size that holds one full chunk and is
        smaller than the full step."""
        step = self.max_num_scheduled_tokens
        chunk = self.scheduler_config.long_prefill_token_threshold
        compile_sizes = self.vllm_config.compilation_config.compile_sizes
        assert compile_sizes is not None, "the TPU platform sets compile_sizes"
        sizes = [int(size) for size in compile_sizes if chunk <= int(size) < step]
        if not sizes:
            raise NotImplementedError(
                "TPU_PREFIX_AWARE_SHORT_PROMPTS=first-short-step needs a "
                f"compiled step size of at least the {chunk}-token chunk and "
                f"below the {step}-token step; add one to compile_sizes or use "
                "TPU_PREFIX_AWARE_SHORT_PROMPTS=first."
            )
        return min(sizes)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._forget_blocks_hashed_last_step()
        if self.follow_shared_prefix:
            self._update_followers()
        held = (
            self._hold_prompts_behind_running_prefixes()
            if self.hold_shared_prefix
            else []
        )
        try:
            if self.short_prompt_tokens is not None:
                output = self._schedule_short_prompts_first(throttle_prefills)
            else:
                output = super().schedule(throttle_prefills)
        finally:
            self._carriers = set()
            for request in reversed(held):
                self.waiting.prepend_request(request)
        if self.short_prompt_tokens is not None:
            self._note_admitted_prompts(output)
        elif self.hold_shared_prefix or self._tier is not None:
            self._forget_requests_that_left_waiting()
        if self.follow_shared_prefix:
            self._restore_rejoined_places()
        return output

    def _forget_blocks_hashed_last_step(self) -> None:
        """Let this step's lookups hit the blocks the last step hashed.

        The forward of the last step writes them before this step's forward
        runs. Scheduler.schedule forgets them the same way, but the hold and
        the short-prompt lookups run before it and would count them as
        misses.
        """
        pool_state = self.kv_cache_manager.block_pool.__dict__
        hashed_last_step = pool_state.get(BLOCKS_HASHED_THIS_STEP)
        if hashed_last_step:
            hashed_last_step.clear()

    def _schedule_short_prompts_first(self, throttle_prefills: bool) -> SchedulerOutput:
        full_step = self.max_num_scheduled_tokens
        reserved = 0
        if not self._skip_short_prompts:
            self._short_step = self._starts_short_step()
            if self._short_step:
                assert self.short_step_tokens is not None
                self.max_num_scheduled_tokens = self.short_step_tokens
            self._short_prompts_present = (
                self.shorts_take_oldest_chunk and self._has_short_prompts()
            )
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
            self.max_num_scheduled_tokens = full_step
            self._short_step = False
            self._short_prompts_present = False
            self._running_prefill_budget = None
        # A reservation that scheduled nothing (the short prompts got no KV
        # blocks) would cap the running long prefills again next step, and a
        # prefill capped at 0 never reaches allocate_slots, so nothing is
        # preempted or freed: every step would schedule nothing. The next step
        # reserves nothing and keeps the full step.
        self._skip_short_prompts = (
            bool(reserved) and output.total_num_scheduled_tokens == 0
        )
        return output

    def _has_short_prompts(self) -> bool:
        """Whether a short prompt is in prefill, or waits with a sequence slot
        free to start it."""
        if self._pause_state != PauseState.UNPAUSED:
            return False
        if any(r.is_prefill_chunk and self._is_short_prompt(r) for r in self.running):
            return True
        return self._free_sequence_slots() > 0 and bool(self._short_waiting_chunks())

    def _starts_short_step(self) -> bool:
        """Whether this step holds at most the short step's tokens: a long
        prefill is running and a short waiting prompt's first chunk fits in
        the short step beside the chunks of running short prefills."""
        if (
            self.short_step_tokens is None
            or self._pause_state != PauseState.UNPAUSED
            or self._decode_tokens() > 0
        ):
            return False
        prefills = [r for r in self.running if r.is_prefill_chunk]
        if all(self._is_short_prompt(r) for r in prefills):
            return False
        _, started = fit_short_prompts(
            ((None, chunk) for chunk in self._short_waiting_chunks()),
            self._short_prefill_chunks(prefills),
            self.short_step_tokens,
            self._free_sequence_slots(),
        )
        return bool(started)

    def _hold_prompts_behind_running_prefixes(self) -> list[Request]:
        """Take out of the waiting queue, for this step, every prompt whose
        first uncached block a running prefill has still to compute, and mark
        those prefills as carriers.

        A prompt the hold lets go loses the cache hit looked up while it was
        held, so the short-prompt count looks it up again with the blocks the
        prefill has cached since."""
        held = self._held_prompts()
        held_ids = {request.request_id for request in held}
        for request_id in self._held_last_step - held_ids:
            self._uncached_by_request.pop(request_id, None)
        self._held_last_step = held_ids
        if held:
            waiting = self.waiting
            assert isinstance(waiting, FCFSRequestQueue)
            waiting.remove_requests(held)
        return held

    def _held_prompts(self) -> list[Request]:
        if self._pause_state != PauseState.UNPAUSED:
            return []
        prefills = [r for r in self.running if r.is_prefill_chunk]
        if not prefills:
            return []
        held = []
        for request in islice(self.waiting, self.WINDOW):
            carrier = self._carrier_of(request, prefills)
            if carrier is not None:
                held.append(request)
                self._carriers.add(carrier)
        return held

    def _carrier_of(self, request: Request, prefills: list[Request]) -> str | None:
        """The running prefill that has still to compute this waiting prompt's
        first uncached block, a block a cache hit of the prompt can cover, or
        None.

        The first uncached block is past both the prompt's cache hit and the
        blocks the prefill has computed, so a hit that comes up short of the
        cached blocks (the EAGLE/MTP drop of the last hit block) does not hold
        the prompt behind blocks the prefill has already cached.
        """
        if (
            request.status != RequestStatus.WAITING
            or request.num_computed_tokens > 0
            or request.skip_reading_prefix_cache
        ):
            return None
        cached = (request.num_tokens - self._uncached_once(request)) // (
            self.hash_block_size
        )
        readable = (request.num_tokens - 1) // self.hash_block_size
        for prefill in prefills:
            index = max(cached, prefill.num_computed_tokens // self.hash_block_size)
            if (
                index < min(readable, len(prefill.block_hashes))
                and prefill.block_hashes[index] == request.block_hashes[index]
            ):
                return prefill.request_id
        return None

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
        """Whether this running long prefill keeps its full chunk beside short
        prompts: every prefill a held prompt waits for does, and so does the
        oldest long prefill unless short prompts take its chunk too. In a short
        step every long prefill gives way."""
        if self._short_step:
            return False
        if request.request_id in self._carriers:
            return True
        if self._short_prompts_present:
            return False
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
        running long prefill keeps its full chunk, unless short prompts take
        it too (TPU_PREFIX_AWARE_SHORTS_TAKE_OLDEST_CHUNK), so short prompts
        take budget only from the long prefills that do not keep theirs."""
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
        reserved, _ = fit_short_prompts(
            ((None, chunk) for chunk in self._short_waiting_chunks()),
            self._short_prefill_chunks(prefills),
            free_tokens,
            self._free_sequence_slots(),
        )
        return min(reserved, free_tokens)

    def _short_prefill_chunks(self, prefills: list[Request]) -> int:
        """Tokens of this step's chunks of the running short prefills."""
        return sum(
            self._chunk(r.num_tokens - r.num_computed_tokens)
            for r in prefills
            if self._is_short_prompt(r)
        )

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

        Such a request waits this step as a follower.
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
        self._followers.setdefault(request_id, _Follower())
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
        last step, so its admission takes the cached tokens.

        The cached prefix is compared between step starts, so a follower that
        the last step did not reach, for example because the requests ahead of
        it used the token budget, still sees its sibling's growth."""
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
            start = follower.cached_at_step_start
            if start is None or cached > start:
                follower.cached_at_step_start = cached
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
        logger.debug(
            "follow shared prefix: %s took %d cached tokens past its own %d "
            "after %d followed steps",
            request.request_id,
            taken,
            rejoin.computed_tokens,
            rejoin.followed_steps,
        )
        now = time.monotonic()
        last = self._follow_stats_logged_at
        if last is not None and now - last < _FOLLOW_STATS_INTERVAL_S:
            return
        self._follow_stats_logged_at = now
        logger.info(
            "follow shared prefix totals: %d rejoins, %d followed steps, "
            "%d cached tokens taken",
            self._num_rejoins,
            self._num_followed_steps,
            self._num_followed_tokens,
        )

    def _init_tier_warmth(self, enabled: bool) -> None:
        self._tier_warmth = bool(enabled)
        self._tier_resolved = False
        self._tier_scheduler: TPURaidenOffloadingScheduler | None = None
        self._tier_probe_broken = False
        # request_id -> (probe time, HBM-hit tokens probed from, tier tokens)
        self._tier_cache: dict[str, tuple[float, int, int]] = {}
        self._paw_tier_tokens = 0

    def _offload_scheduler(self) -> TPURaidenOffloadingScheduler | None:
        """The Raiden offloading connector's scheduler side, or None.

        Looks through a multi-connector's children for a
        ``TPURaidenOffloadingScheduler`` instance.
        """
        if self._tier_resolved:
            return self._tier_scheduler
        self._tier_resolved = True
        connector = self.connector
        if connector is None:
            return None
        children = getattr(connector, "_connectors", None) or (connector,)
        for child in children:
            sched = getattr(child, "connector_scheduler", None)
            if isinstance(sched, TPURaidenOffloadingScheduler):
                self._tier_scheduler = sched
                break
        if self._tier_scheduler is None:
            logger.info(
                "PrefixAwareScheduler: no offloading connector; tier warmth is off."
            )
        return self._tier_scheduler

    def _tier_tokens(self, request: Request, hbm_tokens: int) -> int:
        """Prompt tokens right past the HBM hit that the offload tier can load.

        Counts the run of whole chunks the tier holds (or is storing) from the
        chunk that starts at the HBM hit, in the host pool or, when one is
        configured, the storage tier behind it. A hit ending inside a chunk gets
        nothing: tier loads start only from a chunk-aligned local hit (see
        ``_patch_vllm_offload_fine_prefix_alignment``).
        """
        if not self._tier_warmth or self._tier_probe_broken:
            return 0
        sched = self._offload_scheduler()
        if sched is None:
            return 0
        group_configs = sched.config.kv_group_configs
        if len(group_configs) != 1:
            return 0
        group = group_configs[0]
        chunk_tokens = int(group.tokens_per_chunk)
        hashes_per_chunk = int(group.hashes_per_chunk)
        if chunk_tokens <= 0 or hbm_tokens % chunk_tokens:
            return 0
        # A hit covers at most num_tokens - 1 tokens.
        max_chunks = max(request.num_tokens - 1, 0) // chunk_tokens
        first_chunk = hbm_tokens // chunk_tokens
        if max_chunks - first_chunk < self.TIER_PROBE_MIN_CHUNKS:
            return 0

        now = time.monotonic()
        cached = self._tier_cache.get(request.request_id)
        if (
            cached is not None
            and cached[1] == hbm_tokens
            and now - cached[0] < self.TIER_PROBE_TTL_S
        ):
            return cached[2]

        stop_chunk = min(max_chunks, first_chunk + self.TIER_PROBE_MAX_CHUNKS)
        try:
            block_hashes = request.block_hashes
            keys = []
            for chunk in range(first_chunk, stop_chunk):
                hash_idx = (chunk + 1) * hashes_per_chunk - 1
                if hash_idx >= len(block_hashes):
                    break
                keys.append(make_offload_key(block_hashes[hash_idx], group.group_idx))
            # One store call for the whole run: RaidenOffloadingManager's
            # probe counts the chunks the host pool or the storage tier holds
            # (or is still storing), never asks the global registry, and
            # neither pins, reorders nor stashes anything.
            chunks = sched._raiden_manager.probe_warm_chunks(keys) if keys else 0
        except Exception:
            self._tier_probe_broken = True
            logger.exception(
                "PrefixAwareScheduler: offload-tier probe failed; "
                "ranking on HBM residency only."
            )
            return 0

        if group.is_eagle_group:
            # OffloadingConnectorScheduler pops the trailing chunk of a flagged
            # drafter group's tier hit as a volatile draft tail (and drops a
            # one-chunk hit).
            chunks = max(chunks - 1, 0)

        tier_tokens = chunks * chunk_tokens
        self._tier_cache[request.request_id] = (now, hbm_tokens, tier_tokens)
        if len(self._tier_cache) > 8 * self.WINDOW:
            horizon = now - 10 * self.TIER_PROBE_TTL_S
            self._tier_cache = {
                rid: entry
                for rid, entry in self._tier_cache.items()
                if entry[0] >= horizon
            }
            while len(self._tier_cache) > 4 * self.WINDOW:
                self._tier_cache.pop(next(iter(self._tier_cache)))
        return tier_tokens

    def _prefix_warmth_and_tier(self, request: Request) -> tuple[int, int]:
        """Return ``(warmth_tokens, tier_tokens)`` of ``request``."""
        kvm = self.kv_cache_manager
        if not kvm.enable_caching or request.skip_reading_prefix_cache:
            return 0, 0
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens, 0
        try:
            result = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max(request.num_tokens - 1, 0)
            )
            hbm_tokens = int(result[1])
        except Exception:
            if not self._paw_scoring_broken:
                self._paw_scoring_broken = True
                logger.exception(
                    "PrefixAwareScheduler: warmth scoring failed; falling back to FCFS."
                )
            return 0, 0
        tier_tokens = (
            self._tier_tokens(request, hbm_tokens)
            if self._tier is None
            else self._tier_prefix_once(request, hbm_tokens) - hbm_tokens
        )
        return hbm_tokens + tier_tokens, tier_tokens

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of ``request`` resident in the KV cache."""
        return self._prefix_warmth_and_tier(request)[0]

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int, int]:
        """Return ``(sort_key, warmth, tier_tokens)``. Lower sort_key admits first."""
        warmth, tier_tokens = self._prefix_warmth_and_tier(request)
        return (
            self._rank_key(request, request.num_tokens - warmth),
            warmth,
            tier_tokens,
        )

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
        (best_rank, best_warmth, best_tier_tokens), best = min(
            scored, key=lambda t: t[0][0]
        )

        if best is not head and best_rank < scored[0][0][0]:
            # Only FCFS reaches here (see the early return above).
            cast(FCFSRequestQueue, queue).remove(best)
            queue.prepend_request(best)

        self._paw_n += 1
        self._paw_cold_tokens += max(best.num_tokens - best_warmth, 0)
        self._paw_prompt_tokens += best.num_tokens
        self._paw_tier_tokens += best_tier_tokens

        if self._paw_n % _LOG_EVERY == 0:
            logger.info(
                "warmest-first: n=%d mean_cold_tokens=%.0f "
                "cold_frac=%.3f tier_frac=%.3f queue_depth=%d",
                self._paw_n,
                self._paw_cold_tokens / self._paw_n,
                self._paw_cold_tokens / max(self._paw_prompt_tokens, 1),
                self._paw_tier_tokens / max(self._paw_prompt_tokens, 1),
                len(queue),
            )
