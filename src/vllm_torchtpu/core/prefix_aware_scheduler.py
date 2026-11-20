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

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
    VLLM_TPU_FOLLOW_SHARED_PREFIX=1 (optional)
"""

import time
from dataclasses import dataclass
from itertools import islice
from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
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
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


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


class PrefixAwareScheduler(AsyncScheduler):
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

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, tier_warmth=%s, "
            "follow shared prefix: %s)",
            self.WINDOW,
            self._tier_warmth,
            "on" if self.follow_shared_prefix else "off",
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if not self.follow_shared_prefix:
            return super().schedule(throttle_prefills)
        self._update_followers()
        output = super().schedule(throttle_prefills)
        self._restore_rejoined_places()
        return output

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        num_new_tokens = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if (
            self.follow_shared_prefix
            and num_new_tokens > 0
            and self._chunk_is_cached(request, num_computed_tokens + num_new_tokens)
        ):
            return 0
        return num_new_tokens

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
        tier_tokens = self._tier_tokens(request, hbm_tokens)
        return hbm_tokens + tier_tokens, tier_tokens

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of ``request`` resident in the KV cache."""
        return self._prefix_warmth_and_tier(request)[0]

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int, int]:
        """Return ``(sort_key, warmth, tier_tokens)``. Lower sort_key admits first."""
        warmth, tier_tokens = self._prefix_warmth_and_tier(request)
        cold_tokens = float(request.num_tokens - warmth)
        tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        return (tier, cold_tokens, request.arrival_time), warmth, tier_tokens

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
