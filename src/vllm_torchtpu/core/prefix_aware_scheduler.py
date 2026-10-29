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

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
"""

import time
from itertools import islice
from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.kv_offload.base import LookupResult, make_offload_key
from vllm.v1.request import Request

from vllm_torchtpu import envs
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000
_WARM_RESULTS = (LookupResult.HIT, LookupResult.HIT_PENDING)

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


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

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, tier_warmth=%s)",
            self.WINDOW,
            self._tier_warmth,
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
        chunk that starts at the HBM hit. A hit ending inside a chunk gets
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
            chunks = 0
            for chunk in range(first_chunk, stop_chunk):
                hash_idx = (chunk + 1) * hashes_per_chunk - 1
                if hash_idx >= len(block_hashes):
                    break
                key = make_offload_key(block_hashes[hash_idx], group.group_idx)
                # Local-only probe: RaidenOffloadingManager.lookup neither pins
                # nor reorders local hits, and local_only=True skips remote
                # global-registry RPCs and _remote_slices mutation.
                if (
                    sched._raiden_manager.lookup(key, None, local_only=True)
                    not in _WARM_RESULTS
                ):
                    break
                chunks += 1
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
