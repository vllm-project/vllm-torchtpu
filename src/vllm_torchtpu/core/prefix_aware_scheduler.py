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

Cached means resident in HBM or, when the KV connector implements
`ExternalPrefixProbe` (host offload), loadable from the connector's tier: a
host-resident prefix costs a DMA, not a prefill, so it must not rank as cold.

Warmest-first alone starves long cold prompts: while warmer requests keep
arriving they never reach the front, their session stalls, and its cached
prefix ages out. A request that has waited `paw_max_wait_s` seconds (additional
config, default 0 = off) is promoted ahead of every normal request,
oldest first.

With ``VLLM_TPU_SIBLING_HOLD`` (default off), requests that share a prefix
with a request still prefilling it (subagent forks sent together) are held:
a fresh waiting request whose next uncached KV block is a prompt block that a
running request has not yet scheduled is skipped for the step, so it does not
prefill the shared prefix a second time in parallel. vLLM caches a block when
it is scheduled (hittable from the next step on TPU), so the held request
resumes with that prefix as a hit one step after the running request's chunk
covers it. A request is held at most
``VLLM_TPU_SIBLING_HOLD_MAX_S`` seconds in total, never when the connector can
load the block, and never when it is resumed, preempted or holds KV blocks.

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
"""

import time
from itertools import islice
from typing import Any, cast

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.core import reuse_aware_eviction
from vllm_torchtpu.core.eviction_aware_connector import EvictionAwareConnector
from vllm_torchtpu.core.external_prefix_probe import ExternalPrefixProbe
from vllm_torchtpu.runner.utils import PAW_MAX_WAIT_S_KEY

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_AGED = 2
_TIER_NORMAL = 3

# Seconds a request may wait before it is promoted ahead of warmer requests
# (additional_config PAW_MAX_WAIT_S_KEY; 0 disables aging).
DEFAULT_PAW_MAX_WAIT_S = 0.0

# Connector probes are cached per request and refreshed every this many
# scheduler steps: the host tier changes slowly, and the window is re-scored
# every step.
_EXTERNAL_PROBE_REFRESH_STEPS = 8

# Sibling hold: first-held time of a request whose hold ran out.
_HOLD_EXPIRED = float("-inf")


class PrefixAwareScheduler(AsyncScheduler):
    """Admit ready requests prioritizing fewest uncached prompt tokens."""

    #: Lookahead window of ready requests to evaluate for cached prefix length.
    WINDOW = 64

    # Class-level defaults so partially constructed instances (tests) work.
    _paw_max_wait_s: float = DEFAULT_PAW_MAX_WAIT_S
    _paw_probe: ExternalPrefixProbe | None = None
    _paw_step: int = 0
    _paw_external_cache: dict[str, tuple[int, int, int, int]] | None = None
    _paw_aged: int = 0
    # Sibling hold (VLLM_TPU_SIBLING_HOLD).
    _sib_hold: bool = False
    _sib_max_s: float = 0.0
    _sib_first: dict[str, float] | None = None
    _sib_local: dict[str, int] | None = None
    _sib_holds: int = 0
    _sib_expired: int = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        self._paw_max_wait_s = _max_wait_from_config(getattr(self, "vllm_config", None))
        # Before the connector sees the block pool: contiguous_alloc keeps a
        # reference to the free queue this replaces.
        kvm = getattr(self, "kv_cache_manager", None)
        if kvm is not None and reuse_aware_eviction.enabled():
            reuse_aware_eviction.install(kvm)
        connector = getattr(self, "connector", None)
        self._paw_probe = (
            connector if isinstance(connector, ExternalPrefixProbe) else None
        )
        if isinstance(connector, EvictionAwareConnector):
            connector.attach_block_pool(self.kv_cache_manager.block_pool)
        self._paw_external_cache = {}
        self._sib_hold = envs.VLLM_TPU_SIBLING_HOLD
        self._sib_max_s = envs.VLLM_TPU_SIBLING_HOLD_MAX_S
        # Request id -> wall time it was first held; request id -> local
        # prefix hit (tokens) at its last warmth scoring.
        self._sib_first = {}
        self._sib_local = {}
        if self._sib_hold:
            logger.info(
                "PrefixAwareScheduler: sibling hold on (max %.1fs)", self._sib_max_s
            )

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, max_wait=%.0fs, "
            "connector warmth=%s)",
            self.WINDOW,
            self._paw_max_wait_s,
            self._paw_probe is not None,
        )

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of ``request`` resident in the KV cache."""
        kvm = self.kv_cache_manager
        if not kvm.enable_caching or request.skip_reading_prefix_cache:
            return 0
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens
        try:
            result = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max(request.num_tokens - 1, 0)
            )
            local = int(result[1])
            if self._sib_hold and self._sib_local is not None:
                self._sib_local[request.request_id] = local
        except Exception:
            if not self._paw_scoring_broken:
                self._paw_scoring_broken = True
                logger.exception(
                    "PrefixAwareScheduler: warmth scoring failed; falling back to FCFS."
                )
            return 0
        return local + self._external_warmth(request, local)

    def _external_warmth(self, request: Request, local: int) -> int:
        """Prompt tokens past `local` the KV connector could load (0 if none)."""
        probe = self._paw_probe
        if probe is None:
            return 0
        cache = self._paw_external_cache
        if cache is None:
            cache = self._paw_external_cache = {}
        num_hashes = len(request.block_hashes)
        cached = cache.get(request.request_id)
        if (
            cached is not None
            and cached[0] == local
            and cached[1] == num_hashes
            and self._paw_step - cached[3] < _EXTERNAL_PROBE_REFRESH_STEPS
        ):
            return cached[2]
        try:
            external = int(probe.peek_external_prefix_tokens(request, local))
        except Exception:
            logger.exception(
                "PrefixAwareScheduler: connector warmth probe failed; "
                "ranking by local warmth only."
            )
            self._paw_probe = None
            return 0
        external = max(0, min(external, request.num_tokens - local))
        cache[request.request_id] = (local, num_hashes, external, self._paw_step)
        return external

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        ready = super()._handle_blocked_waiting_request(request)
        if (
            ready
            and self._sib_hold
            and self._sib_max_s > 0
            and self._sibling_held(request, start=True)
        ):
            # Skipped for this step only: it stays at its place in the queue.
            return False
        return ready

    def _sibling_held(self, request: Request, start: bool = False) -> bool:
        """Whether ``request`` waits this step for a running request to
        schedule the next block of their shared prefix. ``start`` starts (and
        counts) the hold; without it this only checks."""
        first_held = self._sib_first
        if first_held is None:
            first_held = self._sib_first = {}
        now = time.time()
        first = first_held.get(request.request_id)
        if first is not None and now - first >= self._sib_max_s:
            if start and first != _HOLD_EXPIRED:
                # Count each expired hold once.
                first_held[request.request_id] = _HOLD_EXPIRED
                self._sib_expired += 1
            return False
        if not self._sibling_leader_pending(request):
            return False
        if start:
            if first is None:
                first_held[request.request_id] = now
                if len(first_held) > 4 * self.WINDOW:
                    self._sib_prune()
            self._sib_holds += 1
            if self._sib_holds % _LOG_EVERY == 0:
                logger.info(
                    "sibling hold: held_steps=%d held_requests=%d expired=%d",
                    self._sib_holds,
                    len(first_held),
                    self._sib_expired,
                )
        return True

    def _sib_prune(self) -> None:
        """Drop hold state of requests no longer waiting."""
        reqs = getattr(self, "requests", {})
        for state in (self._sib_first, self._sib_local):
            if not state:
                continue
            for req_id in [
                k
                for k in state
                if (r := reqs.get(k)) is None or r.status != RequestStatus.WAITING
            ]:
                del state[req_id]

    def _local_hit(self, request: Request) -> int:
        result = self.kv_cache_manager.coordinator.find_longest_cache_hit(
            request.block_hashes, max(request.num_tokens - 1, 0)
        )
        return int(result[1])

    def _sibling_leader_pending(self, request: Request) -> bool:
        """Whether a running request still has to prefill the scheduler
        block that would extend ``request``'s prefix-cache hit.

        Block hashes chain their parent, so equal hashes at one index mean an
        equal prompt up to there. Blocks are cached when scheduled and are
        hits from the next step on (the TPU same-step patch hides blocks the
        current step has not written). So a shared block past the hit is one
        the running request schedules this step or later, and its hold ends
        the step after the running request has scheduled it.
        """
        if (
            request.status != RequestStatus.WAITING
            or request.num_computed_tokens > 0
            or request.num_preemptions > 0
            or request.skip_reading_prefix_cache
            or self._paw_scoring_broken
        ):
            return False
        kvm = self.kv_cache_manager
        if not kvm.enable_caching:
            return False
        hashes = request.block_hashes
        hash_bs = int(getattr(self, "hash_block_size", 0) or self.block_size)
        sched_bs = int(self.block_size)
        if hash_bs <= 0 or sched_bs % hash_bs:
            return False
        try:
            local_cache = self._sib_local
            if local_cache is None:
                local_cache = self._sib_local = {}
            local = local_cache.get(request.request_id)
            if local is None:
                local = local_cache[request.request_id] = self._local_hit(request)
            # Cheap filter at the last scored hit: a running prefill sharing
            # the block there. Its scheduled blocks only extend the hit.
            i = local // hash_bs
            if i >= len(hashes):
                return False
            h = hashes[i]
            leaders = [
                r
                for r in self.running
                if r is not request
                and r.num_computed_tokens < r.num_prompt_tokens
                and len(r.block_hashes) > i
                and r.block_hashes[i] == h
            ]
            if not leaders:
                return False
            local = local_cache[request.request_id] = self._local_hit(request)
        except Exception:
            logger.exception("sibling hold: prefix lookup failed; hold disabled.")
            self._sib_hold = False
            return False
        # Last hash block of the next scheduler block past the hit; a hit
        # stops one token short of the prompt end.
        j = local // hash_bs + sched_bs // hash_bs - 1
        end = (j + 1) * hash_bs
        if j >= len(hashes) or end > request.num_tokens - 1:
            return False
        hj = hashes[j]
        if not any(
            end <= r.num_prompt_tokens
            and len(r.block_hashes) > j
            and r.block_hashes[j] == hj
            for r in leaders
        ):
            return False
        # The connector loads it: a DMA, not a prefill, so do not wait.
        return local + self._external_warmth(request, local) < end

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth). Lower sort_key is admitted first."""
        warmth = self._prefix_warmth(request)
        cold_tokens = float(request.num_tokens - warmth)
        if request.num_preemptions > 0:
            tier = _TIER_PREEMPTED
        elif self._is_aged(request):
            # Oldest first among aged requests, whatever their warmth.
            return (_TIER_AGED, 0.0, request.arrival_time), warmth
        else:
            tier = _TIER_NORMAL
        return (tier, cold_tokens, request.arrival_time), warmth

    def _is_aged(self, request: Request) -> bool:
        max_wait = self._paw_max_wait_s
        return max_wait > 0 and time.time() - request.arrival_time >= max_wait

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        return queue

    def _drop_erased_external_warmth(self) -> None:
        """Forget cached connector warmth that covers host blocks the
        connector has since erased (exclusive host tier), so admission and
        sibling hold do not count on a load that would miss. A connector
        without `take_erased_block_hashes` never erases: nothing to do."""
        take = getattr(self._paw_probe, "take_erased_block_hashes", None)
        cache = self._paw_external_cache
        if take is None:
            return
        erased = take()
        if erased is not None and not erased:
            return
        if not cache:
            return
        if erased is None:
            cache.clear()
            return
        reqs = getattr(self, "requests", None) or {}
        hash_bs = int(getattr(self, "hash_block_size", 0) or self.block_size)
        for req_id, (local, _, external, _) in list(cache.items()):
            if external <= 0:
                continue
            request = reqs.get(req_id)
            if request is None:
                del cache[req_id]
                continue
            lo = local // hash_bs
            hi = (local + external) // hash_bs + 1
            if not erased.isdisjoint(request.block_hashes[lo:hi]):
                del cache[req_id]

    def schedule(self, throttle_prefills: bool = False):
        self._drop_erased_external_warmth()
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        return super().schedule(throttle_prefills)

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
        if self.policy != SchedulingPolicy.FCFS or len(queue) < 2:
            return

        self._paw_step += 1
        window = list(islice(queue, self.WINDOW))
        head = window[0]
        scored = [(self._rank_and_warmth(r), r) for r in window]
        (best_rank, best_warmth), best = min(scored, key=lambda t: t[0][0])
        if best_rank[0] == _TIER_AGED:
            self._paw_aged += 1
        cache = self._paw_external_cache
        if cache and len(cache) > 4 * self.WINDOW:
            live = {r.request_id for r in window}
            for req_id in [k for k in cache if k not in live]:
                del cache[req_id]

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
                "cold_frac=%.3f queue_depth=%d aged=%d",
                self._paw_n,
                self._paw_cold_tokens / self._paw_n,
                self._paw_cold_tokens / max(self._paw_prompt_tokens, 1),
                len(queue),
                self._paw_aged,
            )


def _max_wait_from_config(vllm_config: Any) -> float:
    """`paw_max_wait_s` from `--additional-config` (default 60 s, 0 = off)."""
    additional = getattr(vllm_config, "additional_config", None)
    raw = (
        additional.get(PAW_MAX_WAIT_S_KEY, DEFAULT_PAW_MAX_WAIT_S)
        if isinstance(additional, dict)
        else DEFAULT_PAW_MAX_WAIT_S
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"additional_config[{PAW_MAX_WAIT_S_KEY!r}] must be a number of "
            f"seconds, got {raw!r}"
        ) from None
    if value < 0:
        raise ValueError(
            f"additional_config[{PAW_MAX_WAIT_S_KEY!r}] must be >= 0, got {value}"
        )
    return value
