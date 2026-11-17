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
"""Data-parallel engine choice by pending prompt tokens.

vLLM's load balancer (DPLBAsyncMPClient.get_core_engine_for_request) scores
an engine by its waiting and running request counts. A running prefill of
100,000 tokens counts as 1 request, though it holds the engine's whole
token budget for about 100 steps, so a new prompt can wait behind it while
another engine has no prompt work at all.

With VLLM_TPU_DP_PROMPT_BACKLOG=1:

* Each engine core publishes its pending prompt tokens with the request
  counts it already sends the DP coordinator: the uncached prompt tokens of
  its waiting requests (kv_holding_waiting and waiting, the two queues the
  scheduler admits from) plus the prompt tokens its running requests have
  not computed yet. The coordinator relays [waiting, running, kv_cache_usage] per
  engine without reading the third value, so the engine sends
  ``pending + kv_cache_usage`` there (KV usage capped below 1) and the front
  end splits it again. Only these coordinator messages carry the sum; the
  scheduler stats the API server logs and exports are unchanged.
* The front end's client splits each relayed row back into vLLM's
  [waiting, running, kv_cache_usage] and the pending prompt tokens, so vLLM's
  own score reads the same numbers as without the switch. It takes vLLM's
  own choice unless that engine's pending prompt
  tokens exceed the least backlogged engine's by more than MARGIN times the
  new prompt's length; then it sends the request to the least backlogged
  engine (ties broken by fewer waiting plus running requests, then by the
  lower engine index). With MARGIN >= 1, a moved request has fewer prompt
  tokens to wait for and compute on its new engine than on vLLM's choice,
  even in the worst case where vLLM's choice holds its whole prompt in cache
  and the new engine computes all of it.
* Only prompt tokens are compared. The choice does not look at free
  sequence slots: an engine running max_num_seqs requests with no pending
  prompt tokens still takes a moved request, which then waits there for a
  slot.

The choice only decides where a request runs; an engine computes the same
tokens with or without a prefix-cache hit.
"""

import math
from itertools import chain

from vllm.v1.pool.late_interaction import get_late_interaction_engine_index

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LOG_EVERY_REQUESTS = 100
KV_USAGE_CAP = 0.999999


def pack(pending_tokens: int, kv_cache_usage: float) -> float:
    return float(pending_tokens) + min(max(kv_cache_usage, 0.0), KV_USAGE_CAP)


def unpack(value: float) -> tuple[int, float]:
    pending = math.floor(value)
    return pending, value - pending


class PendingPrompts:
    """Pending prompt tokens of one scheduler. A waiting request's cached
    prefix is looked up once, the first time it is seen waiting."""

    def __init__(self) -> None:
        self.cached_when_waiting: dict[str, int] = {}

    def cached_tokens(self, scheduler, request) -> int:
        cached = self.cached_when_waiting.get(request.request_id)
        if cached is None:
            manager = scheduler.kv_cache_manager
            cached = 0
            if request.block_hashes and manager.prefix_cache_lookup_enabled(request):
                cached = manager.coordinator.find_longest_cache_hit(
                    request.block_hashes, request.num_tokens - 1
                )[1]
            self.cached_when_waiting[request.request_id] = cached
        return cached

    def tokens(self, scheduler) -> int:
        pending = 0
        waiting_ids = set()
        for request in chain(scheduler.kv_holding_waiting, scheduler.waiting):
            waiting_ids.add(request.request_id)
            done = max(
                request.num_computed_tokens, self.cached_tokens(scheduler, request)
            )
            pending += max(0, request.num_prompt_tokens - done)
        for request in scheduler.running:
            pending += max(0, request.num_prompt_tokens - request.num_computed_tokens)
        if len(self.cached_when_waiting) > len(waiting_ids):
            self.cached_when_waiting = {
                rid: n
                for rid, n in self.cached_when_waiting.items()
                if rid in waiting_ids
            }
        return pending


def make_publisher():
    """A DPEngineCoreProc._maybe_publish_request_counts that also publishes
    pending prompt tokens and republishes when only they change.

    It replaces DPEngineCoreProc._maybe_publish_request_counts in vLLM's
    vllm/v1/engine/core.py (v0.31.1rc0, the pinned version) instead of
    wrapping it: the original publishes only when the request counts change
    and sends the KV usage as is, which are the two things this one changes.
    The rest, the SchedulerStats it builds and the (-1, EngineCoreOutputs)
    message it queues, is the original's body; compare it with upstream
    when the pin moves.
    """
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.stats import SchedulerStats

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return
        scheduler = self.scheduler
        prompts = getattr(self, "_pending_prompts", None)
        if prompts is None:
            prompts = self._pending_prompts = PendingPrompts()
        counts = scheduler.get_request_counts()
        pending = prompts.tokens(scheduler)
        if (counts, pending) != self.last_counts:
            self.last_counts = (counts, pending)
            stats = SchedulerStats(
                *counts,
                kv_cache_usage=pack(pending, scheduler.get_kv_cache_usage()),
                step_counter=self.step_counter,
                current_wave=self.current_wave,
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    return _maybe_publish_request_counts


class BacklogStats:
    def __init__(self) -> None:
        self.requests = 0
        self.moved = 0
        self.tokens_skipped = 0

    def log(self) -> None:
        logger.info(
            "dp prompt backlog: %d requests, %d sent away from vLLM's choice to "
            "an engine with fewer pending prompt tokens, %d pending prompt "
            "tokens skipped in total",
            self.requests,
            self.moved,
            self.tokens_skipped,
        )


def _get_lb_engines(self):
    if "_lb_counts" not in self.__dict__:
        # A client built before the switch was installed holds plain rows.
        _set_lb_engines(self, self.__dict__.pop("lb_engines"))
    return self._lb_counts


def _set_lb_engines(self, rows):
    """The coordinator's rows arrive as [waiting, running, pending + KV usage];
    keep them in vLLM's [waiting, running, KV usage] form and the pending
    prompt tokens beside them."""
    self._lb_counts = []
    self._lb_pending = []
    for waiting, running, packed in rows:
        pending, kv_cache_usage = unpack(packed)
        self._lb_counts.append([waiting, running, kv_cache_usage])
        self._lb_pending.append(pending)


def make_router(stock, margin: float):
    """A get_core_engine_for_request that takes vLLM's choice unless that
    engine's pending prompt tokens exceed the least backlogged engine's by
    more than `margin` times the new prompt's length."""
    stats = BacklogStats()

    def get_core_engine_for_request(self, request):
        if (
            request.request_id in self.reqs_in_flight
            or request.data_parallel_rank is not None
            or get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
            is not None
        ):
            return stock(self, request)

        engine = stock(self, request)
        chosen = self.core_engines.index(engine)
        counts, pending = self.lb_engines, self._lb_pending
        least = min(
            range(len(pending)),
            key=lambda idx: (pending[idx], counts[idx][0] + counts[idx][1]),
        )
        prompt_tokens = len(request.prompt_token_ids or ())
        stats.requests += 1
        if pending[chosen] - pending[least] > margin * prompt_tokens:
            stats.moved += 1
            stats.tokens_skipped += pending[chosen] - pending[least]
            counts[chosen][0] -= self.client_count
            self.engine_inflight[engine] -= 1
            chosen = least
            engine = self.core_engines[chosen]
            counts[chosen][0] += self.client_count
            self.engine_inflight[engine] += 1
            self.reqs_in_flight[request.request_id] = engine
        pending[chosen] += prompt_tokens
        if stats.requests % LOG_EVERY_REQUESTS == 0:
            stats.log()
        return engine

    # pyrefly: ignore[missing-attribute]
    get_core_engine_for_request.backlog_stats = stats
    return get_core_engine_for_request


def install(margin: float) -> None:
    from vllm.v1.engine.core import DPEngineCoreProc
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    if getattr(DPLBAsyncMPClient.get_core_engine_for_request, "backlog_stats", None):
        return
    DPEngineCoreProc._maybe_publish_request_counts = make_publisher()
    # pyrefly: ignore[no-access]
    DPLBAsyncMPClient.lb_engines = property(_get_lb_engines, _set_lb_engines)
    DPLBAsyncMPClient.get_core_engine_for_request = make_router(
        DPLBAsyncMPClient.get_core_engine_for_request, margin
    )
    logger.info(
        "dp prompt backlog: engines publish pending prompt tokens; a request "
        "leaves vLLM's choice when that engine's backlog exceeds the least "
        "backlogged engine's by more than %.2f x its prompt",
        margin,
    )


def install_if_enabled() -> None:
    from vllm_torchtpu import envs

    if envs.VLLM_TPU_DP_PROMPT_BACKLOG:
        install(envs.VLLM_TPU_DP_PROMPT_BACKLOG_MARGIN)
