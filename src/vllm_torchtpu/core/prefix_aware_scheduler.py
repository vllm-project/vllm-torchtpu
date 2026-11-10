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

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
"""

import time
from itertools import islice
from typing import cast

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.request import Request

logger = init_logger("vllm." + __name__)


def _host_phases_every() -> int:
    from vllm_torchtpu import envs

    return envs.VLLM_TPU_HOST_PHASES


def _prompt_ids_ndarray() -> bool:
    from vllm_torchtpu import envs

    return envs.VLLM_TPU_PROMPT_IDS_NDARRAY


def ship_prompt_ids_as_ndarray(scheduler_output, requests) -> None:
    """Replace each new request's ``prompt_token_ids`` list in the scheduler
    output by the int32 ndarray cached on its ``Request`` at add time.

    Only the IPC copy changes: the engine's ``Request`` keeps its list. The
    workers' consumers (``len``, slicing into ``token_ids_cpu``, indexing)
    see the same values; the copy into ``token_ids_cpu`` becomes a memcpy
    and the pickle carries one buffer instead of ~100K int objects.
    """
    for new_req in scheduler_output.scheduled_new_reqs:
        ids = new_req.prompt_token_ids
        if ids is None or not isinstance(ids, list):
            continue
        request = requests.get(new_req.req_id)
        arr = getattr(request, "_prompt_ids_np", None)
        # A streaming-input request may have grown its list since add time.
        if arr is None or request.prompt_token_ids is not ids or len(arr) != len(ids):
            continue
        new_req.prompt_token_ids = arr

_LOG_EVERY = 1000

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


class PrefixAwareScheduler(AsyncScheduler):
    """Admit ready requests prioritizing fewest uncached prompt tokens."""

    #: Lookahead window of ready requests to evaluate for cached prefix length.
    WINDOW = 64

    # VLLM_TPU_HOST_PHASES instrumentation; class defaults keep it off for
    # instances built without __init__.
    _hp_every = 0
    _hp_n = 0
    _hp_promote_ns = 0
    _prompt_np = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        # VLLM_TPU_HOST_PHASES: engine-side host time per scheduler call.
        self._hp_every = _host_phases_every()
        self._hp_samples: dict[str, list[float]] = {}
        self._hp_n = 0
        self._hp_promote_ns = 0
        self._prompt_np = _prompt_ids_ndarray()

        logger.info("PrefixAwareScheduler ACTIVE (window=%d)", self.WINDOW)

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
            return int(result[1])
        except Exception:
            if not self._paw_scoring_broken:
                self._paw_scoring_broken = True
                logger.exception(
                    "PrefixAwareScheduler: warmth scoring failed; falling back to FCFS."
                )
            return 0

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth). Lower sort_key is admitted first."""
        warmth = self._prefix_warmth(request)
        cold_tokens = float(request.num_tokens - warmth)
        tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        return (tier, cold_tokens, request.arrival_time), warmth

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        return queue

    def schedule(self, throttle_prefills: bool = False):
        if self._hp_every:
            out = self._schedule_timed(throttle_prefills)
        else:
            queue = getattr(self, "kv_holding_waiting", None) or getattr(
                self, "waiting", None
            )
            if queue is not None:
                self._promote_warmest(queue)
            out = super().schedule(throttle_prefills)
        if self._prompt_np and out.scheduled_new_reqs:
            ship_prompt_ids_as_ndarray(out, self.requests)
        return out

    # ---- VLLM_TPU_HOST_PHASES instrumentation (off by default) ----
    def _hp_note(self, name: str, us: float) -> None:
        self._hp_samples.setdefault(name, []).append(us)

    def _schedule_timed(self, throttle_prefills: bool):
        t0 = time.perf_counter_ns()
        self._hp_promote_ns = 0
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        out = super().schedule(throttle_prefills)
        sched_us = (time.perf_counter_ns() - t0) / 1e3
        promote_us = self._hp_promote_ns / 1e3
        kind = "admit" if out.scheduled_new_reqs else "other"
        self._hp_note(f"{kind}.schedule", sched_us)
        self._hp_note(f"{kind}.promote", promote_us)
        self._hp_note(f"{kind}.waiting", float(len(queue) if queue is not None else 0))
        # Read by the runner's host phases (pickled with the dataclass).
        try:
            out._host_phases_stamp = (time.time(), promote_us, sched_us)
        except AttributeError:  # a slotted SchedulerOutput: no stamp
            pass
        self._hp_tick()
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        if not self._hp_every:
            return super().update_from_output(scheduler_output, model_runner_output)
        t0 = time.perf_counter_ns()
        out = super().update_from_output(scheduler_output, model_runner_output)
        self._hp_note("update_from_output", (time.perf_counter_ns() - t0) / 1e3)
        return out

    def add_request(self, request: Request) -> None:
        t0 = time.perf_counter_ns() if self._hp_every else 0
        if self._prompt_np and isinstance(request.prompt_token_ids, list):
            # Once per request, at arrival (usually off the admit step).
            request._prompt_ids_np = np.asarray(request.prompt_token_ids, np.int32)
        super().add_request(request)
        if self._hp_every:
            self._hp_note("add_request", (time.perf_counter_ns() - t0) / 1e3)

    def _hp_tick(self) -> None:
        self._hp_n += 1
        if self._hp_n % self._hp_every:
            return
        parts = []
        for name, values in sorted(self._hp_samples.items()):
            v = np.asarray(values, dtype=np.float64)
            parts.append(
                f"{name}={np.percentile(v, 50):.0f}/{np.percentile(v, 90):.0f}"
                f"/{v.max():.0f}/n{len(v)}"
            )
        logger.info(
            "SCHEDPHASES calls=%d host us (p50/p90/max/count) %s",
            self._hp_n,
            " ".join(parts),
        )
        self._hp_samples.clear()

    def _promote_warmest(self, queue: RequestQueue) -> None:
        if not self._hp_every:
            return self._promote_warmest_impl(queue)
        t0 = time.perf_counter_ns()
        try:
            return self._promote_warmest_impl(queue)
        finally:
            self._hp_promote_ns += time.perf_counter_ns() - t0

    def _promote_warmest_impl(self, queue: RequestQueue) -> None:
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
