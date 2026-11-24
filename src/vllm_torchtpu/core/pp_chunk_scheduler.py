# SPDX-License-Identifier: Apache-2.0
"""Scheduler that sizes prefill chunks against a per-step time budget.

vLLM's scheduler fills every step up to a token budget and clips each
request's chunk at one point, ``_mamba_block_aligned_split``. This subclass
routes every request through that point, clips prefill chunks to what the
step's remaining time holds (``core/pp_chunks.py``) and then applies the
base scheduler's mamba block alignment when the model has a mamba cache.
Decode requests count toward the step before any prefill chunk is sized,
whatever their position in the running queue. When the pending prefill
work cannot fill every stage with a full step, the step shrinks to that
work spread over the pipeline, so a lone prompt runs as one small step per
stage instead of trailing itself through the pipeline one full step at a
time; a busy server keeps full steps and first-come order. A step also holds
at most as many (query block, KV block) attention pairs as the batched
attention kernel can schedule, which bounds a chunk deep in a very long
prompt so the kernel never overruns its SMEM. The cost model
arrives after the engine core has built its executor and scheduler; until
then chunks are cut by token count alone.

With ``TPU_PP_WARM_FIRST`` the head of the waiting queue is ordered before
each step by uncached prompt tokens, fewest first, aged by the time each
request has waited (``_order_waiting``), so a warm request is admitted ahead
of a cold one that arrived shortly before it. Running requests keep their
order. ``TPU_PP_WARM_FIRST_TIER`` also counts the tokens the offload
connector's host pool holds as cached.
"""

import math
import time
from itertools import islice
from typing import Any, cast

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import envs
from vllm_torchtpu.core.pp_chunks import StepCostModel
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_CLS = "vllm_torchtpu.core.pp_chunk_scheduler.TpuPipelineChunkScheduler"

# TPU_PP_WARM_FIRST: requests at the head of the waiting queue that are
# ordered each step; a request's uncached prompt tokens are looked up again
# once the last lookup is this old, so an eviction while it waits is seen.
_WARM_FIRST_WINDOW = 128
_WARM_FIRST_LOOKUP_TTL_S = 10.0
_WARM_FIRST_LOG_INTERVAL_S = 60.0


def uses_dynamic_chunks(vllm_config: Any) -> bool:
    """Whether this configuration sizes pipeline steps by time; the platform
    installs the scheduler class only when it does."""
    return (
        vllm_config is not None
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.scheduler_config.scheduler_cls == SCHEDULER_CLS
    )


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


def _host_tier(connector: Any) -> Any | None:
    """The scheduler-side connector that reports host-tier hits: the
    connector itself or a member of a multi-connector; None without one."""
    if connector is None:
        return None
    members = getattr(connector, "sub_connectors", None)
    for member in [connector] if members is None else members:
        if callable(getattr(member, "host_tier_hit_tokens", None)):
            return member
    return None


def _started(request: Any) -> bool:
    """Whether a waiting request resumes work: it holds computed tokens or
    was preempted."""
    return request.num_computed_tokens > 0 or request.num_preemptions > 0


class TpuPipelineChunkScheduler(Scheduler):
    # Prompt tokens of a request in the local prefix cache, counted as the
    # prefix-aware scheduler counts them; it reads only kv_cache_manager and
    # the _paw_scoring_broken latch.
    _prefix_warmth = PrefixAwareScheduler._prefix_warmth

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_warm_first(
            envs.TPU_PP_WARM_FIRST,
            float(envs.TPU_PP_WARM_FIRST_AGING),
            envs.TPU_PP_WARM_FIRST_TIER,
            int(envs.TPU_PP_WARM_FIRST_WINDOW),
        )
        self._cost: StepCostModel | None = None
        self._step_limit: int | None = None
        self._pp_size = 1
        self._mamba_split = self.need_mamba_block_aligned_split
        self.need_mamba_block_aligned_split = True
        self._step_tokens = 0
        self._step_pairs = 0.0
        self._step_squares = 0.0
        self._step_prefill_tokens = 0
        self._step_schedule = 0
        self._seeded: set[str] = set()
        self._clips = 0
        self._waits = 0
        logger.info("Pipeline chunk scheduler: waiting for the step cost model")

    def install_cost_model(self, model: StepCostModel) -> None:
        self._cost = model
        self._pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        logger.info("Pipeline chunk scheduler: %s", model.describe())

    def _account(self, tokens: int, prefix: int) -> None:
        self._step_tokens += tokens
        self._step_pairs += float(tokens) * prefix
        self._step_squares += float(tokens) * tokens

    def _init_warm_first(
        self,
        enabled: bool,
        aging: float,
        tier: bool = False,
        window: int = 0,
    ) -> None:
        if aging < 0:
            raise ValueError(f"TPU_PP_WARM_FIRST_AGING must be >= 0, got {aging}")
        if window < -1:
            raise ValueError(
                f"TPU_PP_WARM_FIRST_WINDOW must be >= -1, got {window}"
            )
        self._warm_first = enabled
        self._warm_first_aging = aging
        self._warm_first_window: int | None = (
            None
            if window == -1
            else (_WARM_FIRST_WINDOW if window == 0 else window)
        )
        # The connector whose host pool counts as cached (TPU_PP_WARM_FIRST_TIER).
        self._tier = _host_tier(self.connector) if enabled and tier else None
        if enabled and tier and self._tier is None:
            logger.warning(
                "Pipeline chunk scheduler: TPU_PP_WARM_FIRST_TIER is set but no "
                "KV connector reports host-tier hits; only the local prefix "
                "cache counts as cached."
            )
        self._wf_tier_hits = 0
        # Request id -> (uncached prompt tokens, wall time of the lookup).
        self._uncached: dict[str, tuple[int, float]] = {}
        self._paw_scoring_broken = False
        self._wf_sorts = 0
        self._wf_reorders = 0
        self._wf_lookups = 0
        self._wf_log_t = time.monotonic()
        if enabled:
            window_desc = (
                "all"
                if self._warm_first_window is None
                else str(self._warm_first_window)
            )
            logger.info(
                "Pipeline chunk scheduler: warm-first waiting order "
                "(window %s, aging %g tokens/s, host tier %s)",
                window_desc,
                aging,
                "counted" if self._tier is not None else "not counted",
            )

    def _tier_tokens(self, request: Any, local: int) -> int:
        """Prompt tokens past the local hit that the host tier holds; 0, and
        the tier no longer asked, once a lookup has failed."""
        try:
            return int(self._tier.host_tier_hit_tokens(request, local))
        except Exception:
            logger.exception(
                "Pipeline chunk scheduler: host-tier lookup failed; only the "
                "local prefix cache counts as cached from now on."
            )
            self._tier = None
            return 0

    def _uncached_tokens(self, request: Any, now: float) -> int:
        """Prompt tokens of a fresh waiting request that miss the local
        prefix cache (and, with TPU_PP_WARM_FIRST_TIER, the host tier), from a
        lookup at most _WARM_FIRST_LOOKUP_TTL_S old."""
        cached = self._uncached.get(request.request_id)
        if cached is not None and now - cached[1] < _WARM_FIRST_LOOKUP_TTL_S:
            return cached[0]
        warmth = self._prefix_warmth(request)
        if self._tier is not None:
            tier = self._tier_tokens(request, warmth)
            if tier > 0:
                self._wf_tier_hits += 1
                warmth += tier
        uncached = max(request.num_tokens - warmth, 0)
        self._uncached[request.request_id] = (uncached, now)
        self._wf_lookups += 1
        return uncached

    def _order_waiting(self, now: float) -> None:
        """Orders the first TPU_PP_WARM_FIRST_WINDOW waiting requests warm first
        (default 128; -1 scans the full waiting queue).

        Requests that resume work (``_started``) stay at the head in their
        order. The fresh ones follow, by uncached prompt tokens less
        TPU_PP_WARM_FIRST_AGING tokens per second waited, then by arrival.
        The aged key orders requests as arrival + uncached / rate does, so a
        request with U uncached tokens is passed only by requests that arrive
        less than U / rate seconds after it. ``running`` and
        ``kv_holding_waiting``, which the base scheduler drains first, are
        not touched; neither is the queue past the window.
        """
        queue = self.waiting
        if self.policy != SchedulingPolicy.FCFS or len(queue) < 2:
            return
        window = list(islice(queue, self._warm_first_window))
        started = [r for r in window if _started(r)]
        fresh = [r for r in window if not _started(r)]
        rate = self._warm_first_aging
        keys = {
            r.request_id: (
                self._uncached_tokens(r, now) - rate * max(now - r.arrival_time, 0.0),
                r.arrival_time,
            )
            for r in fresh
        }
        # Keep the lookups of the requests still in the window only.
        self._uncached = {req_id: self._uncached[req_id] for req_id in keys}
        fresh.sort(key=lambda r: keys[r.request_id])
        order = started + fresh
        self._wf_sorts += 1
        if any(a is not b for a, b in zip(order, window)):
            self._wf_reorders += 1
            for _ in window:
                queue.pop_request()
            # FCFSRequestQueue.prepend_requests reverses its argument.
            queue.prepend_requests(reversed(order))
        now_m = time.monotonic()
        if now_m - self._wf_log_t >= _WARM_FIRST_LOG_INTERVAL_S:
            self._wf_log_t = now_m
            head = order[0]
            logger.info(
                "Pipeline chunk scheduler warm-first: sorts=%d reordered=%d "
                "lookups=%d tier_hits=%d waiting=%d head_uncached=%s",
                self._wf_sorts,
                self._wf_reorders,
                self._wf_lookups,
                self._wf_tier_hits,
                len(queue),
                self._uncached[head.request_id][0] if not _started(head) else "started",
            )

    def schedule(self, throttle_prefills: bool = False):
        if self._warm_first:
            self._order_waiting(time.time())
        self._step_tokens = 0
        self._step_pairs = 0.0
        self._step_squares = 0.0
        self._step_prefill_tokens = 0
        self._step_schedule = 0
        self._seeded = set()
        self._step_limit = self._pending_step_limit()
        # Every running request past its prefill with a token left to
        # compute takes it this step; count them before the loop sizes any
        # prefill chunk. A request whose last chunk is still in flight has
        # nothing to compute yet.
        for request in self.running:
            computed = request.num_computed_tokens
            if computed >= _prefill_end(request) and request.num_tokens > computed:
                self._account(1, computed)
                self._seeded.add(request.request_id)
        return super().schedule(throttle_prefills)

    def step_estimate_ms(self, scheduler_output: Any) -> float | None:
        """Modelled time on one stage of the step the last ``schedule``
        produced, in ms; None until the cost model is installed."""
        if self._cost is None:
            return None
        pairs = squares = 0.0
        for req_id, tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            # schedule() already counted the step's tokens as computed.
            prefix = request.num_computed_tokens - tokens if request is not None else 0
            pairs += float(tokens) * prefix
            squares += float(tokens) * tokens
        buckets = self._cost.buckets
        total = scheduler_output.total_num_scheduled_tokens
        bucket = next((b for b in buckets if b >= total), buckets[-1])
        return self._cost.step_ms(bucket, pairs, squares)

    def _pending_step_limit(self) -> int | None:
        """The bucket a step is held to: the prefill work still to schedule
        spread over up to as many steps as the pipeline has stages, rounded
        up to a granule, at most a full step. Of those spreads the one that
        gets the work through the pipeline soonest is taken: a step of half
        the tokens that takes as long as the whole one only adds steps.
        None until the cost model is installed or when no prefill work is
        pending."""
        if self._cost is None:
            return None
        # TPU_PP_SPREAD_MAX_PREFIX (q38): requests this deep into their
        # prompt stay out of the spread and are cut by the time budget alone.
        # The cost model is fit at short prefixes and prices each spread with
        # no prefix at all, so it splits a long-context chunk into halves
        # that cost nearly as much as the whole (sparse attention at 100K of
        # context costs about the same for 3,200 and 6,400 tokens).
        max_prefix = int(envs.TPU_PP_SPREAD_MAX_PREFIX)
        pending = 0
        for request in (*self.running, *self.waiting):
            if max_prefix > 0 and request.num_computed_tokens >= max_prefix:
                continue
            pending += max(_prefill_end(request) - request.num_computed_tokens, 0)
        if pending == 0:
            return None
        granularity = self._cost.granularity
        buckets = self._cost.buckets
        limit, soonest = buckets[-1], math.inf
        for spread in range(1, self._pp_size + 1):
            share = -(-pending // spread)
            share = -(-share // granularity) * granularity
            share = min(buckets[-1], max(granularity, share))
            bucket = next(b for b in buckets if b >= share)
            # The last of the work leaves the pipeline a stage count minus
            # one steps after its step enters.
            steps = -(-pending // share) + self._pp_size - 1
            done_ms = steps * self._cost.step_ms(bucket, 0.0, float(share) * share)
            if done_ms < soonest:
                limit, soonest = share, done_ms
        return limit

    def _mamba_block_aligned_split(
        self,
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        if start >= _prefill_end(request):
            if request.request_id not in self._seeded:
                self._account(num_new_tokens, start)
            return num_new_tokens
        if self._cost is not None and num_new_tokens > 0:
            cap = self._cost.chunk(
                start,
                num_new_tokens,
                self._step_tokens,
                self._step_pairs,
                self._step_squares,
                self._step_limit,
                self._step_schedule,
            )
            if cap == 0 and self._step_prefill_tokens == 0:
                # The first prefill chunk of a step always advances.
                cap = min(num_new_tokens, self._cost.granularity)
            if cap == 0:
                self._waits += 1
            elif cap < num_new_tokens:
                self._clips += 1
                logger.debug(
                    "Pipeline chunk: request %s prefix %d, %d -> %d tokens "
                    "(step holds %d)",
                    request.request_id,
                    start,
                    num_new_tokens,
                    cap,
                    self._step_tokens,
                )
            num_new_tokens = cap
        if self._mamba_split and num_new_tokens > 0:
            num_new_tokens = super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        self._account(num_new_tokens, start)
        self._step_prefill_tokens += num_new_tokens
        if self._cost is not None:
            self._step_schedule += self._cost.pairs(num_new_tokens, start)
        return num_new_tokens


def profile_and_install_cost_model(scheduler: Any, executor: Any) -> None:
    """Time every stage's forwards and hand the scheduler the model."""
    if not isinstance(scheduler, TpuPipelineChunkScheduler):
        return
    reports = executor.collective_rpc("profile_pipeline_chunks")
    granularity = max(int(report["granularity"]) for report in reports)
    schedules = [
        tuple(int(x) for x in report["schedule"])
        for report in reports
        if report.get("schedule") is not None
    ]
    model = StepCostModel.fit(
        [[tuple(s) for s in report["samples"]] for report in reports],
        envs.TPU_PP_CHUNK_SLACK,
        granularity,
        # Each worker reports its kernel schedule as (pairs, bq, bkv).
        cast(tuple[int, int, int], min(schedules)) if schedules else None,
    )
    scheduler.install_cost_model(model)


def patch_engine_core_for_pp_chunks(vllm_config: Any) -> None:
    """Have the engine core time the stages once its executor and scheduler
    exist. Installed from the platform's config check (in-process engines)
    and from the engine-core process entry point; idempotent."""
    if not uses_dynamic_chunks(vllm_config):
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_pp_chunks_patch", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        profile_and_install_cost_model(self.scheduler, self.model_executor)

    EngineCore.__init__ = __init__
    EngineCore._tpu_pp_chunks_patch = True
