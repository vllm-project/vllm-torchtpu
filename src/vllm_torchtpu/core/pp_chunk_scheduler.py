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
"""Scheduler that sizes prefill chunks for a pipeline.

vLLM's scheduler fills every step up to a token budget and clips each
request's chunk at one point, ``_mamba_block_aligned_split``. This subclass
routes every request through that point, clips prefill chunks to the
step's limits (``core/pp_chunks.py``) and then applies the base scheduler's
mamba block alignment when the model has a mamba cache. Decode requests
count toward the step before any prefill chunk is sized, whatever their
position in the running queue.

Two policies set the limits, each behind its own env var. With lone-prompt
chunking (``TPU_PP_LONE_PROMPT_CHUNKS``), when the pending prefill work
cannot fill every stage with a full step, the step shrinks to that work
spread over the pipeline, so a lone prompt runs as one small step per stage
instead of trailing itself through the pipeline one full step at a time; a
busy server keeps full steps and first-come order. With equal-cost chunking
(``TPU_PP_EQUAL_COST_CHUNKS``), prefill chunks are clipped to what the
step's remaining time holds under a cost model the stages time at startup.
Either way a step holds at most as many (query block, KV block) attention
pairs as the batched attention kernel can schedule, which bounds a chunk
deep in a very long prompt so the kernel never overruns its SMEM. The
limits arrive after the engine core has built its executor and scheduler;
until then chunks are cut by token count alone.

On a Qwen3.5 MoE model (``RETENTION_ARCHITECTURES``) the scheduler also
keeps the GDN state snapshots a later turn hits on
(``core/hybrid_retention.py``) on a KV producer: the retention stops are
applied to a chunk after the step's limits have clipped it. Other hybrid
models keep no snapshots.

The quadratic attention cost and the prefix-aware admission it borrows from
``PrefixAwareScheduler`` are options (``core/pp_scheduler_options.py``), each
off by default, so a pipeline that sets none of them is scheduled as before.
"""

import math
import time
from itertools import islice
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import envs
from vllm_torchtpu.core import pp_scheduler_options, scheduler_options
from vllm_torchtpu.core.hybrid_retention import (
    HybridStateRetention,
    retention_wanted,
)
from vllm_torchtpu.core.pp_chunks import StepCostModel, chunk_pairs, step_limit
from vllm_torchtpu.core.prefix_aware_scheduler import (
    _TIER_NORMAL,
    _TIER_PREEMPTED,
    PrefixAwareScheduler,
    _is_decoding,
    memo_local_prefix_cache_hit,
    trim_resend_token_ids,
)
from vllm_torchtpu.core.scheduler_utils import kv_roles, run_decoders_first
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)

SCHEDULER_CLS = "vllm_torchtpu.core.pp_chunk_scheduler.TpuPipelineChunkScheduler"

# The architectures whose KV producers keep hybrid-state snapshots: the Qwen3.5
# MoE models, whose prefill runs used them. Other hybrid models keep the
# scheduler without retention.
RETENTION_ARCHITECTURES = frozenset(
    {"Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"}
)


# A request whose uncached blocks fit the free blocks while the head's do not
# ranks ahead of PrefixAwareScheduler's tiers.
_TIER_FREE_BLOCK_FIT = _TIER_PREEMPTED - 1


def _wait_seconds(request: Any, now: float) -> float:
    """Seconds since ``request.arrival_time``.

    ``arrival_time`` is the wall-clock time (``time.time()``) at which this
    engine's API server received the request; on a disaggregated prefill leg
    that is the proxy's prefill hop. It covers time in the waiting queue and
    time in ``running``, and does not reset on preemption.
    """
    return max(now - request.arrival_time, 0.0)


@runtime_checkable
class ExternalHitEstimator(Protocol):
    """A KV connector that estimates how many prompt tokens past a request's
    local hit a lookup would load from its external tier (the Raiden
    offloading connector and the TPU multi-connector)."""

    def estimate_external_hit_tokens(
        self, request: Any, num_local_tokens: int
    ) -> int: ...


@runtime_checkable
class TransferPoller(Protocol):
    """A KV connector that completes finished async loads when the scheduler
    polls it at the start of a step (the TPU multi-connector and the Raiden
    offloading connector)."""

    def poll_and_complete_transfers(self, scheduler: Any) -> Any: ...


@runtime_checkable
class DivergentHitResolver(Protocol):
    """A KV connector that decides how much of a diverged local hybrid hit to
    keep: None keeps the reconciled hit, otherwise the number of local tokens
    to keep (the TPU multi-connector and the Raiden offloading
    connector)."""

    def resolve_divergent_local_hit_tokens(
        self, request: Any, num_local: int, num_reconciled: int, block_size: int
    ) -> int | None: ...


def _retains_hybrid_state(vllm_config: Any) -> bool:
    """Whether this engine keeps hybrid-state snapshots: a KV producer
    (``retention_wanted``) serving one of ``RETENTION_ARCHITECTURES``."""
    return retention_wanted(vllm_config) and not RETENTION_ARCHITECTURES.isdisjoint(
        vllm_config.model_config.architectures or ()
    )


def uses_dynamic_chunks(vllm_config: Any) -> bool:
    """Whether this configuration sizes pipeline chunks; the platform
    installs the scheduler class only when it does."""
    return (
        vllm_config is not None
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.scheduler_config.scheduler_cls == SCHEDULER_CLS
    )


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


class TpuPipelineChunkScheduler(Scheduler):
    # The prefix-aware admission (option prefix_aware) borrows these from
    # PrefixAwareScheduler; none of them calls super().
    # The ones this class calls are typed Any: their self parameter is a
    # PrefixAwareScheduler, which this class is not.
    _MEMO_LOG_CAP = PrefixAwareScheduler._MEMO_LOG_CAP
    _MEMO_VERIFY_REUSES = PrefixAwareScheduler._MEMO_VERIFY_REUSES
    WINDOW = PrefixAwareScheduler.WINDOW
    _enable_memo_xstep: Any = PrefixAwareScheduler._enable_memo_xstep
    _memo_trim_log: Any = PrefixAwareScheduler._memo_trim_log
    _memo_still_exact = PrefixAwareScheduler._memo_still_exact
    _memo_get: Any = PrefixAwareScheduler._memo_get
    _memo_compute = PrefixAwareScheduler._memo_compute
    _get_prefix_cache_epoch: Any = PrefixAwareScheduler._get_prefix_cache_epoch
    _get_block_size: Any = PrefixAwareScheduler._get_block_size
    _prefix_warmth: Any = PrefixAwareScheduler._prefix_warmth
    _select_waiting_queue_for_scheduling: Any = (
        PrefixAwareScheduler._select_waiting_queue_for_scheduling
    )
    _promote_warmest = PrefixAwareScheduler._promote_warmest
    _maybe_pack_prompt_ids: Any = PrefixAwareScheduler._maybe_pack_prompt_ids

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lone_prompt = envs.TPU_PP_LONE_PROMPT_CHUNKS
        self._equal_cost = envs.TPU_PP_EQUAL_COST_CHUNKS
        self._cost: StepCostModel | None = None
        self._granularity: int | None = None
        # (pairs, bq, bkv) the batched attention kernel schedules per step.
        self._schedule: tuple[int, int, int] | None = None
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
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        # The trimmed token-id resend of the prefix-aware admission, until a
        # trim fails.
        self._trim_resend = True
        # kv_both (an offloading connector on an aggregated engine) has
        # neither role, as in PrefixAwareScheduler and retention.
        roles = kv_roles(self.vllm_config.kv_transfer_config)
        self._is_kv_producer = roles.producer
        self._is_kv_consumer = roles.consumer
        options = pp_scheduler_options.get(self.vllm_config)
        if options.quadratic_cost and not self._equal_cost:
            # The quadratic pricing runs on the equal-cost step cost model,
            # which the stages are timed for only under equal-cost chunking.
            raise ValueError(
                f"additional_config.{pp_scheduler_options.SECTION}.quadratic_cost "
                "prices prefill work with the equal-cost step cost model; it "
                "needs TPU_PP_EQUAL_COST_CHUNKS on"
            )
        self._quadratic_cost = options.quadratic_cost
        self._prefix_aware = options.prefix_aware
        # The aging rate is read only by the prefix-aware orders.
        self._paw_aging_tok_per_s = (
            options.paw_aging_tok_per_s if options.prefix_aware else 0
        )
        # The wall time of the schedule() call in progress (None between
        # steps), which the prefix-aware ranking ages requests against.
        self._step_wall_time: float | None = None
        self._srpf_running = options.prefix_aware
        self._promote_waiting = (
            options.prefix_aware
            and scheduler_options.get(self.vllm_config).promote_warmest
        )
        # The connector's host-tier estimate, read by _external_warmth.
        self._hit_estimator: ExternalHitEstimator | None = (
            self.connector if isinstance(self.connector, ExternalHitEstimator) else None
        )
        # The connector's async-load poll and divergent-hit decision, which
        # the prefix-aware admission uses.
        self._transfer_poller: TransferPoller | None = (
            self.connector if isinstance(self.connector, TransferPoller) else None
        )
        self._hit_resolver: DivergentHitResolver | None = (
            self.connector if isinstance(self.connector, DivergentHitResolver) else None
        )
        if self._prefix_aware and self.connector is not None:
            missing = [
                name
                for name, hook in (
                    ("poll_and_complete_transfers", self._transfer_poller),
                    ("resolve_divergent_local_hit_tokens", self._hit_resolver),
                )
                if hook is None
            ]
            if missing:
                logger.warning(
                    "Pipeline chunk scheduler: the KV connector %s has no %s; "
                    "the prefix-aware admission runs without it.",
                    type(self.connector).__name__,
                    " or ".join(missing),
                )
        self._retention: HybridStateRetention | None = None
        if _retains_hybrid_state(self.vllm_config):
            retention = HybridStateRetention(self.kv_cache_manager)
            if retention.enabled:
                self._retention = retention
        self._memo_xstep = False
        self._paw_sort = False
        if options.prefix_aware or options.quadratic_cost:
            # Both look up the prefix hit of every waiting request on every
            # step; the memo reuses a lookup until the cache changes under it.
            # The patch registry installs the prefix-cache epoch and change
            # log at engine start; this opts the scheduler's pool in.
            self._enable_memo_xstep()
        if options.prefix_aware:
            self._init_paw_sort()
        logger.info(
            "Pipeline chunk scheduler (lone-prompt %s, equal-cost %s): waiting "
            "for the chunk limits (hybrid_state_retention=%s, quadratic_cost=%s, "
            "prefix_aware=%s, paw_sort=%s, promote_warmest=%s, memo_xstep=%s)",
            self._lone_prompt,
            self._equal_cost,
            self._retention is not None,
            self._quadratic_cost,
            self._prefix_aware,
            self._paw_sort,
            self._promote_waiting,
            self._memo_xstep,
        )

    def _get_num_free_blocks(self) -> int:
        """Free HBM blocks, with the Mamba blocks hybrid-state retention pins
        counted as free: retention evicts them when an allocation needs
        them."""
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if self._retention is not None:
            free_blocks += self._retention.num_pinned_blocks()
        return free_blocks

    def _quad_cost(self, tokens: int, prefix: int) -> float:
        """Quadratic FLOPs-equivalent token cost under the step cost model
        (option quadratic_cost); the token count otherwise."""
        if self._cost is not None and self._quadratic_cost:
            return self._cost.effective_tokens(tokens, prefix)
        return float(max(tokens, 0))

    def _aged_prefill_key(self, request: Any, now: float) -> float:
        """SRPF key of a running prefill: remaining (effective) tokens minus
        paw_aging_tok_per_s x seconds since arrival."""
        rem = max(_prefill_end(request) - request.num_computed_tokens, 0)
        eff = self._quad_cost(rem, request.num_computed_tokens)
        rate = self._get_paw_aging_rate()
        if rate <= 0:
            return eff
        return eff - rate * _wait_seconds(request, now)

    # ------------------------------------------------------------------
    # Waiting-queue sort of the prefix-aware admission, on a KV producer,
    # together with the aging rate (option paw_aging_tok_per_s): without an
    # aging rate a large prompt could be overtaken without bound.
    #
    # ``_promote_warmest`` moves one request per queue to the head. With the
    # sort on, ``schedule()`` also stable-sorts the first _PAW_SORT_WINDOW
    # requests of ``waiting`` before the base scheduler admits from it, so
    # every admission of the step follows the order. The key is the
    # request's estimated uncached prompt tokens (prompt minus local HBM hit
    # minus a loadable external hit), priced quadratically under the option
    # quadratic_cost, minus the aging rate x seconds waited, which bounds how
    # long a large prompt can be overtaken. Requests that already hold
    # computed tokens or were preempted go first, and requests whose uncached
    # tokens do not fit the free HBM blocks go last, so they cannot stall the
    # head.
    # ------------------------------------------------------------------
    _PAW_SORT_LOG_INTERVAL_S = 60.0
    _PAW_SORT_WINDOW = 128

    def _init_paw_sort(self) -> None:
        self._paw_sort = self._is_kv_producer
        self._paw_sort_window = self._PAW_SORT_WINDOW
        self._paw_sort_steps = 0
        self._paw_sort_reordered = 0
        self._paw_sort_moved = 0
        self._paw_sort_head_cold = 0
        self._paw_sort_log_t = time.monotonic()
        if self._paw_sort:
            logger.info(
                "Pipeline chunk scheduler: waiting-queue SRPF sort ACTIVE "
                "(window=%d, aging_tok_per_s=%g)",
                self._paw_sort_window,
                self._get_paw_aging_rate(),
            )

    def _sort_waiting_queue(self) -> None:
        """Stable-sort the first _PAW_SORT_WINDOW waiting requests by
        ``_paw_sort_key`` in place."""
        queue = self.waiting
        if self.policy != SchedulingPolicy.FCFS or len(queue) < 2:
            return
        window = list(islice(queue, self._paw_sort_window))
        free_blocks = self._get_num_free_blocks()
        # The sort runs on a KV producer only (_init_paw_sort), whose async
        # loads reserve blocks before the host-tier hit can land.
        free_blocks_for_ext = max(
            0, free_blocks - self._inflight_prefill_reserved_blocks()
        )
        # The step's wall time, the clock _promote_warmest ages on.
        now = self._now()
        keys = [
            self._paw_sort_key(r, now, free_blocks, free_blocks_for_ext) for r in window
        ]
        order = sorted(range(len(window)), key=lambda i: (keys[i], i))
        self._paw_sort_steps += 1
        head = window[order[0]]
        self._paw_sort_head_cold += max(
            int(head.num_tokens) - self._prefix_warmth(head), 0
        )
        if order != list(range(len(window))):
            self._paw_sort_reordered += 1
            self._paw_sort_moved += sum(1 for pos, i in enumerate(order) if pos != i)
            # The window is the queue's head: pop it and put it back sorted.
            for _ in window:
                queue.pop_request()
            # prepend_requests reverses its argument (deque.extendleft).
            queue.prepend_requests([window[i] for i in reversed(order)])
        now_m = time.monotonic()
        if now_m - self._paw_sort_log_t >= self._PAW_SORT_LOG_INTERVAL_S:
            self._paw_sort_log_t = now_m
            logger.info(
                "paw-sort telemetry: steps=%d reordered=%d moved=%d "
                "mean_head_cold_tokens=%.0f waiting=%d",
                self._paw_sort_steps,
                self._paw_sort_reordered,
                self._paw_sort_moved,
                self._paw_sort_head_cold / max(self._paw_sort_steps, 1),
                len(queue),
            )

    def _paw_sort_key(
        self,
        request: Any,
        now: float,
        free_blocks: int,
        free_blocks_for_ext: int,
    ) -> tuple[int, int, float, float]:
        """Sort key of the waiting-queue sort, by (effective) uncached
        tokens; lower is admitted first."""
        local_warmth = self._prefix_warmth(request)
        num_tokens = int(request.num_tokens)
        hbm_cold_tokens = max(num_tokens - local_warmth, 0)
        block_size = self._get_block_size()
        fits_hbm = math.ceil(hbm_cold_tokens / block_size) <= free_blocks
        ext_warmth = self._external_warmth(request, local_warmth)
        if ext_warmth > 0 and not (
            fits_hbm and math.ceil(ext_warmth / block_size) <= free_blocks_for_ext
        ):
            ext_warmth = 0
        warmth = local_warmth + ext_warmth
        uncached = max(num_tokens - warmth, 0)
        eff_uncached = self._quad_cost(uncached, warmth)
        aged = eff_uncached - self._get_paw_aging_rate() * _wait_seconds(request, now)
        started = request.num_preemptions > 0 or request.num_computed_tokens > 0
        return (0 if fits_hbm else 1, 0 if started else 1, aged, request.arrival_time)

    def _get_paw_aging_rate(self) -> float:
        """Uncached prompt tokens credited per second a request has waited
        (option paw_aging_tok_per_s; 0 turns the aging off)."""
        return float(self._paw_aging_tok_per_s)

    def _now(self) -> float:
        """The wall time of the schedule() call in progress, so every ranking
        of a step ages requests against one clock; time.time() between
        steps."""
        now = self._step_wall_time
        return time.time() if now is None else now

    def _external_warmth(self, request: Any, local_warmth: int) -> int:
        """Prompt tokens past ``local_warmth`` the connector expects to load
        from its external tier (``_hit_estimator``); 0 on a KV consumer,
        without an estimating connector, or with less than a block past the
        local hit. A failing estimate counts as 0; it is logged once per
        exception type (``warning_once`` keys on the arguments, so the request
        and the exception text go to the debug log instead)."""
        if self._is_kv_consumer:
            return 0
        estimator = self._hit_estimator
        if estimator is None:
            return 0
        if request.num_tokens - local_warmth < self._get_block_size():
            return 0
        try:
            estimate = estimator.estimate_external_hit_tokens(request, local_warmth)
            return max(0, int(estimate or 0))
        except Exception as exc:
            logger.warning_once(
                "estimate_external_hit_tokens raised %s; a request whose "
                "estimate fails is ranked by its local prefix hit only.",
                type(exc).__name__,
            )
            logger.debug(
                "estimate_external_hit_tokens failed for request %s: %r",
                request.request_id,
                exc,
            )
            return 0

    def _rank_and_warmth(
        self,
        request: Any,
        free_blocks_for_fit: int | None = None,
        now: float | None = None,
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth) by (effective) uncached tokens."""
        local_warmth = self._prefix_warmth(request)
        hbm_cold_tokens = max(request.num_tokens - local_warmth, 0)
        hbm_eff_cost = self._quad_cost(hbm_cold_tokens, local_warmth)
        rate = self._get_paw_aging_rate()
        aging = 0.0
        if rate > 0:
            aging = rate * _wait_seconds(request, self._now() if now is None else now)
        if self._is_kv_consumer:
            tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
            return (tier, hbm_eff_cost - aging, request.arrival_time), local_warmth
        if (
            free_blocks_for_fit is not None
            and local_warmth > 0
            and math.ceil(hbm_cold_tokens / self._get_block_size())
            <= free_blocks_for_fit
        ):
            return (
                (
                    _TIER_FREE_BLOCK_FIT,
                    hbm_eff_cost - aging,
                    request.arrival_time,
                ),
                local_warmth,
            )
        if request.num_preemptions > 0:
            return (
                (_TIER_PREEMPTED, hbm_eff_cost - aging, request.arrival_time),
                local_warmth,
            )
        if local_warmth > 0:
            return (
                (_TIER_NORMAL, hbm_eff_cost - aging, request.arrival_time),
                local_warmth,
            )
        ext_warmth = self._external_warmth(request, 0)
        if ext_warmth > 0:
            ext_cold_tokens = max(request.num_tokens - ext_warmth, 0)
            ext_eff_cost = self._quad_cost(ext_cold_tokens, ext_warmth)
            return (
                (_TIER_NORMAL + 1, ext_eff_cost - aging, request.arrival_time),
                ext_warmth,
            )
        return (
            (_TIER_NORMAL + 2, hbm_eff_cost - aging, request.arrival_time),
            0,
        )

    def _free_blocks_for_fit(self, head: Any) -> int | None:
        """The free HBM blocks when the uncached tokens of ``head`` (the
        queue's head) do not fit them, else None: the input of the fit tier
        of ``_rank_and_warmth``, for the borrowed ``_promote_warmest``."""
        if self._is_kv_consumer:
            return None
        head_local_warmth = self._prefix_warmth(head)
        free_blocks = self._get_num_free_blocks()
        head_cold_tokens = max(head.num_tokens - head_local_warmth, 0)
        if free_blocks < math.ceil(head_cold_tokens / self._get_block_size()):
            return free_blocks
        return None

    def _make_cached_request_data(self, *args, **kwargs):
        cached = super()._make_cached_request_data(*args, **kwargs)
        if not self._prefix_aware:
            return cached
        if self._trim_resend and getattr(cached, "all_token_ids", None):
            try:
                trim_resend_token_ids(cached)
            except Exception:
                logger.exception(
                    "Pipeline chunk scheduler: all_token_ids trim failed; disabling it."
                )
                self._trim_resend = False
        return cached

    def _get_local_prefix_cache_hit(
        self, request: Any, max_cache_hit_length: int | None = None
    ) -> tuple["KVCacheBlocks", int, int, bool]:
        if not self._prefix_aware:
            return super()._get_local_prefix_cache_hit(request, max_cache_hit_length)
        hit = memo_local_prefix_cache_hit(self, request, max_cache_hit_length)
        if hit is not None:
            request._tpu_local_hit_tokens = hit[1]
            return hit
        resolver = self._hit_resolver
        blocks, num_local, shared_prefix_boundary, hit_diverged = (
            super()._get_local_prefix_cache_hit(request, max_cache_hit_length)
        )
        if hit_diverged and resolver is not None:
            kvm = self.kv_cache_manager
            reconciled_blocks, num_reconciled, reconciled_boundary = (
                kvm.get_computed_blocks(request, max_cache_hit_length)
            )
            block_size = self._get_block_size()
            target_local = resolver.resolve_divergent_local_hit_tokens(
                request, num_local, num_reconciled, block_size
            )
            if target_local is None:
                request._tpu_local_hit_tokens = num_reconciled
                return (
                    reconciled_blocks,
                    num_reconciled,
                    reconciled_boundary,
                    False,
                )
            if target_local <= 0:
                request._tpu_local_hit_tokens = 0
                return (
                    kvm.empty_kv_cache_blocks,
                    0,
                    reconciled_boundary,
                    False,
                )
            if target_local < num_local:
                blocks = kvm.truncate_computed_blocks(blocks, target_local)
                num_local = target_local
        request._tpu_local_hit_tokens = num_local
        return blocks, num_local, shared_prefix_boundary, hit_diverged

    def install_chunk_limits(
        self, granularity: int, schedule: tuple[int, int, int] | None
    ) -> None:
        """Cut chunks in ``granularity`` tokens and keep a step within the
        attention ``schedule`` capacity, without a cost model."""
        self._granularity = granularity
        self._schedule = schedule
        self._pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        text = f"granularity {granularity} tokens"
        if schedule is not None:
            pairs, bq, bkv = schedule
            text += f", at most {pairs} attention pairs ({bq}x{bkv}) per step"
        logger.info("Pipeline chunk scheduler: %s", text)

    def install_cost_model(self, model: StepCostModel) -> None:
        self._cost = model
        self._granularity = model.granularity
        self._schedule = model.schedule
        self._pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        logger.info("Pipeline chunk scheduler: %s", model.describe())

    def _account(self, tokens: int, prefix: int) -> None:
        self._step_tokens += tokens
        self._step_pairs += float(tokens) * prefix
        self._step_squares += float(tokens) * tokens

    def _retention_failed(self) -> None:
        """Turns retention off after one of its calls raised: it releases
        the blocks it holds and its hooks fall through to vLLM's methods."""
        logger.exception(
            "Pipeline chunk scheduler: hybrid state retention failed; disabling it."
        )
        retention, self._retention = self._retention, None
        if retention is None:
            return
        try:
            retention.shutdown()
        except Exception:
            # The blocks it still pins stay pinned; scheduling goes on.
            logger.exception(
                "Pipeline chunk scheduler: shutting hybrid state retention down "
                "failed; its pinned blocks stay pinned."
            )

    def _connector_finished(self, request: Any):
        result = super()._connector_finished(request)
        if result[0] and self._retention is not None:
            try:
                self._retention.pin_running((request,))
            except Exception:
                self._retention_failed()
        return result

    def _free_request_blocks(self, request: Any):
        retention = self._retention
        if retention is None:
            return super()._free_request_blocks(request)
        try:
            attn_blocks = retention.snapshot_attention_blocks(request)
        except Exception:
            self._retention_failed()
            return super()._free_request_blocks(request)
        result = super()._free_request_blocks(request)
        try:
            retention.on_request_freed(request, attn_blocks)
        except Exception:
            self._retention_failed()
        return result

    def schedule(self, throttle_prefills: bool = False):
        prefix_aware = self._prefix_aware
        poller = self._transfer_poller
        if prefix_aware and poller is not None:
            # Completed async loads promote their requests this step.
            try:
                poller.poll_and_complete_transfers(self)
            except Exception:
                logger.exception(
                    "Pipeline chunk scheduler: poll_and_complete_transfers failed."
                )
        if self._retention is not None:
            try:
                self._retention.pin_running(self.running)
            except Exception:
                self._retention_failed()
            else:
                self._retention.begin_step()
        if self._memo_xstep:
            self._memo_trim_log()
        now_wall = time.time()
        self._step_wall_time = now_wall
        if self._srpf_running and len(self.running) > 1:
            # Decoders first, then running prefills shortest remaining (aged)
            # first.
            run_decoders_first(
                self.running,
                _is_decoding,
                lambda r: self._aged_prefill_key(r, now_wall),
            )
        if self._paw_sort:
            try:
                self._sort_waiting_queue()
            except Exception:
                logger.exception(
                    "Pipeline chunk scheduler: waiting-queue sort failed; turning "
                    "the sort off."
                )
                self._paw_sort = False
        if self._promote_waiting:
            try:
                self._select_waiting_queue_for_scheduling()
            except Exception:
                logger.exception(
                    "Pipeline chunk scheduler: warmest promotion failed; disabling it."
                )
                self._promote_waiting = False
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
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self._step_wall_time = None
        if prefix_aware and getattr(output, "scheduled_new_reqs", None):
            self._maybe_pack_prompt_ids(output)
        return output

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

    def _waiting_warmth(self, request: Any) -> int:
        """Prompt tokens of a waiting request that need no prefill: its
        computed tokens, or under quadratic_cost its prefix hit."""
        computed = request.num_computed_tokens
        if computed > 0 or self._cost is None or not self._quadratic_cost:
            return computed
        if not self.kv_cache_manager.enable_caching:
            return 0
        local_warmth = self._prefix_warmth(request)
        ext_warmth = self._external_warmth(request, local_warmth)
        return max(0, min(local_warmth + ext_warmth, _prefill_end(request)))

    def _pending_step_limit(self) -> int | None:
        """The bucket a step is held to under lone-prompt chunking: the
        prefill work still to schedule spread over up to as many steps as the
        pipeline has stages, rounded up to a granule, at most a full step.
        With a cost model, of those spreads the one that gets the work
        through the pipeline soonest is taken: a step of half the tokens
        that takes as long as the whole one only adds steps. Under
        quadratic_cost each spread is also priced by its share of the
        pending (chunk, prefix) pairs, with a waiting request's prefix hit
        counted as done. Without a cost model the work is spread over every
        stage (`step_limit`). None with lone-prompt chunking off, until the
        limits are installed, or when no prefill work is pending."""
        granularity = self._granularity
        if not self._lone_prompt or granularity is None:
            return None
        quadratic = self._quadratic_cost
        pending = 0
        pending_pairs = 0.0
        for request in self.running:
            rem = max(_prefill_end(request) - request.num_computed_tokens, 0)
            pending += rem
            pending_pairs += float(rem) * request.num_computed_tokens
        # Requests holding KV blocks (including async KV loads) wait in a
        # separate queue. Count both queues before the base scheduler scans
        # them. deferred_waiting tracks a subset of these requests, so adding
        # it separately would double-count their remaining prompt work.
        for queue in (self.waiting, self.kv_holding_waiting):
            for request in queue:
                warmth = self._waiting_warmth(request)
                rem = max(_prefill_end(request) - warmth, 0)
                pending += rem
                pending_pairs += float(rem) * warmth
        if self._cost is None:
            return step_limit(
                pending,
                self._pp_size,
                granularity,
                self.scheduler_config.max_num_batched_tokens,
            )
        if pending == 0:
            return None
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
            share_pairs = (
                pending_pairs * (min(share, pending) / pending) if quadratic else 0.0
            )
            step_ms = self._cost.step_ms(bucket, share_pairs, float(share) * share)
            done_ms = steps * step_ms
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
        # A waiting request with one prompt token left after its prefix hit
        # arrives padded to 1 + num_spec_tokens rows of a verify step. A
        # pipeline runs MTP over the prompt only and never verifies
        # (TPUModelRunner._prefill_only_mtp), so the chunk ends at the last
        # prompt token and the base scheduler drops the padding.
        num_new_tokens = min(num_new_tokens, request.num_tokens - start)
        granularity = self._granularity
        if granularity is not None and num_new_tokens > 0:
            cap = self._chunk_cap(start, num_new_tokens, granularity)
            if cap == 0 and self._step_prefill_tokens == 0:
                # The first prefill chunk of a step always advances.
                cap = min(num_new_tokens, granularity)
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
        if num_new_tokens > 0:
            num_new_tokens = self._retention_split(
                request,
                start,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        self._account(num_new_tokens, start)
        self._step_prefill_tokens += num_new_tokens
        self._step_schedule += chunk_pairs(self._schedule, num_new_tokens, start)
        return num_new_tokens

    def _chunk_cap(self, start: int, num_new_tokens: int, granularity: int) -> int:
        """Largest chunk of a prefill behind ``start`` computed tokens that
        the step's limits hold: the cost model's time budget with equal-cost
        chunking, else the lone-prompt step limit; either way within the
        attention schedule capacity and leaving the step total on a granule
        multiple when clipped. 0 when nothing fits."""
        if self._cost is not None:
            return self._cost.chunk(
                start,
                num_new_tokens,
                self._step_tokens,
                self._step_pairs,
                self._step_squares,
                self._step_limit,
                self._step_schedule,
            )
        cap = num_new_tokens
        if self._step_limit is not None:
            cap = min(cap, self._step_limit - self._step_tokens)
        if self._schedule is not None:
            room = self._schedule[0] - self._step_schedule
            while cap > 0 and chunk_pairs(self._schedule, cap, start) > room:
                # Back off a granule, keeping the step total aligned.
                total = self._step_tokens + cap - 1
                cap = total - total % granularity - self._step_tokens
        return max(cap, 0)

    def _retention_split(
        self,
        request: Any,
        start: int,
        num_new_tokens: int,
        num_new_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> int:
        """Ends the clipped chunk where the GDN state must be written: at the
        base scheduler's block alignment and the retention stops."""
        retention = self._retention
        if retention is not None:
            try:
                retention.prepare_prefill_split(request, start, num_new_tokens)
            except Exception:
                self._retention_failed()
                retention = None
        aligned = num_new_tokens
        if self._mamba_split:
            aligned = super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        if retention is not None:
            try:
                aligned = retention.adjust_prefill_split(request, start, aligned)
            except Exception:
                self._retention_failed()
        return aligned


def install_chunk_limits_from_workers(scheduler: Any, executor: Any) -> None:
    """Hand the scheduler its chunk limits from every stage: the largest
    granule, the smallest attention schedule capacity and, with equal-cost
    chunking, a step cost model fit to each stage's timed forwards."""
    if not isinstance(scheduler, TpuPipelineChunkScheduler):
        return
    method = (
        "profile_pipeline_chunks" if scheduler._equal_cost else "pipeline_chunk_limits"
    )
    reports = executor.collective_rpc(method)
    granularity = max(int(report["granularity"]) for report in reports)
    schedules = [
        tuple(int(x) for x in report["schedule"])
        for report in reports
        if report.get("schedule") is not None
    ]
    # Each worker reports its kernel schedule as (pairs, bq, bkv).
    schedule = cast(tuple[int, int, int], min(schedules)) if schedules else None
    if not scheduler._equal_cost:
        scheduler.install_chunk_limits(granularity, schedule)
        return
    model = StepCostModel.fit(
        [[tuple(s) for s in report["samples"]] for report in reports],
        envs.TPU_PP_CHUNK_SLACK,
        granularity,
        schedule,
    )
    scheduler.install_cost_model(model)


def patch_engine_core_for_pp_chunks(vllm_config: Any) -> None:
    """Have the engine core collect the chunk limits once its executor and
    scheduler exist. Installed from the platform's config check (in-process
    engines) and from the engine-core process entry point; idempotent."""
    if not uses_dynamic_chunks(vllm_config):
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_pp_chunks_patch", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        install_chunk_limits_from_workers(self.scheduler, self.model_executor)

    EngineCore.__init__ = __init__
    EngineCore._tpu_pp_chunks_patch = True
