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

With the option ``hybrid_state_retention`` (which the Qwen3.5 config hook
sets on the measured prefill legs) the scheduler also keeps the GDN state
snapshots a later turn hits on (``core/hybrid_retention.py``) on a KV
producer: the retention stops are applied to a chunk after the step's
limits have clipped it. Without it no snapshots are kept.

The quadratic attention cost and the prefix-aware admission it shares with
``PrefixAwareScheduler`` (``PrefixAwareAdmission``) are options
(``core/pp_scheduler_options.py``), each off by default, so a pipeline that
sets none of them is scheduled as before.
"""

import math
import time
from itertools import islice
from typing import TYPE_CHECKING, Any, cast

from vllm.v1.core.sched.request_queue import SchedulingPolicy

from vllm_torchtpu import envs
from vllm_torchtpu.core import pp_scheduler_options, scheduler_options
from vllm_torchtpu.core.hybrid_retention import (
    HybridStateRetention,
    pinned_block_cap,
    retention_wanted,
)
from vllm_torchtpu.core.pp_chunks import StepCostModel, chunk_pairs, step_limit
from vllm_torchtpu.core.prefix_aware_scheduler import (
    _TIER_NORMAL,
    _TIER_PREEMPTED,
    PrefixAwareAdmission,
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


def uses_dynamic_chunks(vllm_config: Any) -> bool:
    """Whether this configuration sizes pipeline chunks; the platform
    installs the scheduler class only when it does."""
    return (
        vllm_config is not None
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.scheduler_config.scheduler_cls == SCHEDULER_CLS
    )


def pipeline_chunks_off_reason(vllm_config: Any) -> str | None:
    """Why the platform leaves pipeline chunk sizing off on this pipeline
    engine (and does not install this scheduler), or None when it installs
    it: both chunking policies off, another scheduler class, the DP
    scheduler, or chunked prefill off."""
    scheduler_config = vllm_config.scheduler_config
    if not (envs.TPU_PP_LONE_PROMPT_CHUNKS or envs.TPU_PP_EQUAL_COST_CHUNKS):
        return "both chunking policies are off"
    if scheduler_config.scheduler_cls not in (None, SCHEDULER_CLS):
        return f"scheduler {scheduler_config.scheduler_cls} is configured"
    if envs.DP_SCHED_ENABLED and vllm_config.parallel_config.data_parallel_size > 1:
        return "the DP scheduler is enabled"
    if not scheduler_config.enable_chunked_prefill:
        return "chunked prefill is disabled"
    return None


def check_options(vllm_config: Any) -> None:
    """Parse ``additional_config["tpu_pp_scheduler"]`` at start-up, before
    the model loads: the platform's config check calls this for every engine,
    after it has decided whether to install this scheduler.

    Raises ``ValueError`` for a malformed section; for a section set on an
    engine that does not run this scheduler (``uses_dynamic_chunks`` is
    false), where it would be ignored; for ``quadratic_cost`` without
    equal-cost chunking, whose step cost model it prices with; and for
    ``hybrid_state_retention`` off a KV producer.
    """
    options = pp_scheduler_options.get(vllm_config)
    section = pp_scheduler_options.SECTION
    additional = vllm_config.additional_config
    if (
        isinstance(additional, dict)
        and section in additional
        and not uses_dynamic_chunks(vllm_config)
    ):
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            why = pipeline_chunks_off_reason(vllm_config)
        else:
            why = "it runs only with more than one pipeline stage"
        raise ValueError(
            f"additional_config.{section} sets options of the pipeline chunk "
            f"scheduler, which this engine does not run: {why}"
        )
    if options.quadratic_cost and not envs.TPU_PP_EQUAL_COST_CHUNKS:
        # The stages are timed for the step cost model only under equal-cost
        # chunking.
        raise ValueError(
            f"additional_config.{section}.quadratic_cost prices prefill work "
            "with the equal-cost step cost model; it needs "
            "TPU_PP_EQUAL_COST_CHUNKS on"
        )
    if options.hybrid_state_retention and not retention_wanted(vllm_config):
        raise ValueError(
            f"additional_config.{section}.hybrid_state_retention keeps the "
            "snapshots a later turn hits on a KV producer (a prefill leg); "
            "this engine's kv_role is not kv_producer"
        )


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


class TpuPipelineChunkScheduler(PrefixAwareAdmission):
    # The prefix-aware admission (option prefix_aware) is the one
    # PrefixAwareScheduler runs (PrefixAwareAdmission); this scheduler ranks
    # its own promotion window (_rank_window).

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
        # The platform's config check (check_options) has already refused
        # a section this engine cannot run.
        options = pp_scheduler_options.get(self.vllm_config)
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
        self._retention: HybridStateRetention | None = None
        # The platform's check refuses the option off a KV producer.
        if options.hybrid_state_retention and self._is_kv_producer:
            # Retention's cross-turn pins, capped by the pool's size.
            pin_cap = pinned_block_cap(self.kv_cache_manager.block_pool.num_gpu_blocks)
            if pin_cap > 0:
                retention = HybridStateRetention(
                    self.kv_cache_manager, max_pinned_blocks=pin_cap
                )
                if retention.enabled:
                    self._retention = retention
                elif retention.off_reason is not None:
                    logger.warning_once(
                        "Pipeline chunk scheduler: hybrid_state_retention is "
                        "on, but this engine has %s; retention stays off.",
                        retention.off_reason,
                    )
            else:
                logger.warning(
                    "Pipeline chunk scheduler: a pool of %d blocks has no room "
                    "for hybrid-state retention (its pin cap is 0 blocks); "
                    "retention stays off.",
                    self.kv_cache_manager.block_pool.num_gpu_blocks,
                )
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
            "for the chunk limits (hybrid_state_retention=%s, max_pinned_blocks=%s, "
            "quadratic_cost=%s, prefix_aware=%s, paw_sort=%s, promote_warmest=%s, "
            "memo_xstep=%s)",
            self._lone_prompt,
            self._equal_cost,
            self._retention is not None,
            None if self._retention is None else self._retention.max_pinned_blocks,
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
    # request's uncached prompt tokens (prompt minus its HBM prefix hit),
    # priced quadratically under the option quadratic_cost, minus the aging
    # rate x seconds waited, which bounds how long a large prompt can be
    # overtaken. Requests that already hold computed tokens or were
    # preempted go first, and requests whose admission takes more blocks
    # than are free (``_blocks_to_admit``) go last, so they cannot stall the
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
        # The step's wall time, the clock _promote_warmest ages on.
        now = self._now()
        keys = [self._paw_sort_key(r, now, free_blocks) for r in window]
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
    ) -> tuple[int, int, float, float]:
        """Sort key of the waiting-queue sort, by (effective) uncached
        tokens; lower is admitted first."""
        warmth = self._prefix_warmth(request)
        fits_hbm = self._blocks_to_admit(request) <= free_blocks
        uncached = max(int(request.num_tokens) - warmth, 0)
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
            and self._blocks_to_admit(request) <= free_blocks_for_fit
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
        return (
            (_TIER_NORMAL + 1, hbm_eff_cost - aging, request.arrival_time),
            0,
        )

    def _rank_window(
        self, window: list[Any]
    ) -> list[tuple[tuple[int, float, float], int]]:
        """The promotion's ranking: every request aged to the step time, and
        the rest of the window ranked against the free blocks when the head
        does not fit them (``_free_blocks_for_fit``)."""
        now = self._now()
        head = window[0]
        head_scored = self._rank_and_warmth(head, now=now)
        free_blocks_for_fit = self._free_blocks_for_fit(head)
        return [head_scored] + [
            self._rank_and_warmth(r, free_blocks_for_fit=free_blocks_for_fit, now=now)
            for r in window[1:]
        ]

    def _free_blocks_for_fit(self, head: Any) -> int | None:
        """The free HBM blocks when admitting ``head`` (the queue's head)
        takes more of them than are free (``_blocks_to_admit``), else None:
        the input of the fit tier of ``_rank_and_warmth`` in the promotion
        (``_rank_window``)."""
        if self._is_kv_consumer:
            return None
        free_blocks = self._get_num_free_blocks()
        if free_blocks < self._blocks_to_admit(head):
            return free_blocks
        return None

    def _blocks_to_admit(self, request: Any) -> int:
        """Blocks admitting all of ``request``'s uncached prompt takes from the
        free blocks, counted as ``KVCacheManager.allocate_slots()`` counts
        them (and ``SrptPrefillScheduler._kv_fits_hold``): the coordinator's
        count over every KV cache group, which includes the prefix hit's
        blocks that sit in the free queue and the speculative lookahead
        slots, plus the watermark.

        The count is made as ``allocate_slots`` makes its whole-sequence check,
        with ``apply_admission_cap=True``: without it, vLLM's Mamba "align"
        manager records the request's prefill checkpoint for the counted
        length, which for a request whose async KV load is in flight
        (``kv_holding_waiting``) points past the blocks it holds and fails
        vLLM's ``cache_blocks`` when the load completes. Otherwise the flag
        changes only sliding-window and chunked-local groups' counts, which it
        caps as vLLM's whole-sequence check does.
        vLLM keeps the watermark free for a waiting request only once another
        request is scheduled in the step, and it schedules ``running`` first,
        so it is added when a request is running."""
        kvm = self.kv_cache_manager
        warmth, hit_blocks = self._prefix_hit(request)
        if hit_blocks is None:
            hit_blocks = kvm.empty_kv_cache_blocks.blocks
        num_tokens = min(request.num_tokens, self.max_model_len)
        blocks = kvm.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(num_tokens + self.num_lookahead_tokens, self.max_model_len),
            new_computed_blocks=hit_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=warmth,
            num_local_computed_tokens=warmth,
            num_tokens_main_model=num_tokens,
            apply_admission_cap=True,
            prefill_end=_prefill_end(request),
        )
        return blocks + (kvm.watermark_blocks if self.running else 0)

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
        """With prefix_aware, the admission lookup is answered from a
        still-exact memo when it can be (as in PrefixAwareScheduler)."""
        if self._prefix_aware:
            hit = memo_local_prefix_cache_hit(self, request, max_cache_hit_length)
            if hit is not None:
                return hit
        return super()._get_local_prefix_cache_hit(request, max_cache_hit_length)

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
        return max(0, min(self._prefix_warmth(request), _prefill_end(request)))

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
