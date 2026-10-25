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
time; a busy server keeps full steps and first-come order. The stages advance
in lockstep, so while a larger step is still in the pipeline every step lasts
at least as long as it does: the spreads are costed against the steps in
flight, which the scheduler tracks from its own steps and from the pushes and
settles the engine sends between them (``pipeline_advanced``). A step also
holds at most as many (query block, KV block) attention pairs as the batched
attention kernel can schedule, which bounds a chunk deep in a very long
prompt so the kernel never overruns its SMEM. The cost model
arrives after the engine core has built its executor and scheduler; until
then chunks are cut by token count alone.

On a hybrid attention + GDN model the scheduler also keeps the GDN state
snapshots a later turn hits on (``core/hybrid_retention.py``) and writes
them inside a chunk rather than at an extra step end
(``core/single_step_prefill.py``), the way ``PrefixAwareScheduler`` does for
a single-stage engine: the retention stops are applied to a chunk after the
time budget has clipped it.
"""

import math
import os
import time
from typing import Any, cast

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import envs
from vllm_torchtpu.core import single_step_prefill
from vllm_torchtpu.core.hybrid_retention import HybridStateRetention
from vllm_torchtpu.core.pp_chunks import StepCostModel
from vllm_torchtpu.core.prefix_aware_scheduler import (
    _TIER_FREE_BLOCK_FIT,
    _TIER_NORMAL,
    _TIER_PREEMPTED,
    PrefixAwareScheduler,
    _is_decoding,
    _read_paw_aging_knobs,
    _wait_seconds,
    trim_resend_token_ids,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_CLS = "vllm_torchtpu.core.pp_chunk_scheduler.TpuPipelineChunkScheduler"
# ix-unified-port: the final lineage's flavour, selected with TPU_PP_SCHEDULER=final.
SCHEDULER_CLS_FINAL = (
    "vllm_torchtpu.core.pp_chunk_scheduler_final.TpuPipelineChunkSchedulerFinal"
)


def pp_scheduler_cls() -> str:
    """The pipeline chunk scheduler class to install: unified's unless
    TPU_PP_SCHEDULER=final (ix-unified-port)."""
    flavour = os.environ.get("TPU_PP_SCHEDULER", "").strip().lower()
    return SCHEDULER_CLS_FINAL if flavour == "final" else SCHEDULER_CLS


def uses_dynamic_chunks(vllm_config: Any) -> bool:
    """Whether this configuration sizes pipeline steps by time; the platform
    installs the scheduler class only when it does."""
    return (
        vllm_config is not None
        and vllm_config.parallel_config.pipeline_parallel_size > 1
        and vllm_config.scheduler_config.scheduler_cls in (SCHEDULER_CLS, SCHEDULER_CLS_FINAL)
    )


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


def _env_switch(name: str) -> bool:
    """An ix-unified-port switch: on unless the variable is set to 0 / false / off."""
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "off")


class TpuPipelineChunkScheduler(Scheduler):
    _MEMO_LOG_CAP = PrefixAwareScheduler._MEMO_LOG_CAP
    _PAW_SORT_LOG_INTERVAL_S = PrefixAwareScheduler._PAW_SORT_LOG_INTERVAL_S
    WINDOW = PrefixAwareScheduler.WINDOW
    _enable_memo_xstep = PrefixAwareScheduler._enable_memo_xstep
    _memo_trim_log = PrefixAwareScheduler._memo_trim_log
    _memo_still_exact = PrefixAwareScheduler._memo_still_exact
    _memo_get = PrefixAwareScheduler._memo_get
    _memo_compute = PrefixAwareScheduler._memo_compute
    _get_prefix_cache_epoch = PrefixAwareScheduler._get_prefix_cache_epoch
    _get_block_size = PrefixAwareScheduler._get_block_size
    _get_num_free_blocks = PrefixAwareScheduler._get_num_free_blocks
    _prefix_warmth = PrefixAwareScheduler._prefix_warmth
    _external_warmth = PrefixAwareScheduler._external_warmth
    _get_paw_aging_knobs = PrefixAwareScheduler._get_paw_aging_knobs
    _now = PrefixAwareScheduler._now
    _init_paw_sort = PrefixAwareScheduler._init_paw_sort
    _sort_waiting_queue = PrefixAwareScheduler._sort_waiting_queue
    _select_waiting_queue_for_scheduling = (
        PrefixAwareScheduler._select_waiting_queue_for_scheduling
    )
    _promote_warmest = PrefixAwareScheduler._promote_warmest
    _maybe_pack_prompt_ids = PrefixAwareScheduler._maybe_pack_prompt_ids

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        # Modelled time of the steps still in the pipeline, newest first: a
        # step launched k steps ago runs for pp_size - k more steps. A push
        # enters as a step of no cost and ages the steps before it; a settle
        # leaves the pipeline empty (pipeline_advanced).
        self._inflight_ms: list[float] = []
        self._clips = 0
        self._waits = 0
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        self._is_kv_producer = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_role == "kv_producer"
        )
        self._is_kv_consumer = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_role != "kv_producer"
        )
        self._srpf_running: bool = bool(envs.TPU_SCHED_SRPF_RUNNING)
        self._paw_aging_tok_per_s, self._paw_max_wait_s = _read_paw_aging_knobs()
        # The MambaManager class patches go in before HybridStateRetention
        # binds its per-instance hooks; both read the align split flag as
        # the base scheduler set it.
        self._single_step = False
        self._n_single_step_reqs = 0
        if not self._is_kv_consumer and single_step_prefill.enabled():
            ok, why = single_step_prefill.supported(self)
            if ok:
                single_step_prefill.install_mamba_manager_patches()
                self._single_step = True
            else:
                logger.warning(
                    "TPU_SCHED_SINGLE_STEP_PREFILL ignored: %s. Workers must "
                    "then run without it too.",
                    why,
                )
        self._retention: HybridStateRetention | None = None
        wanted = envs.TPU_HYBRID_STATE_RETENTION
        if wanted is None:
            wanted = not self._is_kv_consumer
        if wanted:
            retention = HybridStateRetention(self.kv_cache_manager)
            if retention.enabled:
                self._retention = retention
        self._enable_memo_xstep()
        self._init_paw_sort()
        # ix-unified-port switches, each default 1 = this branch's behaviour; 0 restores the
        # scheduler the old trees ran: TPU_SCHED_PAW_PROMOTE_WARMEST (warmest-first promotion),
        # TPU_PP_QUAD_COST (PR #2243 quadratic cost: keys, pending pairs, live target ceiling),
        # TPU_PP_INFLIGHT_COST (in-flight step costing, 4ec32d152 / 06f7b9bf9).
        self._promote_waiting = _env_switch("TPU_SCHED_PAW_PROMOTE_WARMEST")
        self._quad = _env_switch("TPU_PP_QUAD_COST")
        self._inflight_cost = _env_switch("TPU_PP_INFLIGHT_COST")
        logger.info(
            "Pipeline chunk scheduler: waiting for the step cost model "
            "(hybrid_state_retention=%s, single_step_prefill=%s, "
            "paw_sort=%s, srpf_running=%s, memo_xstep=%s)",
            self._retention is not None,
            self._single_step,
            self._paw_sort,
            self._srpf_running,
            getattr(self, "_memo_xstep", False),
        )

    def _quad_cost(self, tokens: int, prefix: int) -> float:
        """Quadratic FLOPs-equivalent token cost under the step cost model."""
        if self._cost is not None and getattr(self, "_quad", True):
            return self._cost.effective_tokens(tokens, prefix)
        return float(max(tokens, 0))

    def _aged_prefill_key(self, request: Any, now: float) -> float:
        """Quadratic SRPF key of a running prefill: remaining effective tokens
        minus TPU_SCHED_PAW_AGING_TOK_PER_S x seconds since arrival."""
        rem = max(_prefill_end(request) - request.num_computed_tokens, 0)
        eff = self._quad_cost(rem, request.num_computed_tokens)
        rate, _ = self._get_paw_aging_knobs()
        if rate <= 0:
            return eff
        return eff - rate * _wait_seconds(request, now)

    def _paw_sort_key(
        self,
        request: Any,
        now: float,
        free_blocks: int | None,
        free_blocks_for_ext: int | None,
    ) -> tuple[int, int, float, float]:
        """Sort key for TPU_SCHED_PAW_SORT_WAITING using quadratic effective
        token cost; lower is admitted first."""
        local_warmth = self._prefix_warmth(request)
        num_tokens = int(request.num_tokens)
        hbm_cold_tokens = max(num_tokens - local_warmth, 0)
        block_size = self._get_block_size()
        fits_hbm = (
            free_blocks is None
            or math.ceil(hbm_cold_tokens / block_size) <= free_blocks
        )
        ext_warmth = self._external_warmth(request, local_warmth)
        if ext_warmth > 0 and not (
            fits_hbm
            and (
                free_blocks_for_ext is None
                or math.ceil(ext_warmth / block_size) <= free_blocks_for_ext
            )
        ):
            ext_warmth = 0
        warmth = local_warmth + ext_warmth
        uncached = max(num_tokens - warmth, 0)
        eff_uncached = self._quad_cost(uncached, warmth)
        arrival = float(getattr(request, "arrival_time", now) or now)
        rate, _ = self._get_paw_aging_knobs()
        aged = eff_uncached - rate * max(0.0, now - arrival)
        started = (
            getattr(request, "num_preemptions", 0) > 0
            or getattr(request, "num_computed_tokens", 0) > 0
        )
        return (0 if fits_hbm else 1, 0 if started else 1, aged, arrival)

    def _rank_and_warmth(
        self,
        request: Any,
        free_blocks_for_fit: int | None = None,
        now: float | None = None,
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth) using quadratic effective token cost."""
        local_warmth = self._prefix_warmth(request)
        hbm_cold_tokens = max(request.num_tokens - local_warmth, 0)
        hbm_eff_cost = self._quad_cost(hbm_cold_tokens, local_warmth)
        rate, _ = self._get_paw_aging_knobs()
        aging = 0.0
        if rate > 0:
            aging = rate * _wait_seconds(
                request, time.time() if now is None else now
            )
        if getattr(self, "_is_kv_consumer", False):
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

    def _make_cached_request_data(self, *args, **kwargs):
        cached = super()._make_cached_request_data(*args, **kwargs)
        if getattr(self, "_trim_resend", None) is None:
            self._trim_resend = True
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
        self, request: Any
    ) -> tuple[object, int, int, bool]:
        connector = getattr(self, "connector", None)
        if connector is None or not connector.supports_divergent_local_hybrid_hits:
            kvm = self.kv_cache_manager
            epoch = self._get_prefix_cache_epoch()
            if (
                epoch is not None
                and not (
                    getattr(kvm, "enable_kv_cache_events", False)
                    and getattr(request, "kv_cache_report_mode", "incremental")
                    == "full"
                )
                and kvm.prefix_cache_lookup_enabled(request)
            ):
                max_len = request.num_tokens - 1
                cached = self._memo_get(request, epoch, max_len)
                if cached is not None:
                    hit = cached[2]
                    if len(hit) == 3:
                        computed_blocks, num_new_computed_tokens, num_uncached = hit
                    else:
                        computed_blocks, num_new_computed_tokens = hit
                        num_uncached = 0
                    if isinstance(computed_blocks, tuple):
                        computed_blocks = tuple(list(b) for b in computed_blocks)
                    shared_prefix_boundary = (
                        num_new_computed_tokens + num_uncached if num_uncached else 0
                    )
                    blocks = kvm.create_kv_cache_blocks(computed_blocks)
                    request._tpu_local_hit_tokens = num_new_computed_tokens
                    return (
                        blocks,
                        num_new_computed_tokens,
                        shared_prefix_boundary,
                        False,
                    )
        blocks, num_local, shared_prefix_boundary, hit_diverged = (
            super()._get_local_prefix_cache_hit(request)
        )
        if hit_diverged and hasattr(connector, "resolve_divergent_local_hit_tokens"):
            kvm = self.kv_cache_manager
            reconciled_blocks, num_reconciled, reconciled_boundary = (
                kvm.get_computed_blocks(request)
            )
            block_size = self._get_block_size()
            target_local = connector.resolve_divergent_local_hit_tokens(
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

    def install_cost_model(self, model: StepCostModel) -> None:
        self._cost = model
        self._pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        logger.info("Pipeline chunk scheduler: %s", model.describe())

    def _enter(self, step_ms: float) -> None:
        """Records a step entering the pipeline; the oldest one leaves."""
        if not getattr(self, "_inflight_cost", True):
            return
        self._inflight_ms = [step_ms, *self._inflight_ms][: max(self._pp_size - 1, 0)]

    def pipeline_advanced(self, pushes: int = 0, settled: bool = False) -> None:
        """Keeps the in-flight history in step with the pipeline outside
        scheduler steps: each push is a step that carries nothing, so it
        enters at no cost and ages the steps before it; a settle, sent once
        nothing is in flight, leaves the pipeline empty. Without this a
        burst that follows a wait would be costed against steps that have
        long left the pipeline."""
        if not getattr(self, "_inflight_cost", True):
            return
        if settled:
            self._inflight_ms = []
            return
        for _ in range(max(int(pushes), 0)):
            self._enter(0.0)

    def _account(self, tokens: int, prefix: int) -> None:
        self._step_tokens += tokens
        self._step_pairs += float(tokens) * prefix
        self._step_squares += float(tokens) * tokens

    def _retention_failed(self) -> None:
        logger.exception(
            "Pipeline chunk scheduler: hybrid state retention failed; disabling it."
        )
        self._retention = None

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
        connector = getattr(self, "connector", None)
        poll = getattr(connector, "poll_and_complete_transfers", None)
        if callable(poll):
            # Completed async loads promote their requests this step.
            try:
                poll(self)
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
        if self._single_step:
            single_step_prefill.begin_step(self.kv_cache_manager)
        if getattr(self, "_memo_xstep", False):
            self._memo_trim_log()
        now_wall = time.time()
        self._step_wall_time = now_wall
        if getattr(self, "_srpf_running", False) and len(self.running) > 1:
            decoding = [r for r in self.running if _is_decoding(r)]
            prefilling = [r for r in self.running if not _is_decoding(r)]
            if len(prefilling) > 1:
                prefilling.sort(key=lambda r: self._aged_prefill_key(r, now_wall))
            if prefilling and (decoding or len(prefilling) > 1):
                self.running[:] = decoding + prefilling
        if getattr(self, "_paw_sort", False):
            try:
                self._sort_waiting_queue()
            except Exception:
                logger.exception(
                    "Pipeline chunk scheduler: waiting-queue sort failed; disabling "
                    "TPU_SCHED_PAW_SORT_WAITING."
                )
                self._paw_sort = False
        if getattr(self, "_promote_waiting", False):
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
        if getattr(output, "scheduled_new_reqs", None):
            self._maybe_pack_prompt_ids(output)
        if self._single_step:
            cuts = single_step_prefill.confirmed_cuts(
                self.kv_cache_manager, output.num_scheduled_tokens
            )
            if cuts:
                # Read by TPUModelRunner; the KV manager already hashed the
                # checkpoint blocks these name.
                output.tpu_mamba_seg_cuts = cuts
                self._n_single_step_reqs += len(cuts)
        if self._cost is not None and output.total_num_scheduled_tokens > 0:
            estimate = self.step_estimate_ms(output)
            assert estimate is not None
            self._enter(estimate)
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
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        if computed > 0:
            return computed
        kvm = getattr(self, "kv_cache_manager", None)
        if not getattr(kvm, "enable_caching", False):
            return 0
        local_warmth = self._prefix_warmth(request)
        ext_warmth = self._external_warmth(request, local_warmth)
        return max(0, min(local_warmth + ext_warmth, _prefill_end(request)))

    def _pending_step_limit(self) -> int | None:
        """The bucket a step is held to: the prefill work still to schedule
        spread over up to as many steps as the pipeline has stages, rounded
        up to a granule, at most a full step. Of those spreads the one that
        gets the work through the pipeline soonest is taken: a step of half
        the tokens that takes as long as the whole one only adds steps. The
        stages advance in lockstep, so while a larger step is still in the
        pipeline every step lasts at least as long as it does and a smaller
        one finishes no sooner; the spreads are costed against the steps in
        flight. None until the cost model is installed or when no prefill
        work is pending."""
        if self._cost is None:
            return None
        pending = 0
        pending_pairs = 0.0
        for request in self.running:
            rem = max(_prefill_end(request) - request.num_computed_tokens, 0)
            pending += rem
            pending_pairs += float(rem) * request.num_computed_tokens
        quad = getattr(self, "_quad", True)
        for request in self.waiting:
            warmth = (
                self._waiting_warmth(request) if quad else request.num_computed_tokens
            )
            rem = max(_prefill_end(request) - warmth, 0)
            pending += rem
            pending_pairs += float(rem) * warmth
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
            share_pairs = (
                pending_pairs * (min(share, pending) / pending) if quad else 0.0
            )
            step_ms = self._cost.step_ms(bucket, share_pairs, float(share) * share)
            done_ms = 0.0
            for step in range(steps):
                # The step launched k steps ago is still in the pipeline
                # during candidate step s while k + s < pp_size - 1; past
                # that no step in flight overlaps, and the bound must not
                # go negative, which would read the oldest entries back in.
                live = self._inflight_ms[: max(self._pp_size - 1 - step, 0)]
                done_ms += max(step_ms, *live) if live else step_ms
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
        if self._cost is not None and num_new_tokens > 0:
            live = self._inflight_ms[: max(self._pp_size - 1, 0)]
            target_ms = (
                max(self._cost.target_ms, *live) if live else self._cost.target_ms
            )
            if not getattr(self, "_quad", True):
                target_ms = None
            cap = self._cost.chunk(
                start,
                num_new_tokens,
                self._step_tokens,
                self._step_pairs,
                self._step_squares,
                self._step_limit,
                self._step_schedule,
                target_ms=target_ms,
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
        if self._cost is not None:
            self._step_schedule += self._cost.pairs(num_new_tokens, start)
        return num_new_tokens

    def _retention_split(
        self,
        request: Any,
        start: int,
        num_new_tokens: int,
        num_new_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> int:
        """Ends the clipped chunk where the GDN state must be written: inside
        the chunk with the single-step cuts, else at the base scheduler's
        block alignment and the retention stops."""
        retention = self._retention
        if retention is not None:
            try:
                retention.prepare_prefill_split(request, start, num_new_tokens)
            except Exception:
                self._retention_failed()
                retention = None
        if self._single_step:
            aligned, cuts = single_step_prefill.plan_single_step_split(
                self, request, start, num_new_tokens, retention
            )
            single_step_prefill.set_plan(
                self.kv_cache_manager,
                request.request_id,
                (start, *cuts) if cuts else None,
            )
            return aligned
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


def profile_and_install_cost_model(scheduler: Any, executor: Any) -> None:
    """Time every stage's forwards and hand the scheduler the model."""
    from vllm_torchtpu.core.pp_chunk_scheduler_final import TpuPipelineChunkSchedulerFinal

    if not isinstance(scheduler, (TpuPipelineChunkScheduler, TpuPipelineChunkSchedulerFinal)):
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


def attach_pipeline_listener(scheduler: Any, executor: Any) -> None:
    """Have the executor's push and settle accounting (``pp_push``) age the
    scheduler's in-flight step costs; a no-op for other schedulers and for
    executors without a pipeline."""
    if not isinstance(scheduler, TpuPipelineChunkScheduler):
        return
    if getattr(executor, "vllm_config", None) is None:
        return
    from vllm_torchtpu.distributed.pp_push import wave_state

    wave_state(executor).listener = scheduler.pipeline_advanced


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
        attach_pipeline_listener(self.scheduler, self.model_executor)

    EngineCore.__init__ = __init__
    EngineCore._tpu_pp_chunks_patch = True
