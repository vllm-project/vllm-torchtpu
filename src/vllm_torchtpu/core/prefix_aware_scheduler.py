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
"""Prefix-aware request scheduler with Shortest-Remaining-Prefix-First (SRPF) ordering.

Extends the vLLM V1 scheduler to maximize KV prefix cache reuse across concurrent
requests while bounding mixed-step chunked prefill interference on active decode batches.
"""

import math
import os
import time
from bisect import bisect_right
from itertools import islice
from operator import itemgetter
from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.core import prefix_change_log, single_step_prefill
from vllm_torchtpu.core.hybrid_retention import (
    ALLOC_NONE_REASONS,
    HybridStateRetention,
    format_counts,
    is_waiting_status,
)
from vllm_torchtpu.core.packed_ids import pack_new_request_prompt_ids

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000
# Telemetry counts a step of at most this many Mamba blocks of tokens, taken
# while requests wait, as a small step (a lone cold chunk under the cold cap).
_SMALL_STEP_BLOCKS = 2

# Priority tiers: lower values admit earlier.
_TIER_FREE_BLOCK_FIT = 0
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


def _is_decoding(request: Request) -> bool:
    return request.num_computed_tokens >= request.num_prompt_tokens > 0


def _remaining_prefill_tokens(request: Request) -> int:
    return max(
        getattr(request, "num_prompt_tokens", 0)
        - getattr(request, "num_computed_tokens", 0),
        0,
    )


def _remaining_output_tokens(request: Request) -> int | None:
    max_toks = getattr(request, "max_tokens", None)
    if max_toks is None:
        sp = getattr(request, "sampling_params", None)
        max_toks = getattr(sp, "max_tokens", None) if sp is not None else None
    if max_toks is None:
        return None
    try:
        max_toks_i = int(max_toks)
    except (TypeError, ValueError):
        return None
    if max_toks_i <= 0:
        return None
    out_toks = getattr(request, "num_output_tokens", None)
    if out_toks is None:
        all_ids = getattr(request, "all_token_ids", None)
        prompt_toks = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if all_ids is not None and prompt_toks > 0:
            out_toks = max(len(all_ids) - prompt_toks, 0)
        else:
            comp_toks = int(getattr(request, "num_computed_tokens", 0) or 0)
            out_toks = max(comp_toks - prompt_toks, 0)
    return max_toks_i - int(out_toks)


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _read_paw_aging_knobs() -> tuple[float, float]:
    """Returns (TPU_SCHED_PAW_AGING_TOK_PER_S, TPU_SCHED_PAW_MAX_WAIT_S); 0 = off."""
    return (
        max(_env_float("TPU_SCHED_PAW_AGING_TOK_PER_S", 0.0), 0.0),
        max(_env_float("TPU_SCHED_PAW_MAX_WAIT_S", 0.0), 0.0),
    )


def _read_exempt_warm_reserve() -> float:
    """Returns TPU_SCHED_PAW_EXEMPT_WARM_RESERVE, clamped to [0, 1]."""
    return min(max(_env_float("TPU_SCHED_PAW_EXEMPT_WARM_RESERVE", 0.5), 0.0), 1.0)


def _wait_seconds(request: Request, now: float) -> float:
    """Seconds since ``request.arrival_time``.

    ``arrival_time`` is the wall-clock time (``time.time()``) at which this
    engine's API server received the request; on a disaggregated prefill leg
    that is the proxy's prefill hop. It covers time in the waiting queue and
    time in ``running``, and does not reset on preemption.
    """
    arrival = getattr(request, "arrival_time", None)
    if arrival is None:
        return 0.0
    return max(now - float(arrival), 0.0)


def _hit_signature(result) -> tuple:
    """Comparable form of a find_longest_cache_hit result (block ids)."""
    blocks = result[0]
    try:
        ids = tuple(tuple(getattr(b, "block_id", b) for b in group) for group in blocks)
    except TypeError:
        ids = None
    return (result[1], result[2] if len(result) > 2 else None, ids)


def trim_resend_token_ids(cached_reqs) -> int:
    """Shrink ``CachedRequestData.all_token_ids`` to what the worker reads.

    Upstream (v0.29.0 ``Scheduler._make_cached_request_data``) re-sends the
    request's *full* token list for every running request that was not
    scheduled in the previous step. It is pickled into the SchedulerOutput and
    unpickled by each of the 8 TP workers (~5 ms at 94k tokens). The only
    reader on our path is ``GPUModelRunner._update_states`` (reached through
    ``TPUModelRunner._update_states``), and only when the request is not in
    the persistent batch, async scheduling is on and ``num_output_tokens > 0``::

        req_state.output_token_ids = all_token_ids[req_id][-num_output_tokens:]

    so ``toks[-n:]`` with the same ``n`` is exactly what it consumes (for
    ``n == 0`` it reads nothing). Requests resumed from preemption keep the
    full list (rare; left exactly as upstream). Returns token ids removed.
    """
    all_ids = getattr(cached_reqs, "all_token_ids", None)
    if not all_ids:
        return 0
    resumed = cached_reqs.resumed_req_ids or ()
    num_output_tokens = cached_reqs.num_output_tokens
    saved = 0
    for i, req_id in enumerate(cached_reqs.req_ids):
        toks = all_ids.get(req_id)
        if toks is None or req_id in resumed:
            continue
        n = num_output_tokens[i]
        if n >= len(toks):
            continue
        trimmed = toks[-n:] if n > 0 else []
        saved += len(toks) - len(trimmed)
        all_ids[req_id] = trimmed
    return saved


class PrefixAwareScheduler(AsyncScheduler):
    """Admit ready requests prioritizing fewest uncached prompt tokens."""

    #: Lookahead window of ready requests to evaluate for cached prefix length.
    WINDOW = 64

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if os.environ.get("VLLM_TPU_T3_TIMERS", "0").strip() == "1":
            # IX track 3 (default off): time the engine core's step loop. The
            # EngineCore binds step_fn after building its scheduler, so a
            # class-level patch made here is the one the busy loop calls.
            from vllm_torchtpu.runner import t3_host

            t3_host.install_engine_core_timers()
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        kv_transfer_cfg = getattr(
            getattr(self, "vllm_config", None), "kv_transfer_config", None
        )
        kv_role = getattr(kv_transfer_cfg, "kv_role", None)
        self._is_disagg = (
            kv_transfer_cfg is not None
            or getattr(self, "connector", None) is not None
        )
        self._is_kv_producer = self._is_disagg and kv_role == "kv_producer"
        self._is_kv_consumer = self._is_disagg and not self._is_kv_producer
        self._srpf_running: bool = (
            bool(envs.TPU_SCHED_SRPF_RUNNING)
            if "TPU_SCHED_SRPF_RUNNING" in os.environ
            else not self._is_kv_consumer
        )
        self._mk_max_running: int = self._resolve_mk_max_running()
        budget = int(getattr(self, "max_num_scheduled_tokens", 8192) or 8192)
        eff_seqs = (
            self._mk_max_running
            if self._mk_max_running > 0
            else min(int(getattr(self, "max_num_running_reqs", 16) or 16), 16)
        )
        self._cold_threshold: int = (
            int(envs.TPU_SCHED_COLD_THRESHOLD)
            if "TPU_SCHED_COLD_THRESHOLD" in os.environ
            else budget
        )
        self._step_total_cap: int = (
            int(envs.TPU_SCHED_STEP_TOTAL_CAP)
            if "TPU_SCHED_STEP_TOTAL_CAP" in os.environ
            else (0 if self._is_disagg else max(1024, budget // 2))
        )
        self._cold_chunk_cap: int = (
            int(envs.TPU_SCHED_COLD_CHUNK_CAP)
            if "TPU_SCHED_COLD_CHUNK_CAP" in os.environ
            else (
                max(512, budget // 4)
                if self._is_kv_producer
                else (0 if self._is_disagg else max(512, self._step_total_cap // 2))
            )
        )
        self._cold_min_decoders: int = (
            int(envs.TPU_SCHED_COLD_MIN_DECODERS)
            if "TPU_SCHED_COLD_MIN_DECODERS" in os.environ
            else max(1, (3 * eff_seqs) // 4)
        )
        self._warm_streak: int = (
            int(envs.TPU_SCHED_WARM_STREAK)
            if "TPU_SCHED_WARM_STREAK" in os.environ
            else (0 if self._is_disagg else int(envs.TPU_SCHED_WARM_STREAK))
        )
        self._warm_min_decoders: int = int(envs.TPU_SCHED_WARM_MIN_DECODERS)
        self._finish_decode_steps: int = (
            int(envs.TPU_SCHED_FINISH_DECODE_STEPS)
            if "TPU_SCHED_FINISH_DECODE_STEPS" in os.environ
            else (0 if self._is_disagg else int(envs.TPU_SCHED_FINISH_DECODE_STEPS))
        )
        self._finish_decode_max_deferral_steps: int = int(
            envs.TPU_SCHED_FINISH_DECODE_MAX_DEFERRAL_STEPS
        )
        self._finish_decode_prefill_min_tokens: int = int(
            envs.TPU_SCHED_FINISH_DECODE_PREFILL_MIN_TOKENS
        )
        self._paw_aging_tok_per_s, self._paw_max_wait_s = _read_paw_aging_knobs()
        self._paw_exempt_warm_reserve = _read_exempt_warm_reserve()
        self._warm_reserve_wasted = False
        self._finish_decode_defer_streak: int = 0
        self._step_has_near_finish_decoder: bool | None = None
        self._step_finish_decode_deferred: bool = False
        self._skip_midblock_align_stop: bool = bool(
            envs.TPU_SCHED_SKIP_MIDBLOCK_ALIGN_STOP
        )
        self._n_midblock_stops_skipped = 0
        self._step_num_decoding: int = 0
        self._step_running_prefill_ids: set[int] = set()
        self._step_prefill_tokens_by_req: dict[int, int] | None = None
        self._pure_decode_streak: int = 999999
        self._retention = None
        # Single-step warm prefill: the MambaManager class patches must be in
        # place before HybridStateRetention binds its per-instance hooks.
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
        default_retention = "0" if self._is_kv_consumer else "1"
        if os.environ.get("TPU_HYBRID_STATE_RETENTION", default_retention) != "0":
            retention = HybridStateRetention(self.kv_cache_manager)
            if retention.enabled:
                self._retention = retention
        self._enable_memo_xstep()
        self._init_paw_sort()
        self._init_kv_prefetch()

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, decoders first, "
            "srpf_running=%s, finish_decode_steps=%d@%d(min_toks=%d), "
            "cold_chunk_cap=%d, cold_threshold=%d, "
            "cold_min_decoders=%d, step_total_cap=%d, mk_max_running=%d, "
            "warm_streak=%d@%d, hybrid_state_retention=%s, memo_xstep=%s, "
            "skip_midblock_align_stop=%s, single_step_prefill=%s, "
            "paw_aging_tok_per_s=%g, paw_max_wait_s=%g, "
            "paw_exempt_warm_reserve=%g)",
            self.WINDOW,
            self._srpf_running,
            self._finish_decode_steps,
            self._finish_decode_max_deferral_steps,
            self._finish_decode_prefill_min_tokens,
            self._cold_chunk_cap,
            self._cold_threshold,
            self._cold_min_decoders,
            self._step_total_cap,
            self._mk_max_running,
            self._warm_streak,
            self._warm_min_decoders,
            self._retention is not None,
            getattr(self, "_memo_xstep", False),
            self._skip_midblock_align_stop,
            self._single_step,
            self._paw_aging_tok_per_s,
            self._paw_max_wait_s,
            self._paw_exempt_warm_reserve,
        )

    # ------------------------------------------------------------------
    # Decode KV prefetch past a full running set
    # (TPU_SCHED_KV_PREFETCH_WAITING=N, default 0 = off).
    #
    # vLLM v0.31.0's waiting loop stops at ``len(running) >=
    # max_num_active_reqs`` before it asks the connector about the next
    # waiting request. On a PD consumer that also defers the remote KV pull,
    # although an async load never joins ``running``: it waits in
    # WAITING_FOR_REMOTE_KVS holding only KV blocks. So pulls start only
    # while a slot is free, and a slot that frees while no KV-ready request
    # is held idles for the whole pull.
    #
    # With N > 0 on a KV consumer, schedule() lifts max_num_active_reqs for
    # the stock loop and enforces the real cap itself, so that at the cap the
    # loop can only start async loads for the head of ``waiting``:
    #   * a request that already holds KV (loading or loaded) is skipped
    #     instead of admitted (_handle_blocked_waiting_request);
    #   * a request that would run without an async load (local hit,
    #     preemption resume) stops the scan, so nothing is pulled ahead of it
    #     and nothing joins ``running`` past the cap
    #     (_reserve_prefill_lookahead returns 0, the loop breaks);
    #   * a new load is refused once N requests are held (in
    #     ``_inflight_prefills`` and not RUNNING), and must otherwise fit in
    #     the free blocks minus the stock async-load reservation minus
    #     TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS kept for the running set
    #     (_inflight_prefill_reserved_blocks).
    # Below the cap the loop is unchanged. When a slot frees, the stock loop
    # drains ``kv_holding_waiting`` first, so the oldest KV-ready prefetched
    # request takes it on the same step. Aborts of prefetched requests free
    # their blocks through the stock WAITING_FOR_REMOTE_KVS path.
    # ------------------------------------------------------------------
    _KV_PREFETCH_LOG_INTERVAL_S = 60.0

    def _init_kv_prefetch(self) -> None:
        n = int(envs.TPU_SCHED_KV_PREFETCH_WAITING)
        reserve = int(envs.TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS)
        consumer = bool(getattr(self, "_is_kv_consumer", False)) and (
            getattr(self, "connector", None) is not None
        )
        if n > 0 and not consumer:
            logger.info(
                "TPU_SCHED_KV_PREFETCH_WAITING=%d ignored: not a KV consumer.", n
            )
        self._kv_prefetch_n = max(n, 0) if consumer else 0
        self._kv_prefetch_reserve = max(reserve, 0)
        # Real admission cap while schedule() runs with prefetch on, else None.
        self._kv_prefetch_cap: int | None = None
        self._kv_prefetch_in_waiting = False
        self._kv_prefetch_cands: set[int] = set()
        self._kv_prefetch_started = 0
        self._kv_prefetch_steps = 0
        self._kv_prefetch_held_skips = 0
        self._kv_prefetch_full_refusals = 0
        self._kv_prefetch_log_t = time.monotonic()
        if self._kv_prefetch_n:
            logger.info(
                "PrefixAwareScheduler: decode KV prefetch past a full running "
                "set ACTIVE (max_held=%d, reserve_blocks=%d)",
                self._kv_prefetch_n,
                self._kv_prefetch_reserve,
            )

    def _kv_prefetch_at_cap(self) -> bool:
        """Whether prefetch is enforcing the cap and the running set is full."""
        cap = getattr(self, "_kv_prefetch_cap", None)
        if cap is None:
            return False
        num_running = len(self.running) + int(
            getattr(self, "num_waiting_for_streaming_input", 0) or 0
        )
        return num_running >= cap

    def _kv_prefetch_num_held(self) -> int:
        """Requests holding KV for an async load (loading or loaded) that have
        not run yet."""
        return sum(
            1
            for r in getattr(self, "_inflight_prefills", ())
            if r.status != RequestStatus.RUNNING
        )

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        if getattr(self, "_kv_prefetch_cap", None) is not None:
            self._kv_prefetch_in_waiting = True
            if self._kv_prefetch_at_cap():
                if self._holds_kv_blocks(request):
                    # Held for a free slot; never admitted past the cap.
                    self._kv_prefetch_held_skips += 1
                    return False
                self._kv_prefetch_cands.add(id(request))
        return super()._handle_blocked_waiting_request(request)

    def _kv_prefetch_begin_step(self) -> bool:
        """Lift the stock loop's cap for this step if prefetch could start a
        load. Returns whether it did."""
        if not getattr(self, "_kv_prefetch_n", 0):
            return False
        cap = int(self.max_num_active_reqs)
        num_running = len(self.running) + int(
            getattr(self, "num_waiting_for_streaming_input", 0) or 0
        )
        if num_running >= cap and (
            not self.waiting or self._kv_prefetch_num_held() >= self._kv_prefetch_n
        ):
            # Nothing to prefetch: keep the stock early stop at the cap.
            return False
        self._kv_prefetch_cap = cap
        self._kv_prefetch_in_waiting = False
        self._kv_prefetch_cands = set()
        self._kv_prefetch_before = {id(r) for r in self._inflight_prefills}
        # Any value above the cap works: running never passes the cap here.
        self.max_num_active_reqs = cap + self._kv_prefetch_n
        return True

    def _kv_prefetch_end_step(self) -> None:
        cap = self._kv_prefetch_cap
        if cap is None:
            return
        self.max_num_active_reqs = cap
        self._kv_prefetch_cap = None
        self._kv_prefetch_in_waiting = False
        self._kv_prefetch_steps += 1
        before = getattr(self, "_kv_prefetch_before", set())
        cands = self._kv_prefetch_cands
        self._kv_prefetch_started += sum(
            1
            for r in self._inflight_prefills
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and id(r) not in before
            and id(r) in cands
        )
        self._kv_prefetch_cands = set()
        self._kv_prefetch_before = set()
        if len(self.running) > cap:
            logger.error(
                "KV prefetch: running=%d exceeds max_num_active_reqs=%d; "
                "disabling TPU_SCHED_KV_PREFETCH_WAITING.",
                len(self.running),
                cap,
            )
            self._kv_prefetch_n = 0

    def _kv_prefetch_maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._kv_prefetch_log_t < self._KV_PREFETCH_LOG_INTERVAL_S:
            return
        self._kv_prefetch_log_t = now
        logger.info(
            "kv-prefetch telemetry: started_at_cap=%d lifted_steps=%d "
            "held_skips=%d full_refusals=%d held=%d running=%d waiting=%d",
            self._kv_prefetch_started,
            self._kv_prefetch_steps,
            self._kv_prefetch_held_skips,
            self._kv_prefetch_full_refusals,
            self._kv_prefetch_num_held(),
            len(self.running),
            len(self.waiting) + len(getattr(self, "kv_holding_waiting", ())),
        )

    # ------------------------------------------------------------------
    # Waiting-queue sort (TPU_SCHED_PAW_SORT_WAITING, default off).
    #
    # ``_promote_warmest`` moves one request per queue to the head. With the
    # flag on, ``schedule()`` also stable-sorts the first
    # TPU_SCHED_PAW_SORT_WINDOW requests of ``waiting`` before the base
    # scheduler admits from it, so every admission of the step follows the
    # order. The key is the request's estimated uncached prompt tokens
    # (prompt minus local HBM hit minus a loadable external hit) minus
    # TPU_SCHED_PAW_AGING_TOK_PER_S x seconds waited, which bounds how long a
    # large prompt can be overtaken. Requests that already hold computed
    # tokens or were preempted go first, and requests whose uncached tokens
    # do not fit the free HBM blocks go last, so they cannot stall the head.
    # ------------------------------------------------------------------
    _PAW_SORT_LOG_INTERVAL_S = 60.0

    def _init_paw_sort(self) -> None:
        self._paw_sort = bool(envs.TPU_SCHED_PAW_SORT_WAITING)
        self._paw_sort_window = max(2, int(envs.TPU_SCHED_PAW_SORT_WINDOW))
        self._paw_sort_steps = 0
        self._paw_sort_reordered = 0
        self._paw_sort_moved = 0
        self._paw_sort_head_cold = 0
        self._paw_sort_log_t = time.monotonic()
        if self._paw_sort:
            logger.info(
                "PrefixAwareScheduler: waiting-queue SRPF sort ACTIVE "
                "(window=%d, aging_tok_per_s=%g)",
                self._paw_sort_window,
                self._get_paw_aging_knobs()[0],
            )

    def _paw_sort_key(
        self,
        request: Request,
        now: float,
        free_blocks: int | None,
        free_blocks_for_ext: int | None,
    ) -> tuple[int, int, float, float]:
        """Sort key for TPU_SCHED_PAW_SORT_WAITING; lower is admitted first."""
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
        uncached = max(num_tokens - local_warmth - ext_warmth, 0)
        arrival = float(getattr(request, "arrival_time", now) or now)
        rate, _ = self._get_paw_aging_knobs()
        aged = uncached - rate * max(0.0, now - arrival)
        started = (
            getattr(request, "num_preemptions", 0) > 0
            or getattr(request, "num_computed_tokens", 0) > 0
        )
        return (0 if fits_hbm else 1, 0 if started else 1, aged, arrival)

    def _sort_waiting_queue(self) -> None:
        """Stable-sort the first TPU_SCHED_PAW_SORT_WINDOW waiting requests by
        ``_paw_sort_key`` in place."""
        queue = self.waiting
        if self.policy != SchedulingPolicy.FCFS or len(queue) < 2:
            return
        window = list(islice(queue, self._paw_sort_window))
        free_blocks = self._get_num_free_blocks()
        free_blocks_for_ext = free_blocks
        if (
            free_blocks is not None
            and getattr(self, "_is_kv_producer", False)
            and hasattr(self, "_inflight_prefill_reserved_blocks")
        ):
            try:
                reserved = int(self._inflight_prefill_reserved_blocks() or 0)
            except Exception:
                reserved = 0
            free_blocks_for_ext = max(0, free_blocks - reserved)
        now = time.time()
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

    # ------------------------------------------------------------------
    # Cross-step memo of find_longest_cache_hit (TPU_SCHED_PAW_MEMO_XSTEP).
    #
    # The epoch-keyed memo below was invalidated by *any* prefix-cache
    # mutation (every hash insert, every eviction, every schedule start with
    # unwritten blocks), so it was stale on almost every step and
    # _promote_warmest re-walked up to WINDOW waiting requests per call.
    #
    # find_longest_cache_hit(hashes, max_len) is a deterministic function of
    # the answers BlockPool.get_cached_block(h, groups) gives for the hashes it
    # probes. With the memo on, core/prefix_change_log.py records the probed
    # hashes during the computation and logs every hash whose answer can
    # change (insert, removal, unwritten-flag flip, reset) on top of the
    # same-step patch. A memo is reused iff none of its probed hashes was
    # logged after its epoch, so the reused result equals a fresh lookup
    # exactly. The first
    # TPU_SCHED_PAW_MEMO_VERIFY (default 64) cross-step reuses are re-checked
    # against a fresh lookup; any mismatch disables the cross-step memo.
    # ------------------------------------------------------------------
    _MEMO_LOG_CAP = 65536

    def _enable_memo_xstep(self) -> None:
        self._memo_xstep = False
        self._memo_hits = 0
        self._memo_computes = 0
        self._memo_verified = 0
        try:
            self._memo_verify_left = int(
                os.environ.get("TPU_SCHED_PAW_MEMO_VERIFY", "64")
            )
        except ValueError:
            self._memo_verify_left = 64
        if not _env_on("TPU_SCHED_PAW_MEMO_XSTEP"):
            return
        try:
            from vllm.v1.core.block_pool import BlockPool

            pool = getattr(self.kv_cache_manager, "block_pool", None)
            if not isinstance(pool, BlockPool) or not prefix_change_log.install():
                return
            prefix_change_log.enable(pool)
            self._memo_xstep = True
        except Exception:
            logger.exception("PrefixAwareScheduler: memo_xstep init failed.")

    def _memo_trim_log(self) -> None:
        pool = getattr(self.kv_cache_manager, "block_pool", None)
        d = getattr(pool, "__dict__", None)
        log = d.get("_tpu_prefix_change_log") if isinstance(d, dict) else None
        if log is not None and len(log) > self._MEMO_LOG_CAP:
            cut = len(log) // 2
            d["_tpu_prefix_change_log_base"] = log[cut - 1][0]
            del log[:cut]

    def _memo_still_exact(self, cached, epoch: int) -> bool:
        """True iff no hash probed by ``cached`` changed since its epoch."""
        if len(cached) < 6 or cached[4] is None:
            return False
        d = self.kv_cache_manager.block_pool.__dict__
        log = d.get("_tpu_prefix_change_log")
        if log is None:
            return False
        e0 = cached[0]
        if e0 < d.get("_tpu_prefix_change_log_base", 0) or e0 > epoch:
            return False
        probes = cached[4]
        start = bisect_right(log, e0, key=itemgetter(0))
        for i in range(start, len(log)):
            if log[i][1] in probes:
                return False
        return True

    def _memo_get(self, request: Request, epoch: int | None, max_len: int):
        """Return a still-exact memo tuple for ``request`` or None."""
        if epoch is None:
            return None
        cached = getattr(request, "_paw_hit_cache", None)
        if cached is None or cached[1] != max_len:
            return None
        if cached[0] == epoch:
            return cached
        if not getattr(self, "_memo_xstep", False):
            return None
        if len(cached) < 6 or cached[5] != len(request.block_hashes):
            return None
        if not self._memo_still_exact(cached, epoch):
            return None
        if self._memo_verify_left > 0:
            self._memo_verify_left -= 1
            fresh = self._memo_compute(request, epoch, max_len)
            if _hit_signature(fresh[2]) != _hit_signature(cached[2]):
                logger.error(
                    "PrefixAwareScheduler: cross-step memo mismatch for %s "
                    "(memo=%s fresh=%s); disabling TPU_SCHED_PAW_MEMO_XSTEP.",
                    getattr(request, "request_id", "?"),
                    _hit_signature(cached[2])[:2],
                    _hit_signature(fresh[2])[:2],
                )
                self._memo_xstep = False
                return fresh
            self._memo_verified += 1
            return fresh
        cached = (epoch,) + cached[1:]
        request._paw_hit_cache = cached
        self._memo_hits += 1
        return cached

    def _memo_compute(self, request: Request, epoch: int | None, max_len: int):
        kvm = self.kv_cache_manager
        record = epoch is not None and getattr(self, "_memo_xstep", False)
        probes = None
        if record:
            result, probes = prefix_change_log.record_probes(
                kvm.block_pool,
                kvm.coordinator.find_longest_cache_hit,
                request.block_hashes,
                max_len,
            )
            if isinstance(result, tuple) and isinstance(result[0], tuple):
                result = (tuple(tuple(b) for b in result[0]),) + result[1:]
        else:
            result = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max_len
            )
        warmth = int(result[1])
        self._memo_computes = getattr(self, "_memo_computes", 0) + 1
        cached = (epoch, max_len, result, warmth, probes, len(request.block_hashes))
        if epoch is not None:
            request._paw_hit_cache = cached
        return cached

    def _resolve_mk_max_running(self) -> int:
        """Return the megakernel-derived active running-sequence cap, or 0 if disabled.

        When ``VLLM_TPU_QWEN35_MEGAKERNEL=1`` and ``dcp_world_size == 1``, the
        Pallas megakernel covers verify batches up to ``mk_max_verify_rows``
        (default 64 within the 62 MiB v7x VMEM ceiling). Admitting a 17th
        concurrent sequence into ``self.running`` under MTP=3 (4 verify rows per
        sequence) pushes pure decode steps off the megakernel (~7.5 ms/step,
        ~2,150 tok/s) onto ``stock_t4`` (~28 ms/step, ~1,500 tok/s), reducing
        both cluster generation throughput and per-request interactivity.
        """
        raw = int(envs.TPU_SCHED_MK_MAX_RUNNING)
        if raw >= 0:
            return raw
        if not _env_on("VLLM_TPU_QWEN35_MEGAKERNEL", "0"):
            return 0
        if int(getattr(self, "dcp_world_size", 1) or 1) != 1:
            return 0
        if not _env_on("VLLM_TPU_QWEN35_MK_VERIFY_48", "1"):
            mk_max_verify_rows = 32
        elif not _env_on("VLLM_TPU_QWEN35_MK_VERIFY_64", "1"):
            mk_max_verify_rows = 48
        elif _env_on("VLLM_TPU_QWEN35_MK_SPLIT_WIDE", "1"):
            mk_max_verify_rows = 128
        else:
            mk_max_verify_rows = 64
        num_spec = max(int(getattr(self, "num_spec_tokens", 3) or 0), 0)
        if num_spec > 0:
            return max(1, mk_max_verify_rows // (1 + num_spec))
        return 32

    def _promote_warmest_on(self) -> bool:
        on = getattr(self, "_promote_warmest_switch", None)
        if on is None:
            on = os.environ.get("TPU_SCHED_PAW_PROMOTE_WARMEST", "1").strip().lower() not in (
                "0",
                "false",
                "off",
            )
            self._promote_warmest_switch = on
        return on

    def _inflight_prefill_reserved_blocks(self) -> int:
        reserved = self._decode_inflight_reserved_blocks()
        if (
            getattr(self, "_kv_prefetch_in_waiting", False)
            and self._kv_prefetch_at_cap()
        ):
            # The stock loop calls this only for an async load, right before
            # allocating it. At the cap that load is a prefetch.
            if self._kv_prefetch_num_held() >= self._kv_prefetch_n:
                self._kv_prefetch_full_refusals += 1
                return 1 << 30
            return reserved + self._kv_prefetch_reserve
        return reserved

    def _decode_inflight_reserved_blocks(self) -> int:
        """The stock async-load reservation, or, under
        TPU_SCHED_DECODE_INFLIGHT_THROTTLE=1 (ix-unified-port, default off), the
        final-lineage decode admission throttle (fp2q-r5 0447ff58b, from ae5e50b95):
        on a KV consumer, refuse a new remote load while TPU_SCHED_DECODE_MAX_STAGE3_INFLIGHT
        loads are in flight or running + in flight reaches max_running +
        TPU_SCHED_DECODE_STAGE3_OVERPROVISION, else reserve
        max(TPU_SCHED_DECODE_RESERVED_BLOCKS, _PER_SEQ x (running + in flight)) blocks.
        Defaults 16 / 32 / 128 / 8 at DCP > 1 and 6 / 4 / 512 / 16 otherwise, as in that tree."""
        base = int(super()._inflight_prefill_reserved_blocks() or 0)
        on = getattr(self, "_decode_throttle_switch", None)
        if on is None:
            on = os.environ.get("TPU_SCHED_DECODE_INFLIGHT_THROTTLE", "0").strip() == "1"
            self._decode_throttle_switch = on
        if not on or not getattr(self, "_is_kv_consumer", False):
            return base
        num_inflight = len(getattr(self, "_inflight_prefills", ()))
        num_running = len(getattr(self, "running", ()))
        max_running = int(getattr(self, "max_num_running_reqs", 32) or 32)
        dcp_w = int(getattr(self, "dcp_world_size", 1) or 1)
        max_stage3_inflight = int(
            os.environ.get("TPU_SCHED_DECODE_MAX_STAGE3_INFLIGHT", "16" if dcp_w > 1 else "6")
            or 0
        )
        stage3_overprovision = int(
            os.environ.get("TPU_SCHED_DECODE_STAGE3_OVERPROVISION", "32" if dcp_w > 1 else "4")
        )
        if (max_stage3_inflight > 0 and num_inflight >= max_stage3_inflight) or (
            stage3_overprovision >= 0
            and (num_running + num_inflight) >= (max_running + stage3_overprovision)
        ):
            return 1 << 30
        min_reserved = int(
            os.environ.get("TPU_SCHED_DECODE_RESERVED_BLOCKS", "128" if dcp_w > 1 else "512")
            or 0
        )
        per_seq_reserved = int(
            os.environ.get("TPU_SCHED_DECODE_RESERVED_BLOCKS_PER_SEQ", "8" if dcp_w > 1 else "16")
            or 0
        )
        return max(base, max(min_reserved, per_seq_reserved * (num_running + num_inflight)))

    def _get_sched_knobs(self) -> tuple[bool, int, int, int]:
        srpf = getattr(self, "_srpf_running", None)
        if srpf is None:
            srpf = bool(envs.TPU_SCHED_SRPF_RUNNING)
        cap = getattr(self, "_cold_chunk_cap", None)
        if cap is None:
            cap = int(envs.TPU_SCHED_COLD_CHUNK_CAP)
        thresh = getattr(self, "_cold_threshold", None)
        if thresh is None:
            thresh = int(envs.TPU_SCHED_COLD_THRESHOLD)
        min_dec = getattr(self, "_cold_min_decoders", None)
        if min_dec is None:
            min_dec = int(envs.TPU_SCHED_COLD_MIN_DECODERS)
        return srpf, cap, thresh, min_dec

    def _get_paw_aging_knobs(self) -> tuple[float, float]:
        rate = getattr(self, "_paw_aging_tok_per_s", None)
        max_wait = getattr(self, "_paw_max_wait_s", None)
        if rate is None or max_wait is None:
            rate, max_wait = _read_paw_aging_knobs()
            self._paw_aging_tok_per_s, self._paw_max_wait_s = rate, max_wait
        return rate, max_wait

    def _get_exempt_warm_reserve(self) -> float:
        reserve = getattr(self, "_paw_exempt_warm_reserve", None)
        if reserve is None:
            reserve = self._paw_exempt_warm_reserve = _read_exempt_warm_reserve()
        return reserve

    def _now(self) -> float:
        """Wall-clock time of the current schedule() call."""
        now = getattr(self, "_step_wall_time", None)
        return time.time() if now is None else now

    def _aged_prefill_key(self, request: Request, now: float) -> float:
        """SRPF key of a running prefill: remaining prompt tokens minus
        TPU_SCHED_PAW_AGING_TOK_PER_S x seconds since arrival."""
        rate, _ = self._get_paw_aging_knobs()
        rem = _remaining_prefill_tokens(request)
        if rate <= 0:
            return rem
        return rem - rate * _wait_seconds(request, now)

    def _cold_cap_exempt(self, request: Request) -> bool:
        """Returns whether ``request`` arrived over TPU_SCHED_PAW_MAX_WAIT_S ago."""
        _, max_wait = self._get_paw_aging_knobs()
        return max_wait > 0 and _wait_seconds(request, self._now()) > max_wait

    def _note_waiting_stop(self, request: Request, reason: str) -> None:
        """Records why the waiting loop stopped at ``request`` (first stop of
        the step wins; running requests are ignored)."""
        if getattr(self, "_step_wait_event", None) is None and is_waiting_status(
            request
        ):
            self._step_wait_event = reason

    def _get_step_cap_knobs(self) -> tuple[int, int, int]:
        step_cap = getattr(self, "_step_total_cap", None)
        if step_cap is None:
            step_cap = int(envs.TPU_SCHED_STEP_TOTAL_CAP)
        warm_streak = getattr(self, "_warm_streak", None)
        if warm_streak is None:
            warm_streak = int(envs.TPU_SCHED_WARM_STREAK)
        warm_min_dec = getattr(self, "_warm_min_decoders", None)
        if warm_min_dec is None:
            warm_min_dec = int(envs.TPU_SCHED_WARM_MIN_DECODERS)
        return step_cap, warm_streak, warm_min_dec

    def _get_finish_decode_knobs(self) -> tuple[int, int, int]:
        steps = getattr(self, "_finish_decode_steps", None)
        if steps is None:
            steps = int(envs.TPU_SCHED_FINISH_DECODE_STEPS)
        max_defer = getattr(self, "_finish_decode_max_deferral_steps", None)
        if max_defer is None:
            max_defer = int(envs.TPU_SCHED_FINISH_DECODE_MAX_DEFERRAL_STEPS)
        min_toks = getattr(self, "_finish_decode_prefill_min_tokens", None)
        if min_toks is None:
            min_toks = int(envs.TPU_SCHED_FINISH_DECODE_PREFILL_MIN_TOKENS)
        return steps, max_defer, min_toks

    def _has_near_finish_decoder(self, finish_decode_steps: int) -> bool:
        if finish_decode_steps <= 0:
            return False
        step_flag = getattr(self, "_step_has_near_finish_decoder", None)
        if step_flag is not None:
            return bool(step_flag)
        tokens_per_dec = 1 + max(int(getattr(self, "num_spec_tokens", 3) or 0), 0)
        threshold = finish_decode_steps * tokens_per_dec
        for r in getattr(self, "running", ()):
            if not _is_decoding(r):
                continue
            rem_out = _remaining_output_tokens(r)
            if rem_out is not None and 1 <= rem_out <= threshold:
                return True
        return False

    def _retention_failed(self) -> None:
        logger.exception(
            "PrefixAwareScheduler: hybrid state retention failed; disabling it."
        )
        self._retention = None

    def _maybe_pack_prompt_ids(self, out) -> None:
        """TPU_SCHED_PACK_PROMPT_IDS (default 1): ship new requests' prompt
        ids to the workers as packed int32 (see core/packed_ids.py)."""
        if getattr(self, "_pack_ids", None) is None:
            self._pack_ids = _env_on("TPU_SCHED_PACK_PROMPT_IDS")
            try:
                self._pack_ids_min = int(
                    os.environ.get("TPU_SCHED_PACK_PROMPT_IDS_MIN", "1024")
                )
            except ValueError:
                self._pack_ids_min = 1024
        if not self._pack_ids:
            return
        try:
            pack_new_request_prompt_ids(out, self._pack_ids_min)
        except Exception:
            logger.exception(
                "PrefixAwareScheduler: prompt-id packing failed; disabling it."
            )
            self._pack_ids = False

    def _make_cached_request_data(self, *args, **kwargs):
        cached = super()._make_cached_request_data(*args, **kwargs)
        if getattr(self, "_trim_resend", None) is None:
            self._trim_resend = _env_on("TPU_SCHED_TRIM_RESEND_TOKEN_IDS")
            self._trim_resend_n = 0
            self._trim_resend_saved = 0
            self._trim_resend_log_t = time.monotonic()
        if self._trim_resend and cached.all_token_ids:
            try:
                saved = trim_resend_token_ids(cached)
            except Exception:
                logger.exception(
                    "PrefixAwareScheduler: all_token_ids trim failed; disabling it."
                )
                self._trim_resend = False
                return cached
            if saved:
                self._trim_resend_n += 1
                self._trim_resend_saved += saved
                now = time.monotonic()
                if now - self._trim_resend_log_t >= 60.0:
                    self._trim_resend_log_t = now
                    logger.info(
                        "resend-trim telemetry: steps=%d token_ids_not_sent=%d",
                        self._trim_resend_n,
                        self._trim_resend_saved,
                    )
        return cached

    def _connector_finished(self, request: Request):
        res = super()._connector_finished(request)
        if res[0]:
            retention = getattr(self, "_retention", None)
            if retention is not None:
                try:
                    retention.pin_running((request,))
                except Exception:
                    self._retention_failed()
        return res

    def _free_request_blocks(self, request: Request):
        retention = getattr(self, "_retention", None)
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

    def schedule(self, *args, **kwargs):
        from vllm_torchtpu.runner import lightweight_step_stats
        from vllm_torchtpu.runner import t3_host as _t3_host

        if _t3_host.STEP_HOOK:
            _t3_host.on_step()

        _stats_on = lightweight_step_stats.is_enabled()
        _sched_t0_ns = time.perf_counter_ns() if _stats_on else 0
        _kv_wait_ns = 0
        _early_reqs = 0
        _waited_reqs = 0
        _waited = False

        connector = getattr(self, "connector", None)
        if connector is not None:
            poll_fn = getattr(connector, "poll_and_complete_transfers", None)
            if callable(poll_fn):
                try:
                    _kv_wait_ns, _early_reqs, _waited_reqs, _waited = poll_fn(
                        self
                    )
                except Exception:
                    logger.exception(
                        "PrefixAwareScheduler: poll_and_complete_transfers failed."
                    )

        retention = getattr(self, "_retention", None)
        if retention is not None:
            try:
                retention.pin_running(self.running)
            except Exception:
                self._retention_failed()
        # Decoders first: vLLM schedules ``running`` in list order and a
        # partial prefill takes min(remaining, budget), so a prefill ahead of
        # a decoder can leave it no tokens. Stably move every decoding request
        # ahead of every prefilling one. When TPU_SCHED_SRPF_RUNNING is enabled,
        # also stably order prefilling requests by fewest remaining prompt
        # tokens (Shortest-Remaining-Prefill-First) so warm turns finish ahead
        # of multi-step cold prefills, less TPU_SCHED_PAW_AGING_TOK_PER_S per
        # second since arrival so a long cold prefill cannot starve.
        now_wall = time.time()
        self._step_wall_time = now_wall
        srpf_running, _, _, _ = self._get_sched_knobs()
        decoding = [r for r in self.running if _is_decoding(r)]
        prefilling = [r for r in self.running if not _is_decoding(r)]
        if srpf_running and len(prefilling) > 1:
            prefilling.sort(key=lambda r: self._aged_prefill_key(r, now_wall))
        if prefilling and (decoding or (srpf_running and len(prefilling) > 1)):
            self.running[:] = decoding + prefilling
        self._step_num_decoding = len(decoding)
        if getattr(self, "_paw_sort", False):
            try:
                self._sort_waiting_queue()
            except Exception:
                logger.exception(
                    "PrefixAwareScheduler: waiting-queue sort failed; disabling "
                    "TPU_SCHED_PAW_SORT_WAITING."
                )
                self._paw_sort = False
        self._step_decode_tokens = sum(
            max(
                getattr(r, "num_tokens_with_spec", 0)
                + getattr(r, "num_output_placeholders", 0)
                - getattr(r, "num_computed_tokens", 0),
                1,
            )
            for r in decoding
        )
        finish_steps, finish_max_defer, _ = self._get_finish_decode_knobs()
        if finish_steps > 0 and decoding:
            tokens_per_dec = 1 + max(
                int(getattr(self, "num_spec_tokens", 3) or 0), 0
            )
            finish_thresh = finish_steps * tokens_per_dec
            near_finish_ids = {
                id(r)
                for r in decoding
                if (rem_out := _remaining_output_tokens(r)) is not None
                and 1 <= rem_out <= finish_thresh
            }
            self._step_has_near_finish_decoder = bool(near_finish_ids)
            prev_nf_ids = getattr(self, "_prev_near_finish_ids", None)
            if prev_nf_ids and (prev_nf_ids - near_finish_ids):
                total_nf_defer = int(
                    getattr(self, "_total_nf_defer_streak", 0) or 0
                )
                if total_nf_defer < 2 * finish_max_defer:
                    self._finish_decode_defer_streak = 0
            self._prev_near_finish_ids = near_finish_ids
        else:
            self._step_has_near_finish_decoder = False
            self._prev_near_finish_ids = set()
        if not self._step_has_near_finish_decoder:
            self._finish_decode_defer_streak = 0
            self._total_nf_defer_streak = 0
        self._step_finish_decode_deferred = False
        self._step_running_prefill_ids = {id(r) for r in prefilling}
        self._step_prefill_tokens_by_req = {}
        warm_req_ids: set[str] = set()
        self._step_warm_waiting_tokens = (
            self._compute_warm_waiting_tokens(self._cold_threshold, warm_req_ids)
            if getattr(self, "_is_kv_producer", False)
            else 0
        )
        # Warm budget release (only with TPU_SCHED_PAW_MAX_WAIT_S > 0): the cold
        # group is not capped on this step when the previous step capped it for
        # warm requests and none was admitted.
        self._step_cold_release = bool(
            getattr(self, "_warm_reserve_wasted", False)
        ) and (self._get_paw_aging_knobs()[1] > 0)
        self._step_warm_reserve_applied = False
        self._step_max_cold_budget = None
        self._step_cold_exempt_ids = set()
        self._step_wait_event = None
        waiting_at_start = len(getattr(self, "waiting", ()) or ())
        retention = getattr(self, "_retention", None)
        if retention is not None:
            retention.begin_step()
        if getattr(self, "_memo_xstep", False):
            self._memo_trim_log()
        single_step = getattr(self, "_single_step", False)
        if single_step:
            single_step_prefill.begin_step(self.kv_cache_manager)
        mk_max_running = getattr(self, "_mk_max_running", None)
        if mk_max_running is None:
            mk_max_running = self._resolve_mk_max_running()
        # The scan admits into ``running`` against max_num_active_reqs;
        # max_num_running_reqs is the runner's slot count and must stay.
        orig_max_active = getattr(self, "max_num_active_reqs", None)
        if (
            mk_max_running > 0
            and isinstance(orig_max_active, int)
            and orig_max_active > mk_max_running
        ):
            self.max_num_active_reqs = mk_max_running
        if getattr(self, "_kv_prefetch_n", 0):
            try:
                self._kv_prefetch_begin_step()
            except Exception:
                logger.exception(
                    "PrefixAwareScheduler: KV prefetch setup failed; disabling "
                    "TPU_SCHED_KV_PREFETCH_WAITING."
                )
                self._kv_prefetch_n = 0
                self._kv_prefetch_end_step()
        try:
            # vLLM's Scheduler walks the waiting queues without a selection
            # hook, so the warmest-first promotion runs here, once per step.
            # TPU_SCHED_PAW_PROMOTE_WARMEST=0 (ix-unified-port) skips it: FCFS
            # admission, as the final-lineage trees ran on vLLM 0.31.
            if self._promote_warmest_on():
                self._select_waiting_queue_for_scheduling()
            out = super().schedule(*args, **kwargs)
            if getattr(out, "scheduled_new_reqs", None):
                self._maybe_pack_prompt_ids(out)
            if single_step:
                cuts = single_step_prefill.confirmed_cuts(
                    self.kv_cache_manager, out.num_scheduled_tokens
                )
                if cuts:
                    # Read by TPUModelRunner (same env switch); the scheduler
                    # already hashed the checkpoint blocks these name.
                    out.tpu_mamba_seg_cuts = cuts
                    self._n_single_step_reqs += len(cuts)
            if any(v > 0 for v in self._step_prefill_tokens_by_req.values()):
                self._pure_decode_streak = 0
                self._finish_decode_defer_streak = 0
                self._total_nf_defer_streak = 0
            else:
                self._pure_decode_streak = (
                    getattr(self, "_pure_decode_streak", 999999) + 1
                )
                if getattr(self, "_step_finish_decode_deferred", False):
                    self._finish_decode_defer_streak = (
                        int(getattr(self, "_finish_decode_defer_streak", 0) or 0)
                        + 1
                    )
                    self._total_nf_defer_streak = (
                        int(getattr(self, "_total_nf_defer_streak", 0) or 0)
                        + 1
                    )
            scheduled = getattr(out, "num_scheduled_tokens", None) or {}
            warm_admitted = any(rid in scheduled for rid in warm_req_ids)
            self._warm_reserve_wasted = (
                self._step_warm_reserve_applied and not warm_admitted
            )
            stop = self._waiting_loop_stop_reason(out, waiting_at_start)
            self._record_step_telemetry(out, waiting_at_start, stop)
            now = time.monotonic()
            if now - getattr(self, "_last_sched_log_t", 0.0) >= 60.0:
                self._last_sched_log_t = now
                logger.info(
                    "prefix-scheduler telemetry: running=%d waiting=%d decoding=%d "
                    "prefilling=%d free_blocks=%s sched_tokens=%d pure_decode_streak=%d "
                    "finish_defer_streak=%d midblock_stops_skipped=%d "
                    "single_step_reqs=%d %s",
                    len(self.running),
                    len(getattr(self, "waiting", ())),
                    len(decoding),
                    len(prefilling),
                    self._get_num_free_blocks(),
                    getattr(out, "total_num_scheduled_tokens", 0),
                    self._pure_decode_streak,
                    getattr(self, "_finish_decode_defer_streak", 0),
                    getattr(self, "_n_midblock_stops_skipped", 0),
                    getattr(self, "_n_single_step_reqs", 0),
                    self._admission_telemetry(stop),
                )
                if getattr(self, "_memo_xstep", False) or getattr(
                    self, "_memo_computes", 0
                ):
                    logger.info(
                        "paw-memo telemetry: xstep=%s xstep_hits=%d computes=%d "
                        "verified=%d",
                        getattr(self, "_memo_xstep", False),
                        getattr(self, "_memo_hits", 0),
                        getattr(self, "_memo_computes", 0),
                        getattr(self, "_memo_verified", 0),
                    )
            return out
        finally:
            if getattr(self, "_kv_prefetch_cap", None) is not None:
                self._kv_prefetch_end_step()
            if getattr(self, "_kv_prefetch_n", 0):
                self._kv_prefetch_maybe_log()
            if _stats_on:
                _sched_elapsed_ns = time.perf_counter_ns() - _sched_t0_ns
                lightweight_step_stats.get_step_stats_recorder().record_scheduler_step(
                    schedule_ns=_sched_elapsed_ns,
                    kv_wait_ns=_kv_wait_ns,
                    early_completed_reqs=_early_reqs,
                    waited_completed_reqs=_waited_reqs,
                    waited=_waited,
                )
            if orig_max_active is not None:
                self.max_num_active_reqs = orig_max_active
            self._step_running_prefill_ids = set()
            self._step_prefill_tokens_by_req = None
            self._step_warm_waiting_tokens = 0
            self._step_has_near_finish_decoder = None
            self._step_finish_decode_deferred = False
            self._step_wall_time = None
            self._step_cold_release = False
            self._step_warm_reserve_applied = False
            self._step_max_cold_budget = None
            self._step_cold_exempt_ids = set()
            self._step_wait_event = None

    def _waiting_loop_stop_reason(self, out, waiting_at_start: int) -> str:
        """Names what ended vLLM's waiting loop on this step.

        ``cap0:<why>`` / ``align0`` / ``lookahead0``: the chunk of the request
        at the head came out 0 tokens (our cold cap, the Mamba block-aligned
        split, the MTP lookahead reserve). ``alloc:<reason>``: allocate_slots
        refused it (``ALLOC_NONE_REASONS``). Otherwise ``idle`` (nothing was
        waiting), ``preempted`` (the loop did not run), ``max_running``,
        ``budget`` (no token budget left), ``drained`` (every waiting request
        was admitted or skipped) or ``other``.
        """
        event = getattr(self, "_step_wait_event", None)
        if event is not None:
            return event
        retention = getattr(self, "_retention", None)
        nones = getattr(retention, "step_waiting_none", None)
        if nones:
            return "alloc:" + nones[0]
        if waiting_at_start == 0 and not getattr(self, "kv_holding_waiting", None):
            return "idle"
        if getattr(out, "preempted_req_ids", None):
            return "preempted"
        # The admission check the scan stops at; schedule() restores the
        # megakernel cap on max_num_active_reqs only after this runs.
        max_active = getattr(self, "max_num_active_reqs", None)
        num_active = len(self.running) + int(
            getattr(self, "num_waiting_for_streaming_input", 0) or 0
        )
        if isinstance(max_active, int) and num_active >= max_active:
            return "max_running"
        budget = int(getattr(self, "max_num_scheduled_tokens", 8192) or 8192)
        if getattr(out, "total_num_scheduled_tokens", 0) >= budget:
            return "budget"
        if not self._has_undeferred_waiting():
            return "drained"
        return "other"

    def _has_undeferred_waiting(self) -> bool:
        """Whether ``waiting`` holds a request vLLM has not deferred.

        The scan puts a skipped request that holds no KV blocks back into
        ``waiting`` and keeps it in ``deferred_waiting`` until it is scheduled.
        """
        deferred = getattr(self, "deferred_waiting", None) or ()
        return any(r not in deferred for r in getattr(self, "waiting", None) or ())

    def _record_step_telemetry(self, out, waiting_at_start: int, stop: str) -> None:
        """Accumulates the per-minute step counters of the telemetry line.

        A small step counts only when requests are still waiting after it, so
        a step that admitted every waiting request with a short chunk is not
        one.
        """
        tm = getattr(self, "_tm", None)
        if tm is None:
            tm = self._tm = {
                "steps": 0,
                "small_steps_waiting": 0,
                "exempt_steps": 0,
                "stops": {},
                "alloc_none_base": None,
            }
        tm["steps"] += 1
        if getattr(self, "_step_cold_exempt_ids", None):
            tm["exempt_steps"] = tm.get("exempt_steps", 0) + 1
        if waiting_at_start > 0:
            tokens = getattr(out, "total_num_scheduled_tokens", 0)
            if (
                tokens <= _SMALL_STEP_BLOCKS * self._get_block_size()
                and self._has_undeferred_waiting()
            ):
                tm["small_steps_waiting"] += 1
            stops = tm["stops"]
            stops[stop] = stops.get(stop, 0) + 1

    def _admission_telemetry(self, stop: str) -> str:
        """Formats the sampled step's admission state plus the per-minute step
        counters, and resets the counters."""
        retention = getattr(self, "_retention", None)
        tm = getattr(self, "_tm", None) or {}
        step_none = getattr(retention, "step_waiting_none", None)
        alloc_total = getattr(retention, "alloc_none", None)
        alloc_minute = "-"
        if alloc_total is not None:
            base = tm.get("alloc_none_base") or dict.fromkeys(ALLOC_NONE_REASONS, 0)
            alloc_minute = format_counts(
                {k: v - base.get(k, 0) for k, v in alloc_total.items()}
            )
            tm["alloc_none_base"] = dict(alloc_total)
        reserved = "-"
        fn = getattr(self, "_inflight_prefill_reserved_blocks", None)
        if callable(fn) and hasattr(self, "_inflight_prefills"):
            try:
                reserved = str(fn())
            except Exception:
                reserved = "-"
        pool_free = "-"
        pool = getattr(getattr(self, "kv_cache_manager", None), "block_pool", None)
        get_free = getattr(pool, "get_num_free_blocks", None)
        if callable(get_free):
            try:
                pool_free = str(int(get_free()))
            except Exception:
                pool_free = "-"
        max_cold = getattr(self, "_step_max_cold_budget", None)
        stops = dict(sorted((tm.get("stops") or {}).items(), key=lambda kv: -kv[1]))
        text = (
            f"warm_waiting_toks={getattr(self, '_step_warm_waiting_tokens', 0)} "
            f"max_cold_budget={'-' if max_cold is None else max_cold} "
            f"cold_release={int(bool(getattr(self, '_step_cold_release', False)))} "
            f"cold_exempt={len(getattr(self, '_step_cold_exempt_ids', ()) or ())} "
            f"admits={getattr(retention, 'step_admits', 0)} "
            f"async_admits={getattr(retention, 'step_async_admits', 0)} "
            f"stop={stop} "
            f"step_alloc_none={','.join(step_none) if step_none else '-'} "
            f"inflight_reserved={reserved} pool_free={pool_free} "
            f"min_steps={tm.get('steps', 0)} "
            f"min_small_steps_waiting={tm.get('small_steps_waiting', 0)} "
            f"min_exempt_steps={tm.get('exempt_steps', 0)} "
            f"min_alloc_none={alloc_minute} "
            f"min_stops={format_counts(stops)}"
        )
        tm["steps"] = 0
        tm["small_steps_waiting"] = 0
        tm["exempt_steps"] = 0
        tm["stops"] = {}
        return text

    def _cap_cold_prefill_tokens(
        self,
        request: Request,
        start_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        """Cap prefill tokens across single-cold and multi-prefill mixed steps.

        1. First prefill chunk on a step: subtracts scheduled decode tokens from
           ``cold_chunk_cap`` so ``decode_tokens + cold_chunk <= cold_chunk_cap``
           (e.g. ``(1 + K) * num_decoding + cold_chunk <= cold_chunk_cap``),
           keeping single-cold steps inside the ``<= cold_chunk_cap`` XLA bucket.
        2. Subsequent prefill requests in the same step at high concurrency
           (``num_decoding >= cold_min_decoders``): subtracts cumulative scheduled
           step tokens (``decode_tokens + prior_prefill_tokens``) from
           ``step_total_cap`` (default ``budget // 2``), returning ``0`` if fewer
           than one Mamba block remains so multi-prefill collisions never spill
           into the ``budget`` XLA compile bucket.
        3. KV producer, cold request already running: while warm requests wait,
           the cold group shares ``max(block, min(cold_chunk_cap, budget -
           warm_waiting_toks))`` per step. For a request that arrived more than
           TPU_SCHED_PAW_MAX_WAIT_S ago the group budget is ``max(cold_chunk_cap,
           budget - 1 - min(warm_waiting_toks, reserve x budget))`` with reserve
           = TPU_SCHED_PAW_EXEMPT_WARM_RESERVE, so the waiting warm requests keep
           up to that share of the step. With TPU_SCHED_PAW_MAX_WAIT_S > 0, on
           the step after a step whose warm reservation admitted no warm request
           the cold group keeps only the phase-1 gap (``budget - 1``) that lets
           the waiting loop start async loads. With both PAW aging knobs at 0
           this branch is the S0 cold cap.
        """
        _, cold_chunk_cap, cold_threshold, cold_min_decoders = self._get_sched_knobs()
        step_total_cap, warm_streak, warm_min_decoders = self._get_step_cap_knobs()
        prefill_end = max(
            getattr(request, "num_prompt_tokens", 0),
            getattr(request, "num_tokens", 0) - 1,
        )
        rem = prefill_end - start_computed_tokens
        if rem <= 0 or num_new_tokens <= 0:
            return num_new_tokens

        num_decoding = getattr(self, "_step_num_decoding", 0)
        tokens_per_dec = 1 + max(int(getattr(self, "num_spec_tokens", 3) or 0), 0)
        decode_tokens = getattr(
            self, "_step_decode_tokens", tokens_per_dec * num_decoding
        )
        block_size = self._get_block_size()

        if (
            warm_streak > 0
            and 0 < warm_min_decoders <= num_decoding
            and getattr(self, "_pure_decode_streak", 999999) < warm_streak
        ):
            self._last_cap_zero = "warm_streak"
            return 0

        sched_map = getattr(self, "_step_prefill_tokens_by_req", None)
        req_key = id(request)
        if sched_map is not None:
            prior_prefill_tokens = sum(v for k, v in sched_map.items() if k != req_key)
            has_prior_prefill = any(v > 0 for k, v in sched_map.items() if k != req_key)
        else:
            prior_prefill_tokens = 0
            has_prior_prefill = False
        already_scheduled = decode_tokens + prior_prefill_tokens

        finish_steps, finish_max_defer, finish_min_toks = (
            self._get_finish_decode_knobs()
        )
        if finish_steps > 0:
            if self._has_near_finish_decoder(finish_steps):
                if max(prior_prefill_tokens + num_new_tokens, rem) >= finish_min_toks:
                    defer_streak = int(
                        getattr(self, "_finish_decode_defer_streak", 0) or 0
                    )
                    if defer_streak < finish_max_defer:
                        self._step_finish_decode_deferred = True
                        if sched_map is None:
                            self._finish_decode_defer_streak = defer_streak + 1
                        else:
                            sched_map[req_key] = 0
                        self._last_cap_zero = "finish_defer"
                        return 0
                    self._finish_decode_defer_streak = 0
                    self._step_finish_decode_deferred = False
            elif sched_map is None:
                self._finish_decode_defer_streak = 0

        if (
            step_total_cap > 0
            and has_prior_prefill
            and 0 < cold_min_decoders <= num_decoding
        ):
            rem_step_cap = step_total_cap - already_scheduled
            if rem_step_cap <= 0 or (
                rem_step_cap < block_size
                and (start_computed_tokens + rem_step_cap) < prefill_end
            ):
                if sched_map is not None:
                    sched_map[req_key] = 0
                self._last_cap_zero = "step_cap"
                return 0
            num_new_tokens = min(num_new_tokens, rem_step_cap)

        if cold_chunk_cap > 0 and cold_threshold > 0 and rem > cold_threshold:
            in_running = req_key in getattr(self, "_step_running_prefill_ids", ())
            if getattr(self, "_is_kv_producer", False):
                if in_running:
                    budget = int(
                        getattr(self, "max_num_scheduled_tokens", 8192) or 8192
                    )
                    warm_waiting_toks = getattr(
                        self, "_step_warm_waiting_tokens", None
                    )
                    if warm_waiting_toks is None:
                        warm_waiting_toks = self._compute_warm_waiting_tokens(
                            cold_threshold
                        )
                    any_waiting = bool(getattr(self, "waiting", None)) or bool(
                        getattr(self, "kv_holding_waiting", None)
                    )
                    if warm_waiting_toks > 0 and not getattr(
                        self, "_step_cold_release", False
                    ):
                        if self._cold_cap_exempt(request):
                            warm_keep = min(
                                warm_waiting_toks,
                                int(self._get_exempt_warm_reserve() * budget),
                            )
                            max_cold_budget = max(
                                cold_chunk_cap, budget - 1 - warm_keep
                            )
                            exempt_ids = getattr(self, "_step_cold_exempt_ids", None)
                            if exempt_ids is not None:
                                exempt_ids.add(req_key)
                        else:
                            max_cold_budget = max(
                                block_size,
                                min(cold_chunk_cap, budget - warm_waiting_toks),
                            )
                        self._step_warm_reserve_applied = True
                        self._step_max_cold_budget = max_cold_budget
                        if has_prior_prefill:
                            rem_cold_cap = max_cold_budget - already_scheduled
                            if rem_cold_cap < block_size:
                                if sched_map is not None:
                                    sched_map[req_key] = 0
                                self._last_cap_zero = "cold_cap"
                                return 0
                            num_new_tokens = min(num_new_tokens, rem_cold_cap)
                        else:
                            num_new_tokens = min(
                                num_new_tokens,
                                max(block_size, max_cold_budget - decode_tokens),
                            )
                    elif any_waiting:
                        max_phase1_budget = max(block_size, budget - 1)
                        rem_phase1 = max_phase1_budget - already_scheduled
                        if has_prior_prefill and rem_phase1 < block_size:
                            if sched_map is not None:
                                sched_map[req_key] = 0
                            self._last_cap_zero = "phase1"
                            return 0
                        num_new_tokens = min(
                            num_new_tokens, max(block_size, rem_phase1)
                        )
            else:
                has_waiting = bool(getattr(self, "waiting", None)) or bool(
                    getattr(self, "kv_holding_waiting", None)
                )
                if 0 < cold_min_decoders <= num_decoding or (
                    in_running and has_waiting
                ):
                    if has_prior_prefill:
                        rem_cold_cap = cold_chunk_cap - already_scheduled
                        if rem_cold_cap >= block_size:
                            num_new_tokens = min(num_new_tokens, rem_cold_cap)
                        else:
                            cap = max(block_size, cold_chunk_cap - decode_tokens)
                            num_new_tokens = min(num_new_tokens, cap)
                    else:
                        cap = max(block_size, cold_chunk_cap - decode_tokens)
                        num_new_tokens = min(num_new_tokens, cap)

        if sched_map is not None:
            sched_map[req_key] = num_new_tokens
        return num_new_tokens

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        self._last_cap_zero = None
        aligned = self._capped_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if aligned <= 0:
            why = getattr(self, "_last_cap_zero", None)
            self._note_waiting_stop(request, f"cap0:{why}" if why else "align0")
        return aligned

    def _capped_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> int:
        start = (
            getattr(request, "num_computed_tokens", 0)
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        num_new_tokens = self._cap_cold_prefill_tokens(
            request, start, num_new_tokens
        )
        if num_new_tokens <= 0:
            return 0
        retention = getattr(self, "_retention", None)
        if retention is not None:
            try:
                retention.prepare_prefill_split(request, start, num_new_tokens)
            except Exception:
                self._retention_failed()
                retention = None
        if getattr(self, "_single_step", False):
            aligned, cuts = single_step_prefill.plan_single_step_split(
                self, request, start, num_new_tokens, retention
            )
            single_step_prefill.set_plan(
                self.kv_cache_manager,
                request.request_id,
                (start, *cuts) if cuts else None,
            )
            sched_map = getattr(self, "_step_prefill_tokens_by_req", None)
            if sched_map is not None and id(request) in sched_map:
                sched_map[id(request)] = aligned
            return aligned
        skip = self._midblock_stop_skip_tokens(request, start, num_new_tokens)
        if skip > 0:
            # Run the upstream split from the next block boundary, as if the
            # mid-block chunk had already been scheduled, and merge the two.
            rest = super()._mamba_block_aligned_split(
                request,
                num_new_tokens - skip,
                num_new_local_computed_tokens + skip,
                num_external_computed_tokens,
            )
            aligned = skip + rest
            if rest > 0:
                self._n_midblock_stops_skipped = (
                    getattr(self, "_n_midblock_stops_skipped", 0) + 1
                )
        else:
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
        sched_map = getattr(self, "_step_prefill_tokens_by_req", None)
        if sched_map is not None and id(request) in sched_map:
            sched_map[id(request)] = aligned
        return aligned

    def _midblock_stop_skip_tokens(
        self, request: Request, start: int, num_new_tokens: int
    ) -> int:
        """Tokens from ``start`` to the next block boundary if the upstream
        split's mid-block stop may be merged into the following chunk, else 0.

        Upstream stops a chunk that starts mid-block (a fine-grained 128-token
        partial-tail hit) at the next block boundary, so that "slot p holds the
        state after (p + 1) * block_size tokens" for every block it may hash.
        With sparse retention (``prefix_cache_retention_interval=0``) a Mamba
        block is hashed only at a reachable boundary (the replay boundary or
        ``shared_prefix_boundary``) or as the prompt's partial tail, so the
        state written at that boundary is never cached nor read again; the
        runner seeds the state block forward across any number of block
        crossings (``TPUModelRunner._collect_mamba_state_seed_copies``). The
        skip is refused when an upstream-mandated stop (last cacheable block,
        partial-tail boundary, shared-prefix junction) lies in
        ``(start, boundary]``, or when the block ending at ``boundary`` is one
        that sparse retention hashes. Retention stops are applied afterwards
        from the original ``start``.
        """
        if not getattr(self, "_skip_midblock_align_stop", False):
            return 0
        if not getattr(self, "need_mamba_block_aligned_split", False) or getattr(
            self, "mamba_has_prefill_checkpoint_blocks", False
        ):
            return 0
        kvm = getattr(self, "kv_cache_manager", None)
        if getattr(kvm, "enable_caching", False):
            coordinator = getattr(kvm, "coordinator", None)
            # Dense retention would hash the stale mid-block state slot.
            if getattr(coordinator, "retention_interval", None) != 0:
                return 0
        block_size = self._get_block_size()
        cache_cfg = getattr(self, "cache_config", None)
        if getattr(cache_cfg, "block_size", block_size) != block_size:
            return 0
        if start % block_size == 0:
            return 0
        boundary = (start // block_size + 1) * block_size
        prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
        num_tokens = int(getattr(request, "num_tokens", prompt) or prompt)
        prefill_end = max(prompt, num_tokens - 1)
        if boundary >= prefill_end or num_new_tokens <= boundary - start:
            return 0
        hash_block_size = int(getattr(self, "hash_block_size", 0) or block_size)
        last_cache_position = num_tokens - num_tokens % block_size
        tail_boundary = prompt // hash_block_size * hash_block_size
        junction = int(getattr(request, "shared_prefix_boundary", 0) or 0)
        mandatory = (
            last_cache_position,
            last_cache_position - block_size,  # EAGLE back-off variant
            tail_boundary,
            tail_boundary - hash_block_size,  # EAGLE block-drop variant
            junction,
        )
        if any(start < s <= boundary for s in mandatory):
            return 0
        if getattr(kvm, "enable_caching", False):
            # MambaManager.reachable_block_mask hashes the state block ending at
            # the block floor of each replay boundary and of the junction. A
            # merged chunk leaves the hit's state in the block ending at
            # ``boundary``, so that block must not be one of them.
            reachable = (*kvm.coordinator.get_replay_boundaries(request), junction)
            if any(b // block_size * block_size == boundary for b in reachable):
                return 0
        return boundary - start

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if (
            getattr(self, "_kv_prefetch_in_waiting", False)
            and self._kv_prefetch_at_cap()
        ):
            # KV prefetch: this waiting request would join ``running`` without
            # an async load, past the cap. 0 stops the stock waiting loop.
            return 0
        why = None
        if not getattr(self, "need_mamba_block_aligned_split", False):
            self._last_cap_zero = None
            num_new_tokens = self._cap_cold_prefill_tokens(
                request, num_computed_tokens, num_new_tokens
            )
            why = getattr(self, "_last_cap_zero", None)
        reserved = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if reserved <= 0:
            self._note_waiting_stop(request, f"cap0:{why}" if why else "lookahead0")
        sched_map = getattr(self, "_step_prefill_tokens_by_req", None)
        if sched_map is not None and id(request) in sched_map:
            sched_map[id(request)] = reserved
        return reserved

    def _get_block_size(self) -> int:
        """Return KV cache block size in tokens (defaults to 16 if unset)."""
        block_size = getattr(self, "block_size", None)
        if (
            isinstance(block_size, int)
            and not isinstance(block_size, bool)
            and block_size > 0
        ):
            return block_size
        cache_cfg = getattr(self, "cache_config", None)
        cfg_block_size = getattr(cache_cfg, "block_size", None)
        if (
            isinstance(cfg_block_size, int)
            and not isinstance(cfg_block_size, bool)
            and cfg_block_size > 0
        ):
            return cfg_block_size
        return 16

    def _get_num_free_blocks(self) -> int | None:
        """Return free HBM blocks from BlockPool if available as an int."""
        block_pool = getattr(
            getattr(self, "kv_cache_manager", None), "block_pool", None
        )
        if block_pool is None:
            return None
        get_free = getattr(block_pool, "get_num_free_blocks", None)
        if not callable(get_free):
            return None
        try:
            free_blocks = get_free()
        except Exception:
            return None
        if isinstance(free_blocks, int) and not isinstance(free_blocks, bool):
            retention = getattr(self, "_retention", None) or getattr(
                self, "_hybrid_retention", None
            )
            num_pinned_fn = (
                getattr(retention, "num_pinned_blocks", None)
                if retention is not None
                else None
            )
            if callable(num_pinned_fn):
                free_blocks += int(num_pinned_fn())
            else:
                for lru_attr in ("_conv_lru", "_fork_lru"):
                    lru = (
                        getattr(retention, lru_attr, None)
                        if retention is not None
                        else None
                    )
                    if lru:
                        for entry in lru.values():
                            blocks = getattr(entry, "blocks", None)
                            if (
                                blocks is None
                                and isinstance(entry, tuple)
                                and len(entry) > 1
                            ):
                                blocks = entry[1]
                            if isinstance(blocks, (list, tuple, set)):
                                free_blocks += len(blocks)
            return free_blocks
        return None

    def _get_prefix_cache_epoch(self) -> int | None:
        """Return current BlockPool prefix-cache mutation epoch if backed by a
        real BlockPool, or None if mocked / unavailable."""
        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return None
        d = getattr(block_pool, "__dict__", None)
        if not isinstance(d, dict) or "cached_block_hash_to_block" not in d:
            return None
        return d.get("_tpu_prefix_cache_epoch", 0)

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of ``request`` resident in the KV cache."""
        kvm = self.kv_cache_manager
        if not kvm.enable_caching or request.skip_reading_prefix_cache:
            return 0
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens
        max_len = max(request.num_tokens - 1, 0)
        epoch = self._get_prefix_cache_epoch()
        cached = self._memo_get(request, epoch, max_len)
        if cached is not None:
            return cached[3]
        try:
            return self._memo_compute(request, epoch, max_len)[3]
        except Exception:
            if not self._paw_scoring_broken:
                self._paw_scoring_broken = True
                logger.exception(
                    "PrefixAwareScheduler: warmth scoring failed; falling back to FCFS."
                )
            return 0

    def _get_local_prefix_cache_hit(
        self, request: Request
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
                        # Fresh lists, as a fresh lookup would return; the
                        # memo itself must never be mutated downstream.
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

    def _external_warmth(self, request: Request, local_warmth: int) -> int:
        if getattr(self, "_is_kv_consumer", False):
            return 0
        connector = getattr(self, "connector", None)
        if connector is None:
            return 0
        fn = getattr(connector, "estimate_external_hit_tokens", None)
        if not callable(fn):
            return 0
        if request.num_tokens - local_warmth < self._get_block_size():
            return 0
        try:
            return max(0, int(fn(request, local_warmth) or 0))
        except Exception:
            return 0

    def _compute_warm_waiting_tokens(
        self, cold_threshold: int, ids: set[str] | None = None
    ) -> int:
        """Return block-aligned TPU prefill tokens needed on this step by
        ready warm requests in ``kv_holding_waiting`` (completed async CPU
        loads) or ``waiting`` (pure HBM cache hits). ``ids`` collects their
        request ids."""
        block_size = self._get_block_size()
        budget = int(getattr(self, "max_num_scheduled_tokens", 8192) or 8192)
        total = 0
        finished_recv = getattr(self, "finished_recving_kv_req_ids", None) or ()
        kv_holding = getattr(self, "kv_holding_waiting", None)
        if kv_holding:
            for r in kv_holding:
                # A finished load still reads WAITING_FOR_REMOTE_KVS until the
                # scan promotes it; one promoted on an earlier step but not
                # scheduled stays here unblocked.
                if r.request_id not in finished_recv and (
                    self._is_blocked_waiting_status(r.status)
                ):
                    continue
                rem = max(
                    getattr(r, "num_tokens", 0)
                    - getattr(r, "num_computed_tokens", 0),
                    1,
                )
                if rem <= cold_threshold:
                    total += (math.ceil(rem / block_size) + 1) * block_size
                    if ids is not None:
                        ids.add(r.request_id)
                    if total >= budget:
                        return budget
        waiting = getattr(self, "waiting", None)
        if waiting:
            for r in islice(waiting, self.WINDOW):
                local_warmth = self._prefix_warmth(r)
                if local_warmth <= 0:
                    continue
                rem = max(getattr(r, "num_tokens", 0) - local_warmth, 1)
                if rem > cold_threshold:
                    continue
                if self._external_warmth(r, local_warmth) > 0:
                    continue
                total += (math.ceil(rem / block_size) + 1) * block_size
                if ids is not None:
                    ids.add(r.request_id)
                if total >= budget:
                    return budget
        return total

    def _rank_and_warmth(
        self,
        request: Request,
        free_blocks_for_fit: int | None = None,
        now: float | None = None,
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth). Lower sort_key is admitted first.

        The key is (tier, uncached tokens, arrival time); the uncached tokens
        are reduced by TPU_SCHED_PAW_AGING_TOK_PER_S per second since arrival
        (see ``_wait_seconds``), which reorders requests within a tier only.
        """
        local_warmth = self._prefix_warmth(request)
        hbm_cold_tokens = max(request.num_tokens - local_warmth, 0)
        rate, _ = self._get_paw_aging_knobs()
        aging = 0.0
        if rate > 0:
            aging = rate * _wait_seconds(
                request, time.time() if now is None else now
            )
        if getattr(self, "_is_kv_consumer", False):
            tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
            return (tier, hbm_cold_tokens - aging, request.arrival_time), local_warmth
        if (
            free_blocks_for_fit is not None
            and local_warmth > 0
            and math.ceil(hbm_cold_tokens / self._get_block_size())
            <= free_blocks_for_fit
        ):
            return (
                (
                    _TIER_FREE_BLOCK_FIT,
                    hbm_cold_tokens - aging,
                    request.arrival_time,
                ),
                local_warmth,
            )
        if request.num_preemptions > 0:
            return (
                (_TIER_PREEMPTED, hbm_cold_tokens - aging, request.arrival_time),
                local_warmth,
            )
        if local_warmth > 0:
            return (
                (_TIER_NORMAL, hbm_cold_tokens - aging, request.arrival_time),
                local_warmth,
            )
        ext_warmth = self._external_warmth(request, 0)
        if ext_warmth > 0:
            ext_cold_tokens = max(request.num_tokens - ext_warmth, 0)
            return (
                (_TIER_NORMAL + 1, ext_cold_tokens - aging, request.arrival_time),
                ext_warmth,
            )
        return (
            (_TIER_NORMAL + 2, hbm_cold_tokens - aging, request.arrival_time),
            0,
        )

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        """Promotes the warmest request of each waiting queue to its head.

        vLLM's scan drains ``kv_holding_waiting`` (requests holding KV blocks,
        such as async loads) before ``waiting`` on every step, so ``waiting``
        is promoted even while ``kv_holding_waiting`` is not empty. Returns
        the queue the scan reads first.
        """
        kv_holding = getattr(self, "kv_holding_waiting", None)
        if kv_holding:
            self._promote_warmest(kv_holding)
        waiting = getattr(self, "waiting", None)
        if waiting:
            self._promote_warmest(waiting)
        return kv_holding or waiting

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
        if len(queue) < 2 or self.policy != SchedulingPolicy.FCFS:
            return

        window = list(islice(queue, self.WINDOW))
        head = window[0]
        now = self._now()
        head_scored = self._rank_and_warmth(head, now=now)
        head_local_warmth = self._prefix_warmth(head)

        free_blocks = (
            None
            if getattr(self, "_is_kv_consumer", False)
            else self._get_num_free_blocks()
        )
        free_blocks_for_fit: int | None = None
        if free_blocks is not None:
            head_cold_tokens = max(head.num_tokens - head_local_warmth, 0)
            head_needed_blocks = math.ceil(head_cold_tokens / self._get_block_size())
            if free_blocks < head_needed_blocks:
                free_blocks_for_fit = free_blocks

        scored = [(head_scored, head)] + [
            (
                self._rank_and_warmth(
                    r, free_blocks_for_fit=free_blocks_for_fit, now=now
                ),
                r,
            )
            for r in window[1:]
        ]
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
