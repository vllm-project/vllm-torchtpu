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
prioritizes requests with the fewest uncached prompt tokens. By default it
modifies only the admission ordering; eviction and preemption policies
remain untouched. The two decode options below also change how many remote
loads a step admits and the blocks reserved for them.

On a prefill/decode-disaggregated decode leg (``kv_role`` ``kv_consumer``)
it also runs decoding requests ahead of partial prefills, promotes the
warmest request of both waiting queues (``kv_holding_waiting`` and
``waiting``), memoises ``find_longest_cache_hit`` across steps, ships long
prompts to the workers as packed int32, trims re-sent token ids and logs
per-minute admission telemetry. Every other engine (no KV transfer, a KV
producer, or ``kv_both``) keeps the warmest-first admission alone. With the
``tpu_scheduler`` options (core/scheduler_options.py) a KV consumer also
prefetches remote KV for waiting requests past a full running set
(``kv_prefetch``, which lifts ``max_num_active_reqs`` for vLLM's waiting loop
and enforces the cap itself) and, with decode context parallelism, throttles
how many remote loads it admits (``decode_inflight_throttle``, through the
async-load block reservation). This scheduler does not sort the waiting
queue.

Enable with:
    --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler

Do not name it on a pipeline-parallel (PP > 1) leg: the platform then keeps
the configured class and turns pipeline chunk sizing off.
"""

import time
from bisect import bisect_right
from collections.abc import Sequence
from itertools import islice
from operator import itemgetter
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.core import prefix_change_log, scheduler_options
from vllm_torchtpu.core.packed_ids import pack_new_request_prompt_ids
from vllm_torchtpu.core.scheduler_utils import (
    format_counts,
    is_waiting_status,
    kv_roles,
    run_decoders_first,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger("vllm." + __name__)


_LOG_EVERY = 1000
# New requests whose prompt has at least this many tokens ship it packed.
_PACK_PROMPT_IDS_MIN_TOKENS = 1024
# Telemetry counts a step of at most this many Mamba blocks of tokens, taken
# while requests wait, as a small step.
_SMALL_STEP_BLOCKS = 2

# Priority tiers: lower values admit earlier.
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


def _is_decoding(request: Request) -> bool:
    return request.num_computed_tokens >= request.num_prompt_tokens > 0


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

    vLLM 0.33's ``Scheduler._make_cached_request_data`` re-sends the
    request's *full* token list for every running request that was not
    scheduled in the previous step; it is pickled into the SchedulerOutput
    and unpickled by every worker. The only reader on our path is
    ``GPUModelRunner._update_states`` (reached through
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


def memo_local_prefix_cache_hit(
    scheduler: Any, request: Request, max_cache_hit_length: int | None
) -> tuple["KVCacheBlocks", int, int, bool] | None:
    """The admission lookup answered from a still-exact memo, or None.

    The memo branch of ``_get_local_prefix_cache_hit``, shared by
    PrefixAwareScheduler and the pipeline chunk scheduler's prefix-aware
    admission; each falls back to its own lookup on None.
    """
    connector = getattr(scheduler, "connector", None)
    if connector is None or not connector.supports_divergent_local_hybrid_hits:
        kvm = scheduler.kv_cache_manager
        epoch = scheduler._get_prefix_cache_epoch()
        if (
            epoch is not None
            and not (
                getattr(kvm, "enable_kv_cache_events", False)
                and getattr(request, "kv_cache_report_mode", "incremental") == "full"
            )
            and kvm.prefix_cache_lookup_enabled(request)
        ):
            max_len = (
                request.num_tokens - 1
                if max_cache_hit_length is None
                else max_cache_hit_length
            )
            cached = scheduler._memo_get(request, epoch, max_len)
            if cached is not None:
                computed_blocks, num_new_computed_tokens, num_uncached = cached[2]
                shared_prefix_boundary = (
                    num_new_computed_tokens + num_uncached if num_uncached else 0
                )
                # Fresh lists, as a fresh lookup returns; the memo itself
                # must never be mutated downstream.
                blocks = kvm.create_kv_cache_blocks(
                    tuple(list(b) for b in computed_blocks)
                )
                return (
                    blocks,
                    num_new_computed_tokens,
                    shared_prefix_boundary,
                    False,
                )
    return None


class PrefixAwareAdmission(Scheduler):
    """The prefix-aware admission PrefixAwareScheduler and the pipeline chunk
    scheduler's ``prefix_aware`` option share: the cross-step memo of
    ``find_longest_cache_hit``, a waiting request's prefix warmth, the
    warmest-first promotion of the waiting queues and packed prompt ids.

    Each scheduler inherits it and ranks its own promotion window
    (``_rank_window``); it is not a scheduler of its own. Its methods call
    no ``super()``. The inheriting scheduler's ``__init__`` sets
    ``_paw_scoring_broken`` and the promotion counters (``_paw_n``,
    ``_paw_cold_tokens``, ``_paw_prompt_tokens``), and turns the memo on
    with ``_enable_memo_xstep`` (``_memo_xstep``).
    """

    #: Lookahead window of ready requests to evaluate for cached prefix length.
    WINDOW = 64

    _paw_scoring_broken: bool
    _paw_n: int
    _paw_cold_tokens: int
    _paw_prompt_tokens: int
    _memo_xstep: bool

    # ------------------------------------------------------------------
    # Cross-step memo of find_longest_cache_hit.
    #
    # Within one prefix-cache epoch a memoised result is reused as is. Any
    # prefix-cache mutation (a hash insert, an eviction, a step start with
    # unwritten blocks) moves the epoch, which happens on almost every step,
    # so reuse across epochs needs the change log.
    #
    # find_longest_cache_hit(hashes, max_len) is a deterministic function of
    # the answers BlockPool.get_cached_block(h, groups) gives for the hashes it
    # probes. With the memo on, core/prefix_change_log.py records the probed
    # hashes during the computation and logs every hash whose answer can
    # change (insert, removal, unwritten-flag flip, reset) on top of the
    # same-step patch. A memo is reused iff none of its probed hashes was
    # logged after its epoch, so the reused result equals a fresh lookup
    # exactly. The first _MEMO_VERIFY_REUSES cross-step reuses are re-checked
    # against a fresh lookup; any mismatch disables the cross-step memo.
    # ------------------------------------------------------------------
    _MEMO_LOG_CAP = 65536
    _MEMO_VERIFY_REUSES = 64

    def _enable_memo_xstep(self) -> None:
        """Memoises lookups on this scheduler's block pool: starts its
        prefix-cache epoch (the same-step memo) and its change log (the
        cross-step memo), which the patch registry's wrappers keep."""
        self._memo_hits = 0
        self._memo_computes = 0
        self._memo_verified = 0
        self._memo_verify_left = self._MEMO_VERIFY_REUSES
        pool = self.kv_cache_manager.block_pool
        self._memo_xstep = prefix_change_log.enable_epoch(
            pool
        ) and prefix_change_log.enable(pool)

    def _memo_trim_log(self) -> None:
        pool = getattr(self.kv_cache_manager, "block_pool", None)
        d = getattr(pool, "__dict__", None)
        log = d.get("_tpu_prefix_change_log") if isinstance(d, dict) else None
        if isinstance(d, dict) and log is not None and len(log) > self._MEMO_LOG_CAP:
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
        return all(log[i][1] not in probes for i in range(start, len(log)))

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
                    "%s: cross-step memo mismatch for %s "
                    "(memo=%s fresh=%s); disabling the cross-step memo.",
                    type(self).__name__,
                    getattr(request, "request_id", "?"),
                    _hit_signature(cached[2])[:2],
                    _hit_signature(fresh[2])[:2],
                )
                self._memo_xstep = False
                # Nothing trims the change log once the memo is off.
                prefix_change_log.disable(self.kv_cache_manager.block_pool)
                return fresh
            self._memo_verified += 1
            return fresh
        cached = (epoch,) + cached[1:]
        cast(Any, request)._paw_hit_cache = cached
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
            cast(Any, request)._paw_hit_cache = cached
        return cached

    def _get_block_size(self) -> int:
        """The scheduler's KV cache block size in tokens."""
        return self.block_size

    def _get_prefix_cache_epoch(self) -> int | None:
        """Return the prefix-cache mutation epoch of the block pool, or None
        when the pool does not count mutations (see
        ``prefix_change_log.enable_epoch``)."""
        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return None
        d = getattr(block_pool, "__dict__", None)
        if not isinstance(d, dict):
            return None
        # Without the epoch counters the epoch never moves, and a same-epoch
        # memo would be reused after the cache changed.
        if not prefix_change_log.epoch_installed():
            return None
        return d.get(prefix_change_log.EPOCH)

    def _prefix_warmth(self, request: Request) -> int:
        """Prompt tokens of ``request`` resident in the KV cache."""
        return self._prefix_hit(request)[0]

    def _prefix_hit(
        self, request: Request
    ) -> tuple[int, tuple[Sequence[KVCacheBlock], ...] | None]:
        """(prompt tokens of ``request`` resident in the KV cache, the hit's
        blocks per KV cache group). The blocks are None without a lookup to
        take them from: prefix caching off, a request that skips it, one that
        has computed tokens (it holds its blocks) or a failed lookup."""
        kvm = self.kv_cache_manager
        if not kvm.enable_caching or request.skip_reading_prefix_cache:
            return 0, None
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens, None
        max_len = max(request.num_tokens - 1, 0)
        epoch = self._get_prefix_cache_epoch()
        try:
            # A memo reuse can recompute the lookup to verify it.
            cached = self._memo_get(request, epoch, max_len)
            if cached is None:
                cached = self._memo_compute(request, epoch, max_len)
            return cached[3], cached[2][0]
        except Exception:
            if not self._paw_scoring_broken:
                self._paw_scoring_broken = True
                logger.exception(
                    "%s: warmth scoring failed; falling back to FCFS.",
                    type(self).__name__,
                )
            return 0, None

    def _rank_window(
        self, window: list[Request]
    ) -> list[tuple[tuple[int, float, float], int]]:
        """(sort key, warmth) of every request of a promotion window, head
        first; a lower sort key is admitted first."""
        raise NotImplementedError

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        """Promotes the warmest request of each waiting queue to its head.

        vLLM's scan drains ``kv_holding_waiting`` (requests holding KV blocks,
        such as async loads) before ``waiting`` on every step, so ``waiting``
        is promoted even while ``kv_holding_waiting`` is not empty. Returns
        the queue the scan reads first.
        """
        if self.kv_holding_waiting:
            self._promote_warmest(self.kv_holding_waiting)
        if self.waiting:
            self._promote_warmest(self.waiting)
        return self.kv_holding_waiting or self.waiting

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
        if len(queue) < 2 or self.policy != SchedulingPolicy.FCFS:
            return

        window = list(islice(queue, self.WINDOW))
        head = window[0]
        scored = list(zip(self._rank_window(window), window, strict=True))
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

    def _maybe_pack_prompt_ids(self, out) -> None:
        """Ship new requests' prompt ids to the workers as packed int32 (see
        core/packed_ids.py)."""
        if getattr(self, "_pack_ids", None) is None:
            self._pack_ids = True
            self._pack_ids_min = _PACK_PROMPT_IDS_MIN_TOKENS
        if not self._pack_ids:
            return
        try:
            pack_new_request_prompt_ids(out, self._pack_ids_min)
        except Exception:
            logger.exception(
                "%s: prompt-id packing failed; disabling it.", type(self).__name__
            )
            self._pack_ids = False


class PrefixAwareScheduler(PrefixAwareAdmission, AsyncScheduler):
    """Admit ready requests prioritizing fewest uncached prompt tokens."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        # The decode-leg levers were measured on the decode legs of
        # prefill/decode disaggregation only. kv_both (an offloading connector
        # on an aggregated engine) is not a consumer and keeps the admission.
        self._is_kv_consumer = kv_roles(self.vllm_config.kv_transfer_config).consumer
        options = scheduler_options.get(self.vllm_config)
        self._promote_warmest_switch = options.promote_warmest
        if options.kv_prefetch and not (
            self._is_kv_consumer and self.connector is not None
        ):
            raise ValueError(
                f"additional_config.{scheduler_options.SECTION}.kv_prefetch "
                "starts remote KV loads; it needs a KV consumer with a KV connector"
            )
        if options.decode_inflight_throttle and not (
            self._is_kv_consumer and self.dcp_world_size > 1
        ):
            raise ValueError(
                f"additional_config.{scheduler_options.SECTION}."
                "decode_inflight_throttle needs a KV consumer with "
                "decode_context_parallel_size > 1"
            )
        self._kv_prefetch_switch = options.kv_prefetch
        self._decode_throttle_switch = options.decode_inflight_throttle
        self._memo_xstep = False
        if self._is_kv_consumer:
            self._enable_memo_xstep()
        self._init_kv_prefetch()

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, decode-leg levers=%s, "
            "memo_xstep=%s)",
            self.WINDOW,
            self._is_kv_consumer,
            self._memo_xstep,
        )

    # ------------------------------------------------------------------
    # Decode KV prefetch past a full running set (option
    # tpu_scheduler.kv_prefetch, on a KV consumer with a connector;
    # N = _KV_PREFETCH_MAX_HELD).
    #
    # vLLM 0.33's waiting loop stops at ``len(running) >=
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
    #     _KV_PREFETCH_RESERVE_BLOCKS kept for the running set
    #     (_inflight_prefill_reserved_blocks).
    # Below the cap the loop is unchanged. When a slot frees, the stock loop
    # drains ``kv_holding_waiting`` first, so the oldest KV-ready prefetched
    # request takes it on the same step. Aborts of prefetched requests free
    # their blocks through the stock WAITING_FOR_REMOTE_KVS path.
    # ------------------------------------------------------------------
    _KV_PREFETCH_LOG_INTERVAL_S = 60.0
    # Requests held for a free slot at most, and blocks kept for the running
    # set (every decode leg of the validation entries runs 8 and the 512 default).
    _KV_PREFETCH_MAX_HELD = 8
    _KV_PREFETCH_RESERVE_BLOCKS = 512

    def _init_kv_prefetch(self) -> None:
        # __init__ refuses the option on any other leg than a KV consumer
        # with a connector.
        self._kv_prefetch_n = (
            self._KV_PREFETCH_MAX_HELD if self._kv_prefetch_switch else 0
        )
        self._kv_prefetch_reserve = self._KV_PREFETCH_RESERVE_BLOCKS
        # Real admission cap while schedule() runs with prefetch on, else None.
        self._kv_prefetch_cap: int | None = None
        self._kv_prefetch_in_waiting = False
        self._kv_prefetch_cands: set[int] = set()
        self._kv_prefetch_before: set[int] = set()
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
        cap = self._kv_prefetch_cap
        if cap is None:
            return False
        return len(self.running) + self.num_waiting_for_streaming_input >= cap

    def _kv_prefetch_num_held(self) -> int:
        """Requests holding KV for an async load (loading or loaded) that have
        not run yet."""
        return sum(
            1 for r in self._inflight_prefills if r.status != RequestStatus.RUNNING
        )

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        if self._kv_prefetch_cap is not None:
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
        if not self._kv_prefetch_n:
            return False
        cap = self.max_num_active_reqs
        num_running = len(self.running) + self.num_waiting_for_streaming_input
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
        before = self._kv_prefetch_before
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
                "turning KV prefetch off.",
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
            len(self.waiting) + len(self.kv_holding_waiting),
        )

    def _promote_warmest_on(self) -> bool:
        """``tpu_scheduler.promote_warmest`` (default on)."""
        return self._promote_warmest_switch

    def _inflight_prefill_reserved_blocks(self) -> int:
        reserved = self._decode_inflight_reserved_blocks()
        if self._kv_prefetch_in_waiting and self._kv_prefetch_at_cap():
            # The stock loop calls this only for an async load, right before
            # allocating it. At the cap that load is a prefetch.
            if self._kv_prefetch_num_held() >= self._kv_prefetch_n:
                self._kv_prefetch_full_refusals += 1
                return 1 << 30
            return reserved + self._kv_prefetch_reserve
        return reserved

    # Decode admission throttle (option tpu_scheduler.decode_inflight_throttle,
    # on a KV consumer with DCP > 1): refuse a new remote load while this many
    # loads are in flight; otherwise reserve the largest of vLLM's own
    # async-load reservation, the reserved-block minimum and the per-sequence
    # blocks x (running + in flight). Every decode leg of the validation
    # entries with DCP 4 runs these values.
    _DECODE_MAX_INFLIGHT_LOADS = 16
    _DECODE_RESERVED_BLOCKS = 128
    _DECODE_RESERVED_BLOCKS_PER_SEQ = 8

    def _decode_inflight_reserved_blocks(self) -> int:
        """The stock async-load reservation, or, with the
        decode_inflight_throttle option, the decode admission throttle
        (constants above)."""
        base = int(super()._inflight_prefill_reserved_blocks() or 0)
        if not self._decode_throttle_switch:
            return base
        num_inflight = len(self._inflight_prefills)
        num_running = len(self.running)
        if num_inflight >= self._DECODE_MAX_INFLIGHT_LOADS:
            return 1 << 30
        min_reserved = self._DECODE_RESERVED_BLOCKS
        per_seq_reserved = self._DECODE_RESERVED_BLOCKS_PER_SEQ
        return max(
            base, max(min_reserved, per_seq_reserved * (num_running + num_inflight))
        )

    def _note_waiting_stop(self, request: Request, reason: str) -> None:
        """Records why the waiting loop stopped at ``request`` (first stop of
        the step wins; running requests are ignored)."""
        if (
            self._is_kv_consumer
            and getattr(self, "_step_wait_event", None) is None
            and is_waiting_status(request)
        ):
            self._step_wait_event = reason

    def _make_cached_request_data(self, *args, **kwargs):
        cached = super()._make_cached_request_data(*args, **kwargs)
        if not self._is_kv_consumer:
            return cached
        if getattr(self, "_trim_resend", None) is None:
            self._trim_resend = True
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

    def schedule(self, *args, **kwargs):
        if not self._is_kv_consumer:
            # Off a decode leg: the warmest-first admission alone, unless
            # tpu_scheduler.promote_warmest=false asks for FCFS admission.
            if self._promote_warmest_on():
                self._promote_warmest(self.kv_holding_waiting or self.waiting)
            return super().schedule(*args, **kwargs)
        # Decoders first, prefills in their order: a prefill ahead of a
        # decoder can leave it no tokens, and preemption takes the last
        # prefill before any decoder.
        decoding, prefilling = run_decoders_first(self.running, _is_decoding)
        self._step_wait_event = None
        waiting_at_start = len(self.waiting)
        if getattr(self, "_memo_xstep", False):
            self._memo_trim_log()
        if self._kv_prefetch_n:
            try:
                self._kv_prefetch_begin_step()
            except Exception:
                logger.exception(
                    "PrefixAwareScheduler: KV prefetch setup failed; turning "
                    "KV prefetch off."
                )
                self._kv_prefetch_n = 0
                self._kv_prefetch_end_step()
        try:
            # vLLM's Scheduler walks the waiting queues without a selection
            # hook, so the warmest-first promotion runs here, once per step.
            # tpu_scheduler.promote_warmest=false skips it: FCFS admission.
            if self._promote_warmest_on():
                self._select_waiting_queue_for_scheduling()
            out = super().schedule(*args, **kwargs)
            if getattr(out, "scheduled_new_reqs", None):
                self._maybe_pack_prompt_ids(out)
            stop = self._waiting_loop_stop_reason(out, waiting_at_start)
            self._record_step_telemetry(out, waiting_at_start, stop)
            now = time.monotonic()
            if now - getattr(self, "_last_sched_log_t", 0.0) >= 60.0:
                self._last_sched_log_t = now
                logger.info(
                    "prefix-scheduler telemetry: running=%d waiting=%d decoding=%d "
                    "prefilling=%d sched_tokens=%d %s",
                    len(self.running),
                    len(self.waiting),
                    len(decoding),
                    len(prefilling),
                    getattr(out, "total_num_scheduled_tokens", 0),
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
            if self._kv_prefetch_cap is not None:
                self._kv_prefetch_end_step()
            if self._kv_prefetch_n:
                self._kv_prefetch_maybe_log()
            self._step_wait_event = None

    def _waiting_loop_stop_reason(self, out, waiting_at_start: int) -> str:
        """Names what ended vLLM's waiting loop on this step.

        ``align0`` / ``lookahead0``: the chunk of the request at the head came
        out 0 tokens (the Mamba block-aligned split, the MTP lookahead
        reserve). Otherwise ``idle`` (nothing was waiting), ``preempted`` (the
        loop did not run), ``max_running``, ``budget`` (no token budget left),
        ``drained`` (every waiting request was admitted or skipped) or
        ``other``.
        """
        event = getattr(self, "_step_wait_event", None)
        if event is not None:
            return event
        if waiting_at_start == 0 and not self.kv_holding_waiting:
            return "idle"
        if getattr(out, "preempted_req_ids", None):
            return "preempted"
        # The admission check the scan stops at. On a KV prefetch step,
        # max_num_active_reqs is lifted until the step ends; the cap is the
        # value prefetch saved.
        num_active = len(self.running) + self.num_waiting_for_streaming_input
        cap = self._kv_prefetch_cap
        if num_active >= (self.max_num_active_reqs if cap is None else cap):
            return "max_running"
        if (
            getattr(out, "total_num_scheduled_tokens", 0)
            >= self.max_num_scheduled_tokens
        ):
            return "budget"
        if not self._has_undeferred_waiting():
            return "drained"
        return "other"

    def _has_undeferred_waiting(self) -> bool:
        """Whether ``waiting`` holds a request vLLM has not deferred.

        The scan puts a skipped request that holds no KV blocks back into
        ``waiting`` and keeps it in ``deferred_waiting`` until it is scheduled.
        """
        return any(r not in self.deferred_waiting for r in self.waiting)

    def _record_step_telemetry(self, out, waiting_at_start: int, stop: str) -> None:
        """Accumulates the per-minute step counters of the telemetry line.

        A small step counts only when requests are still waiting after it, so
        a step that admitted every waiting request with a short chunk is not
        one.
        """
        tm: dict[str, Any] | None = getattr(self, "_tm", None)
        if tm is None:
            tm = self._tm = cast(
                dict[str, Any],
                {
                    "steps": 0,
                    "small_steps_waiting": 0,
                    "stops": {},
                },
            )
        tm["steps"] += 1
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
        tm = getattr(self, "_tm", None) or {}
        pool_free = self._get_num_free_blocks()
        stops = dict(sorted((tm.get("stops") or {}).items(), key=lambda kv: -kv[1]))
        # The decode throttle's reservation (vLLM's own without it); not the
        # prefetch-aware hook, which counts refusals while prefetch is armed.
        reserved = self._decode_inflight_reserved_blocks()
        text = (
            f"stop={stop} "
            f"inflight_reserved={reserved} pool_free={pool_free} "
            f"min_steps={tm.get('steps', 0)} "
            f"min_small_steps_waiting={tm.get('small_steps_waiting', 0)} "
            f"min_stops={format_counts(stops)}"
        )
        tm["steps"] = 0
        tm["small_steps_waiting"] = 0
        tm["stops"] = {}
        return text

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        aligned = super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if aligned <= 0:
            self._note_waiting_stop(request, "align0")
        return aligned

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if self._kv_prefetch_in_waiting and self._kv_prefetch_at_cap():
            # KV prefetch: this waiting request would join ``running`` without
            # an async load, past the cap. vLLM 0.33 requeues a request whose
            # chunk comes out 0 and keeps scanning; putting the real cap back
            # ends the scan, so nothing is pulled ahead of this request.
            cap = self._kv_prefetch_cap
            if cap is not None:
                self.max_num_active_reqs = cap
            return 0
        reserved = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if reserved <= 0:
            self._note_waiting_stop(request, "lookahead0")
        return reserved

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
            return free_blocks
        return None

    def _get_local_prefix_cache_hit(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple["KVCacheBlocks", int, int, bool]:
        """Serves the admission lookup from a still-exact memo.

        Without divergent hybrid hits, vLLM 0.33's
        ``Scheduler._get_local_prefix_cache_hit`` returns
        ``KVCacheManager.get_computed_blocks`` and ``False``. For a memoised
        ``find_longest_cache_hit`` result the memo branch returns what that
        method returns: the same blocks (fresh lists), hit length and
        shared-prefix boundary. It only skips the lookup. That method records
        no prefix-cache stats (the scheduler records them at admission), and
        requests whose ``full`` KV-event report mode makes it emit events go
        to it.
        """
        hit = memo_local_prefix_cache_hit(self, request, max_cache_hit_length)
        if hit is not None:
            return hit
        return super()._get_local_prefix_cache_hit(request, max_cache_hit_length)

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int]:
        """Return (sort_key, warmth). Lower sort_key is admitted first."""
        warmth = self._prefix_warmth(request)
        cold_tokens = float(request.num_tokens - warmth)
        tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        return (tier, cold_tokens, request.arrival_time), warmth

    def _rank_window(
        self, window: list[Request]
    ) -> list[tuple[tuple[int, float, float], int]]:
        return [self._rank_and_warmth(r) for r in window]
