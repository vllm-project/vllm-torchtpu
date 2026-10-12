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

import math
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
from vllm.v1.request import Request

from vllm_torchtpu.core.packed_ids import pack_new_request_prompt_ids

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000

# Priority tiers: lower values admit earlier.
_TIER_FREE_BLOCK_FIT = 0
_TIER_PREEMPTED = 1
_TIER_NORMAL = 2


def _is_decoding(request: Request) -> bool:
    return request.num_computed_tokens >= request.num_prompt_tokens > 0
def _hit_signature(result) -> tuple:
    """Comparable form of a find_longest_cache_hit result (block ids)."""
    blocks = result[0]
    try:
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
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        self._enable_memo_xstep()

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, decoders first, "
            self.WINDOW,
            getattr(self, "_memo_xstep", False),
        )

    # ------------------------------------------------------------------
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
    # against a fresh lookup; any mismatch disables the cross-step memo.
    # ------------------------------------------------------------------
    _MEMO_LOG_CAP = 65536

    def _enable_memo_xstep(self) -> None:
        self._memo_xstep = False
        self._memo_hits = 0
        self._memo_computes = 0
        self._memo_verified = 0
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


        """
    def _maybe_pack_prompt_ids(self, out) -> None:
        if getattr(self, "_pack_ids", None) is None:
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
        connector = getattr(self, "connector", None)
        if connector is not None:
            poll_fn = getattr(connector, "poll_and_complete_transfers", None)
            if callable(poll_fn):
                try:
                except Exception:
                    logger.exception(
                        "PrefixAwareScheduler: poll_and_complete_transfers failed."
                    )

        # Decoders first: vLLM schedules ``running`` in list order and a
        # partial prefill takes min(remaining, budget), so a prefill ahead of
        # a decoder can leave it no tokens. Stably move every decoding request
        decoding = [r for r in self.running if _is_decoding(r)]
        prefilling = [r for r in self.running if not _is_decoding(r)]
            self.running[:] = decoding + prefilling
        if getattr(self, "_memo_xstep", False):
            self._memo_trim_log()
        try:
            # hook, so the warmest-first promotion runs here, once per step.
            out = super().schedule(*args, **kwargs)
            if getattr(out, "scheduled_new_reqs", None):
                self._maybe_pack_prompt_ids(out)
            now = time.monotonic()
            if now - getattr(self, "_last_sched_log_t", 0.0) >= 60.0:
                self._last_sched_log_t = now
                logger.info(
                    "prefix-scheduler telemetry: running=%d waiting=%d decoding=%d "
                    len(self.running),
                    len(getattr(self, "waiting", ())),
                    len(decoding),
                    len(prefilling),
                    self._get_num_free_blocks(),
                    getattr(out, "total_num_scheduled_tokens", 0),
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

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        reserved = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        return reserved

    def _get_block_size(self) -> int:
        """Return KV cache block size in tokens (defaults to 16 if unset)."""
        block_size = getattr(self, "block_size", None)
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
    ) -> tuple[object, int, int, bool]:
        connector = getattr(self, "connector", None)
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
                    )
                    blocks = kvm.create_kv_cache_blocks(computed_blocks)
                    return (
                        blocks,
                        num_new_computed_tokens,
                        shared_prefix_boundary,
                        False,
                    )
        blocks, num_local, shared_prefix_boundary, hit_diverged = (
        )
            kvm = self.kv_cache_manager
            reconciled_blocks, num_reconciled, reconciled_boundary = (
            )
            block_size = self._get_block_size()
            target_local = connector.resolve_divergent_local_hit_tokens(
                request, num_local, num_reconciled, block_size
            )
            if target_local is None:
                return (
                    reconciled_blocks,
                    num_reconciled,
                    reconciled_boundary,
                    False,
                )
            if target_local <= 0:
                return (
                    kvm.empty_kv_cache_blocks,
                    0,
                    reconciled_boundary,
                    False,
                )
            if target_local < num_local:
                blocks = kvm.truncate_computed_blocks(blocks, target_local)
                num_local = target_local
        return blocks, num_local, shared_prefix_boundary, hit_diverged

    def _rank_and_warmth(
        self,
        request: Request,
        free_blocks_for_fit: int | None = None,
    ) -> tuple[tuple[int, float, float], int]:
        warmth = self._prefix_warmth(request)
        cold_tokens = float(request.num_tokens - warmth)
        tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        return (tier, cold_tokens, request.arrival_time), warmth

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:


    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
            return

        window = list(islice(queue, self.WINDOW))
        head = window[0]

        free_blocks_for_fit: int | None = None
        if free_blocks is not None:
            if free_blocks < head_needed_blocks:
                free_blocks_for_fit = free_blocks

        scored = [(head_scored, head)] + [
            (
                self._rank_and_warmth(
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
