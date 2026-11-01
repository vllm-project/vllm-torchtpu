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
requests while bounding mixed-step chunked prefill interference on active decode
batches.
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
from vllm.v1.request import Request

from vllm_torchtpu import envs
from vllm_torchtpu.core import prefix_change_log, single_step_prefill
from vllm_torchtpu.core.hybrid_retention import HybridStateRetention
from vllm_torchtpu.core.packed_ids import pack_new_request_prompt_ids

logger = init_logger("vllm." + __name__)

_LOG_EVERY = 1000

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


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


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
        self._paw_n = 0
        self._paw_cold_tokens = 0
        self._paw_prompt_tokens = 0
        self._paw_scoring_broken = False
        self._srpf_running: bool = bool(envs.TPU_SCHED_SRPF_RUNNING)
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
            else max(1024, budget // 2)
        )
        self._cold_chunk_cap: int = (
            int(envs.TPU_SCHED_COLD_CHUNK_CAP)
            if "TPU_SCHED_COLD_CHUNK_CAP" in os.environ
            else max(512, self._step_total_cap // 2)
        )
        self._cold_min_decoders: int = (
            int(envs.TPU_SCHED_COLD_MIN_DECODERS)
            if "TPU_SCHED_COLD_MIN_DECODERS" in os.environ
            else max(1, (3 * eff_seqs) // 4)
        )
        self._warm_streak: int = int(envs.TPU_SCHED_WARM_STREAK)
        self._warm_min_decoders: int = int(envs.TPU_SCHED_WARM_MIN_DECODERS)
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
        if single_step_prefill.enabled():
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
        if os.environ.get("TPU_HYBRID_STATE_RETENTION", "1") != "0":
            retention = HybridStateRetention(self.kv_cache_manager)
            if retention.enabled:
                self._retention = retention
        self._enable_memo_xstep()

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, decoders first, "
            "srpf_running=%s, cold_chunk_cap=%d, cold_threshold=%d, "
            "cold_min_decoders=%d, step_total_cap=%d, mk_max_running=%d, "
            "warm_streak=%d@%d, hybrid_state_retention=%s, memo_xstep=%s, "
            "skip_midblock_align_stop=%s, single_step_prefill=%s)",
            self.WINDOW,
            self._srpf_running,
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
        vcfg = getattr(self, "vllm_config", None)
        add_cfg = getattr(vcfg, "additional_config", None) or {}
        mk_active = (
            _env_on("VLLM_TPU_QWEN35_MEGAKERNEL", "0")
            or bool(envs.VLLM_TPU_MEGAKERNEL)
            or bool(isinstance(add_cfg, dict) and add_cfg.get("megakernel"))
        )
        if not mk_active:
            return 0
        if int(getattr(self, "dcp_world_size", 1) or 1) != 1:
            return 0
        if not _env_on("VLLM_TPU_QWEN35_MK_VERIFY_48", "1"):
            mk_max_verify_rows = 32
        elif not _env_on("VLLM_TPU_QWEN35_MK_VERIFY_64", "1"):
            mk_max_verify_rows = 48
        else:
            mk_max_verify_rows = 64
        num_spec = max(int(getattr(self, "num_spec_tokens", 3) or 0), 0)
        if num_spec > 0:
            return max(1, mk_max_verify_rows // (1 + num_spec))
        return 32

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
        running_list = getattr(self, "running", None)
        retention = getattr(self, "_retention", None)
        if retention is not None and running_list is not None:
            try:
                retention.pin_running(running_list)
            except Exception:
                self._retention_failed()
        # Decoders first: vLLM schedules ``running`` in list order and a
        # partial prefill takes min(remaining, budget), so a prefill ahead of
        # a decoder can leave it no tokens. Stably move every decoding request
        # ahead of every prefilling one. When TPU_SCHED_SRPF_RUNNING is enabled,
        # also stably order prefilling requests by fewest remaining prompt
        # tokens (Shortest-Remaining-Prefill-First) so warm turns finish ahead
        # of multi-step cold prefills.
        srpf_running, _, _, _ = self._get_sched_knobs()
        if running_list is not None:
            decoding = [r for r in running_list if _is_decoding(r)]
            prefilling = [r for r in running_list if not _is_decoding(r)]
            if srpf_running and len(prefilling) > 1:
                prefilling.sort(key=_remaining_prefill_tokens)
            if prefilling and (decoding or (srpf_running and len(prefilling) > 1)):
                running_list[:] = decoding + prefilling
        else:
            decoding = []
            prefilling = []
        self._step_num_decoding = len(decoding)
        self._step_decode_tokens = sum(
            max(
                getattr(r, "num_tokens_with_spec", 0)
                + getattr(r, "num_output_placeholders", 0)
                - getattr(r, "num_computed_tokens", 0),
                1,
            )
            for r in decoding
        )
        self._step_running_prefill_ids = {id(r) for r in prefilling}
        self._step_prefill_tokens_by_req = {}
        if getattr(self, "_memo_xstep", False):
            self._memo_trim_log()
        single_step = getattr(self, "_single_step", False)
        if single_step:
            single_step_prefill.begin_step(self.kv_cache_manager)
        mk_max_running = getattr(self, "_mk_max_running", None)
        if mk_max_running is None:
            mk_max_running = self._resolve_mk_max_running()
        orig_max_running = getattr(self, "max_num_running_reqs", None)
        if (
            mk_max_running > 0
            and isinstance(orig_max_running, int)
            and orig_max_running > mk_max_running
        ):
            self.max_num_running_reqs = mk_max_running
        try:
            if not hasattr(super(), "_select_waiting_queue_for_scheduling"):
                for q_attr in ("kv_holding_waiting", "waiting"):
                    queue = getattr(self, q_attr, None)
                    if queue:
                        self._promote_warmest(queue)
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
            else:
                self._pure_decode_streak = (
                    getattr(self, "_pure_decode_streak", 999999) + 1
                )
            now = time.monotonic()
            if now - getattr(self, "_last_sched_log_t", 0.0) >= 60.0:
                self._last_sched_log_t = now
                logger.info(
                    "prefix-scheduler telemetry: running=%d waiting=%d decoding=%d "
                    "prefilling=%d free_blocks=%s sched_tokens=%d "
                    "pure_decode_streak=%d midblock_stops_skipped=%d "
                    "single_step_reqs=%d",
                    len(running_list) if running_list is not None else 0,
                    len(getattr(self, "waiting", ())),
                    len(decoding),
                    len(prefilling),
                    self._get_num_free_blocks(),
                    getattr(out, "total_num_scheduled_tokens", 0),
                    self._pure_decode_streak,
                    getattr(self, "_n_midblock_stops_skipped", 0),
                    getattr(self, "_n_single_step_reqs", 0),
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
            if orig_max_running is not None:
                self.max_num_running_reqs = orig_max_running
            self._step_running_prefill_ids = set()
            self._step_prefill_tokens_by_req = None

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
                return 0
            num_new_tokens = min(num_new_tokens, rem_step_cap)

        if cold_chunk_cap > 0 and cold_threshold > 0 and rem > cold_threshold:
            in_running = req_key in getattr(self, "_step_running_prefill_ids", ())
            has_waiting = bool(getattr(self, "waiting", None)) or bool(
                getattr(self, "skipped_waiting", None)
            )
            if 0 < cold_min_decoders <= num_decoding or (in_running and has_waiting):
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
        start = (
            getattr(request, "num_computed_tokens", 0)
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        num_new_tokens = self._cap_cold_prefill_tokens(request, start, num_new_tokens)
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
        ``(start, boundary]``. Retention stops are applied afterwards from the
        original ``start``.
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
        mandatory = (
            last_cache_position,
            last_cache_position - block_size,  # EAGLE back-off variant
            prompt // hash_block_size * hash_block_size,
            int(getattr(request, "shared_prefix_boundary", 0) or 0),
        )
        if any(start < s <= boundary for s in mandatory):
            return 0
        return boundary - start

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if not getattr(self, "need_mamba_block_aligned_split", False):
            num_new_tokens = self._cap_cold_prefill_tokens(
                request, num_computed_tokens, num_new_tokens
            )
        reserved = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
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
        """Return (sort_key, warmth). Lower sort_key is admitted first."""
        warmth = self._prefix_warmth(request)
        cold_tokens = float(max(request.num_tokens - warmth, 0))
        if (
            free_blocks_for_fit is not None
            and warmth > 0
            and math.ceil(cold_tokens / self._get_block_size()) <= free_blocks_for_fit
        ):
            tier = _TIER_FREE_BLOCK_FIT
        elif request.num_preemptions > 0:
            tier = _TIER_PREEMPTED
        elif warmth > 0:
            tier = _TIER_NORMAL
        else:
            tier = _TIER_NORMAL + 1
        return (tier, cold_tokens, request.arrival_time), warmth

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        return queue

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Promote the request with the fewest uncached tokens within WINDOW."""
        if self.policy != SchedulingPolicy.FCFS or len(queue) < 2:
            return

        window = list(islice(queue, self.WINDOW))
        head = window[0]
        head_scored = self._rank_and_warmth(head)
        head_warmth = head_scored[1]

        free_blocks = self._get_num_free_blocks()
        free_blocks_for_fit: int | None = None
        if free_blocks is not None:
            head_cold_tokens = max(head.num_tokens - head_warmth, 0)
            head_needed_blocks = math.ceil(head_cold_tokens / self._get_block_size())
            if free_blocks < head_needed_blocks:
                free_blocks_for_fit = free_blocks

        scored = [(head_scored, head)] + [
            (
                self._rank_and_warmth(r, free_blocks_for_fit=free_blocks_for_fit),
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
