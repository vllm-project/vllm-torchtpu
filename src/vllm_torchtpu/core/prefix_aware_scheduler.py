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
# ruff: noqa: E501
"""Prefix-aware request scheduler with Shortest-Remaining-Prefix-First (SRPF) ordering.

Extends the vLLM V1 scheduler to maximize KV prefix cache reuse across concurrent
requests while bounding mixed-step chunked prefill interference on active decode batches.
"""

import math
import os
from itertools import islice
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
from vllm_torchtpu.core.hybrid_retention import HybridStateRetention

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
        self._is_disagg = (
            getattr(getattr(self, "vllm_config", None), "kv_transfer_config", None)
            is not None
        )
        self._srpf_running: bool = (
            bool(envs.TPU_SCHED_SRPF_RUNNING)
            if (not self._is_disagg or "TPU_SCHED_SRPF_RUNNING" in os.environ)
            else False
        )
        self._cold_chunk_cap: int = (
            int(envs.TPU_SCHED_COLD_CHUNK_CAP)
            if (not self._is_disagg or "TPU_SCHED_COLD_CHUNK_CAP" in os.environ)
            else 0
        )
        self._cold_threshold: int = int(envs.TPU_SCHED_COLD_THRESHOLD)
        self._cold_min_decoders: int = int(envs.TPU_SCHED_COLD_MIN_DECODERS)
        self._step_total_cap: int = (
            int(envs.TPU_SCHED_STEP_TOTAL_CAP)
            if (not self._is_disagg or "TPU_SCHED_STEP_TOTAL_CAP" in os.environ)
            else 0
        )
        self._warm_streak: int = int(envs.TPU_SCHED_WARM_STREAK)
        self._warm_min_decoders: int = int(envs.TPU_SCHED_WARM_MIN_DECODERS)
        self._step_num_decoding: int = 0
        self._step_running_prefill_ids: set[int] = set()
        self._step_prefill_tokens_by_req: dict[int, int] | None = None
        self._pure_decode_streak: int = 999999
        self._retention = None
        retention_default = "0" if self._is_disagg else "1"
        if os.environ.get("TPU_HYBRID_STATE_RETENTION", retention_default) != "0":
            retention = HybridStateRetention(self.kv_cache_manager)
            if retention.enabled:
                self._retention = retention

        logger.info(
            "PrefixAwareScheduler ACTIVE (window=%d, decoders first, "
            "srpf_running=%s, cold_chunk_cap=%d, cold_threshold=%d, "
            "cold_min_decoders=%d, step_total_cap=%d, warm_streak=%d@%d, "
            "hybrid_state_retention=%s)",
            self.WINDOW,
            self._srpf_running,
            self._cold_chunk_cap,
            self._cold_threshold,
            self._cold_min_decoders,
            self._step_total_cap,
            self._warm_streak,
            self._warm_min_decoders,
            self._retention is not None,
        )

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
        # of multi-step cold prefills.
        srpf_running, _, _, _ = self._get_sched_knobs()
        decoding = [r for r in self.running if _is_decoding(r)]
        prefilling = [r for r in self.running if not _is_decoding(r)]
        if srpf_running and len(prefilling) > 1:
            prefilling.sort(key=_remaining_prefill_tokens)
        if prefilling and (decoding or (srpf_running and len(prefilling) > 1)):
            self.running[:] = decoding + prefilling
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
        try:
            out = super().schedule(*args, **kwargs)
            if any(v > 0 for v in self._step_prefill_tokens_by_req.values()):
                self._pure_decode_streak = 0
            else:
                self._pure_decode_streak = (
                    getattr(self, "_pure_decode_streak", 999999) + 1
                )
            return out
        finally:
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
           (e.g. ``4 * 32 + 1920 = 2048``), keeping single-cold steps inside the
           ``<= 2048`` XLA compile bucket.
        2. Subsequent prefill requests in the same step at high concurrency
           (``num_decoding >= cold_min_decoders``): subtracts cumulative scheduled
           step tokens (``decode_tokens + prior_prefill_tokens``) from
           ``step_total_cap`` (default ``4096``), returning ``0`` if fewer than
           one Mamba block remains so multi-prefill collisions never spill into
           the ``8192`` XLA compile bucket.
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
        decode_tokens = getattr(self, "_step_decode_tokens", 4 * num_decoding)
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
        aligned = super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        sched_map = getattr(self, "_step_prefill_tokens_by_req", None)
        if sched_map is not None and id(request) in sched_map:
            sched_map[id(request)] = aligned
        return aligned

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
        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
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
        if epoch is not None:
            cached = getattr(request, "_paw_hit_cache", None)
            if cached is not None and cached[0] == epoch and cached[1] == max_len:
                return cached[3]
        try:
            result = kvm.coordinator.find_longest_cache_hit(
                request.block_hashes, max_len
            )
            warmth = int(result[1])
            if epoch is not None:
                request._paw_hit_cache = (epoch, max_len, result, warmth)
            return warmth
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
                cached = getattr(request, "_paw_hit_cache", None)
                if cached is not None and cached[0] == epoch and cached[1] == max_len:
                    hit = cached[2]
                    if len(hit) == 3:
                        computed_blocks, num_new_computed_tokens, num_uncached = hit
                    else:
                        computed_blocks, num_new_computed_tokens = hit
                        num_uncached = 0
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
        if getattr(self, "_is_disagg", False):
            tier = _TIER_PREEMPTED if request.num_preemptions > 0 else _TIER_NORMAL
        elif (
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
        queue = super()._select_waiting_queue_for_scheduling()
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
