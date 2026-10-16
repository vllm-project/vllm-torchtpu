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
"""Shortest-Remaining-Processing-Time (SRPT) prefill scheduler for RUNNING and WAITING.

vLLM schedules the RUNNING queue in admission order and the WAITING queue after
it. When an in-progress long prefill chunk sits in ``self.running``, it consumes
the step's token budget before ``Scheduler.schedule()`` visits ``self.waiting``,
forcing newly arrived short or prefix-cached requests to wait behind the entire
long prefill.

``SrptPrefillScheduler`` extends :class:`PrefixAwareScheduler` so that each
step (under ``SchedulingPolicy.FCFS``):

* keeps every decoding request (decode and speculative verify) ahead of all
  in-progress prefill chunks, preserving their relative order;
* orders RUNNING prefill chunks by remaining prompt tokens, fewest first — so
  shorter prefills finish first and the longest remaining prefill sits at
  ``self.running[-1]``;
* delegates WAITING queue promotion (including ``kv_holding_waiting`` priority,
  preempted-request priority tier, and bounded ``WINDOW`` prefix-cache warmth
  evaluation) to :class:`PrefixAwareScheduler`;
* holds back, for the current step, only the longest running prefill chunks
  needed so that the step's remaining token budget can admit the highest-ranked
  waiting request within ``WINDOW`` when that waiting request has strictly fewer
  remaining uncached tokens and sufficient free KV blocks exist.

Held-back chunks retain their allocated KV blocks and are restored to
``self.running`` immediately after ``super().schedule()`` while lowering both
``max_num_running_reqs`` and ``max_num_active_reqs`` by ``len(held)`` during the
step so admission never exceeds ``max_num_seqs``. To bound starvation, a running
prefill chunk may be held back at most ``MAX_CONSECUTIVE_HOLDS`` steps in a row
(configurable via ``VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS``, default ``8``) before
it is guaranteed a step to advance (under continuous short arrivals, a long
prompt advances on at least 1 step in ``MAX_CONSECUTIVE_HOLDS + 1``, i.e. 1 step
in 9 by default).

Note on preemption: upstream FCFS preemption pops ``self.running[-1]`` when KV
blocks are exhausted. Hold-back is skipped whenever free KV blocks are below
what the non-held running requests, in-flight prefill reservations, and the
waiting head require, ensuring that under KV pressure the longest remaining
prefill chunk stays at ``self.running[-1]`` and active decodes or shorter
prefills are never preempted in its place.

Select with ``--scheduler-cls``
``vllm_torchtpu.core.experimental.srpt_prefill_scheduler.SrptPrefillScheduler``.
"""

from itertools import islice

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request

from vllm_torchtpu import envs
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

logger = init_logger("vllm." + __name__)


class SrptPrefillScheduler(PrefixAwareScheduler):
    """Order RUNNING decodes first and prefill chunks by fewest remaining tokens,
    holding back longer running prefill chunks when shorter waiting requests exist."""

    #: Maximum consecutive steps a running prefill chunk may be held back
    #: before it is guaranteed a step to make progress (at least 1 step in
    #: ``MAX_CONSECUTIVE_HOLDS + 1`` under continuous short arrivals).
    #: Configurable via ``VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS``.
    MAX_CONSECUTIVE_HOLDS = 8

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            type(self).MAX_CONSECUTIVE_HOLDS
            == SrptPrefillScheduler.MAX_CONSECUTIVE_HOLDS
        ):
            self.MAX_CONSECUTIVE_HOLDS = envs.VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS
        self._skip_hold_next_step = False
        self._blocked_hold_state: tuple[str, int] | None = None
        self._blocked_hold_failures: dict[str, int] = {}
        self._blocked_hold_cooldown: dict[str, int] = {}
        self._consecutive_holds: dict[str, int] = {}
        self._step_warmth_cache: dict[tuple[str, int], tuple[int, tuple]] = {}
        logger.info(
            "SrptPrefillScheduler active (window=%d, max_consecutive_holds=%d)",
            self.WINDOW,
            self.MAX_CONSECUTIVE_HOLDS,
        )

    @staticmethod
    def _running_remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    def _effective_long_prefill_threshold(self, num_running: int | None = None) -> int:
        """Return the runtime-adjusted ``long_prefill_token_threshold`` matching
        upstream ``Scheduler.schedule()`` (0 when at most one request is
        eligible, and floored at ``input_budget // num_eligible_reqs`` when
        ``adaptive_long_prefill_threshold`` is enabled)."""
        lptt = self.scheduler_config.long_prefill_token_threshold
        if lptt <= 0:
            return 0
        get_counts = getattr(self, "get_request_counts", None)
        if callable(get_counts):
            cur_running, num_waiting = get_counts()
            running_cnt = cur_running if num_running is None else num_running
        else:
            running_cnt = len(self.running) if num_running is None else num_running
            queue = getattr(self, "kv_holding_waiting", None) or getattr(
                self, "waiting", None
            )
            num_waiting = len(queue) if queue is not None else 0
        num_eligible_reqs = running_cnt + num_waiting
        if num_eligible_reqs <= 1:
            return 0
        if getattr(self, "adaptive_long_prefill_threshold", False):
            input_budget = self.scheduler_config.max_num_batched_tokens
            lptt = max(lptt, input_budget // num_eligible_reqs)
        return lptt

    def _running_step_tokens(self, request: Request, lptt: int | None = None) -> int:
        tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if lptt is None:
            lptt = self._effective_long_prefill_threshold()
        if 0 < lptt < tokens:
            tokens = lptt
        return max(tokens, 0)

    def _draft_slots(self) -> int:
        spec = self.vllm_config.speculative_config
        return spec.max_num_new_slots_for_drafting if spec is not None else 0

    def _step_token_budget(self) -> int:
        draft_slots = self._draft_slots()
        return min(
            self.max_num_scheduled_tokens,
            max(self.scheduler_config.max_num_batched_tokens - draft_slots, 0),
        )

    def _warmth_and_blocks(self, request: Request) -> tuple[int, tuple]:
        """Return ``(cached_prompt_tokens, cached_blocks_by_group)`` for
        ``request``, memoized for the current ``schedule()`` step."""
        key = (request.request_id, request.num_computed_tokens)
        cached = self._step_warmth_cache.get(key)
        if cached is not None:
            return cached

        kvm = self.kv_cache_manager
        empty_blocks = getattr(kvm, "empty_kv_cache_blocks", None)
        default_blocks = empty_blocks.blocks if empty_blocks is not None else ()
        if not kvm.enable_caching or request.skip_reading_prefix_cache:
            res = (0, default_blocks)
        elif request.num_computed_tokens > 0:
            res = (request.num_computed_tokens, default_blocks)
        else:
            try:
                computed_blocks, num_computed, _ = (
                    kvm.coordinator.find_longest_cache_hit(
                        request.block_hashes, max(request.num_tokens - 1, 0)
                    )
                )
                res = (int(num_computed), tuple(computed_blocks))
            except Exception:
                if not self._paw_scoring_broken:
                    self._paw_scoring_broken = True
                    logger.exception(
                        "PrefixAwareScheduler: warmth scoring failed; "
                        "falling back to FCFS."
                    )
                res = (0, default_blocks)

        self._step_warmth_cache[key] = res
        return res

    def _prefix_warmth(self, request: Request) -> int:
        return self._warmth_and_blocks(request)[0]

    def _waiting_remaining(self, request: Request) -> int:
        return max(request.num_tokens - self._prefix_warmth(request), 1)

    def _num_free_blocks(self) -> int:
        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return 0
        return int(block_pool.get_num_free_blocks())

    def _has_kv_blocks_for_hold(
        self,
        kept_running: list[Request],
        best_waiting: Request,
        waiting_step_tokens: int,
        step_budget: int,
    ) -> bool:
        """Return True iff enough free KV blocks exist to schedule all
        ``kept_running`` requests plus ``best_waiting`` without triggering
        preemption or violating full-sequence-fit / in-flight reservations."""
        kvm = self.kv_cache_manager
        coordinator = getattr(kvm, "coordinator", None)
        if coordinator is None or not hasattr(
            coordinator, "get_num_blocks_to_allocate"
        ):
            return True

        free_blocks = self._num_free_blocks()
        empty_blocks = kvm.empty_kv_cache_blocks.blocks
        draft_slots = self._draft_slots()
        rem_token_budget = min(self.max_num_scheduled_tokens, step_budget)
        rem_input_budget = self.scheduler_config.max_num_batched_tokens
        held_lptt = self._effective_long_prefill_threshold(
            num_running=len(kept_running)
        )

        inflight_fn = getattr(self, "_inflight_prefill_reserved_blocks", None)
        spec_step_fn = getattr(self, "_spec_decode_step_blocks", None)
        inflight_reqs = getattr(self, "_inflight_prefills", ()) or ()
        has_inflight_fn = callable(inflight_fn)

        needed_blocks = 0
        for req in kept_running:
            if rem_token_budget <= 0 or rem_input_budget <= draft_slots:
                break
            req_new_tokens = min(
                self._running_step_tokens(req, lptt=held_lptt),
                rem_token_budget,
                rem_input_budget - draft_slots,
            )
            if req_new_tokens <= 0:
                continue
            rem_token_budget -= req_new_tokens
            rem_input_budget -= req_new_tokens + draft_slots
            # Requests in `_inflight_prefills` already have their full remaining
            # blocks counted by `_inflight_prefill_reserved_blocks()` below.
            if has_inflight_fn and req in inflight_reqs:
                continue
            main_tokens = req.num_computed_tokens + req_new_tokens
            slot_tokens = min(
                main_tokens + self.num_lookahead_tokens, self.max_model_len
            )
            needed_blocks += coordinator.get_num_blocks_to_allocate(
                request_id=req.request_id,
                num_tokens=slot_tokens,
                new_computed_blocks=empty_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=req.num_computed_tokens,
                num_local_computed_tokens=req.num_computed_tokens,
                num_tokens_main_model=main_tokens,
            )

        warmth, computed_blocks = self._warmth_and_blocks(best_waiting)
        w_main_tokens = warmth + waiting_step_tokens
        w_slot_tokens = min(
            w_main_tokens + self.num_lookahead_tokens, self.max_model_len
        )
        waiting_blocks = coordinator.get_num_blocks_to_allocate(
            request_id=best_waiting.request_id,
            num_tokens=w_slot_tokens,
            new_computed_blocks=computed_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=warmth,
            num_local_computed_tokens=warmth,
            num_tokens_main_model=w_main_tokens,
        )
        if getattr(self, "scheduler_reserve_full_isl", False):
            full_num_tokens = min(best_waiting.num_tokens, self.max_model_len)
            full_blocks = coordinator.get_num_blocks_to_allocate(
                request_id=best_waiting.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=warmth,
                num_local_computed_tokens=warmth,
                num_tokens_main_model=full_num_tokens,
            )
            waiting_blocks = max(waiting_blocks, full_blocks)
        needed_blocks += waiting_blocks

        if kept_running:
            needed_blocks += getattr(kvm, "watermark_blocks", 0)

        reserved_blocks = (inflight_fn() if callable(inflight_fn) else 0) + (
            spec_step_fn() if callable(spec_step_fn) else 0
        )
        return free_blocks - reserved_blocks >= needed_blocks

    def _order_running(self) -> None:
        """Sort RUNNING: decodes first (preserving their relative order), then
        in-progress prefill chunks by remaining tokens (fewest first)."""
        if self.policy != SchedulingPolicy.FCFS or len(self.running) < 2:
            return
        decoding = [r for r in self.running if not r.is_prefill_chunk]
        prefilling = sorted(
            (r for r in self.running if r.is_prefill_chunk),
            key=self._running_remaining,
        )
        self.running[:] = decoding + prefilling

    def _select_chunks_to_hold(
        self, throttle_prefills: bool = False
    ) -> tuple[list[Request], Request | None]:
        """Return ``(chunks_to_hold, target_waiting_request)``."""
        if self._skip_hold_next_step:
            self._skip_hold_next_step = False
            return [], None
        if throttle_prefills or self.policy != SchedulingPolicy.FCFS:
            return [], None
        if not any(r.is_prefill_chunk for r in self.running):
            return [], None
        num_streaming = getattr(self, "num_waiting_for_streaming_input", 0)
        max_active = getattr(self, "max_num_active_reqs", self.max_num_running_reqs)
        if len(self.running) + num_streaming >= max_active:
            return [], None
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if not queue:
            return [], None
        admissible = [
            r
            for r in islice(queue, self.WINDOW)
            if not self._is_blocked_waiting_status(r.status)
        ]
        if not admissible:
            return [], None

        # Rank using PrefixAwareScheduler._rank_and_warmth so preempted waiting
        # requests (tier 1) take precedence over normal waiting requests (tier 2),
        # matching the exact request _promote_warmest will place at queue[0].
        best_req = min(admissible, key=lambda r: self._rank_and_warmth(r)[0])
        best_id = best_req.request_id
        cooldown = self._blocked_hold_cooldown.get(best_id, 0)
        if cooldown > 0:
            if cooldown <= 1:
                self._blocked_hold_cooldown.pop(best_id, None)
            else:
                self._blocked_hold_cooldown[best_id] = cooldown - 1
            return [], None

        free_blocks = self._num_free_blocks()
        if self._blocked_hold_state is not None:
            blocked_req_id, blocked_free_blocks = self._blocked_hold_state
            if best_id == blocked_req_id and free_blocks <= blocked_free_blocks:
                self._blocked_hold_state = (blocked_req_id, free_blocks)
                return [], None
            self._blocked_hold_state = None

        target_remaining = self._waiting_remaining(best_req)
        draft_slots = self._draft_slots()
        token_budget = self.max_num_scheduled_tokens
        input_budget = self.scheduler_config.max_num_batched_tokens
        lptt = self._effective_long_prefill_threshold()
        for r in self.running:
            if r.is_prefill_chunk:
                continue
            if token_budget <= 0 or input_budget <= draft_slots:
                token_budget = 0
                input_budget = 0
                break
            toks = min(
                self._running_step_tokens(r, lptt=lptt),
                token_budget,
                input_budget - draft_slots,
            )
            if toks <= 0:
                continue
            token_budget -= toks
            input_budget -= toks + draft_slots

        budget_after_decodes = min(token_budget, input_budget - draft_slots)
        if budget_after_decodes <= 0:
            return [], None

        waiting_step_need = target_remaining
        if 0 < lptt < waiting_step_need:
            waiting_step_need = lptt
        waiting_step_need = min(waiting_step_need, budget_after_decodes)

        # Sort prefill chunks locally by remaining tokens so `_chunks_to_hold()`
        # does not depend on `_order_running()` having been called first.
        prefilling = sorted(
            (r for r in self.running if r.is_prefill_chunk),
            key=self._running_remaining,
        )
        prefill_step_tokens = [
            self._running_step_tokens(r, lptt=lptt) for r in prefilling
        ]
        total_prefill_tokens = sum(prefill_step_tokens)
        total_prefill_count = sum(1 for t in prefill_step_tokens if t > 0)

        def _waiting_budget_after_prefills(
            prefill_toks: int, prefill_count: int
        ) -> int:
            return min(
                token_budget - prefill_toks,
                input_budget - prefill_toks - (prefill_count + 1) * draft_slots,
            )

        # If running prefills already leave enough budget for best_req, hold nothing.
        if (
            _waiting_budget_after_prefills(total_prefill_tokens, total_prefill_count)
            >= waiting_step_need
        ):
            return [], None

        # Hold only the longest eligible running prefill chunks needed to free
        # waiting_step_need tokens of budget.
        held_ids: set[str] = set()
        remaining_prefill_tokens = total_prefill_tokens
        remaining_prefill_count = total_prefill_count
        for r, step_toks in zip(reversed(prefilling), reversed(prefill_step_tokens)):
            if (
                _waiting_budget_after_prefills(
                    remaining_prefill_tokens, remaining_prefill_count
                )
                >= waiting_step_need
            ):
                break
            if self._running_remaining(r) <= target_remaining:
                continue
            if (
                self._consecutive_holds.get(r.request_id, 0)
                >= self.MAX_CONSECUTIVE_HOLDS
            ):
                continue
            held_ids.add(r.request_id)
            remaining_prefill_tokens -= step_toks
            if step_toks > 0:
                remaining_prefill_count -= 1

        avail_after_held = _waiting_budget_after_prefills(
            remaining_prefill_tokens, remaining_prefill_count
        )
        if not held_ids or avail_after_held <= 0:
            return [], None

        held = [r for r in prefilling if r.request_id in held_ids]
        kept_running = [r for r in self.running if r.request_id not in held_ids]
        waiting_avail_budget = min(waiting_step_need, avail_after_held)
        if not self._has_kv_blocks_for_hold(
            kept_running,
            best_req,
            waiting_avail_budget,
            self._step_token_budget(),
        ):
            return [], None

        return held, best_req

    def _chunks_to_hold(self, throttle_prefills: bool = False) -> list[Request]:
        """Return running prefill chunks to hold back for the current step."""
        return self._select_chunks_to_hold(throttle_prefills)[0]

    def _prune_hold_tracking(self) -> None:
        """Drop per-request hold counters and failure/cooldown state for
        requests that are no longer active prefill chunks or waiting."""
        active_prefill_ids = {r.request_id for r in self.running if r.is_prefill_chunk}
        for req_id in list(self._consecutive_holds):
            if req_id not in active_prefill_ids:
                del self._consecutive_holds[req_id]

        if (
            not self._blocked_hold_failures
            and not self._blocked_hold_cooldown
            and self._blocked_hold_state is None
        ):
            return
        waiting_ids: set[str] = set()
        for q_attr in ("kv_holding_waiting", "waiting"):
            q = getattr(self, q_attr, None)
            if q:
                waiting_ids.update(r.request_id for r in q)
        for req_id in list(self._blocked_hold_failures):
            if req_id not in waiting_ids:
                del self._blocked_hold_failures[req_id]
        for req_id in list(self._blocked_hold_cooldown):
            if req_id not in waiting_ids:
                del self._blocked_hold_cooldown[req_id]
        if (
            self._blocked_hold_state is not None
            and self._blocked_hold_state[0] not in waiting_ids
        ):
            self._blocked_hold_state = None

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._step_warmth_cache.clear()
        try:
            self._order_running()
            self._prune_hold_tracking()

            held, target_req = self._select_chunks_to_hold(throttle_prefills)
            if not held:
                output = super().schedule(throttle_prefills)
                for req_id, n_tok in output.num_scheduled_tokens.items():
                    if n_tok > 0:
                        self._consecutive_holds.pop(req_id, None)
                        self._blocked_hold_failures.pop(req_id, None)
                        self._blocked_hold_cooldown.pop(req_id, None)
                if self._blocked_hold_state is not None:
                    blocked_id, blocked_free = self._blocked_hold_state
                    self._blocked_hold_state = (
                        blocked_id,
                        min(blocked_free, self._num_free_blocks()),
                    )
                return output

            held_ids = {r.request_id for r in held}
            for r in held:
                self._consecutive_holds[r.request_id] = (
                    self._consecutive_holds.get(r.request_id, 0) + 1
                )
            self.running[:] = [r for r in self.running if r.request_id not in held_ids]
            len_before = len(self.running)
            k = len(held)
            self.max_num_running_reqs -= k
            has_max_active = hasattr(self, "max_num_active_reqs")
            if has_max_active:
                self.max_num_active_reqs -= k
            try:
                output = super().schedule(throttle_prefills)
                for req_id, n_tok in output.num_scheduled_tokens.items():
                    if n_tok > 0:
                        self._consecutive_holds.pop(req_id, None)
                        self._blocked_hold_failures.pop(req_id, None)
                        self._blocked_hold_cooldown.pop(req_id, None)
                # Note: `len(self.running) > len_before` also treats a simultaneous
                # preemption + admission in the same step as a failed hold, which is
                # conservative (harmlessly skips holding on the next step).
                admitted = len(self.running) > len_before
                self._skip_hold_next_step = not admitted
                if not admitted and target_req is not None:
                    target_id = target_req.request_id
                    failures = self._blocked_hold_failures.get(target_id, 0) + 1
                    self._blocked_hold_failures[target_id] = failures
                    # Exponential backoff (in addition to _skip_hold_next_step) so
                    # non-KV admission failures (LoRA limit, stale output, connector
                    # returning None) do not retry every step as decodes free blocks.
                    self._blocked_hold_cooldown[target_id] = 1 << min(failures - 1, 6)
                    self._blocked_hold_state = (
                        target_id,
                        self._num_free_blocks(),
                    )
                else:
                    self._blocked_hold_state = None
                return output
            finally:
                self.max_num_running_reqs += k
                if has_max_active:
                    self.max_num_active_reqs += k
                self.running.extend(held)
        finally:
            self._step_warmth_cache.clear()
