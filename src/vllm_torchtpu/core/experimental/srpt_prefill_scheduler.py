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
"""Shortest-remaining-first ordering and hold-back of running prefill chunks.

vLLM schedules RUNNING in admission order before WAITING, so an in-progress
long prefill chunk can take the step's token budget before a short waiting
request is reached. Each step (FCFS policy, prefills not throttled) this
scheduler:

* orders RUNNING as decodes first, in their existing order, then prefill
  chunks by remaining prompt tokens, fewest first, with stalled chunks (below)
  ahead of the rest;
* holds back, for this step, the longest running prefill chunks needed for the
  request upstream admits next (the first admissible request in
  ``kv_holding_waiting``, then ``waiting``) to fit in the remaining budget,
  when that request has fewer uncached tokens and free KV blocks cover the
  kept requests plus it, so a hold does not cause preemption;
* with ``VLLM_TPU_SRPT_ASYNC_KV_OVERLAP=1``, when admitting that request would
  start an asynchronous KV connector load (``load_kv_async``), which takes no
  step budget, holds nothing for it and instead trims the last running prefill
  chunk just enough to leave one token of budget, so upstream reaches the
  request and starts its load in the same step as the running chunks.

Held chunks keep their KV blocks; ``max_num_running_reqs`` and
``max_num_active_reqs`` are lowered by the number held during the step so
admission never exceeds ``max_num_seqs``. A hold whose target is not
admitted backs off exponentially for that target.

Anti-starvation: a running prefill chunk that goes
``VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS`` steps in a row without getting at least
one KV block of tokens in a step, whether it was held or crowded out by shorter
prefills, is stalled: it moves ahead of the other prefill chunks and is not
held until it gets a block. Decodes still go first, so a stalled chunk can
still wait while decodes fill the budget.

Under KV pressure upstream preempts ``running[-1]``, which is the longest
non-stalled prefill chunk, so a long prompt can be preempted repeatedly.

Select with ``--scheduler-cls``
``vllm_torchtpu.core.experimental.srpt_prefill_scheduler.SrptPrefillScheduler``.
"""

from itertools import chain

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MAX_BACKOFF_SHIFT = 6
_OVERLAP_LOG_EVERY = 1000


class SrptPrefillScheduler(AsyncScheduler):
    """Order RUNNING prefill chunks by fewest remaining tokens and hold back
    longer ones for a shorter waiting request."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_consecutive_holds = envs.VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS
        # Running prefill request id -> consecutive steps with less than one
        # block of progress.
        self._stalled_steps: dict[str, int] = {}
        # Target request id -> (failed holds, first step a hold may retry).
        self._hold_backoff: dict[str, tuple[int, int]] = {}
        self.async_kv_overlap = envs.VLLM_TPU_SRPT_ASYNC_KV_OVERLAP
        # Request id -> most tokens its chunk may get in this step's RUNNING pass.
        self._token_caps: dict[str, int] = {}
        self._overlap_target: Request | None = None
        self._overlap_steps = 0
        self._overlap_loads_started = 0
        logger.info(
            "SrptPrefillScheduler active (max_consecutive_holds=%d, "
            "async_kv_overlap=%s)",
            self.max_consecutive_holds,
            self.async_kv_overlap,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        held: list[Request] = []
        target = None
        prefilling: list[Request] = []
        self._overlap_target = None
        if not throttle_prefills and self.policy == SchedulingPolicy.FCFS:
            self._prune_hold_state()
            self._order_running()
            prefilling = [r for r in self.running if r.is_prefill_chunk]
            held, target = self._select_chunks_to_hold()

        held_ids = {r.request_id for r in held}
        self.running[:] = [r for r in self.running if r.request_id not in held_ids]
        self.max_num_running_reqs -= len(held)
        self.max_num_active_reqs -= len(held)
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self._token_caps.clear()
            self.max_num_running_reqs += len(held)
            self.max_num_active_reqs += len(held)
            self.running.extend(held)

        # One KV cache block of the main attention group; not self.block_size,
        # which for hybrid models is the LCM of all group block sizes and can
        # exceed a whole chunk.
        progress_tokens = self.cache_config.block_size
        for r in prefilling:
            if (
                output.num_scheduled_tokens.get(r.request_id, 0) >= progress_tokens
                or not r.is_prefill_chunk
            ):
                self._stalled_steps.pop(r.request_id, None)
            else:
                self._stalled_steps[r.request_id] = (
                    self._stalled_steps.get(r.request_id, 0) + 1
                )
        if target is not None:
            self._record_hold_outcome(target)
        if self._overlap_target is not None:
            self._log_overlap(self._overlap_target)
        return output

    def _log_overlap(self, target: Request) -> None:
        self._overlap_steps += 1
        if target.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self._overlap_loads_started += 1
        if self._overlap_steps % _OVERLAP_LOG_EVERY == 0:
            logger.info(
                "SrptPrefillScheduler async KV overlap: steps=%d loads_started=%d",
                self._overlap_steps,
                self._overlap_loads_started,
            )

    def _reserve_prefill_lookahead(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        # Scheduler.schedule() calls this on every request it is about to
        # schedule, after its other token limits, so it is where a cap set
        # for this step applies.
        cap = self._token_caps.get(request.request_id)
        if cap is not None:
            num_new_tokens = min(num_new_tokens, cap)
        return super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )

    @staticmethod
    def _remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    def _is_stalled(self, request: Request) -> bool:
        return (
            self._stalled_steps.get(request.request_id, 0) >= self.max_consecutive_holds
        )

    def _order_running(self) -> None:
        decoding = [r for r in self.running if not r.is_prefill_chunk]
        prefilling = sorted(
            (r for r in self.running if r.is_prefill_chunk),
            key=lambda r: (not self._is_stalled(r), self._remaining(r)),
        )
        self.running[:] = decoding + prefilling

    def _prune_hold_state(self) -> None:
        prefilling = {r.request_id for r in self.running if r.is_prefill_chunk}
        self._stalled_steps = {
            k: v for k, v in self._stalled_steps.items() if k in prefilling
        }
        if self._hold_backoff:
            waiting = {
                r.request_id for r in chain(self.kv_holding_waiting, self.waiting)
            }
            self._hold_backoff = {
                k: v for k, v in self._hold_backoff.items() if k in waiting
            }

    def _record_hold_outcome(self, target: Request) -> None:
        # Admission either runs the target or starts its KV load; both move it
        # out of WAITING / PREEMPTED.
        if target.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            self._hold_backoff.pop(target.request_id, None)
            return
        failures = self._hold_backoff.get(target.request_id, (0, 0))[0] + 1
        delay = 1 << min(failures - 1, _MAX_BACKOFF_SHIFT)
        self._hold_backoff[target.request_id] = (
            failures,
            self.current_step + 1 + delay,
        )

    def _long_prefill_threshold(self, num_eligible_reqs: int) -> int:
        # Mirrors the long_prefill_token_threshold computation inline in
        # Scheduler.schedule() (vLLM v0.31.0, scheduler.py ~L611-626). Re-sync
        # on vLLM upgrades.
        lptt = (
            self.scheduler_config.long_prefill_token_threshold
            if num_eligible_reqs > 1
            else 0
        )
        if lptt > 0 and self.adaptive_long_prefill_threshold:
            lptt = max(
                lptt,
                self.scheduler_config.max_num_batched_tokens // num_eligible_reqs,
            )
        return lptt

    def _plan_running(
        self, reqs: list[Request], lptt: int
    ) -> tuple[dict[str, int], int]:
        """Tokens the RUNNING pass of Scheduler.schedule() would give ``reqs``
        and the budget it leaves for WAITING.

        Mirrors the per-request ``num_new_tokens`` math inline in that pass
        (vLLM v0.31.0, scheduler.py ~L628-722); re-sync on vLLM upgrades.
        Encoder, Mamba-alignment and KV-allocation limits are not modelled, so
        a misestimate only makes a hold miss, which the backoff absorbs.
        """
        next_step = self.current_step + 1
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        token_budget = self.max_num_scheduled_tokens
        input_budget = self.scheduler_config.max_num_batched_tokens
        alloc: dict[str, int] = {}
        for r in reqs:
            if token_budget <= 0 or input_budget <= draft_slots:
                break
            if (
                r.num_output_placeholders > 0
                and r.num_computed_tokens + 2 - r.num_output_placeholders
                >= r.num_prompt_tokens + r.max_tokens
            ) or next_step < r.next_decode_eligible_step:
                continue
            n = (
                r.num_tokens_with_spec
                + r.num_output_placeholders
                - r.num_computed_tokens
            )
            if 0 < lptt < n:
                n = lptt
            n = min(
                n,
                token_budget,
                input_budget - draft_slots,
                self.max_model_len
                - r.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            n = self._reserve_prefill_lookahead(r, r.num_computed_tokens, n)
            if n <= 0:
                continue
            alloc[r.request_id] = n
            token_budget -= n
            input_budget -= n + draft_slots
        return alloc, min(token_budget, input_budget - draft_slots)

    def _next_waiting(self) -> Request | None:
        """The request upstream's WAITING pass admits next."""
        return next(
            (
                r
                for r in chain(self.kv_holding_waiting, self.waiting)
                if not self._is_blocked_waiting_status(r.status)
            ),
            None,
        )

    def _prefix_hit(self, request: Request) -> tuple[KVCacheBlocks, int]:
        if request.num_computed_tokens > 0:
            return (
                self.kv_cache_manager.empty_kv_cache_blocks,
                request.num_computed_tokens,
            )
        blocks, num_hit_tokens, _, _ = self._get_local_prefix_cache_hit(request)
        return blocks, num_hit_tokens

    def _starts_async_kv_load(self, request: Request, num_local: int) -> bool:
        """Whether admitting ``request`` with ``num_local`` locally cached
        tokens would start an asynchronous KV connector load.

        Mirrors the connector lookup in Scheduler.schedule()'s WAITING pass
        (vLLM v0.31.0, scheduler.py ~L940-994); re-sync on vLLM upgrades. It is
        only asked about the request upstream admits next, which upstream asks
        about itself when it reaches it in this step.
        """
        if (
            self.connector is None
            or request.num_computed_tokens > 0
            or request.skip_reading_prefix_cache
        ):
            return False
        partial_tail = num_local % self.block_size
        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
            request, num_local - partial_tail
        )
        # None means the connector is still looking up the request.
        if not ext_tokens or not load_kv_async:
            return False
        if self.prefix_replay_tokens:
            ext_tokens -= ext_tokens % self.block_size
        return ext_tokens > partial_tail

    def _leave_one_token(self, alloc: dict[str, int], left: int) -> bool:
        """Cap the last planned prefill chunk so the RUNNING pass leaves at
        least one token of budget for the WAITING pass. Returns False if no
        prefill chunk can give up enough tokens."""
        deficit = 1 - left
        if deficit <= 0:
            return True
        for r in reversed(self.running):
            n = alloc.get(r.request_id, 0)
            if r.is_prefill_chunk and n > deficit:
                self._token_caps[r.request_id] = n - deficit
                return True
        return False

    def _select_chunks_to_hold(self) -> tuple[list[Request], Request | None]:
        """Return ``(chunks_to_hold, target_waiting_request)``."""
        if not any(r.is_prefill_chunk for r in self.running):
            return [], None
        if (
            len(self.running) + self.num_waiting_for_streaming_input
            >= self.max_num_active_reqs
        ):
            return [], None
        target = self._next_waiting()
        if target is None:
            return [], None
        _, retry_step = self._hold_backoff.get(target.request_id, (0, 0))
        if self.current_step + 1 < retry_step:
            return [], None

        hit_blocks, hit_tokens = self._prefix_hit(target)
        _, num_waiting = self.get_request_counts()
        if self.async_kv_overlap and self._starts_async_kv_load(target, hit_tokens):
            lptt = self._long_prefill_threshold(len(self.running) + num_waiting)
            alloc, left = self._plan_running(self.running, lptt)
            if self._leave_one_token(alloc, left):
                self._overlap_target = target
                return [], target
        target_remaining = max(target.num_tokens - hit_tokens, 1)
        kept = list(self.running)
        held: list[Request] = []
        while True:
            lptt = self._long_prefill_threshold(len(kept) + num_waiting)
            alloc, left = self._plan_running(kept, lptt)
            need = min(target_remaining, lptt) if lptt > 0 else target_remaining
            if left >= need:
                break
            # Non-stalled prefill chunks sit last in `kept`, ordered by
            # remaining tokens, so this is the longest one.
            candidate = next(
                (
                    r
                    for r in reversed(kept)
                    if r.request_id in alloc
                    and r.is_prefill_chunk
                    and self._remaining(r) > target_remaining
                    and not self._is_stalled(r)
                ),
                None,
            )
            if candidate is None:
                break
            kept.remove(candidate)
            held.append(candidate)
        if not held or left <= 0:
            return [], None
        if not self._kv_fits_hold(
            kept, alloc, target, hit_blocks, hit_tokens, min(need, left)
        ):
            return [], None
        return held, target

    def _kv_fits_hold(
        self,
        kept: list[Request],
        alloc: dict[str, int],
        target: Request,
        hit_blocks: KVCacheBlocks,
        hit_tokens: int,
        target_tokens: int,
    ) -> bool:
        """Whether free KV blocks cover ``kept`` running requests plus
        ``target``, counted as KVCacheManager.allocate_slots() counts them,
        so a hold never forces a preemption."""
        coordinator = self.kv_cache_manager.coordinator
        empty_blocks = self.kv_cache_manager.empty_kv_cache_blocks.blocks

        def blocks_for(
            request: Request,
            computed_tokens: int,
            num_new_tokens: int,
            computed_blocks: tuple,
        ) -> int:
            num_tokens_main = computed_tokens + num_new_tokens
            return coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(
                    num_tokens_main + self.num_lookahead_tokens, self.max_model_len
                ),
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=computed_tokens,
                num_local_computed_tokens=computed_tokens,
                num_tokens_main_model=num_tokens_main,
                prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
            )

        needed = sum(
            blocks_for(r, r.num_computed_tokens, alloc[r.request_id], empty_blocks)
            for r in kept
            # In-flight prefills are covered by the reservation below.
            if r.request_id in alloc and r not in self._inflight_prefills
        )
        target_blocks = blocks_for(target, hit_tokens, target_tokens, hit_blocks.blocks)
        if self.scheduler_reserve_full_isl:
            full_num_tokens = min(target.num_tokens, self.max_model_len)
            target_blocks = max(
                target_blocks,
                coordinator.get_num_blocks_to_allocate(
                    request_id=target.request_id,
                    num_tokens=full_num_tokens,
                    new_computed_blocks=hit_blocks.blocks,
                    num_encoder_tokens=0,
                    total_computed_tokens=hit_tokens,
                    num_local_computed_tokens=hit_tokens,
                    num_tokens_main_model=full_num_tokens,
                    apply_admission_cap=True,
                    prefill_end=max(target.num_prompt_tokens, target.num_tokens - 1),
                ),
            )
        needed += target_blocks
        if kept:
            # allocate_slots() adds the watermark when others are scheduled.
            needed += self.kv_cache_manager.watermark_blocks
        reserved = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks()
        )
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return free - reserved >= needed
