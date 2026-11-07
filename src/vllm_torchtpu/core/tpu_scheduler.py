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
from collections.abc import Iterator
from itertools import chain

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TPU_DP_SCHEDULER_CLS = "vllm_torchtpu.core.tpu_scheduler.TpuDpScheduler"

# A locked decode wave is released once its running set has shrunk to at most
# 1/_WAVE_TAIL_RELEASE_DIVISOR of its peak size.
_WAVE_TAIL_RELEASE_DIVISOR = 4


def iter_unblocked_waiting(scheduler: AsyncScheduler) -> Iterator[Request]:
    """Yield waiting requests in admission order, skipping blocked statuses."""
    return (
        r
        for r in chain(scheduler.kv_holding_waiting, scheduler.waiting)
        if not scheduler._is_blocked_waiting_status(r.status)
    )


def next_unblocked_waiting(scheduler: AsyncScheduler) -> Request | None:
    """Return the request upstream's WAITING pass admits next."""
    return next(iter_unblocked_waiting(scheduler), None)


def blocks_to_allocate(
    scheduler: AsyncScheduler,
    request: Request,
    computed_tokens: int,
    total_tokens: int,
    computed_blocks: tuple,
    *,
    num_tokens_main_model: int | None = None,
    apply_admission_cap: bool = False,
) -> int:
    """Count additional KV blocks needed for ``request`` at ``total_tokens``."""
    if num_tokens_main_model is None:
        num_tokens_main_model = total_tokens
    return scheduler.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=total_tokens,
        new_computed_blocks=computed_blocks,
        num_encoder_tokens=0,
        total_computed_tokens=computed_tokens,
        num_local_computed_tokens=computed_tokens,
        num_tokens_main_model=num_tokens_main_model,
        apply_admission_cap=apply_admission_cap,
        prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
    )


def warn_if_prefill_wave_inactive(
    data_parallel_size: int, scheduler_cls: object
) -> None:
    """Warn once when ``TPU_DP_SCHED_PREFILL_WAVE=1`` without ``TpuDpScheduler``."""
    if not envs.TPU_DP_SCHED_PREFILL_WAVE:
        return
    if scheduler_cls == TPU_DP_SCHEDULER_CLS or scheduler_cls is TpuDpScheduler:
        return
    missing: list[str] = []
    if not envs.DP_SCHED_ENABLED:
        missing.append("DP_SCHED_ENABLED=1")
    if data_parallel_size <= 1:
        missing.append(f"data_parallel_size > 1 (got {data_parallel_size})")
    if scheduler_cls is not None:
        missing.append(f"scheduler_cls=None (got {scheduler_cls})")
    logger.warning_once(
        "TPU_DP_SCHED_PREFILL_WAVE=1 ignored because TpuDpScheduler is not "
        "selected; missing %s.",
        ", ".join(missing),
    )


class TpuDpScheduler(AsyncScheduler):
    """Scheduler that aligns prefill admission across DP ranks.

    In its default mode, vLLM passes a DP-synchronized scheduling signal
    through ``throttle_prefills``, and this scheduler uses that signal to
    decide when to admit new prefill requests until pending prefills drain.

    When ``TPU_DP_SCHED_PREFILL_WAVE=1`` (non-speculative, non-PP), ``schedule``
    ignores ``throttle_prefills`` and instead fills KV-bounded waves per DP
    rank: completed prefills in the current wave are held via
    ``next_decode_eligible_step`` until the wave fills so prefill steps stay
    pure prefill and all requests in the wave enter full-concurrency decode
    together without preemption. Once the locked decode wave shrinks to
    1/``_WAVE_TAIL_RELEASE_DIVISOR`` of its peak, the lock releases to admit the
    next wave while letting remaining prior-wave stragglers continue decoding so
    they finish and free KV blocks. A request whose worst-case KV footprint
    exceeds the empty-pool wave budget is admitted individually when idle.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drain_prefill = False
        self._prefill_wave_enabled = bool(
            envs.TPU_DP_SCHED_PREFILL_WAVE
            and self.vllm_config.speculative_config is None
            and not self.use_pp
        )
        if envs.TPU_DP_SCHED_PREFILL_WAVE and not self._prefill_wave_enabled:
            logger.warning_once(
                "TPU_DP_SCHED_PREFILL_WAVE=1 ignored: prefill-wave scheduling "
                "requires speculative_config=None and pipeline_parallel_size=1 "
                "(got speculative_config=%s, use_pp=%s).",
                self.vllm_config.speculative_config is not None,
                self.use_pp,
            )
        self._decode_wave_locked = False
        self._wave_peak_running = 0
        self._straggler_req_ids: set[str] = set()
        logger.info(
            "TpuDpScheduler active (prefill_wave=%s)",
            self._prefill_wave_enabled,
        )

    def _schedule_paused_new(self) -> SchedulerOutput:
        """Run one step while keeping newly prefills buffered."""
        prev_pause_state = self._pause_state
        if prev_pause_state == PauseState.UNPAUSED:
            self._pause_state = PauseState.PAUSED_NEW
        try:
            return super().schedule(False)
        finally:
            self._pause_state = prev_pause_state

    def _schedule_with_admission_cap(self, max_new_admissions: int) -> SchedulerOutput:
        """Run one unpaused step admitting at most ``max_new_admissions`` requests."""
        orig_max_running = self.max_num_running_reqs
        orig_max_active = self.max_num_active_reqs
        step_cap = (
            len(self.running)
            + self.num_waiting_for_streaming_input
            + max_new_admissions
        )
        self.max_num_running_reqs = min(orig_max_running, step_cap)
        self.max_num_active_reqs = min(orig_max_active, step_cap)
        try:
            return super().schedule(False)
        finally:
            self.max_num_running_reqs = orig_max_running
            self.max_num_active_reqs = orig_max_active

    def _next_waiting(self) -> Request | None:
        return next_unblocked_waiting(self)

    def _max_wave_tokens(self, request: Request) -> int:
        max_req_tokens = self.max_model_len
        if request.max_tokens is not None:
            max_req_tokens = min(
                max_req_tokens,
                max(
                    request.num_prompt_tokens + request.max_tokens - 1,
                    request.num_tokens,
                ),
            )
        return min(max_req_tokens + self.num_lookahead_tokens, self.max_model_len)

    def _count_admissible_into_wave(self) -> tuple[int, int, int, int]:
        """Count how many waiting requests fit cumulatively in the wave KV budget.

        Returns ``(admissible, slot_room, first_required_blocks, available_blocks)``.
        """
        slot_room = min(self.max_num_active_reqs, self.max_num_running_reqs) - (
            len(self.running) + self.num_waiting_for_streaming_input
        )
        available_blocks = (
            self.kv_cache_manager.block_pool.get_num_free_blocks()
            - self._spec_decode_step_blocks
        )
        if slot_room <= 0:
            return 0, slot_room, 0, available_blocks

        empty_blocks = self.kv_cache_manager.empty_kv_cache_blocks.blocks
        prefill_running_blocks = sum(
            blocks_to_allocate(
                self,
                r,
                r.num_computed_tokens,
                min(r.num_tokens, self.max_model_len),
                empty_blocks,
            )
            for r in self.running
            if r.is_prefill_chunk
        )
        wave_running_blocks = sum(
            blocks_to_allocate(
                self,
                r,
                r.num_computed_tokens,
                self._max_wave_tokens(r),
                empty_blocks,
            )
            for r in self.running
        )

        admissible = 0
        cum_isl_blocks = 0
        cum_wave_blocks = 0
        first_required_blocks = 0

        for idx, req in enumerate(iter_unblocked_waiting(self)):
            if idx >= slot_room:
                break
            if req.num_computed_tokens > 0:
                hit_blocks = empty_blocks
                hit_tokens = req.num_computed_tokens
            else:
                blocks, hit_tokens, _, _ = self._get_local_prefix_cache_hit(req)
                hit_blocks = blocks.blocks

            req_isl_blocks = blocks_to_allocate(
                self,
                req,
                hit_tokens,
                min(req.num_tokens, self.max_model_len),
                hit_blocks,
                apply_admission_cap=True,
            )
            req_wave_blocks = blocks_to_allocate(
                self,
                req,
                hit_tokens,
                self._max_wave_tokens(req),
                hit_blocks,
            )
            need_admission = (
                prefill_running_blocks
                + cum_isl_blocks
                + req_isl_blocks
                + self.kv_cache_manager.watermark_blocks
            )
            need_full_wave = wave_running_blocks + cum_wave_blocks + req_wave_blocks
            required_blocks = max(need_admission, need_full_wave)
            if idx == 0:
                first_required_blocks = required_blocks
            if available_blocks < required_blocks:
                break
            cum_isl_blocks += req_isl_blocks
            cum_wave_blocks += req_wave_blocks
            admissible += 1

        return admissible, slot_room, first_required_blocks, available_blocks

    def _has_unflushed_disjoint_placeholder(self) -> bool:
        prev_scheduled = self.prev_step_scheduled_req_ids
        return any(
            not r.is_prefill_chunk
            and r.request_id not in self._straggler_req_ids
            and r.num_output_placeholders > 0
            and r.request_id not in prev_scheduled
            for r in self.running
        )

    def _hold_decoding_requests_for_next_step(self) -> bool:
        hold_until = self.current_step + 2
        held_any = False
        for r in self.running:
            if (
                not r.is_prefill_chunk
                and r.request_id not in self._straggler_req_ids
                and r.next_decode_eligible_step < hold_until
            ):
                r.next_decode_eligible_step = hold_until
                held_any = True
        return held_any

    def _schedule_prefill_wave(self) -> SchedulerOutput:
        if not self.running:
            self._decode_wave_locked = False
            self._wave_peak_running = 0
            self._straggler_req_ids.clear()
        elif self._straggler_req_ids:
            running_ids = {r.request_id for r in self.running}
            self._straggler_req_ids.intersection_update(running_ids)

        self._wave_peak_running = max(self._wave_peak_running, len(self.running))
        if (
            self._decode_wave_locked
            and len(self.running) * _WAVE_TAIL_RELEASE_DIVISOR
            <= self._wave_peak_running
        ):
            # Only a straggler tail of the wave is left decoding: release the
            # lock so waiting requests can start the next wave, while exempting
            # remaining stragglers from decode hold so they finish and free KV.
            self._decode_wave_locked = False
            self._wave_peak_running = len(self.running)
            self._straggler_req_ids = {r.request_id for r in self.running}

        has_prefilling = any(r.is_prefill_chunk for r in self.running)
        target = None if self._decode_wave_locked else self._next_waiting()
        admissible = 0
        slot_room = 0
        first_required_blocks = 0
        available_blocks = 0
        if self._pause_state == PauseState.UNPAUSED and target is not None:
            (
                admissible,
                slot_room,
                first_required_blocks,
                available_blocks,
            ) = self._count_admissible_into_wave()

        if (
            admissible == 0
            and slot_room > 0
            and not self.running
            and target is not None
            and self._pause_state == PauseState.UNPAUSED
        ):
            # Nothing is running and a request slot is open, so waiting for KV
            # to free up can never succeed: admit this oversized request alone.
            logger.warning_once(
                "TpuDpScheduler prefill wave: request %s requires %d KV blocks "
                "(exceeding %d available KV blocks); falling back to single-request "
                "admission.",
                target.request_id,
                first_required_blocks,
                available_blocks,
            )
            return self._schedule_with_admission_cap(1)

        if has_prefilling or admissible > 0:
            held_any = self._hold_decoding_requests_for_next_step()
            if admissible > 0:
                output = self._schedule_with_admission_cap(admissible)
            else:
                output = self._schedule_paused_new()
            if (
                output.total_num_scheduled_tokens == 0
                and held_any
                and not has_prefilling
            ):
                # Target was rejected upstream (e.g. connector/encoder gate):
                # lock the decode wave so the next step releases held decodes.
                self._decode_wave_locked = True
                self._straggler_req_ids.clear()
            return output

        if self._has_unflushed_disjoint_placeholder():
            # Drain in-flight batch_queue entry before snapshotting all_token_ids
            # for held requests whose final prefill step has not popped yet.
            self._hold_decoding_requests_for_next_step()
            return self._schedule_paused_new()

        if self.running and (
            self._next_waiting() is not None
            or len(self.running)
            >= min(self.max_num_active_reqs, self.max_num_running_reqs)
        ):
            self._decode_wave_locked = True
            self._straggler_req_ids.clear()
        return self._schedule_paused_new()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.prefill_capacity_bound = True
        if self._prefill_wave_enabled:
            return self._schedule_prefill_wave()
        if not throttle_prefills:
            # Open window for draining accumulated prefills.
            self._drain_prefill = True
        if self._drain_prefill and not (
            self.waiting or getattr(self, "kv_holding_waiting", None)
        ):
            # In drain mode until all accumulated prefills are scheduled.
            self._drain_prefill = False
        if self._drain_prefill:
            return super().schedule(False)
        return self._schedule_paused_new()
