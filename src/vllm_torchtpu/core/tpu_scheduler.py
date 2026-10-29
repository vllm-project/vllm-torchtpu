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
from itertools import chain

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm_torchtpu import envs

logger = init_logger(__name__)


class TpuDpScheduler(AsyncScheduler):
    """Scheduler that aligns prefill admission across DP ranks.

    vLLM passes a DP-synchronized scheduling signal through ``throttle_prefills``.
    This scheduler uses that signal only to decide when to admit new prefill
    requests until all pending prefills are drained.

    When ``TPU_DP_SCHED_PREFILL_WAVE=1`` (non-speculative, non-PP), completed
    prefills are held via ``next_decode_eligible_step`` while a KV-bounded
    prefill wave is filling so prefill steps stay pure prefill and all requests
    in the wave enter full-concurrency decode together without preemption.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._drain_prefill = False
        self._prefill_wave_enabled = bool(
            envs.TPU_DP_SCHED_PREFILL_WAVE
            and self.vllm_config.speculative_config is None
            and not self.use_pp
        )
        self._decode_wave_locked = False
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

    def _next_waiting(self) -> Request | None:
        return next(
            (
                r
                for r in chain(getattr(self, "kv_holding_waiting", ()), self.waiting)
                if not self._is_blocked_waiting_status(r.status)
            ),
            None,
        )

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

    def _can_admit_into_wave(self, target: Request) -> bool:
        """Return whether ``target`` fits in both immediate admission and the
        full-wave KV budget so the wave decodes without preemption."""
        if len(self.running) + getattr(
            self, "num_waiting_for_streaming_input", 0
        ) >= min(self.max_num_active_reqs, self.max_num_running_reqs):
            return False

        coordinator = self.kv_cache_manager.coordinator
        empty_blocks = self.kv_cache_manager.empty_kv_cache_blocks.blocks
        if target.num_computed_tokens > 0:
            hit_blocks = empty_blocks
            hit_tokens = target.num_computed_tokens
        else:
            blocks, hit_tokens, _, _ = self._get_local_prefix_cache_hit(target)
            hit_blocks = blocks.blocks

        def blocks_for(
            request: Request,
            computed_tokens: int,
            total_tokens: int,
            computed_blocks: tuple,
            apply_admission_cap: bool = False,
        ) -> int:
            return coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=total_tokens,
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=computed_tokens,
                num_local_computed_tokens=computed_tokens,
                num_tokens_main_model=total_tokens,
                apply_admission_cap=apply_admission_cap,
                prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
            )

        prefill_running_blocks = sum(
            blocks_for(
                r,
                r.num_computed_tokens,
                min(r.num_tokens, self.max_model_len),
                empty_blocks,
            )
            for r in self.running
            if r.is_prefill_chunk
        )
        target_isl_tokens = min(target.num_tokens, self.max_model_len)
        target_isl_blocks = blocks_for(
            target,
            hit_tokens,
            target_isl_tokens,
            hit_blocks,
            apply_admission_cap=True,
        )
        need_admission = (
            prefill_running_blocks
            + target_isl_blocks
            + self.kv_cache_manager.watermark_blocks
        )

        wave_running_blocks = sum(
            blocks_for(
                r,
                r.num_computed_tokens,
                self._max_wave_tokens(r),
                empty_blocks,
            )
            for r in self.running
        )
        target_wave_blocks = blocks_for(
            target,
            hit_tokens,
            self._max_wave_tokens(target),
            hit_blocks,
        )
        need_full_wave = wave_running_blocks + target_wave_blocks

        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        spec_reserved = getattr(self, "_spec_decode_step_blocks", 0)
        return free - spec_reserved >= max(need_admission, need_full_wave)

    def _has_unflushed_disjoint_placeholder(self) -> bool:
        prev_scheduled = getattr(self, "prev_step_scheduled_req_ids", set())
        return any(
            not r.is_prefill_chunk
            and r.num_output_placeholders > 0
            and r.request_id not in prev_scheduled
            for r in self.running
        )

    def _hold_decoding_requests_for_next_step(self) -> list[tuple[Request, int]]:
        hold_until = self.current_step + 2
        saved: list[tuple[Request, int]] = []
        for r in self.running:
            if not r.is_prefill_chunk:
                prev_step = r.next_decode_eligible_step
                if prev_step < hold_until:
                    saved.append((r, prev_step))
                    r.next_decode_eligible_step = hold_until
        return saved

    def _schedule_prefill_wave(self) -> SchedulerOutput:
        if not self.running:
            self._decode_wave_locked = False

        has_prefilling = any(r.is_prefill_chunk for r in self.running)
        target = None if self._decode_wave_locked else self._next_waiting()
        can_admit = (
            self._pause_state == PauseState.UNPAUSED
            and target is not None
            and self._can_admit_into_wave(target)
        )

        if has_prefilling or can_admit:
            saved = self._hold_decoding_requests_for_next_step()
            if can_admit:
                output = super().schedule(False)
            else:
                output = self._schedule_paused_new()
            if output.total_num_scheduled_tokens == 0 and saved and not has_prefilling:
                # Target was rejected upstream (e.g. connector/encoder gate):
                # lock the decode wave so the next step releases held decodes.
                self._decode_wave_locked = True
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
        return self._schedule_paused_new()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.prefill_capacity_bound = True
        if getattr(self, "_prefill_wave_enabled", False):
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
