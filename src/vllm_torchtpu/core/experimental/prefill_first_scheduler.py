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
"""Prefill each long prompt whole, in a step of its own."""

import contextlib
from collections.abc import Iterator

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DEFAULT_SOLO_OVER_TOKENS = 8192
DEFAULT_SOLO_DECODE_STEPS = 1
DEFAULT_SOLO_MAX_CONSECUTIVE = 0

SCHEDULER_CLS = (
    "vllm_torchtpu.core.experimental.prefill_first_scheduler.PrefillFirstScheduler"
)


class PrefillFirstScheduler(AsyncScheduler):
    """Schedule long prompts whole and alone; cap regular steps at
    ``solo_over_tokens``."""

    def __init__(
        self,
        *args,
        solo_over_tokens: int = DEFAULT_SOLO_OVER_TOKENS,
        solo_decode_steps: int = DEFAULT_SOLO_DECODE_STEPS,
        solo_max_consecutive: int = DEFAULT_SOLO_MAX_CONSECUTIVE,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        extra = getattr(self.vllm_config, "additional_config", None) or {}
        if isinstance(extra, dict):
            solo_over_tokens = extra.get("solo_over_tokens", solo_over_tokens)
            solo_decode_steps = extra.get("solo_decode_steps", solo_decode_steps)
            solo_max_consecutive = extra.get(
                "solo_max_consecutive", solo_max_consecutive
            )
        self.solo_over_tokens = int(solo_over_tokens)
        self.solo_decode_steps = int(solo_decode_steps)
        self.solo_max_consecutive = int(solo_max_consecutive)
        self.max_solo_tokens = min(
            self.max_num_scheduled_tokens, self.scheduler_config.max_num_batched_tokens
        )
        self._check_solo_config()
        self._solo_run = 0
        self._regular_run = 0
        self._skip_solo = False
        self._last_unadmitted_solo_id: str | None = None
        self._solo_fallback_ids: set[str] = set()
        logger.info(
            "PrefillFirstScheduler active (solo_over_tokens=%d, "
            "max_solo_tokens=%d, solo_max_consecutive=%d, solo_decode_steps=%d)",
            self.solo_over_tokens,
            self.max_solo_tokens,
            self.solo_max_consecutive,
            self.solo_decode_steps,
        )

    def _check_solo_config(self) -> None:
        par_cfg = self.vllm_config.parallel_config
        dp_size = par_cfg.data_parallel_size
        if dp_size > 1:
            raise ValueError(
                "PrefillFirstScheduler does not support data parallelism "
                f"(data_parallel_size={dp_size}): independent per-rank solo steps "
                "deadlock cross-DP collectives."
            )
        if getattr(par_cfg, "pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "PrefillFirstScheduler does not support pipeline parallelism."
            )
        if (
            getattr(par_cfg, "prefill_context_parallel_size", 1) > 1
            or getattr(par_cfg, "decode_context_parallel_size", 1) > 1
        ):
            raise ValueError(
                "PrefillFirstScheduler does not support standalone PCP or DCP."
            )
        if self.vllm_config.lora_config is not None:
            raise ValueError("PrefillFirstScheduler does not support LoRA.")
        if self.vllm_config.speculative_config is not None:
            raise ValueError(
                "PrefillFirstScheduler does not support speculative decoding."
            )
        if not 0 < self.solo_over_tokens < self.max_solo_tokens:
            raise ValueError(
                f"PrefillFirstScheduler solo_over_tokens ({self.solo_over_tokens}) "
                f"must be positive and below the step budget ({self.max_solo_tokens}); "
                "raise --max-num-batched-tokens to cover the longest solo prompt."
            )
        threshold = self.scheduler_config.long_prefill_token_threshold
        if threshold > 0:
            raise ValueError(
                f"long_prefill_token_threshold ({threshold}) must be 0 when "
                "PrefillFirstScheduler is active."
            )
        if self.solo_max_consecutive < 0:
            raise ValueError("solo_max_consecutive must be non-negative.")
        if self.solo_max_consecutive and self.solo_decode_steps < 1:
            raise ValueError(
                "solo_decode_steps must be at least 1 when solo_max_consecutive > 0."
            )
        if (
            self.vllm_config.kv_transfer_config is not None
            or self.connector is not None
            or self.need_mamba_block_aligned_split
        ):
            raise ValueError(
                "PrefillFirstScheduler supports neither KV connectors nor Mamba "
                "block-aligned splits."
            )
        if self.max_model_len > self.max_solo_tokens:
            logger.warning(
                "PrefillFirstScheduler: requests with more than %d tokens are "
                "prefilled in regular chunks of at most %d tokens.",
                self.max_solo_tokens,
                self.solo_over_tokens,
            )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        unwritten = self.kv_cache_manager.block_pool.__dict__.get(
            "_tpu_unwritten_block_ids"
        )
        if isinstance(unwritten, set):
            unwritten.clear()
        if self._solo_fallback_ids:
            self._solo_fallback_ids.intersection_update(self.requests)
        request = None
        if not throttle_prefills:
            if self._skip_solo:
                if (
                    not self._engine_has_releasable_blocks()
                    and self._last_unadmitted_solo_id is not None
                ):
                    self._solo_fallback_ids.add(self._last_unadmitted_solo_id)
                self._skip_solo = False
                self._last_unadmitted_solo_id = None
            else:
                request = self._solo_candidate()
        if request is not None:
            output = self._schedule_solo(request)
            if output.total_num_scheduled_tokens:
                self._solo_run += 1
                self._regular_run = 0
                return output
            if self._engine_has_releasable_blocks():
                return output
            self._solo_fallback_ids.add(request.request_id)
            self._skip_solo = False
            self._last_unadmitted_solo_id = None
        output = self._schedule_regular(throttle_prefills)
        if output.total_num_scheduled_tokens:
            self._regular_run += 1
            if self._regular_run >= max(1, self.solo_decode_steps):
                self._solo_run = 0
        return output

    def _has_prefix_cache_hit(self, request: Request) -> bool:
        if (
            not self.cache_config.enable_prefix_caching
            or request.skip_reading_prefix_cache
        ):
            return False
        return int(self.kv_cache_manager.get_computed_blocks(request)[1]) > 0

    def _takes_solo_step(self, request: Request) -> bool:
        return (
            request.request_id not in self._solo_fallback_ids
            and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            and request.num_computed_tokens == 0
            and self.solo_over_tokens < request.num_tokens <= self.max_solo_tokens
            and not request.has_encoder_inputs
            and not (request.num_stale_output_tokens and not request.drop_stale_output)
            and not self._has_prefix_cache_hit(request)
        )

    def _engine_has_releasable_blocks(self) -> bool:
        return bool(self.running or self.kv_holding_waiting)

    def _solo_candidate(self) -> Request | None:
        if self._pause_state != PauseState.UNPAUSED or any(
            not self._is_blocked_waiting_status(r.status)
            for r in self.kv_holding_waiting
        ):
            return None
        if self.running and 0 < self.solo_max_consecutive <= self._solo_run:
            return None
        num_active = len(self.running) + self.num_waiting_for_streaming_input
        if num_active >= min(self.max_num_active_reqs, self.max_num_running_reqs):
            return None
        request = next((r for r in self.waiting if self._takes_solo_step(r)), None)
        if request is None or not self._fits(request, request.num_tokens):
            return None
        return request

    def _blocks_needed_from_zero(self, request: Request, num_tokens: int) -> int:
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(num_tokens + self.num_lookahead_tokens, self.max_model_len),
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
            apply_admission_cap=True,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )

    def _exceeds_pool_capacity(self, request: Request, num_tokens: int) -> bool:
        total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
        if not isinstance(total_blocks, int) or total_blocks <= 0:
            return False
        return self._blocks_needed_from_zero(request, num_tokens) > total_blocks - 1

    def _fits(self, request: Request, num_tokens: int) -> bool:
        needed = self._blocks_needed_from_zero(request, num_tokens)
        if self.running:
            needed += self.kv_cache_manager.watermark_blocks
        reserved = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
        )
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return free - reserved >= needed

    def _schedule_solo(self, request: Request) -> SchedulerOutput:
        others = [r for r in self.waiting if r is not request]
        skip_reading_prefix_cache = request.skip_reading_prefix_cache
        request.skip_reading_prefix_cache = True
        prev_step = getattr(self, "current_step", None)
        try:
            with self._hidden_waiting(others), self._hidden_running(self.running[:]):
                output = super().schedule()
        finally:
            request.skip_reading_prefix_cache = skip_reading_prefix_cache
        scheduled = output.num_scheduled_tokens
        if not scheduled:
            if prev_step is not None:
                self.current_step = prev_step
            logger.warning(
                "PrefillFirstScheduler: solo request %s was not admitted; "
                "falling back to a regular step next.",
                request.request_id,
            )
            self._skip_solo = True
            self._last_unadmitted_solo_id = request.request_id
        elif scheduled != {request.request_id: request.num_tokens}:
            raise RuntimeError(
                f"PrefillFirstScheduler scheduled {scheduled} for solo request "
                f"{request.request_id} of {request.num_tokens} tokens; a solo "
                "step must hold that whole prompt alone."
            )
        return output

    def _schedule_regular(self, throttle_prefills: bool) -> SchedulerOutput:
        held = self._held_from_regular()
        budget = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = min(budget, self.solo_over_tokens)
        try:
            with self._hidden_waiting(held):
                return super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = budget

    def _held_from_regular(self) -> list[Request]:
        waiting = list(self.waiting)
        can_free_more = self._engine_has_releasable_blocks()
        for i, req in enumerate(waiting):
            if not self._takes_solo_step(req):
                continue
            if self._exceeds_pool_capacity(req, req.num_tokens):
                continue
            if not can_free_more and not self._fits(req, req.num_tokens):
                continue
            return waiting[i:]
        return []

    @contextlib.contextmanager
    def _hidden_waiting(self, hidden: list[Request]) -> Iterator[None]:
        if not hidden:
            yield
            return
        position = {r.request_id: i for i, r in enumerate(self.waiting)}
        self.waiting.remove_requests(hidden)
        try:
            yield
        finally:
            current = list(self.waiting)
            ahead = [r for r in current if r.request_id not in position]
            rest = sorted(
                [r for r in current if r.request_id in position] + hidden,
                key=lambda r: position[r.request_id],
            )
            self.waiting.remove_requests(current)
            for r in ahead + rest:
                self.waiting.add_request(r)

    @contextlib.contextmanager
    def _hidden_running(self, hidden: list[Request]) -> Iterator[None]:
        if not hidden:
            yield
            return
        order = self.running[:]
        hidden_ids = {r.request_id for r in hidden}
        self.running[:] = [r for r in order if r.request_id not in hidden_ids]
        self.max_num_running_reqs -= len(hidden)
        self.max_num_active_reqs -= len(hidden)
        try:
            yield
        finally:
            self.max_num_running_reqs += len(hidden)
            self.max_num_active_reqs += len(hidden)
            visible_ids = {r.request_id for r in self.running}
            order_ids = {r.request_id for r in order}
            self.running[:] = [
                r
                for r in order
                if r.request_id in hidden_ids or r.request_id in visible_ids
            ] + [r for r in self.running if r.request_id not in order_ids]

    def _make_cached_request_data(self, *args, **kwargs) -> CachedRequestData:
        data = super()._make_cached_request_data(*args, **kwargs)
        if self.connector is not None:
            return data
        for req_id, num_output_tokens in zip(data.req_ids, data.num_output_tokens):
            token_ids = data.all_token_ids.get(req_id)
            req = self.requests.get(req_id)
            if token_ids is not None and req is not None:
                real_out = list(token_ids[req.num_prompt_tokens :])
                if len(real_out) > num_output_tokens:
                    real_out = real_out[:num_output_tokens]
                pad = num_output_tokens - len(real_out)
                data.all_token_ids[req_id] = real_out + [-1] * pad
        return data
