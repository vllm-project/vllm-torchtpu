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
"""Prefill each long prompt whole, in a step of its own.

A context-parallel prefill program takes a step that holds one request's whole
prompt, with no cached context and no other request, while every other step
runs the regular programs. With ``VLLM_TPU_SOLO_PREFILL_OVER_TOKENS=N`` every
step this scheduler makes has one of two shapes:

* solo: one waiting request with more than ``N`` tokens to compute, none
  computed yet and no encoder inputs, scheduled whole and alone. It skips the
  prefix cache read, and the running requests sit the step out. A solo step is
  only made when the request's KV blocks fit (plus the watermark while other
  requests run) and ``max_num_seqs`` has room for it.
* regular: at most ``N`` tokens. Requests that a solo step would take are not
  admitted in it.

A context-parallel threshold ``C`` on the padded token bucket agrees with this
when the bucket that holds ``N`` tokens is below ``C`` and the next bucket is
not. A prompt longer than a step's budget (``max_num_scheduled_tokens``, and
``max_num_batched_tokens`` less draft slots) cannot be solo and is prefilled in
regular chunks, and so is a request that leaves a blocked waiting status (such
as waiting for a grammar) during a regular step's admission pass.

A solo step stalls every running request for a whole prefill. While requests
run, at most ``VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE`` solo steps (0: no limit)
run in a row, then ``VLLM_TPU_SOLO_PREFILL_DECODE_STEPS`` regular steps that
schedule tokens run before the next one: waiting requests' TTFT traded for
running requests' TPOT.

Requests that sit a step out are resent with only the output tokens the model
runner reads (see ``_make_cached_request_data``).

The platform installs this scheduler when that variable is set;
``--scheduler-cls
vllm_torchtpu.core.experimental.prefill_first_scheduler.PrefillFirstScheduler``
selects it too.
"""

import contextlib
from collections.abc import Iterator

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_CLS = (
    "vllm_torchtpu.core.experimental.prefill_first_scheduler.PrefillFirstScheduler"
)


class PrefillFirstScheduler(AsyncScheduler):
    """Schedule each long prompt whole and alone; keep every other step at
    most ``VLLM_TPU_SOLO_PREFILL_OVER_TOKENS`` tokens."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.solo_over_tokens = envs.VLLM_TPU_SOLO_PREFILL_OVER_TOKENS
        self.solo_max_consecutive = envs.VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE
        self.solo_decode_steps = envs.VLLM_TPU_SOLO_PREFILL_DECODE_STEPS
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        # The longest prompt one step can hold whole.
        self.max_solo_tokens = min(
            self.max_num_scheduled_tokens,
            self.scheduler_config.max_num_batched_tokens - draft_slots,
        )
        self._check_solo_config()
        # Solo steps since the last run of `solo_decode_steps` regular steps,
        # and regular steps that scheduled tokens since the last solo step.
        self._solo_run = 0
        self._regular_run = 0
        # Set when a solo step admitted nothing, so the next step is regular.
        self._skip_solo = False
        logger.info(
            "PrefillFirstScheduler active (solo_over_tokens=%d, "
            "max_solo_tokens=%d, solo_max_consecutive=%d, solo_decode_steps=%d)",
            self.solo_over_tokens,
            self.max_solo_tokens,
            self.solo_max_consecutive,
            self.solo_decode_steps,
        )

    def _check_solo_config(self) -> None:
        if not 0 < self.solo_over_tokens < self.max_solo_tokens:
            raise ValueError(
                f"VLLM_TPU_SOLO_PREFILL_OVER_TOKENS ({self.solo_over_tokens}) must "
                f"be positive and below the step budget ({self.max_solo_tokens}); "
                "raise --max-num-batched-tokens to the longest prompt served."
            )
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < self.max_solo_tokens:
            # It caps a step's chunk of a prompt whenever more than one
            # request is running or waiting (a blocked KV-holding request
            # counts), which would split a solo prompt.
            raise ValueError(
                f"long_prefill_token_threshold ({threshold}) would split solo "
                f"prompts of up to {self.max_solo_tokens} tokens; set it to 0 "
                "(vLLM sets it when max_num_partial_prefills > 1) or to at "
                "least the step budget."
            )
        if self.solo_max_consecutive and self.solo_decode_steps < 1:
            raise ValueError(
                "VLLM_TPU_SOLO_PREFILL_DECODE_STEPS must be at least 1 when "
                "VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE is set."
            )
        if self.connector is not None or self.need_mamba_block_aligned_split:
            # A KV connector's external hits and Mamba-aligned splits both
            # shorten the step below the whole prompt.
            raise ValueError(
                "Solo prefill supports neither KV connectors nor Mamba "
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
        request = None
        if not throttle_prefills and not self._skip_solo:
            request = self._solo_candidate()
        self._skip_solo = False
        if request is not None:
            output = self._schedule_solo(request)
            if output.total_num_scheduled_tokens:
                self._solo_run += 1
                self._regular_run = 0
            return output
        output = self._schedule_regular(throttle_prefills)
        if output.total_num_scheduled_tokens:
            self._regular_run += 1
            if self._regular_run >= self.solo_decode_steps:
                self._solo_run = 0
        return output

    def _takes_solo_step(self, request: Request) -> bool:
        return (
            request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            and request.num_computed_tokens == 0
            and self.solo_over_tokens < request.num_tokens <= self.max_solo_tokens
            and not request.has_encoder_inputs
            # Scheduler.schedule() skips a request whose stale output is still
            # to be delivered.
            and not (request.num_stale_output_tokens and not request.drop_stale_output)
        )

    def _solo_candidate(self) -> Request | None:
        """The first waiting request a solo step takes now, if any."""
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

    def _fits(self, request: Request, num_tokens: int) -> bool:
        """Whether free KV blocks hold ``num_tokens`` of ``request`` from its
        start, counted as KVCacheManager.allocate_slots() counts them, plus
        the watermark while other requests run."""
        needed = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(num_tokens + self.num_lookahead_tokens, self.max_model_len),
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
        )
        if self.running:
            needed += self.kv_cache_manager.watermark_blocks
        reserved = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
        )
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return free - reserved >= needed

    def _schedule_solo(self, request: Request) -> SchedulerOutput:
        others = [r for r in self.waiting if r is not request]
        # The step must carry the whole prompt, so no prefix cache hit.
        skip_reading_prefix_cache = request.skip_reading_prefix_cache
        request.skip_reading_prefix_cache = True
        try:
            with self._hidden_waiting(others), self._hidden_running(self.running[:]):
                output = super().schedule()
        finally:
            request.skip_reading_prefix_cache = skip_reading_prefix_cache
        scheduled = output.num_scheduled_tokens
        if not scheduled:
            logger.warning(
                "PrefillFirstScheduler: solo request %s was not admitted; the "
                "next step is regular.",
                request.request_id,
            )
            self._skip_solo = True
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
        """The waiting requests a regular step does not admit: the first one a
        solo step would take, and every request behind it."""
        waiting = list(self.waiting)
        for i, r in enumerate(waiting):
            if self._takes_solo_step(r):
                return waiting[i:]
        return []

    @contextlib.contextmanager
    def _hidden_waiting(self, hidden: list[Request]) -> Iterator[None]:
        """Take ``hidden`` out of the waiting queue for one Scheduler.schedule()
        call, then put them back where they were."""
        if not hidden:
            yield
            return
        position = {r.request_id: i for i, r in enumerate(self.waiting)}
        self.waiting.remove_requests(hidden)
        try:
            yield
        finally:
            current = list(self.waiting)
            # Requests the step put back (preempted ones) stay ahead; the rest
            # return to their order before the step.
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
        """Take ``hidden`` out of RUNNING for one Scheduler.schedule() call,
        with the running-request limits lowered to match, then put them back
        in their places."""
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
            # Preempted requests have left RUNNING; admitted ones join the end.
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
        # A request that sat the last step out is resent with all its token
        # ids, of which GPUModelRunner._update_states reads the last
        # num_output_tokens. Solo steps make such steps routine, and an
        # 80K-token prompt costs milliseconds to serialize and decode per
        # request and step.
        for req_id, num_output_tokens in zip(data.req_ids, data.num_output_tokens):
            token_ids = data.all_token_ids.get(req_id)
            if token_ids is not None:
                data.all_token_ids[req_id] = token_ids[
                    len(token_ids) - num_output_tokens :
                ]
        return data
