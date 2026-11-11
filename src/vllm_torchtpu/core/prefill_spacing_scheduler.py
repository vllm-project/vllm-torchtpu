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
"""Shortest-prefill-first admission with decode-only spacing of long prompts.

Every running request decodes inside every step, so a long prompt admitted as
back-to-back full chunks makes all decoding requests wait through a train of
the slowest steps, and vLLM's RUNNING-before-WAITING order lets that train take
the budget ahead of short waiting prompts. On top of PrefixAwareScheduler's
warmest-first admission, each step this scheduler:

* sorts the waiting window by uncached prompt tokens (not only its head) and
  orders RUNNING as decodes, then prefill chunks by remaining prompt tokens;
* leaves, from a running prefill chunk's share of the budget, the uncached
  tokens of the waiting requests that are shorter than it and that the free
  running slots can admit, so shorter prompts are prefilled first and the
  longer chunk takes what is left (shortest remaining prefill first over both
  queues). A chunk that gets less than one KV block for
  ``VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS`` steps in a row is no longer cut;
* after a step that runs a chunk of a prompt with at least
  ``VLLM_TPU_PREFILL_SPACING_MIN_TOKENS`` tokens left, owes
  ``VLLM_TPU_PREFILL_SPACING_FRAC`` times that kind of step's measured time in
  decode-only steps, and defers all prefill while the debt is positive and
  requests decode (KV loads still start). Step times are moving averages of
  the time between consecutive ``update_from_output`` calls per kind of step,
  timed only while the next step is already queued (async scheduling), so no
  chunk is spaced until a long-chunk step has been timed.

Select with ``--scheduler-cls``
``vllm_torchtpu.core.prefill_spacing_scheduler.PrefillSpacingScheduler``.
"""

import time
from itertools import islice

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Weight of the newest step in the per-kind step-time moving average.
_STEP_TIME_EMA = 0.2
# A gap above this many times the average is idle time, not a step.
_MAX_STEP_RATIO = 4.0
# Steps scheduled but not yet reported; bounds the kind map if outputs drop.
_MAX_PENDING_STEPS = 64
_LOG_EVERY = 10000

_LONG, _PREFILL, _DECODE = "long", "prefill", "decode"


class PrefillSpacingScheduler(PrefixAwareScheduler):
    """Shortest remaining prefill first; decode-only steps after long chunks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spacing_frac = envs.VLLM_TPU_PREFILL_SPACING_FRAC
        self.long_prompt_tokens = envs.VLLM_TPU_PREFILL_SPACING_MIN_TOKENS
        self.max_starved_steps = envs.VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS
        # Decode-only seconds owed to the decoding requests.
        self._debt_s = 0.0
        self._step_s: dict[str, float] = {}
        self._pending_kinds: dict[int, str] = {}
        self._last_output_t: float | None = None
        # Running prefill request id -> budget tokens left to shorter waiting
        # requests this step, and the RUNNING-pass tokens scheduled so far.
        self._reserve: dict[str, int] = {}
        self._running_tokens = 0
        self._in_schedule = False
        # Running prefill request id -> consecutive steps without a KV block of
        # progress, counted on steps that did not defer prefill.
        self._starved: dict[str, int] = {}
        self._sorted_at_step = -1
        self._window_cold: list[int] = []
        self._steps = 0
        self._spacing_steps = 0
        self._long_steps = 0
        logger.info(
            "PrefillSpacingScheduler active (spacing_frac=%.2f, "
            "long_prompt_tokens=%d, max_starved_steps=%d)",
            self.spacing_frac,
            self.long_prompt_tokens,
            self.max_starved_steps,
        )

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Sort the whole window warmest first (once per step) and keep the
        uncached tokens of its admissible requests, in admission order."""
        if self._sorted_at_step == self.current_step:
            return
        self._sorted_at_step = self.current_step
        self._window_cold = []
        if self.policy != SchedulingPolicy.FCFS or not queue:
            return
        # Upkeep PrefixAwareScheduler._promote_warmest does: probe refresh
        # clock and the external-warmth cache pruned to the live window.
        self._paw_step += 1
        window = list(islice(queue, self.WINDOW))
        scored = sorted(
            ((self._rank_and_warmth(r), r) for r in window), key=lambda t: t[0][0]
        )
        cache = self._paw_external_cache
        if cache and len(cache) > 4 * self.WINDOW:
            live = {r.request_id for r in window}
            for req_id in [k for k in cache if k not in live]:
                del cache[req_id]
        ordered = [r for _, r in scored]
        if ordered != window:
            queue.remove_requests(window)
            # prepend_requests() reverses its argument.
            queue.prepend_requests(reversed(ordered))
        self._window_cold = [
            max(r.num_tokens - warmth, 1)
            for (_, warmth), r in scored
            if not self._is_blocked_waiting_status(r.status)
        ]

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        queue = getattr(self, "kv_holding_waiting", None) or getattr(
            self, "waiting", None
        )
        if queue is not None:
            self._promote_warmest(queue)
        decoding = any(not r.is_prefill_chunk for r in self.running)
        if not decoding:
            self._debt_s = 0.0
        spacing = self.spacing_frac > 0 and decoding and self._debt_s > 0
        defer = throttle_prefills or spacing
        prefilling = [r for r in self.running if r.is_prefill_chunk]
        self._reserve = {} if defer else self._plan_reserve(prefilling)
        capacity_bound = self.prefill_capacity_bound
        if spacing:
            # Scheduler.schedule() defers prefill only while not capacity-bound.
            self.prefill_capacity_bound = False
        self._running_tokens = 0
        self._in_schedule = True
        try:
            output = super().schedule(defer)
        finally:
            self._in_schedule = False
            if spacing:
                self.prefill_capacity_bound = capacity_bound
        if not defer:
            self._update_starved(prefilling, output)
        self._account(output, decoding, spacing)
        return output

    def _plan_reserve(self, prefilling: list[Request]) -> dict[str, int]:
        """Order RUNNING and size each prefill chunk's reserve for the
        shorter waiting requests the free running slots can admit."""
        self.running[:] = [r for r in self.running if not r.is_prefill_chunk] + sorted(
            prefilling, key=self._remaining
        )
        room = min(self.max_num_running_reqs, self.max_num_active_reqs) - len(
            self.running
        )
        admissible = self._window_cold[: max(room, 0)]
        reserve = {}
        for r in prefilling:
            if self._starved.get(r.request_id, 0) >= self.max_starved_steps:
                continue
            remaining = self._remaining(r)
            tokens = sum(c for c in admissible if c < remaining)
            if tokens:
                reserve[r.request_id] = tokens
        return reserve

    def _reserve_prefill_lookahead(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        # Scheduler.schedule() calls this on every request it is about to
        # schedule, after its budget limits, so the RUNNING pass's tokens are
        # counted here and a chunk with a reserve is cut to what it leaves.
        # The cut goes first so the base rule keeping a chunk end out of the
        # prefill lookahead still holds for the cut chunk.
        running = self._in_schedule and request.status == RequestStatus.RUNNING
        reserve = self._reserve.get(request.request_id, 0) if running else 0
        if reserve:
            left = self.max_num_scheduled_tokens - self._running_tokens - reserve
            num_new_tokens = max(min(num_new_tokens, left), 0)
        num_new_tokens = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if running:
            self._running_tokens += num_new_tokens
        return num_new_tokens

    def _update_starved(
        self, prefilling: list[Request], output: SchedulerOutput
    ) -> None:
        progress = self.cache_config.block_size
        starved = {}
        for r in prefilling:
            n = output.num_scheduled_tokens.get(r.request_id, 0)
            if r.is_prefill_chunk and n < progress:
                starved[r.request_id] = self._starved.get(r.request_id, 0) + 1
        self._starved = starved

    def _account(self, output: SchedulerOutput, decoding: bool, spacing: bool) -> None:
        """Charge the step to the decode-only debt and remember its kind."""
        kind = None
        for req_id, n in output.num_scheduled_tokens.items():
            r = self.requests[req_id]
            # num_computed_tokens already includes this step's tokens.
            computed = r.num_computed_tokens - n
            if computed < r.num_prompt_tokens:
                if r.num_prompt_tokens - computed >= self.long_prompt_tokens:
                    kind = _LONG
                    break
                kind = _PREFILL
            elif kind is None:
                kind = _DECODE
        if kind is None:
            return
        if not decoding:
            self._debt_s = 0.0
        elif kind == _LONG:
            self._debt_s += self.spacing_frac * self._step_s.get(_LONG, 0.0)
        elif kind == _DECODE and self._debt_s > 0:
            self._debt_s = max(self._debt_s - self._step_s.get(_DECODE, 0.0), 0.0)
        if len(self._pending_kinds) >= _MAX_PENDING_STEPS:
            self._pending_kinds.clear()
        self._pending_kinds[id(output)] = kind
        self._steps += 1
        self._spacing_steps += spacing
        self._long_steps += kind == _LONG
        if self._steps % _LOG_EVERY == 0:
            logger.info(
                "PrefillSpacingScheduler: steps=%d long=%d spacing=%d step_s=%s",
                self._steps,
                self._long_steps,
                self._spacing_steps,
                {k: round(v, 4) for k, v in self._step_s.items()},
            )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # Outputs arrive in step order as the device finishes each step, so the
        # time since the previous output is this step's time while busy.
        # Timed only from an output that already had the next step queued.
        now = time.monotonic()
        kind = self._pending_kinds.pop(id(scheduler_output), None)
        if kind is not None and self._last_output_t is not None:
            dt = now - self._last_output_t
            avg = self._step_s.get(kind)
            if avg is None:
                self._step_s[kind] = dt
            elif dt < _MAX_STEP_RATIO * avg:
                self._step_s[kind] = avg + _STEP_TIME_EMA * (dt - avg)
        self._last_output_t = now if self._pending_kinds else None
        return super().update_from_output(scheduler_output, model_runner_output)

    @staticmethod
    def _remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens
