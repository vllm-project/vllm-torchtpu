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
  chunk is spaced until a long-chunk step has been timed;
* with ``VLLM_TPU_PREFILL_TPOT_TARGET_MS`` set, also defers prefill while a
  long-chunk step would push a decoding request that can still meet that TPOT
  past it. A request's TPOT is projected from the time since it first decoded
  and its tokens left to ``max_tokens`` at the measured tokens per step and
  decode-step time, so the rule only acts when clients set ``max_tokens`` and
  ``ignore_eos`` (as AgentX does). Any pending prefill, short prompts too, is
  priced as a long-chunk step: the next step may admit a long one. Unlike the
  fixed debt, prefill then runs as soon as every such request can afford the
  step. At most ``VLLM_TPU_PREFILL_TPOT_MAX_HOLD_S`` of steps in a row are
  deferred this way.
* with ``VLLM_TPU_PREFILL_SPACING_SHORT_TOKENS`` set, the debt does not defer
  prompts with at most that many uncached tokens left: they run whole, up to
  that many prefill tokens per step, and the step pays the debt like a
  decode-only step.
* with ``VLLM_TPU_PREFILL_TPOT_PRICED`` set, the TPOT rule prices the next
  step's prefill by its tokens: prefill is cut to the tokens whose cost (per
  token, measured on long-chunk steps) fits in the smallest slack it
  protects, and deferred only when less than one KV block fits.

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
# Weight of the newest step in the tokens-per-decode-step moving average.
_TOKENS_PER_STEP_EMA = 0.05
# Weight of the newest long-chunk step in its prefill-tokens moving average.
_LONG_TOKENS_EMA = 0.2

_LONG, _PREFILL, _DECODE = "long", "prefill", "decode"


class PrefillSpacingScheduler(PrefixAwareScheduler):
    """Shortest remaining prefill first; decode-only steps after long chunks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spacing_frac = envs.VLLM_TPU_PREFILL_SPACING_FRAC
        self.long_prompt_tokens = envs.VLLM_TPU_PREFILL_SPACING_MIN_TOKENS
        self.max_starved_steps = envs.VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS
        self.tpot_target_s = envs.VLLM_TPU_PREFILL_TPOT_TARGET_MS / 1e3
        self.max_hold_s = envs.VLLM_TPU_PREFILL_TPOT_MAX_HOLD_S
        self.short_tokens = envs.VLLM_TPU_PREFILL_SPACING_SHORT_TOKENS
        self.tpot_priced = envs.VLLM_TPU_PREFILL_TPOT_PRICED
        # Prefill tokens per long-chunk step (moving average): with the
        # long-chunk step time it prices one prefill token.
        self._long_tokens: float | None = None
        # Prefill tokens this step may still schedule (None: no cap), and
        # whether only whole prompts may take them.
        self._cap_left: int | None = None
        self._cap_whole = False
        self._capped_steps = 0
        # Decoding request id -> when it was first scheduled to decode, and
        # its output tokens at the previous step (TPOT projection).
        self._decode_t0: dict[str, float] = {}
        self._seen_out: dict[str, int] = {}
        self._tokens_per_step = 1.0
        # Start of the current run of steps deferred by the TPOT target.
        self._held_since: float | None = None
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
            "long_prompt_tokens=%d, max_starved_steps=%d, tpot_target_ms=%.1f, "
            "short_tokens=%d, tpot_priced=%s)",
            self.spacing_frac,
            self.long_prompt_tokens,
            self.max_starved_steps,
            self.tpot_target_s * 1e3,
            self.short_tokens,
            self.tpot_priced,
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
        decoders = [r for r in self.running if not r.is_prefill_chunk]
        decoding = bool(decoders)
        if not decoding:
            self._debt_s = 0.0
        in_debt = self.spacing_frac > 0 and decoding and self._debt_s > 0
        # Prefill tokens this step may schedule: None runs prefill freely,
        # 0 defers it.
        cap = None
        if in_debt:
            cap = self.short_tokens
        if self.tpot_target_s > 0:
            fit = self._hold_for_tpot(decoders, queue)
            if fit is not None:
                cap = fit if cap is None else min(cap, fit)
        if cap is not None and cap < self.cache_config.block_size:
            cap = 0
        spacing = cap == 0
        capped = cap is not None and cap > 0
        self._cap_left = cap if capped else None
        # The debt allows only whole short prompts; the TPOT price alone
        # allows a cut chunk.
        self._cap_whole = capped and in_debt
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
            self._cap_left = None
            if spacing:
                self.prefill_capacity_bound = capacity_bound
        if not defer and not capped:
            # A capped step holds chunks back by design, not for lack of
            # budget: it does not count toward starvation.
            self._update_starved(prefilling, output)
        self._account(output, decoding, spacing, capped)
        return output

    def _hold_for_tpot(
        self, decoders: list[Request], queue: RequestQueue | None
    ) -> int | None:
        """Prefill tokens the next step may schedule without pushing a
        decoding request that can still meet the TPOT target past it, while
        prefill is pending: None if a long-chunk step fits, else 0 (or, when
        priced, the tokens whose cost fits in the smallest slack)."""
        now = time.monotonic()
        self._track_decoders(decoders, now)
        pending = bool(queue) or any(r.is_prefill_chunk for r in self.running)
        slack_s = self._short_of_slack(decoders, now) if pending else None
        if slack_s is None:
            self._held_since = None
            return None
        if self._held_since is None:
            self._held_since = now
        elif now - self._held_since > self.max_hold_s:
            # Bounds one run of holds; the next step may start another.
            self._held_since = None
            return None
        if not self.tpot_priced:
            return 0
        return self._tokens_within(slack_s)

    def _tokens_within(self, slack_s: float) -> int:
        """Prefill tokens whose cost fits in slack_s, at the per-token cost of
        the measured long-chunk steps over decode-only steps."""
        decode_s = self._step_s.get(_DECODE)
        long_s = self._step_s.get(_LONG)
        if decode_s is None or long_s is None or not self._long_tokens:
            return 0
        per_token_s = max(long_s - decode_s, 1e-6) / self._long_tokens
        return int(slack_s / per_token_s)

    def _track_decoders(self, decoders: list[Request], now: float) -> None:
        tokens = rows = 0
        for r in decoders:
            seen = self._seen_out.get(r.request_id)
            if seen is None:
                self._decode_t0[r.request_id] = now
            else:
                tokens += r.num_output_tokens - seen
                rows += 1
            self._seen_out[r.request_id] = r.num_output_tokens
        if rows:
            self._tokens_per_step += _TOKENS_PER_STEP_EMA * (
                tokens / rows - self._tokens_per_step
            )
        if len(self._seen_out) > 2 * len(self.running) + _MAX_PENDING_STEPS:
            live = {r.request_id for r in self.running}
            for req_id in [k for k in self._seen_out if k not in live]:
                del self._seen_out[req_id]
                self._decode_t0.pop(req_id, None)

    def _short_of_slack(self, decoders: list[Request], now: float) -> float | None:
        """The smallest slack below one long-chunk step's extra time among
        decoding requests that can still meet the TPOT target, or None."""
        decode_s = self._step_s.get(_DECODE)
        long_s = self._step_s.get(_LONG)
        if decode_s is None or long_s is None:
            return None
        extra_s = long_s - decode_s
        smallest = None
        for r in decoders:
            t0 = self._decode_t0.get(r.request_id)
            if t0 is None or r.max_tokens <= 1:
                continue
            steps_left = max(r.max_tokens - r.num_output_tokens, 0) / max(
                self._tokens_per_step, 1.0
            )
            projected_s = now - t0 + steps_left * decode_s
            slack_s = self.tpot_target_s * (r.max_tokens - 1) - projected_s
            # A request already past the target is not protected: holding
            # prefill for it cannot bring it back under.
            if 0 < slack_s < extra_s:
                if not self.tpot_priced:
                    return slack_s
                if smallest is None or slack_s < smallest:
                    smallest = slack_s
        return smallest

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
        # A step under a prefill-token cap (short prompts during the debt, or
        # the priced TPOT rule): every WAITING request is a prefill.
        capped = (
            self._in_schedule
            and self._cap_left is not None
            and (request.is_prefill_chunk if running else True)
        )
        if capped:
            left = request.num_tokens - num_computed_tokens
            if self._cap_whole and left > self._cap_left:
                num_new_tokens = 0
            else:
                num_new_tokens = min(num_new_tokens, self._cap_left)
                if num_new_tokens < min(left, self.cache_config.block_size):
                    # A cut below one KV block is not worth a step slot.
                    num_new_tokens = 0
        num_new_tokens = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if capped:
            self._cap_left -= num_new_tokens
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

    def _account(
        self,
        output: SchedulerOutput,
        decoding: bool,
        spacing: bool,
        capped: bool = False,
    ) -> None:
        """Charge the step to the decode-only debt and remember its kind."""
        kind = None
        prefill_tokens = 0
        for req_id, n in output.num_scheduled_tokens.items():
            r = self.requests[req_id]
            # num_computed_tokens already includes this step's tokens.
            computed = r.num_computed_tokens - n
            if computed < r.num_prompt_tokens:
                prefill_tokens += n
                if r.num_prompt_tokens - computed >= self.long_prompt_tokens:
                    kind = _LONG
                elif kind != _LONG:
                    kind = _PREFILL
            elif kind is None:
                kind = _DECODE
        if kind is None:
            return
        if capped and kind == _LONG:
            # A cut chunk of a long prompt is neither timed nor charged as a
            # long-chunk step.
            kind = _PREFILL
        if kind == _LONG:
            n = prefill_tokens
            if self._long_tokens is None:
                self._long_tokens = float(n)
            else:
                self._long_tokens += _LONG_TOKENS_EMA * (n - self._long_tokens)
        if not decoding:
            self._debt_s = 0.0
        elif kind == _LONG:
            self._debt_s += self.spacing_frac * self._step_s.get(_LONG, 0.0)
        elif self._debt_s > 0 and (kind == _DECODE or capped):
            # A capped step runs at most a short prefill beside the decodes.
            self._debt_s = max(self._debt_s - self._step_s.get(_DECODE, 0.0), 0.0)
        if len(self._pending_kinds) >= _MAX_PENDING_STEPS:
            self._pending_kinds.clear()
        self._pending_kinds[id(output)] = kind
        self._steps += 1
        self._spacing_steps += spacing
        self._long_steps += kind == _LONG
        self._capped_steps += capped
        if self._steps % _LOG_EVERY == 0:
            logger.info(
                "PrefillSpacingScheduler: steps=%d long=%d spacing=%d capped=%d "
                "step_s=%s",
                self._steps,
                self._long_steps,
                self._spacing_steps,
                self._capped_steps,
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
