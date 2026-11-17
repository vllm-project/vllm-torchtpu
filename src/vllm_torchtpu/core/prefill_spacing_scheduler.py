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
  deferred this way;
* ``VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON_S`` protects only requests whose
  projected remaining decode time fits in that horizon, so a hold is short and
  carries its request to the last token, and
  ``VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S`` stops holding once the oldest pending
  prefill arrived that long ago;
* with ``VLLM_TPU_PREFILL_TPOT_PRICE_RESERVE`` = R > 0, prices the step instead
  of holding all of its prefill: each decoding request that can still meet the
  target keeps R long-chunk extras (long-chunk step minus decode-only step) of
  its slack in reserve, the rest of the smallest slack is the time the step
  may add over a decode-only step, and the step's prefill tokens are capped at
  that time over a per-token cost learned from timed prefill steps
  (``VLLM_TPU_PREFILL_TPOT_US_PER_TOKEN`` until one is timed). Short prompts
  that fit run in full, a long chunk runs cut to the cap, and prefill is held
  (as above) only while the cap is under
  ``VLLM_TPU_PREFILL_TPOT_MIN_CHUNK`` tokens.

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
# Priced TPOT: only steps with at least this many prefill tokens update the
# per-token cost, with this weight; one sample moves it at most this factor.
_PRICE_MIN_TOKENS = 512
_PRICE_EMA = 0.1
_PRICE_MAX_STEP = 4.0

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
        self.hold_horizon_s = envs.VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON_S
        self.max_wait_s = envs.VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S
        self._wait_releases = 0
        self.price_reserve = envs.VLLM_TPU_PREFILL_TPOT_PRICE_RESERVE
        self.priced = self.price_reserve > 0 and self.tpot_target_s > 0
        self.min_chunk = max(envs.VLLM_TPU_PREFILL_TPOT_MIN_CHUNK, 1)
        # Modelled step time added per prefill token (priced TPOT).
        self._s_per_token = envs.VLLM_TPU_PREFILL_TPOT_US_PER_TOKEN / 1e6
        # Prefill tokens this step may still schedule; None = no cap.
        self._prefill_cap: int | None = None
        self._capped_steps = 0
        self._price_samples = 0
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
            "long_prompt_tokens=%d, max_starved_steps=%d, tpot_target_ms=%.1f)",
            self.spacing_frac,
            self.long_prompt_tokens,
            self.max_starved_steps,
            self.tpot_target_s * 1e3,
        )
        if self.priced:
            logger.info(
                "PrefillSpacingScheduler priced TPOT on "
                "(VLLM_TPU_PREFILL_TPOT_PRICE_RESERVE=%.2f, seed %.1f us/token, "
                "min chunk %d tokens)",
                self.price_reserve,
                self._s_per_token * 1e6,
                self.min_chunk,
            )
        if self.tpot_target_s > 0 and (self.hold_horizon_s > 0 or self.max_wait_s > 0):
            logger.info(
                "PrefillSpacingScheduler TPOT hold horizon on "
                "(VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON_S=%.2f, "
                "VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S=%.2f)",
                self.hold_horizon_s,
                self.max_wait_s,
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
        spacing = self.spacing_frac > 0 and decoding and self._debt_s > 0
        self._prefill_cap = None
        if self.tpot_target_s > 0:
            spacing = self._hold_for_tpot(decoders, queue) or spacing
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
        if defer:
            self._prefill_cap = None
        elif self._prefill_cap is not None:
            self._capped_steps += 1
        if not defer:
            self._update_starved(prefilling, output)
        self._account(output, decoding, spacing)
        return output

    def _hold_for_tpot(
        self, decoders: list[Request], queue: RequestQueue | None
    ) -> bool:
        """Whether a long-chunk step now would push a decoding request that
        can still meet the TPOT target past it, while prefill is pending."""
        now = time.monotonic()
        self._track_decoders(decoders, now)
        pending = bool(queue) or any(r.is_prefill_chunk for r in self.running)
        if self.priced:
            hold = pending and self._price_prefill(decoders, now)
        else:
            hold = pending and self._short_of_slack(decoders, now)
        if (
            self.max_wait_s > 0
            and pending
            and (hold or self._prefill_cap is not None)
            and self._oldest_wait_s(queue) > self.max_wait_s
        ):
            self._wait_releases += 1
            self._prefill_cap = None
            hold = False
        if not hold:
            self._held_since = None
        elif self._held_since is None:
            self._held_since = now
        elif now - self._held_since > self.max_hold_s:
            # Bounds one run of holds; the next step may start another.
            self._held_since = None
            self._prefill_cap = None
            hold = False
        return hold

    def _oldest_wait_s(self, queue: RequestQueue | None) -> float:
        """Seconds since the oldest pending prefill (waiting, or a running
        prompt chunk) arrived; arrival times are wall-clock."""
        oldest = None
        for r in self.running:
            if r.is_prefill_chunk and (oldest is None or r.arrival_time < oldest):
                oldest = r.arrival_time
        for r in queue or ():
            if oldest is None or r.arrival_time < oldest:
                oldest = r.arrival_time
        return 0.0 if oldest is None else time.time() - oldest

    def _slack_s(self, r: Request, now: float, decode_s: float) -> float | None:
        """Seconds request r's remaining steps may add over decode-only steps
        and still meet the TPOT target, or None if it is not projected or (with
        a hold horizon) too far from its last token to protect."""
        t0 = self._decode_t0.get(r.request_id)
        if t0 is None or r.max_tokens <= 1:
            return None
        steps_left = max(r.max_tokens - r.num_output_tokens, 0) / max(
            self._tokens_per_step, 1.0
        )
        left_s = steps_left * decode_s
        if self.hold_horizon_s > 0 and left_s > self.hold_horizon_s:
            return None
        projected_s = now - t0 + left_s
        return self.tpot_target_s * (r.max_tokens - 1) - projected_s

    def _price_prefill(self, decoders: list[Request], now: float) -> bool:
        """Priced TPOT: set this step's prefill token cap from the smallest
        slack, less the reserve, of the requests that can still meet the
        target; True (hold) if the cap is too small for a useful chunk."""
        decode_s = self._step_s.get(_DECODE)
        if decode_s is None:
            return False
        min_slack = None
        for r in decoders:
            slack_s = self._slack_s(r, now, decode_s)
            # Past the target: not protected, as in _short_of_slack.
            if slack_s is not None and slack_s > 0:
                min_slack = slack_s if min_slack is None else min(min_slack, slack_s)
        if min_slack is None:
            return False
        # Keep R long-chunk extras of the slack so one mis-priced or long step
        # does not push the request past the target and out of protection.
        long_s = self._step_s.get(_LONG)
        extra_s = max(long_s - decode_s, 0.0) if long_s is not None else 0.0
        avail_s = min_slack - self.price_reserve * extra_s
        if avail_s <= 0:
            return True
        cap = int(avail_s / max(self._s_per_token, 1e-9))
        if cap >= self.max_num_scheduled_tokens:
            return False
        if cap < self.min_chunk:
            return True
        self._prefill_cap = cap
        return False

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

    def _short_of_slack(self, decoders: list[Request], now: float) -> bool:
        decode_s = self._step_s.get(_DECODE)
        long_s = self._step_s.get(_LONG)
        if decode_s is None or long_s is None:
            return False
        extra_s = long_s - decode_s
        for r in decoders:
            slack_s = self._slack_s(r, now, decode_s)
            # A request already past the target is not protected: holding
            # prefill for it cannot bring it back under.
            if slack_s is not None and 0 < slack_s < extra_s:
                return True
        return False

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
        capped = (
            self._in_schedule
            and self._prefill_cap is not None
            and num_computed_tokens < request.num_prompt_tokens
        )
        if capped:
            num_new_tokens = min(num_new_tokens, self._prefill_cap)
        num_new_tokens = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if capped:
            self._prefill_cap -= num_new_tokens
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
        prefill_tokens = 0
        for req_id, n in output.num_scheduled_tokens.items():
            r = self.requests[req_id]
            # num_computed_tokens already includes this step's tokens.
            computed = r.num_computed_tokens - n
            if computed < r.num_prompt_tokens:
                prefill_tokens += n
                if kind == _LONG:
                    continue
                if r.num_prompt_tokens - computed >= self.long_prompt_tokens:
                    kind = _LONG
                    continue
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
        self._pending_kinds[id(output)] = (kind, prefill_tokens)
        self._steps += 1
        self._spacing_steps += spacing
        self._long_steps += kind == _LONG
        if self._steps % _LOG_EVERY == 0:
            logger.info(
                "PrefillSpacingScheduler: steps=%d long=%d spacing=%d step_s=%s "
                "capped=%d us_per_token=%.1f (%d samples)",
                self._steps,
                self._long_steps,
                self._spacing_steps,
                {k: round(v, 4) for k, v in self._step_s.items()},
                self._capped_steps,
                self._s_per_token * 1e6,
                self._price_samples,
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
        kind, prefill_tokens = self._pending_kinds.pop(
            id(scheduler_output), (None, 0)
        )
        if kind is not None and self._last_output_t is not None:
            dt = now - self._last_output_t
            avg = self._step_s.get(kind)
            if avg is None:
                self._step_s[kind] = dt
            elif dt < _MAX_STEP_RATIO * avg:
                self._step_s[kind] = avg + _STEP_TIME_EMA * (dt - avg)
            if self.priced:
                self._learn_price(dt, prefill_tokens)
        self._last_output_t = now if self._pending_kinds else None
        return super().update_from_output(scheduler_output, model_runner_output)

    def _learn_price(self, dt: float, prefill_tokens: int) -> None:
        """Per-token prefill cost from a timed step: its time over the
        decode-only step time, per prefill token."""
        decode_s = self._step_s.get(_DECODE)
        if decode_s is None or prefill_tokens < _PRICE_MIN_TOKENS:
            return
        sample = max(dt - decode_s, 0.0) / prefill_tokens
        cur = self._s_per_token
        if cur > 0:
            sample = min(max(sample, cur / _PRICE_MAX_STEP), cur * _PRICE_MAX_STEP)
        self._s_per_token = cur + _PRICE_EMA * (sample - cur)
        self._price_samples += 1

    @staticmethod
    def _remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens
