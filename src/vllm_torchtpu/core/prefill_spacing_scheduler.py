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
  fixed debt, prefill then runs as soon as every such request's slack covers
  ``VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON`` times that step's extra time (the
  horizon gates only whether to hold, not the priced bins below). At most
  ``VLLM_TPU_PREFILL_TPOT_MAX_HOLD_S`` of steps in a row are deferred this way,
  and none while a pending prefill arrived more than
  ``VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S`` ago (when set).
  With ``VLLM_TPU_PREFILL_TPOT_PRICED`` such a step is not decode-only when a
  shorter step fits: it runs, in the scheduler's order
  (running chunks, then the waiting window, each shortest first), the whole
  remaining prefills that fit the largest prefill-token bin whose measured
  step time stays within every decoding request's slack (no chunk is cut).
* with ``VLLM_TPU_PREFILL_SRPT`` set, ranks the waiting window, the running
  prefill chunks and the reserve above by one aged shortest-remaining key:
  uncached prompt tokens left minus ``VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S``
  per second since the request arrived. Without it the key is the uncached
  tokens alone, and a long prompt waits as long as shorter ones keep arriving;
  with it a prompt with ``R`` tokens left goes ahead of every request that
  arrived ``R / rate`` seconds or more after it. It changes which prefill a
  step runs, never whether a step runs prefill (spacing and TPOT holds are
  unchanged).
* with ``VLLM_TPU_SIBLING_HOLD`` (PrefixAwareScheduler), a waiting request held
  for a running request's shared prefix is left out of the reserve: running
  chunks, the prefix's owner included, are not cut for it.

Select with ``--scheduler-cls``
``vllm_torchtpu.core.prefill_spacing_scheduler.PrefillSpacingScheduler``.
"""

import time
from itertools import chain, islice

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.core.prefix_aware_scheduler import (
    _TIER_NORMAL,
    PrefixAwareScheduler,
)
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

_LONG, _PREFILL, _DECODE = "long", "prefill", "decode"
# Prefill tokens per step a deferred step may still carry (priced TPOT hold),
# each timed by its own moving average.
_FIT_BINS = (512, 1024, 2048, 4096)


class PrefillSpacingScheduler(PrefixAwareScheduler):
    """Shortest remaining prefill first; decode-only steps after long chunks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spacing_frac = envs.VLLM_TPU_PREFILL_SPACING_FRAC
        self.long_prompt_tokens = envs.VLLM_TPU_PREFILL_SPACING_MIN_TOKENS
        self.max_starved_steps = envs.VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS
        self.tpot_target_s = envs.VLLM_TPU_PREFILL_TPOT_TARGET_MS / 1e3
        self.max_hold_s = envs.VLLM_TPU_PREFILL_TPOT_MAX_HOLD_S
        self.priced = envs.VLLM_TPU_PREFILL_TPOT_PRICED
        self.hold_horizon = envs.VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON
        self.max_wait_s = envs.VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S
        self.srpt = envs.VLLM_TPU_PREFILL_SRPT
        self.srpt_aging = envs.VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S
        # SRPT: wall clock of the current step (arrival_time is wall time) and
        # the aged keys of the admissible window, parallel to _window_cold.
        self._srpt_now = time.time()
        self._window_key: list[float] = []
        # Decoding request id -> when it was first scheduled to decode, and
        # its output tokens at the previous step (TPOT projection).
        self._decode_t0: dict[str, float] = {}
        self._seen_out: dict[str, int] = {}
        self._tokens_per_step = 1.0
        # Start of the current run of steps deferred by the TPOT target.
        self._held_since: float | None = None
        # Smallest positive TPOT slack of a decoding request at the last check.
        self._slack_s = float("inf")
        # Priced hold: step time by prefill-token bin, the prefill tokens this
        # step may carry (None: no limit) and how many it has taken.
        self._bin_s: dict[int, float] = {}
        self._pending_bins: dict[int, int] = {}
        self._fit_cap: int | None = None
        self._fit_used = 0
        self._fit_steps = 0
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
            "srpt=%s, srpt_aging_tokens_per_s=%.0f)",
            self.spacing_frac,
            self.long_prompt_tokens,
            self.max_starved_steps,
            self.tpot_target_s * 1e3,
            self.srpt,
            self.srpt_aging,
        )

    def _age_tokens(self, request: Request) -> float:
        """SRPT credit of ``request``: aging rate times seconds since arrival."""
        return self.srpt_aging * max(self._srpt_now - request.arrival_time, 0.0)

    def _srpt_key(self, request: Request) -> float:
        """Aged shortest-remaining key of a running prefill chunk."""
        return self._remaining(request) - self._age_tokens(request)

    def _rank_and_warmth(
        self, request: Request
    ) -> tuple[tuple[int, float, float], int]:
        rank, warmth = super()._rank_and_warmth(request)
        if self.srpt and rank[0] == _TIER_NORMAL:
            rank = (rank[0], rank[1] - self._age_tokens(request), rank[2])
        return rank, warmth

    def _promote_warmest(self, queue: RequestQueue) -> None:
        """Sort the whole window warmest first (once per step) and keep the
        uncached tokens of its admissible requests, in admission order."""
        if self._sorted_at_step == self.current_step:
            return
        self._sorted_at_step = self.current_step
        self._window_cold = []
        self._window_key = []
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
        # Requests the sibling hold keeps waiting this step reserve no budget:
        # a running chunk cut for them would delay the prefix they wait for.
        admissible = [
            (warmth, r)
            for (_, warmth), r in scored
            if not self._is_blocked_waiting_status(r.status)
            and not (self._sib_hold and self._sib_max_s > 0 and self._sibling_held(r))
        ]
        self._window_cold = [max(r.num_tokens - warmth, 1) for warmth, r in admissible]
        if self.srpt:
            self._window_key = [
                max(r.num_tokens - warmth, 1) - self._age_tokens(r)
                for warmth, r in admissible
            ]

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if self.srpt:
            self._srpt_now = time.time()
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
        self._fit_cap = None
        defer_debt = spacing or throttle_prefills
        if self.tpot_target_s > 0 and self._hold_for_tpot(decoders, queue):
            fit = self._fit_tokens() if self.priced and not defer_debt else 0
            if fit:
                self._fit_cap, self._fit_used = fit, 0
                self._fit_steps += 1
            spacing = not fit
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

    def _hold_for_tpot(
        self, decoders: list[Request], queue: RequestQueue | None
    ) -> bool:
        """Whether a long-chunk step now would push a decoding request that
        can still meet the TPOT target past it, while prefill is pending."""
        now = time.monotonic()
        self._track_decoders(decoders, now)
        pending = bool(queue) or any(r.is_prefill_chunk for r in self.running)
        hold = pending and self._short_of_slack(decoders, now)
        if hold and self.max_wait_s > 0 and self._waited_too_long(queue):
            hold = False
        if not hold:
            self._held_since = None
        elif self._held_since is None:
            self._held_since = now
        elif now - self._held_since > self.max_hold_s:
            # Bounds one run of holds; the next step may start another.
            self._held_since = None
            hold = False
        return hold

    def _waited_too_long(self, queue: RequestQueue | None) -> bool:
        """Whether a request still waiting for its first token, with a prefill
        that can run (waiting and not blocked, or a running chunk), arrived
        more than ``max_wait_s`` ago: holds then stop adding to its TTFT. A few
        long prompts held step after step otherwise set the mean TTFT."""
        waiting = (
            r for r in queue or () if not self._is_blocked_waiting_status(r.status)
        )
        chunks = (r for r in self.running if r.is_prefill_chunk)
        oldest = min(
            (
                r.arrival_time
                for r in chain(waiting, chunks)
                if r.num_output_tokens == 0
            ),
            default=None,
        )
        return oldest is not None and time.time() - oldest > self.max_wait_s

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
        self._slack_s = float("inf")
        if decode_s is None or long_s is None:
            return False
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
            if slack_s > 0:
                self._slack_s = min(self._slack_s, slack_s)
        return self._slack_s < self.hold_horizon * (long_s - decode_s)

    def _fit_tokens(self) -> int:
        """Largest prefill-token bin whose measured step fits the slack."""
        decode_s = self._step_s[_DECODE]
        fit = 0
        for tokens in _FIT_BINS:
            step_s = self._bin_s.get(tokens)
            if step_s is not None and step_s - decode_s < self._slack_s:
                fit = tokens
        return fit

    def _plan_reserve(self, prefilling: list[Request]) -> dict[str, int]:
        """Order RUNNING and size each prefill chunk's reserve for the
        shorter waiting requests the free running slots can admit."""
        # Off: both sides rank by tokens left; SRPT: by the aged key.
        rank = self._srpt_key if self.srpt else self._remaining
        window_rank = self._window_key if self.srpt else self._window_cold
        self.running[:] = [r for r in self.running if not r.is_prefill_chunk] + sorted(
            prefilling, key=rank
        )
        room = min(self.max_num_running_reqs, self.max_num_active_reqs) - len(
            self.running
        )
        room = max(room, 0)
        admissible = list(zip(window_rank[:room], self._window_cold[:room]))
        reserve = {}
        for r in prefilling:
            if self._starved.get(r.request_id, 0) >= self.max_starved_steps:
                continue
            key = rank(r)
            tokens = sum(c for k, c in admissible if k < key)
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
        if (
            self._in_schedule
            and self._fit_cap is not None
            and num_computed_tokens < request.num_prompt_tokens
        ):
            # Priced hold: whole remaining prefills only, up to the cap.
            left = request.num_tokens - num_computed_tokens
            if num_new_tokens < left or self._fit_used + left > self._fit_cap:
                num_new_tokens = 0
            self._fit_used += num_new_tokens
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
                if r.num_prompt_tokens - computed >= self.long_prompt_tokens:
                    kind = _LONG
                elif kind != _LONG:
                    kind = _PREFILL
            elif kind is None:
                kind = _DECODE
        if kind is None:
            return
        if self.priced and kind == _PREFILL and prefill_tokens <= _FIT_BINS[-1]:
            if len(self._pending_bins) >= _MAX_PENDING_STEPS:
                self._pending_bins.clear()
            self._pending_bins[id(output)] = next(
                b for b in _FIT_BINS if prefill_tokens <= b
            )
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
                "PrefillSpacingScheduler: steps=%d long=%d spacing=%d fit=%d "
                "step_s=%s bin_s=%s",
                self._steps,
                self._long_steps,
                self._spacing_steps,
                self._fit_steps,
                {k: round(v, 4) for k, v in self._step_s.items()},
                {k: round(v, 4) for k, v in self._bin_s.items()},
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
        fit_bin = self._pending_bins.pop(id(scheduler_output), None)
        if kind is not None and self._last_output_t is not None:
            dt = now - self._last_output_t
            self._time_step(self._step_s, kind, dt)
            if fit_bin is not None:
                self._time_step(self._bin_s, fit_bin, dt)
        self._last_output_t = now if self._pending_kinds else None
        return super().update_from_output(scheduler_output, model_runner_output)

    @staticmethod
    def _time_step(averages: dict, key, dt: float) -> None:
        avg = averages.get(key)
        if avg is None:
            averages[key] = dt
        elif dt < _MAX_STEP_RATIO * avg:
            averages[key] = avg + _STEP_TIME_EMA * (dt - avg)

    @staticmethod
    def _remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens
