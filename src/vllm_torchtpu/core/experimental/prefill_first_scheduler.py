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
"""Prefill-first policies that trade running requests' TPOT for TTFT.

Solo prefill (``VLLM_TPU_SOLO_PREFILL_OVER_TOKENS=N > 0``). A context-parallel
prefill program takes a step that holds one request's whole prompt, with no
cached context and no other request, while every other step runs the regular
programs. Every step then has one of two shapes:

* solo: one waiting request with more than ``N`` tokens to compute, none
  computed yet and no encoder inputs, scheduled whole and alone. It skips the
  prefix cache read, and the running requests sit the step out. A solo step is
  only made when the request's KV blocks fit (plus the watermark while other
  requests run) and ``max_num_seqs`` has room for it.
* regular: at most ``N`` tokens. It admits no waiting request from the first
  one a solo step would take onward, so shorter requests do not pass a long
  prompt waiting for its solo step (upstream FCFS admission likewise stops at
  the first request that does not fit).

A context-parallel threshold ``C`` on the padded token bucket agrees with this
when the bucket that holds ``N`` tokens is below ``C`` and the next bucket is
not. A prompt longer than a step's budget (``max_num_scheduled_tokens``, and
``max_num_batched_tokens`` less draft slots) cannot be solo and is prefilled in
regular chunks, and so is a request that leaves a blocked waiting status (such
as waiting for a grammar) during a regular step's admission pass.

A solo step stalls every running request for a whole prefill. While requests
run, at most ``VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE`` solo steps (0: no limit)
run in a row, then ``VLLM_TPU_SOLO_PREFILL_DECODE_STEPS`` regular steps that
schedule tokens run before the next one.

Decode pacing (``VLLM_TPU_DECODE_INTERVAL_MS=T > 0``). While a regular step has
a prefill to schedule (a running prefill chunk, or a waiting request it may
admit), the running requests' decode tokens run on average once per ``T``
milliseconds, in the steps that start at or after each due time, instead of in
every step; the other steps carry only prefill. With a closed-loop client,
TTFT + output_len * TPOT is the time between a user's requests, so raising TPOT
toward its limit lowers TTFT at the same throughput.

Decode interval floor (``VLLM_TPU_MIN_DECODE_INTERVAL_MS=F > 0``). When nothing
is left to prefill (no running prefill chunk and no waiting request), a step
that would only decode waits until ``F`` milliseconds have passed since the
previous step that decoded: the device would only be decoding, so this raises
TPOT without taking prefill capacity, and lowers TTFT under a closed-loop
client the same way. The wait is spent in steps that schedule nothing; the
engine core yields 1 ms after each (EngineCore._process_engine_step) and, with
DP > 1, also runs a dummy forward, so it does not spin and a request that
arrives meanwhile waits about that long. The held decodes leave the runner's
persistent batch for the hold and are added back when they next decode.

TPOT pacing (``VLLM_TPU_DECODE_TPOT_TARGET_MS=T`` or
``VLLM_TPU_DECODE_TPOT_MS_PER_REQUEST=P`` > 0, or both). Each running request
earns one decode token per ``max(T, P * n)`` milliseconds, ``n`` being the
requests the engine holds (running and waiting) at the time, and a regular step
schedules its decode only while it holds an earned token. Each token scheduled
spends one (drafts included, so with speculative decoding TPOT runs above the
target). A request that fell behind (in long prefill steps, say) holds several
and decodes in every step until it is back on schedule, so each request's TPOT
tends to the target and its decode ends about ``output_len`` intervals after it
began. A held request resumes once it holds ``_RESUME_AFTER_TOKENS``. Prefill
chunks are never held back; a step with no earned decode and nothing to prefill
schedules nothing (the engine core then yields 1 ms and, with DP > 1, runs a
dummy forward). With a closed-loop client a request's completion sends the
user's next one, so decodes paced to a target finish spread out like their
first tokens instead of together, and the next requests do not queue behind
each other's prefills; under open-loop arrivals this only raises TPOT. ``P``
caps the engine's decode rate at ``1000 / P`` tokens per second shared equally
by its requests, so the target follows the load.

TPOT cap (``VLLM_TPU_DECODE_TPOT_CAP_MS=C`` or
``VLLM_TPU_DECODE_TPOT_CAP_MS_PER_REQUEST=Q`` > 0, or both), the converse of
pacing with the same accounting: each running request owes one decode token per
``max(C, Q * n)`` milliseconds from its first decode step on, each token it
schedules pays one, and while any request owes a whole token a step prefills
nothing (no prefill chunk, no admission from the waiting queue, no solo step)
and only decodes, so each request's TPOT stays at or below about the cap. A
request that owes a token is scheduled even if pacing, the decode interval or
the floor would hold it. Requests resuming from the KV-holding queue (streaming
input, remote KV) are not held back. ``VLLM_TPU_DECODE_TPOT_CAP_MS`` also takes
load tiers ``N:C[,N:C...]``: with ``n`` requests on the engine, ``C`` is that
of the largest ``N`` not above ``n``, and below every ``N`` the cap is off and
nothing is owed, so one server can cap only its highest concurrencies, where
TTFT has the room. Under a closed-loop client the requests then finish spread
out like their first tokens, as with pacing, and TPOT settles below the cap at
the cost of prefill time.

Prefill-aware DP load balancing (``VLLM_TPU_DP_LB_PREFILL_UNIT_TOKENS=U > 0``).
vLLM's data-parallel front end sends each new request to the engine with the
fewest waiting plus running requests, as each engine's scheduler reports them;
a replica that still has a long prompt to prefill then looks no busier than one
that only decodes, and a request routed there waits for that prefill. With
DP > 1, the counts the engine core publishes to the DP coordinator add, to the
running requests, one per U tokens (rounded up) that each running prefill chunk
and waiting request still has to prefill; scheduling, the logged stats and the
engine's unfinished-request checks keep the true counts.

Requests that sit a step out are resent with only the output tokens the model
runner reads (see ``_make_cached_request_data``).

The platform installs this scheduler when any of these policies is set;
``--scheduler-cls
vllm_torchtpu.core.experimental.prefill_first_scheduler.PrefillFirstScheduler``
selects it too.
"""

import contextlib
import dataclasses
from collections.abc import Iterator
from itertools import chain
from time import monotonic

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

# Every resume puts a request back into the model runner's persistent batch,
# which rewrites its whole token row (2.25 ms of host time measured for an
# 80K-token prompt; shorter prompts resume cheaper), so TPOT pacing resumes a
# request once per this many tokens, not every token.
_RESUME_AFTER_TOKENS = 16


@dataclasses.dataclass
class _Pace:
    """TPOT-pacing and TPOT-cap state of one decoding request."""

    # Decode tokens earned and not yet scheduled.
    earned: float = 1.0
    # Decoding in every step until back on schedule.
    resumed: bool = True
    # TPOT cap: decode tokens due at the cap and not yet scheduled.
    owed: float = 0.0


class PrefillFirstScheduler(AsyncScheduler):
    """Solo whole-prompt prefill steps, decode pacing, a decode interval floor,
    TPOT pacing, a TPOT cap and prefill-aware DP load-balancing counts (module
    docstring)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.solo_over_tokens = envs.VLLM_TPU_SOLO_PREFILL_OVER_TOKENS
        self.solo_max_consecutive = envs.VLLM_TPU_SOLO_PREFILL_MAX_CONSECUTIVE
        self.solo_decode_steps = envs.VLLM_TPU_SOLO_PREFILL_DECODE_STEPS
        self.decode_interval = envs.VLLM_TPU_DECODE_INTERVAL_MS / 1000
        self.lb_prefill_unit = envs.VLLM_TPU_DP_LB_PREFILL_UNIT_TOKENS
        self.min_decode_interval = envs.VLLM_TPU_MIN_DECODE_INTERVAL_MS / 1000
        self.tpot_target = envs.VLLM_TPU_DECODE_TPOT_TARGET_MS / 1000
        self.tpot_per_request = envs.VLLM_TPU_DECODE_TPOT_MS_PER_REQUEST / 1000
        # (min requests on the engine, cap in seconds), ascending.
        self.tpot_cap_tiers = tuple(
            (n, cap / 1000) for n, cap in envs.VLLM_TPU_DECODE_TPOT_CAP_MS
        )
        self.tpot_cap_per_request = envs.VLLM_TPU_DECODE_TPOT_CAP_MS_PER_REQUEST / 1000
        if not (
            self.solo_over_tokens
            or self.decode_interval
            or self.lb_prefill_unit
            or self.min_decode_interval
            or self.tpot_target
            or self.tpot_per_request
            or self.tpot_cap_tiers
            or self.tpot_cap_per_request
        ):
            raise ValueError(
                "PrefillFirstScheduler needs VLLM_TPU_SOLO_PREFILL_OVER_TOKENS, "
                "VLLM_TPU_DECODE_INTERVAL_MS, VLLM_TPU_MIN_DECODE_INTERVAL_MS, "
                "VLLM_TPU_DP_LB_PREFILL_UNIT_TOKENS, VLLM_TPU_DECODE_TPOT_TARGET_MS, "
                "VLLM_TPU_DECODE_TPOT_MS_PER_REQUEST, VLLM_TPU_DECODE_TPOT_CAP_MS or "
                "VLLM_TPU_DECODE_TPOT_CAP_MS_PER_REQUEST."
            )
        if self.lb_prefill_unit and self.parallel_config.data_parallel_size == 1:
            logger.warning(
                "VLLM_TPU_DP_LB_PREFILL_UNIT_TOKENS has no load balancer to "
                "inform without data parallelism; reporting true counts."
            )
            self.lb_prefill_unit = 0
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        # The longest prompt one step can hold whole.
        self.max_solo_tokens = min(
            self.max_num_scheduled_tokens,
            self.scheduler_config.max_num_batched_tokens - draft_slots,
        )
        if self.solo_over_tokens:
            self._check_solo_config()
        # Solo steps since the last run of `solo_decode_steps` regular steps,
        # and regular steps that scheduled tokens since the last solo step.
        self._solo_run = 0
        self._regular_run = 0
        # Set when a solo step admitted nothing, so the next step is regular.
        self._skip_solo = False
        # When the paced decodes are due next, and whether the next regular
        # step must run them (its predecessor paused them and scheduled
        # nothing).
        self._decode_due = 0.0
        self._force_decode = False
        # When the last step that decoded was scheduled (interval floor).
        self._last_decode = float("-inf")
        # TPOT pacing and cap: per decoding request, and when their clocks last
        # advanced.
        self._pace: dict[str, _Pace] = {}
        self._paced_at = 0.0
        # Set while the base scheduler's own logic reads the request counts:
        # schedule() and the unfinished-request checks see the true counts.
        self._true_counts = False
        logger.info(
            "PrefillFirstScheduler active (solo_over_tokens=%d, "
            "max_solo_tokens=%d, solo_max_consecutive=%d, solo_decode_steps=%d, "
            "decode_interval_ms=%g, min_decode_interval_ms=%g, "
            "dp_lb_prefill_unit_tokens=%d, tpot_target_ms=%g, "
            "tpot_ms_per_request=%g, tpot_cap_ms=%s, tpot_cap_ms_per_request=%g)",
            self.solo_over_tokens,
            self.max_solo_tokens,
            self.solo_max_consecutive,
            self.solo_decode_steps,
            self.decode_interval * 1000,
            self.min_decode_interval * 1000,
            self.lb_prefill_unit,
            self.tpot_target * 1000,
            self.tpot_per_request * 1000,
            ",".join(f"{n}:{cap * 1000:g}" for n, cap in self.tpot_cap_tiers) or "0",
            self.tpot_cap_per_request * 1000,
        )

    def _check_solo_config(self) -> None:
        if self.solo_over_tokens >= self.max_solo_tokens:
            raise ValueError(
                f"VLLM_TPU_SOLO_PREFILL_OVER_TOKENS ({self.solo_over_tokens}) must "
                f"be below the step budget ({self.max_solo_tokens}); raise "
                "--max-num-batched-tokens to the longest prompt served."
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

    @contextlib.contextmanager
    def _reading_true_counts(self) -> Iterator[None]:
        outer = self._true_counts
        self._true_counts = True
        try:
            yield
        finally:
            self._true_counts = outer

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        with self._reading_true_counts():
            return self._schedule(throttle_prefills)

    def get_num_unfinished_requests(self) -> int:
        # Pausing, idling and shutdown drains count real requests.
        with self._reading_true_counts():
            return super().get_num_unfinished_requests()

    def has_finished_requests(self) -> bool:
        with self._reading_true_counts():
            return super().has_finished_requests()

    def make_stats(self, *args, **kwargs):
        # Loggers and metrics count requests; only the coordinator's copy, which
        # EngineCore._maybe_publish_request_counts sends, carries the units.
        with self._reading_true_counts():
            return super().make_stats(*args, **kwargs)

    def get_request_counts(self) -> tuple[int, int]:
        """With prefill-aware load balancing, outside the base scheduler's own
        logic (so for the DP stats publisher), running also counts the units of
        prefill still pending."""
        num_running, num_waiting = super().get_request_counts()
        if not self.lb_prefill_unit or self._true_counts:
            return num_running, num_waiting
        prefilling = chain(
            (r for r in self.running if r.is_prefill_chunk),
            self.kv_holding_waiting,
            self.waiting,
        )
        # Prefix-cache hits a waiting request will get are not subtracted.
        pending_units = sum(
            -(-(r.num_tokens - r.num_computed_tokens) // self.lb_prefill_unit)
            for r in prefilling
        )
        return num_running + pending_units, num_waiting

    def _schedule(self, throttle_prefills: bool) -> SchedulerOutput:
        self._advance_decode_clocks()
        request = None
        if (
            self.solo_over_tokens
            and not throttle_prefills
            and not self._skip_solo
            and not self._behind_tpot_cap()
        ):
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
            self.solo_over_tokens > 0
            and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
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
        capped = self._behind_tpot_cap()
        held = list(self.waiting) if capped else self._held_from_regular()
        decodes = [r for r in self.running if not r.is_prefill_chunk]
        below_floor = self._below_decode_floor(decodes)
        interval_paused = [] if below_floor or capped else self._paused_decodes()
        paused = decodes if below_floor else interval_paused
        paced = self._tpot_paced(decodes)
        hidden = paused + [r for r in paced if r not in paused]
        if capped:
            # A decode that owes a token is scheduled whatever would hold it.
            hidden = [r for r in hidden if self._pace[r.request_id].owed < 1]
            hidden += [r for r in self.running if r.is_prefill_chunk]
        budget = self.max_num_scheduled_tokens
        if self.solo_over_tokens:
            self.max_num_scheduled_tokens = min(budget, self.solo_over_tokens)
        try:
            with self._hidden_waiting(held), self._hidden_running(hidden):
                output = super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = budget
        if interval_paused and not output.total_num_scheduled_tokens:
            self._force_decode = True
        for req_id, pace in self._pace.items():
            num_tokens = output.num_scheduled_tokens.get(req_id, 0)
            pace.earned -= num_tokens
            pace.owed -= num_tokens
        if any(r.request_id in output.num_scheduled_tokens for r in decodes):
            self._last_decode = monotonic()
        return output

    def _advance_decode_clocks(self) -> None:
        """Earn TPOT-pacing tokens and accrue TPOT-cap tokens for the running
        decodes over the time since the previous schedule() call. A request
        starts decoding at once and owes nothing then."""
        pacing = self.tpot_target or self.tpot_per_request
        capping = self.tpot_cap_tiers or self.tpot_cap_per_request
        if not (pacing or capping):
            return
        now = monotonic()
        elapsed, self._paced_at = now - self._paced_at, now
        # The interval is positive whenever there is a decode: the engine
        # holds it. The cap is 0 below every load tier.
        num_held = len(self.running) + len(self.waiting) + len(self.kv_holding_waiting)
        interval = max(self.tpot_target, self.tpot_per_request * num_held)
        tier_cap = next(
            (cap for n, cap in reversed(self.tpot_cap_tiers) if num_held >= n), 0.0
        )
        cap = max(tier_cap, self.tpot_cap_per_request * num_held)
        # Requests that stopped decoding (finished, preempted) drop out.
        pace: dict[str, _Pace] = {}
        for r in self.running:
            if r.is_prefill_chunk:
                continue
            state = self._pace.get(r.request_id)
            if state is None:
                state = _Pace()
            else:
                if pacing:
                    state.earned += elapsed / interval
                if capping:
                    # Below every load tier nothing is owed.
                    state.owed = state.owed + elapsed / cap if cap else 0.0
            pace[r.request_id] = state
        self._pace = pace

    def _tpot_paced(self, decodes: list[Request]) -> list[Request]:
        """The decodes that sit this regular step out for TPOT pacing: those
        without an earned token, and, after such a hold, those still short of
        ``_RESUME_AFTER_TOKENS``. Only regular steps decide this, so a solo
        step does not end a request's run of decodes."""
        if not (self.tpot_target or self.tpot_per_request):
            return []
        paced = []
        for r in decodes:
            state = self._pace[r.request_id]
            state.resumed = state.earned >= (
                1 if state.resumed else _RESUME_AFTER_TOKENS
            )
            if not state.resumed:
                paced.append(r)
        return paced

    def _behind_tpot_cap(self) -> bool:
        """Whether a running decode owes a whole token under the TPOT cap."""
        return any(pace.owed >= 1 for pace in self._pace.values())

    def _below_decode_floor(self, decodes: list[Request]) -> bool:
        """Whether a decode-only step still waits for the interval floor."""
        if not (self.min_decode_interval and decodes):
            return False
        if any(r.is_prefill_chunk for r in self.running) or (
            self.waiting or self.kv_holding_waiting
        ):
            return False
        return monotonic() < self._last_decode + self.min_decode_interval

    def _held_from_regular(self) -> list[Request]:
        """The waiting requests a regular step does not admit: the first one a
        solo step would take, and every request behind it."""
        waiting = list(self.waiting)
        for i, r in enumerate(waiting):
            if self._takes_solo_step(r):
                return waiting[i:]
        return []

    def _paused_decodes(self) -> list[Request]:
        """The running decodes that sit this regular step out for pacing."""
        if not self.decode_interval:
            return []
        decodes = [r for r in self.running if not r.is_prefill_chunk]
        if not decodes:
            return []
        now = monotonic()
        if self._force_decode or not self._prefill_pending():
            # Pacing restarts one interval after the last decode step.
            self._force_decode = False
            self._decode_due = now + self.decode_interval
            return []
        if now >= self._decode_due:
            # Keep the average interval: the next due time follows this one.
            self._decode_due = max(self._decode_due + self.decode_interval, now)
            return []
        return decodes

    def _prefill_pending(self) -> bool:
        """Whether a regular step has a prefill to schedule: a running prefill
        chunk, or a waiting request it admits (the first one ready, whose
        first chunk, or whole prompt under full-ISL reservation, fits)."""
        if any(r.is_prefill_chunk for r in self.running):
            return True
        if self._pause_state == PauseState.PAUSED_ALL:
            return False
        num_active = len(self.running) + self.num_waiting_for_streaming_input
        if num_active >= self.max_num_active_reqs:
            return False
        queues = [self.kv_holding_waiting]
        if self._pause_state == PauseState.UNPAUSED:
            queues.append(self.waiting)
        budget = self.max_num_scheduled_tokens
        if self.solo_over_tokens:
            budget = min(budget, self.solo_over_tokens)
        for r in chain(*queues):
            if self._takes_solo_step(r):
                # It and the requests behind it wait for its solo step.
                break
            if self._is_blocked_waiting_status(r.status) or (
                r.num_stale_output_tokens and not r.drop_stale_output
            ):
                continue
            if r.num_computed_tokens:
                return True
            if self.scheduler_reserve_full_isl:
                return self._fits(r, r.num_tokens)
            return self._fits(r, min(r.num_tokens, budget))
        return False

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
        # num_output_tokens. Both policies make such steps routine, and an
        # 80K-token prompt costs milliseconds to serialize and decode per
        # request and step.
        for req_id, num_output_tokens in zip(data.req_ids, data.num_output_tokens):
            token_ids = data.all_token_ids.get(req_id)
            if token_ids is not None:
                data.all_token_ids[req_id] = token_ids[
                    len(token_ids) - num_output_tokens :
                ]
        return data
