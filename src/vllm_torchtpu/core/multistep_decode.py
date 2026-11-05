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
"""Multi-step greedy decode, scheduler side (TPU_MULTISTEP_DECODE=K).

When nothing would prefill and every running request decodes greedily with
the top-token path (no logprobs, grammar, LoRA or min_tokens), a step
becomes a burst: each running request carries ``k - 1`` placeholder drafts
into ``schedule()``, so vLLM's speculative-decoding accounting schedules it
``k`` tokens, allocates their KV slots, counts ``k`` output placeholders and
later drops any token past a stop. The runner decodes the ``k`` positions
with ``k`` iterations of the same programs (``runner/multistep_decode.py``),
so the tokens are those of ``k`` single steps.

``k`` is ``TPU_MULTISTEP_DECODE``, cut to the most tokens any request has
left before its ``max_tokens`` (no burst computes positions nobody keeps),
and to what every request's ``max_model_len`` room allows. A request whose
in-flight tokens already reach its ``max_tokens`` is not scheduled. Steps
that cannot burst decode one token; the first per reason is logged and every
one is counted (summary at INFO every ``_SUMMARY_EVERY_STEPS`` steps).

A request that arrives waits for the bursts in flight (up to two under async
scheduling) before its prefill starts, so with
``TPU_MULTISTEP_DECODE_MAX_BURST_MS`` a burst is also cut to that many
milliseconds: ``k`` is at most the cap over the median time per decode
iteration of the last decode-only steps, measured between their outputs.
"""

from __future__ import annotations

import collections
import statistics
import time
from typing import Any

from vllm.sampling_params import SamplingType
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm_torchtpu import envs
from vllm_torchtpu.core.experimental.prefill_first_scheduler import (
    PrefillFirstScheduler,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_CLS = "vllm_torchtpu.core.multistep_decode.MultiStepAsyncScheduler"
PREFILL_FIRST_SCHEDULER_CLS = (
    "vllm_torchtpu.core.multistep_decode.MultiStepPrefillFirstScheduler"
)

_SUMMARY_EVERY_STEPS = 20000
# Decode-only steps whose iteration times the burst duration cap reads.
_ITERATION_WINDOW = 8


def _request_refusal(request: Request) -> str | None:
    """Why ``request`` keeps single-token steps, None when it may burst."""
    params = request.sampling_params
    if params is None:
        return "pooling"
    if params.sampling_type != SamplingType.GREEDY:
        return "sampling"
    if params.logprobs is not None:
        return "logprobs"
    if request.use_structured_output:
        return "structured output"
    if request.lora_request is not None:
        return "LoRA"
    if params.min_tokens:
        # The runner keeps such a request off the top-token path until a step
        # after its settled tokens reach min_tokens (runner/min_tokens.py).
        return "min_tokens"
    return None


def tokens_left(request: Request) -> int:
    """Tokens ``request`` may still generate beyond the ones in flight."""
    return request.max_tokens - (
        request.num_output_tokens + request.num_output_placeholders
    )


class MultiStepDecodeMixin:
    """Bursts of up to TPU_MULTISTEP_DECODE greedy tokens per request.

    Mix in ahead of a vLLM ``AsyncScheduler`` subclass.
    """

    # Set by the scheduler this mixes into.
    running: list[Request]
    max_num_active_reqs: int
    max_model_len: int
    max_num_scheduled_tokens: int
    num_spec_tokens: int
    get_request_counts: Any

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.multistep_k = envs.TPU_MULTISTEP_DECODE
        self.multistep_counts: collections.Counter[str] = collections.Counter()
        self._multistep_steps = 0
        self._max_burst_ms = envs.TPU_MULTISTEP_DECODE_MAX_BURST_MS
        self._iteration_ms: collections.deque[float] = collections.deque(
            maxlen=_ITERATION_WINDOW
        )
        # Outputs in flight, in schedule order, with the tokens per request of
        # the decode-only ones (None for the others).
        self._in_flight: collections.deque[tuple[SchedulerOutput, int | None]] = (
            collections.deque()
        )
        self._last_decode_output_s: float | None = None

    def _multistep_burst(self) -> tuple[int, str | None]:
        """This step's burst length, with the reason when it is 1."""
        if self.get_request_counts()[1]:
            return 1, "waiting requests"
        live = []
        for request in self.running:
            if request.num_computed_tokens < request.num_prompt_tokens:
                return 1, "prefill chunk"
            refusal = _request_refusal(request)
            if refusal is not None:
                return 1, refusal
            if tokens_left(request) > 0:
                live.append(request)
        if not live:
            return 1, None
        k = min(self.multistep_k, max(tokens_left(r) for r in live))
        # vLLM caps each request's tokens at its model-length room, which
        # would make the burst ragged.
        k = min([k] + [self.max_model_len - r.num_computed_tokens - 1 for r in live])
        k = min(k, self._burst_token_budget() // len(live))
        if k > 1 and self._max_burst_ms and self._iteration_ms:
            per_iteration = statistics.median(self._iteration_ms)
            capped = max(1, int(self._max_burst_ms / per_iteration))
            if capped < k:
                k = capped
                if k == 1:
                    return 1, "burst duration cap"
        return (k, None) if k > 1 else (1, "few tokens left")

    def _burst_token_budget(self) -> int:
        """Tokens the step that takes a burst may schedule."""
        return self.max_num_scheduled_tokens

    def schedule(self, *args: Any, **kwargs: Any) -> SchedulerOutput:
        k, reason = self._multistep_burst()
        # A request whose in-flight tokens reach its max_tokens has nothing
        # left to schedule; vLLM's own check sees one token in flight, not a
        # burst. Such requests sit out of `running` (keeping a burst uniform)
        # and count against the admission limit while they do.
        held = [r for r in self.running if tokens_left(r) <= 0]
        if held:
            order = {id(r): i for i, r in enumerate(self.running)}
            self.running = [r for r in self.running if tokens_left(r) > 0]
            self.max_num_active_reqs -= len(held)
        if k > 1:
            for request in self.running:
                request.spec_token_ids = [-1] * (k - 1)
        output: SchedulerOutput | None = None
        try:
            output = super().schedule(*args, **kwargs)  # type: ignore[misc]
        finally:
            if k > 1:
                # vLLM clears the drafts of the requests it schedules; one the
                # scheduler held back must not carry placeholders into a step
                # that is no burst.
                scheduled = output.num_scheduled_tokens if output else {}
                for request in self.running:
                    if request.request_id not in scheduled:
                        request.spec_token_ids = []
            if held:
                self.max_num_active_reqs += len(held)
                # Back in their places: the order picks preemption victims.
                self.running = sorted(
                    held + self.running, key=lambda r: order.get(id(r), len(order))
                )
        if k > 1:
            self._check_burst(output, k)
        decode_only = output.num_scheduled_tokens and reason not in (
            "waiting requests",
            "prefill chunk",
        )
        self._in_flight.append((output, k if decode_only else None))
        self._count(reason, k)
        return output

    def update_from_output(self, scheduler_output: SchedulerOutput, *args: Any):
        # Back-to-back decode-only steps: the time between their outputs is
        # the later step's, k iterations of it.
        k = None
        while self._in_flight:  # outputs scheduled earlier that never came back
            output, steps = self._in_flight.popleft()
            if output is scheduler_output:
                k = steps
                break
        now = time.monotonic()
        if k is not None and self._last_decode_output_s is not None:
            iteration_ms = (now - self._last_decode_output_s) * 1e3 / k
            if iteration_ms > 0:
                self._iteration_ms.append(iteration_ms)
        self._last_decode_output_s = now if k is not None else None
        return super().update_from_output(scheduler_output, *args)  # type: ignore[misc]

    def _check_burst(self, output: SchedulerOutput, k: int) -> None:
        scheduled = output.num_scheduled_tokens
        drafts = output.scheduled_spec_decode_tokens
        if output.scheduled_new_reqs or any(
            n != k or len(drafts.get(r, ())) != k - 1 for r, n in scheduled.items()
        ):
            raise RuntimeError(
                f"TPU_MULTISTEP_DECODE: a {k}-token burst came out ragged: "
                f"scheduled {dict(list(scheduled.items())[:8])}, "
                f"{len(output.scheduled_new_reqs)} new requests"
            )

    def _count(self, reason: str | None, k: int) -> None:
        if not self.running:
            return
        self._multistep_steps += 1
        key = f"burst {k}" if k > 1 else f"single: {reason}"
        if reason is not None and key not in self.multistep_counts:
            logger.info(
                "TPU_MULTISTEP_DECODE: a step decodes one token per request (%s)",
                reason,
            )
        self.multistep_counts[key] += 1
        if self._multistep_steps % _SUMMARY_EVERY_STEPS == 0:
            logger.info(
                "TPU_MULTISTEP_DECODE: %d steps with running requests: %s",
                self._multistep_steps,
                dict(self.multistep_counts),
            )

    def make_spec_decoding_stats(self, spec_decoding_stats: Any, *args: Any, **kw: Any):
        # Burst placeholders are not drafts: report no speculative decoding.
        if not self.num_spec_tokens:
            return spec_decoding_stats
        return super().make_spec_decoding_stats(  # type: ignore[misc]
            spec_decoding_stats, *args, **kw
        )


class MultiStepAsyncScheduler(MultiStepDecodeMixin, AsyncScheduler):
    """vLLM's async scheduler with multi-step greedy decode bursts."""


class MultiStepPrefillFirstScheduler(MultiStepDecodeMixin, PrefillFirstScheduler):
    """PrefillFirstScheduler with bursts in its regular steps.

    A burst only starts when nothing waits or prefills, so solo prefill steps,
    which take a waiting request, are the plain scheduler's. A burst counts as
    one regular step between solo steps, which only matters while one waits.
    """

    def _burst_token_budget(self) -> int:
        # Regular steps schedule at most the solo threshold.
        budget = self.max_num_scheduled_tokens
        return min(budget, self.solo_over_tokens) if self.solo_over_tokens else budget
