# SPDX-License-Identifier: Apache-2.0
"""Engine step loop that chooses the next step late while a prefill step runs.

With async scheduling vLLM keeps two steps in flight. Once step ``k - 2``'s
result arrives, which is when step ``k - 1`` starts on the device, the engine
drains its input queue, schedules step ``k`` and blocks on step ``k - 1``. A
request that arrives while step ``k - 1`` runs has missed step ``k`` and waits
for the whole of it as well.

With ``TPU_DEFER_STEP`` set, while the one step in flight carries prefill
tokens, a step waits on the input queue until ``_MARGIN_S`` before that step
is expected to end, and only then lets vLLM schedule step ``k``. A request
queued in between ends the step without scheduling; vLLM's busy loop then
adds every queued request and steps again, which waits for the same deadline.
Requests that arrive in between join step ``k``. Steps keep their token budget
and the scheduler its policy; only the moment step ``k``'s requests are chosen
changes.

A step starts on the device when the step before it ends or when it is sent,
whichever is later, and it ends when the engine reads its result. Its expected
time is a low quantile of the measured times of recent steps of the same size;
a size with fewer than ``_MIN_SAMPLES`` measured steps is not deferred. A step
chosen after the step in flight has ended leaves the device idle, so the
quantile is low and the margin covers the time to send a step.
"""

import dataclasses
import time
from collections import deque
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Steps are timed by size in 512-token buckets. On an 8-chip prefill server a
# step's time grows from 71 ms (under 512 tokens) to 330 ms (4,096-4,607
# tokens), about 30-110 ms per bucket; the low quantile below takes the short
# end of each bucket.
_BUCKET_TOKENS = 512
# Each bucket keeps its newest 64 times: at about 3 timed prefill steps per
# second spread over 9 buckets, the last few minutes of load.
_SAMPLES_PER_BUCKET = 64
# With 8 samples the 10% quantile is the shortest time seen; fewer samples
# give no stable estimate, so the step is not deferred.
_MIN_SAMPLES = 8
# A step chosen after the step in flight ends leaves the device idle, while a
# step chosen early only takes fewer requests, so the estimate errs short.
_QUANTILE = 0.1
# Time from choosing step k to the device being ready for it. With these
# values 0 of 11,401 deferred steps on that server were chosen after the step
# in flight ended; the choice came 27 ms (p10) and 45 ms (p50) before its end.
_MARGIN_S = 0.030


def prefill_tokens(scheduler: Any, scheduler_output: Any) -> int:
    """Tokens of the step that belong to prompts. Read right after the step is
    scheduled, when each request's computed tokens include this step's."""
    total = 0
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests.get(req_id)
        if request is None:
            continue
        if request.num_computed_tokens - num_tokens < request.num_prompt_tokens:
            total += num_tokens
    return total


class StepTimes:
    """Measured device times of recent steps, by step size."""

    def __init__(self):
        self._samples: dict[int, deque[float]] = {}

    @staticmethod
    def bucket(num_tokens: int) -> int:
        return num_tokens // _BUCKET_TOKENS

    def add(self, num_tokens: int, seconds: float) -> None:
        self._samples.setdefault(
            self.bucket(num_tokens), deque(maxlen=_SAMPLES_PER_BUCKET)
        ).append(seconds)

    def expected(self, num_tokens: int) -> float | None:
        samples = self._samples.get(self.bucket(num_tokens))
        if samples is None or len(samples) < _MIN_SAMPLES:
            return None
        ordered = sorted(samples)
        return ordered[int(_QUANTILE * (len(ordered) - 1))]


@dataclasses.dataclass
class _Sent:
    """A step the engine sent to the workers and has not read back."""

    tokens: int
    prefill_tokens: int
    sent: float


class DeferredStepper:
    """Wraps one engine core's batch-queue step with the late choice."""

    def __init__(self, core: Any):
        self.core = core
        self.times = StepTimes()
        self.sent: dict[int, _Sent] = {}
        self.newest: _Sent | None = None
        self.last_end: float | None = None
        self._step = core.step_with_batch_queue
        self._wrap_scheduler(core.scheduler)
        self._wrap_dispatch(core.model_executor)

    def _wrap_scheduler(self, scheduler: Any) -> None:
        schedule = scheduler.schedule
        update_from_output = scheduler.update_from_output

        def timed_schedule(*args, **kwargs):
            scheduler_output = schedule(*args, **kwargs)
            self._on_scheduled(scheduler, scheduler_output)
            return scheduler_output

        def timed_update_from_output(scheduler_output, *args, **kwargs):
            self._on_result(scheduler_output)
            return update_from_output(scheduler_output, *args, **kwargs)

        scheduler.schedule = timed_schedule
        scheduler.update_from_output = timed_update_from_output

    def _wrap_dispatch(self, executor: Any) -> None:
        sample_tokens = executor.sample_tokens

        def timed_sample_tokens(*args, **kwargs):
            future = sample_tokens(*args, **kwargs)
            if self.newest is not None:
                self.newest.sent = time.monotonic()
            return future

        executor.sample_tokens = timed_sample_tokens

    def _on_scheduled(self, scheduler: Any, scheduler_output: Any) -> None:
        step = _Sent(
            tokens=scheduler_output.total_num_scheduled_tokens,
            prefill_tokens=prefill_tokens(scheduler, scheduler_output),
            sent=time.monotonic(),
        )
        # A step's record is dropped when its result is read. A step whose
        # result is never read would stay, so the records are cleared once
        # they hold 4 times the steps that can be in flight.
        if len(self.sent) >= 4 * self.core.batch_queue_size:
            self.sent.clear()
        self.sent[id(scheduler_output)] = step
        self.newest = step

    def _started(self, step: _Sent) -> float:
        if self.last_end is None:
            return step.sent
        return max(step.sent, self.last_end)

    def _on_result(self, scheduler_output: Any) -> None:
        now = time.monotonic()
        step = self.sent.pop(id(scheduler_output), None)
        if step is None:
            return
        if step.prefill_tokens > 0:
            self.times.add(step.tokens, now - self._started(step))
        self.last_end = now

    def _deadline(self) -> float | None:
        batch_queue = self.core.batch_queue
        if len(batch_queue) != 1:
            return None
        step = self.sent.get(id(batch_queue[0][1]))
        if step is None or step.prefill_tokens == 0:
            return None
        expected = self.times.expected(step.tokens)
        if expected is None:
            return None
        return self._started(step) + expected - _MARGIN_S

    def _request_queued_before(self, deadline: float) -> bool:
        input_queue = self.core.input_queue
        with input_queue.not_empty:
            return bool(
                input_queue.not_empty.wait_for(
                    lambda: input_queue.queue, timeout=deadline - time.monotonic()
                )
            )

    def step(self):
        deadline = self._deadline()
        if (
            deadline is not None
            and deadline > time.monotonic()
            and self._request_queued_before(deadline)
        ):
            return None, False
        return self._step()


def _blocker(core: Any) -> str | None:
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

    if not isinstance(core, EngineCoreProc):
        return "an in-process engine core, which has no input queue"
    if isinstance(core, DPEngineCoreProc):
        return (
            "data parallelism, where a step that runs no model makes the "
            "engine run a dummy batch"
        )
    if core.batch_queue_size != 2:
        return "the engine does not keep two steps in flight"
    if core.is_pooling_model:
        return "a pooling model"
    if core.vllm_config.parallel_config.pipeline_parallel_size > 1:
        return "pipeline parallelism"
    return None


def patch_engine_core_deferred_step() -> None:
    """Have every engine core choose its next step late behind prefill steps
    when ``TPU_DEFER_STEP`` is set. Installed in the engine-core process
    before the engine core is built; idempotent."""
    if not envs.TPU_DEFER_STEP:
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_deferred_step_patch", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        blocker = _blocker(self)
        if blocker is not None:
            logger.warning("TPU_DEFER_STEP: deferred step choice is off: %s.", blocker)
            return
        self.step_fn = DeferredStepper(self).step
        logger.info(
            "TPU_DEFER_STEP: deferred step choice is on: the next step is chosen %d ms "
            "before a running prefill step is expected to end.",
            round(_MARGIN_S * 1000),
        )

    EngineCore.__init__ = __init__
    EngineCore._tpu_deferred_step_patch = True
