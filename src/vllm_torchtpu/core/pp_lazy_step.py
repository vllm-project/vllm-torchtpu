# SPDX-License-Identifier: Apache-2.0
"""Engine step loop that pushes the pipeline only when stage 0 would idle.

vLLM's engine keeps up to ``stages`` scheduler steps in flight and, once
that many are out or the scheduler has nothing more to run, waits for the
oldest. A step reaches the last stage only once ``stages - 2`` later steps
have been dispatched, so before that wait the executor sends what the step
still lacks: pushes, steps that carry nothing, or the settle when it is the
newest step (``distributed/pp_push.py``). Two things keep requests from
sharing the pipeline then. The scheduler returns a step without tokens when
everything runnable is in flight, and the engine queues it like any other,
so behind a lone request the queue fills with empty steps and the engine
waits; and that wait settles, so the next request opens a new burst whose
first hand-off waits until the last stage has finished the previous one.

This loop replaces ``EngineCore.step_with_batch_queue`` when
``TPU_PP_LAZY_PUSH`` is set. It drops steps without tokens that change no
worker state (their scheduler stats come with the next step), keeps up to
``stages`` steps with tokens in flight, never blocks on a result, and
settles once none of them is in flight, and before any other call to the
workers. While a step is still on its way to the last stage and the
scheduler has nothing to run, it pushes, but only once stage 0 has idled
for a while: a request that arrives before then takes the push's place in
the pipeline.

When stage 0 starts to idle: when every stage waits for all workers before
each hand-off, hand-off ``p`` happens at one moment on every stage, and
until hand-off ``p + 1`` stage ``r`` runs the forward of the burst's step
``p - r``. The last stage replies
to a step's forward or push call right after the hand-off that starts that
step on it, so its reply for step ``m`` marks hand-off ``m + stages - 1``.
Earlier hand-offs are estimated from the dispatch times and the expected
forward times: the step cost model's (``core/pp_chunks.py``) scaled by the
measured ones, or the measured ones alone. Step ``p`` is pushed once the
time since hand-off ``p`` exceeds ``_PUSH_DELAY`` of the longest forward
the other stages run before hand-off ``p + 1``.

Two things about the setup are needed, and other setups keep vLLM's loop:
the engine reads a reply only once it has arrived, which needs the last
stage on another host than the engine (a remote reader of its reply ring),
and the hand-off times come from the last stage's replies, which needs
every stage to drain the slice before each hand-off.
"""

import dataclasses
import time
from collections import deque
from concurrent.futures import Future
from typing import Any, cast

import vllm.envs as vllm_envs

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pp_push import wave_state
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Fraction of the other stages' longest forward that stage 0 waits for a
# real step before it pushes instead.
_PUSH_DELAY = 0.5
# Weight of the newest measurement in the running forward-time estimate,
# and the factor by which one measurement may differ from the estimate.
_MEASURE_WEIGHT = 0.25
_MEASURE_BAND = 2.0

# Executor calls that belong to a step; any other call settles first.
_STEP_METHODS = frozenset({"execute_model", "sample_tokens", "pp_push", "pp_settle"})

# Fields of a step without tokens that hold nothing for the workers.
_STATELESS_FIELDS = frozenset(
    {
        "scheduled_cached_reqs",
        "num_common_prefix_blocks",
        "scheduled_encoder_input_stats",
    }
)


def _carries_state(scheduler_output: Any) -> bool:
    """Whether a step without tokens changes the workers' state: finished
    or preempted requests, freed encoder outputs, or anything else the
    scheduler set."""
    if scheduler_output.scheduled_cached_reqs.req_ids:
        return True
    return any(
        getattr(scheduler_output, field.name)
        for field in dataclasses.fields(scheduler_output)
        if field.name not in _STATELESS_FIELDS
    )


def _banded(value: float, estimate: float) -> float:
    return min(max(value, estimate / _MEASURE_BAND), estimate * _MEASURE_BAND)


@dataclasses.dataclass
class _Slot:
    """One step of the burst on its way through the pipeline, pushes
    included."""

    dispatched: float
    # Expected forward time on one stage in seconds; 0 for a push.
    expected: float
    # The cost model's time for the step in ms, None without it.
    modelled_ms: float | None = None
    # When the last stage's reply to the step's forward or push call was
    # read, and in which drain.
    started_last: float | None = None
    started_drain: int = -1


@dataclasses.dataclass
class _Step:
    """A dispatched scheduler step whose result the engine has not read."""

    scheduler_output: Any
    exec_future: Future
    # Sampling, or the forward for a step without tokens.
    future: Future
    dispatched: float
    # None for a step without tokens.
    slot: _Slot | None = None


class LazyPushStepper:
    """One engine core's step loop; see the module docstring."""

    def __init__(self, core: Any):
        self.core = core
        self.executor = core.model_executor
        self.wave = wave_state(self.executor)
        self.stages = self.wave.stages
        self.socket = self.executor.response_mqs[
            self.executor.output_rank
        ].remote_socket
        self.timeout = vllm_envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
        self.modelled = getattr(core.scheduler, "step_estimate_ms", None)
        # Dispatched steps, oldest first; the engine core's batch queue.
        self.in_flight: deque[_Step] = deque()
        # This burst's newest steps in dispatch order, pushes included, and
        # the burst index of the oldest one held.
        self.slots: list[_Slot] = []
        self.first_slot = 0
        self.last_real = -1
        # Measured over modelled forward time, and the measured time in
        # seconds for steps the cost model does not cover.
        self.ratio = 1.0
        self.measured = 0.0
        self.drains = 0
        self.dropped = 0

    def step(self) -> tuple[dict[int, Any] | None, bool]:
        """Reads the replies that arrived, processes the oldest step if it
        finished, dispatches the next step if there is one, then pushes or
        settles the pipeline."""
        progressed = self._drain()
        outputs = None
        if self.in_flight and self._finished(self.in_flight[0]):
            outputs = self._process(self.in_flight.popleft())
            progressed = True
        real = self._real_in_flight()
        if (
            real < self.stages
            and len(self.in_flight) < 2 * self.stages
            and self.core.scheduler.has_requests()
            and self._dispatch()
        ):
            progressed = True
            real = self._real_in_flight()
        if self.wave.open:
            if real == 0:
                self._settle()
            elif self._push_due():
                self._push()
                progressed = True
        if not progressed and self.in_flight:
            self.socket.poll(1)
        return outputs, real > 0

    def _real_in_flight(self) -> int:
        return sum(1 for step in self.in_flight if step.slot is not None)

    def _drain(self) -> bool:
        """Reads every reply that has arrived, oldest call first; waits for
        a call that is not part of a step."""
        self.drains += 1
        queue = self.executor.futures_queue
        read = False
        while queue:
            future = queue[-1]
            if getattr(future, "_tpu_last_stage_reply", False) and not self.socket.poll(
                0
            ):
                break
            queue.pop()
            future._wait_for_response()
            read = True
        return read

    def _finished(self, step: _Step) -> bool:
        return step.future.done() or time.monotonic() - step.dispatched > self.timeout

    def _process(self, step: _Step) -> dict[int, Any] | None:
        core = self.core
        with (
            core.capture_iteration_details(step.scheduler_output) as iteration_details,
            core.log_error_detail(step.scheduler_output),
        ):
            if not step.future.done():
                # Past the timeout the step raises as vLLM's wait would.
                method = "sample_tokens" if step.slot else "execute_model"
                raise TimeoutError(f"RPC call to {method} timed out.")
            model_output = step.future.result()
            if model_output is None:
                # None from sample_tokens() means the forward failed.
                step.exec_future.result()
                raise RuntimeError("unexpected error")
        core._process_aborts_queue()
        engine_core_outputs = core.scheduler.update_from_output(
            step.scheduler_output, model_output
        )
        core._attach_iteration_details(engine_core_outputs, iteration_details)
        return engine_core_outputs

    def _dispatch(self) -> bool:
        """Schedules a step and dispatches it unless it holds nothing for
        the workers."""
        core = self.core
        scheduler = core.scheduler
        scheduler_output = scheduler.schedule(core._should_throttle_prefills())
        # A request whose grammar failed to compile finishes when any step's
        # output is processed; a step without tokens is sent for it only
        # when none is in flight.
        if (
            scheduler_output.total_num_scheduled_tokens <= 0
            and not _carries_state(scheduler_output)
            and not (
                getattr(scheduler, "grammar_compile_error_reqs", None)
                and not self.in_flight
            )
        ):
            self.dropped += 1
            return False
        opens = not self.wave.open
        with core.log_error_detail(scheduler_output):
            exec_future = self.executor.execute_model(scheduler_output, non_block=True)
        now = time.monotonic()
        if scheduler_output.total_num_scheduled_tokens <= 0:
            self.in_flight.append(
                _Step(scheduler_output, exec_future, exec_future, now)
            )
            return True
        grammar_output = scheduler.get_grammar_bitmask(scheduler_output)
        future = self.executor.sample_tokens(grammar_output, non_block=True)
        if opens:
            self.slots.clear()
            self.first_slot = 0
        modelled_ms = (
            self.modelled(scheduler_output) if self.modelled is not None else None
        )
        expected = (
            self.ratio * modelled_ms / 1000.0
            if modelled_ms is not None
            else self.measured
        )
        slot = _Slot(now, expected, modelled_ms)
        self._add_slot(slot)
        self.last_real = self.first_slot + len(self.slots) - 1
        exec_future.add_done_callback(lambda _: self._started_last(slot))
        future.add_done_callback(lambda _: self._sampled(slot))
        self.in_flight.append(_Step(scheduler_output, exec_future, future, now, slot))
        return True

    def _push_due(self) -> bool:
        """Whether to push the burst's next step now: a step with tokens
        still needs it to reach the last stage, no finished step waits to
        be processed first, and stage 0 has idled long enough since the
        hand-off that lets it run the next step."""
        nxt = self.first_slot + len(self.slots)
        if nxt > self.last_real + self.stages - 2:
            return False
        if self.in_flight and self.in_flight[0].future.done():
            return False
        start = self._handoff(nxt)
        if start is None:
            return False
        return time.monotonic() >= start + _PUSH_DELAY * self._others(nxt)

    def _settle(self) -> None:
        self.wave.settle(self.executor)
        settles = self.wave.settles
        if settles <= 5 or settles % 100 == 0:
            logger.info(
                "PP lazy push: %d settles, %d pushes, %d steps, %d empty "
                "steps dropped; forwards take %.2fx the cost model's time",
                settles,
                self.wave.pushes,
                self.wave.real_steps,
                self.dropped,
                self.ratio,
            )

    def _push(self) -> None:
        _, future = self.wave.push(self.executor)
        slot = _Slot(time.monotonic(), 0.0)
        self._add_slot(slot)
        future.add_done_callback(lambda _: self._started_last(slot))

    def _add_slot(self, slot: _Slot) -> None:
        """Appends a slot and drops any older than the ``stages`` newest,
        the only ones the hand-off times still read."""
        self.slots.append(slot)
        if len(self.slots) > self.stages:
            del self.slots[0]
            self.first_slot += 1

    def _handoff(self, k: int) -> float | None:
        """When hand-off ``k`` of the burst happened: read from the last
        stage's reply for step ``k - stages + 1``, None while that reply is
        due, estimated before that step exists."""
        observed = k - self.stages + 1
        if observed >= 0:
            return self.slots[observed - self.first_slot].started_last
        start = self.slots[0].dispatched
        for j in range(k):
            slot = self.slots[j]
            start = max(
                start + self._others(j), max(start, slot.dispatched) + slot.expected
            )
        return start

    def _others(self, k: int) -> float:
        """The longest expected forward stages 1 and up run between
        hand-offs ``k`` and ``k + 1``."""
        first = max(k - self.stages + 1, 0) - self.first_slot
        return max(
            (slot.expected for slot in self.slots[first : k - self.first_slot]),
            default=0.0,
        )

    def _started_last(self, slot: _Slot) -> None:
        slot.started_last = time.monotonic()
        slot.started_drain = self.drains

    def _sampled(self, slot: _Slot) -> None:
        """Measures the step's forward on the last stage, from its start to
        the sampling reply, within ``_MEASURE_BAND`` of the running
        estimate; replies read together say nothing."""
        if slot.started_last is None or slot.started_drain == self.drains:
            return
        seconds = time.monotonic() - slot.started_last
        if slot.modelled_ms:
            self.ratio += _MEASURE_WEIGHT * (
                _banded(seconds * 1000.0 / slot.modelled_ms, self.ratio) - self.ratio
            )
        elif self.measured == 0.0:
            self.measured = seconds
        else:
            self.measured += _MEASURE_WEIGHT * (
                _banded(seconds, self.measured) - self.measured
            )


def _patch_collective_rpc() -> None:
    """Have every lazy-push executor settle before a call that is not part
    of a step, and mark the futures of step calls, whose replies come from
    the last stage alone."""
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    if getattr(MultiprocExecutor, "_tpu_pp_lazy_patch", False):
        return
    orig = MultiprocExecutor.collective_rpc

    def collective_rpc(self, method, *args, **kwargs):
        state = getattr(self, "_tpu_wave_state", None)
        if state is None or not state.lazy:
            return orig(self, method, *args, **kwargs)
        step_call = isinstance(method, str) and method in _STEP_METHODS
        if not step_call:
            state.settle(self)
        result = orig(self, method, *args, **kwargs)
        if step_call and isinstance(result, Future):
            cast(Any, result)._tpu_last_stage_reply = True
        return result

    MultiprocExecutor.collective_rpc = collective_rpc
    MultiprocExecutor._tpu_pp_lazy_patch = True


def _lazy_push_blocker(core: Any, vllm_config: Any) -> str | None:
    """Why this engine core keeps vLLM's step loop, None if it need not."""
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    executor = core.model_executor
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size > 1:
        return "data parallelism"
    if getattr(parallel_config, "enable_fault_tolerance", False):
        return "fault tolerance"
    if core.is_pooling_model:
        return "a pooling model"
    if vllm_config.speculative_config is not None:
        return "speculative decoding"
    if (
        vllm_config.kv_transfer_config is not None
        or vllm_config.ec_transfer_config is not None
    ):
        return "a KV or encoder cache connector"
    if core.async_scheduling or core.batch_queue is None:
        return "async scheduling or no batch queue"
    if not isinstance(executor, MultiprocExecutor) or not getattr(
        MultiprocExecutor, "_tpu_pp_wave_patch", False
    ):
        return "an executor without the pipeline push"
    if (
        executor.kv_output_aggregator is not None
        or executor.ec_output_aggregator is not None
    ):
        return "an output aggregator"
    if not executor.response_mqs[executor.output_rank]._is_remote_reader:
        return "the last stage on the engine's host"
    if not all(executor.collective_rpc("pp_drains_before_handoff")):
        return "stages that hand off without draining the slice"
    return None


def patch_engine_core_for_lazy_push(vllm_config: Any) -> None:
    """Have every engine core of a pipeline deeper than two stages run
    ``LazyPushStepper`` when ``TPU_PP_LAZY_PUSH`` is set. Installed from the
    platform's config check (in-process engines) and from the engine-core
    process entry point; idempotent."""
    if (
        vllm_config is None
        or not envs.TPU_PP_LAZY_PUSH
        or vllm_config.parallel_config.pipeline_parallel_size <= 2
    ):
        return
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_tpu_pp_lazy_patch", False):
        return
    orig_init = EngineCore.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        blocker = _lazy_push_blocker(self, self.vllm_config)
        if blocker is not None:
            logger.warning("PP lazy push is off: %s.", blocker)
            return
        _patch_collective_rpc()
        stepper = LazyPushStepper(self)
        wave_state(self.model_executor).lazy = True
        self.batch_queue = stepper.in_flight
        self.step_fn = stepper.step
        logger.info(
            "PP lazy push: the engine core pushes %d stages itself.", stepper.stages
        )

    EngineCore.__init__ = __init__
    EngineCore._tpu_pp_lazy_patch = True
