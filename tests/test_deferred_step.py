# SPDX-License-Identifier: Apache-2.0
import sys
from collections import deque
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core import deferred_step

pytestmark = pytest.mark.cpu_test

LONG = 4096
LONG_S = 0.35
DECODE_S = 0.01
PREP_S = 0.002


class _Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _Inputs:
    """The engine's input queue, with requests arriving at fixed times. Like
    queue.Queue it has a ``not_empty`` condition and holds the queued requests
    in ``queue``; waiting on the condition moves the clock."""

    def __init__(self, clock, arrivals):
        self.clock = clock
        self.arrivals = sorted(arrivals)
        self.not_empty = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def queue(self):
        return deque(a for a in self.arrivals if a <= self.clock.now)

    def wait_for(self, predicate, timeout):
        until = self.clock.now + timeout
        if not predicate():
            if self.arrivals and self.arrivals[0] <= until:
                self.clock.now = self.arrivals[0]
            else:
                self.clock.now = until
        return predicate()

    def take_queued(self):
        queued = list(self.queue)
        del self.arrivals[: len(queued)]
        return queued


class _Device:
    """Runs sent steps one after another; a step starts once it is prepared
    and the step before it has ended."""

    def __init__(self, clock):
        self.clock = clock
        self.end = 0.0
        self.idle = 0.0

    def send(self, seconds):
        start = max(self.clock.now + PREP_S, self.end)
        self.idle += max(0.0, start - self.end) if self.end else 0.0
        self.end = start + seconds
        end = self.end
        clock = self.clock

        def result():
            clock.now = max(clock.now, end)
            return "output"

        return SimpleNamespace(result=result)


class _Scheduler:
    """Long prompts of ``prompt_chunks`` 4,096-token chunks each, run back to
    back for ``long_chunks`` steps, then decode; each arrival is a short prompt
    that joins the next step it is scheduled in."""

    def __init__(self, long_chunks, prompt_chunks):
        self.long_chunks = long_chunks
        self.prompt_chunks = prompt_chunks
        self.long_steps = 0
        self.waiting = []
        self.requests = {}
        self.first_step = {}
        self.steps = 0

    def has_requests(self):
        return True

    def add(self, request):
        self.waiting.append(request.at)

    def schedule(self, *args):
        self.steps += 1
        tokens = {}
        if self.long_chunks:
            self.long_chunks -= 1
            self.long_steps += 1
            tokens["long"] = LONG
            self.requests["long"] = SimpleNamespace(
                num_computed_tokens=LONG
                * ((self.long_steps - 1) % self.prompt_chunks + 1),
                num_prompt_tokens=LONG * self.prompt_chunks,
            )
        for arrival in self.waiting:
            self.first_step[arrival] = self.steps
            tokens[f"short-{arrival}"] = 16
            self.requests[f"short-{arrival}"] = SimpleNamespace(
                num_computed_tokens=16, num_prompt_tokens=10**6
            )
        self.waiting = []
        tokens["decode"] = 8
        self.requests["decode"] = SimpleNamespace(
            num_computed_tokens=10**7, num_prompt_tokens=1
        )
        return SimpleNamespace(
            num_scheduled_tokens=tokens,
            total_num_scheduled_tokens=sum(tokens.values()),
            long="long" in tokens,
        )

    def update_from_output(self, scheduler_output, model_output):
        return {}


def _core(clock, long_chunks, arrivals, prompt_chunks=1, decode_s=DECODE_S):
    device = _Device(clock)
    core = SimpleNamespace(
        batch_queue=deque(maxlen=2),
        batch_queue_size=2,
        is_pooling_model=False,
        scheduler=_Scheduler(long_chunks, prompt_chunks),
        input_queue=_Inputs(clock, arrivals),
        device=device,
    )
    core.model_executor = SimpleNamespace(sample_tokens=lambda *a, **k: core.pending)

    def step_with_batch_queue():
        """vLLM's batch-queue step without the branches the test does not use."""
        scheduler_output = core.scheduler.schedule()
        core.pending = device.send(LONG_S if scheduler_output.long else decode_s)
        future = core.model_executor.sample_tokens(None, non_block=True)
        core.batch_queue.appendleft((future, scheduler_output, None))
        if len(core.batch_queue) < core.batch_queue_size:
            return None, True
        future, scheduler_output, _ = core.batch_queue.pop()
        output = future.result()
        return core.scheduler.update_from_output(scheduler_output, output), True

    core.step_with_batch_queue = step_with_batch_queue
    return core


def _run(core, steps, step_fn):
    """vLLM's busy loop: add every queued request, then step. A step that ran
    no model is followed by a 1 ms sleep."""
    while steps:
        for arrival in core.input_queue.take_queued():
            core.scheduler.add(SimpleNamespace(at=arrival))
        if step_fn() == (None, False):
            core.input_queue.clock.now += 0.001
        else:
            steps -= 1


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(deferred_step, "time", clock)
    return clock


def test_step_times_need_samples_and_take_a_low_quantile():
    times = deferred_step.StepTimes()
    for seconds in (0.40, 0.36, 0.35, 0.38, 0.37, 0.39, 0.41):
        times.add(4100, seconds)
    assert times.expected(4100) is None
    times.add(4100, 0.42)
    assert times.expected(4100) == pytest.approx(0.35)
    assert times.expected(1024) is None


def test_prefill_tokens_count_only_prompt_tokens():
    scheduler = SimpleNamespace(
        requests={
            "prompt": SimpleNamespace(num_computed_tokens=4096, num_prompt_tokens=5000),
            "decode": SimpleNamespace(num_computed_tokens=5003, num_prompt_tokens=5000),
        }
    )
    output = SimpleNamespace(
        num_scheduled_tokens={"prompt": 4096, "decode": 3, "finished": 7}
    )
    assert deferred_step.prefill_tokens(scheduler, output) == 4096
    scheduler.requests["prompt"].num_computed_tokens = 5000
    output.num_scheduled_tokens["prompt"] = 904
    assert deferred_step.prefill_tokens(scheduler, output) == 904


def _first_step_of_arrival(clock, step_of, prompt_chunks=1):
    warmup = deferred_step._MIN_SAMPLES + 2
    clock.now = 100.0
    core = _core(clock, warmup + 4, [], prompt_chunks)
    step_fn = step_of(core)
    _run(core, warmup + 2, step_fn)
    arrival = clock.now + 0.2
    core.input_queue.arrivals.append(arrival)
    _run(core, 6, step_fn)
    return core.scheduler.first_step[arrival], core.device.idle


def _plain(core):
    return core.step_with_batch_queue


def _deferred(core):
    return deferred_step.DeferredStepper(core).step


@pytest.mark.parametrize("prompt_chunks", [1, 10**6])
def test_arrival_during_a_prefill_step_joins_the_next_step(clock, prompt_chunks):
    plain_step, _ = _first_step_of_arrival(clock, _plain, prompt_chunks)
    step, idle = _first_step_of_arrival(clock, _deferred, prompt_chunks)
    assert step == plain_step - 1
    assert idle == pytest.approx(0.0, abs=1e-9)


def test_steps_without_prefill_tokens_are_never_deferred(clock):
    """Long decode steps the size of steps that carried short prompts."""
    decode_s = 0.1
    arrivals = [100.0 + 3 * decode_s * i for i in range(2 * deferred_step._MIN_SAMPLES)]
    core = _core(clock, 0, arrivals, decode_s=decode_s)
    stepper = deferred_step.DeferredStepper(core)
    wait = stepper._request_queued_before

    def request_queued_before(deadline):
        in_flight = stepper.sent[id(core.batch_queue[0][1])]
        if in_flight.prefill_tokens == 0:
            pytest.fail("a step without prefill tokens was deferred")
        return wait(deadline)

    stepper._request_queued_before = request_queued_before
    _run(core, 100, stepper.step)


def test_steps_without_results_are_forgotten(clock):
    core = _core(clock, 0, [])
    stepper = deferred_step.DeferredStepper(core)
    outputs = [core.scheduler.schedule() for _ in range(4 * core.batch_queue_size + 1)]
    assert len(stepper.sent) <= 4 * core.batch_queue_size
    core.scheduler.update_from_output(outputs[0], None)
    assert stepper.last_end is None


class _EngineCore:
    def __init__(self, batch_queue_size=2, is_pooling_model=False, pp_size=1):
        self.batch_queue = (
            deque(maxlen=batch_queue_size) if batch_queue_size > 1 else None
        )
        self.batch_queue_size = batch_queue_size
        self.is_pooling_model = is_pooling_model
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
        )
        self.scheduler = SimpleNamespace(schedule=None, update_from_output=None)
        self.model_executor = SimpleNamespace(sample_tokens=None)
        self.step_with_batch_queue = lambda: None
        self.step_fn = self.step_with_batch_queue


@pytest.fixture
def engine_core(monkeypatch):
    in_process = type("EngineCore", (_EngineCore,), {})
    engine_core = type("EngineCoreProc", (in_process,), {})
    data_parallel = type("DPEngineCoreProc", (engine_core,), {})
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.core",
        SimpleNamespace(
            EngineCore=in_process,
            EngineCoreProc=engine_core,
            DPEngineCoreProc=data_parallel,
        ),
    )
    return engine_core


def test_engine_core_is_unchanged_when_switched_off(engine_core, monkeypatch):
    monkeypatch.setattr(deferred_step.envs, "TPU_DEFER_STEP", False, raising=False)
    init = engine_core.__init__
    deferred_step.patch_engine_core_deferred_step()
    assert engine_core.__init__ is init


def test_engine_core_steps_with_the_late_choice(engine_core, monkeypatch):
    monkeypatch.setattr(deferred_step.envs, "TPU_DEFER_STEP", True, raising=False)
    deferred_step.patch_engine_core_deferred_step()
    init = engine_core.__init__
    deferred_step.patch_engine_core_deferred_step()
    assert engine_core.__init__ is init
    core = engine_core()
    assert core.step_fn.__self__.core is core


@pytest.mark.parametrize(
    "config",
    [
        {"batch_queue_size": 1},
        {"batch_queue_size": 3},
        {"is_pooling_model": True},
        {"pp_size": 2},
    ],
)
def test_engine_core_without_two_steps_in_flight_is_unchanged(
    engine_core, monkeypatch, config
):
    monkeypatch.setattr(deferred_step.envs, "TPU_DEFER_STEP", True, raising=False)
    deferred_step.patch_engine_core_deferred_step()
    core = engine_core(**config)
    assert core.step_fn == core.step_with_batch_queue


def test_in_process_engine_core_is_unchanged(engine_core, monkeypatch):
    monkeypatch.setattr(deferred_step.envs, "TPU_DEFER_STEP", True, raising=False)
    deferred_step.patch_engine_core_deferred_step()
    core = engine_core.__base__()
    assert core.step_fn == core.step_with_batch_queue


def test_data_parallel_engine_core_is_unchanged(engine_core, monkeypatch):
    monkeypatch.setattr(deferred_step.envs, "TPU_DEFER_STEP", True, raising=False)
    deferred_step.patch_engine_core_deferred_step()
    data_parallel = sys.modules["vllm.v1.engine.core"].DPEngineCoreProc
    core = data_parallel()
    assert core.step_fn == core.step_with_batch_queue
