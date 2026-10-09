# SPDX-License-Identifier: Apache-2.0
import contextlib
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_torchtpu.core import pp_lazy_step
from vllm_torchtpu.distributed import pp_push

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def fake_rings(monkeypatch):
    """Keeps the ring widening off vLLM's MessageQueue."""
    from vllm.distributed.device_communicators import shm_broadcast

    monkeypatch.setattr(pp_push, "_ring_chunks_needed", 0)
    monkeypatch.setattr(shm_broadcast, "MessageQueue", type("_Queue", (), {}))


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=64.0)
    monkeypatch.setattr(
        pp_lazy_step, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    return clock


def _config(pp=4, dp=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enable_fault_tolerance=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
    )


def _output(tokens, **fields):
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.total_num_scheduled_tokens = tokens
    if tokens:
        scheduler_output.num_scheduled_tokens = {"r": tokens}
    for name, value in fields.items():
        setattr(scheduler_output, name, value)
    return scheduler_output


class _Socket:
    """The engine's end of the last stage's reply ring."""

    def __init__(self):
        self.arrived = 0
        self.polls = []

    def poll(self, timeout=None):
        self.polls.append(timeout)
        return self.arrived


def _executor(monkeypatch, stages=4):
    """A MultiprocExecutor stand-in with the push and the lazy-push patches;
    replies are read from the last stage's socket in call order."""
    from vllm.v1.executor import multiproc_executor

    class _Reply(Future):
        def __init__(self, executor, method, arg):
            super().__init__()
            self.executor = executor
            self.method = method
            self.arg = arg
            executor.futures_queue.appendleft(self)

        def _wait_for_response(self):
            socket = self.executor.socket
            if socket.arrived <= 0:
                self.set_exception(TimeoutError(self.method))
                return
            socket.arrived -= 1
            self.set_result(
                self.executor.replies.get(self.method, (self.method, self.arg))
            )

        def result(self, timeout=None):
            while not self.done():
                self.executor.futures_queue.pop()._wait_for_response()
            return super().result()

    class _Executor:
        kv_output_aggregator = None
        ec_output_aggregator = None

        def __init__(self):
            self.vllm_config = _config(stages)
            self.output_rank = stages - 1
            self.futures_queue = deque()
            self.socket = _Socket()
            self.response_mqs = {
                self.output_rank: SimpleNamespace(
                    remote_socket=self.socket, _is_remote_reader=True
                )
            }
            self.calls = []
            self.replies = {"pp_drains_before_handoff": [True] * stages}

        def collective_rpc(
            self,
            method,
            timeout=None,
            args=(),
            kwargs=None,
            non_block=False,
            unique_reply_rank=None,
        ):
            self.calls.append(method)
            future = _Reply(self, method, args[0] if args else None)
            return future if non_block else future.result()

        def execute_model(self, scheduler_output, non_block=False):
            return self.collective_rpc(
                "execute_model",
                args=(scheduler_output,),
                non_block=non_block,
                unique_reply_rank=self.output_rank,
            )

        def sample_tokens(self, grammar_output, non_block=False):
            return self.collective_rpc(
                "sample_tokens",
                args=(grammar_output,),
                non_block=non_block,
                unique_reply_rank=self.output_rank,
            )

    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", _Executor)
    pp_push.patch_executor_for_pp_wave(_config(stages))
    pp_lazy_step._patch_collective_rpc()
    return _Executor()


class _Scheduler:
    def __init__(self, *steps, estimate_ms=125.0):
        self.steps = deque(steps)
        self.estimate_ms = estimate_ms
        self.grammar_compile_error_reqs = set()
        self.idle = False
        self.schedules = 0

    def has_requests(self):
        return not self.idle

    def schedule(self, throttle_prefills):
        self.schedules += 1
        return self.steps.popleft() if self.steps else _output(0)

    def get_grammar_bitmask(self, scheduler_output):
        return None

    def step_estimate_ms(self, scheduler_output):
        return self.estimate_ms

    def update_from_output(self, scheduler_output, model_output):
        return {0: scheduler_output}


class _Core:
    def __init__(self, executor, scheduler):
        self.model_executor = executor
        self.scheduler = scheduler

    @contextlib.contextmanager
    def capture_iteration_details(self, scheduler_output):
        yield None

    @contextlib.contextmanager
    def log_error_detail(self, scheduler_output):
        yield

    def _should_throttle_prefills(self):
        return False

    def _process_aborts_queue(self):
        pass

    def _attach_iteration_details(self, outputs, iteration_details):
        pass


def _stepper(monkeypatch, *steps, estimate_ms=125.0):
    executor = _executor(monkeypatch)
    scheduler = _Scheduler(*steps, estimate_ms=estimate_ms)
    stepper = pp_lazy_step.LazyPushStepper(_Core(executor, scheduler))
    stepper.wave.lazy = True
    return stepper, executor, scheduler


def test_only_steps_that_change_worker_state_carry_it():
    assert not pp_lazy_step._carries_state(_output(0))
    assert not pp_lazy_step._carries_state(_output(0, num_common_prefix_blocks=[0, 0]))
    cached = CachedRequestData.make_empty()
    cached.req_ids = ["r"]
    assert pp_lazy_step._carries_state(_output(0, scheduled_cached_reqs=cached))
    assert pp_lazy_step._carries_state(_output(0, finished_req_ids={"r"}))
    assert pp_lazy_step._carries_state(_output(0, preempted_req_ids={"r"}))
    assert pp_lazy_step._carries_state(_output(0, free_encoder_mm_hashes=["h"]))


def test_empty_steps_are_dropped_unless_something_needs_them(monkeypatch, clock):
    finished = _output(0, finished_req_ids={"r"})
    stepper, executor, scheduler = _stepper(monkeypatch, _output(0), finished)
    assert stepper.step() == (None, False)
    assert executor.calls == [] and stepper.dropped == 1
    # finished requests reach the workers in a step without tokens
    assert stepper.step() == (None, False)
    assert executor.calls == ["execute_model"]
    assert len(stepper.in_flight) == 1 and not stepper.wave.open
    # a request whose grammar failed to compile finishes when that step's
    # output is processed; with nothing in flight it needs a step of its own
    scheduler.grammar_compile_error_reqs.add("g")
    stepper.step()
    assert executor.calls == ["execute_model"] and stepper.dropped == 2
    executor.socket.arrived = 1
    assert stepper.step() == ({0: finished}, False)
    assert executor.calls == ["execute_model"] * 2
    assert stepper.in_flight[0].slot is None
    scheduler.idle = True
    stepper.step()
    assert executor.calls == ["execute_model"] * 2
    executor.socket.arrived = 1
    assert stepper.step()[0] is not None and not stepper.in_flight


def test_a_lone_step_is_pushed_when_stage_0_would_idle(monkeypatch, clock):
    step = _output(8192)
    stepper, executor, scheduler = _stepper(monkeypatch, step)
    assert stepper.step() == (None, True)
    assert executor.calls == ["execute_model", "sample_tokens"]
    # a 125 ms forward: hand-off 1 at 64.125, after which stage 1 runs the
    # step for 125 ms; stage 0 waits half of that for another request
    clock.now = 64.1874
    assert stepper.step() == (None, True)
    assert executor.calls == ["execute_model", "sample_tokens"]
    # nothing to do: the loop sleeps on the last stage's reply ring
    assert executor.socket.polls[-1] == 1
    clock.now = 64.1875
    stepper.step()
    assert executor.calls[-1] == "pp_push"
    # the push carries nothing: hand-off 2 at 64.25, the step on stage 2
    clock.now = 64.3124
    stepper.step()
    assert executor.calls.count("pp_push") == 1
    clock.now = 64.3125
    stepper.step()
    assert executor.calls.count("pp_push") == 2
    # two later steps carry the step to the last stage: no third push
    clock.now = 64.5
    stepper.step()
    assert executor.calls.count("pp_push") == 2
    # the last stage starts the step, then samples 156.25 ms later
    executor.socket.arrived = 1
    stepper.step()
    assert stepper.slots[0].started_last == 64.5
    clock.now = 64.65625
    executor.socket.arrived = 1
    assert stepper.step() == ({0: step}, False)
    assert stepper.ratio == 1.0625
    # nothing with tokens is in flight any more: the burst settles
    assert executor.calls[-1] == "pp_settle" and not stepper.wave.open


def test_a_measurement_counts_within_a_band_of_the_estimate(monkeypatch, clock):
    stepper, executor, scheduler = _stepper(monkeypatch, _output(8192))
    stepper.step()
    clock.now = 64.5
    executor.socket.arrived = 1
    stepper.step()
    # a 6 s forward against a 125 ms estimate counts as a 250 ms one
    clock.now = 70.5
    executor.socket.arrived = 1
    stepper.step()
    assert stepper.ratio == 1.25


def test_a_long_burst_keeps_only_the_newest_slots(monkeypatch, clock):
    steps = [_output(4096) for _ in range(6)]
    stepper, executor, scheduler = _stepper(monkeypatch, *steps)
    for _ in range(4):
        stepper.step()
    # the first two steps finish and two more go out
    for _ in range(2):
        executor.socket.arrived = 2
        stepper.step()
    assert stepper.last_real == 5 and len(stepper.in_flight) == 4
    assert stepper.first_slot == 2 and len(stepper.slots) == 4
    assert "pp_push" not in executor.calls
    # the last stage starts the third and fourth steps: hand-off 6 is known
    # and slot 6 is pushed after half of the other stages' forward
    clock.now = 65.0
    executor.socket.arrived = 3
    stepper.step()
    assert "pp_push" not in executor.calls
    clock.now = 65.0625
    stepper.step()
    assert executor.calls.count("pp_push") == 1
    assert stepper.first_slot == 3 and len(stepper.slots) == 4
    # slot 7 waits for the fifth step's reply
    clock.now = 70.0
    stepper.step()
    assert executor.calls.count("pp_push") == 1


def test_a_request_arriving_before_the_push_takes_its_place(monkeypatch, clock):
    stepper, executor, scheduler = _stepper(monkeypatch, _output(8192))
    stepper.step()
    clock.now = 64.15625
    scheduler.steps.append(_output(8192))
    stepper.step()
    assert executor.calls == ["execute_model", "sample_tokens"] * 2
    # hand-off 2 once stage 0 ran the second step, at 64.28125
    clock.now = 64.3437
    stepper.step()
    assert "pp_push" not in executor.calls
    clock.now = 64.34375
    stepper.step()
    assert executor.calls.count("pp_push") == 1
    # hand-off 3 is read from the last stage's reply for the first step
    clock.now = 70.0
    stepper.step()
    assert executor.calls.count("pp_push") == 1
    executor.socket.arrived = 1
    stepper.step()
    assert executor.calls.count("pp_push") == 1
    clock.now = 70.0625
    stepper.step()
    assert executor.calls.count("pp_push") == 2
    # the second step has two later steps now
    clock.now = 80.0
    stepper.step()
    assert executor.calls.count("pp_push") == 2


def test_at_most_stages_steps_with_tokens_in_flight(monkeypatch, clock):
    steps = [_output(4096) for _ in range(6)]
    stepper, executor, scheduler = _stepper(monkeypatch, *steps)
    for _ in range(6):
        stepper.step()
    assert scheduler.schedules == 4 and len(stepper.in_flight) == 4
    # the oldest finishes and the next one goes out
    executor.socket.arrived = 2
    assert stepper.step() == ({0: steps[0]}, True)
    assert scheduler.schedules == 5 and len(stepper.in_flight) == 4
    assert "pp_push" not in executor.calls


def test_in_flight_steps_stay_within_the_message_rings(monkeypatch, clock):
    steps = [_output(0, finished_req_ids={str(i)}) for i in range(10)]
    stepper, executor, scheduler = _stepper(monkeypatch, *steps)
    for _ in range(10):
        stepper.step()
    assert scheduler.schedules == 8 and len(stepper.in_flight) == 8


def test_other_calls_settle_the_burst_first(monkeypatch, clock):
    stepper, executor, scheduler = _stepper(monkeypatch, _output(4096))
    stepper.step()
    executor.collective_rpc("profile", non_block=True)
    assert executor.calls == ["execute_model", "sample_tokens", "pp_settle", "profile"]
    assert not stepper.wave.open
    # step calls are answered by the last stage alone
    assert [
        getattr(f, "_tpu_last_stage_reply", False)
        for f in reversed(executor.futures_queue)
    ] == [True, True, True, False]
    executor.collective_rpc("profile", non_block=True)
    assert executor.calls[-2:] == ["profile", "profile"]
    # every reply read in call order, the step processed
    executor.socket.arrived = 5
    scheduler.idle = True
    assert stepper.step()[0] is not None
    assert not executor.futures_queue and not stepper.in_flight


def test_a_blocking_call_settles_and_reads_every_step(monkeypatch, clock):
    stepper, executor, scheduler = _stepper(monkeypatch, _output(4096), _output(4096))
    stepper.step()
    stepper.step()
    executor.socket.arrived = 6
    assert executor.collective_rpc("profile") == ("profile", None)
    assert executor.calls[-2:] == ["pp_settle", "profile"]
    assert not executor.futures_queue and not stepper.wave.open
    # the steps are processed in order; nothing is pushed into the settled
    # burst, and replies read together measure nothing
    assert stepper.step()[0] is not None
    assert stepper.step()[0] is not None
    assert not stepper.in_flight and "pp_push" not in executor.calls
    assert stepper.ratio == 1.0


def test_executors_without_lazy_pushes_call_through(monkeypatch):
    executor = _executor(monkeypatch)
    pp_push.wave_state(executor).open = True
    future = executor.collective_rpc("profile", non_block=True)
    assert executor.calls == ["profile"]
    assert not getattr(future, "_tpu_last_stage_reply", False)


def test_a_step_past_the_timeout_raises(monkeypatch, clock):
    stepper, executor, scheduler = _stepper(monkeypatch, _output(4096))
    stepper.timeout = 10.0
    stepper.step()
    clock.now = 64.1875
    stepper.step()
    assert executor.calls[-1] == "pp_push"
    clock.now = 74.5
    with pytest.raises(TimeoutError, match="sample_tokens"):
        stepper.step()
    # the step is not waited for, nor the push ahead of it
    assert "pp_settle" not in executor.calls


def test_a_forward_that_failed_raises(monkeypatch, clock):
    stepper, executor, scheduler = _stepper(monkeypatch, _output(4096))
    executor.replies["sample_tokens"] = None
    stepper.step()
    executor.socket.arrived = 2
    with pytest.raises(RuntimeError, match="unexpected error"):
        stepper.step()


def test_a_new_burst_counts_its_steps_from_zero(monkeypatch, clock):
    first = _output(4096)
    stepper, executor, scheduler = _stepper(monkeypatch, first)
    stepper.step()
    executor.socket.arrived = 2
    assert stepper.step()[0] == {0: first}
    assert executor.calls[-1] == "pp_settle"
    # replies read together measure nothing
    assert stepper.ratio == 1.0
    clock.now = 65.0
    scheduler.steps.append(_output(4096))
    stepper.step()
    assert len(stepper.slots) == 1 and stepper.last_real == 0
    assert stepper.slots[0].dispatched == 65.0


def test_without_the_cost_model_the_measured_forward_is_expected(monkeypatch, clock):
    stepper, executor, scheduler = _stepper(
        monkeypatch, _output(4096), estimate_ms=None
    )
    # nothing measured yet: the push goes out at once
    stepper.step()
    assert executor.calls == ["execute_model", "sample_tokens", "pp_push"]
    clock.now = 64.5
    executor.socket.arrived = 1
    stepper.step()
    clock.now = 64.75
    executor.socket.arrived = 1
    stepper.step()
    assert stepper.measured == 0.25
    executor.socket.arrived = 2
    stepper.step()
    scheduler.steps.append(_output(4096))
    stepper.step()
    assert stepper.slots[-1].expected == 0.25
    # a 10 s forward counts as twice the 250 ms measured so far
    clock.now = 65.0
    executor.socket.arrived = 2
    stepper.step()
    clock.now = 75.0
    executor.socket.arrived = 1
    stepper.step()
    assert stepper.measured == 0.3125


def _engine_core(monkeypatch, executor):
    from vllm.v1.engine import core as engine_core

    class FakeCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.model_executor = executor
            self.scheduler = _Scheduler()
            self.is_pooling_model = False
            self.async_scheduling = False
            self.batch_queue = deque(maxlen=4)
            self.step_fn = self.step_with_batch_queue

        def step_with_batch_queue(self):
            return None, False

    monkeypatch.setattr(engine_core, "EngineCore", FakeCore)
    return FakeCore


def test_the_engine_core_runs_the_lazy_loop(monkeypatch):
    monkeypatch.setattr(pp_lazy_step.envs, "TPU_PP_LAZY_PUSH", True)
    executor = _executor(monkeypatch)
    core_cls = _engine_core(monkeypatch, executor)
    pp_lazy_step.patch_engine_core_for_lazy_push(executor.vllm_config)
    executor.socket.arrived = 1
    core = core_cls(executor.vllm_config)
    assert executor.calls == ["pp_drains_before_handoff"]
    stepper = core.step_fn.__self__
    assert isinstance(stepper, pp_lazy_step.LazyPushStepper)
    assert core.batch_queue is stepper.in_flight
    assert pp_push.wave_state(executor).lazy
    # a second install is a no-op
    init = core_cls.__init__
    pp_lazy_step.patch_engine_core_for_lazy_push(executor.vllm_config)
    assert core_cls.__init__ is init


def test_the_engine_core_keeps_vllms_loop_with_a_local_last_stage(monkeypatch):
    monkeypatch.setattr(pp_lazy_step.envs, "TPU_PP_LAZY_PUSH", True)
    executor = _executor(monkeypatch)
    executor.response_mqs[executor.output_rank]._is_remote_reader = False
    core_cls = _engine_core(monkeypatch, executor)
    pp_lazy_step.patch_engine_core_for_lazy_push(executor.vllm_config)
    core = core_cls(executor.vllm_config)
    assert core.step_fn == core.step_with_batch_queue
    assert not pp_push.wave_state(executor).lazy


def test_the_patch_needs_the_flag_and_more_than_two_stages(monkeypatch):
    from vllm.v1.engine import core as engine_core

    class FakeCore:
        pass

    monkeypatch.setattr(engine_core, "EngineCore", FakeCore)
    monkeypatch.setattr(pp_lazy_step.envs, "TPU_PP_LAZY_PUSH", False)
    pp_lazy_step.patch_engine_core_for_lazy_push(_config(4))
    monkeypatch.setattr(pp_lazy_step.envs, "TPU_PP_LAZY_PUSH", True)
    pp_lazy_step.patch_engine_core_for_lazy_push(_config(2))
    pp_lazy_step.patch_engine_core_for_lazy_push(None)
    assert not hasattr(FakeCore, "_tpu_pp_lazy_patch")


def test_lazy_push_is_off_in_setups_it_does_not_cover(monkeypatch):
    executor = _executor(monkeypatch)
    core = SimpleNamespace(
        model_executor=executor,
        is_pooling_model=False,
        async_scheduling=False,
        batch_queue=deque(),
    )
    blocker = pp_lazy_step._lazy_push_blocker

    def core_with(**changes):
        return SimpleNamespace(**{**vars(core), **changes})

    def config_with(**changes):
        config = _config()
        for name, value in changes.items():
            setattr(config, name, value)
        return config

    executor.socket.arrived = 1
    assert blocker(core, _config()) is None
    assert executor.calls == ["pp_drains_before_handoff"]
    assert blocker(core, _config(dp=2)) == "data parallelism"
    config = _config()
    config.parallel_config.enable_fault_tolerance = True
    assert blocker(core, config) == "fault tolerance"
    assert blocker(core_with(is_pooling_model=True), _config()) == "a pooling model"
    assert (
        blocker(core, config_with(speculative_config=object()))
        == "speculative decoding"
    )
    for name in ("kv_transfer_config", "ec_transfer_config"):
        assert (
            blocker(core, config_with(**{name: object()}))
            == "a KV or encoder cache connector"
        )
    for changes in ({"async_scheduling": True}, {"batch_queue": None}):
        assert (
            blocker(core_with(**changes), _config())
            == "async scheduling or no batch queue"
        )
    assert (
        blocker(core_with(model_executor=SimpleNamespace()), _config())
        == "an executor without the pipeline push"
    )
    executor.kv_output_aggregator = object()
    assert blocker(core, _config()) == "an output aggregator"
    executor.kv_output_aggregator = None
    executor.response_mqs[executor.output_rank]._is_remote_reader = False
    assert blocker(core, _config()) == "the last stage on the engine's host"
    executor.response_mqs[executor.output_rank]._is_remote_reader = True
    executor.replies["pp_drains_before_handoff"] = [True, False, True, True]
    executor.socket.arrived = 1
    assert blocker(core, _config()) == (
        "stages that hand off without draining the slice"
    )
