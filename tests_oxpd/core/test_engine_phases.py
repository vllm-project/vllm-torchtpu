# SPDX-License-Identifier: Apache-2.0
"""q38 D1k: the engine core's per-step phase timers (VLLM_TPU_ENGINE_PHASES).

A fake engine core advances a fake clock by a known amount in each method, so
every phase's per-step time is known exactly and the timers' attribution
(nested phases, the ``wait`` and ``other`` remainders, the slow-step dump)
can be checked without a TPU or a real scheduler.
"""

import re
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core import engine_phases

US = 1000  # ns per microsecond


class Clock:
    def __init__(self):
        self.t = 1_000_000_000

    def __call__(self):
        return self.t

    def adv(self, us):
        self.t += us * US


class FakeCore:
    """Costs in microseconds per call. step_fn mirrors step_with_batch_queue:
    schedule -> execute_model -> sample_tokens -> wait -> update_from_output."""

    def __init__(self, clock, slow_wait_steps=()):
        c = clock
        self.clock = c
        self.slow_wait_steps = set(slow_wait_steps)
        self.n = 0
        self.new_next = []
        self.input_extra = []
        kvm = SimpleNamespace(
            allocate_slots=lambda *a: (c.adv(1000), "blocks")[1],
            free=lambda *a: c.adv(200),
        )
        conn = SimpleNamespace(
            build_connector_meta=lambda *a: (c.adv(500), "meta")[1],
            update_connector_output=lambda *a: c.adv(300),
        )

        def schedule():
            c.adv(1500)
            self.scheduler.kv_cache_manager.allocate_slots()
            self.scheduler.connector.build_connector_meta()
            new, self.new_next = self.new_next, []
            return SimpleNamespace(scheduled_new_reqs=new, finished_req_ids={"x"})

        def update_from_output(so, out):
            c.adv(1000)
            self.scheduler.kv_cache_manager.free()
            self.scheduler.connector.update_connector_output()
            return {"outputs": out}

        def add_request(req):
            c.adv(400)

        self.scheduler = SimpleNamespace(
            schedule=schedule,
            update_from_output=update_from_output,
            add_request=add_request,
            kv_cache_manager=kvm,
            connector=conn,
            running=[1, 2, 3],
            waiting=[4],
        )
        self.model_executor = SimpleNamespace(
            execute_model=lambda so, non_block=True: (c.adv(2000), "fut")[1],
            sample_tokens=lambda g, non_block=True: (c.adv(500), "fut2")[1],
        )
        self.step_fn = self._step

    def _step(self):
        so = self.scheduler.schedule()
        self.model_executor.execute_model(so)
        self.model_executor.sample_tokens(None)
        self.n += 1
        self.clock.adv(60000 if self.n in self.slow_wait_steps else 4000)  # wait
        return self.scheduler.update_from_output(so, "out"), True

    def preprocess_add_request(self, req):
        self.clock.adv(2500)  # request creation + block hashing (input thread)
        return req, 0

    def _process_input_queue(self):
        self.clock.adv(300 + self.input_extra.pop(0) if self.input_extra else 300)
        self.scheduler.add_request("r")

    def loop_once(self):
        self._process_input_queue()
        out = self.step_fn()
        self.clock.adv(600)  # output queue, post_step
        return out


class Capture:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)

    def exception(self, msg, *args):
        self.lines.append("EXC " + msg % args)


@pytest.fixture
def env(monkeypatch):
    clock = Clock()
    cap = Capture()
    monkeypatch.setattr(engine_phases.time, "perf_counter_ns", clock)
    monkeypatch.setattr(engine_phases, "logger", cap)
    return clock, cap


def _phase(line, name):
    m = re.search(r" %s=(\d+)/(\d+)/(\d+)" % name, line)
    return tuple(int(x) for x in m.groups()) if m else None


def test_off_wraps_nothing(env):
    clock, _ = env
    core = FakeCore(clock)
    before = (core.step_fn, core.scheduler.schedule, core.model_executor.execute_model)
    assert engine_phases.install(core, 0) is None
    assert (core.step_fn, core.scheduler.schedule, core.model_executor.execute_model) == before


def test_phase_attribution_and_report(env):
    clock, cap = env
    core = FakeCore(clock)
    engine_phases.install(core, 4)
    outs = [core.loop_once() for _ in range(4)]
    assert outs[0] == ({"outputs": "out"}, True)  # return values pass through
    rep = [ln for ln in cap.lines if ln.startswith("ENGINEPHASES")]
    assert len(rep) == 1
    line = rep[0]
    # step_fn = schedule 3000 (1500 + alloc 1000 + meta 500) + exec 2000 + sample 500
    #          + wait 4000 + update 1500 (1000 + free 200 + conn_update 300)
    assert _phase(line, "schedule") == (3000, 3000, 3000)
    assert _phase(line, "alloc") == (1000, 1000, 1000)
    assert _phase(line, "conn_meta") == (500, 500, 500)
    assert _phase(line, "exec_dispatch") == (2000, 2000, 2000)
    assert _phase(line, "sample_dispatch") == (500, 500, 500)
    assert _phase(line, "update") == (1500, 1500, 1500)
    assert _phase(line, "free") == (200, 200, 200)
    assert _phase(line, "conn_update") == (300, 300, 300)
    assert _phase(line, "wait") == (4000, 4000, 4000)
    assert _phase(line, "input") == (700, 700, 700)
    assert _phase(line, "add_request") == (400, 400, 400)
    # loop period = input 700 + step 11000 + other 600 = 12300 (3 periods in 4 steps)
    assert re.search(r"loop=12300/12300/12300 ", line)
    # "other" is attributed to the step that follows it: 0 for the first step
    assert re.search(r" other=600/600/450 ", line)
    assert "running=3 waiting=1 calls add_request=4 alloc=4 free=4 preproc=0 slow=0" in line
    assert not [ln for ln in cap.lines if ln.startswith("ENGINESLOW")]


def test_other_excludes_the_input_of_its_own_period(env):
    # The input queue processed between two steps varies; "other" must stay the
    # 600 us the loop spends outside step_fn and the input queue.
    clock, cap = env
    core = FakeCore(clock)
    core.input_extra = [0, 600, 0, 1200]
    engine_phases.install(core, 4)
    for _ in range(4):
        core.loop_once()
    line = [ln for ln in cap.lines if ln.startswith("ENGINEPHASES")][0]
    assert re.search(r" other=600/600/450 ", line)
    assert _phase(line, "input") == (1000, 1720, 1150)


def test_slow_step_dump_names_the_admission(env):
    clock, cap = env
    core = FakeCore(clock, slow_wait_steps={3})
    engine_phases.install(core, 100)
    for i in range(5):
        if i == 2:
            core.new_next = [SimpleNamespace(prompt_token_ids=list(range(5000)))]
        core.loop_once()
    slow = [ln for ln in cap.lines if ln.startswith("ENGINESLOW")]
    # step 3 waits 60 ms; the period that includes it is measured at step 4's start
    assert len(slow) == 1
    assert slow[0].startswith("ENGINESLOW step=4 period_us=68300 running=3 ")
    head, prev = slow[0].split(" | prev ")
    assert "new=0 new_tokens=0" in head
    assert "wait=60000" in prev and "new=1 new_tokens=5000 finished=1" in prev


def test_input_thread_preprocessing_is_its_own_phase(env):
    clock, cap = env
    core = FakeCore(clock)
    engine_phases.install(core, 2)
    assert core.preprocess_add_request("r") == ("r", 0)  # the input thread's call
    core.loop_once()
    core.loop_once()
    line = [ln for ln in cap.lines if ln.startswith("ENGINEPHASES")][0]
    assert _phase(line, "preproc") == (1250, 2250, 1250)
    assert " preproc=1 slow=0" in line


def test_exceptions_propagate_and_still_count(env):
    clock, cap = env
    core = FakeCore(clock)

    def boom():
        clock.adv(250)
        raise ValueError("schedule failed")

    core.scheduler.schedule = boom
    engine_phases.install(core, 1)
    with pytest.raises(ValueError, match="schedule failed"):
        core.step_fn()
    line = [ln for ln in cap.lines if ln.startswith("ENGINEPHASES")][0]
    assert _phase(line, "schedule") == (250, 250, 250)


def test_compile_neutral():
    from vllm_torchtpu.compilation import tpu_compiler

    assert "VLLM_TPU_ENGINE_PHASES" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
