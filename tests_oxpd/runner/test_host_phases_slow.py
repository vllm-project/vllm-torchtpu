# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-HOST-B: the HOSTSLOW slow-step dump (VLLM_TPU_HOST_PHASES_SLOW_MS).

A step's record is the idle charged at its begin_step plus its own marks;
sub-phases ("~") split a phase and are not added to the total. The aggregate
HOSTPHASES samples must be the same with the dump on or off.
"""

from types import SimpleNamespace

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.runner import host_phases
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


def _reset(monkeypatch, every, slow_ms, step_us):
    monkeypatch.setattr(host_phases, "_EVERY", every)
    monkeypatch.setattr(host_phases, "_SLOW_NS", int(slow_ms * 1e6))
    monkeypatch.setattr(host_phases, "_last", None)
    monkeypatch.setattr(host_phases, "_steps", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    monkeypatch.setattr(host_phases, "_cur", {})
    monkeypatch.setattr(host_phases, "_ctx", [])
    monkeypatch.setattr(host_phases, "_sub_last", None)
    monkeypatch.setattr(host_phases, "_cur_wall", 0.0)
    monkeypatch.setattr(host_phases, "_report", lambda: None)
    log = _Log()
    monkeypatch.setattr(host_phases, "logger", log)
    clock = iter(range(0, 10**12, step_us * 1000))
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", lambda: next(clock))
    return log


def _three_steps(slow_step=1):
    """Each clock read advances one tick. Step `slow_step` notes context and
    calls sub() twice inside phase "a"."""
    for i in range(3):
        host_phases.begin_step()
        if i == slow_step:
            host_phases.note("admit:prompt=300000")
            host_phases.sub("s1")
            host_phases.sub("s2")
            host_phases.charge("add_request", 7_000)
        host_phases.mark("a")
        host_phases.mark("b")
    host_phases.begin_step()


def test_slow_step_is_logged_with_idle_phases_subs_and_context(monkeypatch):
    # 1 ms per clock read; threshold 3.5 ms. Plain steps read 3 ticks (idle,
    # a, b) = 3 ms; the noted step reads 5 ticks (two subs) = 5 ms.
    log = _reset(monkeypatch, every=1000, slow_ms=3.5, step_us=1000)
    _three_steps(slow_step=1)
    assert len(log.lines) == 1
    line = log.lines[0]
    assert line.startswith("HOSTSLOW rank=")
    assert " t_begin=" in line and "total_us=5000" in line
    assert "ctx=[admit:prompt=300000]" in line
    # The step's record opens with the idle before it; "a" holds the two subs.
    assert "idle=1000 ~s1=1000 ~s2=1000 ~add_request=7 a=3000 b=1000" in line


def test_fast_steps_are_not_logged(monkeypatch):
    log = _reset(monkeypatch, every=1000, slow_ms=6.0, step_us=1000)
    _three_steps(slow_step=1)
    assert log.lines == []


def test_aggregate_samples_are_unchanged_by_the_dump(monkeypatch):
    # Same clock and calls with the dump off and on: the aggregate samples
    # must match (sub/charge/note never touch them).
    _reset(monkeypatch, every=1000, slow_ms=0.0, step_us=1000)
    for i in range(3):
        host_phases.begin_step()
        host_phases.mark("a")
        host_phases.mark("b")
    host_phases.begin_step()
    off = {k: list(v) for k, v in host_phases._samples.items()}
    _reset(monkeypatch, every=1000, slow_ms=0.5, step_us=1000)
    for i in range(3):
        host_phases.begin_step()
        host_phases.mark("a")
        host_phases.mark("b")
    host_phases.begin_step()
    on = {k: list(v) for k, v in host_phases._samples.items()}
    assert off == on == {"a": [1000000] * 3, "b": [1000000] * 3, "idle": [1000000] * 3}


def test_off_is_a_no_op(monkeypatch):
    log = _reset(monkeypatch, every=1000, slow_ms=0.0, step_us=1000)
    _three_steps(slow_step=1)
    assert log.lines == []
    assert host_phases._cur == {} and host_phases._ctx == []
    assert not host_phases.slow_enabled()


def test_slow_needs_host_phases_on(monkeypatch):
    # With HOST_PHASES=0 nothing is recorded even if the threshold is set.
    log = _reset(monkeypatch, every=0, slow_ms=0.001, step_us=1000)
    monkeypatch.setattr(host_phases, "_SLOW_NS", 0)  # as computed at import
    _three_steps(slow_step=1)
    assert log.lines == [] and host_phases._samples == {}


def test_flag_is_compile_neutral():
    assert "VLLM_TPU_HOST_PHASES_SLOW_MS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_note_admissions_records_context_and_times_add_request(monkeypatch):
    _reset(monkeypatch, every=1000, slow_ms=0.5, step_us=1)
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", lambda: 0)
    calls = []
    batch = SimpleNamespace(num_reqs=9, add_request=lambda req: calls.append(req) or 3)
    runner = SimpleNamespace(input_batch=batch)
    ticks = iter([1_000, 51_000, 60_000, 90_000])
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.time.perf_counter_ns", lambda: next(ticks)
    )
    new = [
        SimpleNamespace(
            prompt_token_ids=[0] * 300_000, num_computed_tokens=299_000,
            block_ids=([1, 2, 3], [4]),
        ),
        SimpleNamespace(prompt_token_ids=None, num_computed_tokens=0, block_ids=([],)),
    ]
    so = SimpleNamespace(
        scheduled_new_reqs=new,
        total_num_scheduled_tokens=40,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"x"}),
    )
    TPUModelRunner._note_admissions(runner, so)
    assert host_phases._ctx == [
        "reqs=9 sched=40 new=2 resumed=1",
        "admit:prompt=300000:computed=299000:blocks=4:tail=-",
        "admit:prompt=-1:computed=0:blocks=0:tail=-",
    ]
    # The wrapper is installed once and charges add_request's host time.
    first = batch.add_request
    TPUModelRunner._note_admissions(runner, so)
    assert batch.add_request is first
    assert batch.add_request("r1") == 3 and batch.add_request("r2") == 3
    assert calls == ["r1", "r2"]
    assert host_phases._cur["~add_request"] == 50_000 + 30_000


def test_threshold_is_read_at_import_and_needs_host_phases(monkeypatch):
    import importlib

    try:
        monkeypatch.setenv("VLLM_TPU_HOST_PHASES_SLOW_MS", "50")
        monkeypatch.setenv("VLLM_TPU_HOST_PHASES", "0")
        importlib.reload(host_phases)
        assert host_phases._SLOW_NS == 0
        monkeypatch.setenv("VLLM_TPU_HOST_PHASES", "200")
        importlib.reload(host_phases)
        assert host_phases._SLOW_NS == 50_000_000 and host_phases.slow_enabled()
    finally:
        monkeypatch.delenv("VLLM_TPU_HOST_PHASES_SLOW_MS", raising=False)
        monkeypatch.delenv("VLLM_TPU_HOST_PHASES", raising=False)
        importlib.reload(host_phases)
    assert host_phases._SLOW_NS == 0 and host_phases._EVERY == 0


def test_subs_and_notes_never_enter_the_aggregate(monkeypatch):
    _reset(monkeypatch, every=1000, slow_ms=0.5, step_us=1000)
    _three_steps(slow_step=1)
    assert sorted(host_phases._samples) == ["a", "b", "idle"]
    assert [len(v) for _, v in sorted(host_phases._samples.items())] == [3, 3, 3]
