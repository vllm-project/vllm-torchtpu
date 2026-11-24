# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu.runner import host_phases


def _reset(monkeypatch, every):
    monkeypatch.setattr(host_phases, "_EVERY", every)
    monkeypatch.setattr(host_phases, "_last", None)
    monkeypatch.setattr(host_phases, "_steps", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    monkeypatch.setattr(host_phases, "_kind_steps", {})
    monkeypatch.setattr(host_phases, "_kind", "all")
    monkeypatch.setattr(host_phases, "_cur", [])
    monkeypatch.setattr(host_phases, "_cur_info", "")
    clock = iter(range(0, 10**9, 1000))
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", lambda: next(clock))


def test_marks_charge_time_to_phases_and_report_every_n_steps(monkeypatch):
    _reset(monkeypatch, 2)
    reports = []
    monkeypatch.setattr(
        host_phases,
        "_report",
        lambda: reports.append(
            {k: {n: list(v) for n, v in d.items()} for k, d in host_phases._samples.items()}
        ),
    )
    for _ in range(2):
        host_phases.begin_step()
        host_phases.mark("a")
        host_phases.mark("b")
    # The second step's begin charges the gap since the first step's last mark
    # to idle, then reports the window it closes.
    assert reports == [{"all": {"a": [1000], "b": [1000], "idle": [1000]}}]


def test_steps_are_bucketed_by_kind_and_admits_logged(monkeypatch):
    _reset(monkeypatch, 100)
    admits = []
    monkeypatch.setattr(
        host_phases, "_log_admit", lambda phases, info: admits.append((info, phases))
    )
    host_phases.begin_step("dec")
    host_phases.mark("a")
    host_phases.begin_step("admit", "new=1")
    host_phases.note("x_sched_total", 5.0)
    host_phases.mark("a")
    host_phases.begin_step("dec")
    assert host_phases._kind_steps == {"dec": 2, "admit": 1}
    # The idle before the admit step is charged to the admit step.
    assert host_phases._samples["admit"]["idle"] == [1000]
    assert host_phases._samples["admit"]["x_sched_total"] == [5000]
    assert admits == [
        ("new=1", [("idle", 1000), ("x_sched_total", 5000), ("a", 1000)])
    ]


def test_off_by_default_is_a_no_op(monkeypatch):
    monkeypatch.setattr(host_phases, "_EVERY", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    host_phases.begin_step()
    host_phases.mark("a")
    host_phases.note("x", 1.0)
    assert host_phases._samples == {}


def test_cpu_time_per_phase_is_recorded_as_x_cpu(monkeypatch):
    _reset(monkeypatch, 100)
    monkeypatch.setattr(host_phases, "_CPU", True)
    cpu = iter(range(0, 10**9, 400))
    monkeypatch.setattr(host_phases.time, "thread_time_ns", lambda: next(cpu))
    host_phases.begin_step("dec")
    host_phases.mark("a")
    # Wall 1000 ns, on-CPU 400 ns; x_ entries are not summed into total_mean.
    assert host_phases._samples["dec"] == {"a": [1000], "x_cpu_a": [400]}


def test_thread_usage_logs_deltas_since_the_previous_call(monkeypatch):
    snaps = iter(
        [
            {1: ("step", 0, 0, 0), 2: ("busy", 0, 0, 0), 3: ("idle", 0, 0, 0)},
            {1: ("step", 4_000_000, 1_000_000, 9), 2: ("busy", 8_000_000, 0, 3), 3: ("idle", 1, 0, 1)},
        ]
    )
    monkeypatch.setattr(host_phases, "_read_threads", lambda: next(snaps))
    monkeypatch.setattr(host_phases, "_thread_prev", {})
    monkeypatch.setattr(host_phases, "_thread_prev_wall", 0)
    monkeypatch.setattr(host_phases.threading, "get_native_id", lambda: 1)
    clock = iter([5_000_000, 15_000_000])
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", lambda: next(clock))
    lines = []
    monkeypatch.setattr(host_phases.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    host_phases.log_thread_usage("rank=0")
    host_phases.log_thread_usage("rank=0")
    # Busiest first; the calling thread is starred; idle threads are left out.
    assert lines == ["HOSTTHREADS rank=0 wall_ms=10 threads=3 busy/2=8/0/3 *step/1=4/1/9"]
