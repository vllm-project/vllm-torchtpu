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
