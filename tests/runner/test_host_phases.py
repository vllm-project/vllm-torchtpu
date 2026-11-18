# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu.runner import host_phases


def test_marks_charge_time_to_phases_and_report_every_n_steps(monkeypatch):
    monkeypatch.setattr(host_phases, "_EVERY", 2)
    monkeypatch.setattr(host_phases, "_last", None)
    monkeypatch.setattr(host_phases, "_steps", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    reports = []
    monkeypatch.setattr(
        host_phases,
        "_report",
        lambda: reports.append({k: list(v) for k, v in host_phases._samples.items()}),
    )
    clock = iter(range(0, 10**9, 1000))
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", lambda: next(clock))
    for _ in range(2):
        host_phases.begin_step()
        host_phases.mark("a")
        host_phases.mark("b")
    # The second step's begin charges the gap since the first step's last mark
    # to idle, then reports the window it closes.
    assert reports == [{"a": [1000], "b": [1000], "idle": [1000]}]


def test_off_by_default_is_a_no_op(monkeypatch):
    monkeypatch.setattr(host_phases, "_EVERY", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    host_phases.begin_step()
    host_phases.mark("a")
    assert host_phases._samples == {}
