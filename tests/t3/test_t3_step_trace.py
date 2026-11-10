# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_T3_STEP_TRACE (R-HOST measurement): per-step wall-clock events, captured in windows aligned to
wall-clock time, one log line per window. Default off; needs VLLM_TPU_T3_TIMERS=1."""
from __future__ import annotations

import logging
import re
import sys
import time
import types

import pytest

from vllm_torchtpu.runner import t3_host

LINE = re.compile(r"\[t3trace\](.*?) win=(\S+) base=([\d.]+) n=(\d+) ev=(\S*)")


class _FakeRunner:
    def execute_model(self, x):
        return x + 1

    def forward_model(self, x):
        return x

    def sample_tokens(self, g):
        return g


def _engine(monkeypatch):
    class Sched:
        def schedule(self, *a):
            return "so"

        def update_from_output(self, so, mo):
            return {"out": 1}

    class Exec:
        def execute_model(self, so, non_block=True):
            return "f"

        def sample_tokens(self, g, non_block=True):
            return "f2"

    class EngineCore:
        def __init__(self):
            self.scheduler = Sched()
            self.model_executor = Exec()

        def step_with_batch_queue(self):
            so = self.scheduler.schedule()
            self.model_executor.execute_model(so)
            self.model_executor.sample_tokens(None)
            return self.scheduler.update_from_output(so, None), True

    fake = types.ModuleType("vllm.v1.engine.core")
    fake.EngineCore = EngineCore
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", fake)
    import vllm.v1.engine as _eng

    monkeypatch.setattr(_eng, "core", fake, raising=False)
    return EngineCore


def _lines(caplog):
    return [m for r in caplog.records if (m := LINE.search(r.getMessage()))]


def _parse(m):
    return [(c, int(t)) for c, t in (x.split(":") for x in m.group(5).split(","))]


@pytest.fixture
def trace_on(monkeypatch):
    monkeypatch.setattr(t3_host, "TIMERS", True)
    monkeypatch.setattr(t3_host, "STEP_TRACE", True)
    monkeypatch.setattr(t3_host, "_aggs", {})
    monkeypatch.setattr(t3_host, "_trace_state", {"win": None, "ev": [], "off": 0.0})
    # every second is a capture window of its own (window 1 s of every 1 s), so a window change can be
    # forced by moving the clock to the next second
    monkeypatch.setattr(t3_host, "_TRACE_EVERY_S", 1.0)
    monkeypatch.setattr(t3_host, "_TRACE_WINDOW_S", 1.0)
    yield


def test_step_trace_off_by_default_logs_nothing(monkeypatch, caplog):
    monkeypatch.setattr(t3_host, "TIMERS", True)
    monkeypatch.setattr(t3_host, "STEP_TRACE", False)
    monkeypatch.setattr(t3_host, "_aggs", {})
    monkeypatch.setattr(t3_host, "_trace_state", {"win": None, "ev": [], "off": 0.0})
    cls = type("R", (_FakeRunner,), {})
    t3_host.install_runner_timers(cls)
    r = cls()
    with caplog.at_level(logging.INFO):
        for i in range(5):
            r.execute_model(i)
            r.sample_tokens(i)
        t3_host.trace("E", time.perf_counter())
    assert not _lines(caplog)
    assert t3_host._trace_state["ev"] == []


def test_worker_and_engine_events_in_one_window(trace_on, monkeypatch, caplog):
    now = [1000.25]
    real_time = time.time
    monkeypatch.setattr(t3_host._time, "time", lambda: now[0])
    cls = type("R", (_FakeRunner,), {})
    t3_host.install_runner_timers(cls)
    EngineCore = _engine(monkeypatch)
    t3_host.install_engine_core_timers()
    ec = EngineCore()
    r = cls()
    with caplog.at_level(logging.INFO):
        for i in range(3):
            assert ec.step_with_batch_queue() == ({"out": 1}, True)
            assert r.execute_model(i) == i + 1  # results unchanged
            assert r.sample_tokens(i) == i
        now[0] = 1001.25  # next window: the first one is emitted on the next event
        r.execute_model(0)
    ls = _lines(caplog)
    assert len(ls) == 1
    m = ls[0]
    assert m.group(2) == "1000"
    ev = _parse(m)
    assert int(m.group(4)) == len(ev) == 3 * (10 + 4)
    codes = [c for c, _ in sorted(ev, key=lambda x: x[1])]
    # engine call order inside a step: Q S s X x Y y U u q; every start precedes its end
    one = "".join(c for c in codes if c in "QSsXxYyUuq")
    assert one == "QSsXxYyUuq" * 3
    w = "".join(c for c in codes if c in "EePp")
    assert w == "EePp" * 3
    assert min(t for _, t in ev) == 0
    # base is wall clock (offset applied), not perf_counter
    assert abs(float(m.group(3)) - 1000.25) < 5.0
    del real_time


def test_events_outside_capture_window_are_dropped(trace_on, monkeypatch, caplog):
    monkeypatch.setattr(t3_host, "_TRACE_EVERY_S", 60.0)
    monkeypatch.setattr(t3_host, "_TRACE_WINDOW_S", 2.0)
    now = [1210.5]  # 1210.5 % 60 = 10.5 -> outside the first 2 s
    monkeypatch.setattr(t3_host._time, "time", lambda: now[0])
    with caplog.at_level(logging.INFO):
        t3_host.trace("E", time.perf_counter())
        assert t3_host._trace_state["ev"] == []
        now[0] = 1260.5  # inside window 21
        t3_host.trace("E", time.perf_counter())
        t3_host.trace("e", time.perf_counter())
        now[0] = 1263.0  # window closed -> emitted
        t3_host.trace("P", time.perf_counter())
    ls = _lines(caplog)
    assert len(ls) == 1 and ls[0].group(2) == "21" and [c for c, _ in _parse(ls[0])] == ["E", "e"]


def test_step_trace_env_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_T3_STEP_TRACE", raising=False)
    assert t3_host._step_trace_env() is False
    monkeypatch.setenv("VLLM_TPU_T3_STEP_TRACE", "0")
    assert t3_host._step_trace_env() is False
    monkeypatch.setenv("VLLM_TPU_T3_STEP_TRACE", "1")
    assert t3_host._step_trace_env() is True
