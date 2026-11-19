# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-HOST-4 D1-norm windowed A/B (core/d1n_ab.py): the window classes, the refusal
next to D1e's windows, uniform mode, and the per-window line with its counts."""

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.core import d1n_ab

P = 20000
MS = 1_000_000


class _Log:
    def __init__(self):
        self.info_lines, self.errors = [], []

    def info(self, fmt, *a):
        self.info_lines.append(fmt % a)

    def error(self, fmt, *a):
        self.errors.append(fmt % a)


@pytest.fixture
def ab(monkeypatch):
    log = _Log()
    clock = {"t": 0}
    monkeypatch.setattr(d1n_ab, "logger", log)
    monkeypatch.setattr(d1n_ab, "_now_ns", lambda: clock["t"])
    monkeypatch.setattr(envs, "VLLM_TPU_GDN_SEED_COPY_PACKED", True, raising=False)

    def setup(period, d1e=0):
        monkeypatch.setattr(d1n_ab, "PERIOD_MS", period)
        monkeypatch.setattr(d1n_ab, "D1E_PERIOD_MS", d1e)
        d1n_ab._resolve()
    yield setup, clock, log
    d1n_ab._resolve()


def test_window_classes_gate_each_arm(ab):
    setup, clock, _ = ab
    setup(P)
    seen = {}
    for k in range(8):
        clock["t"] = (k * P + P // 2) * MS
        seen[k % 4] = (d1n_ab.arm_on("prep", True), d1n_ab.arm_on("gdncopy", True))
    assert seen == {0: (True, False), 1: (False, False), 2: (False, True), 3: (False, False)}


def test_flag_off_is_off_in_every_window(ab):
    setup, clock, _ = ab
    setup(P)
    for k in range(4):
        clock["t"] = (k * P + 1) * MS
        assert not d1n_ab.arm_on("gdncopy", False)


def test_refused_next_to_d1e_windows(ab):
    setup, clock, log = ab
    setup(P, d1e=P)
    for k in range(4):
        clock["t"] = (k * P + 1) * MS
        assert not d1n_ab.arm_on("gdncopy", True)
    assert len(log.errors) == 1 and "one windowed A/B per decode" in log.errors[0]


def test_uniform_mode_always_on_and_logs_every_20s(ab):
    setup, clock, log = ab
    setup(0)
    clock["t"] = 1 * MS
    assert d1n_ab.arm_on("gdncopy", True)
    d1n_ab.note("vectorized")
    clock["t"] = (20000 + 5) * MS
    assert d1n_ab.arm_on("gdncopy", True)
    assert log.info_lines == ["q38 D1N window=0 class=uniform vectorized=1 fallback=0 steps=0 syncs=0"]


def test_window_line_carries_the_ended_windows_counts(ab):
    setup, clock, log = ab
    setup(P)
    clock["t"] = (2 * P + 10) * MS  # class 2
    d1n_ab.note("steps")
    d1n_ab.note("steps")
    d1n_ab.note("syncs")
    d1n_ab.note("vectorized")
    clock["t"] = (3 * P + 10) * MS  # next window
    d1n_ab.note("steps")
    assert log.info_lines == ["q38 D1N window=2 class=2 vectorized=1 fallback=0 steps=2 syncs=1"]
    assert d1n_ab.counts() == dict(vectorized=0, fallback=0, steps=1, syncs=0)
