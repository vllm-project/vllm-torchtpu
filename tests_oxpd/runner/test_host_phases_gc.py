# SPDX-License-Identifier: Apache-2.0
"""q38 L6-INSTR (H1 rung 1): the GC census of runner/host_phases.py.

With VLLM_TPU_HOST_PHASES=N > 0 (and VLLM_TPU_HOST_PHASES_GC not 0) the
worker's first ``begin_step`` appends a ``gc.callbacks`` timer and every N
steps each rank logs a GCCENSUS line with the window's cyclic GC pauses:
``gc_us=<sum>/<collections>/<gen-2 collections>``, the longest pause and the
per-generation split. With N = 0 nothing is installed and nothing is logged;
the HOSTPHASES line is the same with the census on or off.
"""

import gc
import os
import subprocess
import sys
import time

import pytest

from vllm_torchtpu.runner import host_phases


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


class _Clock:
    """Virtual time: reads do not advance it; the test advances it."""

    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def tick(self, us):
        self.ns += us * 1000


@pytest.fixture
def hp(monkeypatch):
    """host_phases with N=4, the GC census on, a virtual step clock and a
    separate virtual GC clock. The callback counts as installed but is not in
    gc.callbacks, so only the test's own start / stop calls feed it (a real
    collection mid-test would add counts); the install test opts in."""
    monkeypatch.setattr(host_phases, "_EVERY", 4)
    monkeypatch.setattr(host_phases, "_SLOW_NS", 0)
    monkeypatch.setattr(host_phases, "_SUBS", False)
    monkeypatch.setattr(host_phases, "_GC", True)
    monkeypatch.setattr(host_phases, "_gc_hooked", True)
    monkeypatch.setattr(host_phases, "_gc_t0", None)
    monkeypatch.setattr(host_phases, "_gc_win", [0, 0, 0, 0])
    monkeypatch.setattr(host_phases, "_gc_gen", [[0, 0], [0, 0], [0, 0]])
    monkeypatch.setattr(host_phases, "_last", None)
    monkeypatch.setattr(host_phases, "_steps", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    monkeypatch.setattr(host_phases, "_sub_acc", {})
    monkeypatch.setattr(host_phases, "_sub_samples", {})
    monkeypatch.setattr(host_phases, "_rank", 3)
    log = _Log()
    monkeypatch.setattr(host_phases, "logger", log)
    clock = _Clock()
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", clock)
    gclock = _Clock()
    monkeypatch.setattr(host_phases, "_gc_clock", gclock)
    yield log, clock, gclock
    while host_phases._gc_cb in gc.callbacks:
        gc.callbacks.remove(host_phases._gc_cb)


def _pause(gclock, gen, us):
    host_phases._gc_cb("start", {"generation": gen})
    gclock.tick(us)
    host_phases._gc_cb("stop", {"generation": gen, "collected": 0, "uncollectable": 0})


def _step(clock):
    host_phases.begin_step()
    clock.tick(30)
    host_phases.mark("a")
    clock.tick(20)


def test_gc_census_line_carries_the_callback_sums(hp):
    log, clock, gclock = hp
    # Window 1 (steps 0..2 after the first begin_step): gen0 120 us, gen2 300
    # us, gen1 45 us, gen0 5 us.
    _step(clock)
    _pause(gclock, 0, 120)
    _step(clock)
    _pause(gclock, 2, 300)
    _pause(gclock, 1, 45)
    _step(clock)
    _pause(gclock, 0, 5)
    host_phases.begin_step()  # steps=4: report
    # Window 2: one gen2 pause of 7 us, then the report at steps=8.
    _pause(gclock, 2, 7)
    for _ in range(3):
        _step(clock)
    host_phases.begin_step()
    gcl = [ln for ln in log.lines if ln.startswith("GCCENSUS")]
    assert gcl == [
        "GCCENSUS rank=3 steps=4 window=4 cyclic GC pauses (sum_us/collections/gen2_collections) "
        "gc_us=470/4/1 max_us=300 g0=125/2 g1=45/1 g2=300/1",
        "GCCENSUS rank=3 steps=8 window=4 cyclic GC pauses (sum_us/collections/gen2_collections) "
        "gc_us=7/1/1 max_us=7 g0=0/0 g1=0/0 g2=7/1",
    ], log.lines
    # Each GCCENSUS line follows its window's HOSTPHASES line.
    order = [ln.split()[0] for ln in log.lines]
    assert order == ["HOSTPHASES", "GCCENSUS", "HOSTPHASES", "GCCENSUS"], order


def test_callback_is_installed_at_the_first_step_and_times_a_real_collection(hp, monkeypatch):
    log, clock, _ = hp
    monkeypatch.setattr(host_phases, "_gc_hooked", False)
    host_phases._gc_clock = time.monotonic_ns  # a real clock (perf_counter_ns is virtual here)
    assert host_phases._gc_cb not in gc.callbacks
    host_phases.begin_step()
    assert host_phases._gc_cb in gc.callbacks and host_phases._gc_hooked
    host_phases.begin_step()  # a second step does not install twice
    assert gc.callbacks.count(host_phases._gc_cb) == 1
    before = list(host_phases._gc_win)
    gc.collect()  # a full (gen-2) collection
    assert host_phases._gc_win[1] >= before[1] + 1
    assert host_phases._gc_win[2] >= before[2] + 1
    assert host_phases._gc_gen[2][1] >= 1


def test_census_off_installs_nothing_and_logs_nothing(hp, monkeypatch):
    log, clock, gclock = hp
    monkeypatch.setattr(host_phases, "_GC", False)
    monkeypatch.setattr(host_phases, "_gc_hooked", False)
    for _ in range(9):
        _step(clock)
    gc.collect()
    assert host_phases._gc_cb not in gc.callbacks and not host_phases._gc_hooked
    assert host_phases._gc_win == [0, 0, 0, 0]
    assert not any(ln.startswith("GCCENSUS") for ln in log.lines)
    assert sum(ln.startswith("HOSTPHASES") for ln in log.lines) == 2


def test_hostphases_line_is_the_same_with_the_census_on_or_off(hp, monkeypatch):
    log, clock, gclock = hp
    for _ in range(9):
        _step(clock)
        _pause(gclock, 0, 11)
    on = [ln for ln in log.lines if ln.startswith("HOSTPHASES")]
    monkeypatch.setattr(host_phases, "_GC", False)
    monkeypatch.setattr(host_phases, "_last", None)
    monkeypatch.setattr(host_phases, "_steps", 0)
    monkeypatch.setattr(host_phases, "_samples", {})
    log.lines.clear()
    clock.ns = 0
    for _ in range(9):
        _step(clock)
    off = [ln for ln in log.lines if ln.startswith("HOSTPHASES")]
    assert on == off and len(on) == 2, (on, off)


_PROBE = (
    "import gc; from vllm_torchtpu.runner import host_phases as h; "
    "print('GC', h._GC, 'EVERY', h._EVERY, 'HOOKED', h._gc_cb in gc.callbacks)"
)


def _probe(extra):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VLLM_TPU_HOST_PHASES", "VLLM_TPU_HOST_PHASES_GC")
    }
    env.update(extra)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("GC ")]
    assert len(line) == 1, out.stdout[-2000:]
    return line[0]


def test_fresh_process_flag_gating():
    # Off by default; on with host phases; the opt-out wins. Importing the
    # module never installs the callback (the worker's first step does).
    assert _probe({}) == "GC False EVERY 0 HOOKED False"
    assert _probe({"VLLM_TPU_HOST_PHASES": "64"}) == "GC True EVERY 64 HOOKED False"
    assert (
        _probe({"VLLM_TPU_HOST_PHASES": "64", "VLLM_TPU_HOST_PHASES_GC": "0"})
        == "GC False EVERY 64 HOOKED False"
    )
    assert _probe({"VLLM_TPU_HOST_PHASES_GC": "1"}) == "GC False EVERY 0 HOOKED False"
