"""q38 D1e host knobs (core/host_knobs.py): default off, switch interval, gc.freeze once
after N steps, engine-core hook, worker hook and compile-ignore registration."""

import gc
import pathlib
import sys

import pytest

from vllm_torchtpu.core import host_knobs as hk

SRC = pathlib.Path(hk.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore():
    sw, th, fc = sys.getswitchinterval(), gc.get_threshold(), gc.get_freeze_count()
    saved = (hk.SWITCH_US, hk.GC_MODE, hk.GC_AFTER, hk.GC_GEN0, hk.ENABLED, hk.AB_PERIOD_MS, hk.AB_REFUSED, hk._now_ns)
    hk._reset_for_tests()
    yield
    hk.SWITCH_US, hk.GC_MODE, hk.GC_AFTER, hk.GC_GEN0, hk.ENABLED, hk.AB_PERIOD_MS, hk.AB_REFUSED, hk._now_ns = saved
    hk._reset_for_tests()
    if hk._gc_cb in gc.callbacks:
        gc.callbacks.remove(hk._gc_cb)
    hk._gc["hooked"] = False
    if gc.get_freeze_count() > fc:  # the interpreter may start with a frozen set (375 in the decode venv)
        gc.unfreeze()
    gc.set_threshold(*th)
    sys.setswitchinterval(sw)


def test_default_off_changes_nothing():
    assert hk.SWITCH_US == 0 and hk.GC_MODE == "" and not hk.ENABLED
    sw, th, fc = sys.getswitchinterval(), gc.get_threshold(), gc.get_freeze_count()
    for _ in range(600):
        hk.on_step("worker")
    assert sys.getswitchinterval() == sw
    assert gc.get_threshold() == th
    assert gc.get_freeze_count() == fc


def test_switch_interval_applied_once():
    hk.SWITCH_US, hk.ENABLED = 200.0, True
    hk.on_step("worker")
    assert sys.getswitchinterval() == pytest.approx(200e-6)
    assert hk._state["applied"]


def test_gc_freeze_after_n_steps_once():
    hk.GC_MODE, hk.GC_AFTER, hk.GC_GEN0, hk.ENABLED = "freeze", 3, 12345, True
    g1, g2 = gc.get_threshold()[1:]
    fc = gc.get_freeze_count()
    hk.on_step("worker")
    hk.on_step("worker")
    assert gc.get_freeze_count() == fc and not hk._state["frozen"]
    keep = [object() for _ in range(10)]  # live objects for the freeze to take
    hk.on_step("worker")
    n = gc.get_freeze_count()
    assert n > fc and hk._state["frozen"]
    assert gc.get_threshold() == (12345, g1, g2)
    junk = [[i] for i in range(1000)]  # new objects after the freeze
    hk.on_step("worker")
    assert gc.get_freeze_count() == n  # frozen once, not again
    del keep, junk


def test_engine_core_hook_counts_and_calls_through():
    calls = []

    class Sched:
        def schedule(self, x):
            calls.append(x)
            return x + 1

    class Core:
        scheduler = Sched()

    core = Core()
    hk.SWITCH_US, hk.ENABLED = 100.0, True
    assert hk.install_engine_core(core)
    assert not hk.install_engine_core(core)  # idempotent
    assert core.scheduler.schedule(1) == 2 and calls == [1]
    assert hk._state["n"] == 1
    assert sys.getswitchinterval() == pytest.approx(100e-6)


def test_worker_and_entry_hooks_present():
    runner = (SRC / "runner/tpu_runner.py").read_text()
    assert 'if _host_knobs.ENABLED:\n            _host_knobs.on_step("worker")' in runner
    entry = (SRC / "__init__.py").read_text()
    assert "patch_engine_core_for_host_knobs()" in entry


def test_keys_compile_ignored():
    from vllm_torchtpu.compilation import tpu_compiler

    for k in ("VLLM_TPU_HOST_SWITCH_US", "VLLM_TPU_HOST_GC", "VLLM_TPU_HOST_GC_AFTER", "VLLM_TPU_HOST_GC_GEN0",
              "VLLM_TPU_HOST_AB_PERIOD_MS"):
        assert k in tpu_compiler._TPU_COMPILE_ENV_IGNORED, k


# ---- in-process A/B windows (VLLM_TPU_HOST_AB_PERIOD_MS)
class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


def _ab(period_ms=1000, switch=200.0, mode="", after=2, gen0=12345):
    hk.SWITCH_US, hk.GC_MODE, hk.GC_AFTER, hk.GC_GEN0, hk.AB_PERIOD_MS = switch, mode, after, gen0, period_ms
    hk._resolve()
    clk = Clock()
    hk._now_ns = clk
    return clk


def test_ab_switch_interval_follows_window_parity():
    clk = _ab()
    sw = sys.getswitchinterval()
    clk.t = 0                      # window 0: on
    hk.on_step("worker")
    assert sys.getswitchinterval() == pytest.approx(200e-6)
    clk.t = 1_500_000_000          # window 1: off -> interpreter default
    hk.on_step("worker")
    assert sys.getswitchinterval() == pytest.approx(hk._DEFAULT_SWITCH)
    clk.t = 2_000_000_000          # window 2: on again
    hk.on_step("worker")
    assert sys.getswitchinterval() == pytest.approx(200e-6)
    assert hk._state["windows"] == [(0, True), (1, False), (2, True)]
    sys.setswitchinterval(sw)


def test_ab_threshold_toggles_after_n_steps():
    clk = _ab(switch=0.0, mode="threshold", after=2)
    g1, g2 = gc.get_threshold()[1:]
    clk.t = 0
    hk.on_step("worker")           # step 1: not yet ready
    assert gc.get_threshold()[0] == hk._DEFAULT_GEN0
    hk.on_step("worker")           # step 2: ready, window 0 on
    assert gc.get_threshold() == (12345, g1, g2)
    clk.t = 1_000_000_000          # window 1: off
    hk.on_step("worker")
    assert gc.get_threshold()[0] == hk._DEFAULT_GEN0
    assert not hk._state["frozen"]


def test_ab_refuses_freeze():
    _ab(mode="freeze")
    assert hk.AB_REFUSED and not hk.ENABLED
    _ab(mode="threshold")
    assert not hk.AB_REFUSED and hk.ENABLED


def test_ab_window_stats_count_gc_per_window():
    clk = _ab(switch=0.0, mode="")
    hk.ENABLED = True
    clk.t = 0
    hk.on_step("worker")
    gc.collect()                   # a gen-2 collection inside window 0
    hk.on_step("worker")
    clk.t = 1_000_000_000
    hk.on_step("worker")           # window change: window 0's stats are closed
    st = hk._state["last_stats"]
    assert st["window"] == 0 and st["on"] is True and st["steps"] == 2
    assert st["gc_n"][2] >= 1 and st["gc_ms"][2] >= 0.0
    clk.t = 2_000_000_000
    hk.on_step("worker")
    assert hk._state["last_stats"]["gc_n"][2] == 0  # reset per window


# ---- q38 L6 SYNC-GC: the GC timer whatever the AB period (served runs: threshold mode, no AB)


def test_gc_timer_is_installed_without_ab():
    hk.GC_MODE, hk.GC_AFTER, hk.AB_PERIOD_MS = "threshold", 500, 0
    hk._resolve()
    assert hk.ENABLED and hk.AB_PERIOD_MS == 0
    hk.on_step("worker")
    assert hk._gc_cb in gc.callbacks and hk._gc["hooked"]
    gc.collect()
    assert hk._gc["n"][2] >= 1


def test_gc_line_every_20_s_without_ab(monkeypatch):
    hk.GC_MODE, hk.GC_AFTER, hk.AB_PERIOD_MS = "threshold", 500, 0
    hk._resolve()
    lines = []
    monkeypatch.setattr(hk.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    clk = Clock()
    hk._now_ns = clk
    hk.on_step("worker")              # t = 0: the line clock starts
    gc.collect()
    clk.t = 19_000_000_000
    hk.on_step("worker")
    assert not any(ln.startswith("Host knobs GC") for ln in lines)
    clk.t = 21_000_000_000
    hk.on_step("worker")
    gc_lines = [ln for ln in lines if ln.startswith("Host knobs GC (q38 D1e): role=worker steps=3 ")]
    assert len(gc_lines) == 1, lines
    assert "gc_n=[" in gc_lines[0] and gc_lines[0].split("gc_n=")[1] != "[0, 0, 0]", gc_lines
    assert hk._gc["n"] == [0, 0, 0]  # reset after each line
