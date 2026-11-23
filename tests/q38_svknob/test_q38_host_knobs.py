# SPDX-License-Identifier: Apache-2.0
"""q38 SV-KNOB GIL-SW (core/host_knobs.py): default off, the switch interval, the in-process A/B windows on a fake
clock, the window-change log, refusals, the engine-core and worker hooks, compile-ignore registration, no GC.

Port of the oxpd form's tests_oxpd/core/test_host_knobs.py (74c03159ab; IX track-3 host trims), GC parts dropped."""

import gc
import inspect
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from vllm_torchtpu.core import host_knobs as hk

SRC = pathlib.Path(hk.__file__).resolve().parents[1]  # .../src/vllm_torchtpu
KEYS = ("VLLM_TPU_HOST_SWITCH_US", "VLLM_TPU_HOST_AB_PERIOD_MS")
ALL_KEYS = KEYS + hk.REFUSED_GC_KEYS
DAY_NS = 86_400 * 10**9
P20 = 20_000  # the car's period (ms)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    sw = sys.getswitchinterval()
    hk._load()
    hk._reset_for_tests()
    saved_now = hk._now_ns
    yield
    for k in ALL_KEYS:
        os.environ.pop(k, None)
    hk._now_ns = saved_now
    hk._load()
    hk._reset_for_tests()
    sys.setswitchinterval(sw)


class Clock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t


def _capture(monkeypatch):
    lines = []
    monkeypatch.setattr(hk.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    return lines


def _set(monkeypatch, switch=None, period=None):
    if switch is not None:
        monkeypatch.setenv("VLLM_TPU_HOST_SWITCH_US", str(switch))
    if period is not None:
        monkeypatch.setenv("VLLM_TPU_HOST_AB_PERIOD_MS", str(period))
    hk._load()


# ---- off


def test_default_off_changes_nothing(monkeypatch):
    lines = _capture(monkeypatch)
    from vllm.v1.engine.core import EngineCore

    init = EngineCore.__init__
    sw = sys.getswitchinterval()
    assert hk.SWITCH_US == 0 and hk.AB_PERIOD_MS == 0 and hk.ENABLED is False and hk._PERIOD_NS == 0
    assert hk.patch_engine_core_for_host_knobs() is False
    assert EngineCore.__init__ is init
    assert not getattr(EngineCore, "_q38_host_knobs_patch", False)
    assert sys.getswitchinterval() == sw
    assert lines == []


def test_off_in_a_fresh_interpreter():
    """Import, the engine-core patch call and the worker guard in a new process with every key unset."""
    code = textwrap.dedent(
        """
        import sys
        sw = sys.getswitchinterval()
        from vllm_torchtpu.core import host_knobs as hk
        from vllm.v1.engine.core import EngineCore
        init = EngineCore.__init__
        r = hk.patch_engine_core_for_host_knobs()
        if hk.ENABLED:
            hk.on_step("worker")
        print("OFFCHECK", r, hk.ENABLED, EngineCore.__init__ is init, sys.getswitchinterval() == sw,
              getattr(EngineCore, "_q38_host_knobs_patch", False))
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in ALL_KEYS}
    out = subprocess.run([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:]
    assert "OFFCHECK False False True True False" in out.stdout, out.stdout[-3000:]
    assert "Host knobs" not in out.stdout


def test_worker_guard_calls_only_when_enabled(monkeypatch):
    """The exact two runner lines, executed: the off path never reaches on_step."""
    text = (SRC / "runner/tpu_runner.py").read_text()
    anchor = '        host_phases.begin_step()\n        if _host_knobs.ENABLED:\n            _host_knobs.on_step("worker")\n'
    assert text.count(anchor) == 1
    assert text.count("from vllm_torchtpu.core import host_knobs as _host_knobs") == 1
    snippet = textwrap.dedent("\n".join(anchor.splitlines()[1:]) + "\n")
    calls = []
    monkeypatch.setattr(hk, "on_step", lambda role: calls.append(role))
    hk.ENABLED = False
    exec(snippet, {"_host_knobs": hk})
    assert calls == []
    hk.ENABLED = True
    exec(snippet, {"_host_knobs": hk})
    assert calls == ["worker"]


# ---- on, no A/B


def test_switch_interval_applied_at_first_step(monkeypatch):
    lines = _capture(monkeypatch)
    _set(monkeypatch, switch=200)
    assert hk.ENABLED and hk._PERIOD_NS == 0
    sw = sys.getswitchinterval()
    assert lines == []  # nothing at load
    assert sys.getswitchinterval() == sw
    hk.on_step("worker")
    assert sys.getswitchinterval() == pytest.approx(200e-6)
    for _ in range(5):
        hk.on_step("worker")
    assert sys.getswitchinterval() == pytest.approx(200e-6)
    assert len(lines) == 1 and lines[0].startswith("Host knobs (q38 SV-KNOB): role=worker pid=")
    assert "switch_us=200 ab_period_ms=0" in lines[0]
    assert not any(ln.startswith("Host knobs AB") for ln in lines)


# ---- A/B windows on a fake clock


def test_ab_exact_pattern_and_window_lines(monkeypatch):
    lines = _capture(monkeypatch)
    _set(monkeypatch, switch=200, period=P20)
    base = 20_414 * DAY_NS  # midnight UTC: window parity is even (on)
    clk = Clock()
    hk._now_ns = clk
    k0 = base // (P20 * 10**6)
    assert k0 % 2 == 0
    default = hk._DEFAULT_SWITCH
    seq = [  # (offset s, expected interval after the step)
        (0.0, 200e-6), (5.0, 200e-6), (19.999, 200e-6), (20.0, default), (39.999, default), (40.0, 200e-6),
        (100.0, default), (100.5, default),
    ]
    got = []
    for off, _ in seq:
        clk.t = base + int(round(off * 1e9))
        hk.on_step("engine_core")
        got.append(sys.getswitchinterval())
    assert got == pytest.approx([w for _, w in seq])
    assert hk._state["windows"] == [(k0, True), (k0 + 1, False), (k0 + 2, True), (k0 + 5, False)]
    ab = [ln for ln in lines if ln.startswith("Host knobs AB (q38 SV-KNOB): ")]
    assert len(lines) == 1 + 4 and len(ab) == 4
    want = [(k0, 1, 0.0, None, 0, 200e-6), (k0 + 1, 0, 20.0, k0, 3, default), (k0 + 2, 1, 40.0, k0 + 1, 2, 200e-6),
            (k0 + 5, 0, 100.0, k0 + 2, 1, default)]
    for ln, (k, on, off, prev, steps, sw) in zip(ab, want):
        t = base + int(round(off * 1e9))
        assert f"role=engine_core pid={os.getpid()} window={k} on={on} t_ns={t} prev_window={prev} steps={steps} " in ln, ln
        assert ln.endswith("switch_s=%.6f" % sw), ln


def test_ab_parity_is_fixed_by_time_of_day():
    for period in (P20, 1, 60_000, 43_200_000):
        n = hk.DAY_MS // period
        assert hk.DAY_MS % period == 0 and n % 2 == 0
        for day in (0, 1, 20_414, 20_415):
            assert ((day * DAY_NS) // (period * 10**6)) % 2 == 0


def test_ab_without_switch_is_an_aa_control(monkeypatch):
    lines = _capture(monkeypatch)
    _set(monkeypatch, period=P20)
    assert hk.ENABLED and hk.SWITCH_US == 0
    sw = sys.getswitchinterval()
    clk = Clock(20_414 * DAY_NS)
    hk._now_ns = clk
    for i in range(6):
        clk.t += 10 * 10**9
        hk.on_step("worker")
        assert sys.getswitchinterval() == sw
    ab = [ln for ln in lines if ln.startswith("Host knobs AB")]
    assert len(ab) == 4 and [("on=1" in ln) for ln in ab] == [True, False, True, False]


def test_every_window_change_is_logged_once(monkeypatch):
    lines = _capture(monkeypatch)
    _set(monkeypatch, switch=200, period=1000)
    clk = Clock(20_414 * DAY_NS - 250_000_000)  # the first step lands on a window start
    hk._now_ns = clk
    for i in range(200):
        clk.t += 250_000_000  # 4 steps a window
        hk.on_step("worker")
    ab = [ln for ln in lines if ln.startswith("Host knobs AB")]
    assert len(ab) == len(hk._state["windows"]) == 50
    assert [ln.split(" window=")[1].split()[0] for ln in ab] == [str(k) for k, _ in hk._state["windows"]]
    assert all(" steps=4 " in ln for ln in ab[1:])


# ---- refusals


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("bad", ["abc", "-5", "1.5", "200us", "0x10"])
def test_malformed_values_refused(monkeypatch, key, bad):
    monkeypatch.setenv(key, bad)
    with pytest.raises(ValueError):
        hk._load()


@pytest.mark.parametrize(
    "switch,period,ok",
    [
        (100_000, 0, True), (100_001, 0, False), (1, 0, True), (0, 0, True),
        (200, 20_000, True), (200, 1, True), (200, 43_200_000, True), (0, 20_000, True),
        (200, 7_000, False), (200, 28_800_000, False), (200, 86_400_000, False), (200, 50_000, True),
        (200, 17_280_000, False),
    ],
)
def test_ranges_and_period_rule(monkeypatch, switch, period, ok):
    monkeypatch.setenv("VLLM_TPU_HOST_SWITCH_US", str(switch))
    monkeypatch.setenv("VLLM_TPU_HOST_AB_PERIOD_MS", str(period))
    if ok:
        hk._load()
        assert hk.ENABLED == (switch > 0 or period > 0)
    else:
        with pytest.raises(ValueError):
            hk._load()


@pytest.mark.parametrize("key", hk.REFUSED_GC_KEYS)
def test_oxpd_gc_keys_refused(monkeypatch, key):
    monkeypatch.setenv(key, "threshold" if key == "VLLM_TPU_HOST_GC" else "500")
    with pytest.raises(ValueError):
        hk._load()
    monkeypatch.setenv(key, " ")
    hk._load()  # empty is unset


def test_refusal_stops_import_in_a_fresh_interpreter():
    env = {k: v for k, v in os.environ.items() if k not in ALL_KEYS}
    env["VLLM_TPU_HOST_AB_PERIOD_MS"] = "7000"
    out = subprocess.run([sys.executable, "-c", "import vllm_torchtpu.core.host_knobs"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    assert out.returncode != 0 and "VLLM_TPU_HOST_AB_PERIOD_MS" in out.stdout and "ValueError" in out.stdout


# ---- hooks


def test_engine_core_hook_calls_through_unchanged(monkeypatch):
    calls = []
    sentinel = object()

    class Sched:
        def schedule(self, *a, **k):
            calls.append((a, k))
            return sentinel

    class Core:
        scheduler = Sched()

    core = Core()
    _set(monkeypatch, switch=100)
    assert hk.install_engine_core(core)
    assert not hk.install_engine_core(core)  # idempotent
    assert core.scheduler.schedule(True, x=1) is sentinel and calls == [((True,), {"x": 1})]
    assert hk._state["n"] == 1
    assert sys.getswitchinterval() == pytest.approx(100e-6)


def test_patch_engine_core_real_class(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    class Sched:
        def schedule(self, *a):
            return a

    def fake_init(self, *a, **k):
        self.scheduler = Sched()

    monkeypatch.setattr(EngineCore, "__init__", fake_init)
    monkeypatch.setattr(EngineCore, "_q38_host_knobs_patch", False, raising=False)
    _set(monkeypatch, switch=200)
    assert hk.patch_engine_core_for_host_knobs() is True
    assert EngineCore.__init__ is not fake_init and EngineCore._q38_host_knobs_patch
    assert hk.patch_engine_core_for_host_knobs() is True and hk._state["n"] == 0  # idempotent, no step yet
    core = EngineCore.__new__(EngineCore)
    EngineCore.__init__(core)
    assert core.scheduler.schedule._q38_host_knobs and core.scheduler.schedule(7) == (7,)
    assert hk._state["n"] == 1 and sys.getswitchinterval() == pytest.approx(200e-6)


def test_vllm_engine_core_looks_schedule_up_per_call():
    """vLLM 0.31 EngineCore (and K3's PP lazy stepper) call scheduler.schedule through the instance each step, so
    the instance-attribute wrapper is reached (the same mechanism as on the oxpd form)."""
    from vllm.v1.engine.core import EngineCore

    for fn in (EngineCore.step, EngineCore.step_with_batch_queue):
        assert "self.scheduler.schedule(" in inspect.getsource(fn), fn
    lazy = (SRC / "core/pp_lazy_step.py").read_text()
    assert "scheduler = core.scheduler\n        scheduler_output = scheduler.schedule(" in lazy
    entry = (SRC / "__init__.py").read_text()
    body = entry.split("def _run_engine_core_with_tpu_patches(", 1)[1].split("\ndef ", 1)[0]
    assert body.count("patch_engine_core_for_host_knobs()") == 1
    assert body.index("patch_engine_core_for_lazy_push(") < body.index("patch_engine_core_for_host_knobs()") < body.index(
        "original_run(*args, **kwargs)")


# ---- compile key, GC, IX-neutral


def test_keys_registered_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    for k in KEYS:
        assert k in envs.environment_variables and k in tpu_compiler._TPU_COMPILE_ENV_IGNORED, k
    base = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv("VLLM_TPU_HOST_SWITCH_US", "200")
    monkeypatch.setenv("VLLM_TPU_HOST_AB_PERIOD_MS", "20000")
    assert tpu_compiler._tpu_compile_env_factors() == base
    keyed = {p.relative_to(SRC.parent).as_posix() for p in tpu_compiler._iter_runtime_cache_key_files(SRC.parent)}
    assert keyed and not keyed & {"vllm_torchtpu/core/host_knobs.py", "vllm_torchtpu/runner/tpu_runner.py",
                                  "vllm_torchtpu/__init__.py", "vllm_torchtpu/envs.py",
                                  "vllm_torchtpu/compilation/tpu_compiler.py"}


def test_no_gc_mechanism(monkeypatch):
    src = (SRC / "core/host_knobs.py").read_text()
    assert "import gc" not in src and "gc." not in src.replace("runner/gc_at_waits.py", "")
    th, ncb, fc = gc.get_threshold(), len(gc.callbacks), gc.get_freeze_count()
    _set(monkeypatch, switch=200, period=1000)
    clk = Clock(20_414 * DAY_NS)
    hk._now_ns = clk
    for _ in range(40):
        clk.t += 300_000_000
        hk.on_step("worker")
    assert (gc.get_threshold(), len(gc.callbacks), gc.get_freeze_count()) == (th, ncb, fc)


def test_ix_neutral_host_only():
    src = (SRC / "core/host_knobs.py").read_text()
    imports = sorted(ln.strip() for ln in src.splitlines() if ln.startswith(("import ", "from ")))
    assert imports == ["from __future__ import annotations", "from typing import Any", "from vllm_torchtpu import envs",
                       "from vllm_torchtpu.logger import init_logger", "import os", "import sys", "import time"]
    code = src.split('"""', 2)[2]
    assert code.count("sys.setswitchinterval(") == 1
    for tok in ("import torch", "import jax", "import numpy", "scheduler_output", "num_scheduled", "gc."):
        assert tok not in code, tok
