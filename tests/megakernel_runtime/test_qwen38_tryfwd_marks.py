# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_HOST_PHASES laps inside the Qwen3.8 megakernel try_forward
(runner/mk_call_marks.py): inert with HOST_PHASES off, and with it on they
change no decision, call or counter, sum to the call, and do not count the
x_mk_try_forward wrapper twice."""

import random

import pytest
from test_qwen38_tryfwd_lean import D, _driver, _runner, _step

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv
from vllm_torchtpu.runner import host_phases
from vllm_torchtpu.runner import mk_call_marks as mkm

SUMMED = {
    "x_mkf_inspect",
    "x_mkf_gate",
    "x_mkf_pre",
    "x_mkf_ctx_enter",
    "x_mkf_call",
    "x_mkf_ctx_exit",
    "x_mkf_tail",
}


def _reset_host_phases(monkeypatch, every):
    monkeypatch.setattr(host_phases, "_EVERY", every)
    monkeypatch.setattr(host_phases, "_samples", {})
    monkeypatch.setattr(host_phases, "_cur", [])
    monkeypatch.setattr(host_phases, "_kind", "dec")


def _run(monkeypatch, *, timed=False, steps=400, seed=0):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_TRYFWD_LEAN", True, raising=False)
    monkeypatch.setattr(drv.logger, "info", lambda *a, **k: None)
    log: list = []
    d, runner, rng = _driver(log), _runner(), random.Random(seed)
    call = D.try_forward.__get__(d)
    if timed:
        # What tpu_runner's HOST_PHASES fine targets do to the instance.
        d.try_forward = host_phases.timed(call, "x_mk_try_forward")
        call = d.try_forward
    trace = []
    for _ in range(steps):
        ids = _step(rng, runner)
        ok, out = call(
            runner, ids, ids, None, None, model_kwargs={"context_lens": None}
        )
        trace.append((ok, None if out is None else out.tolist()))
    return trace, log, dict(stats=d.stats, whole=d.whole_calls, seq=d.seq)


def test_defaults_are_off():
    assert envs.VLLM_TPU_HOST_PHASES == 0
    assert mkm.ON is False and mkm.FLUSH is False and mkm.STEP_EVERY == 0
    assert drv._HP is False and mkm.active is False


def test_off_records_nothing(monkeypatch):
    _reset_host_phases(monkeypatch, 0)

    def boom(*a, **k):
        raise AssertionError("mk_call_marks reached with HOST_PHASES off")

    for name in ("start", "lap", "finish", "call_begin", "call_end", "flush"):
        monkeypatch.setattr(mkm, name, boom)
    trace, _, state = _run(monkeypatch)
    assert state["whole"] > 0 and any(not ok for ok, _ in trace)
    assert host_phases._samples == {} and mkm.active is False


def test_on_changes_no_decision_and_laps_sum_to_the_call(monkeypatch):
    _reset_host_phases(monkeypatch, 0)
    off = _run(monkeypatch)

    _reset_host_phases(monkeypatch, 10**9)
    monkeypatch.setattr(drv, "_HP", True)
    monkeypatch.setattr(mkm, "STEP_EVERY", 4)
    lines = []
    monkeypatch.setattr(mkm.logger, "info", lambda fmt, *a: lines.append(a))
    clock = iter(range(0, 10**12, 1000))
    tick = lambda: next(clock)  # noqa: E731
    monkeypatch.setattr(mkm.time, "perf_counter_ns", tick)
    monkeypatch.setattr(host_phases.time, "perf_counter_ns", tick)
    on = _run(monkeypatch, timed=True)

    assert on == off
    trace, _, state = on
    s = host_phases._samples["dec"]
    calls = len(trace)
    # The wrapper counts each call once (the inner call bypasses it).
    assert len(s["x_mk_try_forward"]) == calls
    # Every call ends with a tail lap; whole-step calls carry every lap.
    assert len(s["x_mkf_tail"]) == calls
    for name in ("x_mkf_pre", "x_mkf_ctx_enter", "x_mkf_call", "x_mkf_ctx_exit"):
        assert len(s[name]) == state["whole"], name
    assert len(s["x_mkf_call_total"]) == state["whole"]
    # Laps are contiguous: their sum is the time from start() to finish(),
    # which lies within the wrapper's own interval (two clock reads outside).
    lap_sum = sum(sum(v) for k, v in s.items() if k in SUMMED)
    wrap_sum = sum(s["x_mk_try_forward"])
    assert 0 < lap_sum <= wrap_sum
    assert wrap_sum - lap_sum <= 3 * 1000 * calls
    assert not any(
        k.startswith("x_mkf_") and k not in SUMMED | {"x_mkf_call_total"} for k in s
    )
    # One HOSTMKF line per 4th driver step that ran the whole-step program.
    assert lines and all(a[1] % 4 == 0 for a in lines)
    assert all("x_mkf_call=" in a[-1] for a in lines)
    assert mkm.active is False


def test_laps_stop_when_the_call_raises(monkeypatch):
    _reset_host_phases(monkeypatch, 10**9)
    monkeypatch.setattr(drv, "_HP", True)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_TRYFWD_LEAN", True, raising=False)
    d, runner = _driver([]), _runner()

    def bad(*a, **k):
        raise RuntimeError("inspect failed")

    monkeypatch.setattr(drv.MegakernelDriver, "_inspect_step_detail", bad)
    ids = _step(random.Random(0), runner)
    with pytest.raises(RuntimeError):
        D.try_forward(d, runner, ids, ids, None, None)
    assert mkm.active is False
    assert list(host_phases._samples["dec"]) == ["x_mkf_tail"]
