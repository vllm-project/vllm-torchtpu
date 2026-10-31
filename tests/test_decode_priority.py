# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DECODE_PRIORITY=N:K: K throttled (decode-only) schedules after a
step with >= N prefill tokens."""

import inspect
import types

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import decode_priority


def _run(monkeypatch, flag, steps):
    """`steps`: per schedule call, the {req: tokens} the fake original returns.
    Returns the throttle flag each call passed to the original."""
    calls = []
    plan = iter(steps)

    def fake(self, throttle_prefills=False):
        calls.append((throttle_prefills, self.prefill_capacity_bound))
        return types.SimpleNamespace(num_scheduled_tokens=next(plan))

    monkeypatch.setattr(Scheduler, "schedule", fake)
    monkeypatch.setenv("VLLM_TPU_DECODE_PRIORITY", flag)
    decode_priority.patch_decode_priority()
    sched = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            speculative_config=types.SimpleNamespace(num_speculative_tokens=3)
        ),
        prefill_capacity_bound=True,
    )
    for _ in steps:
        Scheduler.schedule(sched)
    return calls


DEC = {"a": 4, "b": 4}  # decode-only step: 1 + 3 speculative tokens each


def test_window_after_large_prefill(monkeypatch):
    big = {**DEC, "p": 8192}
    calls = _run(monkeypatch, "4096:3", [big, DEC, DEC, DEC, DEC, big, DEC])
    assert [t for t, _ in calls] == [False, True, True, True, False, False, True]
    # The DP capacity release is disabled inside the window only.
    assert [c for t, c in calls if t] == [False] * 4


def test_small_prefill_and_decode_steps_do_not_open_a_window(monkeypatch):
    small = {**DEC, "p": 1000}
    calls = _run(monkeypatch, "4096:3", [small, DEC, {"a": 4, "b": 3}, DEC])
    assert [t for t, _ in calls] == [False] * 4


def test_caller_throttle_is_kept(monkeypatch):
    calls = []

    def fake(self, throttle_prefills=False):
        calls.append(throttle_prefills)
        return types.SimpleNamespace(num_scheduled_tokens=DEC)

    monkeypatch.setattr(Scheduler, "schedule", fake)
    monkeypatch.setenv("VLLM_TPU_DECODE_PRIORITY", "4096:3")
    decode_priority.patch_decode_priority()
    sched = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(speculative_config=None), prefill_capacity_bound=False
    )
    Scheduler.schedule(sched, True)
    assert calls == [True]


def test_off_and_bad_values(monkeypatch):
    orig = Scheduler.schedule
    monkeypatch.delenv("VLLM_TPU_DECODE_PRIORITY", raising=False)
    decode_priority.patch_decode_priority()
    assert Scheduler.schedule is orig
    monkeypatch.setenv("VLLM_TPU_DECODE_PRIORITY", "4096:0")
    with pytest.raises(ValueError):
        decode_priority.patch_decode_priority()


def test_upstream_schedule_takes_throttle_prefills():
    assert "throttle_prefills" in inspect.signature(inspect.unwrap(Scheduler.schedule)).parameters
