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


def _req(prompt, computed, out, max_tokens, arrival=0.0):
    return types.SimpleNamespace(
        num_prompt_tokens=prompt, num_computed_tokens=computed, num_output_tokens=out,
        max_tokens=max_tokens, arrival_time=arrival,
    )


def _run_sched(monkeypatch, flag, steps, running=(), waiting=()):
    calls = []
    plan = iter(steps)

    def fake(self, throttle_prefills=False):
        calls.append(throttle_prefills)
        return types.SimpleNamespace(num_scheduled_tokens=next(plan))

    monkeypatch.setattr(Scheduler, "schedule", fake)
    monkeypatch.setenv("VLLM_TPU_DECODE_PRIORITY", flag)
    decode_priority.patch_decode_priority()
    sched = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            speculative_config=types.SimpleNamespace(num_speculative_tokens=3)
        ),
        prefill_capacity_bound=True, running=list(running), waiting=list(waiting),
    )
    for _ in steps:
        Scheduler.schedule(sched)
    return calls


def test_scaled_window(monkeypatch):
    half = {**DEC, "p": 4096}
    calls = _run_sched(monkeypatch, "512:8:scale=8192", [half] + [DEC] * 5)
    # 8 * 4096 / 8192 = 4 decode-only steps, then prefill is allowed again.
    assert calls == [False, True, True, True, True, False]


def test_protect_holds_prefill_for_near_done_decode(monkeypatch):
    import time as _t
    now = _t.time()
    near = _req(100, 120, 20, 24)            # 4 tokens left
    far = _req(100, 101, 1, 500)
    partial = _req(9000, 4096, 0, 50, now)   # prefill in progress
    small = {**DEC, "p": 1000}
    calls = _run_sched(monkeypatch, "100000:1:protect=8:hold=2", [DEC, DEC, small, DEC, DEC, DEC],
                       running=[near, far, partial])
    # hold=2, then a chunk runs and resets the hold, then hold again for 2 steps
    assert calls == [True, True, False, True, True, False]


def test_protect_needs_prefill_work(monkeypatch):
    near = _req(100, 120, 20, 24)
    calls = _run_sched(monkeypatch, "100000:1:protect=8", [DEC] * 2, running=[near])
    assert calls == [False, False]


def test_deadline_releases_window(monkeypatch):
    import time as _t
    old = _req(9000, 0, 0, 50, _t.time() - 10.0)  # waited 10 s
    big = {**DEC, "p": 8192}
    calls = _run_sched(monkeypatch, "512:4:deadline=5000", [big, DEC, DEC],
                       running=[_req(100, 101, 1, 500)], waiting=[old])
    assert calls == [False, False, False]


def test_option_errors(monkeypatch):
    for bad in ["512:4:foo=1", "512:4:hold=0", "512"]:
        monkeypatch.setenv("VLLM_TPU_DECODE_PRIORITY", bad)
        with pytest.raises(ValueError):
            decode_priority._parse()
