# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS


def test_logs_streak_with_request_counters(monkeypatch):
    """A 2 s run of empty schedules with running requests logs each request's
    counters and whether the last update_from_output carried it; a scheduled
    step resets the streak."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_torchtpu.core import zero_schedule_diag as diag

    monkeypatch.setenv("VLLM_TPU_STALL_WATCHDOG_S", "15")
    tokens = [0]
    monkeypatch.setattr(
        Scheduler, "schedule", lambda self, *a, **k: NS(total_num_scheduled_tokens=tokens[0])
    )
    monkeypatch.setattr(Scheduler, "update_from_output", lambda self, so, mro: None)
    diag.patch_zero_schedule_diag()
    clock = [100.0]
    monkeypatch.setattr(diag.time, "monotonic", lambda: clock[0])
    logs = []
    monkeypatch.setattr(diag.logger, "warning", lambda msg, *a: logs.append(msg % a))

    sched = object.__new__(Scheduler)
    req = lambda rid: NS(
        request_id=rid, num_computed_tokens=10, num_output_placeholders=4,
        num_tokens_with_spec=6, num_prompt_tokens=5, max_tokens=99,
        spec_token_ids=[1, 2, 3], status="RUNNING",
    )
    sched.running = [req("req-a"), req("req-b")]
    sched.waiting = []
    sched.update_from_output(None, NS(req_ids=["req-a"]))
    for _ in range(3):
        sched.schedule()
        clock[0] += 1.0
    assert len(logs) == 1
    assert "2 running" in logs[0] and "3 empty schedules" in logs[0]
    assert "req-a: computed=10 placeholders=4" in logs[0]
    assert "in_last_update=True" in logs[0] and "in_last_update=False" in logs[0]
    tokens[0] = 8
    sched.schedule()
    assert sched._tpu_zero_streak is None
