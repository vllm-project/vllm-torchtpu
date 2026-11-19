# SPDX-License-Identifier: Apache-2.0
"""q38 D1b: VLLM_TPU_STAGE_TIMING_OFF removes the runner's stage timing.

With a KV connector the stage timing is on even under
VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK (the connector is the sink), so the PD
decode pays two TpuEvent records per stage and a collector thread. The off
switch must remove all of it and leave the stage bodies running.
"""

import threading

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.runner import stage_timing


class _FakeEvent:
    made = 0
    recorded = 0

    def __init__(self):
        type(self).made += 1

    def record(self):
        type(self).recorded += 1

    def synchronize(self):
        pass


def _setup(monkeypatch, off):
    _FakeEvent.made = 0
    _FakeEvent.recorded = 0
    threads = []
    real_thread = threading.Thread

    def counting_thread(*args, **kwargs):
        t = real_thread(*args, **kwargs)
        threads.append(t)
        return t

    monkeypatch.setattr(stage_timing, "TpuEvent", _FakeEvent)
    monkeypatch.setattr(stage_timing.threading, "Thread", counting_thread)
    # A connector is present: the PD case, where the sink gate is always true.
    monkeypatch.setattr(stage_timing, "has_kv_transfer_group", lambda: True)
    monkeypatch.setenv("VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK", "1")
    if off:
        monkeypatch.setenv("VLLM_TPU_STAGE_TIMING_OFF", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_STAGE_TIMING_OFF", raising=False)
    return threads


def _run_stages(timing):
    ran = []
    with timing.stage("main_forward"):
        ran.append("main_forward")
    with timing.stage("draft_propose"):
        ran.append("draft_propose")
    with timing.host_stage("output_assembly"):
        ran.append("output_assembly")
    return ran


def test_default_with_connector_records_events(monkeypatch):
    # Control: without the switch, a connector turns the timing on, so the
    # off test below can see records if the switch failed.
    threads = _setup(monkeypatch, off=False)
    timing = stage_timing.RunnerStageTiming(rank=3)
    made_at_init = _FakeEvent.made
    assert len(threads) == 1
    assert _run_stages(timing) == ["main_forward", "draft_propose", "output_assembly"]
    assert _FakeEvent.made - made_at_init == 4
    cpu, _ = timing.take_samples()
    assert [s[1] for s in cpu] == ["main_forward", "draft_propose", "output_assembly"]


def test_off_removes_events_thread_and_samples(monkeypatch):
    threads = _setup(monkeypatch, off=True)
    timing = stage_timing.RunnerStageTiming(rank=3)
    assert threads == []
    assert _FakeEvent.made == 0
    assert _run_stages(timing) == ["main_forward", "draft_propose", "output_assembly"]
    assert _FakeEvent.made == 0 and _FakeEvent.recorded == 0
    assert timing.take_samples() == ([], [])


def test_off_stage_body_exceptions_propagate(monkeypatch):
    _setup(monkeypatch, off=True)
    timing = stage_timing.RunnerStageTiming(rank=0)
    try:
        with timing.stage("sample"):
            raise ValueError("body")
    except ValueError as e:
        assert str(e) == "body"
    else:
        raise AssertionError("the stage swallowed the body's exception")


def test_off_flag_is_compile_neutral():
    # Host-side only: the flag must not change the compile-cache key.
    assert "VLLM_TPU_STAGE_TIMING_OFF" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
