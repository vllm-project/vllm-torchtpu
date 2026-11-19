# SPDX-License-Identifier: Apache-2.0
"""q38 D1 CP0: RunnerStageTiming logs whether the stage timing is on or off.

The CP0 A/B reads this line once per decode TP rank as engagement evidence
for VLLM_TPU_STAGE_TIMING_OFF (DEC-HOST-B's switch), so the line must say
"off" exactly when the switch is set and "on" otherwise.
"""

import threading

from vllm_torchtpu.runner import stage_timing


class _FakeEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


class _Capture:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)


def _make(monkeypatch, off):
    cap = _Capture()
    monkeypatch.setattr(stage_timing, "logger", cap)
    monkeypatch.setattr(stage_timing, "TpuEvent", _FakeEvent)
    monkeypatch.setattr(stage_timing, "has_kv_transfer_group", lambda: True)
    real_thread = threading.Thread
    # No real collector thread: the fake events never complete.
    monkeypatch.setattr(
        stage_timing.threading, "Thread", lambda *a, **k: real_thread(target=lambda: None)
    )
    if off:
        monkeypatch.setenv("VLLM_TPU_STAGE_TIMING_OFF", "1")
    else:
        monkeypatch.delenv("VLLM_TPU_STAGE_TIMING_OFF", raising=False)
    stage_timing.RunnerStageTiming(rank=5)
    return cap.lines


def test_logs_off_when_switch_set(monkeypatch):
    assert _make(monkeypatch, off=True) == [
        "Runner stage timing: off (VLLM_TPU_STAGE_TIMING_OFF=1) rank=5"
    ]


def test_logs_on_by_default(monkeypatch):
    assert _make(monkeypatch, off=False) == [
        "Runner stage timing: on (VLLM_TPU_STAGE_TIMING_OFF=0) rank=5"
    ]
