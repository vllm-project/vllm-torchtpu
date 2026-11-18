# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU component tests for MiniCPM-o 4.5 Code2Wav watchdog."""

from __future__ import annotations

import time

import pytest
import torch
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

WATCHDOG_TIMEOUT_S = 0.05
STALL_SLEEP_S = 0.15


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_minicpmo_4_5_code2wav_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MiniCPM-o 4.5 Code2Wav Watchdog.

    - TPU path: import ``Code2WavWatchdog`` in-body, configure
      ``VLLM_OMNI_CODE2WAV_WATCHDOG_TIMEOUT_S="0.05"``, and run watched TPU
      tensor phases.
    - Setup check: ``current_omni_platform.device_type == "tpu"``,
      ``watchdog.enabled`` is True and ``timeout_s == 0.05`` (vs disabled when
      env var is unset).
    - Outcome check: fast ``decode/encoder`` phase emits 0 reports; stalled
      ``decode/cfm`` phase emits 1 ``Code2WavWatchdogReport`` with matching
      phase, bounded details, ``elapsed_s >= timeout_s``, and owner stack.
    """
    from vllm_omni.model_executor.models.minicpmo_4_5.code2wav_watchdog import (
        CODE2WAV_WATCHDOG_TIMEOUT_ENV,
        Code2WavWatchdog,
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    monkeypatch.delenv(CODE2WAV_WATCHDOG_TIMEOUT_ENV, raising=False)
    disabled_watchdog = Code2WavWatchdog.from_environment()
    assert disabled_watchdog.enabled is False, (
        "Expected watchdog to be disabled when env var is unset"
    )
    disabled_watchdog.close()

    reports = []
    monkeypatch.setenv(CODE2WAV_WATCHDOG_TIMEOUT_ENV, str(WATCHDOG_TIMEOUT_S))
    watchdog = Code2WavWatchdog.from_environment(reporter=reports.append)
    try:
        assert watchdog.enabled is True, (
            "Expected watchdog to be enabled when timeout env var is set"
        )
        assert watchdog.timeout_s == WATCHDOG_TIMEOUT_S, (
            f"Expected timeout_s {WATCHDOG_TIMEOUT_S}, got {watchdog.timeout_s}"
        )

        x = torch.ones(4, 4, device="tpu")
        with watchdog.watch("decode/encoder", {"step": 1}):
            y = x + 1.0
            torch.accelerator.synchronize()
        assert len(reports) == 0, (
            f"Expected 0 reports for fast decode/encoder phase, got {len(reports)}"
        )

        with watchdog.watch("decode/cfm", {"step": 2, "items": list(range(32))}):
            _ = y * 2.0
            time.sleep(STALL_SLEEP_S)
            completed = True

        assert completed is True, "Watchdog must not interrupt execution"
        assert len(reports) == 1, (
            f"Expected 1 report for stalled decode/cfm phase, got {len(reports)}"
        )
        report = reports[0]
        assert report.phase == "decode/cfm", (
            f"Expected report phase 'decode/cfm', got {report.phase!r}"
        )
        assert len(report.details["items"]) == 16, (
            f"Expected bounded detail list of 16, got {len(report.details['items'])}"
        )
        assert report.elapsed_s >= WATCHDOG_TIMEOUT_S, (
            f"Expected elapsed_s >= {WATCHDOG_TIMEOUT_S}, got {report.elapsed_s}"
        )
        assert bool(report.stack.strip()), (
            "Expected non-empty owner-thread stack trace in watchdog report"
        )
    finally:
        watchdog.close()
