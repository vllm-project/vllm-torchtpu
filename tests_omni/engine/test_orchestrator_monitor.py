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
"""TPU engine tests for OrchestratorMonitor window tracing."""

import json
from pathlib import Path

import numpy as np
import pytest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

_EXPECTED_WINDOW_S = 1.0
_PROMPT = {"prompt": "a red cube on a white table"}


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=1,
        seed=0,
    )


def test_orchestrator_window_monitor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Orchestrator Window Monitor

    - TPU path: Runs single-device ``OmniRunner`` on ``tiny_flux_schnell()`` with
      ``enable_orch_monitor=True`` and ``VLLM_OMNI_ORCH_MONITOR_PATH`` set, then
      closes the engine to flush ``OrchestratorMonitor``.
    - Setup check: Verifies ``_enable_orch_monitor`` on the engine matches the flag
      and no trace file is written when ``enable_orch_monitor=False``.
    - Outcome check: Verifies the flushed JSON schema, ``loop_active >= 1``,
      matching window series lengths, and bitwise image parity against baseline.
    """
    model_dir = tiny_flux_schnell()
    monitor_path = tmp_path / "orch_monitor.json"
    monkeypatch.setenv("VLLM_OMNI_ORCH_MONITOR_PATH", str(monitor_path))

    baseline_runner = OmniRunner(model_dir, enable_orch_monitor=False)
    try:
        assert baseline_runner.omni.engine._enable_orch_monitor is False, (
            "baseline engine _enable_orch_monitor must be False"
        )
        baseline_out = list(baseline_runner.generate(_PROMPT, _sampling_params()))[0]
        baseline_img = np.asarray(baseline_out.images[0])
    finally:
        baseline_runner.close()

    assert not monitor_path.exists(), (
        f"monitor file {monitor_path} should not exist when enable_orch_monitor=False"
    )

    monitored_runner = OmniRunner(model_dir, enable_orch_monitor=True)
    try:
        assert monitored_runner.omni.engine._enable_orch_monitor is True, (
            "monitored engine _enable_orch_monitor must be True"
        )
        monitored_out = list(monitored_runner.generate(_PROMPT, _sampling_params()))[0]
        monitored_img = np.asarray(monitored_out.images[0])
    finally:
        monitored_runner.close()

    assert monitor_path.is_file(), (
        f"OrchestratorMonitor did not flush JSON file to {monitor_path}"
    )
    payload = json.loads(monitor_path.read_text(encoding="utf-8"))
    assert payload["configured_window_s"] == _EXPECTED_WINDOW_S, (
        f"configured_window_s {payload['configured_window_s']} != {_EXPECTED_WINDOW_S}"
    )

    windows = payload["windows"]
    n_windows = len(windows["duration_s"])
    assert n_windows >= 1, f"expected >= 1 recorded window, got {n_windows}"
    assert sum(windows["loop_active"]) >= 1, (
        f"expected loop_active sum >= 1, got {windows['loop_active']}"
    )
    for key in (
        "loop_idle",
        "loop_active",
        "dispatch_queue_size",
        "dispatch_queue_high_water",
    ):
        assert len(windows[key]) == n_windows, (
            f"windows[{key!r}] length {len(windows[key])} != {n_windows}"
        )

    assert "stage=0,replica=0" in payload["replicas"], (
        f"missing 'stage=0,replica=0' in replicas {list(payload['replicas'].keys())}"
    )
    replica_series = payload["replicas"]["stage=0,replica=0"]
    for key in ("outputs_queue_size", "inflight"):
        assert len(replica_series[key]) == n_windows, (
            f"replicas['stage=0,replica=0'][{key!r}] length "
            f"{len(replica_series[key])} != {n_windows}"
        )

    assert np.array_equal(monitored_img, baseline_img), (
        "generated image pixels diverged with enable_orch_monitor=True"
    )
