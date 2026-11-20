# SPDX-License-Identifier: Apache-2.0
"""RunnerStageTiming."""

from unittest.mock import patch

import pytest

from vllm_torchtpu.runner import stage_timing
from vllm_torchtpu.runner.stage_timing import RunnerStageTiming


@pytest.mark.cpu_test
def test_disabled_stage_timing_records_nothing():
    with patch.object(
        stage_timing, "TpuEvent", side_effect=AssertionError("no device event")
    ):
        timing = RunnerStageTiming(rank=0, enabled=False)
        with timing.stage("main_forward"):
            pass
        with timing.host_stage("output_assembly"):
            pass
    assert timing.take_samples() == ([], [])
