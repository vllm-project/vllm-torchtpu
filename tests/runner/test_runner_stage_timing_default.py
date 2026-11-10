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
"""The KV-transfer stage timings are off by default; ``tpu_runner.stage_timing``
turns them on when the KV connector has a stats sink to feed."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.runner import runner_options
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.utils import RUNNER_OPTIONS_SECTION


@pytest.mark.parametrize(
    "stage_timing,kv_uses,want",
    [
        (False, True, False),  # off by default under P/D
        (False, False, False),
        (True, True, True),
        (True, False, False),  # no stats sink, nothing to feed
    ],
)
def test_stage_timing_enabled(stage_timing, kv_uses, want):
    options = runner_options.RunnerOptions(stage_timing=stage_timing)
    assert TPUModelRunner._stage_timing_enabled(options, kv_uses) is want


def test_the_former_environment_switches_have_no_effect(monkeypatch):
    monkeypatch.setenv("TPU_RUNNER_STAGE_TIMING_MODE", "on")
    monkeypatch.setenv("TPU_RUNNER_STAGE_TIMING", "1")
    options = runner_options.get(SimpleNamespace(additional_config={}))
    assert TPUModelRunner._stage_timing_enabled(options, True) is False


def test_options_come_from_the_tpu_runner_section():
    assert runner_options.SECTION == RUNNER_OPTIONS_SECTION == "tpu_runner"
    on = SimpleNamespace(additional_config={"tpu_runner": {"stage_timing": True}})
    assert runner_options.get(on).stage_timing is True
    assert runner_options.get(SimpleNamespace(additional_config=None)) == (
        runner_options.RunnerOptions()
    )
    assert runner_options.get(SimpleNamespace()).stage_timing is False


@pytest.mark.parametrize(
    "raw,match",
    [
        ({"stage_timing": 1}, "must be a bool"),
        ({"stage_timing": "true"}, "must be a bool"),
        ({"stage_timeing": True}, "unknown keys"),
        (["stage_timing"], "must be a JSON object"),
    ],
)
def test_parse_is_strict(raw, match):
    with pytest.raises(ValueError, match=match):
        runner_options.parse(raw)
