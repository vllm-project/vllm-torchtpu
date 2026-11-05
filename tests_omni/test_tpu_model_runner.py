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
"""Unit tests for src/vllm_torchtpu/omni/worker/tpu_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm_omni.config import OmniModelConfig
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.omni


def _fake_base_init(self, vllm_config, *args, **kwargs):
    """Stand-in for TPUModelRunner.__init__: only the config plumbing."""
    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config


def _omni_model_config(**fields) -> OmniModelConfig:
    """An OmniModelConfig without the ModelConfig machinery behind it."""
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in fields.items():
        object.__setattr__(config, name, value)
    return config


def test_base_runner_inheritance():
    # The connector mixin belongs to the leaf runners, as upstream.
    assert issubclass(OmniTPUModelRunner, TPUModelRunner)
    assert not issubclass(OmniTPUModelRunner, OmniConnectorModelRunnerMixin)


def test_base_runner_records_stage_from_omni_config():
    config = _omni_model_config(model_stage="talker", stage_id=1)
    with patch.object(TPUModelRunner, "__init__", _fake_base_init):
        runner = OmniTPUModelRunner(SimpleNamespace(model_config=config), "tpu")

    assert runner.model_stage == "talker"
    assert runner.stage_id == 1


def test_base_runner_without_omni_config_has_no_stage():
    # A plain vLLM ModelConfig carries no stage: the runner reports None and
    # the leaf runners treat it like a single-stage thinker.
    plain = SimpleNamespace(model_config=SimpleNamespace())
    with patch.object(TPUModelRunner, "__init__", _fake_base_init):
        runner = OmniTPUModelRunner(plain, "tpu")

    assert runner.model_stage is None
    assert runner.stage_id is None
