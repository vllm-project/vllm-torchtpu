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
"""Unit tests for src/vllm_torchtpu/omni/worker/tpu_ar_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm_omni.config import OmniModelConfig
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.omni

# The connector fields needs_omni_connector() looks at, all off.
_SINGLE_STAGE = dict(
    requires_full_payload_input=False,
    custom_process_next_stage_input_func=None,
    stage_connector_config=None,
)


def _fake_base_init(self, vllm_config, *args, **kwargs):
    """Stand-in for TPUModelRunner.__init__: only the config plumbing."""
    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config


def _stage_config(**fields) -> OmniModelConfig:
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in {**_SINGLE_STAGE, **fields}.items():
        object.__setattr__(config, name, value)
    return config


def _build(model_config) -> TPUARModelRunner:
    with patch.object(TPUModelRunner, "__init__", _fake_base_init):
        return TPUARModelRunner(SimpleNamespace(model_config=model_config), "tpu")


def test_tpu_ar_model_runner_inheritance():
    # Same shape as upstream GPUARModelRunner: the Omni base first, the
    # connector mixin on the leaf.
    assert TPUARModelRunner.__bases__ == (
        OmniTPUModelRunner,
        OmniConnectorModelRunnerMixin,
    )
    assert frozenset({"thinker"}) == TPUARModelRunner.SUPPORTED_STAGES


def test_thinker_only_stage_is_accepted():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    assert runner.model_stage == "thinker"
    assert runner.stage_id == 0


def test_plain_vllm_config_is_accepted():
    # No stage metadata at all: a single-stage deployment outside vLLM-Omni.
    runner = _build(SimpleNamespace())
    assert runner.model_stage is None


def test_talker_stage_is_refused_at_construction():
    with pytest.raises(NotImplementedError, match="talker_mtp"):
        _build(_stage_config(model_stage="talker", stage_id=1))


@pytest.mark.parametrize(
    "connector_fields",
    [
        dict(requires_full_payload_input=True),
        dict(
            custom_process_next_stage_input_func="pkg.mod.thinker2talker_full_payload"
        ),
        dict(stage_connector_config={"extra": {"role": "sender"}}),
    ],
    ids=["full_payload_consumer", "producer_hook", "explicit_role"],
)
def test_multi_stage_thinker_is_refused_at_construction(connector_fields):
    # The thinker of a multi-stage pipeline produces inter-stage payloads and
    # therefore needs connectors, which are not wired on TPU yet.
    with pytest.raises(NotImplementedError, match="connectors"):
        _build(_stage_config(model_stage="thinker", stage_id=0, **connector_fields))
