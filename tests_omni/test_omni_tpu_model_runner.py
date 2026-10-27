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
"""Unit tests for src/vllm_torchtpu/omni/omni_tpu_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.omni


def _make_bare_runner(cls=OmniTPUModelRunner):
    """Instantiate a runner class without invoking TPUModelRunner.__init__."""
    runner = cls.__new__(cls)
    runner.init_omni_connectors = MagicMock()
    return runner


def test_omni_tpu_model_runner_inheritance():
    # The connector mixin lives on the shared base so every Omni runner (AR
    # now, generation next) gets one connector lifecycle.
    assert issubclass(OmniTPUModelRunner, TPUModelRunner)
    assert issubclass(OmniTPUModelRunner, OmniConnectorModelRunnerMixin)
    assert OmniTPUModelRunner._omni_connector_initialized is False


@pytest.mark.parametrize(
    "model_config",
    [
        SimpleNamespace(requires_full_payload_input=True),
        SimpleNamespace(custom_process_next_stage_input_func="pkg.mod.func"),
        SimpleNamespace(stage_connector_config={"extra": {"role": "sender"}}),
    ],
)
def test_load_model_inits_connectors_when_needed(model_config):
    runner = _make_bare_runner()
    runner.model_config = model_config

    with patch.object(TPUModelRunner, "load_model") as mock_super_load:
        runner.load_model()
        mock_super_load.assert_called_once_with()

    runner.init_omni_connectors.assert_called_once_with(model_config)


def test_load_model_skips_connectors_when_not_needed():
    runner = _make_bare_runner()
    # A stage that neither consumes a full payload, produces inter-stage data,
    # nor has an explicit connector role must not start connector threads.
    runner.model_config = SimpleNamespace(
        requires_full_payload_input=False,
        custom_process_next_stage_input_func=None,
        stage_connector_config=None,
    )

    with patch.object(TPUModelRunner, "load_model") as mock_super_load:
        runner.load_model()
        mock_super_load.assert_called_once_with()

    runner.init_omni_connectors.assert_not_called()
    assert runner._omni_connector_initialized is False


def test_update_states_cleans_finished_requests():
    runner = _make_bare_runner()
    runner._omni_connector_initialized = True
    runner.cleanup_finished_request = MagicMock()
    sched_out = SimpleNamespace(finished_req_ids=["req-1"])

    with patch.object(TPUModelRunner, "_update_states", return_value="updated"):
        assert runner._update_states(sched_out) == "updated"

    runner.cleanup_finished_request.assert_called_once_with("req-1")


def test_update_states_without_connectors():
    runner = _make_bare_runner()
    runner.cleanup_finished_request = MagicMock()
    sched_out = SimpleNamespace(finished_req_ids=["req-1"])

    with patch.object(TPUModelRunner, "_update_states", return_value="updated"):
        assert runner._update_states(sched_out) == "updated"

    runner.cleanup_finished_request.assert_not_called()
