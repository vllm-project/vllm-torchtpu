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
"""Unit tests for src/vllm_torchtpu/omni/omni_tpu_ar_model_runner.py."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.omni_tpu_ar_model_runner import OmniTPUARModelRunner
from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput

pytestmark = pytest.mark.omni


def _make_bare_ar_runner(connector_initialized=True):
    runner = OmniTPUARModelRunner.__new__(OmniTPUARModelRunner)
    runner._omni_connector_initialized = connector_initialized
    # The real attach reads connector state this bare runner does not have;
    # stand in with a tag so the tests can see where the output was routed.
    runner.attach_omni_connector_output = lambda output: ("connector_attached", output)
    return runner


def _async_output(inner):
    return AsyncTPUModelRunnerOutput(
        model_runner_output=inner,
        copy_state=None,
        discard_sampled_tokens_req_indices=[],
    )


def test_omni_tpu_ar_model_runner_inheritance():
    # The AR runner only adds the AR hooks; the mixin comes from the base.
    assert OmniTPUARModelRunner.__bases__ == (OmniTPUModelRunner,)
    assert issubclass(OmniTPUARModelRunner, OmniConnectorModelRunnerMixin)


def test_sample_tokens_sync_output():
    runner = _make_bare_ar_runner()
    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value="raw_tokens"):
        assert runner.sample_tokens() == ("connector_attached", "raw_tokens")


def test_sample_tokens_without_connectors_is_passthrough():
    runner = _make_bare_ar_runner(connector_initialized=False)
    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value="raw_tokens"):
        assert runner.sample_tokens() == "raw_tokens"


def test_sample_tokens_async_output_attaches_to_inner_output():
    """Under async scheduling the engine reads get_output(), not the wrapper."""
    runner = _make_bare_ar_runner()
    inner = SimpleNamespace(sampled_token_ids=None)
    async_output = _async_output(inner)

    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value=async_output):
        result = runner.sample_tokens()

    assert result is async_output
    assert result._model_runner_output == ("connector_attached", inner)


def test_sample_tokens_none_is_passthrough():
    runner = _make_bare_ar_runner()
    with patch.object(OmniTPUModelRunner, "sample_tokens", return_value=None):
        assert runner.sample_tokens() is None


def test_execute_model_two_phase_path_defers_to_sample_tokens():
    runner = _make_bare_ar_runner()
    sched_out = SimpleNamespace(total_num_scheduled_tokens=4)
    with patch.object(OmniTPUModelRunner, "execute_model", return_value=None) as m:
        assert runner.execute_model(sched_out) is None
    m.assert_called_once_with(sched_out, None)


def test_execute_model_zero_token_early_return_attaches_connector_output():
    """A zero-token step returns from execute_model and skips sample_tokens()."""
    runner = _make_bare_ar_runner()
    sched_out = SimpleNamespace(total_num_scheduled_tokens=0)
    with patch.object(OmniTPUModelRunner, "execute_model", return_value="empty_output"):
        assert runner.execute_model(sched_out) == ("connector_attached", "empty_output")
