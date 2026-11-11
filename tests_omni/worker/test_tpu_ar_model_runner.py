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
import torch
from vllm.sequence import IntermediateTensors
from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.omni


def _stage_config(**fields) -> OmniModelConfig:
    defaults = dict(
        model_stage="thinker",
        stage_id=0,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func=None,
        stage_connector_config=None,
    )
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in {**defaults, **fields}.items():
        object.__setattr__(config, name, value)
    return config


def _build(model_config=None, model=None) -> TPUARModelRunner:
    if model_config is None:
        model_config = _stage_config()

    def _fake_init(self, vllm_config, *args, **kwargs):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

    with patch.object(TPUModelRunner, "__init__", _fake_init):
        runner = TPUARModelRunner(SimpleNamespace(model_config=model_config), "tpu")
    if model is not None:
        with (
            patch(
                "vllm_torchtpu.omni.worker.tpu_ar_model_runner.maybe_patch_qwen3_omni_thinker"
            ) as mock_patch,
            patch.object(
                TPUModelRunner, "load_model", lambda self: setattr(self, "model", model)
            ),
        ):
            runner.load_model()
            mock_patch.assert_called_once_with(runner.model_config)
    return runner


class _StageModel:
    def __init__(self, result, *, has_preprocess=False, has_postprocess=False):
        self.result = result
        self.has_preprocess = has_preprocess
        self.has_postprocess = has_postprocess
        self.wrapped = []

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.result

    def make_omni_output(self, model_outputs, **kwargs) -> OmniOutput:
        self.wrapped.append(model_outputs)
        hidden = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"embed": {}})


def test_tpu_ar_model_runner_init_and_inheritance():
    assert TPUARModelRunner.__bases__ == (
        OmniTPUModelRunner,
        OmniConnectorModelRunnerMixin,
    )
    assert OmniConnectorModelRunnerMixin not in OmniTPUModelRunner.__mro__

    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    assert (runner.model_stage, runner.stage_id) == ("thinker", 0)
    plain = _build(SimpleNamespace())
    assert (plain.model_stage, plain.stage_id) == (None, None)

    with pytest.raises(NotImplementedError, match="connectors"):
        _build(
            _stage_config(custom_process_next_stage_input_func="pkg.mod.thinker2talker")
        )


def test_load_model_checks_per_step_hook_capabilities():
    thinker = _StageModel(torch.zeros(1, 4))
    assert _build(model=thinker).get_model() is thinker

    talker = _StageModel(torch.zeros(1, 4), has_preprocess=True, has_postprocess=True)
    with pytest.raises(NotImplementedError, match="has_preprocess, has_postprocess"):
        _build(model=talker)


def test_forward_model_normalizes_stage_outputs():
    ids, pos = torch.tensor([1, 2]), torch.tensor([0, 1])
    hidden = torch.arange(8.0).reshape(2, 4)
    staged = (hidden, {"layer_0": torch.zeros(2, 4)})

    for raw, expected_wrapped in [
        (hidden, [hidden]),
        ([hidden], [hidden]),
        (staged, [staged]),
        (OmniOutput(text_hidden_states=hidden), []),
    ]:
        runner = _build(model=(model := _StageModel(raw)))
        assert runner.forward_model(ids, pos) == (hidden, None)
        assert model.wrapped == expected_wrapped
        assert "intermediate_tensors" not in model.kwargs

    tensors = IntermediateTensors({"hidden_states": hidden})
    runner = _build(model=(model := _StageModel(tensors)))
    assert runner.forward_model(ids, pos, intermediate_tensors=tensors) == (
        tensors,
        None,
    )
    assert model.kwargs["intermediate_tensors"] is tensors
