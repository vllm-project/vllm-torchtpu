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
"""The Qwen3.5 rule that installs the fused GDN gated norm + FP8 quantization
after loading: a Qwen3.5 target routes its model through the kernel bridge,
and any other model is left as it was."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.models.qwen3_5.kernels import gated_norm_quant_op
from vllm_torchtpu.runner import tpu_runner


@pytest.mark.parametrize(
    "architectures, routed",
    [
        (None, False),
        (["Qwen3NextForCausalLM"], False),
        (["Qwen3_5ForConditionalGeneration"], False),
        (["Qwen3_5MoeForConditionalGeneration"], True),
    ],
)
def test_the_runner_routes_qwen3_5_models_only(monkeypatch, architectures, routed):
    install = Mock(return_value=3)
    monkeypatch.setattr(gated_norm_quant_op, "install_if_supported", install)
    model = torch.nn.Linear(2, 2)
    model_config = (
        None if architectures is None else SimpleNamespace(architectures=architectures)
    )
    got = tpu_runner._install_gdn_fused_norm_quant(model, model_config)
    if routed:
        install.assert_called_once_with(model)
        assert got == 3
    else:
        install.assert_not_called()
        assert got == 0
