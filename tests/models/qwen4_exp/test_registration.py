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
"""models/qwen4_exp register(): the base registration, then the experimental
tier's hand-off."""

from unittest import mock

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

import vllm_torchtpu.models.qwen4_exp as pkg
from vllm_torchtpu.models.qwen4_exp.config import TPUQwen4ExpConfig
from vllm_torchtpu.models.qwen4_exp.experimental import hooks

_ARCHS = ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")


def test_register_base_then_experimental():
    order = []
    with (
        mock.patch.object(
            ModelRegistry,
            "register_model",
            side_effect=lambda arch, _: order.append(("base", arch)),
        ),
        mock.patch.dict(MODELS_CONFIG_MAP),
        mock.patch.object(
            hooks, "register", side_effect=lambda: order.append(("hooks", None))
        ),
    ):
        pkg.register()
        for arch in _ARCHS:
            assert MODELS_CONFIG_MAP[arch] is TPUQwen4ExpConfig
    assert order == [("base", arch) for arch in _ARCHS] + [("hooks", None)]


def test_register_models_leaves_qwen4_exp_to_its_directory():
    from vllm_torchtpu.models.vllm import register_models

    with (
        mock.patch.object(ModelRegistry, "register_model") as register_model,
        mock.patch.dict(MODELS_CONFIG_MAP),
    ):
        register_models()
    registered = {call.args[0] for call in register_model.call_args_list}
    assert not registered & set(_ARCHS)
