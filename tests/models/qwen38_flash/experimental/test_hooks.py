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
"""Registration hook of experimental Qwen3.8-Flash."""

from unittest import mock

import pytest

from vllm_torchtpu.models.qwen38_flash.experimental import hooks, settings


def test_selector_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", raising=False)
    assert settings.VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL is False
    assert not hooks.enabled()


def test_register_is_noop_without_selector(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", raising=False)
    with mock.patch(
        "vllm.model_executor.models.ModelRegistry.register_model"
    ) as register:
        hooks.register()
    register.assert_not_called()


def test_register_takes_over_qwen4_exp_architectures(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", "1")
    with mock.patch(
        "vllm.model_executor.models.ModelRegistry.register_model"
    ) as register:
        hooks.register()
    registered = {call.args[0]: call.args[1] for call in register.call_args_list}
    assert set(registered) == set(hooks.QWEN4_EXP_ARCHITECTURES)
    for target in registered.values():
        assert target.startswith("vllm_torchtpu.models.qwen38_flash.experimental.")


def test_unknown_setting_raises():
    with pytest.raises(AttributeError):
        _ = settings.NOT_A_QWEN38_FLAG

