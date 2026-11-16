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

from vllm_torchtpu.models.qwen4_exp.experimental import hooks


def test_selector_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", raising=False)
    assert not hooks.enabled()


@pytest.mark.parametrize(
    ("value", "on"), [("1", True), ("true", True), ("True", True), ("0", False)]
)
def test_selector_values(monkeypatch, value, on):
    monkeypatch.setenv("VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", value)
    assert hooks.enabled() is on


def test_register_is_noop_without_selector(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", raising=False)
    with mock.patch(
        "vllm.model_executor.models.ModelRegistry.register_model"
    ) as register:
        hooks.register()
    register.assert_not_called()


def test_register_takes_over_qwen4_exp_architectures(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", "1")
    with (
        mock.patch(
            "vllm.model_executor.models.ModelRegistry.register_model"
        ) as register,
        mock.patch.dict(
            "vllm.model_executor.models.config.MODELS_CONFIG_MAP", clear=True
        ),
        mock.patch(
            "vllm_torchtpu.compilation.tpu_compiler.register_compile_cache_key"
        ) as cache_key,
        mock.patch(
            "vllm_torchtpu.runner.model_adapters.register_model_adapter"
        ) as model_adapter,
        mock.patch(
            "vllm_torchtpu.worker.cpu_pinning.register_cpu_pinning"
        ) as cpu_pinning,
    ):
        hooks.register()
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP

        config_map = dict(MODELS_CONFIG_MAP)
    registered = {call.args[0]: call.args[1] for call in register.call_args_list}
    assert set(registered) == {*hooks.QWEN4_EXP_ARCHITECTURES, "Qwen4ExpMTP"}
    assert set(config_map) == set(hooks.QWEN4_EXP_ARCHITECTURES)
    assert cache_key.call_args.args == ("qwen4_exp",)
    assert cpu_pinning.call_args.args == ("numa", *hooks.QWEN4_EXP_ARCHITECTURES)
    adapter_archs, make_adapter = model_adapter.call_args.args
    assert set(adapter_archs) == set(hooks.QWEN4_EXP_ARCHITECTURES)
    assert make_adapter is hooks._make_adapter
    factors = cache_key.call_args.kwargs["env_factors"]()
    assert set(factors) == {"model_plan"}
    for target in registered.values():
        assert target.startswith("vllm_torchtpu.models.qwen4_exp.experimental.")
