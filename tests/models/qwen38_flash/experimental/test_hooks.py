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
            "vllm_torchtpu.platforms.tpu_platform.register_tpu_kv_connector"
        ) as kv_connector,
        mock.patch(
            "vllm_torchtpu.offload.model_owned_pool.register_model_owned_pool"
        ) as offload_pool,
        mock.patch(
            "vllm_torchtpu.runner.model_adapters.register_model_adapter"
        ) as model_adapter,
    ):
        hooks.register()
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP

        config_map = dict(MODELS_CONFIG_MAP)
    registered = {call.args[0]: call.args[1] for call in register.call_args_list}
    assert set(registered) == {*hooks.QWEN4_EXP_ARCHITECTURES, "Qwen4ExpMTP"}
    assert set(config_map) == set(hooks.QWEN4_EXP_ARCHITECTURES)
    assert cache_key.call_args.args == ("qwen38_flash",)
    assert set(offload_pool.call_args.args[0]) == set(hooks.QWEN4_EXP_ARCHITECTURES)
    adapter_archs, make_adapter = model_adapter.call_args.args
    assert set(adapter_archs) == set(hooks.QWEN4_EXP_ARCHITECTURES)
    assert make_adapter is hooks._make_adapter
    kv_connector.assert_called_once_with(
        "Qwen4ExpRaidenConnector",
        "vllm_torchtpu.models.qwen38_flash.experimental.pd_connector",
    )
    factors = cache_key.call_args.kwargs["env_factors"]()
    assert "QWEN4_EXP_POOL_SIZING" not in factors
    assert "VLLM_TPU_QSA_FUSED_SPARSE_DECODE" in factors
    for target in registered.values():
        assert target.startswith("vllm_torchtpu.models.qwen38_flash.experimental.")


def test_unknown_setting_raises():
    with pytest.raises(AttributeError):
        _ = settings.NOT_A_QWEN38_FLAG
