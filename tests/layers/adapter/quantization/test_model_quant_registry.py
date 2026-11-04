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
"""Model directories register checkpoint quant methods with the TPU dispatch."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import quantization
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


class _ConfigA(VllmQuantConfig):
    pass


class _ConfigB(VllmQuantConfig):
    pass


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(quantization, "_MODEL_QUANT_CONFIGS", {})
    monkeypatch.setattr(
        TpuPlatform, "supported_quantization", list(TpuPlatform.supported_quantization)
    )
    return quantization._MODEL_QUANT_CONFIGS


def test_registered_method_passes_platform_check(registry) -> None:
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    assert TpuPlatform.supported_quantization.count("model_fmt") == 1


def test_registered_method_is_dispatched(registry, monkeypatch) -> None:
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)  # idempotent
    seen = {}

    def fake_get_quantization_config(model_config, load_config):
        seen["method"] = model_config.quantization
        return object()

    monkeypatch.setattr(
        quantization.VllmConfig,
        "get_quantization_config",
        staticmethod(fake_get_quantization_config),
    )
    monkeypatch.setattr(_ConfigA, "get_name", classmethod(lambda cls: "model_fmt"))
    monkeypatch.setattr(quantization, "attach_online_fp8", lambda config, _: config)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(quantization="model_fmt"), load_config=None
    )

    quantization.get_tpu_quantization_config(vllm_config)

    assert seen["method"] == "tpu-model_fmt"
    assert _ConfigA.vllm_config is vllm_config


def test_conflicting_registration_is_rejected(registry) -> None:
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    with pytest.raises(ValueError, match="already served by _ConfigA"):
        quantization.register_tpu_quant_config("model_fmt", _ConfigB)
