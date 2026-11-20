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
import torch
import vllm.model_executor.layers.quantization as vllm_quantization

from vllm_torchtpu.layers.adapter import quantization
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


class _ConfigA(VllmUnquantizedConfig):
    @classmethod
    def get_name(cls) -> str:
        return "model_fmt"


class _ConfigB(_ConfigA):
    pass


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(quantization, "_MODEL_QUANT_CONFIGS", {})
    monkeypatch.setattr(
        TpuPlatform, "supported_quantization", list(TpuPlatform.supported_quantization)
    )
    monkeypatch.setattr(
        vllm_quantization,
        "QUANTIZATION_METHODS",
        list(vllm_quantization.QUANTIZATION_METHODS),
    )
    monkeypatch.setattr(
        vllm_quantization,
        "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG",
        dict(vllm_quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG),
    )
    return quantization._MODEL_QUANT_CONFIGS


def test_registration_adds_supported_method_once(registry) -> None:
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    assert TpuPlatform.supported_quantization.count("model_fmt") == 1
    assert vllm_quantization.get_quantization_config("tpu-model_fmt") is _ConfigA


def test_registered_method_is_dispatched(registry) -> None:
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization="model_fmt",
            quantization_config=None,
            hf_config=SimpleNamespace(quantization_config={"quant_method": "x"}),
            dtype=torch.bfloat16,
            model="unused",
            revision=None,
        ),
        load_config=None,
        additional_config={},
    )

    config = quantization.get_tpu_quantization_config(vllm_config)

    assert type(config) is _ConfigA
    assert _ConfigA.vllm_config is vllm_config


def test_conflicting_registration_is_rejected(registry) -> None:
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    with pytest.raises(ValueError, match="already served by _ConfigA"):
        quantization.register_tpu_quant_config("model_fmt", _ConfigB)


def test_builtin_method_conflict_is_rejected(registry) -> None:
    class _Fp8(_ConfigA):
        @classmethod
        def get_name(cls) -> str:
            return "fp8"

    with pytest.raises(ValueError, match="already served by VllmFp8Config"):
        quantization.register_tpu_quant_config("fp8", _Fp8)
    quantization.register_tpu_quant_config("fp8", VllmFp8Config)  # same class: ok
    assert registry == {}


def test_name_mismatch_is_rejected(registry) -> None:
    with pytest.raises(ValueError, match="get_name"):
        quantization.register_tpu_quant_config("other_fmt", _ConfigA)
