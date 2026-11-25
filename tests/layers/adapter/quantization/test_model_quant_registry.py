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
import vllm.config.model as vllm_model_config
import vllm.model_executor.layers.quantization as vllm_quantization
from vllm.config import ModelConfig
from vllm.platforms import current_platform

from vllm_torchtpu.layers.adapter import quantization
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
)


class _ConfigA(VllmUnquantizedConfig):
    @classmethod
    def get_name(cls) -> str:
        return "model_fmt"


class _ConfigB(_ConfigA):
    pass


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(quantization, "_MODEL_QUANT_CONFIGS", {})
    platform_cls = type(current_platform)
    monkeypatch.setattr(
        platform_cls,
        "supported_quantization",
        list(platform_cls.supported_quantization),
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
    # ModelConfig reads the module through a LazyLoader that copies the module
    # dict on first use; point it at the module so it sees this test's lists.
    monkeypatch.setattr(vllm_model_config, "me_quant", vllm_quantization)
    return quantization._MODEL_QUANT_CONFIGS


def test_registration_adds_supported_method_once(registry) -> None:
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    quantization.register_tpu_quant_config("model_fmt", _ConfigA)
    assert current_platform.supported_quantization.count("model_fmt") == 1
    assert vllm_quantization.QUANTIZATION_METHODS.count("model_fmt") == 1
    assert vllm_quantization.get_quantization_config("tpu-model_fmt") is _ConfigA


def _verify_quantization(quant_method: str) -> str:
    """Run vLLM's own checkpoint name check on a config declaring the method."""
    model_config = SimpleNamespace(
        quantization=None,
        model_arch_config=SimpleNamespace(
            quantization_config={"quant_method": quant_method}
        ),
        hf_config=SimpleNamespace(),
        allow_deprecated_quantization=False,
    )
    ModelConfig._verify_quantization(model_config)  # pyrefly: ignore[bad-argument-type]
    return model_config.quantization


def test_unlisted_method_passes_vllm_name_check(registry) -> None:
    assert "model_fmt" not in vllm_quantization.QUANTIZATION_METHODS
    with pytest.raises(ValueError, match="Unknown quantization method"):
        _verify_quantization("model_fmt")

    quantization.register_tpu_quant_config("model_fmt", _ConfigA)

    assert vllm_quantization.get_quantization_config("model_fmt") is _ConfigA
    assert _verify_quantization("model_fmt") == "model_fmt"


def test_listed_method_keeps_vllm_class(registry) -> None:
    class _Mxfp8(_ConfigA):
        @classmethod
        def get_name(cls) -> str:
            return "mxfp8"

    vllm_cls = vllm_quantization.get_quantization_config("mxfp8")
    quantization.register_tpu_quant_config("mxfp8", _Mxfp8)

    assert vllm_quantization.get_quantization_config("mxfp8") is vllm_cls
    assert vllm_quantization.get_quantization_config("tpu-mxfp8") is _Mxfp8
    assert current_platform.supported_quantization.count("mxfp8") == 1
    assert _verify_quantization("mxfp8") == "mxfp8"


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

    # The built-in unquantized config sits under None in the table but is
    # registered with vLLM as tpu-unquantized, its get_name().
    class _Unquantized(_ConfigA):
        @classmethod
        def get_name(cls) -> str:
            return "unquantized"

    with pytest.raises(ValueError, match="already served by VllmUnquantizedConfig"):
        quantization.register_tpu_quant_config("unquantized", _Unquantized)
    assert (
        vllm_quantization.get_quantization_config("tpu-unquantized")
        is VllmUnquantizedConfig
    )
    assert registry == {}


def test_name_mismatch_is_rejected(registry) -> None:
    with pytest.raises(ValueError, match="get_name"):
        quantization.register_tpu_quant_config("other_fmt", _ConfigA)
