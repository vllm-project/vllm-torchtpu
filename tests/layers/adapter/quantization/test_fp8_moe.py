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
"""Unit tests for VllmFp8MoEConfig (quantization="fp8_moe")."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config

from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8MoEConfig,
    VllmFp8MoEMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core import quant_methods
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


class FakeRoutedExperts(RoutedExperts):
    """Subclass RoutedExperts to satisfy isinstance checks without full vLLM setup."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.moe_parallel_config = MagicMock()


@pytest.fixture
def linear_layer():
    """A bare LinearBase instance for testing get_quant_method dispatch."""
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.output_size = 32
    layer.input_size = 64
    return layer


@pytest.fixture
def routed_experts_layer():
    """A FakeRoutedExperts instance for testing get_quant_method dispatch."""
    return FakeRoutedExperts()


@pytest.fixture
def attention_layer():
    """A non-linear, non-MoE module representing attention."""
    return torch.nn.Identity()


@pytest.fixture(autouse=True)
def mock_vllm_config(monkeypatch):
    """Ensure VllmQuantConfig.vllm_config is set for get_moe_config."""
    mock_cfg = MagicMock()
    mock_cfg.parallel_config.enable_expert_parallel = False
    mock_comp = MagicMock()
    mock_comp.custom_ops = ["all"]
    mock_comp.enabled_custom_ops = set()
    mock_comp.disabled_custom_ops = set()
    mock_cfg.compilation_config = mock_comp
    monkeypatch.setattr(VllmQuantConfig, "vllm_config", mock_cfg)
    monkeypatch.setattr("vllm.config.vllm._current_vllm_config", mock_cfg)


class TestFp8MoERegistration:
    def test_config_name(self):
        assert VllmFp8MoEConfig.get_name() == "fp8_moe"
        assert VllmFp8MoEConfig.get_name() == quant_methods.FP8_MOE

    def test_vllm_quantization_registry(self):
        with pytest.raises(ValueError, match="Invalid quantization method"):
            get_quantization_config("fp8_moe")
        assert (
            get_quantization_config(
                quant_methods.get_tpu_quant_method(quant_methods.FP8_MOE)
            )
            is VllmFp8MoEConfig
        )

    def test_platform_supported_quantization(self):
        assert "fp8_per_channel" in TpuPlatform.supported_quantization
        assert "fp8_moe" not in TpuPlatform.supported_quantization

    def test_get_tpu_quantization_config_moe_only_dict(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization import (
            get_tpu_quantization_config,
        )

        vllm_config = MagicMock()
        vllm_config.model_config.quantization = quant_methods.FP8_PER_CHANNEL
        vllm_config.model_config.quantization_config = {
            "linear": {"activation": None, "weight": None},
        }
        vllm_config.load_config = MagicMock()
        vllm_config.additional_config = {}

        mock_instance = VllmFp8MoEConfig.from_config({})
        mock_get_qc = MagicMock(return_value=mock_instance)
        monkeypatch.setattr(
            "vllm.config.VllmConfig.get_quantization_config", mock_get_qc
        )

        config = get_tpu_quantization_config(vllm_config)
        assert isinstance(config, VllmFp8MoEConfig)
        mock_get_qc.assert_called_once()
        passed_model_config = mock_get_qc.call_args[0][0]
        assert passed_model_config.quantization == "tpu-fp8_moe"

    def test_get_tpu_quantization_config_moe_only_linear_none(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization import (
            get_tpu_quantization_config,
        )

        vllm_config = MagicMock()
        vllm_config.model_config.quantization = quant_methods.FP8
        vllm_config.model_config.quantization_config = {"linear": None}
        vllm_config.load_config = MagicMock()
        vllm_config.additional_config = {}

        mock_instance = VllmFp8MoEConfig.from_config({})
        mock_get_qc = MagicMock(return_value=mock_instance)
        monkeypatch.setattr(
            "vllm.config.VllmConfig.get_quantization_config", mock_get_qc
        )

        config = get_tpu_quantization_config(vllm_config)
        assert isinstance(config, VllmFp8MoEConfig)
        mock_get_qc.assert_called_once()
        passed_model_config = mock_get_qc.call_args[0][0]
        assert passed_model_config.quantization == "tpu-fp8_moe"

    def test_get_tpu_quantization_config_moe_only_object_linear_none(self, monkeypatch):
        """Test QuantizationConfigArgs-like object with linear=None."""
        from vllm_torchtpu.layers.adapter.quantization import (
            get_tpu_quantization_config,
        )

        class FakeQuantConfigArgs:
            linear = None
            moe = None

        vllm_config = MagicMock()
        vllm_config.model_config.quantization = quant_methods.FP8_PER_CHANNEL
        vllm_config.model_config.quantization_config = FakeQuantConfigArgs()
        vllm_config.load_config = MagicMock()
        vllm_config.additional_config = {}

        mock_instance = VllmFp8MoEConfig.from_config({})
        mock_get_qc = MagicMock(return_value=mock_instance)
        monkeypatch.setattr(
            "vllm.config.VllmConfig.get_quantization_config", mock_get_qc
        )

        config = get_tpu_quantization_config(vllm_config)
        assert isinstance(config, VllmFp8MoEConfig)
        mock_get_qc.assert_called_once()
        passed_model_config = mock_get_qc.call_args[0][0]
        assert passed_model_config.quantization == "tpu-fp8_moe"

    def test_get_tpu_quantization_config_full_fp8(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization import (
            get_tpu_quantization_config,
        )

        vllm_config = MagicMock()
        vllm_config.model_config.quantization = quant_methods.FP8_PER_CHANNEL
        vllm_config.model_config.quantization_config = None
        vllm_config.load_config = MagicMock()
        vllm_config.additional_config = {}

        mock_instance = MagicMock()
        mock_get_qc = MagicMock(return_value=mock_instance)
        monkeypatch.setattr(
            "vllm.config.VllmConfig.get_quantization_config", mock_get_qc
        )

        config = get_tpu_quantization_config(vllm_config)
        assert config is mock_instance
        mock_get_qc.assert_called_once()
        passed_model_config = mock_get_qc.call_args[0][0]
        assert passed_model_config.quantization == "tpu-fp8"


class TestFp8MoEConfig:
    def test_from_config_defaults(self):
        cfg = VllmFp8MoEConfig.from_config({})
        assert cfg.get_name() == "fp8_moe"
        assert cfg.activation_scheme == "dynamic"
        assert not cfg.is_checkpoint_fp8_serialized
        assert not cfg.ignored_layers
        assert cfg.weight_block_size is None
        assert cfg.store_dtype is None

    def test_from_config_custom(self):
        cfg = VllmFp8MoEConfig.from_config(
            {
                "activation_scheme": "dynamic",
                "is_checkpoint_fp8_serialized": True,
                "ignored_layers": ["model.layers.0.mlp.experts"],
                "weight_block_size": [128, 128],
            }
        )
        assert cfg.activation_scheme == "dynamic"
        assert cfg.is_checkpoint_fp8_serialized is True
        assert cfg.ignored_layers == ["model.layers.0.mlp.experts"]
        assert cfg.weight_block_size == [128, 128]

    def test_from_config_modules_to_not_convert_alias(self):
        cfg = VllmFp8MoEConfig.from_config(
            {"modules_to_not_convert": ["expert_0", "expert_1"]}
        )
        assert cfg.ignored_layers == ["expert_0", "expert_1"]

    def test_from_config_rejects_store_dtype(self):
        with pytest.raises(NotImplementedError, match="store_dtype"):
            VllmFp8MoEConfig.from_config({"store_dtype": "mxfp4"})


class TestFp8MoEDispatch:
    def test_linear_layer_uses_unquantized_linear_method(self, linear_layer):
        cfg = VllmFp8MoEConfig.from_config({})
        method = cfg.get_quant_method(
            linear_layer, prefix="model.layers.0.self_attn.q_proj"
        )
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_attention_layer_returns_none(self, attention_layer):
        cfg = VllmFp8MoEConfig.from_config({})
        method = cfg.get_quant_method(
            attention_layer, prefix="model.layers.0.self_attn"
        )
        assert method is None

    def test_routed_experts_uses_fp8_moe_method(self, routed_experts_layer):
        cfg = VllmFp8MoEConfig.from_config({})
        method = cfg.get_quant_method(
            routed_experts_layer, prefix="model.layers.0.mlp.experts"
        )
        assert isinstance(method, VllmFp8MoEMethodTPU)

    def test_ignored_routed_experts_uses_unquantized_moe_method(
        self, routed_experts_layer
    ):
        prefix = "model.layers.0.mlp.experts"
        cfg = VllmFp8MoEConfig.from_config({"ignored_layers": [prefix]})
        method = cfg.get_quant_method(routed_experts_layer, prefix=prefix)
        assert isinstance(method, VllmUnquantizedFusedMoEMethod)
