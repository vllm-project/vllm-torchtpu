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
"""Unit tests for Compressed Tensors quantization config and dispatch."""

from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
    CompressedTensorsScheme,
)

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    VllmCompressedTensorsConfig,
    _build_fp8_config,
    _build_fp8_linear_method,
    _is_int4_w4aN,
    _is_weight_fp8,
    _raise_not_implemented,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quant_methods import (
    COMPRESSED_TENSORS,
    get_tpu_quant_method,
)


@pytest.fixture
def linear_layer():
    """A bare LinearBase instance that bypasses tensor-parallel initialization."""
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.output_size = 32
    layer.input_size = 64
    return layer


class FakeRoutedExperts(RoutedExperts):
    """Subclass RoutedExperts to satisfy isinstance checks."""

    def __init__(self, experts_per_token: int = 2):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.experts_per_token = experts_per_token
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = False
        self.use_grouped_topk = False


class TestCompressedTensorsHelpers:
    """Unit tests for standalone helper functions in compressed_tensors module."""

    def test_is_weight_fp8(self):
        # None quant args
        assert not _is_weight_fp8(None)

        # 8-bit float -> True
        fp8_args = QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT)
        assert _is_weight_fp8(fp8_args)

        # 8-bit int -> False
        int8_args = QuantizationArgs(num_bits=8, type=QuantizationType.INT)
        assert not _is_weight_fp8(int8_args)

        # 4-bit float -> False
        fp4_args = QuantizationArgs(num_bits=4, type=QuantizationType.FLOAT)
        assert not _is_weight_fp8(fp4_args)

    def test_is_int4_w4aN(self):
        # None quant args
        assert not _is_int4_w4aN(None)

        # 4-bit int, GROUP strategy, static -> True
        w4_group = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            dynamic=False,
        )
        assert _is_int4_w4aN(w4_group)

        # 4-bit int, CHANNEL strategy, static -> True
        w4_channel = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            dynamic=False,
        )
        assert _is_int4_w4aN(w4_channel)

        # 4-bit int, TENSOR strategy (not group/channel) -> False
        w4_tensor = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
            dynamic=False,
        )
        assert not _is_int4_w4aN(w4_tensor)

        # 4-bit int, dynamic -> False
        w4_dynamic = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            dynamic=True,
        )
        assert not _is_int4_w4aN(w4_dynamic)

        # 8-bit int -> False
        w8_group = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            dynamic=False,
        )
        assert not _is_int4_w4aN(w8_group)

        # 4-bit float -> False
        w4_float = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            dynamic=False,
        )
        assert not _is_int4_w4aN(w4_float)

    def test_build_fp8_config_defaults(self):
        weight_quant = QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT)
        fp8_config = _build_fp8_config(weight_quant, None)

        assert isinstance(fp8_config, VllmFp8Config)
        assert fp8_config.is_checkpoint_fp8_serialized is True
        assert fp8_config.activation_scheme == "dynamic"
        assert fp8_config.weight_block_size is None
        assert fp8_config.is_channel_quant is False

    def test_build_fp8_config_dynamic_and_static_activations(self):
        weight_quant = QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT)

        # Dynamic activations
        input_dynamic = QuantizationArgs(
            num_bits=8, type=QuantizationType.FLOAT, dynamic=True
        )
        cfg_dynamic = _build_fp8_config(weight_quant, input_dynamic)
        assert cfg_dynamic.activation_scheme == "dynamic"

        # Static activations
        input_static = QuantizationArgs(
            num_bits=8, type=QuantizationType.FLOAT, dynamic=False
        )
        cfg_static = _build_fp8_config(weight_quant, input_static)
        assert cfg_static.activation_scheme == "static"

    def test_build_fp8_config_block_and_channel_strategy(self):
        # BLOCK strategy
        weight_block = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.BLOCK,
            block_structure=[128, 128],
        )
        cfg_block = _build_fp8_config(weight_block, None)
        assert cfg_block.weight_block_size == [128, 128]
        assert cfg_block.is_channel_quant is False

        # CHANNEL strategy
        weight_channel = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.CHANNEL,
        )
        cfg_channel = _build_fp8_config(weight_channel, None)
        assert cfg_channel.weight_block_size is None
        assert cfg_channel.is_channel_quant is True

    def test_build_fp8_linear_method(self, linear_layer):
        weight_quant = QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT)
        method = _build_fp8_linear_method(linear_layer, weight_quant, None)

        assert isinstance(method, VllmFp8LinearMethodTPU)
        assert isinstance(method.quant_config, VllmFp8Config)
        assert method.quant_config.activation_scheme == "dynamic"

    def test_raise_not_implemented(self):
        with pytest.raises(
            NotImplementedError,
            match=(
                "Compressed-tensors scheme 'SampleScheme' is not implemented yet in "
                "vllm-torchtpu."
            ),
        ):
            _raise_not_implemented("SampleScheme")


class TestCompressedTensorsConfig:
    """Unit tests for VllmCompressedTensorsConfig registration and methods."""

    def test_config_name_and_registration(self):
        assert VllmCompressedTensorsConfig.get_name() == COMPRESSED_TENSORS
        registered_cls = get_quantization_config(
            get_tpu_quant_method(COMPRESSED_TENSORS)
        )
        assert registered_cls is VllmCompressedTensorsConfig

    def test_set_configs_propagates_to_fp8_config(self):
        mock_vllm_config = MagicMock()
        orig_ct_config = VllmCompressedTensorsConfig.vllm_config
        orig_fp8_config = VllmFp8Config.vllm_config
        try:
            VllmCompressedTensorsConfig.set_configs(mock_vllm_config)
            assert VllmCompressedTensorsConfig.vllm_config is mock_vllm_config
            assert VllmFp8Config.vllm_config is mock_vllm_config
        finally:
            VllmCompressedTensorsConfig.vllm_config = orig_ct_config
            VllmFp8Config.vllm_config = orig_fp8_config

    def test_get_scheme_empty_target_scheme_map(self, linear_layer):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={}, ignore=[], quant_format="dense"
        )
        assert cfg.get_scheme(linear_layer, "model.layers.0.qkv_proj") is None

    def test_get_scheme_no_matched_target(self, linear_layer):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"other.layer": {"weights": None}},
            ignore=[],
            quant_format="dense",
        )
        assert cfg.get_scheme(linear_layer, "model.layers.0.qkv_proj") is None

    def test_get_scheme_no_weights_in_matched_target(self, linear_layer):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"model.layers.0.qkv_proj": {}},
            ignore=[],
            quant_format="dense",
        )
        assert cfg.get_scheme(linear_layer, "model.layers.0.qkv_proj") is None

    def test_get_scheme_unsupported_w4a8_fp8_raises(self, linear_layer):
        weight_quant = QuantizationArgs(
            num_bits=4,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
            symmetric=True,
            dynamic=False,
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        )
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={
                "proj": {"weights": weight_quant, "input_activations": input_quant}
            },
            ignore=[],
            quant_format="dense",
        )
        with pytest.raises(
            NotImplementedError,
            match=(
                "Compressed-tensors scheme 'VllmCompressedTensorsW4A8Fp8' is not "
                "implemented yet"
            ),
        ):
            cfg.get_scheme(linear_layer, "proj")

    def test_get_scheme_unsupported_nvfp4_raises(self, linear_layer):
        weight_quant = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=16,
            symmetric=True,
        )
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"proj": {"weights": weight_quant}},
            ignore=[],
            quant_format="dense",
        )
        with pytest.raises(
            NotImplementedError,
            match=(
                "Compressed-tensors scheme 'VllmCompressedTensorsW4A4Fp4' is not "
                "implemented yet"
            ),
        ):
            cfg.get_scheme(linear_layer, "proj")

    def test_get_scheme_unsupported_w8a8_fp8_raises(self, linear_layer):
        weight_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
            dynamic=True,
        )
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={
                "proj": {"weights": weight_quant, "input_activations": input_quant}
            },
            ignore=[],
            quant_format="dense",
        )
        with pytest.raises(
            NotImplementedError,
            match=(
                "Compressed-tensors scheme 'VllmCompressedTensorsW8A8Fp8' is not "
                "implemented yet"
            ),
        ):
            cfg.get_scheme(linear_layer, "proj")

    def test_get_scheme_unsupported_dynamic_token_w8a8_int8_raises(self, linear_layer):
        weight_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        )
        input_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        )
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={
                "proj": {"weights": weight_quant, "input_activations": input_quant}
            },
            ignore=[],
            quant_format="dense",
        )
        with pytest.raises(
            NotImplementedError,
            match=(
                "Compressed-tensors scheme 'VllmCompressedTensorsW8A8Int8' is not "
                "implemented yet"
            ),
        ):
            cfg.get_scheme(linear_layer, "proj")

    def test_get_scheme_unrecognized_scheme_raises(self, linear_layer):
        weight_quant = QuantizationArgs(
            num_bits=16, type=QuantizationType.INT, strategy=QuantizationStrategy.TENSOR
        )
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"proj": {"weights": weight_quant}},
            ignore=[],
            quant_format="dense",
        )
        with pytest.raises(
            NotImplementedError,
            match="No compressed-tensors compatible scheme was found for layer proj",
        ):
            cfg.get_scheme(linear_layer, "proj")


class TestCompressedTensorsQuantMethod:
    """Unit tests for VllmCompressedTensorsConfig.get_quant_method dispatch."""

    def test_get_quant_method_ignored_layer(self, linear_layer):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={}, ignore=["model.layers.0.skip"], quant_format="dense"
        )
        method = cfg.get_quant_method(linear_layer, "model.layers.0.skip")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_get_quant_method_linear_no_scheme_dict(self, linear_layer):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={}, ignore=[], quant_format="dense"
        )
        method = cfg.get_quant_method(linear_layer, "model.layers.0.linear")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_get_quant_method_linear_no_weights(self, linear_layer):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"proj": {}}, ignore=[], quant_format="dense"
        )
        method = cfg.get_quant_method(linear_layer, "proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_get_quant_method_linear_fp8(self, linear_layer):
        weight_quant = QuantizationArgs(num_bits=8, type=QuantizationType.FLOAT)
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"proj": {"weights": weight_quant}},
            ignore=[],
            quant_format="dense",
        )
        method = cfg.get_quant_method(linear_layer, "proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)

    def test_get_quant_method_linear_scheme_none_fallback(
        self, linear_layer, monkeypatch
    ):
        weight_quant = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
        )
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"proj": {"weights": weight_quant}},
            ignore=[],
            quant_format="dense",
        )
        monkeypatch.setattr(cfg, "get_scheme", lambda layer, layer_name: None)
        method = cfg.get_quant_method(linear_layer, "proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_get_quant_method_linear_with_scheme(self, linear_layer, monkeypatch):
        weight_quant = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=128,
        )
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={"proj": {"weights": weight_quant}},
            ignore=[],
            quant_format="dense",
        )
        mock_scheme = MagicMock(spec=CompressedTensorsScheme)
        monkeypatch.setattr(cfg, "get_scheme", lambda layer, layer_name: mock_scheme)
        method = cfg.get_quant_method(linear_layer, "proj")
        assert isinstance(method, CompressedTensorsLinearMethod)
        assert linear_layer.scheme is mock_scheme

    def test_get_quant_method_moe_delegates_to_ct_moe_method(self, monkeypatch):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={}, ignore=[], quant_format="dense"
        )
        layer = FakeRoutedExperts()
        mock_moe_method = MagicMock()
        mock_moe_config = MagicMock()

        monkeypatch.setattr(cfg, "get_moe_config", lambda layer_: mock_moe_config)
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors.VllmCompressedTensorsMoEMethod.get_moe_method",
            lambda c, layer_, layer_name: mock_moe_method,
        )

        method = cfg.get_quant_method(layer, "model.layers.0.block")
        assert method is mock_moe_method
        assert layer.moe_config is mock_moe_config

    def test_get_quant_method_attention_returns_none(self):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={}, ignore=[], quant_format="dense"
        )
        attn = Attention.__new__(Attention)
        torch.nn.Module.__init__(attn)
        method = cfg.get_quant_method(attn, "model.layers.0.self_attn")
        assert method is None

    def test_get_quant_method_other_layer_returns_none(self):
        cfg = VllmCompressedTensorsConfig(
            target_scheme_map={}, ignore=[], quant_format="dense"
        )
        other = torch.nn.Identity()
        method = cfg.get_quant_method(other, "model.layers.0.act")
        assert method is None
