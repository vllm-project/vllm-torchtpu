from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts

from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import \
    VllmCompressedTensorsMoEMethod
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16 import \
    VllmCompressedTensorsW4A16MoEMethod
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import \
    VllmCompressedTensorsW4ANMxfp4MoEMethod
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8MoEMethodTPU


class FakeQuantArgs:
    """Mock for QuantizationArgs from compressed-tensors."""

    def __init__(self,
                 num_bits=4,
                 strategy="group",
                 group_size=16,
                 symmetric=True,
                 dynamic=False,
                 quant_type="int",
                 actorder="none",
                 format="int4"):
        self.num_bits = num_bits
        self.strategy = strategy
        self.group_size = group_size
        self.symmetric = symmetric
        self.dynamic = dynamic
        self.type = quant_type
        self.actorder = actorder
        self.format = format

    def __getitem__(self, key):
        if key == "format":
            return self.format
        raise KeyError(key)

    def get(self, key, default=None):
        if key == "format":
            return self.format
        return default


class FakeActivation:
    """Mimics MoEActivation enum."""

    def __init__(self, value):
        self.value = value


# Need to mock vllm.model_executor.layers.fused_moe.RoutedExperts isinstance check
# Because VllmCompressedTensorsMoEMethod.get_moe_method asserts isinstance(layer, RoutedExperts)
class FakeRoutedExperts(RoutedExperts):
    """Subclass RoutedExperts to pass isinstance check without triggering full vLLM config requirements."""

    def __init__(self, experts_per_token=2):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.experts_per_token = experts_per_token
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = False
        self.moe_config.activation = FakeActivation("silu")
        self.use_grouped_topk = False


class TestCompressedTensorsConfigRouting:
    """Verify that VllmCompressedTensorsConfig correctly routes configs."""

    def test_get_moe_method_routing(self, monkeypatch):
        # We need to mock _is_int4_w4aN, _is_weight_fp8 to make them independent of full configs, or use correct FakeQuantArgs
        config = MagicMock(spec=VllmCompressedTensorsConfig)

        # Monkeypatch _is_mxfp4
        config._is_mxfp4 = MagicMock(return_value=False)

        # 1. Test W4A16 routing
        weight_quant = FakeQuantArgs(num_bits=4,
                                     strategy="group",
                                     group_size=16,
                                     format="pack")  # using int4
        input_quant = None
        scheme_dict = {
            "weights": weight_quant,
            "input_activations": input_quant
        }
        config.get_scheme_dict.return_value = scheme_dict

        # Monkey patch routing methods
        monkeypatch.setattr(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors._is_int4_w4aN",
            lambda w: w.num_bits == 4 and w.format != "mxfp4")
        monkeypatch.setattr(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors._is_weight_fp8",
            lambda w: w.format == "float8")
        monkeypatch.setattr(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors._build_fp8_config",
            MagicMock())

        layer = FakeRoutedExperts(experts_per_token=2)

        method = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block")
        assert isinstance(method, VllmCompressedTensorsW4A16MoEMethod)

        # 2. Test W4A8 routing (should route to the same runner)
        input_quant_8 = FakeQuantArgs(num_bits=8,
                                      strategy="token",
                                      dynamic=True,
                                      format="int8")
        scheme_dict_w4a8 = {
            "weights": weight_quant,
            "input_activations": input_quant_8
        }
        config.get_scheme_dict.return_value = scheme_dict_w4a8

        method_w4a8 = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block")
        assert isinstance(method_w4a8, VllmCompressedTensorsW4A16MoEMethod)

        # 3. Test MXFP4 routing
        weight_quant_mxfp4 = FakeQuantArgs(num_bits=4, format="mxfp4")
        scheme_dict_mxfp4 = {
            "weights": weight_quant_mxfp4,
            "input_activations": None
        }
        config.get_scheme_dict.return_value = scheme_dict_mxfp4
        config._is_mxfp4 = MagicMock(return_value=True)

        method_mxfp4 = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block")
        assert isinstance(method_mxfp4,
                          VllmCompressedTensorsW4ANMxfp4MoEMethod)

        # 4. Test FP8 routing
        config._is_mxfp4 = MagicMock(return_value=False)
        weight_quant_fp8 = FakeQuantArgs(num_bits=8, format="float8")
        scheme_dict_fp8 = {
            "weights": weight_quant_fp8,
            "input_activations": None
        }
        config.get_scheme_dict.return_value = scheme_dict_fp8

        method_fp8 = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block")
        assert isinstance(method_fp8, VllmFp8MoEMethodTPU)

        # 5. Test unsupported strategy routing fallback
        weight_quant_unsupported = FakeQuantArgs(num_bits=16, format="unknown")
        scheme_dict_unsupported = {
            "weights": weight_quant_unsupported,
            "input_activations": None
        }
        config.get_scheme_dict.return_value = scheme_dict_unsupported

        with pytest.raises(RuntimeError,
                           match="Unsupported TPU FusedMoe scheme"):
            VllmCompressedTensorsMoEMethod.get_moe_method(
                config, layer, "model.layers.0.block")
