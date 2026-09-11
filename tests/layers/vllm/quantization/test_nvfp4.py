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
"""Tests for NVFP4 (ModelOpt FP4) quantization support for TPU.

Verifies helper functions (_fresh, _to_kernel_scale, _requant_moe_w4a8),
VllmNvfp4Config dispatch, VllmNvfp4LinearMethod (create_weights,
process_weights_after_loading, apply), and VllmNvfp4MoEMethod (create_weights,
process_weights_after_loading across W4A16 and W4A8 paths, apply_monolithic).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoeWeightScaleSupported,
                                                  RoutedExperts)
from vllm.model_executor.layers.linear import LinearBase

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.vllm.quantization.nvfp4 import (
    VllmNvfp4Config, VllmNvfp4LinearMethod, VllmNvfp4MoEMethod, _fresh,
    _NullInputQuantKernel, _requant_moe_w4a8, _to_kernel_scale)
from vllm_torchtpu.layers.vllm.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod)


class FakeActivation:
    """Mimics MoEActivation enum or object with a value attribute."""

    def __init__(self, value: str):
        self.value = value


class FakeRoutedExperts(RoutedExperts):
    """Subclass RoutedExperts for isolated unit tests without full vLLM setup."""

    def __init__(
        self,
        num_experts: int = 4,
        experts_per_token: int = 2,
        activation: str = "silu",
        use_ep: bool = False,
    ):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.num_experts = num_experts
        self.moe_config.experts_per_token = experts_per_token
        self.moe_config.ep_size = 1
        self.moe_config.ep_rank = 0
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = use_ep
        self.global_num_experts = num_experts
        self.expert_placement_strategy = "linear"
        self.activation = FakeActivation(activation)
        self.use_grouped_topk = False


def make_linear_layer(input_size: int = 64,
                      output_size: int = 32) -> LinearBase:
    """Create a bare LinearBase instance for testing without initializing TP groups."""
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = input_size
    layer.output_size = output_size
    return layer


class TestNvfp4Helpers:
    """Tests for low-level memory and scale helper routines."""

    def test_fresh(self):
        t = torch.randn(4, 8, dtype=torch.float32)
        f = _fresh(t)
        assert torch.equal(f, t)
        assert f.data_ptr() != t.data_ptr()
        assert f.dtype == t.dtype
        assert f.device == t.device

    def test_to_kernel_scale_2d(self):
        out_dim, num_blocks = 32, 4
        s = torch.randn(out_dim, num_blocks, dtype=torch.float32)
        k = _to_kernel_scale(s)
        # Expected shape: [num_blocks, 1, out_dim]
        assert k.shape == (num_blocks, 1, out_dim)
        assert torch.equal(k.squeeze(1), s.transpose(-1, -2))
        assert k.is_contiguous()

    def test_to_kernel_scale_3d(self):
        num_experts, out_dim, num_blocks = 2, 32, 4
        s = torch.randn(num_experts, out_dim, num_blocks, dtype=torch.float32)
        k = _to_kernel_scale(s)
        # Expected shape: [num_experts, num_blocks, 1, out_dim]
        assert k.shape == (num_experts, num_blocks, 1, out_dim)
        assert torch.equal(k.squeeze(2), s.transpose(-1, -2))
        assert k.is_contiguous()

    def test_requant_moe_w4a8_aligned(self):
        E, two_i, H, block = 2, 32, 32, 16
        inter = two_i // 2
        w13_u8 = torch.randint(0, 256, (E, two_i, H // 2), dtype=torch.uint8)
        w13_scale_f = torch.rand(E, two_i, H // 16, dtype=torch.float32) + 0.1
        w2_u8 = torch.randint(0, 256, (E, H, inter // 2), dtype=torch.uint8)
        w2_scale_f = torch.rand(E, H, inter // 16, dtype=torch.float32) + 0.1

        w13_out, w13_s4, w2_out, w2_s4 = _requant_moe_w4a8(
            w13_u8, w13_scale_f, w2_u8, w2_scale_f, block)

        assert w13_out.shape == (E, two_i, H // 2)
        assert w13_s4.shape == (E, H // block, 1, two_i)
        assert w2_out.shape == (E, H, inter // 2)
        assert w2_s4.shape == (E, inter // block, 1, H)

    def test_requant_moe_w4a8_padded(self):
        # When inter is not a multiple of block (e.g., inter=16, block=32)
        E, two_i, H, block = 2, 32, 32, 32
        inter = two_i // 2  # 16
        # align_to(16, 32) -> 32, inter_pad = 16, 2 * I_pad = 64
        w13_u8 = torch.randint(0, 256, (E, two_i, H // 2), dtype=torch.uint8)
        w13_scale_f = torch.rand(E, two_i, H // 16, dtype=torch.float32) + 0.1
        w2_u8 = torch.randint(0, 256, (E, H, inter // 2), dtype=torch.uint8)
        w2_scale_f = torch.rand(E, H, inter // 16, dtype=torch.float32) + 0.1

        w13_out, w13_s4, w2_out, w2_s4 = _requant_moe_w4a8(
            w13_u8, w13_scale_f, w2_u8, w2_scale_f, block)

        assert w13_out.shape == (E, 64, H // 2)
        assert w13_s4.shape == (E, H // block, 1, 64)
        assert w2_out.shape == (E, H, 32 // 2)
        assert w2_s4.shape == (E, 32 // block, 1, H)

    def test_requant_moe_w4a8_unaligned_hidden_raises(self):
        # Contracting dim H must be divisible by block
        E, two_i, H_unaligned, block = 2, 32, 30, 16
        w13_u8 = torch.randint(0,
                               256, (E, two_i, H_unaligned // 2),
                               dtype=torch.uint8)
        w13_scale_f = torch.rand(E, two_i, 2, dtype=torch.float32)
        w2_u8 = torch.randint(0, 256, (E, H_unaligned, 8), dtype=torch.uint8)
        w2_scale_f = torch.rand(E, H_unaligned, 1, dtype=torch.float32)

        with pytest.raises(AssertionError, match="divisible by block"):
            _requant_moe_w4a8(w13_u8, w13_scale_f, w2_u8, w2_scale_f, block)


class TestNvfp4Config:
    """Tests for VllmNvfp4Config get_name and get_quant_method dispatch."""

    def test_get_name(self):
        assert VllmNvfp4Config.get_name() == "modelopt_fp4"

    def test_get_quant_method_linear_quantized(self):
        cfg = VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        cfg.vllm_config = MagicMock()
        layer = make_linear_layer(64, 32)
        method = cfg.get_quant_method(layer, "model.layers.0.self_attn.q_proj")
        assert isinstance(method, VllmNvfp4LinearMethod)
        assert method.quant_config is cfg

    def test_get_quant_method_linear_excluded(self):
        cfg = VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        cfg.exclude_modules = ["model.layers.0.skip"]
        layer = make_linear_layer(64, 32)
        method = cfg.get_quant_method(layer, "model.layers.0.skip")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_get_quant_method_moe_quantized(self):
        cfg = VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        cfg.vllm_config = MagicMock()
        cfg.vllm_config.parallel_config.enable_expert_parallel = False
        layer = FakeRoutedExperts()
        method = cfg.get_quant_method(layer, "model.layers.0.block")
        assert isinstance(method, VllmNvfp4MoEMethod)
        assert method.quant_config is cfg

    def test_get_quant_method_moe_excluded(self, vllm_config_context):
        cfg = VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        cfg.vllm_config = MagicMock()
        cfg.vllm_config.parallel_config.enable_expert_parallel = False
        cfg.exclude_modules = ["model.layers.0.skip_moe"]
        layer = FakeRoutedExperts()
        method = cfg.get_quant_method(layer, "model.layers.0.skip_moe")
        assert isinstance(method, VllmUnquantizedFusedMoEMethod)

    def test_get_quant_method_attention_returns_none(self):
        cfg = VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        layer = Attention.__new__(Attention)
        torch.nn.Module.__init__(layer)
        assert cfg.get_quant_method(layer, "model.layers.0.self_attn") is None

    def test_get_quant_method_other_layer_returns_none(self):
        cfg = VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        layer = torch.nn.Linear(16, 16)
        assert cfg.get_quant_method(layer, "model.norm") is None


class TestNvfp4LinearMethod:
    """Tests for VllmNvfp4LinearMethod weight creation, processing, and apply."""

    def test_null_input_quant_kernel(self):
        kernel = _NullInputQuantKernel()
        assert kernel.input_quant_key() is None

    def test_create_weights_registers_scalar_loader(self):
        quant_config = MagicMock(group_size=16)
        linear_config = MagicMock()
        method = VllmNvfp4LinearMethod(quant_config, linear_config)
        layer = make_linear_layer(64, 32)

        def mock_base_create_weights(self, lyr, *args, **kwargs):
            lyr.input_scale = torch.nn.Parameter(torch.zeros(1),
                                                 requires_grad=False)
            lyr.weight_scale_2 = torch.nn.Parameter(torch.zeros(1),
                                                    requires_grad=False)

        with patch(
                "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4LinearMethod.create_weights",
                mock_base_create_weights):
            method.create_weights(layer, 64, [32], 64, 32, torch.bfloat16)

        assert hasattr(layer.input_scale, "weight_loader")
        assert hasattr(layer.weight_scale_2, "weight_loader")

        # Test scalar_weight_loader fills single-item scalar
        layer.input_scale.weight_loader(layer.input_scale, torch.tensor([3.5]))
        assert layer.input_scale.item() == 3.5

        # Test scalar_weight_loader raises when tensor has multiple elements
        with pytest.raises(AssertionError):
            layer.input_scale.weight_loader(layer.input_scale,
                                            torch.tensor([1.0, 2.0]))

    def test_process_weights_after_loading(self):
        N, K, group_size = 32, 64, 16
        layer = make_linear_layer(K, N)
        layer.weight = torch.nn.Parameter(torch.randint(0,
                                                        256, (N, K // 2),
                                                        dtype=torch.uint8),
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.ones(
            N, K // group_size, dtype=torch.float32),
                                                requires_grad=False)
        layer.weight_scale_2 = torch.nn.Parameter(torch.tensor(
            [2.5], dtype=torch.float32),
                                                  requires_grad=False)
        layer.input_scale = torch.nn.Parameter(torch.tensor(
            [1.0], dtype=torch.float32),
                                               requires_grad=False)

        method = VllmNvfp4LinearMethod(MagicMock(group_size=group_size),
                                       MagicMock())

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.load_kmajor_fp4"
        ) as mock_load:
            mock_load.side_effect = lambda t: torch.zeros(
                K, N, dtype=torch.uint8)
            method.process_weights_after_loading(layer)

        assert layer.weight.shape == (K, N)
        assert layer.weight_scale.shape == (1, K // group_size, 1, N)
        expected_scale = _to_kernel_scale(
            torch.ones(N, K // group_size) * 2.5).unsqueeze(0)
        assert torch.allclose(layer.weight_scale, expected_scale)
        assert not hasattr(layer, "weight_scale_2")
        assert not hasattr(layer, "input_scale")

    def test_process_weights_mismatched_global_scale_raises(self):
        layer = make_linear_layer(64, 32)
        layer.weight_scale_2 = torch.nn.Parameter(torch.tensor([1.0, 2.0]),
                                                  requires_grad=False)
        method = VllmNvfp4LinearMethod(MagicMock(group_size=16), MagicMock())

        with pytest.raises(
                AssertionError,
                match="Fused NVFP4 projections must share one global scale"):
            method.process_weights_after_loading(layer)

    def test_apply_without_bias(self):
        layer = make_linear_layer(64, 32)
        layer.weight = torch.nn.Parameter(torch.empty(64, 32),
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.empty(1, 4, 1, 32),
                                                requires_grad=False)
        x = torch.randn(4, 64)
        method = VllmNvfp4LinearMethod(MagicMock(group_size=16), MagicMock())

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.quantized_matmul_fp4"
        ) as mock_matmul:
            mock_matmul.return_value = torch.ones(4, 32)
            out = method.apply(layer, x)
            assert torch.equal(out, torch.ones(4, 32))
            mock_matmul.assert_called_once_with(x, layer.weight,
                                                layer.weight_scale)

    def test_apply_with_bias(self):
        layer = make_linear_layer(64, 32)
        layer.weight = torch.nn.Parameter(torch.empty(64, 32),
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.empty(1, 4, 1, 32),
                                                requires_grad=False)
        x = torch.randn(4, 64)
        bias = torch.full((32, ), 3.0)
        method = VllmNvfp4LinearMethod(MagicMock(group_size=16), MagicMock())

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.quantized_matmul_fp4",
                return_value=torch.ones(4, 32)):
            out = method.apply(layer, x, bias=bias)
            assert torch.equal(out, torch.full((4, 32), 4.0))


class TestNvfp4MoEMethod:
    """Tests for VllmNvfp4MoEMethod weight creation, processing, and forward execution."""

    def test_moe_properties(self):
        moe_cfg = MagicMock(has_bias=False, is_act_and_mul=True)
        method = VllmNvfp4MoEMethod(MagicMock(group_size=16), moe_cfg)
        assert method.is_monolithic is True
        assert method.group_size == 16
        assert method.use_global_sf is False
        assert method.get_fused_moe_quant_config(None) is None

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.enable_pipelined_collective_and_compute",
                return_value=True):
            assert method.supports_internal_mk is True

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.enable_pipelined_collective_and_compute",
                return_value=False):
            assert method.supports_internal_mk is False

    def test_moe_create_weights_sets_quant_method_attrs(self):
        layer = FakeRoutedExperts()
        method = VllmNvfp4MoEMethod(MagicMock(group_size=16), layer.moe_config)

        def mock_moe_create_weights(self, lyr, *args, **kwargs):
            lyr.w13_weight_scale = torch.nn.Parameter(torch.empty(0),
                                                      requires_grad=False)
            lyr.w2_weight_scale = torch.nn.Parameter(torch.empty(0),
                                                     requires_grad=False)
            lyr.w13_weight_scale_2 = torch.nn.Parameter(torch.empty(0),
                                                        requires_grad=False)
            lyr.w2_weight_scale_2 = torch.nn.Parameter(torch.empty(0),
                                                       requires_grad=False)

        with patch(
                "vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4FusedMoE.create_weights",
                mock_moe_create_weights):
            method.create_weights(layer, 4, 128, 64, torch.bfloat16)

        assert (getattr(
            layer.w13_weight_scale,
            "quant_method") == FusedMoeWeightScaleSupported.BLOCK.value)
        assert (getattr(
            layer.w2_weight_scale,
            "quant_method") == FusedMoeWeightScaleSupported.BLOCK.value)
        assert (getattr(
            layer.w13_weight_scale_2,
            "quant_method") == FusedMoeWeightScaleSupported.TENSOR.value)
        assert (getattr(
            layer.w2_weight_scale_2,
            "quant_method") == FusedMoeWeightScaleSupported.TENSOR.value)

    def test_process_weights_validation_guards(self):
        method = VllmNvfp4MoEMethod(
            MagicMock(group_size=16),
            MagicMock(has_bias=False, is_act_and_mul=True))

        # 1. Non-RoutedExperts layer
        with pytest.raises(AssertionError):
            method.process_weights_after_loading(torch.nn.Module())

        # 2. Layer with bias
        layer = FakeRoutedExperts()
        method_bias = VllmNvfp4MoEMethod(
            MagicMock(group_size=16),
            MagicMock(has_bias=True, is_act_and_mul=True))
        with pytest.raises(AssertionError, match="does not support bias"):
            method_bias.process_weights_after_loading(layer)

        # 3. Non act_and_mul activation
        method_non_gated = VllmNvfp4MoEMethod(
            MagicMock(group_size=16),
            MagicMock(has_bias=False, is_act_and_mul=False))
        with pytest.raises(AssertionError, match="expects gated"):
            method_non_gated.process_weights_after_loading(layer)

        # 4. swigluoai activation unsupported
        layer_swiglu = FakeRoutedExperts(activation="swigluoai")
        with pytest.raises(NotImplementedError, match="swigluoai"):
            method.process_weights_after_loading(layer_swiglu)

    def test_process_weights_w4a16_default(self, monkeypatch):
        monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", None)
        E, two_i, H, group_size = 2, 32, 64, 16
        inter = two_i // 2
        layer = FakeRoutedExperts(num_experts=E,
                                  activation="silu",
                                  use_ep=False)

        layer.w13_weight = torch.nn.Parameter(torch.randint(0,
                                                            256,
                                                            (E, two_i, H // 2),
                                                            dtype=torch.uint8),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.randint(0,
                                                           256,
                                                           (E, H, inter // 2),
                                                           dtype=torch.uint8),
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            E, two_i, H // group_size, dtype=torch.float32),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            E, H, inter // group_size, dtype=torch.float32),
                                                   requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(torch.tensor(
            [[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
                                                      requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(torch.tensor(
            [2.0, 4.0], dtype=torch.float32),
                                                     requires_grad=False)
        layer.w13_input_scale = torch.nn.Parameter(torch.tensor([1.0]),
                                                   requires_grad=False)
        layer.w2_input_scale = torch.nn.Parameter(torch.tensor([1.0]),
                                                  requires_grad=False)

        method = VllmNvfp4MoEMethod(
            MagicMock(group_size=group_size),
            MagicMock(has_bias=False, is_act_and_mul=True))

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.load_kmajor_fp4"
        ) as mock_load, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.moe_routing.register_experts_start_buffer"
        ) as mock_reg, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.prebuild_fused_moe_kernel"
        ) as mock_prebuild:
            # Load unpacks [E, N, K/2] -> [E, K, N]
            mock_load.side_effect = lambda t: torch.zeros(
                t.shape[0], t.shape[-1] * 2, t.shape[1], dtype=torch.uint8)
            method.process_weights_after_loading(layer)

            assert mock_load.call_count == 2
            mock_reg.assert_called_once_with(layer,
                                             device=layer.w13_weight.device)
            mock_prebuild.assert_called_once_with(topk=2,
                                                  activation="silu",
                                                  use_ep=False)

        assert layer.w13_weight.shape == (E, H, two_i)
        assert layer.w2_weight.shape == (E, inter, H)
        assert layer.w13_weight_scale.shape == (E, H // group_size, 1, two_i)
        assert layer.w2_weight_scale.shape == (E, inter // group_size, 1, H)
        assert not hasattr(layer, "w13_weight_scale_2")
        assert not hasattr(layer, "w2_weight_scale_2")
        assert not hasattr(layer, "w13_input_scale")
        assert not hasattr(layer, "w2_input_scale")

    def test_process_weights_w4a8_jax_requant(self, monkeypatch):
        monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", 32)
        E, two_i, H, group_size = 2, 64, 64, 16
        inter = two_i // 2  # 32; both H=64 and inter=32 are divisible by 32
        layer = FakeRoutedExperts(num_experts=E,
                                  activation="silu",
                                  use_ep=False)

        layer.w13_weight = torch.nn.Parameter(torch.randint(0,
                                                            256,
                                                            (E, two_i, H // 2),
                                                            dtype=torch.uint8),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.randint(0,
                                                           256,
                                                           (E, H, inter // 2),
                                                           dtype=torch.uint8),
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            E, two_i, H // group_size, dtype=torch.float32),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            E, H, inter // group_size, dtype=torch.float32),
                                                   requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(
            E, 2, dtype=torch.float32),
                                                      requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(
            E, dtype=torch.float32),
                                                     requires_grad=False)

        method = VllmNvfp4MoEMethod(
            MagicMock(group_size=group_size),
            MagicMock(has_bias=False, is_act_and_mul=True))

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.requant_load_kmajor_fp4"
        ) as mock_requant, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.prebuild_fused_moe_kernel"
        ):
            mock_requant.return_value = (torch.zeros(E,
                                                     64,
                                                     64,
                                                     dtype=torch.uint8),
                                         torch.zeros(E, 2, 1, 64))
            method.process_weights_after_loading(layer)
            assert mock_requant.call_count == 2

    def test_process_weights_w4a8_torch_requant_with_padding(
            self, monkeypatch):
        monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", 32)
        E, two_i, H, group_size = 2, 32, 64, 16
        inter = two_i // 2  # 16; H=64 divisible by 32, but inter=16 is NOT
        layer = FakeRoutedExperts(num_experts=E,
                                  activation="silu",
                                  use_ep=False)

        layer.w13_weight = torch.nn.Parameter(torch.randint(0,
                                                            256,
                                                            (E, two_i, H // 2),
                                                            dtype=torch.uint8),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.randint(0,
                                                           256,
                                                           (E, H, inter // 2),
                                                           dtype=torch.uint8),
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            E, two_i, H // group_size, dtype=torch.float32),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            E, H, inter // group_size, dtype=torch.float32),
                                                   requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(
            E, 2, dtype=torch.float32),
                                                      requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(
            E, dtype=torch.float32),
                                                     requires_grad=False)

        method = VllmNvfp4MoEMethod(
            MagicMock(group_size=group_size),
            MagicMock(has_bias=False, is_act_and_mul=True))

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4._requant_moe_w4a8"
        ) as mock_torch_requant, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.load_kmajor_fp4"
        ) as mock_load, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.prebuild_fused_moe_kernel"
        ):
            mock_torch_requant.return_value = (
                torch.zeros(E, 64, 32, dtype=torch.uint8),
                torch.zeros(E, 2, 1, 64),
                torch.zeros(E, 64, 16, dtype=torch.uint8),
                torch.zeros(E, 1, 1, 64),
            )
            mock_load.side_effect = lambda t: t
            method.process_weights_after_loading(layer)
            mock_torch_requant.assert_called_once()
            assert mock_load.call_count == 2

    def test_process_weights_with_expert_parallel(self):
        layer = FakeRoutedExperts(use_ep=True)
        E, two_i, H, group_size = 2, 32, 64, 16
        inter = two_i // 2
        layer.w13_weight = torch.nn.Parameter(torch.empty(E,
                                                          two_i,
                                                          H // 2,
                                                          dtype=torch.uint8),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.empty(E,
                                                         H,
                                                         inter // 2,
                                                         dtype=torch.uint8),
                                             requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(torch.ones(
            E, two_i, H // group_size),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(torch.ones(
            E, H, inter // group_size),
                                                   requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(E, 2),
                                                      requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(torch.ones(E),
                                                     requires_grad=False)

        method = VllmNvfp4MoEMethod(
            MagicMock(group_size=group_size),
            MagicMock(has_bias=False, is_act_and_mul=True))

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.load_kmajor_fp4",
                side_effect=lambda t: t
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.moe_routing.validate_linear_ep_placement"
        ) as mock_ep, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.prebuild_fused_moe_kernel"
        ):
            method.process_weights_after_loading(layer)
            mock_ep.assert_called_once_with(layer)
            assert method._tpu_activation_str == "silu"
            assert not hasattr(layer, "_tpu_activation_str")

    def test_apply_monolithic_standard(self):
        layer = FakeRoutedExperts()
        layer._experts_start = torch.zeros(4)
        layer.w13_weight = torch.empty(0)
        layer.w2_weight = torch.empty(0)
        layer.w13_weight_scale = torch.empty(0)
        layer.w2_weight_scale = torch.empty(0)

        x = torch.randn(4, 64)
        router_logits = torch.randn(4, 4)
        topk_weights = torch.full((4, 2), 0.5)
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]],
                                dtype=torch.int32)

        method = VllmNvfp4MoEMethod(MagicMock(group_size=16), layer.moe_config)
        method._tpu_activation_str = "silu"

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.moe_routing.route",
                return_value=(topk_weights, topk_ids)
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.fused_moe_gmm"
        ) as mock_gmm, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.enable_pipelined_collective_and_compute",
                return_value=False):
            mock_gmm.return_value = torch.ones(4, 64)
            out = method.apply_monolithic(layer, x, router_logits)
            assert torch.equal(out, torch.ones(4, 64))
            mock_gmm.assert_called_once()
            kwargs = mock_gmm.call_args.kwargs
            assert kwargs["activation"] == "silu"
            assert kwargs["topk"] == 2

    def test_apply_monolithic_pipelined(self):
        layer = FakeRoutedExperts()
        layer._experts_start = torch.zeros(4)
        layer.w13_weight = torch.empty(0)
        layer.w2_weight = torch.empty(0)
        layer.w13_weight_scale = torch.empty(0)
        layer.w2_weight_scale = torch.empty(0)

        x = torch.randn(4, 64)
        router_logits = torch.randn(4, 4)

        method = VllmNvfp4MoEMethod(MagicMock(group_size=16), layer.moe_config)
        method._tpu_activation_str = "silu"

        with patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.moe_routing.route",
                return_value=(torch.full(
                    (4, 2), 0.5), torch.zeros((4, 2), dtype=torch.int32))
        ), patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.pipelined_fused_moe_gmm"
        ) as mock_pipe, patch(
                "vllm_torchtpu.layers.vllm.quantization.nvfp4.enable_pipelined_collective_and_compute",
                return_value=True):
            mock_pipe.return_value = torch.ones(4, 64)
            out = method.apply_monolithic(layer, x, router_logits)
            assert torch.equal(out, torch.ones(4, 64))
            mock_pipe.assert_called_once()
