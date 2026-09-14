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
"""Tests for Compressed Tensors quantization integration on TPU."""

import ctypes
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import RoutedExperts

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16 import \
    VllmCompressedTensorsW4A16MoEMethod


def _reference_moe(x: torch.Tensor,
                   router_logits: torch.Tensor,
                   w1: torch.Tensor,
                   w2: torch.Tensor,
                   w1_bias: Optional[torch.Tensor],
                   w2_bias: Optional[torch.Tensor],
                   top_k: int,
                   renormalize: bool,
                   activation: str,
                   scoring_func: str = "softmax") -> torch.Tensor:
    """Reference implementation of MoE forward pass in Torch."""
    match scoring_func:
        case "softmax":
            expert_weights = F.softmax(router_logits, dim=-1)
        case "sigmoid":
            expert_weights = F.sigmoid(router_logits)
        case _:
            raise NotImplementedError(
                f"No reference implementation for {scoring_func} scoring")

    expert_weights, expert_indices = torch.topk(expert_weights, top_k, dim=-1)
    if renormalize:
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

    x_expanded = torch.einsum("ti,eoi->teo", x, w1)
    if w1_bias is not None:
        x_expanded += w1_bias.unsqueeze(0)

    match activation:
        case "silu":
            x1, x3 = x_expanded.chunk(chunks=2, dim=-1)
            x_expanded = F.silu(x1) * x3
        case "swigluoai":
            x1, x3 = x_expanded[..., ::2], x_expanded[..., 1::2]
            x1 = x1.clamp(min=None, max=7.0)
            x3 = x3.clamp(min=-7.0, max=7.0)
            gated_activation = x1 * torch.sigmoid(x1 * 1.702)
            x_expanded = gated_activation * (x3 + 1)
        case _:
            raise NotImplementedError(
                f"No reference implementation for {activation} activation")

    x_expanded = torch.einsum("teo,eio->tei", x_expanded, w2)
    if w2_bias is not None:
        x_expanded += w2_bias.unsqueeze(0)

    seq_indexes = torch.arange(x_expanded.shape[0]).unsqueeze(1)
    x_expanded = x_expanded[seq_indexes, expert_indices]

    return torch.einsum("tai,ta->ti", x_expanded, expert_weights)


def quantize_and_pack_shard_w4(weight_bf16: torch.Tensor, group_size: int):
    # weight_bf16 shape: [out_dim, in_dim]
    out_dim, in_dim = weight_bf16.shape

    # Group along the input dimension (in_dim, dim=1)
    w_grouped = weight_bf16.reshape(out_dim, in_dim // group_size, group_size)
    scales = w_grouped.abs().max(dim=-1, keepdim=True).values / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    w_quant = torch.round(w_grouped / scales).clamp(-8, 7).to(torch.int8)
    w_dequant = (w_quant.to(torch.float32) * scales.to(torch.float32)).reshape(
        out_dim, in_dim)

    w_u4 = (w_quant + 8) & 0xF
    w_to_pack = w_u4.reshape(out_dim, in_dim // group_size, group_size // 8, 8)

    packed = torch.zeros(out_dim,
                         in_dim // group_size,
                         group_size // 8,
                         dtype=torch.int32)
    for i in range(8):
        packed |= w_to_pack[..., i].to(torch.int32) << (i * 4)

    packed = packed.reshape(out_dim, in_dim // 8)
    scales = scales.squeeze(-1)  # [out_dim, in_dim // group_size]

    return packed, scales, w_dequant


class FakeQuantArgs:
    """Mock for QuantizationArgs from compressed-tensors."""

    def __init__(self,
                 num_bits=4,
                 strategy="group",
                 group_size=16,
                 symmetric=True,
                 dynamic=False,
                 quant_type="int",
                 actorder="none"):
        self.num_bits = num_bits
        self.strategy = strategy
        self.group_size = group_size
        self.symmetric = symmetric
        self.dynamic = dynamic
        self.type = quant_type
        self.actorder = actorder


class FakeActivation:
    """Mimics MoEActivation enum."""

    def __init__(self, value):
        self.value = value


class FakeRoutedExperts(RoutedExperts):
    """Subclass RoutedExperts to pass isinstance check without triggering full vLLM config requirements."""

    def __init__(self, experts_per_token=2):
        # Call Module.__init__ to initialize parameters dictionary, bypassing RoutedExperts.__init__
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.experts_per_token = experts_per_token
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = False
        self.moe_config.activation = FakeActivation("silu")
        self.use_grouped_topk = False
        self.custom_routing_function = None
        self.scoring_func = "softmax"
        self.e_score_correction_bias = None
        self.routed_scaling_factor = 1.0


class TestW4MoEWeightPreprocessing:
    """Verify CPU weight packing and sign-extension logic."""

    def test_xor_sign_conversion(self):
        """Symmetric unsigned INT4 [0, 15] should be converted to signed [-8, 7] via XOR."""
        layer = FakeRoutedExperts(experts_per_token=2)
        layer.activation = FakeActivation("silu")

        # Packed INT32 carrier can fit 8 values. Let's write one carrier.
        # Original signed INT4: [-8, -7, -1, 0, 1, 7, -8, 7]
        # In two's complement hex (4-bit):
        # -8 = 0x8, -7 = 0x9, -1 = 0xF, 0 = 0x0, 1 = 0x1, 7 = 0x7, -8 = 0x8, 7 = 0x7
        # Packed LSB-first into 32-bit INT32: 0x78710F98
        initial_carrier = 0x78710F98

        num_experts = 2

        # 1. Setup parameters using create_weights
        weight_quant = FakeQuantArgs(num_bits=4,
                                     strategy="group",
                                     group_size=8)
        method = VllmCompressedTensorsW4A16MoEMethod(weight_quant, None,
                                                     layer.moe_config)

        # Mock TPU runner config for method
        method.moe = MagicMock()
        method.moe.tp_size = 1
        method.moe.tp_rank = 0

        # Mock map_global_expert_id_to_local_expert_id
        layer._map_global_expert_id_to_local_expert_id = MagicMock(
            side_effect=lambda x: x)

        extra_weight_attrs = {
            "weight_loader": MagicMock(),
            "intermediate_size_full": 8,
        }

        with patch.object(VllmCompressedTensorsW4A16MoEMethod, "_validate_w4a16_scheme"), \
             patch.object(VllmCompressedTensorsW4A16MoEMethod, "_validate_int32_weight_carriers"), \
             patch("vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.prebuild_fused_moe_kernel"):
            method.create_weights(layer, num_experts, 8, 8, torch.float32,
                                  **extra_weight_attrs)

            loader_hook = layer.w13_weight_packed.weight_loader
            assert loader_hook is not None

            # Zero-initialize TPU param to ensure we verify the copy
            layer.w13_weight_packed.data.zero_()
            layer.w2_weight_packed.data.zero_()

            # 2. Simulate loading shards using the hook
            # w13_weight_packed has shape [2, 1, 16] (num_experts, in_dim, out_dim_packed)
            # Shards from loader: [weight_shard_size, ...]
            # We load shape [8, 1] filled with initial_carrier
            shard_w1_e0 = torch.tensor([[initial_carrier]] * 8,
                                       dtype=torch.int32)
            shard_w3_e0 = torch.tensor([[initial_carrier]] * 8,
                                       dtype=torch.int32)
            shard_w1_e1 = torch.tensor([[initial_carrier]] * 8,
                                       dtype=torch.int32)
            shard_w3_e1 = torch.tensor([[initial_carrier]] * 8,
                                       dtype=torch.int32)

            # Load all shards for w13 to trigger copy
            loader_hook(layer.w13_weight_packed, shard_w1_e0,
                        "w13_weight_packed", "w1", 0)
            loader_hook(layer.w13_weight_packed, shard_w3_e0,
                        "w13_weight_packed", "w3", 0)
            loader_hook(layer.w13_weight_packed, shard_w1_e1,
                        "w13_weight_packed", "w1", 1)
            loader_hook(layer.w13_weight_packed, shard_w3_e1,
                        "w13_weight_packed", "w3", 1)

            # Call process_weights_after_loading to perform the H2D copy and cleanup scratchpad
            method.process_weights_after_loading(layer)

            # Expected output after XORing with 0x88888888: 0xF0F98710
            # signed int32 equivalent: -252082416
            expected_carrier = ctypes.c_int32(0xF0F98710).value

            # Verify TPU param content
            assert layer.w13_weight_packed[0, 0, 0].item() == expected_carrier
            assert layer.w13_weight_packed[1, 0, 0].item() == expected_carrier


class TestW4MoECorrectness:
    """Verify computation correctness of W4 MoE."""

    def test_custom_routing_called(self, device):
        """W4 MoE apply_monolithic should use custom_routing_function when present."""
        layer = MagicMock()
        layer.moe_config.experts_per_token = 2
        layer.moe_config.moe_parallel_config.use_ep = False
        layer.renormalize = True
        layer.experts_start = None

        mock_routing = MagicMock(
            return_value=(torch.ones(4, 2),
                          torch.zeros(4, 2, dtype=torch.int32)))
        layer.custom_routing_function = mock_routing

        layer.activation = "silu"
        method = MagicMock(spec=VllmCompressedTensorsW4A16MoEMethod)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)

        # Call the real apply_monolithic
        with patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.fused_moe_gmm",
                return_value=x,
        ):
            VllmCompressedTensorsW4A16MoEMethod.apply_monolithic(
                method, layer, x, router_logits)

        mock_routing.assert_called_once()

    def test_fused_moe_w4a16_correctness(self, device):
        """Verify mathematical correctness of W4A16 MoE on TPU against CPU reference."""
        if device.type != "tpu":
            pytest.skip(
                "Pallas W4A16 MoE kernel execution requires TPU device.")

        # Test parameters
        num_experts = 4
        topk = 2
        hidden_size = 128
        intermediate_size = 256
        group_size = 16
        num_tokens = 8

        # 1. Setup Layer Mock
        layer = FakeRoutedExperts(experts_per_token=topk)
        layer.activation = FakeActivation("silu")
        layer.renormalize = True
        layer.experts_start = None

        # 2. Setup Quant Config
        weight_quant = FakeQuantArgs(num_bits=4,
                                     strategy="group",
                                     group_size=group_size)
        method = VllmCompressedTensorsW4A16MoEMethod(weight_quant, None,
                                                     layer.moe_config)

        method.moe = MagicMock()
        method.moe.tp_size = 1
        method.moe.tp_rank = 0

        layer._map_global_expert_id_to_local_expert_id = MagicMock(
            side_effect=lambda x: x)

        # 3. Create Weights on TPU (and cpu scratchpads) directly
        extra_weight_attrs = {
            "weight_loader": MagicMock(),
            "intermediate_size_full": intermediate_size,
        }
        with torch.device(device):
            method.create_weights(layer, num_experts, hidden_size,
                                  intermediate_size, torch.float32,
                                  **extra_weight_attrs)

        loader_hook = layer.w13_weight_packed.weight_loader

        # 4. Generate random weights & pack them on CPU, then load them
        torch.manual_seed(42)

        # Lists for CPU reference
        w1_dequants = []
        w3_dequants = []
        w2_dequants = []

        for e in range(num_experts):
            w1_bf16 = (torch.rand(
                intermediate_size, hidden_size, dtype=torch.bfloat16) -
                       0.5) / 10
            w3_bf16 = (torch.rand(
                intermediate_size, hidden_size, dtype=torch.bfloat16) -
                       0.5) / 10
            w2_bf16 = (torch.rand(
                hidden_size, intermediate_size, dtype=torch.bfloat16) -
                       0.5) / 10

            # Pack
            packed_w1, scale_w1, dequant_w1 = quantize_and_pack_shard_w4(
                w1_bf16, group_size)
            packed_w3, scale_w3, dequant_w3 = quantize_and_pack_shard_w4(
                w3_bf16, group_size)
            packed_w2, scale_w2, dequant_w2 = quantize_and_pack_shard_w4(
                w2_bf16, group_size)

            # Save dequants for ref model
            w1_dequants.append(dequant_w1)
            w3_dequants.append(dequant_w3)
            w2_dequants.append(dequant_w2)

            # Load shards
            loader_hook(layer.w13_weight_packed, packed_w1,
                        "w13_weight_packed", "w1", e)
            loader_hook(layer.w13_weight_packed, packed_w3,
                        "w13_weight_packed", "w3", e)
            loader_hook(layer.w2_weight_packed, packed_w2, "w2_weight_packed",
                        "w2", e)

            loader_hook(layer.w13_weight_scale, scale_w1, "w13_weight_scale",
                        "w1", e)
            loader_hook(layer.w13_weight_scale, scale_w3, "w13_weight_scale",
                        "w3", e)
            loader_hook(layer.w2_weight_scale, scale_w2, "w2_weight_scale",
                        "w2", e)

        # Apply CPU-side transposition, XOR sign flip and move to TPU device
        method.process_weights_after_loading(layer)

        # 5. Build CPU reference tensors
        w1_ref = torch.stack([
            torch.cat([w1_dequants[e], w3_dequants[e]], dim=0)
            for e in range(num_experts)
        ],
                             dim=0)
        w2_ref = torch.stack(w2_dequants, dim=0)

        # 6. Generate Inputs
        x = torch.rand(num_tokens,
                       hidden_size,
                       dtype=torch.bfloat16,
                       device=device)
        router_logits = (torch.rand(
            num_tokens, num_experts, dtype=torch.bfloat16, device=device) -
                         0.5) * 5

        # 7. Execute TPU forward pass
        result_tpu = method.apply_monolithic(layer, x, router_logits)
        result_cpu = result_tpu.cpu().to(torch.float32)

        # 8. Execute CPU reference
        expected_cpu = _reference_moe(x.cpu().to(torch.float32),
                                      router_logits.cpu().to(torch.float32),
                                      w1_ref.to(torch.float32),
                                      w2_ref.to(torch.float32),
                                      w1_bias=None,
                                      w2_bias=None,
                                      top_k=topk,
                                      renormalize=True,
                                      activation="silu")

        # 9. Verify close match (allowing small tolerances since dynamic TPU compiles kernel optimizations)
        torch.testing.assert_close(result_cpu,
                                   expected_cpu,
                                   rtol=0.03,
                                   atol=0.03)

    def test_fused_moe_fp4_requant_correctness(self, device):
        """Verify execution of FP4 requantized MoE on TPU."""
        if device.type != "tpu":
            pytest.skip(
                "Pallas W4A16 MoE kernel execution requires TPU device.")

        # Test parameters
        num_experts = 4
        topk = 2
        hidden_size = 256
        intermediate_size = 256
        group_size = 16
        requant_block_size = 64
        num_tokens = 8

        layer = FakeRoutedExperts(experts_per_token=topk)
        layer.activation = FakeActivation("silu")
        layer.renormalize = True
        layer.experts_start = None

        weight_quant = FakeQuantArgs(num_bits=4,
                                     strategy="group",
                                     group_size=group_size)
        method = VllmCompressedTensorsW4A16MoEMethod(weight_quant, None,
                                                     layer.moe_config)
        method.moe = MagicMock()
        method.moe.tp_size = 1
        method.moe.tp_rank = 0
        layer._map_global_expert_id_to_local_expert_id = MagicMock(
            side_effect=lambda x: x)

        extra_weight_attrs = {
            "weight_loader": MagicMock(),
            "intermediate_size_full": intermediate_size,
        }
        with torch.device(device):
            method.create_weights(layer, num_experts, hidden_size,
                                  intermediate_size, torch.float32,
                                  **extra_weight_attrs)

        loader_hook = layer.w13_weight_packed.weight_loader
        torch.manual_seed(42)

        for e in range(num_experts):
            w1_bf16 = (torch.rand(
                intermediate_size, hidden_size, dtype=torch.bfloat16) -
                       0.5) / 10
            w3_bf16 = (torch.rand(
                intermediate_size, hidden_size, dtype=torch.bfloat16) -
                       0.5) / 10
            w2_bf16 = (torch.rand(
                hidden_size, intermediate_size, dtype=torch.bfloat16) -
                       0.5) / 10

            packed_w1, scale_w1, _ = quantize_and_pack_shard_w4(
                w1_bf16, group_size)
            packed_w3, scale_w3, _ = quantize_and_pack_shard_w4(
                w3_bf16, group_size)
            packed_w2, scale_w2, _ = quantize_and_pack_shard_w4(
                w2_bf16, group_size)

            loader_hook(layer.w13_weight_packed, packed_w1,
                        "w13_weight_packed", "w1", e)
            loader_hook(layer.w13_weight_packed, packed_w3,
                        "w13_weight_packed", "w3", e)
            loader_hook(layer.w2_weight_packed, packed_w2, "w2_weight_packed",
                        "w2", e)

            loader_hook(layer.w13_weight_scale, scale_w1, "w13_weight_scale",
                        "w1", e)
            loader_hook(layer.w13_weight_scale, scale_w3, "w13_weight_scale",
                        "w3", e)
            loader_hook(layer.w2_weight_scale, scale_w2, "w2_weight_scale",
                        "w2", e)

        with patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.envs.MOE_REQUANTIZE_WEIGHT_DTYPE",
                "float4_e2m1fn"
        ), patch(
                "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.envs.MOE_REQUANTIZE_BLOCK_SIZE",
                str(requant_block_size)):
            method.process_weights_after_loading(layer)

        x = torch.rand(num_tokens,
                       hidden_size,
                       dtype=torch.bfloat16,
                       device=device)
        router_logits = (torch.rand(
            num_tokens, num_experts, dtype=torch.bfloat16, device=device) -
                         0.5) * 5

        result_tpu = method.apply_monolithic(layer, x, router_logits)
        assert result_tpu.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result_tpu.cpu()).any()


def test_w4a16_requantize_block():
    """Verify that INT4 MoE weights are correctly requantized when MOE_REQUANTIZE_BLOCK_SIZE is set."""
    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.utils import \
        requantize_and_transpose_int4_weights

    num_experts = 4
    hidden_size = 128
    intermediate_size_per_partition = 64
    in_group_size = 16
    requant_block_size = 64

    layer = FakeRoutedExperts(experts_per_token=2)
    layer.activation = FakeActivation("silu")
    layer._map_global_expert_id_to_local_expert_id = lambda x: x

    weight_quant = FakeQuantArgs(num_bits=4,
                                 strategy="group",
                                 group_size=in_group_size)
    method = VllmCompressedTensorsW4A16MoEMethod(weight_quant, None,
                                                 layer.moe_config)
    method.moe = MagicMock()
    method.moe.tp_size = 1
    method.moe.tp_rank = 0

    extra_weight_attrs = {
        "weight_loader": MagicMock(),
        "intermediate_size_full": intermediate_size_per_partition,
    }

    with patch.object(VllmCompressedTensorsW4A16MoEMethod, "_validate_w4a16_scheme"), \
         patch.object(VllmCompressedTensorsW4A16MoEMethod, "_validate_int32_weight_carriers"):
        method.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=torch.float32,
            **extra_weight_attrs,
        )

    weight_loader = layer.w13_weight_packed.weight_loader

    # w13: in_dim = hidden_size (128), out_dim = 64
    # w13 packed carrier per shard: [64, 128 // 8] = [64, 16]
    # w13 scale per shard: [64, 128 // 16] = [64, 8]
    w13_packed_val = torch.randint(-2147483648,
                                   2147483647, (64, 16),
                                   dtype=torch.int32)
    w13_scale_val = torch.rand(64, 8, dtype=torch.float32) + 0.1

    # w2: in_dim = intermediate_size (64), out_dim = hidden_size (128)
    # w2 packed carrier: [128, 64 // 8] = [128, 8]
    # w2 scale: [128, 64 // 16] = [128, 4]
    w2_packed_val = torch.randint(-2147483648,
                                  2147483647, (128, 8),
                                  dtype=torch.int32)
    w2_scale_val = torch.rand(128, 4, dtype=torch.float32) + 0.1

    for expert_id in range(num_experts):
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w1", expert_id)
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w3", expert_id)
        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w1", expert_id)
        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w3", expert_id)
        weight_loader(layer.w2_weight_packed, w2_packed_val,
                      "w2_weight_packed", "w2", expert_id)
        weight_loader(layer.w2_weight_scale, w2_scale_val, "w2_weight_scale",
                      "w2", expert_id)

    # Save original CPU scratch data (prior to XOR and requant)
    w13_packed_orig = layer.w13_weight_packed._cpu_scratch.data.clone()
    w13_scale_orig = layer.w13_weight_scale._cpu_scratch.data.clone()

    with patch(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.envs.MOE_REQUANTIZE_BLOCK_SIZE",
            str(requant_block_size)
    ), patch(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.prebuild_fused_moe_kernel"
    ):
        method.process_weights_after_loading(layer)

    # Verify shapes:
    # w13_weight_packed: [E, K//8, 2N] = [4, 16, 128]
    # w13_weight_scale:  [E, K//block, 2N] = [4, 2, 128]
    # w2_weight_packed:  [E, N//8, K] = [4, 8, 128]
    # w2_weight_scale:   [E, N//block, K] = [4, 1, 128]
    assert layer.w13_weight_packed.shape == (4, 16, 128)
    assert layer.w13_weight_scale.shape == (4, 2, 128)
    assert layer.w2_weight_packed.shape == (4, 8, 128)
    assert layer.w2_weight_scale.shape == (4, 1, 128)

    # Reference calculation on w13:
    # requantize and transpose directly
    w13_req_packed_expected_kmajor, w13_scale_expected_transposed = requantize_and_transpose_int4_weights(
        w13_packed_orig,
        w13_scale_orig,
        requant_block_size,
        ctypes.c_int32(0x88888888).value,
        in_group_size=in_group_size)

    torch.testing.assert_close(layer.w13_weight_packed.cpu(),
                               w13_req_packed_expected_kmajor.cpu())
    torch.testing.assert_close(layer.w13_weight_scale.cpu(),
                               w13_scale_expected_transposed.cpu())


def test_w4a16_requantize_to_fp4():
    """Verify that INT4 MoE weights are correctly requantized to FP4 (torch.float8_e4m3fn) when MOE_REQUANTIZE_WEIGHT_DTYPE is set."""
    import jax.numpy as jnp

    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.utils import \
        requantize_and_transpose_int4_to_fp4_weights

    num_experts = 4
    hidden_size = 128
    intermediate_size_per_partition = 64
    in_group_size = 16
    requant_block_size = 64

    layer = FakeRoutedExperts(experts_per_token=2)
    layer.activation = FakeActivation("silu")
    layer._map_global_expert_id_to_local_expert_id = lambda x: x

    weight_quant = FakeQuantArgs(num_bits=4,
                                 strategy="group",
                                 group_size=in_group_size)
    method = VllmCompressedTensorsW4A16MoEMethod(weight_quant, None,
                                                 layer.moe_config)
    method.moe = MagicMock()
    method.moe.tp_size = 1
    method.moe.tp_rank = 0

    extra_weight_attrs = {
        "weight_loader": MagicMock(),
        "intermediate_size_full": intermediate_size_per_partition,
    }

    with patch.object(VllmCompressedTensorsW4A16MoEMethod, "_validate_w4a16_scheme"), \
         patch.object(VllmCompressedTensorsW4A16MoEMethod, "_validate_int32_weight_carriers"):
        method.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=torch.float32,
            **extra_weight_attrs,
        )

    weight_loader = layer.w13_weight_packed.weight_loader

    w13_packed_val = torch.randint(-2147483648,
                                   2147483647, (64, 16),
                                   dtype=torch.int32)
    w13_scale_val = torch.rand(64, 8, dtype=torch.float32) + 0.1

    w2_packed_val = torch.randint(-2147483648,
                                  2147483647, (128, 8),
                                  dtype=torch.int32)
    w2_scale_val = torch.rand(128, 4, dtype=torch.float32) + 0.1

    for expert_id in range(num_experts):
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w1", expert_id)
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w3", expert_id)
        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w1", expert_id)
        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w3", expert_id)
        weight_loader(layer.w2_weight_packed, w2_packed_val,
                      "w2_weight_packed", "w2", expert_id)
        weight_loader(layer.w2_weight_scale, w2_scale_val, "w2_weight_scale",
                      "w2", expert_id)

    w13_packed_orig = layer.w13_weight_packed._cpu_scratch.data.clone()
    w13_scale_orig = layer.w13_weight_scale._cpu_scratch.data.clone()

    with patch(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.envs.MOE_REQUANTIZE_WEIGHT_DTYPE",
            "float4_e2m1fn"
    ), patch(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.envs.MOE_REQUANTIZE_BLOCK_SIZE",
            str(requant_block_size)
    ), patch(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16.prebuild_fused_moe_kernel"
    ):
        method.process_weights_after_loading(layer)

    # Verify shapes and types:
    # w13_weight_packed: [E, K//2, 2N] = [4, 64, 128]
    # w13_weight_scale:  [E, K//block, 2N] = [4, 2, 128]
    # w2_weight_packed:  [E, N//2, K] = [4, 32, 128]
    # w2_weight_scale:   [E, N//block, K] = [4, 1, 128]
    assert layer.w13_weight_packed.shape == (4, 64, 128)
    assert layer.w13_weight_scale.shape == (4, 2, 128)
    assert layer.w2_weight_packed.shape == (4, 32, 128)
    assert layer.w2_weight_scale.shape == (4, 1, 128)

    assert layer.w13_weight_packed.dtype == torch.float8_e4m3fn
    assert layer.w2_weight_packed.dtype == torch.float8_e4m3fn
    assert layer._rhs_quant_dtype == jnp.float4_e2m1fn

    # Reference calculation on w13:
    w13_req_packed_expected_kmajor, w13_scale_expected_transposed = requantize_and_transpose_int4_to_fp4_weights(
        w13_packed_orig,
        w13_scale_orig,
        requant_block_size,
        in_group_size=in_group_size)

    torch.testing.assert_close(
        layer.w13_weight_packed.cpu().view(torch.uint8),
        w13_req_packed_expected_kmajor.cpu().view(torch.uint8))
    torch.testing.assert_close(layer.w13_weight_scale.cpu(),
                               w13_scale_expected_transposed.cpu())
