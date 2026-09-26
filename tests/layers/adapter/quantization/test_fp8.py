# Copyright 2025 Google LLC
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
"""
Tests for FP8 quantization weight processing on TPU.

Verifies that process_weights_after_loading correctly transforms FP8
weights and scales for the GMM kernel, and that FP8 linear layers keep
runtime FP8 weights with dynamically quantized activations.

Run on CPU (default):
    pytest tests/layers/adapter/quantization/test_fp8.py -v

Run on TPU (when available):
    pytest tests/layers/adapter/quantization/test_fp8.py -v --use-tpu
"""

import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)


def test_fp8_config_accepts_default_store_dtype():
    config = VllmFp8Config.from_config({"activation_scheme": "dynamic"})
    assert config.store_dtype is None


def test_fp8_config_rejects_store_dtype():
    with pytest.raises(NotImplementedError, match="store_dtype"):
        VllmFp8Config.from_config(
            {
                "activation_scheme": "dynamic",
                "store_dtype": "mxfp4",
            }
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "block_size,shard_size,deepseek_v4",
    [
        ([128, 128], 256, False),
        ([128, 128], 192, False),
        (None, 256, False),
        ([128, 128], 256, True),
    ],
)
def test_fp8_routed_experts_checkpoint_allocation(
    monkeypatch, block_size, shard_size, deepseek_v4
):
    """Exercise upstream construction and loading with TPU checkpoint geometry."""
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        MoEActivation,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    parallel = FusedMoEParallelConfig(
        tp_size=2,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=1,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=256,
        intermediate_size=2 * shard_size,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    expert_map = ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=1,
        global_num_experts=2,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    quant_cls = VllmFp8Config
    if deepseek_v4:
        from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
            VllmDeepseekV4Fp8Config,
        )

        quant_cls = VllmDeepseekV4Fp8Config
    quant = quant_cls(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=block_size,
    )
    quant.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
    )
    with set_current_vllm_config(quant.vllm_config):
        layer = RoutedExperts(
            layer_name="model.layers.0.mlp.experts",
            params_dtype=torch.bfloat16,
            moe_config=config,
            quant_config=quant,
            expert_map_manager=expert_map,
        )

    assert isinstance(layer.quant_method, VllmFp8MoEMethodTPU)
    padded = (shard_size + 127) // 128 * 128 if block_size else shard_size
    assert layer.w13_weight.shape == (2, 2 * padded, 256)
    assert layer.w2_weight.shape == (2, 256, padded)
    assert layer.w13_weight.dtype == layer.w2_weight.dtype == torch.float8_e4m3fn
    suffix = "weight_scale_inv" if block_size else "weight_scale"
    w13_scale = getattr(layer, f"w13_{suffix}")
    w2_scale = getattr(layer, f"w2_{suffix}")
    assert w13_scale.shape == ((2, 2 * padded // 128, 2) if block_size else (2, 2))
    assert w2_scale.shape == ((2, 2, padded // 128) if block_size else (2,))
    assert w13_scale.dtype == w2_scale.dtype == torch.float32
    assert layer.w13_weight.weight_loader == layer.weight_loader
    assert w13_scale.weight_loader == layer.weight_loader
    assert layer.w13_input_scale is None and layer.w2_input_scale is None
    if block_size and shard_size == padded:
        # The second TP rank loads the original checkpoint block grid without
        # native GPU scale refinement. Distinct columns expose wrong slicing.
        scales = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        layer.weight_loader(w2_scale, scales, "w2_weight_scale_inv", "w2", 0)
        torch.testing.assert_close(w2_scale[0], scales[:, 2:])


class FakeQuant:
    """Minimal quant config for testing."""

    def __init__(
        self,
        weight_block_size=None,
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=True,
    ):
        self.weight_block_size = weight_block_size
        self.activation_scheme = activation_scheme
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized


class FakeMoEConfig:
    """Minimal MoE config for testing."""

    def __init__(self, has_bias=False, experts_per_token=2):
        self.has_bias = has_bias
        self.experts_per_token = experts_per_token


class FakeActivation:
    """Mimics MoEActivation enum."""

    def __init__(self, value):
        self.value = value


class FakeFusedMoELayer(torch.nn.Module):
    """Minimal FusedMoE-like layer for testing process_weights_after_loading."""

    def __init__(
        self,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        activation="silu",
        has_bias=False,
        experts_per_token=2,
    ):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        self.w13_weight_scale_inv = torch.nn.Parameter(w13_scale, requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(w2_scale, requires_grad=False)
        self.activation = FakeActivation(activation)
        self.moe_config = FakeMoEConfig(
            has_bias=has_bias, experts_per_token=experts_per_token
        )

    # Make isinstance(layer, FusedMoE) work via duck typing in the assert
    def __class_getitem__(cls, item):
        return cls


class TestFp8MoEScaleReshape:
    """Tests for block-quantized FP8 MoE scale reshaping."""

    @pytest.mark.parametrize(
        "num_experts,intermediate,hidden,block_size",
        [
            (4, 256, 512, 128),
            (8, 384, 768, 128),
            (2, 128, 256, 128),
        ],
    )
    def test_scale_output_shape(
        self, device, num_experts, intermediate, hidden, block_size
    ):
        """Verify scales are reshaped to [E, in/B, 1, out_dim] for GMM."""
        E, inter, H, B = num_experts, intermediate, hidden, block_size

        # Input scale shape: [E, 2*inter/B, H/B] (w13) and [E, H/B, inter/B] (w2)
        w13_scale = torch.randn(E, 2 * inter // B, H // B, device=device)
        w2_scale = torch.randn(E, H // B, inter // B, device=device)

        # Simulate the reshape logic from process_weights_after_loading
        # Step 1: transpose [E, out/B, in/B] -> [E, in/B, out/B]
        w13_out = w13_scale.transpose(1, 2)
        # Step 2: expand by block_h then reshape
        w13_out = (
            w13_out.unsqueeze(-1)
            .expand(*w13_out.shape, B)
            .reshape(w13_out.shape[0], w13_out.shape[1], -1)
        )
        # Step 3: unsqueeze for GMM
        w13_out = w13_out.unsqueeze(2).to(torch.float32)

        # Expected: [E, H/B, 1, 2*inter]
        assert w13_out.shape == (E, H // B, 1, 2 * inter)

        # Same for w2
        w2_out = w2_scale.transpose(1, 2)
        w2_out = (
            w2_out.unsqueeze(-1)
            .expand(*w2_out.shape, B)
            .reshape(w2_out.shape[0], w2_out.shape[1], -1)
        )
        w2_out = w2_out.unsqueeze(2).to(torch.float32)

        assert w2_out.shape == (E, inter // B, 1, H)

    def test_scale_values_are_repeated(self, device):
        """Each block scale should be repeated block_h times in output dim."""
        E, B = 2, 128
        out_blocks, in_blocks = 4, 2  # out=512, in=256

        scale = torch.arange(
            E * out_blocks * in_blocks, dtype=torch.float32, device=device
        ).reshape(E, out_blocks, in_blocks)

        # Apply reshape
        out = scale.transpose(1, 2)
        out = (
            out.unsqueeze(-1)
            .expand(*out.shape, B)
            .reshape(out.shape[0], out.shape[1], -1)
        )
        out = out.unsqueeze(2)

        # Shape: [E, in_blocks, 1, out_blocks * B]
        assert out.shape == (E, in_blocks, 1, out_blocks * B)

        # Check each block_h-sized chunk has the same value
        for e in range(E):
            for ib in range(in_blocks):
                for ob in range(out_blocks):
                    chunk = out[e, ib, 0, ob * B : (ob + 1) * B]
                    assert torch.all(chunk == chunk[0]), (
                        f"Scale not constant within block at e={e}, ib={ib}, ob={ob}"
                    )


class TestFp8LinearRuntimeQuant:
    """Tests for ullm-style runtime FP8 dense linear weights."""

    def test_block_dequant_shape(self, device):
        """Runtime FP8 weight should preserve original shape."""
        out_dim, in_dim = 256, 512
        block_h, block_w = 128, 128

        weight_fp8 = torch.randn(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(
            out_dim // block_h, in_dim // block_w, device=device, dtype=torch.float32
        )

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale_inv, requires_grad=False)

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        # Canonical (k, n): the runtime matmul is (m, k) @ (k, n), so the
        # checkpoint's [n_out, n_in] weight is stored transposed.
        assert layer.weight.shape == (in_dim, out_dim)
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (out_dim,)
        assert layer.weight_scale.dtype == torch.float32
        assert not hasattr(layer, "weight_scale_inv")

    def test_block_dequant_applies_scales(self, device):
        """Block scales should actually scale the weights."""
        out_dim, in_dim = 256, 256
        block_h, block_w = 128, 128

        # Use ones as FP8 weight — after dequant, result should equal scale
        weight_fp8 = torch.ones(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        # Set different scales per block
        scale_inv = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], device=device, dtype=torch.float32
        )

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(scale_inv, requires_grad=False)

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        # Weight is stored (k, n), so it indexes [in, out] and the
        # per-output-channel scale broadcasts along the trailing axis.
        w = layer.weight.float() * layer.weight_scale[None, :]
        # fp8 ones = 1.0, so each block should be scaled by its scale_inv.
        # w[i, o] == w_checkpoint[o, i], so the off-diagonal blocks swap
        # relative to the checkpoint's [out, in] grid:
        # scale_inv[0,0] -> w[:128, :128]  ~1.0
        # scale_inv[1,0] -> w[:128, 128:]  ~3.0
        # scale_inv[0,1] -> w[128:, :128]  ~2.0
        # scale_inv[1,1] -> w[128:, 128:]  ~4.0
        assert torch.allclose(
            w[:128, :128].mean(), torch.tensor(1.0, device=device), rtol=0.06, atol=0.05
        )
        assert torch.allclose(
            w[:128, 128:].mean(), torch.tensor(3.0, device=device), rtol=0.06, atol=0.05
        )
        assert torch.allclose(
            w[128:, :128].mean(), torch.tensor(2.0, device=device), rtol=0.06, atol=0.05
        )
        assert torch.allclose(
            w[128:, 128:].mean(), torch.tensor(4.0, device=device), rtol=0.06, atol=0.05
        )

    def test_per_tensor_dequant(self, device):
        """Per-tensor (non-block) dequant should scale entire weight."""
        out_dim, in_dim = 64, 128

        weight_fp8 = torch.ones(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor([2.5], device=device, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=None))
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (out_dim,)
        # fp8 ones * 2.5 should be ~2.5
        # (k, n) storage: scale is per output channel, so it broadcasts
        # along the trailing axis.
        runtime_deq = layer.weight.float() * layer.weight_scale[None, :]
        assert torch.allclose(
            runtime_deq.mean(), torch.tensor(2.5, device=device), atol=0.1
        )

    def test_per_tensor_dequant_with_logical_widths(self, device):
        """Per-tensor dequant with multiple scales should use logical_widths repeat_interleave."""
        # Simulate QKVParallelLinear with logical_widths [64, 32, 32] (total out_dim = 128)
        logical_widths = [64, 32, 32]
        out_dim = sum(logical_widths)
        in_dim = 128

        weight_fp8 = torch.ones(
            out_dim, in_dim, device=device, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        scale = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.logical_widths = logical_widths

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=None))
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (out_dim,)

        w = layer.weight.float() * layer.weight_scale[:, None]
        assert torch.allclose(w[:64].mean(), torch.tensor(1.0, device=device), atol=0.1)
        assert torch.allclose(
            w[64:96].mean(), torch.tensor(2.0, device=device), atol=0.1
        )
        assert torch.allclose(w[96:].mean(), torch.tensor(3.0, device=device), atol=0.1)

    def test_apply_is_linear(self, device):
        """apply() should use the runtime FP8 quantized matmul."""
        if device.type != "tpu":
            pytest.skip("Pallas quantized matmul bridge requires TPU.")

        out_dim, in_dim = 64, 128
        batch = 4

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[128, 128]))

        layer = torch.nn.Module()
        # apply() consumes the canonical (k, n) layout that
        # process_weights_after_loading produces, i.e. [n_in, n_out].
        layer.weight = torch.nn.Parameter(
            torch.ones(in_dim, out_dim, device=device, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(out_dim, device=device, dtype=torch.float32), requires_grad=False
        )

        x = torch.ones(batch, in_dim, device=device, dtype=torch.bfloat16)
        result = method.apply(layer, x)

        assert result.shape == (batch, out_dim)
        expected = torch.full(
            (batch, out_dim), float(in_dim), device=device, dtype=torch.bfloat16
        )
        assert torch.allclose(result, expected, rtol=0.01, atol=0.01)

    def test_blockwise_runtime_scale_shape(self, device, monkeypatch):
        """ullm blockwise env gate changes dense-linear runtime scale layout."""
        monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")

        out_dim, in_dim = 256, 256
        block_h, block_w = 128, 128

        layer = torch.nn.Module()
        layer.logical_widths = None
        layer.weight = torch.nn.Parameter(
            torch.ones(out_dim, in_dim, device=device, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.ones(
                out_dim // block_h,
                in_dim // block_w,
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        method = VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[block_h, block_w]))
        method.process_weights_after_loading(layer)

        assert layer.weight.dtype == torch.float8_e4m3fn
        # gmm_v2 rhs_scale layout: [size_group, num_blocks, 1, out_size].
        assert layer.weight_scale.shape == (1, in_dim // 128, 1, out_dim)

    def test_blockwise_kernel_requires_block_size(self, monkeypatch):
        """Match ullm: enabling the blockwise kernel requires block size."""
        monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
        monkeypatch.delenv("REQUANTIZE_BLOCK_SIZE", raising=False)

        with pytest.raises(ValueError, match="REQUANTIZE_BLOCK_SIZE"):
            VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[128, 128]))

    def test_block_size_requires_blockwise_kernel(self, monkeypatch):
        """Match ullm: REQUANTIZE_BLOCK_SIZE only applies to the kernel path."""
        monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")

        with pytest.raises(ValueError, match="Blockwise quantization"):
            VllmFp8LinearMethodTPU(FakeQuant(weight_block_size=[128, 128]))

    def test_blockwise_scale_output_dim_validated(self):
        """Blockwise matmul should reject scales with the wrong output dim."""
        x = jnp.ones((2, 4), dtype=jnp.bfloat16)
        w_q = jnp.ones((4, 3), dtype=jnp.float8_e4m3fn)  # (k, n)
        w_scale = jnp.ones((1, 1, 1, 2), dtype=jnp.float32)  # n=2 != 3

        with pytest.raises(ValueError, match="output dim"):
            _quantized_matmul_jax(x, w_q, w_scale)

    def test_blockwise_scale_block_count_validated(self):
        """Blockwise matmul should reject ambiguous scale block counts."""
        x = jnp.ones((2, 5), dtype=jnp.bfloat16)
        w_q = jnp.ones((5, 3), dtype=jnp.float8_e4m3fn)  # (k, n), k=5
        w_scale = jnp.ones((1, 2, 1, 3), dtype=jnp.float32)  # 5 % 2 != 0

        with pytest.raises(ValueError, match="divisible by block scale count"):
            _quantized_matmul_jax(x, w_q, w_scale)


class TestFp8WeightTranspose:
    """Tests for weight transpose in MoE processing."""

    def test_weights_transposed(self, device):
        """Weights should be transposed from [E, out, in] to [E, in, out]."""
        E, inter, H = 4, 256, 512

        w13 = torch.randn(E, 2 * inter, H, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(E, H, inter, device=device, dtype=torch.bfloat16)

        w13_t = w13.transpose(1, 2).contiguous()
        w2_t = w2.transpose(1, 2).contiguous()

        assert w13_t.shape == (E, H, 2 * inter)
        assert w2_t.shape == (E, inter, H)
        # Verify the transpose is correct
        assert torch.equal(w13_t[0, :, 0], w13[0, 0, :])

    def test_swigluoai_deinterleave(self, device):
        """swigluoai activation should de-interleave w13 weights."""
        E, inter, H = 2, 128, 256

        # Create w13 with known interleaved pattern
        w13 = torch.zeros(E, 2 * inter, H, device=device, dtype=torch.bfloat16)
        # Even rows (gate/w1) = 1.0, odd rows (up/w3) = 2.0
        w13[:, ::2, :] = 1.0
        w13[:, 1::2, :] = 2.0

        # De-interleave
        w1 = w13[:, ::2, :]
        w3 = w13[:, 1::2, :]
        w13_deinterleaved = torch.cat([w1, w3], dim=1)

        # First half should be all 1.0 (w1), second half all 2.0 (w3)
        assert torch.all(w13_deinterleaved[:, :inter, :] == 1.0)
        assert torch.all(w13_deinterleaved[:, inter:, :] == 2.0)

    def test_swigluoai_deinterleave_scale(self, device):
        """swigluoai should de-interleave w13_scale to match w13."""
        E, inter = 2, 128

        # Create scale with known interleaved pattern: one scale per row
        w13_scale = torch.zeros(E, 2 * inter, 1, device=device)
        w13_scale[:, ::2, :] = 1.0  # w1 scales
        w13_scale[:, 1::2, :] = 2.0  # w3 scales

        s1 = w13_scale[:, ::2, :]
        s3 = w13_scale[:, 1::2, :]
        w13_scale_deinterleaved = torch.cat([s1, s3], dim=1)

        # First half should be w1 scales (1.0), second half w3 scales (2.0)
        assert torch.all(w13_scale_deinterleaved[:, :inter, :] == 1.0)
        assert torch.all(w13_scale_deinterleaved[:, inter:, :] == 2.0)


class TestOnlineFp8Quantization:
    """Tests for _quantize_bf16_moe_weights."""

    def test_roundtrip_error(self, device):
        """FP8 quantize -> dequantize should have small error."""
        E, inter, H = 4, 64, 128
        w13 = torch.randn(E, 2 * inter, H, device=device, dtype=torch.float32)

        from vllm_torchtpu.layers.core.quantization import quantize_tensor

        w13_q, w13_s = quantize_tensor(w13, quant_dtype=torch.float8_e4m3fn, axis=-1)
        w13_deq = w13_q.to(torch.float32) * w13_s

        rel_err = (w13 - w13_deq).abs().mean() / (w13.abs().mean() + 1e-8)
        assert rel_err < 0.05, f"Roundtrip relative error too high: {rel_err:.4f}"

    def test_output_shapes(self, device):
        """_quantize_bf16_moe_weights should produce correct shapes."""
        from unittest.mock import MagicMock

        from vllm.model_executor.layers.fused_moe import RoutedExperts

        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _quantize_bf16_moe_weights,
        )

        E, inter, H = 4, 64, 128
        layer = MagicMock(spec=RoutedExperts)
        layer.w13_weight = torch.nn.Parameter(
            torch.randn(E, 2 * inter, H, dtype=torch.bfloat16)
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.randn(E, H, inter, dtype=torch.bfloat16)
        )

        w13, w13_s, w2, w2_s, dtype_name, block_size = _quantize_bf16_moe_weights(
            layer, activation="silu"
        )

        # _quantize_and_format_moe_weights rounds the intermediate size up to
        # a multiple of 128 before quantizing, padding w13/w2 accordingly.
        aligned_inter = (inter + 127) // 128 * 128

        # Weights should be transposed: [E, in, out]
        assert w13.shape == (E, H, 2 * aligned_inter)
        assert w2.shape == (E, aligned_inter, H)
        assert w13.dtype == torch.float8_e4m3fn
        assert w2.dtype == torch.float8_e4m3fn

        # Scales should be 4D: [E, num_blocks, 1, N]
        assert w13_s.shape == (E, 1, 1, 2 * aligned_inter)
        assert w2_s.shape == (E, 1, 1, H)
        assert w13_s.dtype == torch.float32

    def test_custom_routing_called(self):
        """FP8 apply_monolithic should use custom_routing_function when present."""
        from unittest.mock import MagicMock, patch

        from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

        # Create mock layer with custom routing
        layer = MagicMock()
        layer.moe_config.experts_per_token = 2
        layer.moe_config.moe_parallel_config.use_ep = False
        layer.renormalize = True

        mock_routing = MagicMock(
            return_value=(torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int32))
        )
        layer.custom_routing_function = mock_routing

        method = MagicMock(spec=VllmFp8MoEMethodTPU)
        # MagicMock does not inherit the class attribute's False value.
        method._use_adaptive_fused_moe = False
        method._tpu_activation_str = "silu"
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        router_logits = torch.randn(4, 8, dtype=torch.bfloat16)

        # Call the real apply_monolithic
        with patch(
            "vllm_torchtpu.layers.adapter.quantization.fp8.fused_moe_gmm",
            return_value=x,
        ):
            VllmFp8MoEMethodTPU.apply_monolithic(method, layer, x, router_logits)

        mock_routing.assert_called_once()


class FakeInt8MoEConfig:
    """The `moe_config` attributes `_process_fp8_moe_weights` actually reads."""

    def __init__(self, intermediate):
        self.intermediate_size_per_partition = intermediate
        self.intermediate_size_per_partition_unpadded = intermediate
        self.tp_size = 1
        self.tp_rank = 0
        self.has_bias = False


class TestInt8MoERequantization:
    """`MOE_REQUANTIZE_WEIGHT_DTYPE=int8` end to end through the loader.

    int8 is a supported value of that env var but no recipe, CI config or other
    test exercises it, which is how the truncating cast in `quantize_tensor`
    survived. This runs the real `_process_fp8_moe_weights`. CPU-only, so it
    takes no `device` fixture.
    """

    E, INTER, H, BLOCK = 2, 128, 384, 128

    def _layer(self):
        gen = torch.Generator().manual_seed(0)
        E, inter, H, B = self.E, self.INTER, self.H, self.BLOCK
        layer = FakeFusedMoELayer(
            (torch.randn(E, 2 * inter, H, generator=gen) * 40).to(torch.float8_e4m3fn),
            (torch.randn(E, H, inter, generator=gen) * 40).to(torch.float8_e4m3fn),
            torch.rand(E, 2 * inter // B, H // B, generator=gen) + 0.5,
            torch.rand(E, H // B, inter // B, generator=gen) + 0.5,
        )
        layer.moe_config = FakeInt8MoEConfig(inter)
        return layer

    def test_int8_requant_produces_int8_weights_rounded_to_nearest(self, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            _process_fp8_moe_weights,
        )
        from vllm_torchtpu.layers.core.quantization import dequantize_tensor

        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "int8")
        layer = self._layer()

        w13, w13_scale, w2, _, dtype_name, block_size = _process_fp8_moe_weights(
            layer, weight_block_size=(self.BLOCK, self.BLOCK), activation="silu"
        )

        assert dtype_name == "int8" and block_size is None
        assert w13.dtype == torch.int8 and w2.dtype == torch.int8
        # The GMM kernel wants [E, in, out]; the loader transposes on the way.
        assert w13.shape == (self.E, self.H, 2 * self.INTER)
        assert w2.shape == (self.E, self.INTER, self.H)

        # Per-channel scale over the contracting axis, broadcast back to [E, out, in].
        scale = w13_scale.reshape(self.E, 1, -1).transpose(1, 2)
        reference = dequantize_tensor(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            axis=(1, 2),
            out_dtype=torch.float32,
        )
        err_lsb = (w13.float().transpose(1, 2) * scale - reference).abs() / scale
        # Round-to-nearest bounds this by half an LSB; truncation gives a full one.
        assert err_lsb.max().item() <= 0.5 + 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestMoERequantizationChunks:
    """Requantizing a few experts at a time must give the whole-tensor
    result."""

    @staticmethod
    def _layer(device, dtype, num_experts=6, inter=256, hidden=512, block=128):
        from types import SimpleNamespace

        def param(t):
            return torch.nn.Parameter(t, requires_grad=False)

        w13 = torch.randn(
            num_experts, 2 * inter, hidden, device=device, dtype=torch.bfloat16
        ).to(dtype)
        w2 = torch.randn(
            num_experts, hidden, inter, device=device, dtype=torch.bfloat16
        ).to(dtype)
        layer = SimpleNamespace(
            w13_weight=param(w13),
            w2_weight=param(w2),
            w13_weight_scale_inv=param(
                torch.rand(
                    num_experts, 2 * inter // block, hidden // block, device=device
                )
                + 0.5
            ),
            w2_weight_scale_inv=param(
                torch.rand(num_experts, hidden // block, inter // block, device=device)
                + 0.5
            ),
            moe_config=SimpleNamespace(
                intermediate_size_per_partition=inter,
                intermediate_size_per_partition_unpadded=inter,
                tp_size=1,
                tp_rank=0,
            ),
        )
        # w13 chunks of two experts: 2 * (2*inter*hidden) float32 bytes
        return layer, 2 * 2 * inter * hidden * 4

    @staticmethod
    def _same(whole, chunked):
        for a, b in zip(whole[:4], chunked[:4]):
            assert a.shape == b.shape
            assert torch.equal(a.float(), b.float())

    def test_fp8_checkpoint(self, device, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

        torch.manual_seed(0)
        layer, chunk_bytes = self._layer(device, torch.float8_e4m3fn)
        kwargs = dict(weight_block_size=(128, 128), activation="silu")
        whole = fp8_mod._process_fp8_moe_weights(layer, **kwargs)
        monkeypatch.setattr(fp8_mod, "_REQUANT_CHUNK_BYTES", chunk_bytes)
        chunked = fp8_mod._process_fp8_moe_weights(layer, **kwargs)
        self._same(whole, chunked)

    def test_bf16_checkpoint(self, device, monkeypatch):
        from vllm_torchtpu.layers.adapter.quantization import fp8 as fp8_mod

        torch.manual_seed(0)
        layer, chunk_bytes = self._layer(device, torch.bfloat16)
        whole = fp8_mod._quantize_bf16_moe_weights(layer, activation="silu")
        monkeypatch.setattr(fp8_mod, "_REQUANT_CHUNK_BYTES", chunk_bytes)
        chunked = fp8_mod._quantize_bf16_moe_weights(layer, activation="silu")
        self._same(whole, chunked)
