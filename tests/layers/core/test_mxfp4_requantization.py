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
Tests for PyTorch MXFP4 requantization functions.

Verifies that PyTorch implementations in `vllm_torchtpu.layers.core.quantization`
match JAX reference behavior for:
- e8m0 to float32 conversion
- uint8 to FP4 unpacking
- MXFP4 dequantization
- FP4 quantization with block scaling
- Full dequant->requant cycle (block-32 to block-512)

Run on CPU (default):
    pytest tests/layers/core/test_mxfp4_requantization.py -v

Run on TPU (when available):
    pytest tests/layers/core/test_mxfp4_requantization.py -v --use-tpu
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

# Import functions under test from production code
from vllm_torchtpu.layers.core.quantization import (  # isort: skip
    dequantize_mxfp4_packed,
    e8m0_to_fp32,
    fp4_indices_to_float,
    pack_fp4_indices,
    quantize_tensor_to_fp4,
    unpack_uint8_to_fp4,
)

# ============================================================================
# JAX REFERENCE IMPLEMENTATION (from legacy vllm_torchtpu)
# These are copied here ONLY for comparison in tests.
# ============================================================================


def jax_e8m0_to_fp32(u8: jax.Array) -> jax.Array:
    """Convert e8m0 (that was bitcasted to u8) into fp32."""
    assert u8.dtype == jnp.uint8
    e8_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    exponents = u8.astype(jnp.int32) + e8_finfo.minexp
    ones = jnp.ones_like(u8, dtype=jnp.float32)
    return jnp.ldexp(ones, exponents)


def jax_u8_unpack_e2m1(u8_packed_e2m1: jax.Array) -> jax.Array:
    """Unpack e2m1 tensor that was packed into u8."""
    assert u8_packed_e2m1.dtype == jnp.uint8
    e2m1 = jax.lax.bitcast_convert_type(u8_packed_e2m1, jnp.float4_e2m1fn)
    return jnp.reshape(e2m1, e2m1.shape[:-2] + (-1,))


def jax_dequantize_tensor(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int = -1,
    out_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Dequantize a quantized tensor with block scaling."""

    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor_q.shape
    if tensor_q.ndim == scale.ndim:
        blocked_shape = [[i] for i in orig_shape]
        for i in axis:
            num_blocks = scale.shape[i]
            block_size = tensor_q.shape[i] // num_blocks
            blocked_shape[i] = (num_blocks, block_size)

        axis = sorted([(i + tensor_q.ndim) % tensor_q.ndim for i in axis])
        axis = [1 + n + i for n, i in enumerate(axis)]
        blocked_shape = list(itertools.chain(*blocked_shape))
        tensor_q = tensor_q.reshape(blocked_shape)

    scale = jnp.expand_dims(scale, axis)
    tensor = (tensor_q.astype(jnp.float32) * scale).astype(out_dtype)
    return tensor.reshape(orig_shape)


def jax_dequantize_mxfp4_packed(
    tensor_q: jax.Array,
    scale: jax.Array,
    axis: int = -1,
    out_dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Dequantize packed mxfp4 tensor - reference implementation."""
    tensor_e2m1 = jax_u8_unpack_e2m1(tensor_q)
    scale_fp32 = jax_e8m0_to_fp32(scale)
    return jax_dequantize_tensor(tensor_e2m1, scale_fp32, axis, out_dtype)


def jax_quantize_tensor(
    dtype: jnp.dtype,
    tensor: jax.Array,
    axis: int = -1,
    block_size: int = 512,
) -> tuple[jax.Array, jax.Array]:
    """Quantize tensor to given dtype with block scaling - reference."""
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor.shape
    block_size_list = [block_size] * len(axis)

    blocked_shape = [[i] for i in orig_shape]
    for i, block in zip(axis, block_size_list):
        num_blocks = tensor.shape[i] // block
        blocked_shape[i] = (num_blocks, block)

    axis = sorted([i % tensor.ndim for i in axis])
    axis = [1 + n + i for n, i in enumerate(axis)]
    blocked_shape = list(itertools.chain(*blocked_shape))
    tensor = tensor.reshape(blocked_shape)

    dtype_info = jnp.finfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    abs_max = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)
    scale = abs_max / dtype_max
    scale_inv = jnp.nan_to_num(1 / scale, jnp.inf)

    tensor_q = jnp.clip(tensor * scale_inv, dtype_min, dtype_max)
    tensor_q = tensor_q.reshape(orig_shape)
    tensor_q = tensor_q.astype(dtype)

    scale = jnp.squeeze(scale, axis).astype(jnp.float32)
    return tensor_q, scale


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestE8m0Conversion:
    """Tests for e8m0 to float32 conversion.

    FIXME: On TPU, these tests fail for extreme exponent values (u8 >= 254):
        - test_e8m0_to_fp32_matches_jax: fails because torch.ldexp(1.0, 127)
          returns inf on TPU but JAX correctly returns 1.7e38
        - test_e8m0_specific_values[254]: same issue

        Root cause: torch.ldexp on TPU overflows to inf for exponent=127,
        while jnp.ldexp handles it correctly. This only affects unrealistic
        scale values that never appear in real MXFP4 models (u8 > 200 is ~1e22).
        Real MXFP4 scales are typically u8=100-160.
    """

    def test_e8m0_to_fp32_matches_jax(self, device):
        """PyTorch e8m0 conversion should match JAX reference.

        FIXME: Fails on TPU for u8=254.
        """
        u8_np = np.arange(256, dtype=np.uint8)

        # JAX reference (runs on TPU automatically when available)
        u8_jax = jnp.array(u8_np)
        jax_result = np.array(jax_e8m0_to_fp32(u8_jax))

        # PyTorch (from production code) - runs on device
        u8_torch = torch.tensor(u8_np, dtype=torch.uint8, device=device)
        torch_result = e8m0_to_fp32(u8_torch).cpu().numpy()

        # Compare finite values only (u8=255 gives inf)
        finite_mask = np.isfinite(jax_result) & np.isfinite(torch_result)
        assert np.allclose(jax_result[finite_mask], torch_result[finite_mask])
        # Check inf values match
        assert np.all(np.isinf(jax_result) == np.isinf(torch_result))

    @pytest.mark.parametrize("u8_val", [0, 1, 127, 128, 253, 254, 255])
    def test_e8m0_specific_values(self, device, u8_val):
        """Test specific u8 values match between JAX and PyTorch.

        FIXME: Fails on TPU for u8=254.
        """
        u8_torch = torch.tensor([u8_val], dtype=torch.uint8, device=device)
        u8_jax = jnp.array([u8_val], dtype=jnp.uint8)

        torch_val = e8m0_to_fp32(u8_torch).cpu().item()
        jax_val = float(jax_e8m0_to_fp32(u8_jax)[0])

        if np.isfinite(jax_val):
            assert np.isclose(torch_val, jax_val)
        else:
            assert np.isinf(torch_val) == np.isinf(jax_val)


class TestFP4Unpacking:
    """Tests for uint8 to FP4 unpacking."""

    def test_unpack_matches_jax(self, device):
        """PyTorch FP4 unpacking should match JAX bitcast."""
        u8_np = np.arange(256, dtype=np.uint8).reshape(16, 16)

        # JAX reference
        u8_jax = jnp.array(u8_np)
        jax_result = np.array(jax_u8_unpack_e2m1(u8_jax).astype(jnp.float32))

        # PyTorch (from production code) - runs on device
        u8_torch = torch.tensor(u8_np, dtype=torch.uint8, device=device)
        torch_result = unpack_uint8_to_fp4(u8_torch).cpu().numpy()

        assert jax_result.shape == torch_result.shape
        np.testing.assert_allclose(torch_result, jax_result, atol=1e-6)

    @pytest.mark.parametrize("byte_val", [0, 1, 15, 16, 128, 255])
    def test_unpack_specific_bytes(self, device, byte_val):
        """Test specific byte values unpack correctly."""
        u8_torch = torch.tensor([[byte_val]], dtype=torch.uint8, device=device)
        u8_jax = jnp.array([[byte_val]], dtype=jnp.uint8)

        torch_result = unpack_uint8_to_fp4(u8_torch).cpu().numpy()
        jax_result = np.array(jax_u8_unpack_e2m1(u8_jax).astype(jnp.float32))

        np.testing.assert_allclose(torch_result, jax_result, atol=1e-6)


class TestMXFP4Dequantization:
    """Tests for full MXFP4 dequantization."""

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 64, 128),
            (2, 32, 64),
            (8, 16, 256),
        ],
    )
    def test_dequantize_matches_jax(self, device, shape):
        """Full dequantization should match JAX reference."""
        E, out_dim, in_dim = shape
        block_size = 32

        np.random.seed(42)
        weight_packed_np = np.random.randint(
            0, 256, size=(E, out_dim, in_dim // 2), dtype=np.uint8
        )
        scale_u8_np = np.random.randint(
            100, 160, size=(E, out_dim, in_dim // block_size), dtype=np.uint8
        )

        # JAX reference
        weight_packed_jax = jnp.array(weight_packed_np)
        scale_u8_jax = jnp.array(scale_u8_np)
        jax_result = np.array(
            jax_dequantize_mxfp4_packed(weight_packed_jax, scale_u8_jax, axis=2)
        )

        # PyTorch (from production code) - runs on device
        weight_packed_torch = torch.tensor(
            weight_packed_np, dtype=torch.uint8, device=device
        )
        scale_u8_torch = torch.tensor(scale_u8_np, dtype=torch.uint8, device=device)
        torch_result = (
            dequantize_mxfp4_packed(weight_packed_torch, scale_u8_torch, axis=2)
            .cpu()
            .numpy()
        )

        np.testing.assert_allclose(torch_result, jax_result, atol=1e-5)


class TestFP4Quantization:
    """Tests for FP4 quantization with block scaling."""

    @pytest.mark.parametrize("block_size", [512, 256])
    def test_quantize_scales_match_jax(self, device, block_size):
        """Quantization scales should match JAX reference."""
        E, out_dim, in_dim = 4, 64, 1024

        np.random.seed(42)
        tensor_np = np.random.randn(E, out_dim, in_dim).astype(np.float32) * 3.0

        # JAX reference
        tensor_jax = jnp.array(tensor_np)
        _, jax_scale = jax_quantize_tensor(
            jnp.float4_e2m1fn, tensor_jax, axis=2, block_size=block_size
        )
        jax_scale_np = np.array(jax_scale)

        # PyTorch (from production code) - runs on device
        tensor_torch = torch.tensor(tensor_np, dtype=torch.float32, device=device)
        _, torch_scale = quantize_tensor_to_fp4(
            tensor_torch, axis=2, block_size=block_size
        )
        torch_scale_np = torch_scale.cpu().numpy()

        np.testing.assert_allclose(torch_scale_np, jax_scale_np, atol=1e-5)

    def test_quantize_values_match_jax(self, device):
        """Quantized values should match JAX reference."""
        E, out_dim, in_dim = 4, 64, 1024
        block_size = 512

        np.random.seed(42)
        tensor_np = np.random.randn(E, out_dim, in_dim).astype(np.float32) * 3.0

        # JAX reference
        tensor_jax = jnp.array(tensor_np)
        jax_quantized, _ = jax_quantize_tensor(
            jnp.float4_e2m1fn, tensor_jax, axis=2, block_size=block_size
        )
        jax_quantized_np = np.array(jax_quantized.astype(jnp.float32))

        # PyTorch (from production code) - runs on device
        tensor_torch = torch.tensor(tensor_np, dtype=torch.float32, device=device)
        torch_indices, _ = quantize_tensor_to_fp4(
            tensor_torch, axis=2, block_size=block_size
        )
        torch_quantized = fp4_indices_to_float(torch_indices)
        torch_quantized_np = torch_quantized.cpu().numpy()

        np.testing.assert_allclose(torch_quantized_np, jax_quantized_np, atol=1e-5)


class TestFP4Packing:
    """Tests for FP4 packing/unpacking roundtrip."""

    def test_pack_unpack_roundtrip(self, device):
        """Packing then unpacking should recover original FP4 values."""
        # Create indices covering all 16 FP4 values
        np.random.seed(42)
        indices_np = np.random.randint(0, 16, size=(4, 64, 512), dtype=np.int8)
        indices = torch.tensor(indices_np, dtype=torch.int8, device=device)

        # Pack to uint8
        packed = pack_fp4_indices(indices)

        # Unpack back to floats
        unpacked_floats = unpack_uint8_to_fp4(packed)

        # Convert original indices to floats for comparison
        original_floats = fp4_indices_to_float(indices)

        np.testing.assert_allclose(
            unpacked_floats.cpu().numpy(), original_floats.cpu().numpy(), atol=1e-6
        )

    def test_pack_shape(self, device):
        """Packed output should have half the last dimension."""
        indices = torch.randint(0, 16, (4, 64, 1024), dtype=torch.int8, device=device)
        packed = pack_fp4_indices(indices)

        assert packed.shape == (4, 64, 512)
        assert packed.dtype == torch.uint8

    @pytest.mark.parametrize(
        "low_idx,high_idx",
        [
            (0, 0),  # 0x00
            (15, 15),  # 0xFF
            (1, 0),  # 0x01
            (0, 1),  # 0x10
            (5, 10),  # 0xA5
        ],
    )
    def test_pack_specific_values(self, device, low_idx, high_idx):
        """Test specific index pairs pack correctly."""
        indices = torch.tensor([[low_idx, high_idx]], dtype=torch.int8, device=device)
        packed = pack_fp4_indices(indices)

        expected = low_idx | (high_idx << 4)
        assert packed.cpu().item() == expected


class TestFullDequantRequantCycle:
    """Tests for complete dequant->requant cycle (block-32 to block-512)."""

    def test_full_cycle_dequant_matches(self, device):
        """Dequantization step should match exactly."""
        E, out_dim, in_dim = 4, 64, 1024
        original_block_size = 32

        np.random.seed(42)
        weight_packed_np = np.random.randint(
            0, 256, size=(E, out_dim, in_dim // 2), dtype=np.uint8
        )
        scale_u8_np = np.random.randint(
            110, 150, size=(E, out_dim, in_dim // original_block_size), dtype=np.uint8
        )

        # JAX
        weight_packed_jax = jnp.array(weight_packed_np)
        scale_u8_jax = jnp.array(scale_u8_np)
        jax_dequant = np.array(
            jax_dequantize_mxfp4_packed(
                weight_packed_jax, scale_u8_jax, axis=2, out_dtype=jnp.float32
            )
        )

        # PyTorch (from production code) - runs on device
        weight_packed_torch = torch.tensor(
            weight_packed_np, dtype=torch.uint8, device=device
        )
        scale_u8_torch = torch.tensor(scale_u8_np, dtype=torch.uint8, device=device)
        torch_dequant = (
            dequantize_mxfp4_packed(weight_packed_torch, scale_u8_torch, axis=2)
            .cpu()
            .numpy()
        )

        np.testing.assert_allclose(torch_dequant, jax_dequant, atol=1e-5)

    def test_full_cycle_high_match_rate(self, device):
        """Full cycle should achieve >99% match rate.

        Note: <1% differences at FP4 boundaries are expected due to
        different tie-breaking strategies between JAX (native cast)
        and PyTorch (LUT argmin). Both are mathematically valid.
        """
        E, out_dim, in_dim = 4, 64, 1024
        original_block_size = 32
        target_block_size = 512

        np.random.seed(42)
        weight_packed_np = np.random.randint(
            0, 256, size=(E, out_dim, in_dim // 2), dtype=np.uint8
        )
        scale_u8_np = np.random.randint(
            110, 150, size=(E, out_dim, in_dim // original_block_size), dtype=np.uint8
        )

        # JAX pipeline
        weight_packed_jax = jnp.array(weight_packed_np)
        scale_u8_jax = jnp.array(scale_u8_np)
        jax_dequant = jax_dequantize_mxfp4_packed(
            weight_packed_jax, scale_u8_jax, axis=2, out_dtype=jnp.float32
        )
        jax_requant, jax_scale = jax_quantize_tensor(
            jnp.float4_e2m1fn, jax_dequant, axis=2, block_size=target_block_size
        )
        jax_requant_np = np.array(jax_requant.astype(jnp.float32))

        # PyTorch pipeline (from production code) - runs on device
        weight_packed_torch = torch.tensor(
            weight_packed_np, dtype=torch.uint8, device=device
        )
        scale_u8_torch = torch.tensor(scale_u8_np, dtype=torch.uint8, device=device)
        torch_dequant = dequantize_mxfp4_packed(
            weight_packed_torch, scale_u8_torch, axis=2
        )
        torch_indices, torch_scale = quantize_tensor_to_fp4(
            torch_dequant, axis=2, block_size=target_block_size
        )
        torch_requant = fp4_indices_to_float(torch_indices)
        torch_requant_np = torch_requant.cpu().numpy()

        # Check match rate
        matching = np.isclose(jax_requant_np, torch_requant_np, atol=1e-6)
        match_rate = matching.mean() * 100

        assert match_rate >= 99.0, (
            f"Match rate {match_rate:.2f}% is below 99% threshold"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
