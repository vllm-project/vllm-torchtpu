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
PyTorch MXFP4 Quantization Utilities for TPU.

This module provides pure PyTorch implementations of MXFP4 quantization
operations, to match the JAX reference implementation in
`vllm_torchtpu/layers/common/quantization.py`.

Key functions:
- e8m0_to_fp32: Convert e8m0 (8-bit exponent) scales to float32
- unpack_uint8_to_fp4: Unpack uint8 to FP4 e2m1 float values
- dequantize_mxfp4_packed: Dequantize MXFP4 packed weights
- quantize_tensor_to_fp4: Quantize float tensor to FP4 with block scaling
- pack_fp4_indices: Pack FP4 indices back to uint8

Usage in MXFP4 MoE pipeline:
1. Load checkpoint (block-32 MXFP4)
2. Dequantize to float32 using dequantize_mxfp4_packed()
3. Re-quantize to block-512 using quantize_tensor_to_fp4()
4. Pack indices using pack_fp4_indices()
5. Use in GMM kernel with rhs_scale
"""

from typing import Tuple

import jax.numpy as jnp
import torch

# MXFP4 block size as stored in checkpoints
MXFP4_BLOCK_SIZE = 32

# Target block size for optimized TPU kernels
REQUANTIZED_BLOCK_SIZE = 512

# e8m0 is a bare biased exponent with no sign and no mantissa:
# value = 2^(bits - 127) for bits in [0, 254] (255 encodes NaN). This matches
# jnp.float8_e8m0fnu / torch.float8_e8m0fnu, and `e8m0_to_fp32` is the decoder
# for the uint8 form used throughout this module.
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127

# Same value as jnp.finfo(jnp.float8_e4m3fn).max (448.0).
FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def dtype_max(dtype: jnp.dtype) -> float:
    info = (jnp.iinfo(dtype)
            if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype))
    return float(info.max)


def is_floating_dtype(dtype: torch.dtype) -> bool:
    return torch.is_floating_point(torch.empty((), dtype=dtype))


def _safe_inverse_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return 1 / scale while keeping zero-scale blocks exactly zero."""
    return torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)


def static_per_tensor_quantize_tensor(
    tensor: torch.Tensor,
    quant_dtype: torch.dtype,
    scale: float,
) -> torch.Tensor:
    """Quantize tensor with a fixed scalar scale and clamp to dtype range."""
    if scale == 0.0:
        raise ValueError("Quantization scale must be non-zero.")
    if is_floating_dtype(quant_dtype):
        dtype_info = torch.finfo(quant_dtype)
    else:
        dtype_info = torch.iinfo(quant_dtype)
    quantized = torch.clamp(
        tensor.to(torch.float32) / scale, dtype_info.min, dtype_info.max)
    return quantized.to(quant_dtype)


def quantize_kv(
    quant_dtype: torch.dtype,
    key: torch.Tensor,
    value: torch.Tensor | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Static quantize key/value tensors with per-tensor scales."""
    key = static_per_tensor_quantize_tensor(key, quant_dtype, k_scale)
    if value is None:
        return key, None
    value = static_per_tensor_quantize_tensor(value, quant_dtype, v_scale)
    return key, value


def e8m0_to_fp32(u8: torch.Tensor) -> torch.Tensor:
    """Convert e8m0 (8-bit exponent only) to float32.

    e8m0 format: 8-bit unsigned exponent with bias 127.
    Value = 2^(exponent - 127)

    Args:
        u8: uint8 tensor representing e8m0 exponents.

    Returns:
        float32 tensor with actual scale values.

    FIXME: TPU-specific behavior difference for large exponents (u8 >= 254).
        - u8=254 -> exponent=127 -> 2^127 ≈ 1.7e38
        - On CPU: torch.ldexp correctly returns 1.7e38
        - On TPU: torch.ldexp overflows to inf (due to float32 limits)
        - JAX on TPU: jnp.ldexp correctly returns 1.7e38
        This only affects unrealistic scale values (u8 > 200 is already ~1e22).
    """
    # e8m0 minexp is -127 (same as jnp.float8_e8m0fnu)
    exponents = u8.to(torch.int32) + E8M0_MIN_EXP
    ones = torch.ones_like(u8, dtype=torch.float32)
    result = torch.ldexp(ones, exponents)
    return result


def unpack_uint8_to_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 tensor containing two fp4 (e2m1) values per byte.

    FP4 e2m1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit.
    Representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6

    Packing format: low nibble (bits 0-3) first, then high nibble (bits 4-7).

    Args:
        packed: uint8 tensor of shape [..., N/2].

    Returns:
        float32 tensor of shape [..., N] with unpacked FP4 values.
    """
    # FP4 e2m1 lookup table mapping 4-bit index to float value
    FP4_LUT = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
        -3.0, -4.0, -6.0
    ],
                           dtype=torch.float32,
                           device=packed.device)

    # Extract low and high nibbles
    low_nibble = (packed & 0x0F).to(torch.int64)
    high_nibble = ((packed >> 4) & 0x0F).to(torch.int64)

    # Convert to float using LUT
    low_fp32 = FP4_LUT[low_nibble]
    high_fp32 = FP4_LUT[high_nibble]

    # Interleave: [low0, high0, low1, high1, ...]
    result = torch.stack([low_fp32, high_fp32], dim=-1)
    return result.reshape(*packed.shape[:-1], -1)


def dequantize_tensor(
    tensor_q: torch.Tensor,
    scale: torch.Tensor,
    axis: int = -1,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a quantized tensor with block scaling.

    Applies per-block scale factors to dequantize a tensor.
    Matches the reference `dequantize_tensor` in JAX.

    Args:
        tensor_q: Quantized tensor of shape [...].
        scale: Scale tensor with same ndim but smaller along quantized axes.
        axis: Axis or axes along which quantization was performed.
        out_dtype: Output dtype.

    Returns:
        Dequantized tensor in out_dtype.
    """
    if isinstance(axis, int):
        axis = [axis]

    orig_shape = tensor_q.shape

    if tensor_q.ndim == scale.ndim:
        # Block quantized - need to reshape for broadcasting
        blocked_shape = []
        for i, dim in enumerate(orig_shape):
            if i in [(a + tensor_q.ndim) % tensor_q.ndim for a in axis]:
                num_blocks = scale.shape[i]
                block_size = dim // num_blocks
                blocked_shape.extend([num_blocks, block_size])
            else:
                blocked_shape.append(dim)

        # Calculate the axis positions after reshaping
        axis_normalized = sorted([(a + tensor_q.ndim) % tensor_q.ndim
                                  for a in axis])
        expanded_axis = [1 + n + a for n, a in enumerate(axis_normalized)]

        tensor_q = tensor_q.reshape(blocked_shape)

        # Expand scale to match blocked tensor shape
        for ax in expanded_axis:
            scale = scale.unsqueeze(ax)

    tensor = (tensor_q.to(torch.float32) * scale).to(out_dtype)
    return tensor.reshape(orig_shape)


def quantize_tensor(
    tensor: torch.Tensor,
    quant_dtype: torch.dtype,
    axis: int = -1,
    block_size: int | None = None,
    use_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor with block scaling.

    Args:
        tensor: Float tensor to quantize.
        quant_dtype: Output quantized dtype.
        axis: Axis along which quantization is performed.
        block_size: Number of elements per block. If unset, quantize over the
            entire axis.
        use_ue8m0: If True, use unsigned e8m0 format for scale. Otherwise, use
            float32 scale.

    Returns:
        Quantized tensor and scale tensor. The scale is float32, or uint8
        holding e8m0 biased exponents when `use_ue8m0` is set (decode it with
        `e8m0_to_fp32`, or view it as bytes to pack alongside the values).
    """
    block_size = block_size or tensor.shape[axis]
    if tensor.shape[axis] % block_size != 0:
        raise ValueError(
            "Quantization expects the requested axis to be divisible by "
            f"block_size, got tensor.shape={tuple(tensor.shape)}, axis={axis}, "
            f"block_size={block_size}.")

    if torch.empty((), dtype=quant_dtype).is_floating_point():
        dtype_info = torch.finfo(quant_dtype)
    else:
        dtype_info = torch.iinfo(quant_dtype)

    axis = axis % tensor.ndim
    moved = tensor.movedim(axis, -1)
    num_blocks = moved.shape[-1] // block_size
    blocked = moved.reshape(*moved.shape[:-1], num_blocks, block_size)

    abs_max = blocked.abs().amax(dim=-1, keepdim=True)
    # Use explicit fp32 reciprocal multiply for stable FP8 scale rounding.
    # Plain division can land one ULP lower and change tie-point buckets.
    dtype_max_recip = torch.tensor(1.0 / float(dtype_info.max),
                                   dtype=torch.float32,
                                   device=tensor.device)
    scale = abs_max * dtype_max_recip

    scale_bits = None
    if use_ue8m0:
        # e8m0 carries only an exponent, so snap the scale to a power of two,
        # rounding UP: a smaller scale pushes the block's largest magnitude
        # past dtype_max, where the clamp below silently clips it. Redo the
        # product in fp32 -- for a bf16 scale, log2(0.01575) rounds to exactly
        # -6.0, and the ceil then hands back a scale a full octave too small.
        scale = abs_max.to(torch.float32) * dtype_max_recip
        scale_exp = torch.ceil(torch.log2(scale))
        # log2 is not exactly rounded either, so repair the octave rather than
        # trusting it. An all-zero block gives log2(0) = -inf: exp2(-inf) == 0
        # is not < 0, so it falls through to the clamp, which pins it to the
        # smallest representable exponent.
        scale_exp = torch.where(
            torch.exp2(scale_exp) < scale, scale_exp + 1,
            scale_exp).clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
        scale = torch.exp2(scale_exp)
        # Keep the biased exponent byte, not the power of two itself. This is
        # the exact inverse of `e8m0_to_fp32`, and the clamp above already
        # bounds the byte to [0, 254].
        scale_bits = (scale_exp - E8M0_MIN_EXP).to(torch.uint8)

    # Keep all-zero blocks quantized as exact zeros instead of producing
    # 0 * inf -> NaN during requantization.
    scale_inv = _safe_inverse_scale(scale)

    blocked_q = torch.clamp(blocked * scale_inv,
                            min=float(dtype_info.min),
                            max=float(dtype_info.max))
    if not is_floating_dtype(quant_dtype):
        # A float->int cast truncates toward zero, biasing every element by
        # half an LSB; float dtypes already round in the cast. Half-to-even,
        # matching `quantized_matmul.util.quantize_block`.
        blocked_q = torch.round(blocked_q)
    tensor_q = blocked_q.reshape_as(moved).to(quant_dtype).movedim(-1, axis)
    scale = scale_bits if scale_bits is not None else scale.to(torch.float32)
    scale = scale.squeeze(-1).movedim(-1, axis)
    return tensor_q, scale


def dequantize_mxfp4_packed(
    weight_packed: torch.Tensor,
    scale_u8: torch.Tensor,
    axis: int = -1,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize MXFP4 packed weights to float.

    MXFP4 format:
    - Weights are packed as uint8 (2 x fp4 per byte)
    - Scales are e8m0 stored as uint8
    - Default block size is 32 (32 values share one scale)

    Args:
        weight_packed: Packed weights [E, out_dim, in_dim/2] as uint8.
        scale_u8: Scales [E, out_dim, in_dim/32] as uint8.
        axis: Quantization axis (typically the in_dim axis).
        out_dtype: Output dtype (default float32).

    Returns:
        Dequantized weights [E, out_dim, in_dim] as out_dtype.
    """
    weight_unpacked = unpack_uint8_to_fp4(weight_packed)
    scale_fp32 = e8m0_to_fp32(scale_u8)
    return dequantize_tensor(weight_unpacked, scale_fp32, axis, out_dtype)


def quantize_tensor_to_fp4(
    tensor: torch.Tensor,
    axis: int = -1,
    block_size: int = REQUANTIZED_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize float tensor to FP4 with block scaling.

    Computes per-block scale factors and quantizes each element to the
    nearest FP4 e2m1 value. Matches the reference `quantize_tensor` in JAX.

    Args:
        tensor: Float tensor to quantize.
        axis: Axis or axes along which to compute block scales.
        block_size: Number of elements per block (must divide tensor dimension).

    Returns:
        indices: int8 tensor with FP4 index values (0-15).
        scale: float32 tensor with per-block scales.
    """
    if isinstance(axis, int):
        axis = [axis]

    # FP4 e2m1 representable values
    FP4_VALUES = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
        -3.0, -4.0, -6.0
    ],
                              dtype=torch.float32,
                              device=tensor.device)
    FP4_MAX = 6.0  # Maximum representable magnitude
    FP4_MIN = -6.0

    orig_shape = tensor.shape

    # Reshape to expose block structure
    blocked_shape = []
    for i, dim in enumerate(orig_shape):
        if i in [(a + tensor.ndim) % tensor.ndim for a in axis]:
            if dim % block_size != 0:
                raise ValueError(
                    f"Dimension {i} of size {dim} is not divisible by block_size={block_size}"
                )
            num_blocks = dim // block_size
            blocked_shape.extend([num_blocks, block_size])
        else:
            blocked_shape.append(dim)

    axis_normalized = sorted([a % tensor.ndim for a in axis])
    expanded_axis = [1 + n + a for n, a in enumerate(axis_normalized)]

    tensor_blocked = tensor.reshape(blocked_shape)

    # Compute per-block scale: abs_max / FP4_MAX
    abs_max = tensor_blocked.abs().amax(dim=expanded_axis, keepdim=True)
    scale = abs_max / FP4_MAX

    # Keep all-zero blocks quantized as exact zeros instead of producing
    # 0 * inf -> NaN before the LUT lookup.
    scale_inv = _safe_inverse_scale(scale)

    # Scale and clip to FP4 range
    tensor_scaled = tensor_blocked * scale_inv
    tensor_clipped = tensor_scaled.clamp(FP4_MIN, FP4_MAX)

    # Find nearest FP4 value (quantize by finding closest in LUT)
    tensor_for_compare = tensor_clipped.unsqueeze(-1)
    fp4_for_compare = FP4_VALUES.view([1] * tensor_clipped.ndim + [16])

    distances = (tensor_for_compare - fp4_for_compare).abs()
    indices = distances.argmin(dim=-1).to(torch.int8)

    # Reshape back to original shape
    indices = indices.reshape(orig_shape)

    # Squeeze scale to remove the keepdim axes
    scale = scale.squeeze(tuple(expanded_axis)).to(torch.float32)

    return indices, scale


def pack_fp4_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack FP4 indices (0-15) into uint8 (two per byte).

    Packing format: even indices in low nibble (bits 0-3),
    odd indices in high nibble (bits 4-7).

    Args:
        indices: int8 tensor with shape [..., N] where N is even.

    Returns:
        uint8 tensor with shape [..., N/2].
    """
    if indices.shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension must be even for packing, got {indices.shape[-1]}"
        )

    indices = indices.to(torch.uint8)
    low = indices[..., 0::2]  # Even indices -> low nibble
    high = indices[..., 1::2]  # Odd indices -> high nibble

    # Pack: low in bits 0-3, high in bits 4-7
    packed = low | (high << 4)
    return packed


def fp4_indices_to_float(indices: torch.Tensor) -> torch.Tensor:
    """Convert FP4 indices (0-15) to float values.

    Args:
        indices: int8 tensor with FP4 indices.

    Returns:
        float32 tensor with FP4 values.
    """
    FP4_VALUES = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
        -3.0, -4.0, -6.0
    ],
                              dtype=torch.float32,
                              device=indices.device)

    return FP4_VALUES[indices.to(torch.int64)]
