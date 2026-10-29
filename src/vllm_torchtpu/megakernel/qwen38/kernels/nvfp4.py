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
"""Independent ModelOpt NVFP4 and static-FP8 equations, without Pallas code.

Checkpoint scales multiply decoded values; runtime quantizers use their
reciprocals. NVFP4 payloads pack consecutive contraction coordinates, even
coordinate in the low nibble. See docs/nvfp4.md for the source contract.
"""

import jax
import jax.numpy as jnp

F32 = jnp.float32
BF16 = jnp.bfloat16
FP8 = jnp.float8_e4m3fn


def unpack_e2m1(packed):
    """Decode checkpoint bytes, preserving negative zero, along the last axis."""
    if packed.dtype != jnp.uint8 or packed.ndim < 1:
        raise ValueError("Packed E2M1 weights require a uint8 contraction axis")
    codes = jnp.stack((packed & 15, packed >> 4), axis=-1)
    codes = codes.reshape((*packed.shape[:-1], packed.shape[-1] * 2))
    magnitudes = jnp.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6], BF16)[codes & 7]
    return jnp.where((codes & 8) != 0, -magnitudes, magnitudes)


def round_e2m1(value):
    """Saturating round-to-nearest, ties-to-even on the E2M1 value grid."""
    value = value.astype(F32)
    magnitude = jnp.abs(value)
    rounded = jnp.where(magnitude > 5, 6.0, 4.0)
    rounded = jnp.where(magnitude < 3.5, 3.0, rounded)
    rounded = jnp.where(magnitude <= 2.5, 2.0, rounded)
    rounded = jnp.where(magnitude < 1.75, 1.5, rounded)
    rounded = jnp.where(magnitude <= 1.25, 1.0, rounded)
    rounded = jnp.where(magnitude < 0.75, 0.5, rounded)
    rounded = jnp.where(magnitude <= 0.25, 0.0, rounded)
    return jnp.where(jnp.signbit(value), -rounded, rounded).astype(BF16)


def _scalar(value, name):
    value = jnp.asarray(value, F32)
    if value.size != 1:
        raise ValueError(f"{name} must be a single calibrated scale")
    return value.reshape(())


def quantize_activation(value, input_scale):
    """Return E2M1 values and dynamic E4M3 scales for each 16 inputs.

    ``input_scale`` is the positive calibrated checkpoint dequantization scale.
    Arithmetic uses FP32 division, E4M3 saturation, and E2M1 nearest-even
    rounding. Zero/underflowed block scales produce zero values. This defines
    portable rounding, rather than CUDA's approximate reciprocal instruction.
    """
    if value.ndim < 1 or value.shape[-1] % 16:
        raise ValueError("NVFP4 activations require complete 16-element blocks")
    global_scale = _scalar(input_scale, "input_scale")
    blocks = value.astype(F32).reshape((*value.shape[:-1], -1, 16))
    maximum = jnp.max(jnp.abs(blocks), axis=-1)
    inverse = jax.lax.optimization_barrier(1.0 / global_scale)
    # Preserve the specified two FP32 multiplications. Reassociating the
    # inverse with 1/6 changes E4M3 midpoint decisions (e.g. 200.00002 -> 200).
    local_scale = jax.lax.optimization_barrier(maximum * jnp.float32(1 / 6))
    scale = jnp.clip(inverse * local_scale, 0, 448).astype(FP8)
    scale32 = scale.astype(F32)
    inverse_block = jax.lax.optimization_barrier(jnp.where(scale32 != 0, inverse / scale32, 0))
    normalized = jnp.clip(blocks * inverse_block[..., None], -6, 6)
    return round_e2m1(normalized).reshape(value.shape), scale


def _check_weight(packed, block_scale):
    if (
        packed.ndim != 2
        or packed.dtype != jnp.uint8
        or packed.shape[1] % 8
        or block_scale.shape != (packed.shape[0], packed.shape[1] // 8)
    ):
        raise ValueError("NVFP4 weights require [N,K/2] bytes and [N,K/16] block scales")


def dequantize_weight(packed, block_scale, weight_scale_2):
    """Expand one selected checkpoint matrix into FP32 [N,K] values."""
    _check_weight(packed, block_scale)
    global_scale = _scalar(weight_scale_2, "weight_scale_2")
    decoded = unpack_e2m1(packed).astype(F32)
    scales = jnp.repeat(block_scale.astype(F32) * global_scale, 16, axis=-1)
    return decoded * scales


def _matrix_tiles(source, columns, tile):
    """Fixed M8/N128 MXU contraction, expressed independently in ordinary JAX."""
    logical_shape = source.shape[:-1]
    source = source.reshape((-1, source.shape[-1])).astype(BF16)
    logical_rows = source.shape[0]
    rows = (logical_rows + 7) // 8 * 8
    source = jnp.pad(source, ((0, rows - logical_rows), (0, 0)))
    padded_columns = (columns + 127) // 128 * 128

    def project(index, output):
        # A single output tile otherwise lets XLA fold the exact BF16
        # conversions into an FP8 convolution. TPU same-operand replay shows
        # that lowering changes results; preserve the specified MXU operands.
        lhs, matrix = jax.lax.optimization_barrier((source, tile(index)))
        value = jnp.matmul(
            lhs, matrix, precision=jax.lax.Precision.HIGHEST, preferred_element_type=F32
        )
        # Keep the whole M8 contraction observable before logical-row slicing.
        value = jax.lax.optimization_barrier(value)
        return jax.lax.dynamic_update_slice(output, value, (0, index * 128))

    result = jax.lax.fori_loop(
        0, padded_columns // 128, project, jnp.zeros((rows, padded_columns), F32)
    )
    return result[:logical_rows, :columns].reshape((*logical_shape, columns))


def nvfp4_matmul(value, packed, block_scale, weight_scale_2, input_scale, *, mode="canonical"):
    """Native W4A4 projection using preserved checkpoint payloads and scales.

    Canonical evaluation expands E2M1 times E4M3 locally to exact BF16 values,
    contracts complete K with FP32 accumulation, and applies the two FP32
    global scales afterwards. Native evaluation independently expands FP32
    operands, including their global scales, before a full-width contraction.
    """
    _check_weight(packed, block_scale)
    if value.shape[-1] != packed.shape[1] * 2:
        raise ValueError("NVFP4 projection contraction dimensions differ")
    if mode not in ("canonical", "native"):
        raise ValueError(f"Unknown NVFP4 projection mode: {mode}")
    ga = _scalar(input_scale, "input_scale")
    gw = _scalar(weight_scale_2, "weight_scale_2")
    quantized, activation_scale = quantize_activation(value, ga)
    local_activation = quantized.astype(F32) * jnp.repeat(activation_scale.astype(F32), 16, axis=-1)
    if mode == "native":
        matrix = dequantize_weight(packed, block_scale, gw)
        return jnp.matmul(local_activation * ga, matrix.T, precision=jax.lax.Precision.HIGHEST)
    columns = packed.shape[0]
    padding = (-columns) % 128
    payload = jnp.pad(packed, ((0, padding), (0, 0)))
    scales = jnp.pad(block_scale, ((0, padding), (0, 0)))

    def tile(index):
        raw = jax.lax.dynamic_slice(payload, (index * 128, 0), (128, payload.shape[1]))
        sf = jax.lax.dynamic_slice(scales, (index * 128, 0), (128, scales.shape[1]))
        # Two- and four-bit significands multiply exactly within BF16's eight
        # significant bits; FP32 globals are deliberately excluded here.
        return (
            (unpack_e2m1(raw).astype(F32) * jnp.repeat(sf.astype(F32), 16, axis=-1)).astype(BF16).T
        )

    return _matrix_tiles(local_activation, columns, tile) * (ga * gw)


def static_fp8_matmul(value, weight, input_scale, weight_scale, *, mode="canonical"):
    """Calibrated ModelOpt W8A8 projection, with weights stored as [K,N]."""
    if weight.ndim != 2 or value.shape[-1] != weight.shape[0]:
        raise ValueError("FP8 projection contraction dimensions differ")
    if mode not in ("canonical", "native"):
        raise ValueError(f"Unknown FP8 projection mode: {mode}")
    ga = _scalar(input_scale, "input_scale")
    gw = _scalar(weight_scale, "weight_scale")
    quantized = jnp.clip(value.astype(F32) / ga, -448, 448).astype(FP8)
    if mode == "native":
        return jnp.matmul(
            quantized.astype(F32) * ga,
            weight.astype(F32) * gw,
            precision=jax.lax.Precision.HIGHEST,
        )
    columns = weight.shape[1]
    matrix = jnp.pad(weight, ((0, 0), (0, (-columns) % 128)))

    def tile(index):
        return jax.lax.dynamic_slice(matrix, (0, index * 128), (matrix.shape[0], 128)).astype(BF16)

    return _matrix_tiles(quantized, columns, tile) * (ga * gw)
