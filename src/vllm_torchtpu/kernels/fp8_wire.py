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
"""FP8 E4M3 scaling and casts for activations sent between chips.

The fused EP MoE kernel scales per row: the scale rides with the row inside the
kernel, and a row with no usable scale is zeroed whole (see `row_scale`).

hierrs_tc scales per chunk: each scale is its own remote DMA, and the minimum DMA
inner slice is 128 f32 lanes, so a scale per row would add one such transfer per
row. It sums partial results for the group, so like its BF16 wire it passes NaN
through and lets the output show it (see `chunk_scale`).
"""

import jax.numpy as jnp

FP8_E4M3_MAX = 448.0
FP8 = jnp.float8_e4m3fn


def quantize(x_f32, inv_scale):
    return jnp.clip(x_f32 * inv_scale, -FP8_E4M3_MAX, FP8_E4M3_MAX).astype(FP8)


def dequantize_to_bf16(x_fp8, scale):
    return (x_fp8.astype(jnp.float32) * scale).astype(jnp.bfloat16)


def chunk_scale(x_f32):
    """Returns (scale, 1 / scale), with x * (1 / scale) in the FP8 range.

    The scale ignores non-finite values so one cannot poison the chunk; the
    values themselves still propagate.
    """
    amax = jnp.max(jnp.abs(jnp.where(jnp.isfinite(x_f32), x_f32, 0.0)))
    scale = jnp.where(amax == 0.0, 1.0, amax / FP8_E4M3_MAX)
    return scale, 1.0 / scale


def row_scale(amax):
    """The row scale and its inverse, with the rows that have no usable one.

    A row whose maximum is zero has nothing to scale by, and a row whose
    maximum is not finite has nothing USEFUL to scale by: the reciprocal of
    an infinite maximum is zero, so every finite value in that row would
    multiply to zero and only the infinity would survive, as a NaN. That
    leaves a plausible-looking mostly-zero row where the row's real content
    was, and sends an infinite scale over the wire for the combine at the
    destination shard to multiply back in. Both cases are answered the same
    way, by the caller zeroing the row whole against `usable`: an obviously
    empty row rather than a quietly wrong one.
    """
    scale = amax.astype(jnp.float32) / FP8_E4M3_MAX
    usable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(usable, scale, 0.0)
    sinv = jnp.where(usable, 1.0 / jnp.where(usable, scale, 1.0), 0.0)
    return scale, sinv, usable


def quant_apply(x, sinv_native, usable):
    """Quantize a row, or zero it whole where its scale is unusable."""
    return jnp.where(usable, x * sinv_native, jnp.zeros_like(x)).astype(FP8)


def rowquant_fp8(x):
    """Per-row dynamic fp8 quant, reducing and applying in bf16."""
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale, sinv, usable = row_scale(amax)
    return quant_apply(x, sinv.astype(x.dtype), usable), scale
