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
"""Per-row dynamic fp8 quantization. Every row that moves between shards
is quantized by these lines, whether XLA or Mosaic lowers them."""

import jax.numpy as jnp

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0


def row_scale(row_amax, scale_dtype=jnp.float32):
    """The scale of each row and its inverse, from the row's absolute maximum.

    The scale is rounded to `scale_dtype` before use, so the row and its
    scale stay consistent whatever the rounding. A row whose maximum is zero
    or not finite has no usable scale: both come back as zero, and the
    caller zeroes the whole row against `scalable` rather than sending an
    infinite scale to another shard.

    Returns:
      (scale f32, inverse scale f32, scalable bool), each [rows, 1].
    """
    scale = (row_amax.astype(jnp.float32) / FP8_MAX).astype(scale_dtype)
    scale = scale.astype(jnp.float32)
    scalable = jnp.isfinite(scale) & (scale > 0)
    scale = jnp.where(scalable, scale, 0.0)
    inverse = jnp.where(scalable, 1.0 / jnp.where(scalable, scale, 1.0), 0.0)
    return scale, inverse, scalable


def apply_row_scale(rows, inverse_scale, scalable, dtype=jnp.float32):
    """Quantizes each row by its inverse scale. A row without one is zeroed.

    `dtype` is the element type the product is formed in before the fp8
    cast (Config.elementwise_dtype). In float32 the product of two bf16
    values is exact and is rounded once, so every compilation of these
    lines agrees to the bit. In bfloat16 the vector unit runs at twice
    the rate and the product carries a second rounding that a compiler
    may or may not perform inside a fusion (XLA dropped it for 128-wide
    blocks and kept it for wider ones), so two compilations can differ by
    an fp8 step on a few percent of the values.
    """
    scaled = rows.astype(dtype) * inverse_scale.astype(dtype)
    return jnp.where(scalable, scaled, jnp.zeros_like(scaled)).astype(FP8)


def quantize_rows(rows, scale_dtype=jnp.float32, block=0, dtype=jnp.float32):
    """Per-row fp8 quantization of a [rows, n] array: the maximum in the
    array's own dtype, the scaling product in `dtype` (apply_row_scale).
    Returns (fp8 rows, f32 scales [rows, 1]).

    With `block` (dividing n), one scale per `block` values of each row
    instead, and the scales come back as [rows, n // block]. Written as
    static column slices so the same lines lower inside a kernel.
    """
    if not block:
        row_amax = jnp.max(jnp.abs(rows), axis=-1, keepdims=True)
        scale, inverse, scalable = row_scale(row_amax, scale_dtype)
        return apply_row_scale(rows, inverse, scalable, dtype), scale
    width = rows.shape[-1]
    if width % block:
        raise ValueError(
            f"a row of {width} values is not a whole number of {block}-value blocks"
        )
    quantized, scales = [], []
    for lo in range(0, width, block):
        chunk = rows[:, lo : lo + block]
        amax = jnp.max(jnp.abs(chunk), axis=-1, keepdims=True)
        scale, inverse, scalable = row_scale(amax, scale_dtype)
        quantized.append(apply_row_scale(chunk, inverse, scalable, dtype))
        scales.append(scale)
    return (jnp.concatenate(quantized, axis=-1), jnp.concatenate(scales, axis=-1))
