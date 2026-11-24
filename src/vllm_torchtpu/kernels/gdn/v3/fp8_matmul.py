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
"""FP8 high/low pair matrix-unit helpers and power-of-two column scaling."""

import struct
import types
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

# Dot orders for 1- or 2-part (high, low) operands. Skips the negligible
# low @ low term and orders pairs so lhs[0] stays resident across the first two
# dots and rhs[0] stays resident across the last two.
_PAIR_DOT_ORDERS: Mapping[tuple[int, int], tuple[tuple[int, int], ...]] = (
    types.MappingProxyType(
        {
            (1, 1): ((0, 0),),
            (1, 2): ((0, 0), (0, 1)),
            (2, 1): ((0, 0), (1, 0)),
            (2, 2): ((0, 1), (0, 0), (1, 0)),
        }
    )
)


_MIN_POW2_SCALE = 2.0**-110
_MAX_POW2_SCALE = 2.0**118
# IEEE-754 bfloat16 exponent mask and 2 * exponent-bias constant (254 << 7):
# subtracting exp_bits from _BF16_TWICE_ONE_BITS negates the exponent to form
# the exact reciprocal 1 / scale.
_BF16_EXPONENT_MASK = 0x7F80
_BF16_TWICE_ONE_BITS = 0x7F00


def _ceil_bf16(val: float) -> jax.Array:
    """Rounds a positive scalar up to the smallest bfloat16 >= `val`."""
    # Compute in Python so pow2_scales_from_amax traces cleanly under jax.jit.
    u16 = struct.unpack(">I", struct.pack(">f", val))[0] >> 16
    bf16_val = struct.unpack(">f", struct.pack(">I", u16 << 16))[0]
    if bf16_val < val:
        bf16_val = struct.unpack(">f", struct.pack(">I", (u16 + 1) << 16))[0]
    return jnp.bfloat16(bf16_val)


def pow2_scales_from_amax(
    amax_bf16: jax.Array,
    max_abs: float,
    *,
    min_scale: float = _MIN_POW2_SCALE,
    max_scale: float = _MAX_POW2_SCALE,
) -> tuple[jax.Array, jax.Array]:
    """Computes bfloat16 power-of-two `(scale, inv_scale)` from `amax_bf16`.

    Extracts IEEE-754 exponent bits in `uint16` so `scale * inv_scale == 1.0`
    bit-exact and `x_bf16 * inv_scale` shifts exponents with zero rounding error
    while keeping `|x_bf16 * inv_scale| < max_abs`.

    Args:
      amax_bf16: Non-negative bfloat16 max-abs array.
      max_abs: Strict positive upper bound for `|x_bf16 * inv_scale|`.
      min_scale: Minimum clamped power-of-two scale (keeps scales and reciprocals
        normal in bfloat16).
      max_scale: Maximum clamped power-of-two scale.

    Returns:
      `(scale_bf16, inv_bf16)` power-of-two bfloat16 scale factors matching
      `amax_bf16.shape`.
    """
    if amax_bf16.dtype != jnp.bfloat16:
        raise ValueError(f"amax_bf16 must be bfloat16; got {amax_bf16.dtype}.")
    if max_abs <= 0:
        raise ValueError(f"max_abs must be positive; got {max_abs}.")
    # Rounding 2 / max_abs up in bfloat16 guarantees floor_pow2(amax * factor)
    # stays strictly above amax / max_abs even for non-power-of-two max_abs.
    target = jnp.clip(
        amax_bf16 * _ceil_bf16(2.0 / max_abs),
        jnp.bfloat16(min_scale),
        jnp.bfloat16(max_scale),
    )
    exp_bits = jnp.bitwise_and(
        jax.lax.bitcast_convert_type(target, jnp.uint16),
        jnp.uint16(_BF16_EXPONENT_MASK),
    )
    scale_bf16 = jax.lax.bitcast_convert_type(exp_bits, jnp.bfloat16)
    inv_bf16 = jax.lax.bitcast_convert_type(
        jnp.uint16(_BF16_TWICE_ONE_BITS) - exp_bits, jnp.bfloat16
    )
    return scale_bf16, inv_bf16


def compute_column_pow2_scales(
    x_bf16: jax.Array, max_abs: float
) -> tuple[jax.Array, jax.Array]:
    """Returns per-column power-of-two scales keeping `|x| < max_abs`.

    Reduces along `axis=0` (inside each TPU lane, with no cross-lane traffic) and
    extracts IEEE-754 exponent bits via `pow2_scales_from_amax`.

    Args:
      x_bf16: 2D bfloat16 array to scale per column along `axis=0`.
      max_abs: Strict positive upper bound for `|x_bf16 * inv_scale|`.

    Returns:
      `(scale_bf16, inv_bf16)` power-of-two bfloat16 scale factors of shape
      `[1, cols]`.
    """
    if x_bf16.dtype != jnp.bfloat16:
        raise ValueError(f"x_bf16 must be bfloat16; got {x_bf16.dtype}.")
    return pow2_scales_from_amax(
        jnp.max(jnp.abs(x_bf16), axis=0, keepdims=True), max_abs
    )


def split_high_low(
    x: jax.Array,
    fp8_dtype: jnp.dtype | None,
    *,
    truncate_e4m3: bool = False,
) -> list[jax.Array]:
    """Splits an array into one BF16 array or an FP8 `[high, low]` residual pair.

    Splitting each 16-bit value into `high + low` lets two 8-bit matrix-unit
    operands recover near-bfloat16 precision.

    Args:
      x: Input float32 or bfloat16 array.
      fp8_dtype: Target 8-bit FP8 dtype, or `None` for single-part BF16 mode.
      truncate_e4m3: When `True` and `fp8_dtype` is `float8_e4m3fn`, truncates the
        low 4 bfloat16 mantissa bits for `high` instead of rounding and upcasting
        back to bfloat16.

    Returns:
      `[x_bf16]` when `fp8_dtype` is `None`, or `[high, low]` in `fp8_dtype`.
    """
    x_bf16 = x if x.dtype == jnp.bfloat16 else x.astype(jnp.bfloat16)
    if fp8_dtype is None:
        return [x_bf16]
    if truncate_e4m3 and fp8_dtype == jnp.float8_e4m3fn:
        u16 = jax.lax.bitcast_convert_type(x_bf16, jnp.uint16)
        # 0xFFF0 keeps sign, exponent, and the top 3 mantissa bits (matching e4m3),
        # avoiding a round-and-upcast step before subtracting to form the low part.
        hi_bf16 = jax.lax.bitcast_convert_type(
            jnp.bitwise_and(u16, jnp.uint16(0xFFF0)), jnp.bfloat16
        )
        return [hi_bf16.astype(fp8_dtype), (x_bf16 - hi_bf16).astype(fp8_dtype)]
    hi = x_bf16.astype(fp8_dtype)
    # bf16(x) - hi has at most 4 significant bits, so it is exact in bf16.
    return [hi, (x_bf16 - hi.astype(jnp.bfloat16)).astype(fp8_dtype)]


def dot_general(
    lhs: jax.Array | Sequence[jax.Array],
    rhs: jax.Array | Sequence[jax.Array],
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> jax.Array:
    """Sums matrix-unit (MXU) float32 dot products for 1- or 2-part operands.

    Args:
      lhs: Single array or `[high, low]` sequence from `split_high_low`.
      rhs: Single array or `[high, low]` sequence from `split_high_low`.
      dimension_numbers: Contraction and batch dimensions for
        `jax.lax.dot_general`.

    Returns:
      Float32 sum of the partial matrix products (omitting `low @ low`).
    """
    lhs_parts: Sequence[jax.Array] = lhs if isinstance(lhs, Sequence) else (lhs,)
    rhs_parts: Sequence[jax.Array] = rhs if isinstance(rhs, Sequence) else (rhs,)
    pairs = _PAIR_DOT_ORDERS.get((len(lhs_parts), len(rhs_parts)))
    if pairs is None:
        raise ValueError(
            "Expected 1 or 2 components per operand; got"
            f" ({len(lhs_parts)}, {len(rhs_parts)})."
        )
    first_i, first_j = pairs[0]
    acc = jax.lax.dot_general(
        lhs_parts[first_i],
        rhs_parts[first_j],
        dimension_numbers,
        preferred_element_type=jnp.float32,
    )
    for i, j in pairs[1:]:
        acc = acc + jax.lax.dot_general(
            lhs_parts[i],
            rhs_parts[j],
            dimension_numbers,
            preferred_element_type=jnp.float32,
        )
    return acc
