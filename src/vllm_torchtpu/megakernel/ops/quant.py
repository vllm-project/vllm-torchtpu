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
"""Stateless low-precision NVFP4 and MXFP4 unpack and sub-tiled dot helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .base import BF16, F32, FP8, U32

FP8_E4M3_MAX = 448.0


def _unpack_fp4_words(packed: jax.Array, *, interpret: bool) -> jax.Array:
    if interpret:
        return lax.bitcast_convert_type(packed, jnp.float4_e2m1fn).transpose(0, 2, 1)
    return pltpu.bitcast(packed, jnp.float4_e2m1fn)


def _expand_nvfp4_bscale(
    bscale_u8: jax.Array, tk_raw: int, n_logical: int
) -> jax.Array:
    bs_bf16 = lax.bitcast_convert_type(bscale_u8, FP8).astype(BF16)
    if n_logical // 8 >= 128:
        return bs_bf16.reshape(tk_raw // 8, 1, n_logical)
    return jnp.concatenate(
        [
            jnp.concatenate([bs_bf16[i * 8 + r, :] for r in range(8)], axis=0)[
                None, None, :
            ]
            for i in range(tk_raw // 8)
        ],
        axis=0,
    )


def unpack_nvfp4_tile(
    w_u8: jax.Array, n_logical: int, *, interpret: bool = False
) -> jax.Array:
    """Unpack `[tk_raw, n_raw] uint8` NVFP4 tile into `[2 * tk_raw, n_logical] bf16`."""
    tk_raw = w_u8.shape[0]
    if tk_raw % 8 != 0 or n_logical % 8 != 0:
        raise ValueError(
            f"unpack_nvfp4_tile requires tk_raw % 8 == 0 and n_logical % 8 == 0, "
            f"got tk_raw={tk_raw}, n_logical={n_logical}"
        )
    packed = w_u8[:, :n_logical]
    scale_cols = n_logical // 8
    bscale_u8 = w_u8[:, n_logical : n_logical + scale_cols]
    bscale = _expand_nvfp4_bscale(bscale_u8, tk_raw, n_logical)
    w4 = _unpack_fp4_words(packed, interpret=interpret)
    w_bf16 = w4.astype(BF16).reshape(tk_raw // 8, 16, n_logical)
    return (w_bf16 * bscale).reshape(2 * tk_raw, n_logical)


def unpack_nvfp4_tile_fp8(
    w_u8: jax.Array, n_logical: int, *, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Unpack `[tk_raw, n_raw] uint8` NVFP4 tile into `(w_fp8, col_scale_f32)`."""
    tk_raw = w_u8.shape[0]
    if tk_raw % 8 != 0 or n_logical % 8 != 0:
        raise ValueError(
            f"unpack_nvfp4_tile_fp8 requires tk_raw % 8 == 0 and n_logical % 8 == 0, "
            f"got tk_raw={tk_raw}, n_logical={n_logical}"
        )
    packed = w_u8[:, :n_logical]
    scale_cols = n_logical // 8
    bscale_u8 = w_u8[:, n_logical : n_logical + scale_cols]
    bscale = _expand_nvfp4_bscale(bscale_u8, tk_raw, n_logical)
    s_max = jnp.max(jnp.abs(bscale), axis=0, keepdims=True)
    bscale_norm = jnp.where(s_max > 0, bscale / s_max, jnp.zeros_like(bscale))
    w4 = _unpack_fp4_words(packed, interpret=interpret)
    w_bf16 = w4.astype(BF16).reshape(tk_raw // 8, 16, n_logical)
    w_fp8 = (w_bf16 * bscale_norm).reshape(2 * tk_raw, n_logical).astype(FP8)
    return w_fp8, s_max.reshape(1, n_logical).astype(F32)


def nvfp4_dot_subtiled(
    x_ref: Any,
    w_ref: Any,
    n_logical: int,
    *,
    subtile_raw_rows: int = 64,
    use_fp8_dot: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Compute `x_ref @ dequant(w_ref)` in 64-raw-row subtiles with zero spills."""
    tk_raw = w_ref.shape[0]
    if subtile_raw_rows % 8 != 0 or tk_raw % subtile_raw_rows != 0:
        raise ValueError(
            f"tk_raw={tk_raw} must be divisible by subtile_raw_rows={subtile_raw_rows}"
        )
    n_steps = tk_raw // subtile_raw_rows
    k_step = 2 * subtile_raw_rows
    fp8_active = use_fp8_dot and not interpret

    def _loop_body(s: jax.Array, acc: jax.Array) -> jax.Array:
        r0 = pl.multiple_of(s * subtile_raw_rows, subtile_raw_rows)
        k0 = pl.multiple_of(s * k_step, k_step)
        w_tile = w_ref[pl.ds(r0, subtile_raw_rows), :]
        x_sub = x_ref[:, pl.ds(k0, k_step)]
        if fp8_active:
            w_fp8, w_s = unpack_nvfp4_tile_fp8(w_tile, n_logical, interpret=interpret)
            x_f32 = x_sub.astype(F32)
            x_max = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
            x_s = jnp.where(x_max > 0, x_max / FP8_E4M3_MAX, jnp.ones_like(x_max))
            x_fp8 = (x_f32 / x_s).astype(FP8)
            dot = lax.dot_general(
                x_fp8, w_fp8, (((1,), (0,)), ((), ())), preferred_element_type=F32
            )
            return acc + dot * (x_s * w_s)
        w_bf16 = unpack_nvfp4_tile(w_tile, n_logical, interpret=interpret)
        dot = lax.dot_general(
            x_sub.astype(BF16),
            w_bf16,
            (((1,), (0,)), ((), ())),
            preferred_element_type=F32,
        )
        return acc + dot

    acc0 = jnp.zeros((x_ref.shape[0], n_logical), dtype=F32)
    return lax.fori_loop(0, n_steps, _loop_body, acc0)


def unpack_e8m0_scale(exponents: jax.Array) -> jax.Array:
    """Decode MXFP4 E8M0 unsigned exponent bytes to exact FP32 powers of two."""
    exp_u32 = exponents.astype(U32) & U32(0xFF)
    return lax.bitcast_convert_type(exp_u32 << U32(23), F32)


def unpack_mxfp4_u32_bf16(
    packed_u32: jax.Array,
    scale_e8m0: jax.Array,
    *,
    group_size: int = 32,
    interpret: bool = False,
) -> jax.Array:
    """Unpack `[k_u32, n]` `uint32` MXFP4 + `[k_groups, n]` E8M0 scales to BF16."""
    k_u32, n_logical = packed_u32.shape
    k_logical = 8 * k_u32
    if group_size <= 0 or k_logical % group_size != 0:
        raise ValueError(
            f"k_logical={k_logical} must be divisible by group_size={group_size}"
        )
    k_groups = k_logical // group_size
    if scale_e8m0.shape != (k_groups, n_logical):
        raise ValueError(
            f"expected scale_e8m0 shape {(k_groups, n_logical)}, got {scale_e8m0.shape}"
        )
    w4 = _unpack_fp4_words(packed_u32, interpret=interpret).reshape(
        k_groups, group_size, n_logical
    )
    scales = unpack_e8m0_scale(scale_e8m0)[:, None, :]
    return (w4.astype(F32) * scales).reshape(k_logical, n_logical).astype(BF16)


def unpack_mxfp4_u32_fp8(
    packed_u32: jax.Array,
    scale_e8m0: jax.Array,
    *,
    group_size: int = 32,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Unpack `[k_u32, n]` `uint32` MXFP4 + E8M0 scales to `(w_fp8, col_scale_f32)`."""
    k_u32, n_logical = packed_u32.shape
    k_logical = 8 * k_u32
    if group_size <= 0 or k_logical % group_size != 0:
        raise ValueError(
            f"k_logical={k_logical} must be divisible by group_size={group_size}"
        )
    k_groups = k_logical // group_size
    if scale_e8m0.shape != (k_groups, n_logical):
        raise ValueError(
            f"expected scale_e8m0 shape {(k_groups, n_logical)}, got {scale_e8m0.shape}"
        )
    w4 = _unpack_fp4_words(packed_u32, interpret=interpret).reshape(
        k_groups, group_size, n_logical
    )
    scales = unpack_e8m0_scale(scale_e8m0)
    s_max = jnp.max(scales, axis=0, keepdims=True)
    s_norm = jnp.where(s_max > 0, scales / s_max, jnp.zeros_like(scales))[:, None, :]
    w_fp8 = (w4.astype(F32) * s_norm).reshape(k_logical, n_logical).astype(FP8)
    return w_fp8, s_max.reshape(1, n_logical)
