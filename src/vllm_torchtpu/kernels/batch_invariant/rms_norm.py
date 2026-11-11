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
"""Batch-invariant RMS normalization and residual-gate kernels on TPU.

The eager torch_tpu `mean` over the hidden dim reduces a row in an order that
depends on the row count and the row offset (4.8e-7 relative in fp32 on v7x),
which flips bf16 roundings downstream. Here every row is reduced inside one
Pallas block of the full hidden width, so its order is fixed by the width.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_BLOCK_BYTES = 2 * 1024 * 1024
_MAX_TILE_M = 512


def _kernel(x_ref, o_ref, *, eps: float):
    x = x_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + jnp.float32(eps))
    o_ref[...] = x * inv


def _kernel_scaled(x_ref, o_ref, *, eps: float, scale: float):
    x = x_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + jnp.float32(eps))
    normed = (x * inv).astype(o_ref.dtype)
    if scale != 1.0:
        normed = (normed.astype(jnp.float32) * jnp.float32(scale)).astype(o_ref.dtype)
    o_ref[...] = normed


def _kernel_o_norm_gate(x_ref, g_ref, o_ref, *, eps: float):
    x = x_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + jnp.float32(eps))
    normed = (x * inv).astype(o_ref.dtype)
    gate = jax.nn.sigmoid(g_ref[...].astype(jnp.float32)).astype(o_ref.dtype)
    o_ref[...] = normed * gate


def _kernel_centered(x_ref, w_ref, o_ref, *, eps: float):
    x = x_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + jnp.float32(eps))
    w = 1.0 + w_ref[...].astype(jnp.float32)
    o_ref[...] = (x * inv * w).astype(o_ref.dtype)


def _kernel_input_scaled(x_ref, w_ref, o_ref, *, eps: float):
    scaled = (
        x_ref[...].astype(jnp.float32) * (1.0 + w_ref[...].astype(jnp.float32))
    ).astype(x_ref.dtype)
    sf = scaled.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(sf * sf, axis=-1, keepdims=True) + jnp.float32(eps))
    o_ref[...] = (sf * inv).astype(o_ref.dtype)


def _kernel_input_scaled_masked(n_ref, x_ref, w_ref, o_ref, *, eps: float, tm: int):
    start = pl.program_id(0) * tm
    n_valid = n_ref[0]

    # Tiles past n_valid alias the last valid block and leave it untouched.
    @pl.when(start < n_valid)
    def _():
        rows = start + jax.lax.broadcasted_iota(jnp.int32, (tm, 1), 0)
        valid = rows < n_valid
        x = jnp.where(valid, x_ref[...], jnp.zeros((), x_ref.dtype))
        scaled = (
            x.astype(jnp.float32) * (1.0 + w_ref[...].astype(jnp.float32))
        ).astype(x_ref.dtype)
        sf = scaled.astype(jnp.float32)
        inv = jax.lax.rsqrt(
            jnp.mean(sf * sf, axis=-1, keepdims=True) + jnp.float32(eps)
        )
        normed = (sf * inv).astype(o_ref.dtype)
        o_ref[...] = jnp.where(valid, normed, jnp.zeros((), o_ref.dtype))


def _kernel_rms_res_gate(res_ref, y_ref, a_ref, b_ref, o_ref, *, eps: float):
    yf = y_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(yf * yf, axis=-1, keepdims=True) + jnp.float32(eps))
    normed = (yf * inv).astype(y_ref.dtype).astype(jnp.float32)
    out = a_ref[...].astype(jnp.float32) * res_ref[...].astype(jnp.float32) + (
        b_ref[...].astype(jnp.float32) * normed
    )
    o_ref[...] = out.astype(o_ref.dtype)


def _kernel_input_scaled_rms_res_gate(
    res_ref, y_ref, w_ref, a_ref, b_ref, o_ref, *, eps: float
):
    scaled = (
        y_ref[...].astype(jnp.float32) * (1.0 + w_ref[...].astype(jnp.float32))
    ).astype(y_ref.dtype)
    sf = scaled.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(sf * sf, axis=-1, keepdims=True) + jnp.float32(eps))
    normed = (sf * inv).astype(y_ref.dtype).astype(jnp.float32)
    out = a_ref[...].astype(jnp.float32) * res_ref[...].astype(jnp.float32) + (
        b_ref[...].astype(jnp.float32) * normed
    )
    o_ref[...] = out.astype(o_ref.dtype)


def row_tile(m: int, width: int) -> int:
    tm = 8
    while tm < m and tm < _MAX_TILE_M and (2 * tm) * width * 4 <= _BLOCK_BYTES:
        tm *= 2
    return tm


@functools.partial(jax.jit, static_argnames=("eps", "tile_m", "interpret"))
def det_rms_normalize(
    x: jax.Array, *, eps: float, tile_m: int | None = None, interpret: bool = False
) -> jax.Array:
    """`x * rsqrt(mean(x^2, -1) + eps)` in fp32 for 2D `x`; row order invariant."""
    m, width = x.shape
    tm = tile_m or row_tile(m, width)
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        grid=(pl.cdiv(m, tm),),
        in_specs=[pl.BlockSpec((tm, width), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tm, width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, width), jnp.float32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x)


@functools.partial(
    jax.jit, static_argnames=("eps", "scale", "out_dtype", "tile_m", "interpret")
)
def det_rms_norm_scaled(
    x: jax.Array,
    *,
    eps: float,
    scale: float = 1.0,
    out_dtype=jnp.bfloat16,
    tile_m: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused RMS norm with optional scalar multiplier and output cast."""
    m, width = x.shape
    tm = tile_m or row_tile(m, width)
    return pl.pallas_call(
        functools.partial(_kernel_scaled, eps=eps, scale=scale),
        grid=(pl.cdiv(m, tm),),
        in_specs=[pl.BlockSpec((tm, width), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((tm, width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, width), jnp.dtype(out_dtype)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x)


@functools.partial(jax.jit, static_argnames=("eps", "tile_m", "interpret"))
def det_o_norm_sigmoid_gate(
    x: jax.Array,
    gate: jax.Array,
    *,
    eps: float,
    tile_m: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused per head `o_norm(x) * sigmoid(gate)` in `x.dtype`."""
    m, width = x.shape
    tm = tile_m or row_tile(m, width)
    return pl.pallas_call(
        functools.partial(_kernel_o_norm_gate, eps=eps),
        grid=(pl.cdiv(m, tm),),
        in_specs=[
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tm, width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, width), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, gate)


@functools.partial(jax.jit, static_argnames=("eps", "out_dtype", "tile_m", "interpret"))
def det_centered_rms_norm(
    x: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    out_dtype=jnp.bfloat16,
    tile_m: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused centered RMSNorm `x * rsqrt(mean(x^2) + eps) * (1 + weight)`."""
    m, width = x.shape
    tm = tile_m or row_tile(m, width)
    w2 = weight.reshape(1, width)
    return pl.pallas_call(
        functools.partial(_kernel_centered, eps=eps),
        grid=(pl.cdiv(m, tm),),
        in_specs=[
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
            pl.BlockSpec((1, width), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tm, width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, width), jnp.dtype(out_dtype)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, w2)


@functools.partial(jax.jit, static_argnames=("eps", "out_dtype", "tile_m", "interpret"))
def det_input_scaled_rms_norm(
    x: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    out_dtype=jnp.bfloat16,
    tile_m: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused InputScaledRMSNorm `rms_norm(round(x * (1 + weight), x.dtype))`."""
    m, width = x.shape
    tm = tile_m or row_tile(m, width)
    w2 = weight.reshape(1, width)
    return pl.pallas_call(
        functools.partial(_kernel_input_scaled, eps=eps),
        grid=(pl.cdiv(m, tm),),
        in_specs=[
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
            pl.BlockSpec((1, width), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tm, width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, width), jnp.dtype(out_dtype)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(x, w2)


@functools.partial(jax.jit, static_argnames=("eps", "out_dtype", "tile_m", "interpret"))
def det_input_scaled_rms_norm_masked(
    x: jax.Array,
    weight: jax.Array,
    n_valid: jax.Array,
    *,
    eps: float,
    out_dtype=jnp.bfloat16,
    tile_m: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """det_input_scaled_rms_norm on rows [0, n_valid).

    Valid rows are bit identical to the unmasked kernel. Rows past n_valid in
    the last valid tile are zero; later rows are left unwritten (undefined),
    so callers must mask them. Tiles past n_valid alias the last valid block,
    which skips both their DMA and their compute.
    """
    m, width = x.shape
    tm = tile_m or row_tile(m, width)
    w2 = weight.reshape(1, width)
    n = jnp.reshape(n_valid.astype(jnp.int32), (1,))

    def row_map(i, n_ref):
        last = jnp.maximum(pl.cdiv(n_ref[0], tm) - 1, 0)
        return (jnp.minimum(i, last), 0)

    return pl.pallas_call(
        functools.partial(_kernel_input_scaled_masked, eps=eps, tm=tm),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(pl.cdiv(m, tm),),
            in_specs=[
                pl.BlockSpec((tm, width), row_map),
                pl.BlockSpec((1, width), lambda i, n_ref: (0, 0)),
            ],
            out_specs=pl.BlockSpec((tm, width), row_map),
        ),
        out_shape=jax.ShapeDtypeStruct((m, width), jnp.dtype(out_dtype)),
        # Revisited output blocks need sequential grid order.
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )(n, x, w2)


@functools.partial(jax.jit, static_argnames=("eps", "out_dtype", "tile_m", "interpret"))
def det_rms_norm_residual_gate(
    residual: jax.Array,
    y: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    *,
    eps: float,
    out_dtype=jnp.float32,
    tile_m: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused `alpha * residual + beta * rms_norm(y)`."""
    m, width = y.shape
    tm = tile_m or row_tile(m, width)
    a2 = alpha.reshape(1, width)
    b2 = beta.reshape(1, width)
    return pl.pallas_call(
        functools.partial(_kernel_rms_res_gate, eps=eps),
        grid=(pl.cdiv(m, tm),),
        in_specs=[
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
            pl.BlockSpec((1, width), lambda i: (0, 0)),
            pl.BlockSpec((1, width), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tm, width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, width), jnp.dtype(out_dtype)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(residual, y, a2, b2)


@functools.partial(jax.jit, static_argnames=("eps", "out_dtype", "tile_m", "interpret"))
def det_input_scaled_rms_norm_residual_gate(
    residual: jax.Array,
    y: jax.Array,
    weight: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    *,
    eps: float,
    out_dtype=jnp.float32,
    tile_m: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused `alpha * residual + beta * input_scaled_rms_norm(y, weight)`."""
    m, width = y.shape
    tm = tile_m or row_tile(m, width)
    w2 = weight.reshape(1, width)
    a2 = alpha.reshape(1, width)
    b2 = beta.reshape(1, width)
    return pl.pallas_call(
        functools.partial(_kernel_input_scaled_rms_res_gate, eps=eps),
        grid=(pl.cdiv(m, tm),),
        in_specs=[
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
            pl.BlockSpec((tm, width), lambda i: (i, 0)),
            pl.BlockSpec((1, width), lambda i: (0, 0)),
            pl.BlockSpec((1, width), lambda i: (0, 0)),
            pl.BlockSpec((1, width), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tm, width), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((m, width), jnp.dtype(out_dtype)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(residual, y, w2, a2, b2)
