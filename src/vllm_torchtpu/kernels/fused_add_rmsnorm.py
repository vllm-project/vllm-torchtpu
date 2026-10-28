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
"""One-pass (fused add +) RMSNorm and per-row FP8 quant over row tiles (Pallas).

At prefill/mixed token counts where intermediate activations spill between
large custom kernels, XLA lowers ``fused_add_rms_norm`` as two fusions (a row
reduction writing the residual plus an apply pass that re-reads both inputs),
followed by two more passes (amax + convert) for dynamic per-row FP8
quantization. Keeping the full hidden row in VMEM collapses all four passes
into one HBM read/write pass while matching ``quantize_tensor(y)`` bit-for-bit:

    s        = f32(x) + f32(residual)
    residual = bf16(s)
    y        = bf16(bf16(s * rsqrt(mean(s^2) + eps)) * weight)
    amax     = max|f32(y)|
    scale    = amax / q_max  (0 -> 1)
    y_q      = fp8(clip(f32(y) / scale, -q_max, q_max))
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK_ROWS = 256


def _kernel(*refs, eps: float, has_residual: bool) -> None:
    if has_residual:
        x_ref, r_ref, w_ref, y_ref, ro_ref = refs
        s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
        ro_ref[...] = s.astype(ro_ref.dtype)
    else:
        x_ref, w_ref, y_ref = refs
        s = x_ref[...].astype(jnp.float32)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    normed = (s * jax.lax.rsqrt(var + eps)).astype(y_ref.dtype)
    y_ref[...] = (normed.astype(jnp.float32) * w_ref[...].astype(jnp.float32)).astype(
        y_ref.dtype
    )


def supports(num_rows: int, hidden: int, block_rows: int = DEFAULT_BLOCK_ROWS) -> bool:
    """Whether ``(num_rows, hidden)`` can run the row-tiled Pallas kernel."""
    return num_rows >= block_rows and num_rows % block_rows == 0 and hidden % 128 == 0


@functools.partial(jax.jit, static_argnames=("eps", "block_rows", "interpret"))
def fused_add_rms_norm(
    x: jax.Array,
    residual: jax.Array | None,
    weight: jax.Array,
    *,
    eps: float,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """One-pass (fused add +) RMSNorm over ``[rows, hidden]`` tiles.

    Args:
        x: Input activation ``[rows, hidden]``.
        residual: Optional residual ``[rows, hidden]`` of matching dtype.
        weight: Scale weight ``[hidden]`` of matching dtype.
        eps: Variance epsilon.
        block_rows: Rows per Pallas grid step (must divide ``rows``).
        interpret: Run ``pallas_call`` in interpret mode.

    Returns:
        ``(y, new_residual)`` when ``residual`` is provided, else ``y``.
    """
    rows, hidden = x.shape
    if rows % block_rows:
        raise ValueError(f"{rows=} must be a multiple of {block_rows=}")
    has_residual = residual is not None
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    out_shape = [jax.ShapeDtypeStruct((rows, hidden), x.dtype)]
    out_specs = [row_spec]
    in_specs = [row_spec]
    args = [x]
    if has_residual:
        in_specs.append(row_spec)
        args.append(residual)
        out_shape.append(jax.ShapeDtypeStruct((rows, hidden), x.dtype))
        out_specs.append(row_spec)
    in_specs.append(w_spec)
    args.append(weight.reshape(1, hidden))
    tile = block_rows * hidden * 4
    vmem_limit = min(100 * 1024 * 1024, max(32 * 1024 * 1024, 12 * tile))
    outs = pl.pallas_call(
        functools.partial(_kernel, eps=eps, has_residual=has_residual),
        grid=(rows // block_rows,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        name=f"fused_add_rmsnorm_r{int(has_residual)}_b{block_rows}_h{hidden}",
        interpret=interpret,
    )(*args)
    if has_residual:
        return outs[0], outs[1]
    return outs[0]


def _quant_kernel(*refs, eps: float, emit_y: bool, q_max: float) -> None:
    if emit_y:
        x_ref, r_ref, w_ref, y_ref, q_ref, s_ref, ro_ref = refs
    else:
        x_ref, r_ref, w_ref, q_ref, s_ref, ro_ref = refs
        y_ref = None
    s = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    ro_ref[...] = s.astype(ro_ref.dtype)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    normed = (s * jax.lax.rsqrt(var + eps)).astype(x_ref.dtype)
    y = (normed.astype(jnp.float32) * w_ref[...].astype(jnp.float32)).astype(
        x_ref.dtype
    )
    if emit_y:
        assert y_ref is not None
        y_ref[...] = y
    yf = y.astype(jnp.float32)
    amax = jnp.max(jnp.abs(yf), axis=-1, keepdims=True)
    scale = amax / jnp.float32(q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    s_ref[...] = scale
    q_ref[...] = jnp.clip(yf / scale, -q_max, q_max).astype(q_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=("eps", "quant_dtype", "emit_y", "block_rows", "interpret")
)
def fused_add_rms_norm_quant(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
    emit_y: bool = True,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array | None, jax.Array, jax.Array, jax.Array]:
    """One-pass fused add + RMSNorm + per-row dynamic FP8 quantization.

    Returns ``(y | None, y_q [rows, hidden], scale f32 [rows, 1], new_residual)``
    where ``(y_q, scale)`` match ``quantize_tensor(y, quant_dtype)`` bit-for-bit.
    """
    return _quant_call(
        x,
        residual,
        weight,
        eps=eps,
        quant_dtype=quant_dtype,
        emit_y=emit_y,
        block_rows=block_rows,
        interpret=interpret,
        body=_quant_kernel,
    )


def _quant_call(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array,
    *,
    eps: float,
    quant_dtype: jnp.dtype,
    emit_y: bool,
    block_rows: int,
    interpret: bool,
    body: Callable[..., None],
) -> tuple[jax.Array | None, jax.Array, jax.Array, jax.Array]:
    rows, hidden = x.shape
    if rows % block_rows:
        raise ValueError(f"{rows=} must be a multiple of {block_rows=}")
    q_max = float(jnp.finfo(quant_dtype).max)
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    w_spec = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    s_spec = pl.BlockSpec((block_rows, 1), lambda i: (i, 0))
    out_shape = []
    out_specs = []
    if emit_y:
        out_shape.append(jax.ShapeDtypeStruct((rows, hidden), x.dtype))
        out_specs.append(row_spec)
    out_shape += [
        jax.ShapeDtypeStruct((rows, hidden), quant_dtype),
        jax.ShapeDtypeStruct((rows, 1), jnp.float32),
        jax.ShapeDtypeStruct((rows, hidden), x.dtype),
    ]
    out_specs += [row_spec, s_spec, row_spec]
    tile = block_rows * hidden * 4
    vmem_limit = min(100 * 1024 * 1024, max(32 * 1024 * 1024, 14 * tile))
    outs = pl.pallas_call(
        functools.partial(body, eps=eps, emit_y=emit_y, q_max=q_max),
        grid=(rows // block_rows,),
        in_specs=[row_spec, row_spec, w_spec],
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit,
        ),
        name=f"fused_add_rmsnorm_quant_y{int(emit_y)}_b{block_rows}_h{hidden}",
        interpret=interpret,
    )(x, residual, weight.reshape(1, hidden))
    if emit_y:
        return outs[0], outs[1], outs[2], outs[3]
    return None, outs[0], outs[1], outs[2]


def reference(
    x: jax.Array,
    residual: jax.Array | None,
    weight: jax.Array,
    eps: float,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Reference (pure-XLA) fused add + RMSNorm matching vLLM numerics."""
    orig = x.dtype
    s = x.astype(jnp.float32)
    new_res = None
    if residual is not None:
        s = s + residual.astype(jnp.float32)
        new_res = s.astype(orig)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * jax.lax.rsqrt(var + eps)).astype(weight.dtype) * weight
    y = y.astype(orig)
    return (y, new_res) if new_res is not None else y
