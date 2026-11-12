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
"""One-pass (fused add +) RMSNorm over row tiles (Pallas).

XLA lowers ``vllm.ir.ops.fused_add_rms_norm`` at prefill shapes as two
fusions: a row reduction (which also writes the new residual) and an apply
pass that re-reads and re-adds both inputs. With the full hidden row resident
in VMEM the whole op is one pass: read ``x`` and ``residual`` once, write the
new residual and the normalized output once.

Numerics follow ``vllm/ir/ops/layernorm.py`` step for step:

    s        = f32(x) + f32(residual)        # unrounded
    residual = bf16(s)
    y        = bf16(bf16(s * rsqrt(mean(s^2) + eps)) * weight)

The only freedom is the f32 summation order of ``mean(s^2)``.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per grid step: one [256, 6144] bf16 tile is 3 MiB, and 256 rows ran
# faster than 128 at 4096 x 6144 on v7x.
DEFAULT_BLOCK_ROWS = 256


def _kernel(*refs, eps: float, has_residual: bool):
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
):
    """Returns ``(y, new_residual)``, or ``y`` alone when ``residual`` is None.

    ``x``/``residual`` are ``[rows, hidden]`` of one dtype; ``weight`` is
    ``[hidden]`` of that dtype. ``rows`` must be a multiple of ``block_rows``.
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
    # 2 (or 4) row tiles double-buffered plus ~4 f32 tile temporaries.
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


def _quant_kernel(*refs, eps: float, emit_y: bool, q_max: float):
    """``_kernel`` plus the per-row activation quant of
    ``quantized_matmul.util.quantize_block`` on the bf16-rounded ``y``:

        amax  = max|y|                       (exact: y is bf16)
        scale = f32(amax) / q_max;  0 -> 1
        y_q   = fp8(clip(f32(y) / scale, -q_max, q_max))
    """
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
    quant_dtype=jnp.float8_e4m3fn,
    emit_y: bool = True,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
):
    """The add + RMSNorm pass that also emits the consumer's per-row fp8
    activation, so the fp8 matmul behind it skips its amax and convert passes
    over ``y``.

    Returns ``(y | None, y_q [rows, hidden], scale f32 [rows, 1], residual)``;
    ``(y_q, scale)`` are bit for bit ``quantize_tensor(y, quant_dtype)``.
    ``y`` is written only with ``emit_y`` (a second consumer needs bf16).
    """
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
        functools.partial(_quant_kernel, eps=eps, emit_y=emit_y, q_max=q_max),
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


def reference(x, residual, weight, eps):
    """``vllm.ir.ops`` numerics in JAX (XLA)."""
    orig = x.dtype
    s = x.astype(jnp.float32)
    if residual is not None:
        s = s + residual.astype(jnp.float32)
        new_res = s.astype(orig)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * jax.lax.rsqrt(var + eps)).astype(weight.dtype) * weight
    y = y.astype(orig)
    return (y, new_res) if residual is not None else y
