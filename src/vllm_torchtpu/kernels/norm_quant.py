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
"""Sandwich RMSNorm + residual + per-token FP8 quantization Pallas kernels."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import fit_block_rows, vmem_limit_bytes

_BLOCK_ROWS = 256


def _rms_norm(x: jax.Array, w: jax.Array, eps: float, dtype: jnp.dtype) -> jax.Array:
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(var + eps) * w).astype(dtype).astype(jnp.float32)


def _kernel(*refs, eps, has_scalar, quant_dtype, bytes_out):
    scalar_ref = None
    if has_scalar:
        x_ref, r_ref, w_post_ref, w_next_ref, scalar_ref, h_ref, q_ref, s_ref = refs
    else:
        x_ref, r_ref, w_post_ref, w_next_ref, h_ref, q_ref, s_ref = refs
    dtype = h_ref.dtype
    x = x_ref[...].astype(jnp.float32)
    w_post = w_post_ref[...].astype(jnp.float32)
    n = _rms_norm(x, w_post, eps, dtype)
    h = (n + r_ref[...].astype(jnp.float32)).astype(dtype).astype(jnp.float32)
    if scalar_ref is not None:
        h = (h * scalar_ref[...].astype(jnp.float32)).astype(dtype).astype(jnp.float32)
    h_ref[...] = h.astype(dtype)
    m = _rms_norm(h, w_next_ref[...].astype(jnp.float32), eps, dtype)
    info = jnp.finfo(quant_dtype)
    scale = jnp.max(jnp.abs(m), axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(m / scale, float(info.min), float(info.max)).astype(quant_dtype)
    q_ref[...] = jax.lax.bitcast_convert_type(q, jnp.int8) if bytes_out else q
    s_ref[...] = scale


@functools.partial(
    jax.jit,
    static_argnames=("eps", "quant_dtype", "bytes_out", "max_block_rows", "interpret"),
)
def add_rms_norm_quantize(
    x: jax.Array,
    residual: jax.Array,
    w_post: jax.Array,
    w_next: jax.Array,
    scalar: jax.Array | None,
    *,
    eps: float,
    quant_dtype: jnp.dtype,
    bytes_out: bool = False,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens, hidden = x.shape
    if hidden % 128 != 0:
        raise ValueError(f"hidden {hidden} must be a multiple of 128")
    vmem_limit = vmem_limit_bytes()
    row_bytes = 2 * (3 * x.dtype.itemsize + 1) * hidden
    rows = fit_block_rows(tokens, row_bytes, max_block_rows, vmem_limit)
    row_block = pl.BlockSpec((rows, hidden), lambda i: (i, 0))
    weight = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    operands = [x, residual, w_post.reshape(1, hidden), w_next.reshape(1, hidden)]
    in_specs = [row_block, row_block, weight, weight]
    if scalar is not None:
        operands.append(scalar.reshape(1, 1))
        in_specs.append(pl.BlockSpec((1, 1), lambda i: (0, 0)))
    return pl.pallas_call(
        functools.partial(
            _kernel,
            eps=eps,
            has_scalar=scalar is not None,
            quant_dtype=quant_dtype,
            bytes_out=bytes_out,
        ),
        grid=(tokens // rows,),
        in_specs=in_specs,
        out_specs=[row_block, row_block, pl.BlockSpec((rows, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(x.shape, jnp.int8 if bytes_out else quant_dtype),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit
        ),
        interpret=interpret,
        name="add_rms_norm_quantize",
    )(*operands)


def _moe_tail_kernel(
    m_ref,
    e_ref,
    r_ref,
    w1_ref,
    w2_ref,
    wp_ref,
    sc_ref,
    wn_ref,
    h_ref,
    q_ref,
    s_ref,
    *,
    eps,
    quant_dtype,
    bytes_out,
):
    dtype = h_ref.dtype
    f32 = jnp.float32
    mixed = _rms_norm(m_ref[...].astype(f32), w1_ref[...].astype(f32), eps, dtype)
    mixed += _rms_norm(e_ref[...].astype(f32), w2_ref[...].astype(f32), eps, dtype)
    mixed = mixed.astype(dtype).astype(f32)
    n = _rms_norm(mixed, wp_ref[...].astype(f32), eps, dtype)
    h = (n + r_ref[...].astype(f32)).astype(dtype).astype(f32)
    h = (h * sc_ref[...].astype(f32)).astype(dtype).astype(f32)
    h_ref[...] = h.astype(dtype)
    m = _rms_norm(h, wn_ref[...].astype(f32), eps, dtype)
    info = jnp.finfo(quant_dtype)
    scale = jnp.max(jnp.abs(m), axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(m / scale, float(info.min), float(info.max)).astype(quant_dtype)
    q_ref[...] = jax.lax.bitcast_convert_type(q, jnp.int8) if bytes_out else q
    s_ref[...] = scale


@functools.partial(
    jax.jit,
    static_argnames=("eps", "quant_dtype", "bytes_out", "max_block_rows", "interpret"),
)
def moe_tail_rms_norm_quantize(
    mlp: jax.Array,
    moe: jax.Array,
    residual: jax.Array,
    w_mlp: jax.Array,
    w_moe: jax.Array,
    w_post: jax.Array,
    scalar: jax.Array,
    w_next: jax.Array,
    *,
    eps: float,
    quant_dtype: jnp.dtype,
    bytes_out: bool = False,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens, hidden = mlp.shape
    if hidden % 128 != 0:
        raise ValueError(f"hidden {hidden} must be a multiple of 128")
    vmem_limit = vmem_limit_bytes()
    row_bytes = 2 * (4 * mlp.dtype.itemsize + 1) * hidden
    rows = fit_block_rows(tokens, row_bytes, max_block_rows, vmem_limit)
    row_block = pl.BlockSpec((rows, hidden), lambda i: (i, 0))
    weight = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    scalar_spec = pl.BlockSpec((1, 1), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _moe_tail_kernel, eps=eps, quant_dtype=quant_dtype, bytes_out=bytes_out
        ),
        grid=(tokens // rows,),
        in_specs=[row_block] * 3 + [weight] * 3 + [scalar_spec, weight],
        out_specs=[row_block, row_block, pl.BlockSpec((rows, 1), lambda i: (i, 0))],
        out_shape=[
            jax.ShapeDtypeStruct(mlp.shape, mlp.dtype),
            jax.ShapeDtypeStruct(mlp.shape, jnp.int8 if bytes_out else quant_dtype),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit
        ),
        interpret=interpret,
        name="moe_tail_rms_norm_quantize",
    )(
        mlp,
        moe,
        residual,
        w_mlp.reshape(1, hidden),
        w_moe.reshape(1, hidden),
        w_post.reshape(1, hidden),
        scalar.reshape(1, 1),
        w_next.reshape(1, hidden),
    )


def _moe_head_kernel(
    x_ref,
    r_ref,
    wp_ref,
    wn_ref,
    wm_ref,
    wr_ref,
    rs_ref,
    h_ref,
    q_ref,
    s_ref,
    m_ref,
    logits_ref,
    *,
    eps,
    router_root,
    quant_dtype,
    bytes_out,
):
    dtype = h_ref.dtype
    f32 = jnp.float32
    n = _rms_norm(x_ref[...].astype(f32), wp_ref[...].astype(f32), eps, dtype)
    h = (n + r_ref[...].astype(f32)).astype(dtype).astype(f32)
    h_ref[...] = h.astype(dtype)
    hn = h * jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
    m = (hn * wn_ref[...].astype(f32)).astype(dtype).astype(f32)
    info = jnp.finfo(quant_dtype)
    scale = jnp.max(jnp.abs(m), axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(m / scale, float(info.min), float(info.max)).astype(quant_dtype)
    q_ref[...] = jax.lax.bitcast_convert_type(q, jnp.int8) if bytes_out else q
    s_ref[...] = scale
    m_ref[...] = (hn * wm_ref[...].astype(f32)).astype(dtype)
    xr = hn.astype(dtype) * jnp.asarray(router_root, dtype)
    xr = xr * rs_ref[...].astype(dtype)
    logits = jax.lax.dot_general(
        xr, wr_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=f32
    )
    logits_ref[...] = logits.astype(dtype).astype(f32)


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "router_root",
        "quant_dtype",
        "bytes_out",
        "max_block_rows",
        "interpret",
    ),
)
def moe_head_rms_norm_quantize(
    x: jax.Array,
    residual: jax.Array,
    w_post: jax.Array,
    w_next: jax.Array,
    w_moe: jax.Array,
    w_router: jax.Array,
    router_scale: jax.Array,
    *,
    eps: float,
    router_root: float,
    quant_dtype: jnp.dtype,
    bytes_out: bool = False,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    tokens, hidden = x.shape
    experts = w_router.shape[0]
    if hidden % 128 != 0 or experts % 128 != 0:
        raise ValueError(
            f"hidden {hidden} and experts {experts} must be multiples of 128"
        )
    if w_router.shape != (experts, hidden):
        raise ValueError(f"w_router {w_router.shape} is not [experts, {hidden}]")
    vmem_limit = vmem_limit_bytes()
    row_bytes = 2 * ((4 * x.dtype.itemsize + 1) * hidden + 4 * experts)
    weight_bytes = 2 * w_router.size * w_router.dtype.itemsize
    if weight_bytes + row_bytes * 8 > vmem_limit:
        raise ValueError(
            f"moe_head_rms_norm_quantize needs {weight_bytes + row_bytes * 8}B "
            f"exceeding vmem_limit {vmem_limit}B"
        )
    rows = fit_block_rows(tokens, row_bytes, max_block_rows, vmem_limit - weight_bytes)
    row_block = pl.BlockSpec((rows, hidden), lambda i: (i, 0))
    weight = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _moe_head_kernel,
            eps=eps,
            router_root=router_root,
            quant_dtype=quant_dtype,
            bytes_out=bytes_out,
        ),
        grid=(tokens // rows,),
        in_specs=[
            row_block,
            row_block,
            weight,
            weight,
            weight,
            pl.BlockSpec((experts, hidden), lambda i: (0, 0)),
            weight,
        ],
        out_specs=[
            row_block,
            row_block,
            pl.BlockSpec((rows, 1), lambda i: (i, 0)),
            row_block,
            pl.BlockSpec((rows, experts), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(x.shape, jnp.int8 if bytes_out else quant_dtype),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct((tokens, experts), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit
        ),
        interpret=interpret,
        name="moe_head_rms_norm_quantize",
    )(
        x,
        residual,
        w_post.reshape(1, hidden),
        w_next.reshape(1, hidden),
        w_moe.reshape(1, hidden),
        w_router,
        router_scale.reshape(1, hidden),
    )
