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
"""Residual add around an RMSNorm, then the next RMSNorm and its per-token FP8
quantization, in one pass over whole rows.

Between two projections a Gemma4 layer runs

    h = rms_norm(x, w_post) + residual    [* layer_scalar]
    q, s = quantize_tensor(rms_norm(h, w_next), fp8)

which XLA runs as four passes over `[tokens, hidden]`: the variance of `x`,
the normalization and the add (writing `h`), the next norm with its row
maximum, and the next norm again with the cast. This kernel holds a block of
whole rows in VMEM and writes `h`, `q` and `s` once.

Arithmetic follows the TPU `rms_norm` provider and `quantize_tensor`: the
norms run in f32 and round to the activation dtype, the add and the scalar
product round to it, and the f32 scale is the row maximum / dtype max (1 where
it is 0). The sums of squares are added in another order than XLA's, so values
can move by one rounding step.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import fit_block_rows

# Rows per block: on tpu7x a 4096-row Gemma4-31B TP2 shard takes 0.142 / 0.121 /
# 0.115 ms with 64 / 128 / 256 rows (device micro, same outputs).
_BLOCK_ROWS = 256
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _rms_norm(x: jax.Array, w: jax.Array, eps: float, dtype: jnp.dtype) -> jax.Array:
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(var + eps) * w).astype(dtype).astype(jnp.float32)


def _kernel(*refs, eps, has_scalar, quant_dtype, bytes_out):
    if has_scalar:
        x_ref, r_ref, w_post_ref, w_next_ref, scalar_ref, h_ref, q_ref, s_ref = refs
    else:
        x_ref, r_ref, w_post_ref, w_next_ref, h_ref, q_ref, s_ref = refs
    dtype = h_ref.dtype
    x = x_ref[...].astype(jnp.float32)
    w_post = w_post_ref[...].astype(jnp.float32)
    n = _rms_norm(x, w_post, eps, dtype)
    h = (n + r_ref[...].astype(jnp.float32)).astype(dtype).astype(jnp.float32)
    if has_scalar:
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
    """`h = rms_norm(x, w_post) + residual` (times `scalar` when given), and
    `quantize_tensor(rms_norm(h, w_next), quant_dtype)`.

    Returns `(h, q, scale)`: `h` in `x`'s dtype, `q` `[tokens, hidden]` in
    `quant_dtype` (with `bytes_out`, its bytes as int8, which a collective moves
    without a dtype view), `scale` f32 `[tokens, 1]` (module docstring for
    rounding).
    """
    tokens, hidden = x.shape
    # x, residual and h in the activation dtype, q in 1 byte, double-buffered.
    row_bytes = 2 * (3 * x.dtype.itemsize + 1) * hidden
    rows = fit_block_rows(tokens, row_bytes, max_block_rows, _VMEM_LIMIT_BYTES)
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
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="add_rms_norm_quantize",
    )(*operands)


def _moe_tail_kernel(*refs, eps, quantize, quant_dtype, bytes_out):
    m_ref, e_ref, r_ref, w1_ref, w2_ref, wp_ref, sc_ref, *rest = refs
    if quantize:
        wn_ref, h_ref, q_ref, s_ref = rest
    else:
        (h_ref,) = rest
        wn_ref = q_ref = s_ref = None
    dtype = h_ref.dtype
    f32 = jnp.float32
    mixed = _rms_norm(m_ref[...].astype(f32), w1_ref[...].astype(f32), eps, dtype)
    mixed += _rms_norm(e_ref[...].astype(f32), w2_ref[...].astype(f32), eps, dtype)
    mixed = mixed.astype(dtype).astype(f32)
    n = _rms_norm(mixed, wp_ref[...].astype(f32), eps, dtype)
    h = (n + r_ref[...].astype(f32)).astype(dtype).astype(f32)
    h = (h * sc_ref[...].astype(f32)).astype(dtype).astype(f32)
    h_ref[...] = h.astype(dtype)
    if wn_ref is None or q_ref is None or s_ref is None:
        return
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
    w_next: jax.Array | None,
    *,
    eps: float,
    quant_dtype: jnp.dtype | None,
    bytes_out: bool = False,
    max_block_rows: int = _BLOCK_ROWS,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """A Gemma4 MoE layer's tail and the next layer's input quantization:

        h = (rms_norm(rms_norm(mlp, w_mlp) + rms_norm(moe, w_moe), w_post)
             + residual) * scalar
        q, scale = quantize_tensor(rms_norm(h, w_next), quant_dtype)

    Returns `(h, q, scale)`, or `(h,)` without `w_next` (the last layer).
    Rounding as `add_rms_norm_quantize`: each norm, add and product rounds to
    the activation dtype, as the unfused ops do.
    """
    tokens, hidden = mlp.shape
    quantize = w_next is not None
    # mlp, moe, residual and h in the activation dtype, q in 1 byte, double-buffered.
    row_bytes = 2 * (4 * mlp.dtype.itemsize + int(quantize)) * hidden
    rows = fit_block_rows(tokens, row_bytes, max_block_rows, _VMEM_LIMIT_BYTES)
    row_block = pl.BlockSpec((rows, hidden), lambda i: (i, 0))
    weight = pl.BlockSpec((1, hidden), lambda i: (0, 0))
    operands = [
        mlp,
        moe,
        residual,
        w_mlp.reshape(1, hidden),
        w_moe.reshape(1, hidden),
        w_post.reshape(1, hidden),
        scalar.reshape(1, 1),
    ]
    in_specs = [row_block] * 3 + [weight] * 3 + [pl.BlockSpec((1, 1), lambda i: (0, 0))]
    out_specs = [row_block]
    out_shape = [jax.ShapeDtypeStruct(mlp.shape, mlp.dtype)]
    if quantize:
        operands.append(w_next.reshape(1, hidden))
        in_specs.append(weight)
        out_specs += [row_block, pl.BlockSpec((rows, 1), lambda i: (i, 0))]
        out_shape += [
            jax.ShapeDtypeStruct(mlp.shape, jnp.int8 if bytes_out else quant_dtype),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
        ]
    return tuple(
        pl.pallas_call(
            functools.partial(
                _moe_tail_kernel,
                eps=eps,
                quantize=quantize,
                quant_dtype=quant_dtype,
                bytes_out=bytes_out,
            ),
            grid=(tokens // rows,),
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shape,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=_VMEM_LIMIT_BYTES,
            ),
            interpret=interpret,
            name="moe_tail_rms_norm_quantize",
        )(*operands)
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
    # The three norms of h share its variance (one epsilon, checked at build).
    hn = h * jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + eps)
    m = (hn * wn_ref[...].astype(f32)).astype(dtype).astype(f32)
    info = jnp.finfo(quant_dtype)
    scale = jnp.max(jnp.abs(m), axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(m / scale, float(info.min), float(info.max)).astype(quant_dtype)
    q_ref[...] = jax.lax.bitcast_convert_type(q, jnp.int8) if bytes_out else q
    s_ref[...] = scale
    m_ref[...] = (hn * wm_ref[...].astype(f32)).astype(dtype)
    # The router's weightless norm, root scale and per-dimension scale, each
    # rounded to the activation dtype as separate ops would (a product of two
    # activation-dtype values rounds the same in that dtype as through f32).
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
    """A Gemma4 MoE layer's post-attention tail and everything read from it:

        h = rms_norm(x, w_post) + residual
        q, scale = quantize_tensor(rms_norm(h, w_next), quant_dtype)
        m = rms_norm(h, w_moe)
        logits = (rms_norm(h) * router_root * router_scale) @ w_router.T

    `rms_norm(h)` has no weight; `router_root` is the router's constant input
    scale already rounded to the activation dtype, `w_router` its [experts,
    hidden] projection. Returns `(h, q, scale, m,
    logits)`: `h` and `m` in `x`'s dtype, `q` as `add_rms_norm_quantize`,
    `logits` f32 [tokens, experts] rounded through the activation dtype, as a
    matmul in that dtype returns them. Every norm, product and the dot round
    as the separate ops do; sums of squares and the dot accumulate in another
    order.
    """
    tokens, hidden = x.shape
    experts = w_router.shape[0]
    if w_router.shape != (experts, hidden):
        raise ValueError(f"w_router {w_router.shape} is not [experts, {hidden}]")
    # x, residual, h and m in the activation dtype, q in 1 byte and the f32
    # logits, double-buffered, beside the double-buffered router weight.
    row_bytes = 2 * ((4 * x.dtype.itemsize + 1) * hidden + 4 * experts)
    weight_bytes = 2 * w_router.size * w_router.dtype.itemsize
    rows = fit_block_rows(
        tokens, row_bytes, max_block_rows, _VMEM_LIMIT_BYTES - weight_bytes
    )
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
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
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
