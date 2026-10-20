# SPDX-License-Identifier: Apache-2.0
"""Single-token mHC with RMSNorm and FP8 activation preparation."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel, pre_kernel
from vllm_torchtpu.kernels.deepseek_v4.mhc.sinkhorn_kernel import single_token_post_comb
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import DEFAULT_VMEM_LIMIT_BYTES
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

GateConstants = tuple[float, float, float, float, int]


def supports_mhc_norm_quant(num_tokens: int, hc_mult: int, hidden_size: int) -> bool:
    """The resident kernel uses one row, four streams and 128-wide tiles."""
    return num_tokens == 1 and hc_mult == 4 and hidden_size % 128 == 0


def _norm(x, weight, eps):
    xf = x.astype(jnp.float32)
    normalized = (
        xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    ).astype(weight.dtype)
    return (normalized * weight).astype(x.dtype)


def _finish(layer, weight, y_ref, quant_refs, eps):
    y = _norm(layer, weight[...], eps)
    y_ref[...] = y
    if quant_refs:
        q, scale = quantize_tensor(y, quant_refs[0].dtype)
        quant_refs[0][...] = q
        quant_refs[1][...] = scale


def _pre_kernel(res, fn, sc, hb, weight, post, comb, y, quant, *, constants, norm_eps):
    rms_eps, hc_eps, sink_eps, alpha, iters = constants
    pre_kernel._mixes_collapse_kernel(
        res,
        (fn,),
        sc,
        hb,
        post,
        comb,
        None,
        hc_mult=4,
        hidden_size=weight.shape[-1],
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        fused_gates=True,
        hc_sinkhorn_eps=sink_eps,
        hc_post_mult_value=alpha,
        sinkhorn_repeat=iters,
        finish_layer=lambda value: _finish(value, weight, y, quant, norm_eps),
    )


def _fused_kernel(
    x,
    res,
    post_in,
    comb_in,
    fn,
    sc,
    hb,
    weight,
    newres,
    post,
    comb,
    y,
    quant,
    *,
    constants,
    norm_eps,
):
    rms_eps, hc_eps, sink_eps, alpha, iters = constants
    fused_post_pre_kernel._fused_kernel(
        x,
        res,
        post_in,
        comb_in,
        (fn,),
        sc,
        hb,
        newres,
        post,
        comb,
        None,
        hc_mult=4,
        hidden_size=weight.shape[-1],
        gemm_precision="highest",
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        fused_gates=True,
        hc_sinkhorn_eps=sink_eps,
        hc_post_mult_value=alpha,
        sinkhorn_repeat=iters,
        finish_layer=lambda value: _finish(value, weight, y, quant, norm_eps),
    )


def _broadcast_pre_kernel(
    x_ref, fn, sc, hb, weight, residual, post, comb, y, quant, *, constants, norm_eps
):
    rms_eps, hc_eps, sink_eps, alpha, iters = constants
    x = x_ref[...].astype(jnp.float32)
    acc = jnp.sum(fn[...] * x, axis=1, keepdims=True).T
    scaled = acc * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + rms_eps)
    pre = jax.nn.sigmoid(scaled[:, :4] * sc[0, 0] + hb[:, :4]) + hc_eps
    layer = pre[:, :1] * x
    for i in range(1, 4):
        layer = layer + pre[:, i : i + 1] * x
    _finish(layer.astype(jnp.bfloat16), weight, y, quant, norm_eps)
    residual[...] = jnp.broadcast_to(x_ref[...], residual.shape)
    post_value, comb_value = single_token_post_comb(
        scaled, sc[...], hb[...], sink_eps, alpha, iters
    )
    post[...] = post_value.T
    comb[...] = comb_value


def _call(inputs, kernel, h, constants, norm_eps, quant_dtype):
    def spec(shape):
        return pl.BlockSpec(shape, lambda: (0,) * len(shape))

    # Keep the stream axis in the physical layout across adjacent mHC calls.
    prefix_shapes = [(4, h)] if kernel in (_fused_kernel, _broadcast_pre_kernel) else []
    shapes = prefix_shapes + [(4, 1), (4, 4), (1, h)]
    dtypes = ([jnp.bfloat16] if prefix_shapes else []) + [
        jnp.float32,
        jnp.float32,
        jnp.bfloat16,
    ]
    q_shapes = [(1, h), (1, 1)] if quant_dtype is not None else []
    q_dtypes = [quant_dtype, jnp.float32] if q_shapes else []
    # Reserving the batched kernel's 100 MiB can force adjacent XLA values
    # into HBM. Leave headroom for FP32 temporaries and DMA staging without
    # consuming that entire budget for a single resident token.
    resident_bytes = sum(a.size * a.dtype.itemsize for a in inputs)
    vmem_limit_bytes = min(
        DEFAULT_VMEM_LIMIT_BYTES, max(16 * 1024 * 1024, 4 * resident_bytes)
    )
    outputs = pl.pallas_call(
        functools.partial(kernel, constants=constants, norm_eps=norm_eps),
        grid=(),
        in_specs=[spec(a.shape) for a in inputs],
        out_specs=[
            *[spec(shape) for shape in shapes],
            tuple(spec(shape) for shape in q_shapes),
        ],
        out_shape=[
            *[
                jax.ShapeDtypeStruct(shape, dtype)
                for shape, dtype in zip(shapes, dtypes)
            ],
            tuple(
                jax.ShapeDtypeStruct(shape, dtype)
                for shape, dtype in zip(q_shapes, q_dtypes)
            ),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name="mhc_"
        + (
            "broadcast_pre"
            if kernel is _broadcast_pre_kernel
            else "fused"
            if prefix_shapes
            else "pre"
        )
        + "_norm"
        + ("_quant" if q_shapes else ""),
    )(*inputs)
    return outputs


def mhc_broadcast_pre_norm_quant(
    x: jax.Array,
    fn: jax.Array,
    sc: jax.Array,
    hb: jax.Array,
    weight: jax.Array,
    *,
    constants: GateConstants,
    norm_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array]]:
    """Open the four-stream chain from an embedding and a summed projection."""
    assert x.ndim == 2 and supports_mhc_norm_quant(x.shape[0], 4, x.shape[1])
    assert x.dtype == jnp.bfloat16 and fn.dtype == jnp.float32
    h = x.shape[1]
    assert fn.shape == (24, h)
    residual, post, comb, y, quant = _call(
        [x, fn, sc.reshape(1, 3), hb.reshape(1, 24), weight.reshape(1, h)],
        _broadcast_pre_kernel,
        h,
        constants,
        norm_eps,
        jnp.float8_e4m3fn,
    )
    return (
        residual.reshape(1, 4, h),
        post.reshape(1, 4, 1),
        comb.reshape(1, 4, 4),
        y,
        quant,
    )


def mhc_pre_norm_quant(
    residual: jax.Array,
    fn: jax.Array,
    sc: jax.Array,
    hb: jax.Array,
    weight: jax.Array,
    *,
    constants: GateConstants,
    norm_eps: float,
    quant_dtype: jnp.dtype = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, jax.Array]]:
    """Return post/comb gates, normalized BF16 input, and FP8 input/scale."""
    assert supports_mhc_norm_quant(*residual.shape)
    assert residual.dtype == jnp.bfloat16 and fn.dtype == jnp.float32
    assert quant_dtype == jnp.float8_e4m3fn
    assert fn.shape == (24, 4 * residual.shape[-1])
    h = residual.shape[-1]
    post, comb, y, q = _call(
        [
            residual.reshape(4, h),
            fn,
            sc.reshape(1, 3),
            hb.reshape(1, 24),
            weight.reshape(1, h),
        ],
        _pre_kernel,
        h,
        constants,
        norm_eps,
        quant_dtype,
    )
    return post.reshape(1, 4, 1), comb.reshape(1, 4, 4), y, q


def mhc_fused_post_pre_norm_quant(
    x: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    fn: jax.Array,
    sc: jax.Array,
    hb: jax.Array,
    weight: jax.Array,
    *,
    constants: GateConstants,
    norm_eps: float,
    quant_dtype: jnp.dtype | None = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """Recombine and normalize; quant_dtype=None omits unused FP8 outputs."""
    assert supports_mhc_norm_quant(*residual.shape)
    assert residual.dtype == jnp.bfloat16 and fn.dtype == jnp.float32
    assert quant_dtype in (None, jnp.float8_e4m3fn)
    assert fn.shape == (24, 4 * residual.shape[-1])
    h = residual.shape[-1]
    res, post, comb, y, q = _call(
        [
            x,
            residual.reshape(4, h),
            post.reshape(4, 1),
            comb.reshape(4, 4),
            fn,
            sc.reshape(1, 3),
            hb.reshape(1, 24),
            weight.reshape(1, h),
        ],
        _fused_kernel,
        h,
        constants,
        norm_eps,
        quant_dtype,
    )
    return res.reshape(1, 4, h), post.reshape(1, 4, 1), comb.reshape(1, 4, 4), y, q
