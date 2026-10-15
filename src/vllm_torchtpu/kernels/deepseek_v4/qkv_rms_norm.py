# SPDX-License-Identifier: Apache-2.0
"""Independent low-rank Q and KV RMSNorms in one TPU kernel."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _norm(x, weight, eps):
    xf = x.astype(jnp.float32)
    normalized = (
        xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    ).astype(weight.dtype)
    return (normalized * weight).astype(x.dtype)


def _kernel(x, qw, kw, q_out, kv_out, *, q_eps, kv_eps):
    q_width = qw.shape[-1]
    kv_width = kw.shape[-1]
    q_out[...] = _norm(x[:, :q_width], qw[...][None, :], q_eps)
    kv_out[...] = _norm(x[:, q_width : q_width + kv_width], kw[...][None, :], kv_eps)


def qkv_rms_norm(
    x: jax.Array,
    q_weight: jax.Array,
    kv_weight: jax.Array,
    *,
    q_eps: float,
    kv_eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Normalize the two packed projection outputs with separate statistics."""
    q_width, kv_width = q_weight.size, kv_weight.size
    assert x.shape == (1, q_width + kv_width)
    assert q_width % 128 == 0 and kv_width % 128 == 0

    def spec(shape):
        return pl.BlockSpec(shape, lambda: (0,) * len(shape))

    # Keep checkpoint vectors in their native layout across the call boundary.
    # Adding the broadcast dimension inside Pallas avoids an external relayout.
    qw, kw = q_weight.reshape(-1), kv_weight.reshape(-1)
    shapes = ((1, q_width), (1, kv_width))
    return pl.pallas_call(
        functools.partial(_kernel, q_eps=q_eps, kv_eps=kv_eps),
        grid=(),
        in_specs=[spec(a.shape) for a in (x, qw, kw)],
        out_specs=tuple(spec(s) for s in shapes),
        out_shape=tuple(jax.ShapeDtypeStruct(s, x.dtype) for s in shapes),
        name="qkv_rms_norm",
    )(x, qw, kw)
