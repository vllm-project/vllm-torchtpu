# SPDX-License-Identifier: Apache-2.0
"""Independent dense JAX references for the serial preprocessing kernels."""

import jax
import jax.numpy as jnp


def projection(x: jax.Array, weight: jax.Array, scale: jax.Array) -> jax.Array:
    """Per-token FP8 matmul with the BF16 linear-output rounding boundary."""
    limit = jnp.asarray(jnp.finfo(weight.dtype).max, jnp.float32)
    absmax = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=1, keepdims=True)
    row_scale = jnp.where(absmax == 0, 1.0, absmax / limit)
    quantized = jnp.clip(x.astype(jnp.float32) / row_scale, -limit, limit)
    product = jax.lax.dot_general(
        quantized.astype(weight.dtype),
        weight,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return (product * row_scale * scale.reshape(1, -1)).astype(jnp.bfloat16)


def normalize(x: jax.Array, gamma: jax.Array, eps: float) -> jax.Array:
    """Weighted RMSNorm with FP32 accumulation and BF16 output."""
    values = x.astype(jnp.float32)
    mean_square = jnp.sum(jnp.square(values), axis=-1, keepdims=True) / x.shape[-1]
    result = values * jax.lax.rsqrt(mean_square + eps) * gamma.astype(jnp.float32)
    return result.astype(jnp.bfloat16)


def rotate(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Interleaved trailing RoPE; non-rotary channels pass through unchanged."""
    rotary_dim = 2 * cos.shape[-1]
    values = x.astype(jnp.float32)
    trailing = values[..., -rotary_dim:]
    partner = jnp.stack((-trailing[..., 1::2], trailing[..., ::2]), -1)
    cosine = jnp.repeat(cos, 2, axis=-1)
    sine = jnp.repeat(sin, 2, axis=-1)
    if x.ndim == 3:
        cosine, sine = cosine[:, None, :], sine[:, None, :]
    rotated = trailing * cosine + partner.reshape(trailing.shape) * sine
    return jnp.concatenate((values[..., :-rotary_dim], rotated), -1).astype(x.dtype)


def qa_kv_norm_swa(
    x: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    gamma_qr: jax.Array,
    gamma_kv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    projected = projection(x, weight, scale)
    qr = normalize(projected[:, : gamma_qr.size], gamma_qr, eps)
    kv = normalize(projected[:, gamma_qr.size :], gamma_kv, eps)
    return qr, rotate(kv, cos, sin)


def qb_rope(
    qr: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    head_dim: int = 512,
) -> jax.Array:
    projected = projection(qr, weight, scale)
    return rotate(projected.reshape(qr.shape[0], -1, head_dim), cos, sin)
