# SPDX-License-Identifier: Apache-2.0
"""Packed Engram row decoding and the original gated residual, in JAX.

Adapted from vllm-project/vllm-torchtpu PR #1853, kernel.py at bbefef5720ee.
The 288-token decode bucket runs these semantics inside ``acc_kernel``; other
buckets compose them with the XLA quantized matmul.
"""

import jax
import jax.numpy as jnp


def unpack_words(words: jax.Array) -> jax.Array:
    """Little-endian int32 words [..., W] to uint8 [..., 4*W]."""
    shifts = jnp.arange(4, dtype=jnp.int32) * 8
    return (
        ((words[..., None] >> shifts) & 255)
        .astype(jnp.uint8)
        .reshape(*words.shape[:-1], words.shape[-1] * 4)
    )


def dequantize_rows(raw: jax.Array, scales: jax.Array) -> jax.Array:
    """Packed FP8 rows and E8M0 scales to BF16; scale 255 produces +0."""
    values = jax.lax.bitcast_convert_type(unpack_words(raw), jnp.float8_e4m3fn)
    scale_bytes = unpack_words(scales)
    block = values.shape[-1] // scale_bytes.shape[-1]
    values = values.reshape(*scale_bytes.shape, block).astype(jnp.float32)
    power = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.int32) << 23, jnp.float32
    )
    values = jnp.where(scale_bytes[..., None] == 255, 0.0, values * power[..., None])
    return values.reshape(*raw.shape[:-1], raw.shape[-1] * 4).astype(jnp.bfloat16)


def gated_residual(
    hidden: jax.Array,
    projection: jax.Array,
    query_weight: jax.Array,
    key_weight: jax.Array,
    token_mask: jax.Array,
    eps: float,
    clamp_value: float,
) -> jax.Array:
    """BF16 [T,S,D] hidden plus gated BF16 [T,(S+1)*D] keys and values."""
    streams, dim = hidden.shape[-2:]
    keys = projection[:, : streams * dim].reshape(-1, streams, dim).astype(jnp.float32)
    values = projection[:, streams * dim :].astype(jnp.float32)
    h = hidden.astype(jnp.float32)
    dot = jnp.sum(h * query_weight * key_weight * keys, axis=-1)
    dot *= jax.lax.rsqrt(jnp.mean(h * h, axis=-1) + eps)
    dot *= jax.lax.rsqrt(jnp.mean(keys * keys, axis=-1) + eps) * dim**-0.5
    magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), clamp_value))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -magnitude, magnitude))
    gate *= token_mask[:, None]
    return (h + gate[..., None] * values[:, None, :]).astype(hidden.dtype)
