# SPDX-License-Identifier: Apache-2.0
"""Dense JAX reference for the V4.1 compressor path up to CSA2 encoding.

Mirrors epic/dsv41 ``DeepseekCompressor.project`` + ``save_compress_norm`` +
``_rotate(latent, group_positions)`` + ``encode_csa2_kv``. Ring rows are
gathered by the caller (``ring_prev``), exactly as ``save_compress_norm`` reads
``stored`` before publishing.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.attention_preprocessing.reference import rotate


def project(x: jax.Array, weight: jax.Array) -> jax.Array:
    """BF16 inputs, FP32 accumulation and output (the lowered ``.float()`` mm)."""
    return jax.lax.dot_general(
        x, weight, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )


def pool_norm(kv_score, has_prev, ring_prev, valid, gamma, eps, ratio):
    """Return BF16 latent [T,512]; rows outside ``valid`` are zero."""
    dim = gamma.size
    values = kv_score[:, :dim]
    if ratio == 2:
        in_chunk = jnp.concatenate((kv_score[:1], kv_score[:-1]), 0)
        previous = jnp.where(has_prev[:, None], in_chunk, ring_prev)
        scores = jnp.stack((previous[:, dim:], kv_score[:, dim:]), 1)
        pair = jnp.stack((previous[:, :dim], values), 1)
        values = (pair * jax.nn.softmax(scores, axis=1)).sum(1)
    normalized = values * jax.lax.rsqrt(
        jnp.mean(values * values, -1, keepdims=True) + eps
    )
    result = (normalized * gamma.astype(jnp.float32)).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, jnp.zeros_like(result))


def encode_csa2(latent_rope: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``encode_csa2_kv``: byte g of word lane holds feature g*128+lane."""
    values = latent_rope.astype(jnp.float32)
    scales = jnp.maximum(jnp.abs(values).max(axis=1), 1e-4) / 448
    quantized = (values / scales[:, None]).astype(jnp.float8_e4m3fn)
    planes = jax.lax.bitcast_convert_type(quantized, jnp.uint8).reshape(-1, 4, 128)
    planes = planes.astype(jnp.uint32)
    words = (
        planes[:, 0] | (planes[:, 1] << 8) | (planes[:, 2] << 16) | (planes[:, 3] << 24)
    )
    return jax.lax.bitcast_convert_type(words, jnp.int32), scales


def compress(
    x, weight, gamma, has_prev, ring_prev, valid, cos, sin, *, eps=1e-6, ratio=2
):
    """Return (kv_score FP32, latent BF16, words INT32 [T,128], scales FP32 [T]).

    ``cos``/``sin`` are gathered at each row's group start position.
    """
    kv_score = project(x, weight)
    latent = pool_norm(kv_score, has_prev, ring_prev, valid, gamma, eps, ratio)
    words, scales = encode_csa2(rotate(latent, cos, sin))
    return kv_score, latent, words, scales
