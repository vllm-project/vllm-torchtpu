# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native V4.1 FP8 records with one FP32 NoPE scale per token."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.cache_layout import pack_cache_values

NOPE_DIM = 448
SCALE_BYTES = 4


def encode_fp8_kv(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Pack 448 FP8 values, one FP32 scale and padding into a 512-byte row.

    The scale occupies logical bytes [448:452] in little-endian order; the
    remaining 60 bytes are zero. Native feature packing splits this payload
    into [4, 128] before the word bitcast. RoPE uses a separate padded row
    in the same combined record, with its original BF16 bits preserved.
    """
    nope = latent[:, :NOPE_DIM].astype(jnp.float32)
    amax = jnp.max(jnp.abs(nope), axis=-1, keepdims=True)
    scale = jnp.maximum(amax, 1e-4) / float(jnp.finfo(jnp.float8_e4m3fn).max)
    values = (nope / scale).astype(jnp.float8_e4m3fn)
    scale_bits = jax.lax.bitcast_convert_type(scale, jnp.uint32)
    scale_bytes = jnp.concatenate(
        [(scale_bits >> (8 * byte)).astype(jnp.uint8) for byte in range(SCALE_BYTES)],
        axis=-1,
    )
    payload = jnp.concatenate(
        (jax.lax.bitcast_convert_type(values, jnp.uint8), scale_bytes), axis=-1
    )
    payload = jnp.pad(payload, ((0, 0), (0, 512 - NOPE_DIM - SCALE_BYTES)))
    return (
        pack_cache_values(payload),
        pack_cache_values(latent[:, NOPE_DIM:].astype(jnp.bfloat16)),
    )


def dequant_fp8_nope(payload: jax.Array) -> jax.Array:
    """Unpack one FP32 scale per gathered row and dequantize NoPE to BF16.

    Only the four scale bytes are assembled across lanes. The 448 FP8
    features keep their native order, and the padding is never interpreted.
    This function also runs inside SparseMLA's Pallas kernel.
    """
    if jax.default_backend() == "tpu":
        scale_bytes = jnp.stack(
            [
                payload[:, NOPE_DIM + byte : NOPE_DIM + byte + 1]
                for byte in range(SCALE_BYTES)
            ],
            axis=-2,
        )
        scale = pltpu.bitcast(scale_bytes, jnp.float32).reshape(payload.shape[0], 1)
        values = pltpu.bitcast(payload[:, :NOPE_DIM], jnp.float8_e4m3fn)
    else:
        scale = jax.lax.bitcast_convert_type(
            payload[:, NOPE_DIM : NOPE_DIM + SCALE_BYTES], jnp.float32
        )[:, None]
        values = jax.lax.bitcast_convert_type(payload[:, :NOPE_DIM], jnp.float8_e4m3fn)
    values = values.astype(jnp.float32)
    return (values * scale).astype(jnp.bfloat16)
