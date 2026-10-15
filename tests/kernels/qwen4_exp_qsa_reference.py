# SPDX-License-Identifier: Apache-2.0
"""Independent QSA cache quantization reference."""

import jax.numpy as jnp


def quantize_kv(value, scale):
    """CUDA reshape_and_cache: scalar FP32 division and saturated E4M3 RNE."""
    encoded = value.astype(jnp.float32) / jnp.float32(scale)
    return jnp.clip(encoded, -448, 448).astype(jnp.float8_e4m3fn)
