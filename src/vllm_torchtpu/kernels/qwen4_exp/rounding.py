# SPDX-License-Identifier: Apache-2.0
"""Observable BF16 rounding boundaries used by QSA arithmetic."""

import jax
import jax.numpy as jnp


def bf16(x: jax.Array) -> jax.Array:
    # Integer RNE makes otherwise removable BF16 boundaries observable to XLA.
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    bits = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)
