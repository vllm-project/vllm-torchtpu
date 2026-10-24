# SPDX-License-Identifier: Apache-2.0
"""Observable BF16 rounding boundaries used by QSA arithmetic."""

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs


def bf16(x: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32)
    if envs.VLLM_TPU_QSA_BF16_REDUCE_PRECISION:
        # reduce_precision is the same RNE, stays a rounding boundary XLA keeps,
        # and is a plain elementwise op that fuses into its neighbours.
        x = jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)
        return x.astype(jnp.bfloat16)
    # Integer RNE makes otherwise removable BF16 boundaries observable to XLA.
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    bits = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)
