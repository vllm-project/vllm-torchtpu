# SPDX-License-Identifier: Apache-2.0
# Provenance: copied from src/vllm_torchtpu/kernels/qwen4_exp/rounding.py
#   at 94aaaf9a11faa401374b80038ee39509f1c8bf2b.
#   Owned by models/qwen38_flash from this copy on.
"""Observable BF16 rounding boundaries used by QSA arithmetic."""

import jax
import jax.numpy as jnp


def bf16(x: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32)
    # Integer RNE makes otherwise removable BF16 boundaries observable to XLA.
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    bits = (bits + jnp.uint32(0x7FFF) + ((bits >> 16) & 1)) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)
