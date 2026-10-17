# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 model math shared by the prefill and decode megakernels.

Value-level functions (``jax.Array`` in, ``jax.Array`` out) that both
kernels evaluate, on VMEM loads or on whole arrays, with the same operations
in the same order. ``prefill_megakernel`` uses them; ``decode_megakernel``
has its own copies of some of them (``_gated_activation``, its inline KDA
normalization and decay, its E8M0 decode).
"""

import jax
import jax.numpy as jnp


def expert_activation(gate, value):
    """K3's expert activation, ``4 tanh(g / 4) sigmoid(g) * 25 tanh(v / 25)``
    (shared, routed and dense FFNs), in the inputs' dtype."""
    return 4 * jnp.tanh(gate / 4) * jax.nn.sigmoid(gate) * 25 * jnp.tanh(value / 25)


def kda_normalize_query_key(query, key, channels):
    """KDA's L2-normalized query (scaled by ``channels ** -0.5``) and key,
    over the last axis, in FP32."""
    query = query * (
        jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
        * channels**-0.5
    )
    key = key * jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    return query, key


def kda_decay(decay_lower_bound, decay_rate, raw_gate, time_step_bias):
    """KDA's per-channel log decay, ``lower_bound * sigmoid(rate * (gate +
    bias))``; ``decay_rate`` is ``exp(A_log)``, broadcast by the caller."""
    return decay_lower_bound * jax.nn.sigmoid(decay_rate * (raw_gate + time_step_bias))


def e8m0_scale(exponents):
    """MXFP4 E8M0 exponent bytes as the exact FP32 powers of two they encode."""
    return jax.lax.bitcast_convert_type(exponents.astype(jnp.uint32) << 23, jnp.float32)
