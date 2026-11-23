# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Source provenance: inferact/k3-megakernel@30c9ba966b
#   src/vllm_torchtpu/megakernel/kimi_k3/kernels/helpers.py
"""Value-level Kimi-K3 model math helpers used by the prefill megakernel.

Pure ``jax.Array`` functions evaluated on VMEM loads or whole arrays with the
same operation order as ``decode_megakernel`` (which keeps its own copies:
``_gated_activation``, inline KDA normalization and decay, and E8M0 decode).
"""

import jax
import jax.numpy as jnp

_EXPONENT_MANTISSA_SHIFT = 23


def expert_activation(gate: jax.Array, value: jax.Array) -> jax.Array:
    """Compute the Kimi-K3 gated expert activation in the inputs' dtype."""
    return 4 * jnp.tanh(gate / 4) * jax.nn.sigmoid(gate) * 25 * jnp.tanh(value / 25)


def kda_normalize_query_key(
    query: jax.Array,
    key: jax.Array,
    channels: float,
) -> tuple[jax.Array, jax.Array]:
    """Return L2-normalized KDA query (scaled by ``channels**-0.5``) and key."""
    query = query * (
        jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
        * channels**-0.5
    )
    key = key * jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    return query, key


def kda_decay(
    decay_lower_bound: float | jax.Array,
    decay_rate: jax.Array,
    raw_gate: jax.Array,
    time_step_bias: jax.Array,
) -> jax.Array:
    """Compute per-channel KDA log decay from ``exp(A_log)`` and gate logits."""
    return decay_lower_bound * jax.nn.sigmoid(decay_rate * (raw_gate + time_step_bias))


def e8m0_scale(exponents: jax.Array) -> jax.Array:
    """Decode MXFP4 E8M0 exponent bytes into exact FP32 powers of two."""
    return jax.lax.bitcast_convert_type(
        exponents.astype(jnp.uint32) << _EXPONENT_MANTISSA_SHIFT,
        jnp.float32,
    )
