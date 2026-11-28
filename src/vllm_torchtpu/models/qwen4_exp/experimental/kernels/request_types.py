# Copyright 2026 Google LLC
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

# Provenance: copied from src/vllm_torchtpu/kernels/qwen4_exp/request_types.py
#   at d33b13e9941efbc17d0296900a0bc3eb81c9a88c.
#   Owned by models/qwen4_exp from this copy on.
"""QSA request segmentation using the runner's decode-first ragged ordering."""

import jax
import jax.numpy as jnp


def decode_prefix(starts: jax.Array) -> jax.Array:
    """Count the leading one-token requests; leave later/empty rows in place."""
    return jnp.sum(jnp.cumprod((jnp.diff(starts) == 1).astype(jnp.int32)))


# Longest request routed to the row-wise RPA kernel by
# ``short_request_prefix_tokens``. Covers decode rows and speculative verify
# rows (1 + num_speculative_tokens). In the block-sparse prefill kernel each
# such request would occupy a whole query block (and force the step to the
# padded, block-aligned layout), while RPA costs one row per token.
SHORT_REQUEST_MAX_TOKENS = 16


def short_request_prefix_tokens(
    starts: jax.Array,
    capacity: int,
    max_request_tokens: int = SHORT_REQUEST_MAX_TOKENS,
) -> jax.Array:
    """Token count of the leading short requests that fit ``capacity`` rows.

    The runner orders decode (and speculative verify) requests first. Count
    the leading requests with 1..max_request_tokens tokens whose end stays
    within ``capacity`` rows and return their total token count; zero when
    the first request is long or empty.
    """
    lengths = jnp.diff(starts)
    short = (lengths >= 1) & (lengths <= max_request_tokens) & (starts[1:] <= capacity)
    count = jnp.sum(jnp.cumprod(short.astype(jnp.int32)))
    return starts[count] - starts[0]
