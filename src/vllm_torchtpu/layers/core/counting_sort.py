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
"""Sort-free dispatch permutation for the MoE routed path."""

import jax
import jax.numpy as jnp


def key_onehot(keys: jax.Array, num_symbols: int) -> jax.Array:
    """``[n, num_symbols]`` int32 indicator of ``keys``."""
    return (
        keys[:, None] == jnp.arange(num_symbols, dtype=keys.dtype)[None, :]
    ).astype(jnp.int32)


def sort_positions(onehot: jax.Array) -> jax.Array:
    """Destination of every route under a **stable** sort of its key."""
    counts = onehot.sum(axis=0)
    bucket_start = jnp.cumsum(counts) - counts  # exclusive
    rank_in_bucket = jnp.cumsum(onehot, axis=0) - onehot  # exclusive
    return (
        (onehot * (bucket_start[None, :] + rank_in_bucket))
        .sum(axis=1)
        .astype(jnp.int32)
    )


def invert_permutation(positions: jax.Array) -> jax.Array:
    """``sorted_indices`` from ``positions``: which route landed where."""
    n = positions.shape[0]
    slots = jnp.arange(n, dtype=jnp.int32)
    hit = positions[:, None] == slots[None, :]
    return jnp.where(hit, slots[:, None], 0).sum(axis=0).astype(jnp.int32)


MAX_ROUTES = 128


def applicable(n_routes: int) -> bool:
    return 0 < n_routes <= MAX_ROUTES
