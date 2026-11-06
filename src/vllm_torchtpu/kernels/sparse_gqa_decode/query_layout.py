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
"""Convert queries into the existing RPA input layout."""

import jax
import jax.numpy as jnp


def query_offsets(batch: int, cached_prefix: jax.Array | None = None) -> jax.Array:
    """Size unit-stride query offsets independently of request segmentation.

    The immutable zero-origin prefix is a cache, so its capacity may differ
    from the current launch's query count.
    """
    if cached_prefix is not None:
        if cached_prefix.ndim != 1 or cached_prefix.dtype != jnp.int32:
            raise ValueError("Cached query offsets must be an int32 vector")
        if len(cached_prefix) >= batch + 1:
            return cached_prefix[: batch + 1]
    return jnp.arange(batch + 1, dtype=jnp.int32)


def planning_query(query: jax.Array, prepared: bool) -> jax.Array:
    """Plan BF16-valued queries held in an FP32 reader layout.

    Prepared producers guarantee exactly representable BF16 values. Arbitrary
    FP32 input retains its dtype and is rejected by the existing planner.
    """
    if prepared:
        assert query.ndim == 3 and query.dtype == jnp.float32
        return query.astype(jnp.bfloat16)
    return query


def prepare_query(q, lengths, kv_heads):
    batch, heads, dim = q.shape
    group = heads // kv_heads
    # The compact schedule never loads zero-length queries. execute_decode
    # still masks returned inactive rows, including NaNs in the unused input.
    return q.reshape(batch, kv_heads, group, 1, dim).swapaxes(0, 1).astype(jnp.float32)
