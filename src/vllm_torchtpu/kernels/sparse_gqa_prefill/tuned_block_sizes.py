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
"""Bounded prefill dispatch and metadata schedules."""

import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# Validated number of query rows per outer dispatch chunk.
PREFILL_DISPATCH_CHUNK_QUERIES = 8192

# Manual query-row ceilings and soft compact selected-KV target.
SC_CHUNK_SIZE = {jnp.dtype(jnp.bfloat16): 32, jnp.dtype(jnp.float8_e4m3fn): 64}
SC_COMPACT_TARGET_BYTES = 144 * 1024**2

# TPU7x: 16 words per reduction batch minimizes measured exact-union latency.
# At 8192 queries / 256K context, scratch stays below 65 MiB.
UNION_WORD_BATCH_SIZE = 16


def sc_chunk_size(
    cache: jax.Array | jax.ShapeDtypeStruct, *, budget: int, ratio: int
) -> int:
    """Choose a compact gather microchunk, or zero if one group exceeds the target."""
    if budget <= 0 or ratio <= 0 or 256 % ratio:
        raise ValueError("SC requires a positive budget and ratio dividing KV256")
    sparse = pltpu.get_tpu_info().sparse_core
    if sparse is None:
        raise NotImplementedError("Requires SparseCore")
    head_axes, packing, dim = cache.shape[2:]
    if min(head_axes, packing, dim) <= 0:
        raise ValueError("SC cache dimensions must be positive")
    capacity = (budget * ratio + ratio - 1 + 255) // 256 * 256
    granularity = math.lcm(
        sparse.num_lanes * sparse.num_cores * sparse.num_subcores * 4, head_axes
    )
    element_bytes = packing * dim * jnp.dtype(cache.dtype).itemsize
    affordable = SC_COMPACT_TARGET_BYTES // (granularity * element_bytes) * granularity
    rows = affordable // (capacity * head_axes)
    # Power-of-two chunks divide both Q256 and the outer dispatch schedule.
    return (
        min(SC_CHUNK_SIZE[jnp.dtype(cache.dtype)], 1 << (rows.bit_length() - 1))
        if rows >= 8
        else 0
    )
