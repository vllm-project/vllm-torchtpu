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
"""Shape-derived TC/SC traffic and matrix-product work estimates."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import sc_chunk_size

# Adaptive kernels dispatch Q256 and SC groups of eight with KV256.
TC_QUERY_BLOCK = 256
_KV_TILE = 256
_SC_QUERY_GROUP = 8
_MASK_WORD_ALIGNMENT = 128


class PrefillCostModel(NamedTuple):
    tc_fixed_ms: float
    tc_query_ms: float
    tc_mask_word_ms: float
    tc_tile_ms: float
    sc_fixed_ms: float
    sc_query_ms: float
    sc_attention_tile_ms: float
    budget: int
    kv_heads: int
    ratio: int
    kv_tile: int
    sc_chunk: int


def prefill_cost_model(
    query: jax.Array | jax.ShapeDtypeStruct,
    blocks: jax.Array | jax.ShapeDtypeStruct,
    table: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    *,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
) -> PrefillCostModel:
    """Convert scheduled traffic and dot products to milliseconds at peak rates."""
    if (
        query.ndim != 3
        or kv_heads <= 0
        or ratio <= 0
        or kv_tile <= 0
        or query.shape[1] % kv_heads
        or kv_tile % (ratio * 32)
        or 256 % ratio
    ):
        raise ValueError("Invalid GQA head, compression or KV-tile geometry")
    info = pltpu.get_tpu_info()
    sparse = info.sparse_core
    if sparse is None:
        raise NotImplementedError("Adaptive prefill requires SparseCore")
    bandwidth = info.mem_bw_bytes_per_second * info.num_cores / 1000
    dot_rate = info.bf16_ops_per_second * info.num_cores / 1000
    heads, dim = query.shape[1:]
    head_axes, packing = cache.shape[2:4]
    budget, columns = blocks.shape[1], table.shape[1]
    index_bytes = jnp.dtype(jnp.int32).itemsize
    query_bytes = jnp.dtype(query.dtype).itemsize
    cache_row_bytes = head_axes * packing * dim * jnp.dtype(cache.dtype).itemsize
    padded_budget = (budget + 127) // 128 * 128
    padded_table = (columns + 127) // 128 * 128
    capacity = (budget * ratio + ratio - 1 + 255) // 256 * 256
    chunk = sc_chunk_size(cache, budget=budget, ratio=ratio)
    if not chunk:
        raise ValueError("One SC query group exceeds the compact workspace target")
    granularity = math.lcm(
        sparse.num_lanes * sparse.num_cores * sparse.num_subcores * 4, head_axes
    )
    indices = chunk * capacity * head_axes
    gather_rows = (indices + granularity - 1) // granularity * granularity
    packed_heads = kv_heads * ((heads // kv_heads + 7) // 8 * 8)
    packed_elements = packed_heads * dim
    index_input_bytes = (
        (capacity // _KV_TILE) * (1 + padded_budget + padded_table) * index_bytes
    )
    index_output_bytes = capacity * head_axes * index_bytes
    gather_bytes = (
        gather_rows
        * (index_bytes + 2 * packing * dim * jnp.dtype(cache.dtype).itemsize)
        / chunk
    )
    query_gate_bytes = 2 * heads * dim * query_bytes
    packed_query_gate_bytes = 2 * packed_elements * query_bytes
    packed_output_bytes = packed_elements * jnp.dtype(jnp.bfloat16).itemsize
    sc_query_bytes = (
        index_input_bytes
        + index_output_bytes
        + gather_bytes
        + query_gate_bytes
        + packed_query_gate_bytes
        + packed_query_gate_bytes
        + packed_output_bytes
        + index_bytes
    )
    tc_query_bytes = (
        query_gate_bytes
        + heads * dim * jnp.dtype(jnp.bfloat16).itemsize
        + (budget + columns + 1) * index_bytes
    )
    tc_table_bytes = columns * index_bytes
    tc_count_bytes = index_bytes
    tc_last_page_bytes = index_bytes
    tc_tile_bytes = kv_tile * cache_row_bytes
    sc_tile_bytes = _SC_QUERY_GROUP * _KV_TILE * cache_row_bytes
    tc_dot_flops = 4 * TC_QUERY_BLOCK * heads * dim * kv_tile
    sc_dot_flops = (
        4 * _SC_QUERY_GROUP * packed_heads * dim * (_SC_QUERY_GROUP * _KV_TILE)
    )
    return PrefillCostModel(
        tc_fixed_ms=(tc_table_bytes + tc_count_bytes + tc_last_page_bytes) / bandwidth,
        tc_query_ms=tc_query_bytes / bandwidth,
        tc_mask_word_ms=2 * index_bytes / bandwidth,
        tc_tile_ms=tc_tile_bytes / bandwidth + tc_dot_flops / dot_rate,
        sc_fixed_ms=0.0,
        sc_query_ms=sc_query_bytes / bandwidth,
        sc_attention_tile_ms=sc_tile_bytes / bandwidth + sc_dot_flops / dot_rate,
        budget=budget,
        kv_heads=kv_heads,
        ratio=ratio,
        kv_tile=kv_tile,
        sc_chunk=chunk,
    )


def estimate_static_bounds_ms(
    model: PrefillCostModel,
    *,
    queries: int,
    num_tiles: int,
    sc_queries: int | None = None,
) -> tuple[float, float]:
    """Return the maximum TC estimate and minimum scheduled SC estimate."""
    words = (
        ((model.kv_tile // (model.ratio * 32)) * num_tiles + (_MASK_WORD_ALIGNMENT - 1))
        // _MASK_WORD_ALIGNMENT
    ) * _MASK_WORD_ALIGNMENT
    upper_tc = (
        model.tc_fixed_ms
        + model.tc_query_ms * queries
        + model.tc_mask_word_ms * queries * words
        + model.tc_tile_ms * (queries // TC_QUERY_BLOCK) * num_tiles
    )
    scheduled_sc = queries if sc_queries is None else sc_queries
    lower_sc = (
        model.sc_fixed_ms
        + model.sc_query_ms * scheduled_sc
        + model.sc_attention_tile_ms * (scheduled_sc // _SC_QUERY_GROUP)
    )
    return upper_tc, lower_sc


def tc_cost(
    model: PrefillCostModel,
    *,
    queries: int | jax.Array,
    num_tiles: int,
    tile_visits: jax.Array,
    launches: int | jax.Array = 1,
) -> jax.Array:
    """Estimate TC time from scheduled queries, mask words, and tile visits."""
    words = (
        ((model.kv_tile // (model.ratio * 32)) * num_tiles + (_MASK_WORD_ALIGNMENT - 1))
        // _MASK_WORD_ALIGNMENT
    ) * _MASK_WORD_ALIGNMENT
    return (
        model.tc_fixed_ms * launches
        + model.tc_query_ms * queries
        + model.tc_mask_word_ms * queries * words
        + model.tc_tile_ms * tile_visits.astype(jnp.float32)
    )


def sc_cost(model: PrefillCostModel, lengths: jax.Array) -> jax.Array:
    """Estimate SC time from scheduled queries and grouped attention tiles."""
    # Gather always transports the padded capacity, including inactive rows.
    # Attention executes one tile even for an all-inactive group of 8 queries.
    selected = (
        jnp.minimum(jnp.maximum(lengths, 0) // model.ratio, model.budget) * model.ratio
        + jnp.maximum(lengths, 0) % model.ratio
    )
    tiles = jnp.maximum(
        1,
        (selected.reshape(-1, _SC_QUERY_GROUP).max(axis=1) + (_KV_TILE - 1))
        // _KV_TILE,
    )
    return (
        model.sc_fixed_ms
        + model.sc_query_ms * lengths.size
        + model.sc_attention_tile_ms * tiles.astype(jnp.float32).sum()
    )
