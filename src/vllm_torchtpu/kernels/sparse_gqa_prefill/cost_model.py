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
"""Calibrated runtime estimates in milliseconds for Q24/KV2/D256 prefill."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Fixed geometry of the calibrated schedule, not hardware capacity limits.
TC_QUERY_BLOCK = 256
_KV_TILE = 256
_SC_QUERY_GROUP = 8
_COMPRESSION_RATIO = 4
_MAX_SELECTED_GROUPS = 512
_MASK_WORD_ALIGNMENT = 128
_MASK_WORDS_PER_KV_TILE = _KV_TILE // _COMPRESSION_RATIO // jnp.iinfo(jnp.uint32).bits


class PrefillCostModel(NamedTuple):
    tc_fixed_ms: float
    tc_query_ms: float
    tc_mask_word_ms: float
    tc_tile_ms: float
    sc_fixed_ms: float
    sc_query_ms: float
    sc_attention_tile_ms: float


# Calibrated coefficients convert the schedule's work counts to milliseconds.
COST_MODELS: dict[jnp.dtype, PrefillCostModel] = {
    jnp.dtype(jnp.bfloat16): PrefillCostModel(
        tc_fixed_ms=0.25000650408450037,
        tc_query_ms=8.991181022104898e-05,
        tc_mask_word_ms=4.814218452898145e-08,
        tc_tile_ms=0.004999265026206821,
        sc_fixed_ms=0.19743081215678976,
        sc_query_ms=0.005833960374709633,
        sc_attention_tile_ms=0.002476169547627867,
    ),
    jnp.dtype(jnp.float8_e4m3fn): PrefillCostModel(
        tc_fixed_ms=0.2491017824685177,
        tc_query_ms=8.889156791776907e-05,
        tc_mask_word_ms=4.724934780077619e-08,
        tc_tile_ms=0.005145153550878911,
        sc_fixed_ms=0.19896896616666457,
        sc_query_ms=0.0032255933216442456,
        sc_attention_tile_ms=0.0038026017010685054,
    ),
}


def estimate_static_bounds_ms(
    model: PrefillCostModel,
    *,
    queries: int,
    num_tiles: int,
    sc_queries: int | None = None,
) -> tuple[float, float]:
    """Return the maximum TC estimate and minimum scheduled SC estimate."""
    words = (
        (_MASK_WORDS_PER_KV_TILE * num_tiles + (_MASK_WORD_ALIGNMENT - 1))
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
    model: PrefillCostModel, *, queries: int, num_tiles: int, tile_visits: jax.Array
) -> jax.Array:
    """Estimate TC time from scheduled queries, mask words, and tile visits."""
    words = (
        (_MASK_WORDS_PER_KV_TILE * num_tiles + (_MASK_WORD_ALIGNMENT - 1))
        // _MASK_WORD_ALIGNMENT
    ) * _MASK_WORD_ALIGNMENT
    return (
        model.tc_fixed_ms
        + model.tc_query_ms * queries
        + model.tc_mask_word_ms * queries * words
        + model.tc_tile_ms * tile_visits.astype(jnp.float32)
    )


def sc_cost(model: PrefillCostModel, lengths: jax.Array) -> jax.Array:
    """Estimate SC time from scheduled queries and grouped attention tiles."""
    # Gather always transports the padded capacity, including inactive rows.
    # Attention executes one tile even for an all-inactive group of 8 queries.
    selected = (
        jnp.minimum(lengths // _COMPRESSION_RATIO, _MAX_SELECTED_GROUPS)
        * _COMPRESSION_RATIO
        + lengths % _COMPRESSION_RATIO
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
