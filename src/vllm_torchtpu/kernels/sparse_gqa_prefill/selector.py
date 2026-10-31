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
"""Refine nonnegative TC/SC work bounds until a schedule is selected."""

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import (
    TC_QUERY_BLOCK,
    PrefillCostModel,
    estimate_static_bounds_ms,
    sc_cost,
    tc_cost,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.occupancy import (
    active_counts,
    bucket_upper_counts,
    exact_counts,
    sampled_metadata,
    span_counts,
)

# Prefer TC within two percent of the SC work estimate.
_TC_PREFERENCE_FACTOR = 1.02
_SAMPLED_QUERIES_PER_BLOCK = 8


def tc_selected_by_static_bound(
    *,
    queries: int,
    num_tiles: int,
    model: PrefillCostModel,
    sc_queries: int | None = None,
) -> bool:
    """Select TC when its maximum estimate is within the minimum SC budget."""
    upper_tc, lower_sc = estimate_static_bounds_ms(
        model, queries=queries, num_tiles=num_tiles, sc_queries=sc_queries
    )
    return upper_tc <= lower_sc * _TC_PREFERENCE_FACTOR


def _tc_selected_by_upper_bounds(
    blocks: jax.Array,
    lengths: jax.Array,
    *,
    num_tiles: int,
    model: PrefillCostModel,
    sc_queries: int | None = None,
) -> jax.Array:
    """Select TC using maximum work, then full-row spans if still unresolved."""
    if tc_selected_by_static_bound(
        queries=lengths.size, num_tiles=num_tiles, model=model, sc_queries=sc_queries
    ):
        return jnp.array(True)
    # TC pads to Q256, whereas SC pads only to its gather microchunk.
    target = sc_cost(model, lengths[:sc_queries]) * _TC_PREFERENCE_FACTOR

    def estimate(visits):
        return tc_cost(
            model,
            queries=lengths.size,
            num_tiles=num_tiles,
            tile_visits=jnp.asarray(visits),
        )

    return lax.cond(
        estimate((lengths.size // TC_QUERY_BLOCK) * num_tiles) <= target,
        lambda _: jnp.array(True),
        lambda _: estimate(
            span_counts(blocks, lengths, ratio=model.ratio, kv_tile=model.kv_tile).sum()
        )
        <= target,
        None,
    )


def select_sc(
    blocks: jax.Array,
    lengths: jax.Array,
    *,
    num_tiles: int,
    model: PrefillCostModel,
    sc_queries: int | None = None,
) -> jax.Array:
    """Return True for SC, False for TC, using progressively tighter bounds.

    Order: static TC bound, full-row TC upper bounds, sampled SC lower bound,
    coarse TC upper bound, then exact counts. Sampling alone cannot select TC.
    Each unresolved branch retains lazy device-side evaluation via lax.cond.
    """
    if tc_selected_by_static_bound(
        queries=lengths.size, num_tiles=num_tiles, model=model, sc_queries=sc_queries
    ):
        return jnp.array(False)
    # TC pads to Q256, whereas SC pads only to its gather microchunk.
    target = sc_cost(model, lengths[:sc_queries]) * _TC_PREFERENCE_FACTOR

    def estimate(visits):
        return tc_cost(
            model, queries=lengths.size, num_tiles=num_tiles, tile_visits=visits
        )

    def choose_from_sampled_lower_bound(_):
        ids, ls = sampled_metadata(blocks, lengths)
        lower = active_counts(
            ids,
            ls,
            num_tiles=num_tiles,
            ratio=model.ratio,
            kv_tile=model.kv_tile,
            query_block=_SAMPLED_QUERIES_PER_BLOCK,
        ).sum()
        return lax.cond(
            estimate(lower) > target,
            lambda _: jnp.array(True),
            choose_from_bucket_upper_bound,
            None,
        )

    def choose_from_bucket_upper_bound(_):
        upper = bucket_upper_counts(
            blocks,
            lengths,
            num_tiles=num_tiles,
            ratio=model.ratio,
            kv_tile=model.kv_tile,
        ).sum()
        return lax.cond(
            estimate(upper) <= target,
            lambda _: jnp.array(False),
            choose_from_exact_counts,
            None,
        )

    def choose_from_exact_counts(_):
        return (
            estimate(
                exact_counts(
                    blocks,
                    lengths,
                    num_tiles=num_tiles,
                    ratio=model.ratio,
                    kv_tile=model.kv_tile,
                ).sum()
            )
            > target
        )

    return lax.cond(
        _tc_selected_by_upper_bounds(
            blocks, lengths, num_tiles=num_tiles, model=model, sc_queries=sc_queries
        ),
        lambda _: jnp.array(False),
        choose_from_sampled_lower_bound,
        None,
    )
