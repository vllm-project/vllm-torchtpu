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
"""Shared JAX ragged sequence-indexing helpers for KDA and Mamba layers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def token_sequence_ids(
    query_start_loc: jax.Array,
    num_tokens: int,
    num_seqs: int,
) -> jax.Array:
    """Map each token row to the 0-indexed sequence that owns it.

    A token's sequence index is the number of interior sequence starts
    (``query_start_loc[1:num_seqs]``) lying at or below its row, so marking each
    start and taking an inclusive cumsum computes the mapping in one linear pass
    over the token axis instead of ``jnp.searchsorted``'s ``O(T * num_seqs)``
    broadcast comparison at small request buckets.

    ``mode="drop"`` discards starts at or past ``num_tokens`` (dummy AOT runs
    may describe more requests than fit in the token bucket), while scatter-add
    accumulates duplicate starts so zero-length interior sequences advance the
    index by more than one.
    """
    if num_tokens == 0 or num_seqs <= 1:
        return jnp.zeros((num_tokens,), dtype=jnp.int32)
    starts = jnp.maximum(query_start_loc[1:num_seqs], 0)
    marks = jnp.zeros((num_tokens,), dtype=jnp.int32).at[starts].add(1, mode="drop")
    return jnp.minimum(jnp.cumsum(marks), num_seqs - 1)


def clip_query_start_loc(
    query_start_loc: jax.Array,
    distribution: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Clip ``query_start_loc`` to ``distribution[2]`` valid sequences.

    Returns ``(num_valid_seqs, effective_query_start_loc, query_lens)``.
    """
    num_seqs = query_start_loc.shape[0] - 1
    if distribution is not None:
        num_valid_seqs = jnp.clip(distribution[2], 0, num_seqs)
        valid_loc_mask = (
            jnp.arange(query_start_loc.shape[0], dtype=jnp.int32) <= num_valid_seqs
        )
        last_valid_loc = query_start_loc[num_valid_seqs]
        effective_query_start_loc = jnp.where(
            valid_loc_mask, query_start_loc, last_valid_loc
        )
    else:
        num_valid_seqs = jnp.asarray(num_seqs, dtype=jnp.int32)
        effective_query_start_loc = query_start_loc

    query_lens = jnp.maximum(
        effective_query_start_loc[1 : num_seqs + 1]
        - effective_query_start_loc[:num_seqs],
        0,
    )
    return num_valid_seqs, effective_query_start_loc, query_lens
