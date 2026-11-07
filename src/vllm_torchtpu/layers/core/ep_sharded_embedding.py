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
"""Token embedding lookup with the vocabulary split over every DP x TP rank.

A vocab-parallel embedding splits the table over the TP group only, so each
of the DP engines holds the whole table. Here rank ``dp_rank * tp + tp_rank``
holds rows ``[rank * rows, (rank + 1) * rows)``. Each rank looks up the ids
of every engine in its slice, a reduce-scatter over the DP axis returns each
engine its own tokens, and the TP all-reduce of the vocab-parallel lookup
finishes the sum. Exactly one rank contributes each row and every other rank
adds +0.0, as in the TP-only lookup with TP > 1, so the result is bitwise that
lookup.
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

DP_AXIS = "dp"
TP_AXIS = "tp"


def ep_embedding_lookup_local(
    ids: jax.Array, table: jax.Array, start: jax.Array, *, dp_size: int, tp_size: int
) -> jax.Array:
    """One rank's part: its engine's ids [T] -> that engine's rows [T, H].

    ``start`` [1] is the first vocabulary row of this rank's slice, passed as
    data: the multi-process SPMD compile does not lower a partition id.
    """
    rows = table.shape[0]
    all_ids = jax.lax.all_gather(ids, DP_AXIS)  # [D, T]
    local = all_ids - start[0]
    owned = (local >= 0) & (local < rows)
    picked = jnp.take(table, jnp.where(owned, local, 0), axis=0)  # [D, T, H]
    picked = jnp.where(owned[..., None], picked, jnp.zeros((), table.dtype))
    if dp_size > 1:
        picked = jax.lax.psum_scatter(picked, DP_AXIS, scatter_dimension=0)
    else:
        picked = picked[0]
    if tp_size > 1:
        picked = jax.lax.psum(picked, TP_AXIS)
    return picked


def ep_embedding_lookup(
    ids: jax.Array, table: jax.Array, start: jax.Array, *, mesh: jax.sharding.Mesh
) -> jax.Array:
    """Global-view entry: operands are the ranks' local arrays stacked on two
    leading axes, ids [dp, tp, T], table [dp, tp, rows, H] and start [dp, tp, 1]
    (rank ``dp_rank * tp + tp_rank`` starts at that rank times ``rows``); the
    result is [dp, tp, T, H] (equal over each engine's TP ranks)."""
    spec = P(DP_AXIS, TP_AXIS)
    local = functools.partial(
        ep_embedding_lookup_local,
        dp_size=mesh.shape[DP_AXIS],
        tp_size=mesh.shape[TP_AXIS],
    )

    def body(i, t, s):
        return local(i[0, 0], t[0, 0], s[0, 0])[None, None]

    mapped = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
    )
    return mapped(ids, table, start)
