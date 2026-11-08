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

The reduce-scatter moves every engine's whole [T, H] partial, almost all
zeros, so its bytes grow with the DP group. With ``owned_rows``, each rank
instead sends only the rows it owns to their engines' members of its TP
column, with one all-to-all over the DP axis, and the TP all-reduce finishes
the sum as before: the same rows plus the same +0.0 terms.
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

DP_AXIS = "dp"
TP_AXIS = "tp"


def _column_rows(all_ids, local, owned, table, start, *, tp_size, ragged):
    """This rank's engine's rows [T, H] that the ranks of this rank's TP column
    own, zeros elsewhere. Each owner sends its rows of engine e's tokens, in
    token order, to member e of the column; member d's rows land in region d
    of the receive buffer. ``ragged`` sends only those rows; otherwise every
    region moves whole (same result, for backends without the ragged op)."""
    dp_size, num_tokens = all_ids.shape
    rows, hidden = table.shape
    rank = start[0] // rows
    dp_pos, column = rank // tp_size, rank % tp_size
    # Region e of the send buffer: this rank's rows of engine e's tokens.
    order = jnp.argsort(~owned, axis=1, stable=True)
    send_sizes = owned.sum(axis=1, dtype=jnp.int32)
    src = jnp.take_along_axis(jnp.where(owned, local, 0), order, axis=1)
    sent = jnp.take(table, src.reshape(-1), axis=0)  # [D * T, H]
    # This engine's tokens: owner rank of each row, and the column's share.
    my_ids = jnp.take(all_ids, dp_pos, axis=0)
    owner = my_ids // rows
    mine = (my_ids >= 0) & (owner < dp_size * tp_size) & (owner % tp_size == column)
    owner_dp = owner // tp_size
    member = jnp.arange(dp_size, dtype=jnp.int32)
    region = member * num_tokens
    if ragged:
        recv_sizes = jnp.sum(
            (owner_dp[None, :] == member[:, None]) & mine[None, :],
            axis=1,
            dtype=jnp.int32,
        )
        received = jax.lax.ragged_all_to_all(
            sent,
            jnp.zeros_like(sent),
            region,
            send_sizes,
            jnp.full((dp_size,), dp_pos * num_tokens, jnp.int32),
            recv_sizes,
            axis_name=DP_AXIS,
        )
    else:
        received = jax.lax.all_to_all(
            sent.reshape(dp_size, num_tokens, hidden), DP_AXIS, 0, 0, tiled=False
        ).reshape(dp_size * num_tokens, hidden)
    # Token i's row: region owner_dp[i], at its rank among this engine's tokens
    # with the same owner.
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    earlier = (owner[None, :] == owner[:, None]) & (token[None, :] < token[:, None])
    slot = jnp.take(region, jnp.clip(owner_dp, 0, dp_size - 1)) + earlier.sum(
        axis=1, dtype=jnp.int32
    )
    picked = jnp.take(received, jnp.where(mine, slot, 0), axis=0)
    return jnp.where(mine[:, None], picked, jnp.zeros((), table.dtype))


def ep_embedding_lookup_local(
    ids: jax.Array,
    table: jax.Array,
    start: jax.Array,
    *,
    dp_size: int,
    tp_size: int,
    owned_rows: str | None = None,
) -> jax.Array:
    """One rank's part: its engine's ids [T] -> that engine's rows [T, H].

    ``start`` [1] is the first vocabulary row of this rank's slice, passed as
    data: the multi-process SPMD compile does not lower a partition id.
    ``owned_rows`` ("ragged", or "dense" where the ragged all-to-all is not
    available) sends each owned row to its engine instead of reduce-scattering
    every engine's partial over DP; the result is the same.
    """
    rows = table.shape[0]
    all_ids = jax.lax.all_gather(ids, DP_AXIS)  # [D, T]
    local = all_ids - start[0]
    owned = (local >= 0) & (local < rows)
    if dp_size > 1 and owned_rows is not None:
        assert owned_rows in ("ragged", "dense"), owned_rows
        picked = _column_rows(
            all_ids,
            local,
            owned,
            table,
            start,
            tp_size=tp_size,
            ragged=owned_rows == "ragged",
        )
    else:
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
    ids: jax.Array,
    table: jax.Array,
    start: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    owned_rows: str | None = None,
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
        owned_rows=owned_rows,
    )

    def body(i, t, s):
        return local(i[0, 0], t[0, 0], s[0, 0])[None, None]

    mapped = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
    )
    return mapped(ids, table, start)
