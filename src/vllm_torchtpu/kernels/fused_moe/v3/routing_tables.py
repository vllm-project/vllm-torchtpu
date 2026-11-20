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
"""Where every routed (token, expert) pair computes and where its result
lands: the routing tables in their reference form (dense jnp passes), and
the shard tables sliced from them. The layer builds the same tables with
routing_tables_kernel.shard_tables_kernel; the tests hold the kernel to
this form.

Rows are ordered by expert, then by the shard that owns the token, each
(expert, shard) run padded to a whole ROW_BLOCK. Every shard holds the
whole set (it is replicated) and slices its own tables from it.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.fused_moe.v3.layout import (
    INT32_LIMIT,
    ROW_BLOCK,
    ROW_BLOCK_SHIFT,
    align_up,
)

# The tables pack an arrival position and an alignment slot into one int32
# as position * field + slot. The slots a mesh needs (up to
# (ROW_BLOCK - 1) * (shards - 1)) have to fit under the field, and the
# position above it. ALIGNMENT_SLOT_FIELD is the field of meshes up to 10
# shards; alignment_slot_field() widens it for wider meshes.
ALIGNMENT_SLOT_FIELD = 64
# The widest block the tables bin routed pairs over. The rank pass inside a
# block is quadratic in it.
MAX_ROUTING_BLOCK = 256
# Rows of the count table the FFN kernel takes: rows sent and received,
# the same two in padded rows, and the experts visited. Each row is still
# spread over the shards, and the kernel's scalar core sums it.
COUNT_ROWS = 5


class RoutingTables(NamedTuple):
    """The tables the kernel indexes: per pair, per expert, per push."""

    arrival_row: jax.Array  # [T, K] arrival row of token t's k-th pair
    routed_row: jax.Array  # [T * K] routed row each pair computes on
    run_rows: jax.Array  # [E, num_shards] true rows of expert e for shard d
    run_rows_aligned: jax.Array  # [E, num_shards] padded to ROW_BLOCK
    run_start_aligned: jax.Array  # [E, num_shards] its padded start
    region_rows: jax.Array  # [source, g, dest] rows one push carries
    recv_base: jax.Array  # [dest, source, g] arrival row it lands on
    recv_rows: jax.Array  # [dest] arrival rows a shard receives
    expert_rows_aligned: jax.Array  # [E] padded rows per expert
    expert_base: jax.Array  # [E] routed row each expert starts at
    routed_rows: int  # routed rows one shard is allocated


def routed_rows_bound(num_tokens, top_k, num_experts, tile_rows):
    """Routed rows per shard the no-drop worst case needs: every routed row on
    one shard, the row-block padding of every expert, one tile of
    tail-read window. A whole number of tiles. Part of the built kernel's
    cache key, so a program is compiled per token bucket."""
    return align_up(
        num_tokens * top_k + (ROW_BLOCK - 1) * num_experts + tile_rows, tile_rows
    )


def routing_block(tokens_per_shard, top_k):
    """The widest power-of-two block dividing a shard's routed pairs."""
    block = MAX_ROUTING_BLOCK
    while (tokens_per_shard * top_k) % block:
        block //= 2
    return block


def alignment_slot_field(num_shards):
    """The power-of-two field a mesh's alignment slots pack into: at least
    ALIGNMENT_SLOT_FIELD, and wide enough for the (ROW_BLOCK - 1) *
    (num_shards - 1) slots the widest run can need."""
    slots = (ROW_BLOCK - 1) * (num_shards - 1) + 1
    return max(ALIGNMENT_SLOT_FIELD, 1 << (slots - 1).bit_length())


def check_routing_shape(*, num_tokens, top_k, num_experts, num_shards):
    """Raises ValueError for a shape the packed table word cannot hold."""
    if num_tokens < 1:
        raise ValueError("the routing tables need at least one token")
    if num_shards < 1 or num_experts % num_shards:
        raise ValueError(
            f"expert count {num_experts} is not divisible by "
            f"the expert-parallel width {num_shards}"
        )
    field = alignment_slot_field(num_shards)
    max_position = num_tokens * top_k + (ROW_BLOCK - 1) * num_experts
    if max_position * field >= INT32_LIMIT:
        raise ValueError(
            f"up to {max_position} arrival rows per shard, packed beside a "
            f"{field}-wide alignment slot, overflow an int32"
        )


def build_routing_tables(
    top_k_indices,
    *,
    num_experts,
    num_shards,
    tokens_per_shard,
    block,
    tile_rows,
    routed_rows_per_shard,
):
    """Assigns every routed pair the routed row it computes on: the five
    O(pairs x experts) passes as dense jnp ops.

    Args:
      top_k_indices: [T, K] int32 expert ids of every token, all-gathered.
    """
    T, K = top_k_indices.shape
    n = T * K
    check_routing_shape(
        num_tokens=T, top_k=K, num_experts=num_experts, num_shards=num_shards
    )
    if n % block or (tokens_per_shard * K) % block:
        raise ValueError(
            f"{n} routed pairs, {tokens_per_shard * K} per shard, "
            f"are not whole {block}-pair blocks"
        )
    if tile_rows % ROW_BLOCK:
        raise ValueError(
            f"tile height {tile_rows} is not a whole number of {ROW_BLOCK}-row blocks"
        )
    if routed_rows_per_shard % tile_rows:
        raise ValueError(
            f"the routed rows stride {routed_rows_per_shard} is not a whole "
            f"number of {tile_rows}-row tiles"
        )
    experts_per_shard = num_experts // num_shards
    slot_field = alignment_slot_field(num_shards)
    expert_of_pair = top_k_indices.reshape(-1).astype(jnp.int32)
    n_blocks = n // block
    expert_blocks = expert_of_pair.reshape(n_blocks, block)
    bins = jnp.arange(num_experts, dtype=jnp.int32)
    one_hot = expert_blocks[:, :, None] == bins[None, None, :]
    block_hist = jnp.sum(one_hot.astype(jnp.int32), axis=1)  # [blocks, E]
    block_offset = jnp.cumsum(block_hist, axis=0) - block_hist
    base_per_slot = jnp.sum(jnp.where(one_hot, block_offset[:, None, :], 0), axis=2)
    same = expert_blocks[:, :, None] == expert_blocks[:, None, :]
    earlier = jnp.tril(jnp.ones((block, block), dtype=jnp.bool_), k=-1)
    rank = jnp.sum((same & earlier[None]).astype(jnp.int32), axis=2)
    pair_rank = (base_per_slot + rank).reshape(-1)
    blocks_per_shard = (tokens_per_shard * K) // block
    run_rows = (
        block_hist.reshape(num_shards, blocks_per_shard, num_experts).sum(axis=1).T
    )  # [E, num_shards]
    run_start = jnp.cumsum(run_rows, axis=1) - run_rows
    run_rows_aligned = (run_rows + (ROW_BLOCK - 1)) & jnp.int32(-ROW_BLOCK)
    run_start_aligned = jnp.cumsum(run_rows_aligned, axis=1) - run_rows_aligned
    slot_shift = run_start_aligned - run_start
    expert_rows_aligned = run_rows_aligned.sum(axis=1)
    rows_by_shard = expert_rows_aligned.reshape(num_shards, experts_per_shard)
    local_base = jnp.cumsum(rows_by_shard, axis=1) - rows_by_shard
    expert_base = (
        local_base
        + jnp.arange(num_shards, dtype=jnp.int32)[:, None] * routed_rows_per_shard
    ).reshape(num_experts)
    region_rows = run_rows_aligned.reshape(num_shards, experts_per_shard, num_shards)
    rows_per_dest = region_rows.transpose(2, 0, 1).reshape(
        num_shards, num_shards * experts_per_shard
    )
    recv_base = (jnp.cumsum(rows_per_dest, axis=1) - rows_per_dest).reshape(
        num_shards, num_shards, experts_per_shard
    )
    recv_rows = rows_per_dest.sum(axis=1)
    position_shift = (
        recv_base.transpose(1, 2, 0).reshape(num_experts, num_shards) - run_start
    )
    shard_of_block = (
        jnp.arange(n_blocks, dtype=jnp.int32) * block // (tokens_per_shard * K)
    )
    packed = position_shift * slot_field + slot_shift
    packed_blocks = jnp.take(packed.T, shard_of_block, axis=0)
    packed_shift = jnp.sum(
        jnp.where(one_hot, packed_blocks[:, None, :], 0), axis=2
    ).reshape(-1)
    base_of_pair = jnp.sum(
        jnp.where(one_hot, expert_base[None, None, :], 0), axis=2
    ).reshape(-1)
    position = pair_rank + jnp.right_shift(packed_shift, slot_field.bit_length() - 1)
    slot = pair_rank + (packed_shift & (slot_field - 1))
    return RoutingTables(
        arrival_row=position.reshape(T, K),
        routed_row=base_of_pair + slot,
        run_rows=run_rows,
        run_rows_aligned=run_rows_aligned,
        run_start_aligned=run_start_aligned,
        region_rows=region_rows,
        recv_base=recv_base,
        recv_rows=recv_rows,
        expert_rows_aligned=expert_rows_aligned,
        expert_base=expert_base,
        routed_rows=num_shards * routed_rows_per_shard,
    )


def local_routed_rows(routing, shard, *, routed_rows_per_shard):
    """`routing.routed_row` rebased onto this shard's routed rows. A pair
    of another
    shard is sent to routed_rows_per_shard, past the end, which a scatter with
    mode="drop" ignores (never negative: that would wrap)."""
    base = shard * routed_rows_per_shard
    return jnp.where(
        routing.routed_row < base,
        jnp.int32(routed_rows_per_shard),
        routing.routed_row - base,
    )


def shard_token_table(routing, shard, *, routed_rows_per_shard):
    """The token each of this shard's routed rows computes,
    [routed_rows_per_shard],
    clamped into the token range: the kernel fetches rows by this value
    with bounds checks off."""
    T, K = routing.arrival_row.shape
    token_of_pair = jnp.broadcast_to(
        jnp.arange(T, dtype=jnp.int32)[:, None], (T, K)
    ).reshape(-1)
    row = local_routed_rows(routing, shard, routed_rows_per_shard=routed_rows_per_shard)
    table = (
        jnp.zeros((routed_rows_per_shard,), jnp.int32)
        .at[row]
        .add(token_of_pair, mode="drop")
    )
    return jnp.clip(table, jnp.int32(0), jnp.int32(max(T - 1, 0)))


def shard_expert_rows(routing, shard, *, num_experts, num_shards):
    """This shard's local experts: (padded rows, start row) int32 [G]."""
    experts_per_shard = num_experts // num_shards
    rows = lax.dynamic_slice(
        routing.expert_rows_aligned, (shard * experts_per_shard,), (experts_per_shard,)
    )
    base = lax.dynamic_slice(
        routing.expert_base, (shard * experts_per_shard,), (experts_per_shard,)
    )
    return rows.astype(jnp.int32), (base - base[0]).astype(jnp.int32)


def visit_order(expert_rows, experts_per_shard):
    """The local experts with rows, most rows first (ties by index), as a
    [G] table whose unvisited tail is never read."""
    order = jnp.arange(experts_per_shard, dtype=jnp.int32)
    key = order - expert_rows.astype(jnp.int32) * experts_per_shard
    return jnp.minimum(
        jnp.argsort(key).astype(jnp.int32), jnp.int32(experts_per_shard - 1)
    )


def _receive_base_row(routing, shard, *, num_shards, experts_per_shard):
    """Where shard `shard`'s runs land in every destination's arrival rows,
    int32 [G, num_shards]."""
    received = lax.dynamic_slice(
        routing.recv_base, (0, shard, 0), (num_shards, 1, experts_per_shard)
    )
    return received[:, 0].T


def shard_push_tables(routing, shard, *, num_experts, num_shards):
    """This shard's push tables in ROW units: the true rows of each (expert,
    destination) run and the arrival row it lands on, int32 [G, num_shards]
    each."""
    experts_per_shard = num_experts // num_shards
    push_rows = lax.dynamic_slice(
        routing.run_rows,
        (shard * experts_per_shard, 0),
        (experts_per_shard, num_shards),
    )
    arrival_offset = _receive_base_row(
        routing, shard, num_shards=num_shards, experts_per_shard=experts_per_shard
    )
    return push_rows.astype(jnp.int32), arrival_offset.astype(jnp.int32)


def shard_transport_tables(routing, shard, *, num_experts, num_shards):
    """This shard's transport tables in ROW_BLOCK units, int32
    [G, num_shards] each:
    where each (expert, destination) run starts in the expert's rows, the
    blocks it holds, the outgoing-routed rows block it commits to (also the
    push source), the blocks a push carries and the arrival block it lands
    on."""
    experts_per_shard = num_experts // num_shards
    run_rows = lax.dynamic_slice(
        routing.run_rows_aligned,
        (shard * experts_per_shard, 0),
        (experts_per_shard, num_shards),
    )
    run_start = lax.dynamic_slice(
        routing.run_start_aligned,
        (shard * experts_per_shard, 0),
        (experts_per_shard, num_shards),
    )
    region_rows = lax.dynamic_slice(
        routing.region_rows, (shard, 0, 0), (1, experts_per_shard, num_shards)
    )[0]
    per_dest = region_rows.sum(axis=0)
    dest_base = jnp.cumsum(per_dest) - per_dest
    outgoing_offset = dest_base[None, :] + (
        jnp.cumsum(region_rows, axis=0) - region_rows
    )
    push_destination = _receive_base_row(
        routing, shard, num_shards=num_shards, experts_per_shard=experts_per_shard
    )

    def blocks(rows):
        return jnp.right_shift(rows, ROW_BLOCK_SHIFT).astype(jnp.int32)

    return (
        blocks(run_start),
        blocks(run_rows),
        blocks(outgoing_offset),
        blocks(region_rows),
        blocks(push_destination),
    )


def shard_counts(routing, expert_rows, shard, *, num_experts, num_shards):
    """The five counts the FFN kernel closes, [COUNT_ROWS, num_shards]
    int32, each
    still spread over the shards: rows sent, rows received, the same two
    in padded rows, and the experts visited."""
    experts_per_shard = num_experts // num_shards
    true_rows = lax.dynamic_slice(
        routing.run_rows,
        (shard * experts_per_shard, 0),
        (experts_per_shard, num_shards),
    ).astype(jnp.int32)
    aligned = lax.dynamic_slice(
        routing.run_rows_aligned,
        (shard * experts_per_shard, 0),
        (experts_per_shard, num_shards),
    ).astype(jnp.int32)
    mine = (jnp.arange(num_shards, dtype=jnp.int32) == shard).astype(jnp.int32)
    other = 1 - mine
    first = (jnp.arange(experts_per_shard, dtype=jnp.int32) == 0).astype(jnp.int32)[
        :, None
    ]
    received = routing.run_rows.sum(axis=0).astype(jnp.int32)
    received_aligned = routing.recv_rows.astype(jnp.int32)
    active = (expert_rows > 0).astype(jnp.int32)[:, None]

    def over_experts(term):
        return term.sum(axis=0, keepdims=True)

    return jnp.concatenate(
        [
            over_experts(true_rows * other[None, :]),
            over_experts(
                (received * mine)[None, :] * first - true_rows * mine[None, :]
            ),
            over_experts(aligned * other[None, :]),
            over_experts(
                (received_aligned * mine)[None, :] * first - aligned * mine[None, :]
            ),
            over_experts(active * mine[None, :]),
        ],
        axis=0,
    ).astype(jnp.int32)
