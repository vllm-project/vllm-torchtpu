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
"""Decode routing inputs, retaining the native slab and SparseCore boundaries."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.decode_preparation import (
    ROUTING_TOKENS,
    admit,
    first_problem,
    fuse_function,
    prefix_sum,
    tpu_version_problem,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe.host import (
    build_routing_tables_sharded,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host

# The kernel's static group slices and [destination, expert] tables were
# lowered and checked for this routing geometry: (topk, experts, ep, tile_m).
ROUTING_GEOMETRY = (6, 384, 16, 128)


def routing_problem(
    gathered, runs, me, *, e_total, ep, t_local, block, tile_m, shard_stride
) -> str | None:
    """First violated contract of the fused routing kernel, or None."""
    geometry = (gathered.topk, e_total, ep, tile_m)
    return (
        first_problem(
            (
                geometry == ROUTING_GEOMETRY,
                f"(topk, experts, ep, tile_m)={geometry}, kernel {ROUTING_GEOMETRY}",
            ),
            (gathered.n_tokens == t_local * ep, "tokens are not t_local * ep"),
            (
                block == host.routing_block(t_local, gathered.topk),
                f"routing block {block} is not the derived block",
            ),
            (
                shard_stride
                == host.ragged_stride_bound(
                    gathered.n_tokens, gathered.topk, e_total, tile_m
                ),
                f"shard stride {shard_stride} is not the ragged stride bound",
            ),
            (
                all(
                    x.dtype == jnp.int32
                    for x in (*runs, gathered.local_blocks, gathered.block_hist, me)
                ),
                "routing tables are not int32",
            ),
        )
        or tpu_version_problem()
    )


class RoutingInputs(NamedTuple):
    pos: jax.Array
    mirror_pos: jax.Array
    slab_row: jax.Array
    kernel_tables: tuple[jax.Array, ...]
    expert_rows: jax.Array
    slab_base: jax.Array
    counts: jax.Array


def _metadata(true, aligned, starts, expert_rows, rank, shard_stride):
    """All tables stay [destination, expert]; no [16,24,16] reshape."""
    ep, experts = true.shape
    g = experts // ep
    d = jnp.arange(ep, dtype=jnp.int32)[:, None]
    mine = d == rank[0]
    recv = prefix_sum(aligned, axis=1) - aligned

    def local(table):
        # Mosaic dynamic gather cannot cross source vregs (E384 > lane128).
        # Static group slices expose the small fixed movement to the compiler.
        result = jnp.zeros((table.shape[0], g), jnp.int32)
        for owner in range(ep):
            result += jnp.where(
                rank[0] == owner, table[:, owner * g : (owner + 1) * g], 0
            )
        return result

    local_true, local_aligned, local_start = local(true), local(aligned), local(starts)
    rows = local(expert_rows)
    slab_base = prefix_sum(rows, axis=1) - rows
    local_base = slab_base + rank[0] * shard_stride
    off = prefix_sum(local_aligned, axis=1) - local_aligned
    out_total = local_aligned.sum(axis=1, keepdims=True)
    contrib = off + prefix_sum(out_total, axis=0) - out_total
    recv_off = local(recv)
    tiles = jnp.where(
        aligned > 0,
        ((starts + jnp.maximum(aligned, 1) - 1) >> 7) - (starts >> 7) + 1,
        0,
    )
    local_tiles = local(tiles)
    true_local_total = local_true.sum(axis=1, keepdims=True)
    tiles_local_total = local_tiles.sum(axis=1, keepdims=True)
    true_recv = mine * (true.sum(axis=1, keepdims=True) - true_local_total)
    aligned_recv = mine * (aligned.sum(axis=1, keepdims=True) - out_total)
    block_totals = (
        jnp.concatenate(
            (
                jnp.sum(out_total * ~mine, keepdims=True),
                jnp.sum(aligned_recv, keepdims=True),
            ),
            axis=1,
        )
        >> host.ROWBLK_SHIFT
    )
    row_totals = jnp.concatenate(
        (
            jnp.sum(true_local_total * ~mine, keepdims=True),
            jnp.sum(true_recv, keepdims=True),
        ),
        axis=1,
    )
    count_cols = (
        true_local_total * ~mine,
        true_recv,
        out_total * ~mine,
        aligned_recv,
        mine * (rows > 0).sum(axis=1, keepdims=True),
        tiles_local_total * ~mine,
        mine * (tiles.sum(axis=1, keepdims=True) - tiles_local_total),
        tiles_local_total * mine,
    )
    return (
        jnp.sum(jnp.where(mine, recv, 0), axis=0, keepdims=True),
        jnp.sum(jnp.where(mine, starts, 0), axis=0, keepdims=True),
        local_base + local_start,
        local_start >> host.ROWBLK_SHIFT,
        local_aligned >> host.ROWBLK_SHIFT,
        contrib >> host.ROWBLK_SHIFT,
        contrib >> host.ROWBLK_SHIFT,
        local_aligned >> host.ROWBLK_SHIFT,
        recv_off >> host.ROWBLK_SHIFT,
        block_totals,
        local_true,
        recv_off,
        row_totals,
        rows,
        slab_base,
        jnp.concatenate(count_cols, axis=1).T,
    )


def _position_metadata(true, aligned, starts, rows, rank, local, hist, *, shard_stride):
    tables = _metadata(true, aligned, starts, rows, rank, shard_stride)
    offset = prefix_sum(hist, axis=0) - hist
    bins = jnp.arange(true.shape[1], dtype=jnp.int32)
    within = host._rank_within_block(local, local.shape[1])
    pos_table = offset + tables[0]
    mirror_table = (
        offset + ((tables[0] >> 7) + 2 * bins[None, :]) * 128 + (tables[1] & 127)
    )
    pos = host._lookup_per_pair(pos_table, local, bins) + within
    mirror = host._lookup_per_pair(mirror_table, local, bins) + within
    # The kernel reads its transport tables as [expert, destination] SMEM
    # arrays; emitting that orientation here removes one XLA transpose copy
    # per table after the call.
    transport = tuple(tables[i].T for i in (3, 4, 5, 8, 10))
    return pos, mirror, *transport, *(tables[i] for i in (13, 14, 15))


def prepare_routing(
    me: jax.Array,
    gathered: host.GatheredPairs,
    runs: tuple[jax.Array, ...],
    *,
    e_total: int,
    ep: int,
    t_local: int,
    block: int,
    tile_m: int,
    shard_stride: int,
    fuse_metadata: bool = True,
) -> RoutingInputs:
    """Build the live consumer inputs, using fusion only in the measured envelope."""
    routing = build_routing_tables_sharded(
        me,
        gathered,
        runs,
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=block,
        tile_m=tile_m,
        shard_stride=shard_stride,
    )
    slab_row = host.local_slab_rows(routing, me, shard_stride=shard_stride)
    supported = fuse_metadata and admit(
        "moe_routing",
        routing_problem(
            gathered,
            runs,
            me,
            e_total=e_total,
            ep=ep,
            t_local=t_local,
            block=block,
            tile_m=tile_m,
            shard_stride=shard_stride,
        ),
        tokens=t_local,
        buckets=ROUTING_TOKENS,
    )
    if not supported:
        blocks = host.shard_transport_tables_in_blocks(
            routing, me, e_total=e_total, ep=ep
        )
        row_tables = host.shard_push_tables_in_rows(routing, me, e_total=e_total, ep=ep)
        expert_rows, base = host.shard_expert_slabs(routing, me, e_total=e_total, ep=ep)
        counts = host.shard_count_vector(
            routing, expert_rows, me, e_total=e_total, ep=ep
        )
        return RoutingInputs(
            routing.pos,
            routing.mirror_pos,
            slab_row,
            (*blocks[:3], blocks[5], row_tables[0]),
            expert_rows,
            base,
            counts,
        )
    true, aligned, starts, rows = runs
    inputs = (
        true.T,
        aligned.T,
        starts.T,
        rows[None, :],
        me.reshape(1),
        gathered.local_blocks,
        gathered.block_hist,
    )
    function = partial(_position_metadata, shard_stride=shard_stride)
    outputs = fuse_function(function, inputs, name="dsv41_routing_position_metadata")(
        *inputs
    )
    pos, mirror, *tables = outputs
    return RoutingInputs(
        pos.reshape(t_local, gathered.topk),
        mirror.reshape(t_local, gathered.topk),
        slab_row,
        tuple(tables[:5]),
        tables[5].reshape(-1),
        tables[6].reshape(-1),
        tables[7],
    )
