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
"""Ragged all-to-all expert-parallel exchange for the grouped-matmul MoE path.

The default data-parallel EP exchange all-gathers every token of the DP group
onto every member and reduce-scatters a full-batch partial output back, so the
bytes each rank moves grow with the group's total token count. Here each
member sends a token only to the members that host at least one of its
experts, once per member, and gets back one partial output row per token it
sent.

Tokens are replicated across the TP ranks of an engine, and each TP rank
exchanges only with the members of its own DP group (the EP ranks that share
its TP rank): a token's experts on the other TP ranks are reached by the
replica that lives there. The sum over TP ranks stays with vLLM's TP
all-reduce, exactly as after the default reduce-scatter.

The expert computation is unchanged. A receiver keeps every source's tokens in
that source's token order, so ``fused_moe_func`` sees the routed rows of every
local expert in the same order as on the all-gathered batch; tokens with no
local expert are left out. Each member's tokens travel as one slice: the
ragged all-to-all costs grow with the number of slices, so packing them with a
gather on each side is cheaper than one slice per token. The home rank adds the
returned partials in DP-position order, where the reduce-scatter used its own
order.

With many DP members, a rank receives only a fraction of the group's tokens
(about a quarter at 32 EP ranks, top-10 of 512 experts) and sees fewer routes
still. The capacity exchange sizes its buffers and the local MoE for a
multiple of those means instead of the whole group; a step whose routing does
not fit on some rank runs the full-size exchange, so results never depend on
the capacity.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func

DP_AXIS = "dp"
TP_AXIS = "tp"
# Capacity buffers grow in whole bf16 row tiles and whole GMM row tiles.
_ROW_TILE = 16
_ROUTE_TILE = 128


def _member_index(
    ids: jax.Array,
    ep_rank: jax.Array,
    *,
    dp_size: int,
    tp_size: int,
    local_num_experts: int,
) -> jax.Array:
    """DP position of the member of this TP rank's DP group that hosts each
    expert id, or ``dp_size`` when no member does (another TP rank's expert,
    or no expert).

    Linear placement puts expert ``e`` on EP rank ``e // local_num_experts``,
    and EP rank ``d * tp_size + r`` is DP position ``d`` at TP rank ``r``.
    """
    owner = ids // local_num_experts
    routed = (ids >= 0) & (owner < dp_size * tp_size)
    here = routed & (owner % tp_size == ep_rank % tp_size)
    return jnp.where(here, owner // tp_size, dp_size)


def exchange_need(
    gathered_ids: jax.Array, ep_rank: jax.Array, *, tp_size: int, local_num_experts: int
) -> jax.Array:
    """Which DP-group members each token of the group must reach.

    ``gathered_ids`` [D, S, K] holds the global expert ids of every token of
    this DP group, by DP position. Returns ``need`` [D, S, D]:
    ``need[s, t, d]`` is true when token ``t`` of member ``s`` has an expert on
    member ``d`` of this TP rank's DP group.
    """
    dp_size = gathered_ids.shape[0]
    dest = _member_index(
        gathered_ids,
        ep_rank,
        dp_size=dp_size,
        tp_size=tp_size,
        local_num_experts=local_num_experts,
    )
    hits = jax.nn.one_hot(dest, dp_size + 1, dtype=jnp.int32)[..., :dp_size]
    return hits.max(axis=2) > 0


def _tile_rows(a: jax.Array) -> jax.Array:
    """[R, H] -> [R, sublanes, H / sublanes]: each row is whole native tiles,
    so the ragged exchange slices the major axis only."""
    rows, hidden = a.shape
    sublanes = 32 // a.dtype.itemsize
    if hidden % (sublanes * 128) == 0:
        return a.reshape(rows, sublanes, hidden // sublanes)
    return a.reshape(rows, 1, hidden)


# A packed route with expert id -1 and weight +0.0: what an empty slot reads.
_EMPTY_ROUTE = -1 << 16


def _gather_routing(topk_ids: jax.Array, topk_weights: jax.Array):
    """All-gather the DP group's routing in one collective when it packs:
    a bf16 weight and an expert id below 2**15 share one int32.

    Returns the gathered ids [D, S, K], a function that reads the routes of
    the given flat (member, token) rows (an out-of-range row reads expert id
    -1 with weight 0), and the packed routes [D * S, K] (None unpacked).
    """
    if topk_weights.dtype != jnp.bfloat16:
        ids = jax.lax.all_gather(topk_ids, DP_AXIS)
        weights = jax.lax.all_gather(topk_weights, DP_AXIS)

        def read(rows):
            flat_ids = ids.reshape(-1, ids.shape[-1])
            flat_weights = weights.reshape(-1, weights.shape[-1])
            return (
                flat_ids.at[rows].get(mode="fill", fill_value=-1),
                flat_weights.at[rows].get(mode="fill", fill_value=0),
            )

        return ids, read, None
    bits = jax.lax.bitcast_convert_type(topk_weights, jnp.uint16).astype(jnp.int32)
    packed = jax.lax.all_gather(jnp.left_shift(topk_ids, 16) | bits, DP_AXIS)

    def read(rows):
        routes = packed.reshape(-1, packed.shape[-1])
        return _unpack_routes(routes.at[rows].get(mode="fill", fill_value=_EMPTY_ROUTE))

    return jnp.right_shift(packed, 16), read, packed.reshape(-1, packed.shape[-1])


def _unpack_routes(routes: jax.Array):
    weights = jax.lax.bitcast_convert_type(
        jnp.bitwise_and(routes, 0xFFFF).astype(jnp.uint16), jnp.bfloat16
    )
    return jnp.right_shift(routes, 16), weights


def exchange_capacity(
    *,
    dp_size: int,
    tp_size: int,
    num_tokens: int,
    topk: int,
    local_num_experts: int,
    factor: float,
    onehot_moe_permute_threshold: int,
    use_sparse_core: bool,
) -> tuple[int, int]:
    """Static sizes of the capacity exchange: ``(rows, routes)``.

    ``rows`` bounds the token rows each rank sends and receives, ``routes``
    the routes hitting its local experts: ``factor`` times their mean under
    uniform routing, rounded up to whole tiles and capped at the sizes that
    always fit (``dp_size * num_tokens`` rows, ``topk`` routes per row). The
    route bound only applies where the bounded routes fit the one-hot path.
    """
    slots = dp_size * num_tokens
    if factor <= 0:
        return slots, slots * topk
    num_ranks = dp_size * tp_size
    num_experts = num_ranks * local_num_experts
    miss = 1.0  # a token's topk distinct experts all on other ranks
    for i in range(topk):
        miss *= max(num_experts - local_num_experts - i, 0) / (num_experts - i)
    rows = -(-math.ceil(factor * slots * (1 - miss)) // _ROW_TILE) * _ROW_TILE
    rows = min(rows, slots)
    routes = math.ceil(factor * slots * topk / num_ranks)
    routes = min(-(-routes // _ROUTE_TILE) * _ROUTE_TILE, rows * topk)
    if not (use_sparse_core and routes <= onehot_moe_permute_threshold):
        return rows, rows * topk
    return rows, routes


def _capacity_overflow(
    gathered_ids: jax.Array,
    *,
    tp_size: int,
    local_num_experts: int,
    rows: int,
    routes: int,
) -> jax.Array:
    """Whether any rank of the mesh would send or receive more than ``rows``
    token rows, or see more than ``routes`` routes to its experts.

    Every rank holds the routing of every engine, so all ranks compute the same
    answer and take the same branch around their collectives.
    """
    dp_size, num_tokens, _ = gathered_ids.shape
    num_ranks = dp_size * tp_size
    owner = gathered_ids // local_num_experts
    owner = jnp.where((gathered_ids >= 0) & (owner < num_ranks), owner, num_ranks)
    hit = owner[..., None] == jnp.arange(num_ranks, dtype=jnp.int32)  # [D, S, K, R]
    rank_routes = hit.sum(axis=(0, 1, 2), dtype=jnp.int32)
    rows_hit = hit.any(axis=2)  # [D, S, R]: token reaches EP rank
    recv_rows = rows_hit.sum(axis=(0, 1), dtype=jnp.int32)
    send_rows = rows_hit.reshape(dp_size, num_tokens, dp_size, tp_size).sum(
        axis=(1, 2), dtype=jnp.int32
    )
    excess = jnp.concatenate(
        [rank_routes - routes, recv_rows - rows, send_rows.reshape(-1) - rows]
    )
    return excess.max() > 0


def _exclusive_cumsum(a: jax.Array, axis: int) -> jax.Array:
    return jnp.cumsum(a, axis=axis, dtype=jnp.int32) - a


def _full_exchange(
    x,
    w1,
    topk_ids,
    gathered_ids,
    read_routes,
    experts_start,
    *,
    tp_size,
    dp_size,
    moe,
):
    """One receive region per DP member, sized for all of its tokens."""
    num_tokens, hidden = x.shape
    local_num_experts = w1.shape[0]
    ep_rank = experts_start[0] // local_num_experts
    dp_pos = ep_rank // tp_size
    slots = dp_size * num_tokens
    member_index = functools.partial(
        _member_index,
        ep_rank=ep_rank,
        dp_size=dp_size,
        tp_size=tp_size,
        local_num_experts=local_num_experts,
    )

    member = jnp.arange(dp_size, dtype=jnp.int32)
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    region = member * num_tokens
    landing = jnp.full((dp_size,), dp_pos * num_tokens, dtype=jnp.int32)

    # Outgoing: member d's region holds this rank's tokens that need d, in
    # token order. This rank's own routing decides it.
    need_out = (member_index(topk_ids)[:, :, None] == member).any(axis=1)  # [S, D]
    send_sizes = need_out.sum(axis=0, dtype=jnp.int32)
    pos_out = jnp.cumsum(need_out, axis=0, dtype=jnp.int32) - 1
    slot_out = jnp.where(need_out, region[None, :] + pos_out, slots)
    send_index = (
        jnp.zeros((slots,), jnp.int32)
        .at[slot_out.reshape(-1)]
        .set(jnp.broadcast_to(token[:, None], need_out.shape).reshape(-1), mode="drop")
    )

    # Incoming: source s's region holds its tokens that need this rank, in
    # its token order; rows past its count read no route.
    need_in = (member_index(gathered_ids) == dp_pos).any(axis=2)  # [D, S]
    recv_sizes = need_in.sum(axis=1, dtype=jnp.int32)
    pos_in = jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1
    slot_in = jnp.where(need_in, region[:, None] + pos_in, slots)
    source_row = (
        jnp.full((slots,), slots, jnp.int32)
        .at[slot_in.reshape(-1)]
        .set((region[:, None] + token[None, :]).reshape(-1), mode="drop")
    )
    recv_ids, recv_weights = read_routes(source_row)

    sent = _tile_rows(x[send_index])
    received = jax.lax.ragged_all_to_all(
        sent,
        jnp.zeros_like(sent),
        region,
        send_sizes,
        landing,
        recv_sizes,
        axis_name=DP_AXIS,
    ).reshape(slots, hidden)

    partial = moe(received, recv_weights, recv_ids)

    # Return path: the rows received from s go back to s, which finds member
    # d's partials at d's region in the order it sent them.
    returned = _tile_rows(partial)
    back = jax.lax.ragged_all_to_all(
        returned,
        jnp.zeros_like(returned),
        region,
        recv_sizes,
        landing,
        send_sizes,
        axis_name=DP_AXIS,
    ).reshape(slots, hidden)
    # A zero row, not a fill-mode gather: with the fill, XLA added the members'
    # partials below in bf16 rather than f32.
    back = jnp.concatenate([back, jnp.zeros((1, hidden), back.dtype)], axis=0)
    rows = back[slot_out.T]  # [D, S, H]; tokens that skipped d read the zero row
    return rows.astype(jnp.float32).sum(axis=0).astype(x.dtype)


def _capacity_exchange(
    x,
    w1,
    topk_ids,
    gathered_ids,
    read_routes,
    packed_routes,
    experts_start,
    overflow,
    *,
    tp_size,
    dp_size,
    rows,
    moe,
):
    """Send and receive buffers of ``rows`` rows, filled without gaps.

    When ``overflow`` is set no rank sends anything (every rank sees the same
    flag), so the buffers cannot overrun; the caller discards the result.

    A rank sends its tokens grouped by destination member, each group in token
    order, and receives the groups of its sources in DP-position order, each
    in that source's token order: the receiver sees the full exchange's rows
    with the empty slots removed. Offsets come from the [D, D] matrix of
    member-to-member token counts, which every member of the DP group derives
    from the gathered routing.
    """
    num_tokens, hidden = x.shape
    local_num_experts = w1.shape[0]
    ep_rank = experts_start[0] // local_num_experts
    dp_pos = ep_rank // tp_size
    member_index = functools.partial(
        _member_index,
        ep_rank=ep_rank,
        dp_size=dp_size,
        tp_size=tp_size,
        local_num_experts=local_num_experts,
    )
    member = jnp.arange(dp_size, dtype=jnp.int32)
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    row = jnp.arange(rows, dtype=jnp.int32)

    fits = jnp.logical_not(overflow)
    need_out = (member_index(topk_ids)[:, :, None] == member).any(axis=1)  # [S, D]
    need_out = need_out & fits
    need = (member_index(gathered_ids)[..., None] == member).any(axis=2)  # [D, S, D]
    need = need & fits
    need_in = (need & (member == dp_pos)).any(axis=2)  # [D, S]
    counts = need.sum(axis=1, dtype=jnp.int32)  # [source, destination]
    send_sizes = need_out.sum(axis=0, dtype=jnp.int32)
    recv_sizes = need_in.sum(axis=1, dtype=jnp.int32)
    send_start = _exclusive_cumsum(send_sizes, 0)
    recv_start = _exclusive_cumsum(recv_sizes, 0)
    earlier = (member < dp_pos).astype(jnp.int32)
    # Where this rank's group lands on each destination: after the groups of
    # the sources before it. Where each source finds this rank's partials:
    # after the groups it sent to the members before this one.
    send_landing = (counts * earlier[:, None]).sum(axis=0)
    back_landing = (counts * earlier[None, :]).sum(axis=1)

    out_row = send_start[None, :] + jnp.cumsum(need_out, axis=0, dtype=jnp.int32) - 1
    out_row = jnp.where(need_out, out_row, rows)  # [S, D]
    send_index = jnp.where(
        out_row[None] == row[:, None, None], token[None, :, None], 0
    ).sum(axis=(1, 2))
    in_row = recv_start[:, None] + jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1
    in_row = jnp.where(need_in, in_row, rows)  # [D, S]
    in_hit = in_row[None] == row[:, None, None]  # [rows, D, S]
    filled = in_hit.any(axis=(1, 2))
    if packed_routes is None:
        source = (member[:, None] * num_tokens + token[None, :])[None]
        source_row = jnp.where(
            filled, jnp.where(in_hit, source, 0).sum(axis=(1, 2)), dp_size * num_tokens
        )
        recv_ids, recv_weights = read_routes(source_row)
    else:
        # At most one hit per row: an integer sum selects the packed route.
        hit = in_hit.reshape(rows, -1, 1)
        routes = jnp.where(hit, packed_routes[None], 0).sum(axis=1)
        recv_ids, recv_weights = _unpack_routes(
            jnp.where(filled[:, None], routes, _EMPTY_ROUTE)
        )

    sent = _tile_rows(x[send_index])
    received = jax.lax.ragged_all_to_all(
        sent,
        jnp.zeros_like(sent),
        send_start,
        send_sizes,
        send_landing,
        recv_sizes,
        axis_name=DP_AXIS,
    ).reshape(rows, hidden)

    partial = moe(received, recv_weights, recv_ids)

    # Every row the home rank reads below is written by this exchange, so the
    # spent send buffer takes the returns.
    back = jax.lax.ragged_all_to_all(
        _tile_rows(partial),
        sent,
        recv_start,
        recv_sizes,
        back_landing,
        send_sizes,
        axis_name=DP_AXIS,
    ).reshape(rows, hidden)
    back = jnp.concatenate([back, jnp.zeros((1, hidden), back.dtype)], axis=0)
    partials = back[out_row.T]  # [D, S, H]; tokens that skipped d read the zero row
    return partials.astype(jnp.float32).sum(axis=0).astype(x.dtype)


def ragged_a2a_moe_local(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array,
    *,
    tp_size: int,
    dp_size: int,
    topk: int,
    activation: str,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool = False,
    capacity_factor: float = 0.0,
) -> jax.Array:
    """Per-rank body (inside shard_map over the (dp, tp) mesh).

    ``x`` [S, H] and the routing [S, K] are this engine's tokens;
    ``experts_start`` [1] is this rank's first global expert id. Returns this
    rank's [S, H] sum over its DP group, before the TP all-reduce.

    With ``capacity_factor`` > 0 and fewer capacity rows than DP-group tokens,
    the exchange and the local MoE are sized by ``exchange_capacity``; a step
    whose routing exceeds those sizes on any rank runs the full-size exchange
    instead. Both give the same result.
    """
    num_tokens = x.shape[0]
    local_num_experts = w1.shape[0]
    slots = dp_size * num_tokens
    gathered_ids, read_routes, packed_routes = _gather_routing(topk_ids, topk_weights)

    def moe(max_local_routes):
        def run(received, recv_weights, recv_ids):
            return fused_moe_func(
                received,
                w1,
                w2,
                w1_scale,
                w2_scale,
                None,
                None,
                recv_weights,
                recv_ids,
                experts_start=experts_start[0],
                topk=topk,
                activation=activation,
                use_ep=True,
                use_sparse_core=use_sparse_core,
                onehot_moe_permute_threshold=onehot_moe_permute_threshold,
                rhs_quant_dtype=rhs_quant_dtype,
                skip_padded_tokens=skip_padded_tokens,
                max_local_routes=max_local_routes,
            )

        return run

    def full():
        return _full_exchange(
            x,
            w1,
            topk_ids,
            gathered_ids,
            read_routes,
            experts_start,
            tp_size=tp_size,
            dp_size=dp_size,
            moe=moe(slots * topk),
        )

    rows, routes = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=local_num_experts,
        factor=capacity_factor,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        use_sparse_core=use_sparse_core,
    )
    if rows >= slots:
        # Bounding the routes alone saves less than the conditional costs
        # (the full exchange already runs the local-routes path over all routes).
        return full()

    overflow = _capacity_overflow(
        gathered_ids,
        tp_size=tp_size,
        local_num_experts=local_num_experts,
        rows=rows,
        routes=routes,
    )
    # The capacity exchange runs outside the conditional, so the schedule
    # around it (weight prefetches for the following layers) is unchanged.
    out = _capacity_exchange(
        x,
        w1,
        topk_ids,
        gathered_ids,
        read_routes,
        packed_routes,
        experts_start,
        overflow,
        tp_size=tp_size,
        dp_size=dp_size,
        rows=rows,
        moe=moe(routes),
    )
    return jax.lax.cond(overflow, full, lambda: out)


def ragged_a2a_moe(
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array,
    w2_scale: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    topk: int,
    activation: str,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool = False,
    capacity_factor: float = 0.0,
) -> jax.Array:
    """Global-view entry: every operand is the ranks' local arrays stacked on
    two leading axes [dp, tp, ...], and so is the result [dp, tp, S, H]."""
    spec = P(DP_AXIS, TP_AXIS)
    local = functools.partial(
        ragged_a2a_moe_local,
        tp_size=mesh.shape[TP_AXIS],
        dp_size=mesh.shape[DP_AXIS],
        topk=topk,
        activation=activation,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
        capacity_factor=capacity_factor,
    )

    def body(*args):
        return local(*(a[0, 0] for a in args))[None, None]

    mapped = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 8, out_specs=spec, check_vma=False
    )
    return mapped(x, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, experts_start)
