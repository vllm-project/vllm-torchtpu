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
gather on each side is cheaper than one slice per token. The sending rank adds
the returned partials in DP-position order, where the reduce-scatter used its
own order.

With many DP members, a rank receives only a fraction of the group's tokens
(about a quarter at 32 EP ranks, top-10 of 512 experts) and sees fewer routes
still. The capacity exchange sizes its buffers and the local MoE for a
multiple of those means instead of the whole group; a step whose routing does
not fit on some rank runs the full-size exchange, so results never depend on
the capacity. Real routing loads the busiest rank with about twice the mean
rows and more than twice the mean routes, so the routes take their own factor.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.dma_all_gather import dma_all_gather
from vllm_torchtpu.kernels.megablox.row_copy import copy_send_rows
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    fused_moe_scatter,
    fused_moe_scatter_planned,
    scatter_moe_supported,
)

DP_AXIS = "dp"
TP_AXIS = "tp"
# Capacity buffers grow in whole bf16 row tiles and whole GMM row tiles.
_ROW_TILE = 16
_ROUTE_TILE = 128
# Largest factor used where the full-size exchange stays on the one-hot path:
# its rerun is cheap there, so larger buffers cost more than they save.
_ONEHOT_FALLBACK_FACTOR = 2.0


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


def _gather_routing(topk_ids: jax.Array, topk_weights: jax.Array, *, dma_members=0):
    """All-gather the DP group's routing in one collective when it packs:
    a bf16 weight and an expert id below 2**15 share one int32. With
    ``dma_members`` (the DP group's size) the packed routing travels by
    `dma_all_gather` instead of XLA's all-gather: the same words.

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
    routes = jnp.left_shift(topk_ids, 16) | bits
    if dma_members:
        # One [K, S] block per member: the tokens fill the lanes.
        packed = dma_all_gather(routes.T, axis=DP_AXIS, num_members=dma_members)
        packed = jnp.swapaxes(packed, 1, 2)
    else:
        packed = jax.lax.all_gather(routes, DP_AXIS)

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
    route_factor: float | None = None,
    row_tile: int = _ROW_TILE,
) -> tuple[int, int]:
    """Static sizes of the capacity exchange: ``(rows, routes)``.

    ``rows`` bounds the token rows each rank sends and receives, ``routes``
    the routes hitting its local experts: ``factor`` (``route_factor`` for
    the routes, when given) times their mean under uniform routing, rounded
    up to whole tiles (rows in ``row_tile``-row tiles) and capped at the
    sizes that always fit (``dp_size * num_tokens`` rows, ``topk`` routes
    per row). The route bound only applies where the bounded routes fit the
    one-hot path. Where the full-size exchange's routes fit it too, both
    factors are capped at ``_ONEHOT_FALLBACK_FACTOR``.
    """
    slots = dp_size * num_tokens
    if factor <= 0:
        return slots, slots * topk
    if use_sparse_core and slots * topk <= onehot_moe_permute_threshold:
        factor = min(factor, _ONEHOT_FALLBACK_FACTOR)
        route_factor = min(route_factor or factor, _ONEHOT_FALLBACK_FACTOR)
    num_ranks = dp_size * tp_size
    num_experts = num_ranks * local_num_experts
    miss = 1.0  # a token's topk distinct experts all on other ranks
    for i in range(topk):
        miss *= max(num_experts - local_num_experts - i, 0) / (num_experts - i)
    rows = -(-math.ceil(factor * slots * (1 - miss)) // row_tile) * row_tile
    rows = min(rows, slots)
    routes = math.ceil((route_factor or factor) * slots * topk / num_ranks)
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


def route_order(ids: jax.Array, in_group: jax.Array) -> jax.Array:
    """Position of each route among its token's routes: the routes to experts of
    this TP rank's DP group first, by expert id (so grouped by member in DP
    order, each member's routes in its expert order), then the others.

    ``ids``/``in_group`` [N, K]; returns int32 [N, K], a permutation of 0..K-1
    per row. Expert and sending ranks compute it from the same routing.
    """
    topk = ids.shape[-1]
    k = jnp.arange(topk, dtype=jnp.int32)
    key = jnp.where(in_group, ids, (1 << 20) + k)
    before = (key[..., None, :] < key[..., :, None]) | (
        (key[..., None, :] == key[..., :, None]) & (k[None, :] < k[:, None])
    )
    return before.sum(axis=-1, dtype=jnp.int32)


def route_plan(
    gathered_ids,
    in_row,
    fits,
    experts_start,
    dests,
    *,
    local_num_experts: int,
    routes: int,
    rows: int,
    gathered_weights=None,
):
    """This rank's local routes in expert order, straight from the DP group's
    routing, for `fused_moe_scatter_planned`.

    ``gathered_ids`` [D, S, K]: the group's expert ids by DP position;
    ``in_row`` [D, S]: the receive row of token ``t`` of member ``s`` when it
    reaches this rank (rows hold the sources in DP order, each in its token
    order); ``dests(ids)`` with ``ids`` [K, D, S] -> (member [D, S], row
    [K, D, S], number of members, receive rows): where each route's output
    goes. ``gathered_weights`` [D, S, K] (skipping padded tokens): a route
    with weight 0 is not computed.

    Returns (token_sorted, group_sizes, dest_member, dest_row) over the first
    ``routes`` routes in (local expert, source, token, k) order. A receive row
    holds the tokens of its source in token order, so this is the order of
    sorting the received rows' routes by (local expert, row, k), as
    `fused_moe_scatter` does. ``fits`` false: no local routes.

    The routes are laid out k-major ([K, D, S], tokens along the minor axis),
    where per-route work over K is cheap; the sort does not depend on it.
    """
    num_members, num_tokens, topk = gathered_ids.shape
    ids = jnp.moveaxis(gathered_ids, -1, 0)
    local = ids - experts_start[0]
    valid = (local >= 0) & (local < local_num_experts) & fits
    if gathered_weights is not None:
        weights = jnp.moveaxis(gathered_weights, -1, 0)
        weights = jnp.where(jnp.isfinite(weights), weights, 0)
        valid = valid & (weights != 0)
    member, row, total_members, recv_rows = dests(ids)
    expert = jnp.where(valid, local, local_num_experts)
    member = jnp.broadcast_to(member[None], ids.shape)
    experts = jnp.arange(local_num_experts, dtype=jnp.int32)[:, None, None, None]
    group_sizes = (expert[None] == experts).sum(axis=(1, 2, 3), dtype=jnp.int32)
    expert_bits = local_num_experts.bit_length()
    token_bits = max(1, (rows - 1).bit_length())
    member_bits = max(1, (total_members - 1).bit_length())
    row_bits = max(1, (recv_rows - 1).bit_length())
    dest_bits = member_bits + row_bits
    if expert_bits + token_bits + dest_bits < 32:
        # One int32 per route: (local expert, receive row, destination). A
        # token's routes name distinct experts, so the computed routes' keys
        # are distinct and in (local expert, row) order; the receive row
        # stands for (source, token).
        token = jnp.where(valid, in_row[None], 0)
        key = (
            jnp.left_shift(expert, token_bits + dest_bits)
            | jnp.left_shift(token, dest_bits)
            | jnp.left_shift(member, row_bits)
            | row
        )
        key = jax.lax.sort(key.reshape(-1))[:routes]
        computed = jnp.right_shift(key, token_bits + dest_bits) < local_num_experts
        token_sorted = jnp.right_shift(key, dest_bits) & ((1 << token_bits) - 1)
        # Routes that are not computed read a zero row.
        token_sorted = jnp.where(computed, token_sorted, rows)
        dest_member = jnp.right_shift(key, row_bits) & ((1 << member_bits) - 1)
        dest_row = key & ((1 << row_bits) - 1)
        return token_sorted, group_sizes, dest_member, dest_row
    num_routes = num_members * num_tokens * topk
    route_bits = max(1, (num_routes - 1).bit_length())
    assert expert_bits + route_bits < 32
    # Route index in (source, token, k) order.
    route = (
        jnp.arange(num_members * num_tokens, dtype=jnp.int32).reshape(
            1, num_members, num_tokens
        )
        * topk
        + jnp.arange(topk, dtype=jnp.int32)[:, None, None]
    )
    key = jnp.left_shift(expert, route_bits) | route
    # Routes that are not computed read a zero row.
    token = jnp.where(valid, in_row[None], rows)
    if max(1, rows.bit_length()) + dest_bits < 32:
        payload = (
            jnp.left_shift(token, dest_bits) | jnp.left_shift(member, row_bits) | row
        )
        _, payload = jax.lax.sort((key.reshape(-1), payload.reshape(-1)), num_keys=1)
        payload = payload[:routes]
        token_sorted = jnp.right_shift(payload, dest_bits)
        dest_member = jnp.right_shift(payload, row_bits) & ((1 << member_bits) - 1)
        dest_row = payload & ((1 << row_bits) - 1)
    else:
        _, token_sorted, dest_member, dest_row = jax.lax.sort(
            tuple(a.reshape(-1) for a in (key, token, member, row)), num_keys=1
        )
        token_sorted = token_sorted[:routes]
        dest_member = dest_member[:routes]
        dest_row = dest_row[:routes]
    return token_sorted, group_sizes, dest_member, dest_row


def _rank_along_k(key):
    """[K, ...] distinct keys per route column -> each key's rank among its
    column (the number of smaller keys), the reduction running over the major
    axis."""
    return (key[:, None] < key[None, :]).sum(axis=0, dtype=jnp.int32)


def streamed_dests(ids, *, member_index, dp_size):
    """`route_plan` destinations of `_streamed_combine` (``ids`` [K, D, S]):
    token ``t`` of member ``s`` gets its routes back at rows ``route_order *
    S + t`` of ``s``."""
    topk, _, num_tokens = ids.shape
    t = jnp.arange(num_tokens, dtype=jnp.int32)
    member = jnp.broadcast_to(
        jnp.arange(dp_size, dtype=jnp.int32)[:, None], ids.shape[1:]
    )
    # route_order, k-major: routes of this TP rank's group first by expert id,
    # then the others by k (keys distinct per token).
    k = jnp.arange(topk, dtype=jnp.int32)[:, None, None]
    key = jnp.where(member_index(ids) < dp_size, ids, (1 << 20) + k)
    row = _rank_along_k(key) * num_tokens + t
    return member, row, dp_size, num_tokens * topk


def owner_dests(ids, *, dp_size, tp_size, local_num_experts):
    """`route_plan` destinations of `_owner_combine` (``ids`` [K, D, S]):
    token ``t`` of member ``s`` is owned by TP rank ``t // blk`` of engine
    ``s``, which gets its routes at rows ``owner_order * blk + t % blk``."""
    topk, _, num_tokens = ids.shape
    blk = -(-num_tokens // tp_size)
    s = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    t = jnp.arange(num_tokens, dtype=jnp.int32)[None, :]
    member = s * tp_size + t // blk
    # owner_order, k-major.
    valid, key, _, _ = owner_route_key(
        ids, dp_size=dp_size, tp_size=tp_size, local_num_experts=local_num_experts
    )
    k = jnp.arange(topk, dtype=jnp.int32)[:, None, None]
    order = _rank_along_k(jnp.where(valid, key, (1 << 24) + k))
    row = order * blk + t % blk
    return member, row, dp_size * tp_size, blk * topk


def _sender_routes(
    topk_ids, topk_weights, *, member_index, dp_size, skip_padded_tokens, dtype
):
    """This rank's routes by position in ``route_order``, each [K, S]: the
    member serving it (``dp_size``: none of this group), whether an output
    comes back for it, and its routing weight in f32."""
    topk = topk_ids.shape[1]
    member = member_index(topk_ids)
    weights = topk_weights.astype(dtype)
    weights = jnp.where(jnp.isfinite(weights), weights, 0)
    valid = member < dp_size
    if skip_padded_tokens:
        valid = valid & (weights != 0)
    order = route_order(topk_ids, member < dp_size)
    pick = order[:, :, None] == jnp.arange(topk, dtype=jnp.int32)  # [S, k, j]
    member_s = jnp.where(pick, member[:, :, None], 0).sum(axis=1).T
    valid_s = (pick & valid[:, :, None]).any(axis=1).T
    weight_s = jnp.where(pick, weights[:, :, None].astype(jnp.float32), 0).sum(axis=1).T
    return member_s, valid_s, weight_s


def _sender_sum(
    recv,
    topk_ids,
    topk_weights,
    *,
    member_index,
    dp_size,
    skip_padded_tokens,
    dtype,
):
    """This rank's [S, H] sum of the expert outputs its tokens' routes got back.

    ``recv`` [K * S, 16, H / 16] holds route ``j`` of token ``t`` (in
    ``route_order``) at row ``j * S + t``, unweighted. As the combine of the
    returned partials: per member, its routes weighted and summed in f32 in its
    expert order and rounded once to the activation dtype; then the members'
    partials summed in f32 in DP order and rounded once.
    """
    num_tokens, topk = topk_ids.shape
    member_s, valid_s, weight_s = _sender_routes(
        topk_ids,
        topk_weights,
        member_index=member_index,
        dp_size=dp_size,
        skip_padded_tokens=skip_padded_tokens,
        dtype=dtype,
    )
    rows = recv.reshape(topk, num_tokens, *recv.shape[1:])
    shape = (num_tokens,) + recv.shape[1:]
    info = jnp.finfo(dtype)

    def round_to(a):
        # The rounding a returned partial gets in memory. A convert pair to
        # ``dtype`` and back may be folded away by XLA; reduce_precision is not.
        return jax.lax.reduce_precision(
            a, exponent_bits=info.nexp, mantissa_bits=info.nmant
        )

    bcast = (slice(None),) + (None,) * (len(shape) - 1)
    acc = jnp.zeros(shape, jnp.float32)
    out = jnp.zeros(shape, jnp.float32)
    for j in range(topk):
        prod = jnp.where(
            valid_s[j][bcast],
            weight_s[j][bcast] * rows[j].astype(jnp.float32),
            0.0,
        )
        if j:
            acc = jnp.where((member_s[j] == member_s[j - 1])[bcast], acc, 0.0)
        acc = acc + prod
        end = member_s[j] < dp_size
        if j + 1 < topk:
            end = end & (member_s[j + 1] != member_s[j])
        out = out + jnp.where(end[bcast], round_to(acc), 0.0)
    return out.astype(dtype).reshape(num_tokens, -1)


def _sum_tokens_per_step(num_tokens: int) -> int:
    """Tokens per grid step of the sum kernels: a divisor of the token
    count, at most 8."""
    return max(d for d in range(1, 9) if num_tokens % d == 0)


def sender_sum_kernel(
    recv,
    topk_ids,
    topk_weights,
    *,
    member_index,
    dp_size,
    skip_padded_tokens,
    dtype,
):
    """`_sender_sum` as one Pallas kernel (same arithmetic and order).

    XLA splits the position loop into several fusions with f32 intermediates
    and relayouts; here each token's rows stream through VMEM once.
    """
    num_tokens, topk = topk_ids.shape
    member_s, valid_s, weight_s = _sender_routes(
        topk_ids,
        topk_weights,
        member_index=member_index,
        dp_size=dp_size,
        skip_padded_tokens=skip_padded_tokens,
        dtype=dtype,
    )
    valid_s = valid_s.astype(jnp.int32)
    rows = recv.reshape(topk, num_tokens, *recv.shape[1:])
    tb = _sum_tokens_per_step(num_tokens)
    row_shape = recv.shape[1:]

    def kernel(member_ref, valid_ref, weight_ref, rows_ref, out_ref):
        base = pl.program_id(0) * tb
        for i in range(tb):
            t = base + i
            acc = jnp.zeros(row_shape, jnp.float32)
            out = jnp.zeros(row_shape, jnp.float32)
            for j in range(topk):
                m = member_ref[j, t]
                w = weight_ref[j, t]
                row = rows_ref[j, i].astype(jnp.float32)
                prod = jnp.where(valid_ref[j, t] != 0, w * row, 0.0)
                if j:
                    acc = jnp.where(m == member_ref[j - 1, t], acc, 0.0)
                acc = acc + prod
                end = m < dp_size
                if j + 1 < topk:
                    end = jnp.logical_and(end, member_ref[j + 1, t] != m)
                # The rounding a returned partial gets in memory.
                out = out + jnp.where(end, acc.astype(dtype).astype(jnp.float32), 0.0)
            out_ref[i] = out.astype(out_ref.dtype)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_tokens,) + row_shape, dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(num_tokens // tb,),
            in_specs=[
                pl.BlockSpec(
                    (topk, tb) + row_shape, lambda g, *_: (0, g) + (0,) * len(row_shape)
                )
            ],
            out_specs=pl.BlockSpec(
                (tb,) + row_shape, lambda g, *_: (g,) + (0,) * len(row_shape)
            ),
        ),
        name="ragged_a2a_sender_sum",
    )(member_s, valid_s, weight_s, rows)
    return out.reshape(num_tokens, -1)


def _receive_buffer(sent, row_gather):
    """The ragged all-to-all's output buffer: zeros, or uninitialized when the
    row gather reads only the rows that the sources send."""
    if row_gather:
        return jax.lax.empty(sent.shape, sent.dtype)
    return jnp.zeros_like(sent)


def _scatter_send(x, out_row_t, *, rows):
    """The send buffer [rows, 16, H / 16] written by `copy_send_rows`: token
    ``t``'s row at ``out_row_t[d, t]`` for each member ``d`` it goes to
    (``out_row_t`` [D, S], destination-major; ``rows`` where it does not)."""
    num_tokens = x.shape[0]
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    row = jnp.arange(rows, dtype=jnp.int32)[:, None, None]
    index = jnp.where(out_row_t[None] == row, token, 0).sum(axis=(1, 2))
    count = jnp.max(jnp.where(out_row_t < rows, out_row_t + 1, 0), initial=0)
    return copy_send_rows(x, index, count[None], rows)


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
    streamed=None,
    plan=None,
    row_gather=False,
    row_scatter=False,
):
    """One receive region per DP member, sized for all of its tokens.

    With ``streamed`` (see ``_streamed_combine``), the expert kernel sends the
    outputs to their senders instead of the return exchange; with ``plan`` too
    (see ``ragged_a2a_moe_local``), it gets its routes from `route_plan`, and
    with ``row_gather`` it copies their rows out of the receive buffer;
    ``row_scatter`` writes the send buffer with `copy_send_rows`."""
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
    planned = streamed is not None and plan is not None
    routes_plan = source_row = recv_ids = recv_weights = None
    if planned:
        dests, gathered_weights = plan
        routes_plan = route_plan(
            gathered_ids,
            region[:, None] + pos_in,
            jnp.bool_(True),
            experts_start,
            dests,
            local_num_experts=local_num_experts,
            routes=slots * topk_ids.shape[1],
            rows=slots,
            gathered_weights=gathered_weights,
        )
    else:
        slot_in = jnp.where(need_in, region[:, None] + pos_in, slots)
        source_row = (
            jnp.full((slots,), slots, jnp.int32)
            .at[slot_in.reshape(-1)]
            .set((region[:, None] + token[None, :]).reshape(-1), mode="drop")
        )
        recv_ids, recv_weights = read_routes(source_row)

    if planned and row_scatter:
        sent = _scatter_send(x, slot_out.T, rows=slots)
    else:
        sent = _tile_rows(x[send_index])
    gather = planned and row_gather
    received = jax.lax.ragged_all_to_all(
        sent,
        _receive_buffer(sent, gather),
        region,
        send_sizes,
        landing,
        recv_sizes,
        axis_name=DP_AXIS,
    )
    if not gather:
        received = received.reshape(slots, hidden)

    if planned:
        # The last source's region ends the received rows.
        src_rows = (region[-1] + recv_sizes[-1])[None]
        return streamed(
            received,
            None,
            None,
            None,
            None,
            dp_pos,
            jnp.bool_(True),
            plan=routes_plan,
            src_rows=src_rows,
        )
    if streamed is not None:
        sender = jnp.where(source_row < slots, source_row, 0)
        return streamed(
            received,
            recv_weights,
            recv_ids,
            sender // num_tokens,
            sender % num_tokens,
            dp_pos,
            jnp.bool_(True),
        )

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
    streamed=None,
    plan=None,
    routes=0,
    row_gather=False,
    row_scatter=False,
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

    With ``streamed`` and ``plan`` (see ``ragged_a2a_moe_local``), the expert
    side gets its ``routes`` routes from `route_plan` and the receive rows'
    sources and routes are never materialized; with ``row_gather`` it copies
    their rows out of the receive buffer, and ``row_scatter`` writes the send
    buffer with `copy_send_rows`.
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
    need_any = (member_index(topk_ids)[:, :, None] == member).any(axis=1)  # [S, D]
    need_out = need_any & fits
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

    if streamed is not None and plan is not None:
        # Destination-major: the running counts go along the minor axis.
        need_out_t = need_out.T  # [D, S]
        out_row_t = (
            send_start[:, None] + jnp.cumsum(need_out_t, axis=1, dtype=jnp.int32) - 1
        )
        out_row_t = jnp.where(need_out_t, out_row_t, rows)
        send_index = jnp.where(
            out_row_t[None] == row[:, None, None], token[None, None, :], 0
        ).sum(axis=(1, 2))
        in_row = recv_start[:, None] + jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1
        dests, gathered_weights = plan
        routes_plan = route_plan(
            gathered_ids,
            in_row,
            fits,
            experts_start,
            dests,
            local_num_experts=local_num_experts,
            routes=routes,
            rows=rows,
            gathered_weights=gathered_weights,
        )
        if row_scatter:
            # Written before the overflow flag is known: a step that does not
            # fit sends nothing, and rows past the buffer are dropped.
            need_any_t = need_any.T  # [D, S]
            scatter_row_t = (
                _exclusive_cumsum(need_any_t.sum(axis=1, dtype=jnp.int32), 0)[:, None]
                + jnp.cumsum(need_any_t, axis=1, dtype=jnp.int32)
                - 1
            )
            sent = _scatter_send(
                x, jnp.where(need_any_t, scatter_row_t, rows), rows=rows
            )
        else:
            sent = _tile_rows(x[send_index])
        received = jax.lax.ragged_all_to_all(
            sent,
            _receive_buffer(sent, row_gather),
            send_start,
            send_sizes,
            send_landing,
            recv_sizes,
            axis_name=DP_AXIS,
        )
        if not row_gather:
            received = received.reshape(rows, hidden)
        # The received rows fill the buffer from row 0.
        src_rows = jnp.sum(recv_sizes, dtype=jnp.int32)[None]
        return streamed(
            received,
            None,
            None,
            None,
            None,
            dp_pos,
            fits,
            plan=routes_plan,
            src_rows=src_rows,
        )

    out_row = send_start[None, :] + jnp.cumsum(need_out, axis=0, dtype=jnp.int32) - 1
    out_row = jnp.where(need_out, out_row, rows)  # [S, D]
    send_index = jnp.where(
        out_row[None] == row[:, None, None], token[None, :, None], 0
    ).sum(axis=(1, 2))
    in_row = recv_start[:, None] + jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1
    in_row = jnp.where(need_in, in_row, rows)  # [D, S]
    in_hit = in_row[None] == row[:, None, None]  # [rows, D, S]
    filled = in_hit.any(axis=(1, 2))
    # The sender of each received row, for the streamed combine.
    sender_member = jnp.where(in_hit, member[None, :, None], 0).sum(axis=(1, 2))
    sender_token = jnp.where(in_hit, token[None, None, :], 0).sum(axis=(1, 2))
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

    if streamed is not None:
        return streamed(
            received, recv_weights, recv_ids, sender_member, sender_token, dp_pos, fits
        )

    partial = moe(received, recv_weights, recv_ids)

    # Every row the sender reads below is written by this exchange, so the
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


def _streamed_combine(
    received,
    recv_weights,
    recv_ids,
    sender_member,
    sender_token,
    dp_pos,
    fits,
    *,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_ids,
    topk_weights,
    experts_start,
    member_index,
    dp_size,
    topk,
    activation,
    rhs_quant_dtype,
    skip_padded_tokens,
    max_local_routes,
    collective_id,
    plan=None,
    row_gather=False,
    src_rows=None,
):
    """The expert kernel sends each output row to the rank that sent its token.

    Received row ``r`` came from token ``sender_token[r]`` of member
    ``sender_member[r]``; its route ``k`` lands at row ``route_order * S +
    sender_token`` of that member's receive buffer, which holds K rows per token
    of this rank's engine. ``fits`` false: nothing was dispatched (the caller
    discards the result), so nothing is expected either. ``plan``: the routes
    from `route_plan` with `streamed_dests` (the received rows' sources and
    routes are not needed); ``row_gather``: ``received`` holds whole-tile rows
    and the routes' rows are copied out of it, all below ``src_rows`` [1].
    """
    num_tokens = x.shape[0]
    member = member_index(topk_ids)
    weights = topk_weights.astype(x.dtype)
    weights = jnp.where(jnp.isfinite(weights), weights, 0)
    remote = (member < dp_size) & (member != dp_pos)
    if skip_padded_tokens:
        remote = remote & (weights != 0)
    expected = jnp.where(fits, remote.sum(dtype=jnp.int32), 0)
    if plan is not None:
        recv = fused_moe_scatter_planned(
            received,
            w1,
            w2,
            w1_scale,
            w2_scale,
            *plan,
            jnp.asarray(dp_pos, jnp.int32)[None],
            expected[None],
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            src_rows=src_rows,
            num_recv_rows=num_tokens * topk,
            axis=DP_AXIS,
            num_members=dp_size,
            collective_id=collective_id,
            row_gather=row_gather,
        )
    else:
        in_group = member_index(recv_ids) < dp_size
        row_dest = route_order(recv_ids, in_group) * num_tokens + sender_token[:, None]
        recv = fused_moe_scatter(
            received,
            w1,
            w2,
            w1_scale,
            w2_scale,
            recv_weights,
            recv_ids,
            experts_start[0],
            sender_member.astype(jnp.int32),
            row_dest.astype(jnp.int32),
            jnp.asarray(dp_pos, jnp.int32)[None],
            expected[None],
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
            max_local_routes=max_local_routes,
            num_recv_rows=num_tokens * topk,
            axis=DP_AXIS,
            num_members=dp_size,
            collective_id=collective_id,
        )
    return sender_sum_kernel(
        recv,
        topk_ids,
        topk_weights,
        member_index=member_index,
        dp_size=dp_size,
        skip_padded_tokens=skip_padded_tokens,
        dtype=x.dtype,
    )


def owner_route_key(ids, *, dp_size, tp_size, local_num_experts):
    """(valid, key, tp, dp) of each route for the owner-side sum: routes ordered
    by the TP rank of their expert, then its DP position, then the local
    expert; ``valid`` false for ids that name no expert of the mesh."""
    num_ranks = dp_size * tp_size
    rank = ids // local_num_experts
    valid = (ids >= 0) & (rank < num_ranks)
    tp = rank % tp_size
    dp = rank // tp_size
    key = (tp * dp_size + dp) * local_num_experts + ids % local_num_experts
    return valid, key, tp, dp


def owner_order(ids, **mesh):
    """Position of each route among its token's routes in owner-key order
    (invalid ids last, by k): a permutation of 0..K-1 per row."""
    topk = ids.shape[-1]
    valid, key, _, _ = owner_route_key(ids, **mesh)
    k = jnp.arange(topk, dtype=jnp.int32)
    key = jnp.where(valid, key, (1 << 24) + k)
    return (key[..., None, :] < key[..., :, None]).sum(axis=-1, dtype=jnp.int32)


def _owner_routes(topk_ids, topk_weights, *, skip_padded_tokens, dtype, **mesh):
    """Per owner-order position, each [K, N]: TP rank of the route's expert
    (``tp_size``: none), its DP position, whether an output comes for it, its
    weight in f32."""
    topk = topk_ids.shape[1]
    tp_size = mesh["tp_size"]
    valid, _, tp, dp = owner_route_key(topk_ids, **mesh)
    weights = topk_weights.astype(dtype)
    weights = jnp.where(jnp.isfinite(weights), weights, 0)
    sent = valid & (weights != 0) if skip_padded_tokens else valid
    order = owner_order(topk_ids, **mesh)
    pick = order[:, :, None] == jnp.arange(topk, dtype=jnp.int32)  # [N, k, j]

    def at(v):
        return jnp.where(pick, v[:, :, None], 0).sum(axis=1).T

    tp_s = at(jnp.where(valid, tp, tp_size))
    dp_s = at(dp)
    sent_s = (pick & sent[:, :, None]).any(axis=1).T
    weight_s = jnp.where(pick, weights[:, :, None].astype(jnp.float32), 0).sum(axis=1).T
    return tp_s, dp_s, sent_s, weight_s


def _owner_sum_ref(
    recv, topk_ids, topk_weights, *, skip_padded_tokens, dtype, shared=None, **mesh
):
    """jnp form of `owner_sum_kernel` (CPU tests)."""
    num_tokens, topk = topk_ids.shape
    tp_size = mesh["tp_size"]
    tp_s, dp_s, sent_s, weight_s = _owner_routes(
        topk_ids,
        topk_weights,
        skip_padded_tokens=skip_padded_tokens,
        dtype=dtype,
        **mesh,
    )
    rows = recv.reshape(topk, num_tokens, *recv.shape[1:])
    shape = (num_tokens,) + recv.shape[1:]
    info = jnp.finfo(dtype)

    def round_to(a):
        return jax.lax.reduce_precision(
            a, exponent_bits=info.nexp, mantissa_bits=info.nmant
        )

    b = (slice(None),) + (None,) * (len(shape) - 1)
    acc_m = jnp.zeros(shape, jnp.float32)
    acc_r = jnp.zeros(shape, jnp.float32)
    out = jnp.zeros(shape, jnp.float32)
    parts = [jnp.zeros(shape, jnp.float32) for _ in range(tp_size)]
    for j in range(topk):
        prod = jnp.where(
            sent_s[j][b], weight_s[j][b] * rows[j].astype(jnp.float32), 0.0
        )
        if j:
            new_r = tp_s[j] != tp_s[j - 1]
            new_m = new_r | (dp_s[j] != dp_s[j - 1])
            acc_r = jnp.where(new_r[b], 0.0, acc_r)
            acc_m = jnp.where(new_m[b], 0.0, acc_m)
        acc_m = acc_m + prod
        group = tp_s[j] < tp_size
        if j + 1 < topk:
            end_r = tp_s[j + 1] != tp_s[j]
            end_m = end_r | (dp_s[j + 1] != dp_s[j])
        else:
            end_r = end_m = jnp.ones_like(group)
        acc_r = acc_r + jnp.where((group & end_m)[b], round_to(acc_m), 0.0)
        if shared is None:
            out = out + jnp.where((group & end_r)[b], round_to(acc_r), 0.0)
        else:
            parts = [
                jnp.where(
                    (group & end_r & (tp_s[j] == rr))[b], round_to(acc_r), parts[rr]
                )
                for rr in range(tp_size)
            ]
    if shared is not None:
        sh = shared.reshape(tp_size, num_tokens, *recv.shape[1:]).astype(jnp.float32)
        for rr in range(tp_size):
            out = out + round_to(sh[rr] + parts[rr])
    return out.astype(dtype).reshape(num_tokens, -1)


def owner_sum_kernel(
    recv, topk_ids, topk_weights, *, skip_padded_tokens, dtype, shared=None, **mesh
):
    """This rank's owned tokens' MoE outputs from all their routes, in one
    Pallas kernel. ``recv`` [K * N, 16, H / 16]: route position ``j`` (owner
    order) of owned token ``i`` at row ``j * N + i``. Per TP rank of the
    experts: per DP member, its routes summed in f32 in expert order and
    rounded once; the members summed in f32 and rounded once (the per-rank
    partial of the returned-partials path); then the TP ranks' partials summed
    in f32 in rank order and rounded once. With ``shared`` [tp * N, 16, H / 16]
    (TP rank ``r``'s shared-expert partial of owned token ``i`` at row
    ``r * N + i``), each TP rank's partial first adds its shared partial with
    one rounding, as the model adds the two before its TP all-reduce."""
    num_tokens, topk = topk_ids.shape
    tp_size = mesh["tp_size"]
    tp_s, dp_s, sent_s, weight_s = _owner_routes(
        topk_ids,
        topk_weights,
        skip_padded_tokens=skip_padded_tokens,
        dtype=dtype,
        **mesh,
    )
    sent_s = sent_s.astype(jnp.int32)
    rows = recv.reshape(topk, num_tokens, *recv.shape[1:])
    tb = _sum_tokens_per_step(num_tokens)
    row_shape = recv.shape[1:]

    def kernel(tp_ref, dp_ref, sent_ref, weight_ref, rows_ref, *rest):
        shared_ref = None
        if shared is None:
            (out_ref,) = rest
        else:
            shared_ref, out_ref = rest
        base = pl.program_id(0) * tb
        for i in range(tb):
            t = base + i
            acc_m = jnp.zeros(row_shape, jnp.float32)
            acc_r = jnp.zeros(row_shape, jnp.float32)
            out = jnp.zeros(row_shape, jnp.float32)
            parts = [jnp.zeros(row_shape, jnp.float32) for _ in range(tp_size)]
            for j in range(topk):
                r = tp_ref[j, t]
                d = dp_ref[j, t]
                row = rows_ref[j, i].astype(jnp.float32)
                prod = jnp.where(sent_ref[j, t] != 0, weight_ref[j, t] * row, 0.0)
                if j:
                    new_r = r != tp_ref[j - 1, t]
                    new_m = jnp.logical_or(new_r, d != dp_ref[j - 1, t])
                    acc_r = jnp.where(new_r, 0.0, acc_r)
                    acc_m = jnp.where(new_m, 0.0, acc_m)
                acc_m = acc_m + prod
                group = r < tp_size
                if j + 1 < topk:
                    end_r = tp_ref[j + 1, t] != r
                    end_m = jnp.logical_or(end_r, dp_ref[j + 1, t] != d)
                else:
                    end_r = end_m = True
                part = acc_m.astype(dtype).astype(jnp.float32)
                acc_r = acc_r + jnp.where(jnp.logical_and(group, end_m), part, 0.0)
                part_r = acc_r.astype(dtype).astype(jnp.float32)
                if shared is None:
                    out = out + jnp.where(jnp.logical_and(group, end_r), part_r, 0.0)
                else:
                    done = jnp.logical_and(group, end_r)
                    parts = [
                        jnp.where(jnp.logical_and(done, r == rr), part_r, parts[rr])
                        for rr in range(tp_size)
                    ]
            if shared_ref is not None:
                for rr in range(tp_size):
                    x = shared_ref[rr, i].astype(jnp.float32) + parts[rr]
                    out = out + x.astype(dtype).astype(jnp.float32)
            out_ref[i] = out.astype(out_ref.dtype)

    in_specs = [
        pl.BlockSpec(
            (topk, tb) + row_shape, lambda g, *_: (0, g) + (0,) * len(row_shape)
        )
    ]
    operands = [rows]
    if shared is not None:
        in_specs.append(
            pl.BlockSpec(
                (tp_size, tb) + row_shape, lambda g, *_: (0, g) + (0,) * len(row_shape)
            )
        )
        operands.append(shared.reshape(tp_size, num_tokens, *row_shape))
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_tokens,) + row_shape, dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(num_tokens // tb,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(
                (tb,) + row_shape, lambda g, *_: (g,) + (0,) * len(row_shape)
            ),
        ),
        name="ragged_a2a_owner_sum" + ("_shared" if shared is not None else ""),
    )(tp_s, dp_s, sent_s, weight_s, *operands)
    return out.reshape(num_tokens, -1)


def owner_tp_sum(partial: jax.Array, *, tp_size: int) -> jax.Array:
    """Sum of the TP ranks' [S, H] partials with the owner ranks' arithmetic:
    each owner rank gets every rank's rows of its owned tokens (one
    all-to-all), adds them in f32 in rank order with one rounding, and the
    engine all-gathers the owned rows. Moves the same bytes as an all-reduce."""
    num_tokens = partial.shape[0]
    blk = -(-num_tokens // tp_size)
    rows = jnp.pad(partial, ((0, blk * tp_size - num_tokens), (0, 0)))
    rows = jax.lax.all_to_all(
        rows.reshape(tp_size, blk, -1), TP_AXIS, 0, 0, tiled=False
    )
    acc = rows[0].astype(jnp.float32)
    for r in range(1, tp_size):
        acc = acc + rows[r].astype(jnp.float32)
    owned = acc.astype(partial.dtype)
    return jax.lax.all_gather(owned, TP_AXIS, tiled=True)[:num_tokens]


def _owner_combine(
    received,
    recv_weights,
    recv_ids,
    sender_member,
    sender_token,
    dp_pos,
    fits,
    *,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_ids,
    topk_weights,
    experts_start,
    member_index,
    dp_size,
    tp_size,
    topk,
    activation,
    rhs_quant_dtype,
    skip_padded_tokens,
    max_local_routes,
    collective_id,
    shared=None,
    plan=None,
    row_gather=False,
    src_rows=None,
):
    """The expert kernel sends each output row to its token's OWNER rank: TP rank
    ``t // ceil(S / tp)`` of the token's engine owns token ``t``. The owner sums
    all of the token's routes (every TP rank's experts) and the engine's TP
    ranks all-gather the owned rows: the result is already summed over TP.
    ``plan``: the routes from `route_plan` with `owner_dests`; ``row_gather``:
    ``received`` holds whole-tile rows and the routes' rows are copied out of
    it, all below ``src_rows`` [1]."""
    num_tokens = x.shape[0]
    local = w1.shape[0]
    mesh = dict(dp_size=dp_size, tp_size=tp_size, local_num_experts=local)
    blk = -(-num_tokens // tp_size)
    ep_rank = experts_start[0] // local
    my_tp = ep_rank % tp_size
    me = dp_pos * tp_size + my_tp
    pad = blk * tp_size - num_tokens
    ids = jnp.pad(topk_ids, ((0, pad), (0, 0)), constant_values=-1)
    weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
    owned_ids = jax.lax.dynamic_slice_in_dim(ids, my_tp * blk, blk)
    owned_weights = jax.lax.dynamic_slice_in_dim(weights, my_tp * blk, blk)
    valid, _, _, _ = owner_route_key(owned_ids, **mesh)
    w = owned_weights.astype(x.dtype)
    w = jnp.where(jnp.isfinite(w), w, 0)
    if skip_padded_tokens:
        valid = valid & (w != 0)
    remote = valid & (owned_ids // local != ep_rank)
    expected = jnp.where(fits, remote.sum(dtype=jnp.int32), 0)
    extra = None
    if shared is not None:
        # This rank's shared-expert partial of every engine token goes to the
        # token's owner (row my_tp * blk + t % blk there), during the experts.
        t = jnp.arange(blk * tp_size, dtype=jnp.int32)
        rows = jnp.pad(shared, ((0, pad), (0, 0)))
        rows = rows.reshape(blk * tp_size, 16, -1)
        extra = (
            rows,
            dp_pos * tp_size + t // blk,
            my_tp * blk + t % blk,
            # An overflowing step sends and expects nothing (the caller runs
            # the full-size exchange).
            jnp.where(fits, jnp.array([(tp_size - 1) * blk, blk * tp_size]), 0).astype(
                jnp.int32
            ),
        )
    if plan is not None:
        recv = fused_moe_scatter_planned(
            received,
            w1,
            w2,
            w1_scale,
            w2_scale,
            *plan,
            jnp.asarray(me, jnp.int32)[None],
            expected[None],
            extra,
            src_rows=src_rows,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            num_recv_rows=blk * topk,
            num_extra_recv_rows=blk * tp_size if shared is not None else 0,
            axis=(DP_AXIS, TP_AXIS),
            num_members=(dp_size, tp_size),
            collective_id=collective_id,
            row_gather=row_gather,
        )
    else:
        dest = sender_member * tp_size + sender_token // blk
        row_dest = owner_order(recv_ids, **mesh) * blk + (sender_token % blk)[:, None]
        recv = fused_moe_scatter(
            received,
            w1,
            w2,
            w1_scale,
            w2_scale,
            recv_weights,
            recv_ids,
            experts_start[0],
            dest.astype(jnp.int32),
            row_dest.astype(jnp.int32),
            jnp.asarray(me, jnp.int32)[None],
            expected[None],
            extra,
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
            max_local_routes=max_local_routes,
            num_recv_rows=blk * topk,
            num_extra_recv_rows=blk * tp_size if shared is not None else 0,
            axis=(DP_AXIS, TP_AXIS),
            num_members=(dp_size, tp_size),
            collective_id=collective_id,
        )
    shared_recv = None
    if shared is not None:
        recv, shared_recv = recv
    owned = owner_sum_kernel(
        recv,
        owned_ids,
        owned_weights,
        skip_padded_tokens=skip_padded_tokens,
        dtype=x.dtype,
        shared=shared_recv,
        **mesh,
    )
    return jax.lax.all_gather(owned, TP_AXIS, tiled=True)[:num_tokens]


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
    route_factor: float = 0.0,
    capacity_row_tile: int = _ROW_TILE,
    row_gather: bool = False,
    row_scatter: bool = False,
    streamed_combine: bool = False,
    owner_combine: bool = False,
    sorted_plan: bool = False,
    routing_dma: bool = False,
    shared: jax.Array | None = None,
) -> jax.Array:
    """Per-rank body (inside shard_map over the (dp, tp) mesh).

    ``x`` [S, H] and the routing [S, K] are this engine's tokens;
    ``experts_start`` [1] is this rank's first global expert id. Returns this
    rank's [S, H] sum over its DP group, before the TP all-reduce.

    With ``capacity_factor`` > 0 and fewer capacity rows than DP-group tokens,
    the exchange and the local MoE are sized by ``exchange_capacity`` (the
    routes by ``route_factor`` when it is > 0); a step
    whose routing exceeds those sizes on any rank runs the full-size exchange
    instead. Both give the same result.

    With ``streamed_combine``, an exchange whose local MoE runs the local-routes
    one-hot path with the fused expert kernel has the kernel DMA each output row
    to the rank that sent its token while the later experts compute
    (see ``_streamed_combine``), instead of combining here and returning partials
    with a second exchange. The sum differs from the returned-partials path only
    in the order of a member's f32 accumulation of three or more routes of one
    token (``_sender_sum``).

    With ``owner_combine`` the expert outputs go to their token's owner rank
    instead (``_owner_combine``) and the result is ALREADY summed over the TP
    ranks: paths without the fused kernel sum here with the same arithmetic
    (``owner_tp_sum``).
    ``shared`` [S, H] (owner mode only): this TP rank's shared-expert partial,
    added before the TP sum as the model would, so the result is final.

    ``sorted_plan`` (with the streamed or owner combine): the expert side orders
    its local routes with one sort over the DP group's gathered routing
    (`route_plan`), with each route's destination carried along, instead of
    reading the received rows' routes, sorting them and looking up their
    destinations. The kernel gets the same rows in the same order.
    ``row_gather`` (with ``sorted_plan``): the expert side copies its routes'
    rows out of the receive buffer with one row DMA each instead of the
    one-hot permute (same rows), and the receive buffers are not zeroed.
    ``row_scatter`` (with ``sorted_plan``): the send buffer is written by a
    kernel that relayouts the tokens once and copies their rows in VMEM,
    instead of an XLA gather of the send rows and their relayout (same rows
    below the send sizes).
    ``routing_dma``: the DP group's packed routing is all-gathered by one
    kernel of remote DMAs (`dma_all_gather`) instead of XLA's ring
    all-gather, whose steps each wait on the previous one (same words).
    """
    assert shared is None or owner_combine
    num_tokens = x.shape[0]
    local_num_experts = w1.shape[0]
    slots = dp_size * num_tokens
    gathered_ids, read_routes, packed_routes = _gather_routing(
        topk_ids, topk_weights, dma_members=dp_size if routing_dma else 0
    )
    ep_rank = experts_start[0] // local_num_experts
    member_index = functools.partial(
        _member_index,
        ep_rank=ep_rank,
        dp_size=dp_size,
        tp_size=tp_size,
        local_num_experts=local_num_experts,
    )

    def streamed(recv_rows, max_local_routes, collective_id):
        """None when the exchange's local MoE does not take the fused kernel."""
        if not (streamed_combine or owner_combine) or not scatter_moe_supported(
            jax.ShapeDtypeStruct((recv_rows, x.shape[1]), x.dtype),
            w1,
            w2,
            w1_scale,
            w2_scale,
            num_routes=max_local_routes,
            max_local_routes=max_local_routes,
            activation=activation,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            rhs_quant_dtype=rhs_quant_dtype,
        ):
            return None
        if owner_combine:
            return functools.partial(
                _owner_combine,
                x=x,
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                experts_start=experts_start,
                member_index=member_index,
                dp_size=dp_size,
                tp_size=tp_size,
                topk=topk,
                activation=activation,
                rhs_quant_dtype=rhs_quant_dtype,
                skip_padded_tokens=skip_padded_tokens,
                max_local_routes=max_local_routes,
                collective_id=collective_id,
                shared=shared,
                row_gather=row_gather,
            )
        return functools.partial(
            _streamed_combine,
            x=x,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts_start=experts_start,
            member_index=member_index,
            dp_size=dp_size,
            topk=topk,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
            max_local_routes=max_local_routes,
            collective_id=collective_id,
            row_gather=row_gather,
        )

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

    def finish(out, reduced):
        # Owner mode returns TP-summed rows; a path without the fused kernel
        # returns this rank's partial, summed here as the owner ranks would.
        if owner_combine and not reduced:
            if shared is not None:
                out = out + shared
            return owner_tp_sum(out, tp_size=tp_size)
        return out

    full_streamed = streamed(slots, slots * topk, 1)
    plan = None
    if sorted_plan and packed_routes is not None:
        if owner_combine:
            dests = functools.partial(
                owner_dests,
                dp_size=dp_size,
                tp_size=tp_size,
                local_num_experts=local_num_experts,
            )
        else:
            dests = functools.partial(
                streamed_dests, member_index=member_index, dp_size=dp_size
            )
        gathered_weights = None
        if skip_padded_tokens:
            gathered_weights = _unpack_routes(packed_routes)[1].reshape(
                gathered_ids.shape
            )
        plan = (dests, gathered_weights)

    def full():
        out = _full_exchange(
            x,
            w1,
            topk_ids,
            gathered_ids,
            read_routes,
            experts_start,
            tp_size=tp_size,
            dp_size=dp_size,
            moe=moe(slots * topk),
            streamed=full_streamed,
            plan=plan,
            row_gather=row_gather,
            row_scatter=row_scatter,
        )
        return finish(out, full_streamed is not None)

    rows, routes = exchange_capacity(
        dp_size=dp_size,
        tp_size=tp_size,
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=local_num_experts,
        factor=capacity_factor,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        use_sparse_core=use_sparse_core,
        route_factor=route_factor or None,
        row_tile=capacity_row_tile,
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
    capacity_streamed = streamed(rows, routes, 2)
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
        streamed=capacity_streamed,
        plan=plan,
        routes=routes,
        row_gather=row_gather,
        row_scatter=row_scatter,
    )
    out = finish(out, capacity_streamed is not None)
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
    route_factor: float = 0.0,
    capacity_row_tile: int = _ROW_TILE,
    row_gather: bool = False,
    row_scatter: bool = False,
    streamed_combine: bool = False,
    owner_combine: bool = False,
    sorted_plan: bool = False,
    routing_dma: bool = False,
    shared: jax.Array | None = None,
) -> jax.Array:
    """Global-view entry: every operand is the ranks' local arrays stacked on
    two leading axes [dp, tp, ...], and so is the result [dp, tp, S, H]
    (``shared``, the shared-expert partials, likewise). Unscaled (bf16)
    experts pass ``None`` for both scales."""
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
        route_factor=route_factor,
        capacity_row_tile=capacity_row_tile,
        row_gather=row_gather,
        row_scatter=row_scatter,
        streamed_combine=streamed_combine,
        owner_combine=owner_combine,
        sorted_plan=sorted_plan,
        routing_dma=routing_dma,
    )

    scaled = w1_scale is not None
    assert scaled == (w2_scale is not None)
    scales = (w1_scale, w2_scale) if scaled else ()
    operands = (x, w1, w2, *scales, topk_weights, topk_ids, experts_start)
    if shared is not None:
        operands += (shared,)

    def body(*args):
        args = [a[0, 0] for a in args]
        if not scaled:
            args[3:3] = [None, None]
        if shared is not None:
            return local(*args[:8], shared=args[8])[None, None]
        return local(*args)[None, None]

    mapped = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(spec,) * len(operands),
        out_specs=spec,
        check_vma=False,
    )
    return mapped(*operands)
