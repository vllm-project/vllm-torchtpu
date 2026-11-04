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
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func

DP_AXIS = "dp"
TP_AXIS = "tp"


def exchange_need(
    gathered_ids: jax.Array, ep_rank: jax.Array, *, tp_size: int, local_num_experts: int
) -> jax.Array:
    """Which DP-group members each token of the group must reach.

    ``gathered_ids`` [D, S, K] holds the global expert ids of every token of
    this DP group, by DP position. Linear placement puts expert ``e`` on EP
    rank ``e // local_num_experts``, and EP rank ``d * tp_size + r`` is DP
    position ``d`` at TP rank ``r``. Returns ``need`` [D, S, D]:
    ``need[s, t, d]`` is true when token ``t`` of member ``s`` has an expert on
    member ``d`` of this TP rank's DP group.
    """
    dp_size = gathered_ids.shape[0]
    owner = gathered_ids // local_num_experts
    routed = (gathered_ids >= 0) & (owner < dp_size * tp_size)
    here = routed & (owner % tp_size == ep_rank % tp_size)
    dest = jnp.where(here, owner // tp_size, dp_size)
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


def _gather_routing(topk_ids: jax.Array, topk_weights: jax.Array):
    """All-gather the DP group's routing in one collective when it packs:
    a bf16 weight and an expert id below 2**15 share one int32."""
    if topk_weights.dtype != jnp.bfloat16:
        return (
            jax.lax.all_gather(topk_ids, DP_AXIS),
            jax.lax.all_gather(topk_weights, DP_AXIS),
        )
    bits = jax.lax.bitcast_convert_type(topk_weights, jnp.uint16).astype(jnp.int32)
    packed = jax.lax.all_gather(jnp.left_shift(topk_ids, 16) | bits, DP_AXIS)
    weights = jax.lax.bitcast_convert_type(
        jnp.bitwise_and(packed, 0xFFFF).astype(jnp.uint16), jnp.bfloat16
    )
    return jnp.right_shift(packed, 16), weights


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
) -> jax.Array:
    """Per-rank body (inside shard_map over the (dp, tp) mesh).

    ``x`` [S, H] and the routing [S, K] are this engine's tokens;
    ``experts_start`` [1] is this rank's first global expert id. Returns this
    rank's [S, H] sum over its DP group, before the TP all-reduce.
    """
    num_tokens, hidden = x.shape
    local_num_experts = w1.shape[0]
    ep_rank = experts_start[0] // local_num_experts
    dp_pos = ep_rank // tp_size
    slots = dp_size * num_tokens

    gathered_ids, gathered_weights = _gather_routing(topk_ids, topk_weights)
    need = exchange_need(
        gathered_ids, ep_rank, tp_size=tp_size, local_num_experts=local_num_experts
    )
    counts = need.sum(axis=1, dtype=jnp.int32)  # [source, destination]
    send_sizes = jnp.take(counts, dp_pos, axis=0)
    recv_sizes = jnp.take(counts, dp_pos, axis=1)
    member = jnp.arange(dp_size, dtype=jnp.int32)
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    region = member * num_tokens
    landing = jnp.full((dp_size,), dp_pos * num_tokens, dtype=jnp.int32)

    # Outgoing: member d's region holds this rank's tokens that need d, in
    # token order.
    need_out = jnp.take(need, dp_pos, axis=0)  # [S, D]
    pos_out = jnp.cumsum(need_out, axis=0, dtype=jnp.int32) - 1
    slot_out = jnp.where(need_out, region[None, :] + pos_out, slots)
    send_index = (
        jnp.zeros((slots,), jnp.int32)
        .at[slot_out.reshape(-1)]
        .set(jnp.broadcast_to(token[:, None], need_out.shape).reshape(-1), mode="drop")
    )

    # Incoming: source s's region holds its tokens that need this rank, in
    # its token order; rows past its count stay empty.
    need_in = jnp.take(need, dp_pos, axis=2)  # [D, S]
    pos_in = jnp.cumsum(need_in, axis=1, dtype=jnp.int32) - 1
    slot_in = jnp.where(need_in, region[:, None] + pos_in, slots)
    source_row = (
        jnp.zeros((slots,), jnp.int32)
        .at[slot_in.reshape(-1)]
        .set((region[:, None] + token[None, :]).reshape(-1), mode="drop")
    )
    filled = (jnp.arange(slots, dtype=jnp.int32) % num_tokens) < jnp.repeat(
        recv_sizes, num_tokens, total_repeat_length=slots
    )
    recv_ids = jnp.where(
        filled[:, None], gathered_ids.reshape(slots, topk)[source_row], -1
    )
    recv_weights = jnp.where(
        filled[:, None],
        gathered_weights.reshape(slots, topk)[source_row],
        jnp.zeros((), gathered_weights.dtype),
    )

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

    partial = fused_moe_func(
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
    )

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
    back = jnp.concatenate([back, jnp.zeros((1, hidden), back.dtype)], axis=0)
    rows = back[slot_out.T]  # [D, S, H]; tokens that skipped d read the zero row
    return rows.astype(jnp.float32).sum(axis=0).astype(x.dtype)


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
    )

    def body(*args):
        return local(*(a[0, 0] for a in args))[None, None]

    mapped = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 8, out_specs=spec, check_vma=False
    )
    return mapped(x, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, experts_start)
