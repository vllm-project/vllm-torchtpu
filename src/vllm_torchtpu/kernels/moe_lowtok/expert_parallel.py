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
"""The few-token kernel with its expert-parallel exchange, for a fused EP op.

A fused expert-parallel op takes each rank's own rows and router logits and
returns each rank's own output rows. For a small step this module does the
same with plain collectives around the few-token kernel: all-gather the rows
and logits over the EP axis, route (softmax top-k), run this shard's experts
on every row, and reduce-scatter the partial sums back to the rows' owners.
It is the exchange vLLM's naive dispatch and combine perform around the
routed GMM path, so a step either kernel takes computes the same layer.

The choice between this and the fused kernel is made from the static row
count inside the op, so the model graph is identical for every token bucket
and the per-bucket compile handover sees the same operands on both sides.
"""

import jax
import jax.numpy as jnp
from jax import lax

try:
    from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs
except ImportError:  # pragma: no cover - optional collective kernel port
    sgs = None

from vllm_torchtpu.kernels.moe_lowtok.kernel import (
    can_use_moe_lowtok,
    moe_lowtok,
    moe_lowtok_scores,
)


def _local_shape(array, ep):
    return jax.ShapeDtypeStruct((array.shape[0] // ep, *array.shape[1:]), array.dtype)


def _combine_partials(
    partial,
    *,
    axis,
    ep: int,
    rank_l,
    out_dtype,
    small_reduce_scatter_max_rows: int = 0,
    interpret: bool = False,
):
    """Combine per-shard partial sums [ep * local_rows, hidden] into
    [local_rows, hidden]."""
    local_rows = partial.shape[0] // ep
    if (
        not interpret
        and sgs is not None
        and small_reduce_scatter_max_rows > 0
        and rank_l is not None
        and partial.dtype == jnp.bfloat16
        and partial.shape[1] % sgs.LANE == 0
        and local_rows <= small_reduce_scatter_max_rows
    ):
        return sgs.small_reduce_scatter(
            partial,
            num_cores=ep,
            axis=axis,
            rank=rank_l.reshape(1),
        ).astype(out_dtype)
    # Sum the shards' partials in float32 and round once.
    out = lax.psum_scatter(
        partial.astype(jnp.float32), axis, scatter_dimension=0, tiled=True
    )
    return out.astype(out_dtype)


def moe_lowtok_ep_serves(
    x, w1, w2, w1_scale, w2_scale, *, ep, max_tokens, activation, token_replica_groups
):
    """Whether the few-token path takes this call: at most `max_tokens` rows
    after the EP all-gather, one shard per token group (no replicated rows),
    and a layer the kernel supports. Arguments are the op's global operands,
    sharded on axis 0 over `ep` shards."""
    rows = x.shape[0]
    if not 0 < rows <= max_tokens or rows % ep:
        return False
    if token_replica_groups is not None and len(token_replica_groups[0]) != 1:
        return False
    return can_use_moe_lowtok(
        _local_shape(x, ep),
        _local_shape(w1, ep),
        _local_shape(w2, ep),
        _local_shape(w1_scale, ep),
        _local_shape(w2_scale, ep),
        None,
        None,
        activation=activation,
    )


def route_softmax_topk(gating, topk, renormalize):
    """Softmax over the experts, the top-k, and optionally renormalized
    weights: [rows, topk] f32 weights and int32 expert ids."""
    scores = jax.nn.softmax(gating.astype(jnp.float32), axis=-1)
    weights, ids = lax.top_k(scores, topk)
    if renormalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return weights, ids.astype(jnp.int32)


def moe_lowtok_ep(
    x,
    gating,
    rank,
    w1,
    w2,
    w1_scale,
    w2_scale,
    *,
    mesh,
    topk,
    renormalize,
    mesh_ep_ranks=None,
    small_reduce_scatter_max_rows: int = 0,
    interpret=False,
):
    """One MoE layer for a small step, sharded over the mesh's one axis.

    x [rows, hidden] and gating [rows, experts] hold every rank's own rows,
    in mesh order; rank [ep, 1] int32 is each shard's mesh index; w1 [experts,
    hidden, 2*inter], w2 [experts, inter, hidden] fp8 with per-channel scales
    w1_scale [experts, 2*inter] and w2_scale [experts, hidden], in contiguous
    expert blocks. mesh_ep_ranks names the EP rank at each mesh index (None
    for the identity); the shard at mesh index i holds EP rank
    mesh_ep_ranks[i]'s block. Returns [rows, hidden], sharded like x.
    """
    (axis,) = mesh.axis_names
    ep = mesh.shape[axis]
    local_experts = w1.shape[0] // ep
    order = jnp.asarray(
        tuple(range(ep)) if mesh_ep_ranks is None else mesh_ep_ranks, jnp.int32
    )

    def local_fn(x_l, gate_l, rank_l, w1_l, w2_l, w1s_l, w2s_l):
        rows = lax.all_gather(x_l, axis, axis=0, tiled=True)
        logits = lax.all_gather(gate_l, axis, axis=0, tiled=True)
        padded_row = jnp.any(jnp.isnan(logits), axis=-1, keepdims=True)
        safe_logits = jnp.where(padded_row, 0.0, logits)
        weights, ids = route_softmax_topk(safe_logits, topk, renormalize)
        weights = jnp.where(padded_row, 0.0, weights)
        ids = jnp.where(padded_row, -1, ids)
        local_ids = ids - order[rank_l[0, 0]] * local_experts
        valid = (local_ids >= 0) & (local_ids < local_experts)
        partial = moe_lowtok(
            rows,
            jnp.where(valid, local_ids, -1),
            jnp.where(valid, weights, 0.0),
            w1_l,
            w1s_l,
            w2_l,
            w2s_l,
            interpret=interpret,
        )
        return _combine_partials(
            partial,
            axis=axis,
            ep=ep,
            rank_l=rank_l,
            out_dtype=x_l.dtype,
            small_reduce_scatter_max_rows=small_reduce_scatter_max_rows,
            interpret=interpret,
        )

    spec = jax.sharding.PartitionSpec(axis)
    return jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(spec,) * 7,
        out_specs=spec,
        check_vma=False,
    )(x, gating, rank, w1, w2, w1_scale, w2_scale)


def zero_scores_for_padding(scores, padding):
    """Zero the router scores of padded rows (padding [rows] nonzero): the
    scores entry's router then selects nothing for them, as the routed path's
    zeroed routing weights do."""
    return jnp.where(padding.reshape(-1, 1) != 0, 0.0, scores)


def moe_lowtok_scores_ep(
    x,
    gating,
    padding,
    tables,
    prim,
    copy,
    hot1h,
    w1,
    w2,
    w1_scale,
    w2_scale,
    copy_weights=None,
    *,
    rank=None,
    mesh,
    topk,
    small_reduce_scatter_max_rows: int = 0,
    interpret=False,
):
    """`moe_lowtok_ep` with the router and the holder choice in the kernel
    (`moe_lowtok_scores`), for placements whose experts may have a copy.

    Beyond `moe_lowtok_ep`'s operands, all sharded on axis 0 over the mesh:
    padding [rows] int32, nonzero for each rank's padded rows; per shard, its
    packed tables (`pack_tables`), the primary and copy holder ranks [1,
    LANES], the hot one-hot [LANES, NC]; copy_weights (w1, w2, w1_scale,
    w2_scale) of the copy slots, or None without copies. The tables name
    ranks in EP rank order, so the mesh order needs no relabelling.
    """
    (axis,) = mesh.axis_names
    ep = mesh.shape[axis]
    has_copies = copy_weights is not None
    has_rank = rank is not None

    def local_fn(x_l, gate_l, pad_l, tab_l, prim_l, copy_l, hot_l, *rest_l):
        if has_rank:
            rank_l, *weights_l = rest_l
        else:
            rank_l, weights_l = None, rest_l
        w1_l, w2_l, w1s_l, w2s_l, *copies_l = weights_l
        padded = (pad_l.reshape(-1, 1) != 0) | jnp.any(
            jnp.isnan(gate_l), axis=-1, keepdims=True
        )
        safe_gate = jnp.where(padded, 0.0, gate_l.astype(jnp.float32))
        scores = jax.nn.softmax(safe_gate, axis=-1)
        scores = jnp.where(padded, 0.0, scores)
        rows = lax.all_gather(x_l, axis, axis=0, tiled=True)
        scores = lax.all_gather(scores, axis, axis=0, tiled=True)
        if has_copies:
            cw1, cw2, cw1s, cw2s = copies_l
            copies = (cw1, cw1s, cw2, cw2s)
        else:
            copies = None
        partial = moe_lowtok_scores(
            rows,
            scores,
            tab_l,
            prim_l,
            copy_l,
            hot_l,
            w1_l,
            w1s_l,
            w2_l,
            w2s_l,
            copies,
            topk=topk,
            ranks=ep,
            has_copies=has_copies,
            interpret=interpret,
        )
        return _combine_partials(
            partial,
            axis=axis,
            ep=ep,
            rank_l=rank_l,
            out_dtype=x_l.dtype,
            small_reduce_scatter_max_rows=small_reduce_scatter_max_rows,
            interpret=interpret,
        )

    operands = (
        x,
        gating,
        padding,
        tables,
        prim,
        copy,
        hot1h,
    )
    if has_rank:
        operands += (rank,)
    operands += (
        w1,
        w2,
        w1_scale,
        w2_scale,
    )
    if has_copies:
        operands += tuple(copy_weights)
    spec = jax.sharding.PartitionSpec(axis)
    return jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(spec,) * len(operands),
        out_specs=spec,
        check_vma=False,
    )(*operands)
