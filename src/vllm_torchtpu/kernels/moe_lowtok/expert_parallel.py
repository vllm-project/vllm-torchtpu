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

from vllm_torchtpu.kernels.moe_lowtok.kernel import can_use_moe_lowtok, moe_lowtok


def _local_shape(array, ep):
    return jax.ShapeDtypeStruct((array.shape[0] // ep, *array.shape[1:]), array.dtype)


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
        weights, ids = route_softmax_topk(logits, topk, renormalize)
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
        # Sum the shards' partials in float32 and round once.
        out = lax.psum_scatter(
            partial.astype(jnp.float32), axis, scatter_dimension=0, tiled=True
        )
        return out.astype(x_l.dtype)

    spec = jax.sharding.PartitionSpec(axis)
    return jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(spec,) * 7,
        out_specs=spec,
        check_vma=False,
    )(x, gating, rank, w1, w2, w1_scale, w2_scale)
