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
"""Torch bridge for tokamax's routing MoE kernel (USE_ROUTING_MOE=1).

tokamax's routing_moe (tokamax/_src/ops/experimental/routing_moe) all-gathers
the tokens over the EP axis; each device then gathers its experts' token rows
by index, runs gate/up and down, and sends each result row straight to the
device that owns the token, which sums the token's top-k rows weighted by the
routing weights. No device builds a full [tokens, hidden] partial sum and there
is no reduce-scatter.

Like the fused EP path (fused_moe_ep.py), the whole layer is one op over the EP
mesh that takes this rank's own tokens and returns this rank's own tokens, built
at weight load. Ranks of one TP group hold the same token rows; each passes the
kernel its own share of them (TokenReplicaLayout), so every token is routed
once, and the group all-gathers the results. The experts are picked here as
moe_router.select_experts picks them: score, top-k, renormalize.

ROUTING_MOE_REMOTE_SCATTER=0 selects the kernel's local mode instead of the remote
scatter above: each device sums its experts' rows into a partial result for
every token, and a reduce-scatter combines the partial results.

Under MoE tensor parallelism (no --enable-expert-parallel) the layer runs the
kernel's TP mode instead: every rank holds all tokens and every expert, sliced
along the intermediate dimension, so each rank computes a partial result for
every token over its slice and the op all-reduces the partials over the TP
group. Each rank's w13 is already [gate | up] over its own slice, which is the
per-shard layout the kernel expects, so the weights go in as they are.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    ep_token_replica_groups,
    get_ep_group,
)
from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ROUTING_MOE_OP_NAME = "pallas::routing_moe"

# The attribute prebuild's op is recorded under, on the quant method that owns
# the layer (one per MoE layer), as fused_moe_ep.FUSED_MOE_EP_OP_ATTR.
ROUTING_MOE_OP_ATTR = "_routing_moe_op"

# One op per distinct closure, shared by the layers that agree on it, as in
# fused_moe_ep._OPS.
_OPS: dict[tuple, Any] = {}

# This rank's index in the EP mesh, as a device tensor. The kernel takes it as
# data: `lax.axis_index` lowers to `partition-id`, which torch_tpu's jax.export
# bridge rejects (see fused_moe_ep._RANK_BUFFER).
_RANK_BUFFER: torch.Tensor | None = None


def routing_moe_supported(owner: Any) -> bool:
    """Whether prebuild_routing_moe armed this layer's quant method."""
    return getattr(owner, ROUTING_MOE_OP_ATTR, None) is not None


def unsupported_reason(layer) -> str | None:
    """Why this integration can't run the layer, or None if it can.

    Only what the integration itself doesn't handle; the kernel raises for what
    it doesn't support. Whether the experts are expert- or tensor-parallel is
    prebuild's to sort out.
    """
    if layer.scoring_func not in ("softmax", "sigmoid"):
        return f"routing here doesn't score with {layer.scoring_func}"
    if layer.e_score_correction_bias is not None:
        return "the score correction bias isn't wired up here"
    if float(layer.routed_scaling_factor) != 1.0:
        return "routed_scaling_factor isn't wired up here"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing isn't wired up here"
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "expert biases aren't passed to the kernel here"
    return None


def _route(
    gating: jax.Array, topk: int, renormalize: bool, scoring_fn: str
) -> tuple[jax.Array, jax.Array]:
    """moe_router.select_experts for the scoring functions admitted above."""
    gating = gating.astype(jnp.float32)
    if scoring_fn == "softmax":
        scores = jax.nn.softmax(gating, axis=-1)
    else:
        scores = jax.nn.sigmoid(gating)
    weights, ids = lax.top_k(scores, topk)
    if renormalize:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return weights, ids.astype(jnp.int32)


def _relabel_table(order: tuple[int, ...], local_experts: int) -> list[int]:
    """Kernel expert id of each global expert id.

    Mesh index m holds the experts of EP rank order[m], and the kernel reads
    expert id e as mesh index e // local_experts.
    """
    index = {rank: m for m, rank in enumerate(order)}
    return [
        index[g // local_experts] * local_experts + g % local_experts
        for g in range(len(order) * local_experts)
    ]


def _build_op(
    mesh,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    activation: str,
    mesh_expert_order: tuple[int, ...] | None,
    replica_groups: tuple[tuple[int, ...], ...],
    remote_scatter: bool,
):
    key = (
        topk,
        renormalize,
        scoring_fn,
        activation,
        mesh_expert_order,
        replica_groups,
        remote_scatter,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached
    # Imported here, where USE_ROUTING_MOE has asked for the kernel: this module
    # is imported for every model, and only this path needs tokamax.
    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.layer import modular as tk_modular

    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.layer import plan as tk_plan

    # A distinct torch op name per closure: two ops registered under one name
    # would share a dispatch entry (see fused_moe_ep._build_op).
    name = ROUTING_MOE_OP_NAME if not _OPS else f"{ROUTING_MOE_OP_NAME}_{len(_OPS)}"
    ep = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    # Every operand is annotated jax.Array: torch_tpu infers static argnums
    # from the signature and refuses an unannotated one.
    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        layout = TokenReplicaLayout.create(
            ep=ep, input_rows=x.shape[0] // ep, groups=replica_groups
        )
        relabel = (
            None
            if mesh_expert_order is None
            else jnp.asarray(
                _relabel_table(mesh_expert_order, w1.shape[0] // ep), jnp.int32
            )
        )

        def split_and_route(x_l, gating_l, rank_l):
            me = rank_l[0, 0]
            weights, ids = _route(
                layout.split(gating_l, me), topk, renormalize, scoring_fn
            )
            if relabel is not None:
                ids = relabel[ids]
            return layout.split(x_l, me), weights, ids

        x, weights, ids = jax.shard_map(
            split_and_route,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=(spec, spec, spec),
            check_vma=False,
        )(x, gating, rank)
        out = tk_modular.routing_moe(
            x,
            weights,
            ids,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,
            None,
            mesh=mesh,
            activation=activation,
            tp_axis=None,
            ep_axis=EP_AXIS_NAME,
            use_sp=True,
            moe_plan=tk_plan.RoutingMoEPlan(remote_scatter=remote_scatter),
            ep_shard_idx=rank.reshape(-1),
        )
        return jax.shard_map(
            lambda out_l: layout.restore(out_l, EP_AXIS_NAME),
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
            check_vma=False,
        )(out)

    op = jax_op(
        name,
        moe,
        mesh=mesh,
        # x, w1, w2, w1_scale, w2_scale, gating, rank; the output is sharded on
        # axis 0 like x.
        input_partition_specs=(spec,) * 7,
        output_partition_specs=spec,
    )

    # The op returns the rows it was given, so the output shape is known
    # without exporting (as fused_moe_ep's fake).
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def _tp_moe_fn(
    mesh,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    activation: str,
    kernel,
):
    """The TP op's body: `kernel` is tokamax's `modular.routing_moe`.

    jax_op hands every operand over sharded on axis 0, one rank's tensor per
    mesh index: x and the gating [ranks * tokens, ...] (every rank's copy of
    the same tokens), the weights and scales [ranks * E, ...]. The kernel wants
    replicated tokens and the weights split along the intermediate dimension.
    Every rank already holds exactly its own slice, so the conversion is an
    identity shard_map that only relabels the sharding, without moving data.
    """
    axis = mesh.axis_names[0]
    rows = PartitionSpec(axis)
    replicated = PartitionSpec()

    def relabel(a: jax.Array, out_spec: PartitionSpec) -> jax.Array:
        return jax.shard_map(
            lambda v: v,
            mesh=mesh,
            in_specs=rows,
            out_specs=out_spec,
            check_vma=False,
        )(a)

    def route(x_l, gating_l):
        weights, ids = _route(gating_l, topk, renormalize, scoring_fn)
        return x_l, weights, ids

    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        del rank  # TP partials are summed, so the mesh order doesn't matter.
        # Every rank holds the same tokens; routing them on each rank gives the
        # same experts everywhere, so the result is replicated.
        x, weights, ids = jax.shard_map(
            route,
            mesh=mesh,
            in_specs=(rows, rows),
            out_specs=(replicated,) * 3,
            check_vma=False,
        )(x, gating)
        # The kernel's TP specs (layer/modular.py): w1 and its scale on the
        # output (intermediate) dim, w2 on its contracting (intermediate) dim.
        # Each rank quantized its own w2 slice, so w2's scale blocks are split
        # along the intermediate dim too; concatenated over ranks there is more
        # than one block, which is what makes the kernel split them as well.
        w1 = relabel(w1, PartitionSpec(None, None, axis))
        w2 = relabel(w2, PartitionSpec(None, axis, None))
        w1_scale = relabel(w1_scale, PartitionSpec(None, None, None, axis))
        w2_scale = relabel(w2_scale, PartitionSpec(None, axis, None, None))
        out = kernel(
            x,
            weights,
            ids,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,
            None,
            mesh=mesh,
            activation=activation,
            tp_axis=axis,
            ep_axis=None,
            use_sp=False,
            reduce_output=True,
        )
        # The kernel all-reduces over the TP axis, so every rank gets the
        # whole result for its own (identical) tokens.
        return jax.shard_map(
            lambda v: v,
            mesh=mesh,
            in_specs=replicated,
            out_specs=rows,
            check_vma=False,
        )(out)

    return moe


def _build_tp_op(
    mesh,
    topk: int,
    renormalize: bool,
    scoring_fn: str,
    activation: str,
):
    key = ("tp", topk, renormalize, scoring_fn, activation)
    cached = _OPS.get(key)
    if cached is not None:
        return cached
    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.routing_moe.layer import modular as tk_modular

    name = ROUTING_MOE_OP_NAME if not _OPS else f"{ROUTING_MOE_OP_NAME}_{len(_OPS)}"
    spec = PartitionSpec(mesh.axis_names[0])
    op = jax_op(
        name,
        _tp_moe_fn(
            mesh, topk, renormalize, scoring_fn, activation, tk_modular.routing_moe
        ),
        mesh=mesh,
        input_partition_specs=(spec,) * 7,
        output_partition_specs=spec,
    )

    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def _tp_group_reason() -> str | None:
    """Why the TP group can't host the op's mesh, or None if it can.

    The op's mesh is `build_ep_mesh`'s, as for moe_tp_decode: vLLM builds the
    EP group even with EP off, and with dp 1 and pcp 1 it is the TP group.
    """
    from vllm.distributed import parallel_state

    tp = parallel_state.get_tp_group()
    if int(tp.world_size) < 2:
        return "the TP group has one device"
    ep = get_ep_group()
    tp_ranks = tuple(int(r) for r in tp.ranks)
    ep_ranks = None if ep is None else tuple(int(r) for r in ep.ranks)
    if tp_ranks != ep_ranks:
        return f"TP ranks {tp_ranks} != EP ranks {ep_ranks} (needs dp 1, pcp 1)"
    return None


def _refusal(reason: str) -> ValueError:
    return ValueError(
        f"routing_moe can't run this MoE layer: {reason}. "
        "Unset USE_ROUTING_MOE to use another MoE kernel."
    )


def _prebuild_tp(layer, parallel, topk: int, renormalize: bool, activation: str) -> Any:
    """prebuild_routing_moe for tensor-parallel experts."""
    global _RANK_BUFFER
    if getattr(parallel, "is_sequence_parallel", False):
        raise _refusal("sequence-parallel MoE is only wired up with EP")
    reason = _tp_group_reason()
    if reason is not None:
        raise _refusal(reason)
    mesh = build_ep_mesh()
    if mesh is None:
        raise _refusal("there is no TP group mesh")
    if layer.global_num_experts != layer.w13_weight.shape[0]:
        raise _refusal(
            f"TP expects all {layer.global_num_experts} experts on every rank, "
            f"got {layer.w13_weight.shape[0]}"
        )
    # Unused by the TP body, but routing_moe passes the same seven operands in
    # both modes.
    _RANK_BUFFER = torch.tensor(
        [[ep_mesh_index()]], dtype=torch.int32, device=layer.w13_weight.device
    )
    tp = mesh.shape[mesh.axis_names[0]]
    if envs.ROUTING_MOE_REMOTE_SCATTER:
        logger.info_once(
            "[MoE]: ROUTING_MOE_REMOTE_SCATTER needs EP; TP routing_moe "
            "all-reduces instead."
        )
    logger.info_once(
        "[MoE]: routing_moe over %d TP devices, all %d experts each, "
        "intermediate %d per device; all-reduce.",
        tp,
        layer.global_num_experts,
        layer.w2_weight.shape[1],
    )
    return _build_tp_op(mesh, topk, renormalize, layer.scoring_func, activation)


def prebuild_routing_moe(
    layer, topk: int, renormalize: bool, activation: str
) -> Any | None:
    """Build the routing MoE op for one layer, or None if USE_ROUTING_MOE is
    off. Raises for a layer it can't run rather than falling back to another
    backend.

    Called from process_weights_after_loading: resolving the mesh gathers from
    every rank, which a compiled forward can't do.
    """
    if not envs.USE_ROUTING_MOE:
        return None
    if envs.USE_MOE_FUSED_EP_KERNEL:
        raise ValueError("USE_ROUTING_MOE and USE_MOE_FUSED_EP_KERNEL are exclusive")
    reason = unsupported_reason(layer)
    if reason is not None:
        raise _refusal(reason)
    global _RANK_BUFFER
    parallel = layer.moe_config.moe_parallel_config
    if not parallel.use_ep:
        return _prebuild_tp(layer, parallel, topk, renormalize, activation)
    mesh = build_ep_mesh()
    order = ep_rank_order()
    if mesh is None or order is None:
        raise _refusal("there is no EP group")
    ep = mesh.shape[EP_AXIS_NAME]
    local_experts = layer.w13_weight.shape[0]
    if layer.global_num_experts != local_experts * ep:
        raise _refusal(
            f"{layer.global_num_experts} experts don't split evenly over "
            f"{ep} EP ranks of {local_experts}"
        )

    _RANK_BUFFER = torch.tensor(
        [[ep_mesh_index()]], dtype=torch.int32, device=layer.w13_weight.device
    )
    order = tuple(order)
    replica_groups = ep_token_replica_groups(
        is_sequence_parallel=layer.moe_config.moe_parallel_config.is_sequence_parallel
    )
    logger.info_once(
        "[MoE]: routing_moe over %d devices, %d experts each, token replica "
        "groups %s; %s scatter.",
        ep,
        local_experts,
        replica_groups,
        "remote" if envs.ROUTING_MOE_REMOTE_SCATTER else "local",
    )
    return _build_op(
        mesh,
        topk,
        renormalize,
        layer.scoring_func,
        activation,
        None if order == tuple(range(ep)) else order,
        replica_groups,
        envs.ROUTING_MOE_REMOTE_SCATTER,
    )


def routing_moe(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Run one MoE layer through the routing MoE op prebuild recorded on
    `owner`, the layer's quant method.

    hidden_states [tokens, hidden] and router_logits [tokens, experts] are this
    rank's; w1 [E, H, 2I] (gate | up), w2 [E, I, H] and their scales
    [E, blocks, 1, N] are this rank's experts. Returns this rank's
    [tokens, hidden].
    """
    op = getattr(owner, ROUTING_MOE_OP_ATTR)
    return op(hidden_states, w1, w2, w1_scale, w2_scale, router_logits, _RANK_BUFFER)
