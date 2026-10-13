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
"""Torch bridge for the fused expert-parallel MoE kernel v3.

Selected with MOE_FUSED_EP_KERNEL_IMPL=v3. Like v2 (`fused_moe_ep.py`) the
whole layer -- top-k, dispatch, both expert matmuls, combine -- runs as one
SPMD program over the EP mesh, taking and returning this rank's own tokens.

v3 serves data parallelism with expert parallelism only (TP=1, PCP=1), with
per-channel fp8 weights and a plain softmax top-k router. A deployment outside
that raises at weight load; a layer outside it (shape, routing, weights) is
declined on its own and keeps the grouped-matmul path.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
)
from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.fused_moe_ep import (
    _pcp_size,
    _squeeze_channel_scale,
    _squeeze_channel_scale_checked,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The FFN tile height (the kernel's `tile_rows`). Smaller tiles are far slower
# with the kernel's DMA strategy, so it is not a knob.
TILE_ROWS = 128
WEIGHT_FORMAT = "fp8"

# One op per distinct closure, as in v2: a per-layer op would re-export the
# same program once per MoE layer, and two `CustomOpDef`s under one qualname
# silently share a dispatch entry. The name must not be v2's
# `pallas::fused_moe_ep`: both modules can be imported in one process.
FUSED_MOE_EP_V3_OP_NAME = "pallas::fused_moe_ep_v3"
_OPS: dict[tuple, Any] = {}

# The attribute the built op is recorded under, on the layer's quant method.
FUSED_MOE_EP_V3_OP_ATTR = "_fused_moe_ep_v3_op"


def _build_op(
    mesh,
    topk: int,
    renormalize: bool,
    rank_order: tuple[int, ...],
    *,
    hidden: int,
    inter: int,
    local_experts: int,
):
    """The op for this closure, built once per distinct closure and reused.

    `rank_order` is `ep_rank_order()`: the EP rank whose expert block sits at
    each mesh index. The kernel reads expert `i * local_experts + j` as mesh
    index i's j-th expert, so the router's columns are put in mesh order before
    its top-k, and each shard is handed its mesh index, not its EP rank.
    """
    key = (topk, renormalize, rank_order, hidden, inter, local_experts)
    if key in _OPS:
        return _OPS[key]
    if MESH_AXIS not in mesh.axis_names:
        raise ValueError(
            f"the v3 kernel splits the experts over the mesh axis named "
            f"{MESH_AXIS!r}; the adapter's mesh has {mesh.axis_names}"
        )
    config = Config.for_shape(
        hidden=hidden, inter=inter, experts_per_shard=local_experts, top_k=topk
    )
    logger.info_once(
        "Fused EP MoE v3 armed | hidden=%d inter=%d local_experts=%d topk=%d "
        "tile_rows=%d %s",
        hidden,
        inter,
        local_experts,
        topk,
        TILE_ROWS,
        config,
    )

    ep = len(rank_order)
    permuted = rank_order != tuple(range(ep))
    # Mesh index of each EP rank, to turn a shard's first expert into its index.
    mesh_index_of_rank = jnp.asarray(
        sorted(range(ep), key=rank_order.__getitem__), jnp.int32
    )

    # Every operand is annotated jax.Array: torch_tpu infers static argnums
    # from the signature and refuses an unannotated one.
    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        experts_start: jax.Array,
    ) -> jax.Array:
        gating = gating.astype(jnp.float32)
        if permuted:
            tokens = gating.shape[0]
            gating = gating.reshape(tokens, ep, local_experts)[:, list(rank_order)]
            gating = gating.reshape(tokens, ep * local_experts)
        # experts_start is int32 [num_shards, 1], each shard's first global
        # expert: its EP rank times local_experts.
        rank = experts_start.astype(jnp.int32) // local_experts
        return fused_ep_moe(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            top_k=topk,
            renormalize=renormalize,
            mesh=mesh,
            tile_rows=TILE_ROWS,
            weight_format=WEIGHT_FORMAT,
            activation="silu",
            config=config,
            experts_per_shard=local_experts,
            # The torch bridge exports the op as one jitted function: its SPMD
            # partitioner refuses the layer's own inner jit, a side-effecting
            # custom call without a sharding, and `lax.axis_index`, so the
            # shard's index goes in as data.
            shard_index=mesh_index_of_rank[rank] if permuted else rank,
            jit=False,
            side_effects=False,
        )

    name = FUSED_MOE_EP_V3_OP_NAME + (f"_{len(_OPS)}" if _OPS else "")
    spec = PartitionSpec(EP_AXIS_NAME)
    # Every operand and the single output are sharded on axis 0 over EP.
    # torch_tpu's jax_op stub types the specs as tuple[tuple[str, ...], ...],
    # but hands each to NamedSharding, which needs a PartitionSpec -- no longer
    # a tuple subclass (jax 0.11.2). Adaptive's identical call is baselined.
    # pyrefly: ignore[no-matching-overload]
    op = jax_op(
        name,
        moe,
        mesh=mesh,
        input_partition_specs=(spec,) * 7,
        output_partition_specs=spec,
    )

    # The kernel returns the rows it was given, so the fake needs no export.
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def fused_moe_ep_v3_supported(owner: Any) -> bool:
    """Whether `prebuild_fused_moe_ep_v3` armed v3 for this layer.

    `owner` is the layer's quant method, where the op was recorded.
    """
    return getattr(owner, FUSED_MOE_EP_V3_OP_ATTR, None) is not None


def fused_moe_ep_v3(owner: Any, layer, x: torch.Tensor, router_logits: torch.Tensor):
    """Run one MoE layer through the v3 kernel.

    Takes this rank's own tokens and returns this rank's own tokens, so the
    caller must not all-gather the input or reduce-scatter the output. Only
    call it when `fused_moe_ep_v3_supported(owner)` said yes.
    """
    return getattr(owner, FUSED_MOE_EP_V3_OP_ATTR)(
        x,
        layer.w13_weight,
        layer.w2_weight,
        _squeeze_channel_scale_checked(layer.w13_weight_scale_inv),
        _squeeze_channel_scale_checked(layer.w2_weight_scale_inv),
        router_logits,
        layer._experts_start.reshape(1, 1),
    )


def _check_deployment(layer) -> bool:
    """Raise on a deployment v3 cannot serve at all; True for tensor-parallel
    attention over the EP group, whose layers fused_moe_ep_v3_tp arms at load.

    The operator selected v3 explicitly, so these are errors rather than a
    silent fallback to the grouped-matmul path.
    """
    from vllm.config import get_current_vllm_config

    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )

    # Read off the config, not `get_tp_group()`: a draft model's weight load
    # rewrites the group coordinator to width 1.
    tp = get_current_vllm_config().parallel_config.tensor_parallel_size
    if tp > 1:
        from vllm_torchtpu.layers.adapter.fused_moe_ep_v3_tp import (
            tp_attention_over_ep,
        )

        if tp_attention_over_ep():
            return True
        raise NotImplementedError(
            f"MOE_FUSED_EP_KERNEL_IMPL=v3 serves data parallelism with expert "
            f"parallelism only; tensor_parallel_size={tp} replicates every "
            "token on each TP rank, so the kernel would do tp_size times the "
            "expert work. Use --data-parallel-size with "
            "--tensor-parallel-size=1, or MOE_FUSED_EP_KERNEL_IMPL=v2."
        )
    pcp = _pcp_size(layer)
    if pcp > 1:
        raise NotImplementedError(
            f"MOE_FUSED_EP_KERNEL_IMPL=v3 does not support prefill context "
            f"parallelism (pcp_size={pcp}). Use MOE_FUSED_EP_KERNEL_IMPL=v2."
        )
    if not layer.moe_config.moe_parallel_config.use_ep:
        raise NotImplementedError(
            "MOE_FUSED_EP_KERNEL_IMPL=v3 addresses experts by EP shard; pass "
            "--enable-expert-parallel."
        )
    if enable_pipelined_collective_and_compute():
        raise NotImplementedError(
            "MOE_FUSED_EP_KERNEL_IMPL=v3 and MoE chunk pipelining "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE) both claim the expert-parallel "
            "exchange; enable one of them."
        )
    return False


def prebuild_fused_moe_ep_v3(layer, activation: str):
    """Resolve the mesh and build the v3 op for one layer, or None to keep
    the grouped-matmul path.

    Called once per MoE layer from `process_weights_after_loading`: mapping EP
    ranks to devices is a collective and a `Mesh` cannot survive Dynamo's
    tracing. Raises on a deployment v3 cannot serve.
    """
    if _check_deployment(layer):
        logger.info_once(
            "Fused EP MoE v3: tensor-parallel attention over the EP group; "
            "fused_moe_ep_v3_tp arms the layers at load"
        )
        return None
    mesh = build_ep_mesh()
    if mesh is None:
        logger.info_once("Fused EP MoE v3 declined: no EP group")
        return None
    ep = mesh.shape[EP_AXIS_NAME]
    w1 = layer.w13_weight
    local_experts, hidden = w1.shape[0], w1.shape[1]
    inter = w1.shape[2] // 2
    reason = (
        _unsupported_layer_reason(layer, local_experts, ep)
        or _unsupported_routing_reason(layer)
        or _unsupported_weights_reason(layer, local_experts, hidden, inter, activation)
    )
    if reason is not None:
        logger.info_once("Fused EP MoE v3 declined a layer: %s", reason)
        return None
    # The mesh is in device-id order (`ep_mesh._ep_mesh_order`), so this rank
    # sits at mesh index `ep_mesh_index()` and holds EP rank
    # `rank_order[ep_mesh_index()]`'s contiguous expert block. A rank that
    # declined here while its peers armed would leave them in a collective it
    # never joins, so a block anywhere else is an error, not a decline.
    order, mesh_index = ep_rank_order(), ep_mesh_index()
    # Both are None only without an EP group, which build_ep_mesh ruled out.
    assert order is not None and mesh_index is not None
    rank_order = tuple(order)
    ep_rank = rank_order[mesh_index]
    if moe_routing.get_experts_start(layer) != ep_rank * local_experts:
        raise RuntimeError(
            f"MOE_FUSED_EP_KERNEL_IMPL=v3: EP rank {ep_rank}'s first expert is "
            f"{moe_routing.get_experts_start(layer)}, not "
            f"{ep_rank} x {local_experts}"
        )
    topk = layer.moe_config.experts_per_token
    return _build_op(
        mesh,
        topk,
        bool(layer.renormalize),
        rank_order,
        hidden=hidden,
        inter=inter,
        local_experts=local_experts,
    )


def _unsupported_layer_reason(layer, local_experts: int, ep: int) -> str | None:
    """Why this layer's expert layout or biases cannot go through v3."""
    if layer.global_num_experts != local_experts * ep:
        return (
            f"{layer.global_num_experts} experts do not split evenly over "
            f"{ep} shards of {local_experts}"
        )
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "expert biases are not wired through the v3 op"
    return None


def _unsupported_routing_reason(layer) -> str | None:
    """Why this layer's routing is not the kernel's, or None.

    The kernel scores with a plain softmax and selects top-k itself, with
    `renormalize` as its only option. Taking the path for any other router
    would route tokens to different experts with different weights, silently.
    """
    if layer.custom_routing_function is not None:
        return "the layer supplies a custom_routing_function"
    if layer.scoring_func != "softmax":
        return f"scoring_func {layer.scoring_func!r}; the kernel scores with softmax"
    if layer.e_score_correction_bias is not None:
        return "the layer has an e_score_correction_bias"
    if layer.routed_scaling_factor != 1.0:
        return f"routed_scaling_factor {layer.routed_scaling_factor}"
    if (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    ):
        return "grouped top-k routing"
    if layer.apply_router_weight_on_input:
        return "the router weight is applied on the input"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "the layer routes by hash table"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return (
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set and the kernel "
            "does its own routing"
        )
    return None


def _unsupported_weights_reason(
    layer, local_experts: int, hidden: int, inter: int, activation: str
) -> str | None:
    """Why this layer's weights or activation cannot use v3, or None."""
    w1, w2 = layer.w13_weight, layer.w2_weight
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        return f"weight rank/expert mismatch {tuple(w1.shape)} {tuple(w2.shape)}"
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return (
            f"the kernel's fp8 form takes float8_e4m3fn expert weights, got "
            f"{w1.dtype} / {w2.dtype}"
        )
    if w2.shape[1] != inter or w2.shape[2] != hidden:
        return f"w2 {tuple(w2.shape)} is not [E, {inter}, {hidden}]"
    if activation != "silu":
        return f"activation {activation!r}; the adapter wires silu only"
    scales = (
        (layer.w13_weight_scale_inv, 2 * inter),
        (layer.w2_weight_scale_inv, hidden),
    )
    if any(_squeeze_channel_scale(s, local_experts, n) is None for s, n in scales):
        return "a weight scale is not the per-output-channel form the kernel takes"
    return None
