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
"""The v3 fused expert-parallel MoE kernel behind `fused_moe_ep`.

Selected with ``MOE_FUSED_EP_KERNEL_IMPL=v3``. `prebuild_fused_moe_ep` keeps
its shared admission (EP group, mesh, rank buffer, token threshold, weight
and routing checks) and asks this module for the v3 refusals and the op. The
op takes the same operands as the v2 op -- x, w1, w2, the per-output-channel
FP8 scales, the router logits and this rank's mesh index -- so `fused_moe_ep`,
`fused_moe_ep_supported` and `VllmMoERunner` serve it unchanged.

``TPU_MOE_FUSED_EP_STEP_MIN_TOKENS`` keeps compiled token shapes below the
threshold off the kernel (`fused_kernel_takes`). Those shapes take the
grouped-matmul path with the layer's own all-gather and reduce-scatter, and
the compile flow traces each token bucket for its side of the threshold
(`compilation/shape_variants`). Both sides must lift the same graph inputs
for a bucket's executable to be handed between the traces, so each side reads
the buffers only the other one needs (`retain_fork_graph_inputs`).
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The FFN tile height (the kernel's `tile_rows`). Smaller tiles are much
# slower with the kernel's DMA strategy, so it is not a knob.
TILE_ROWS = 128

FUSED_MOE_EP_V3_OP_NAME = "pallas::fused_moe_ep_v3"

# One op per distinct closure, for the same reasons as `fused_moe_ep._OPS`:
# the op object is what torch_tpu exports and compiles, so one op per layer
# would export the same program once per MoE layer.
_OPS: dict[tuple, Any] = {}


def _tensor_parallel_size() -> int:
    """This deployment's tensor-parallel width, or 1 if it cannot be read.

    Read off the config rather than `get_tp_group()`: a draft model's weight
    load rewrites the group coordinator to width 1.
    """
    try:
        from vllm.config import get_current_vllm_config

        return int(get_current_vllm_config().parallel_config.tensor_parallel_size)
    except Exception:  # noqa: BLE001 - no config outside a served model
        return 1


def unsupported_reason(layer, activation: str, weight_format: str) -> str | None:
    """Why the v3 kernel cannot serve this layer, or None.

    Checked on top of `fused_moe_ep_unsupported_reason`, which already
    covers the weight shapes, the per-output-channel FP8 scales, expert
    biases and the routing options the v2 kernel refuses.
    """
    if weight_format != "fp8":
        return f"the v3 kernel is wired for FP8 weights only, not {weight_format}"
    if activation != "silu":
        return f"the v3 bridge passes silu only, not {activation!r}"
    # The kernel selects its own softmax top-k from the logits. Every other
    # routing option would route to different experts without an error.
    if layer.scoring_func != "softmax":
        return f"scoring_func {layer.scoring_func!r}; the v3 kernel scores with softmax"
    if layer.e_score_correction_bias is not None:
        return "the v3 kernel takes no expert-score correction bias"
    if float(layer.routed_scaling_factor) != 1.0:
        return (
            f"routed_scaling_factor {layer.routed_scaling_factor}; the v3 "
            "kernel does not scale the routed output"
        )
    if getattr(layer, "hash_indices_table", None) is not None:
        return "the v3 kernel takes no hash-routing table"
    if int(layer.moe_config.moe_parallel_config.pcp_size) > 1:
        return (
            "prefill context parallelism is not validated with the v3 kernel; "
            "use MOE_FUSED_EP_KERNEL_IMPL=v2"
        )
    tp = _tensor_parallel_size()
    if tp > 1:
        return (
            f"tensor parallelism (tp_size={tp}) replicates each batch on every "
            "rank, and the v3 kernel takes each rank's own rows"
        )
    if envs.MOE_FUSED_EP_SORTED_GATHER:
        return "MOE_FUSED_EP_SORTED_GATHER is a v2 option"
    return None


def build_op(
    mesh,
    topk: int,
    renormalize: bool,
    activation: str,
    mesh_expert_order: tuple[int, ...] | None,
    *,
    hidden: int,
    inter: int,
    local_experts: int,
):
    """The v3 op for this closure, built once per distinct closure.

    Called the same way as the v2 op:
    ``op(x, w1, w2, w1_scale, w2_scale, router_logits, rank)``, every operand
    sharded on axis 0 over the EP axis, returning this rank's rows.
    """
    from vllm_torchtpu.kernels.fused_moe.v3 import (
        MESH_AXIS,
        Config,
        fused_ep_moe,
        layout,
    )
    from vllm_torchtpu.kernels.fused_moe.v3 import layer as kernel_layer

    drop_nan_rows = bool(envs.TPU_MOE_SKIP_PADDED_TOKENS)
    key = (
        topk,
        renormalize,
        activation,
        mesh_expert_order,
        hidden,
        inter,
        local_experts,
        drop_nan_rows,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached

    # torch_tpu exports the whole op as one function, and its partitioner
    # refuses the kernel's own inner jit, an unsharded side-effecting custom
    # call, and `lax.axis_index` (a partition-id). So the layer is not
    # wrapped in jit, its programs carry their dependencies in their
    # outputs, and the shard index arrives as data (the rank operand).
    kernel_layer.WRAP_IN_JIT = False
    layout.SIDE_EFFECTS = False
    layout.SHARD_INDEX_OPERAND = True
    if MESH_AXIS != EP_AXIS_NAME:
        raise ValueError(
            f"the v3 kernel splits the experts over the mesh axis named "
            f"{MESH_AXIS!r}; the EP mesh axis is {EP_AXIS_NAME!r}"
        )
    ep = mesh.shape[EP_AXIS_NAME]
    config = Config.for_shape(
        hidden=hidden, inter=inter, experts_per_shard=local_experts, top_k=topk
    )
    expert_order = None if mesh_expert_order is None else list(mesh_expert_order)

    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        gating = gating.astype(jnp.float32)
        if expert_order is not None:
            # The kernel reads expert id e as mesh index e // local_experts,
            # but vLLM numbers experts by EP rank. Move EP rank r's block of
            # logits to the mesh index that rank sits at, with static slices.
            # The output is per token, so nothing is undone afterwards.
            gating = jnp.concatenate(
                [
                    gating[
                        :,
                        rank_at_index * local_experts : (rank_at_index + 1)
                        * local_experts,
                    ]
                    for rank_at_index in expert_order
                ],
                axis=1,
            )
        # A padded row arrives with NaN logits (`fused_moe_ep` masks them
        # under TPU_MOE_SKIP_PADDED_TOKENS); the kernel routes it nowhere and
        # returns zero for it.
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
            weight_format="fp8",
            activation=activation,
            drop_nan_rows=drop_nan_rows,
            config=config,
            shard_index=rank,
            experts_per_shard=local_experts,
        )

    name = (
        FUSED_MOE_EP_V3_OP_NAME
        if not _OPS
        else f"{FUSED_MOE_EP_V3_OP_NAME}_{len(_OPS)}"
    )
    spec = PartitionSpec(EP_AXIS_NAME)
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
    logger.info_once(
        "Fused EP MoE armed | kernel=v3 hidden=%d inter=%d local_experts=%d "
        "ep=%d topk=%d tile_rows=%d skip_padded_tokens=%s step_min_tokens=%d "
        "config=%s",
        hidden,
        inter,
        local_experts,
        ep,
        topk,
        TILE_ROWS,
        drop_nan_rows,
        envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS,
        config,
    )
    return op


def fused_kernel_takes(rows: int) -> bool:
    """Whether an armed layer runs a step of `rows` rows (this rank's own)
    through the v3 kernel. Smaller steps take the grouped-matmul path.

    `rows` is the traced token count, so the comparison leaves a shape guard
    behind, and the compile flow traces each token bucket for its side of the
    threshold (see `compilation/shape_variants`). Every DP rank runs the same
    padded bucket, so every rank makes the same choice and the two paths'
    collectives stay matched.
    """
    if envs.MOE_FUSED_EP_KERNEL_IMPL != "v3":
        return True
    threshold = envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS
    return threshold <= 0 or rows >= threshold


def retain_fork_graph_inputs(
    output: torch.Tensor, *buffers: torch.Tensor
) -> torch.Tensor:
    """`output` plus a zero read from each of `buffers`.

    Under TPU_MOE_FUSED_EP_STEP_MIN_TOKENS the bucket traces on the two sides
    of the threshold must lift the same graph inputs, or the compile flow
    cannot hand a bucket's executable from one trace to the other. The fused
    side reads the rank buffer and the grouped-matmul side the layer's
    `experts_start`; each side reads the other's through this. A no-op
    without the threshold, where there is only one trace.
    """
    if (
        envs.MOE_FUSED_EP_KERNEL_IMPL != "v3"
        or envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS <= 0
    ):
        return output
    for buffer in buffers:
        if buffer is None:
            continue
        value = buffer.reshape(-1)[0].to(output.dtype)
        output = output + (value - value)
    return output
