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
"""The few-token MoE layer with its dispatch and combine in the kernel, as a torch op.

On a DP x TP layout (DP >= 2, TP = 2) whose TP pairs are consecutive cores of
the EP mesh, a replicated sequence-parallel MoE step of a layer the few-token
kernel serves (every TP core holds its engine's rows; the DP engines' rows are
gathered for the expert pass) runs its DP all-gather, the moe_lowtok expert
pass and the DP reduce-scatter plus TP all-reduce as one Pallas call over the
(dp, tp) view of the mesh (kernels/experimental/moe_lowtok/fused_collectives.py).
Steps above the few-token cap keep the served path inside the op: XLA's
all-gather, the grouped-matmul MoE and XLA's reduce-scatter and all-reduce.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
import torch
from jax import lax
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives
from vllm_torchtpu.layers.adapter import (
    fused_moe_ep_v3,
    sequence_parallel,
    small_all_reduce_op,
    token_padding,
)
from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OPS: dict[int, torch.library.CustomOpDef] = {}
# The program's operand and result specs: rows by DP engine; the experts'
# weights, scales and experts_start, and the core's index, by core.
_OUT_SPEC = PartitionSpec(fused_collectives.DP_AXIS)
_IN_SPECS = (_OUT_SPEC, _OUT_SPEC) + (
    PartitionSpec((fused_collectives.DP_AXIS, fused_collectives.TP_AXIS)),
) * 6
# This core's (dp, tp) index on the mesh view, as an int32 [2] device tensor.
_RANK = None
_DP = 0
_TP = 0
_ARMED = False
_CHECKED = False


def maybe_prebuild(layer, scores_op) -> None:
    """:func:`prebuild` for a MoE layer whose steps can reach the op, then the
    op for the layer's top-k once armed; nothing for any other layer.

    The op's only caller is the FP8 method's fused EP v3 entry
    (``_apply_owning_dispatch_v3``), which the method's forward takes for a
    layer v3's prebuild armed while USE_MOE_FUSED_EP_KERNEL is set; call this
    after that prebuild. There the op serves the steps v3's kernel does not
    take when the few-token kernel's scores entry (``scores_op``, from
    ``moe_lowtok_scores.prebuild``) serves the layer without copy slots under
    expert parallelism (:func:`serves`). Elsewhere nothing can call the op,
    so nothing is checked, allocated, built or logged for it.
    """
    if not (
        envs.USE_MOE_FUSED_EP_KERNEL
        and fused_moe_ep_v3.fused_moe_ep_supported(layer)
        and scores_op is not None
        and not layer._lowtok_has_copies
        and layer.moe_config.moe_parallel_config.use_ep
    ):
        return
    prebuild()
    if _ARMED:
        # Every top-k the forward can ask for is built here, outside the
        # traced region (a drafter's layer can differ from the target's).
        _build_op(int(layer.moe_config.experts_per_token))


def prebuild() -> None:
    """Arm the op for this deployment, outside any traced region.

    ``maybe_prebuild`` calls it for a layer the op can serve. Armed when the
    layout is DP x TP over the EP mesh with TP pairs on consecutive cores;
    otherwise a no-op.
    """
    global _RANK, _DP, _TP, _ARMED, _CHECKED
    if _CHECKED:
        return
    _CHECKED = True
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    mesh = build_ep_mesh()
    if mesh is None:
        return
    cores = mesh.shape[EP_AXIS_NAME]
    tp = int(get_tp_group().world_size)
    dp = int(get_dp_group().world_size)
    index = ep_mesh_index()
    assert index is not None  # build_ep_mesh found the EP group
    if (
        dp * tp != cores
        or dp < 2
        or tp != fused_collectives.TP_PAIR
        or not small_all_reduce_op.tp_group_is_one_chip(
            tp, "The moe_lowtok fused collectives"
        )
        or not small_all_reduce_op.groups_match_mesh(index, tp, dp)
    ):
        logger.debug(
            "moe_lowtok fused collectives not armed: dp=%d x tp=%d is not the "
            "%d-core EP mesh with each chip's TP pair on consecutive cores",
            dp,
            tp,
            cores,
        )
        return
    _DP, _TP = dp, tp
    _RANK = torch.tensor([index // tp, index % tp], dtype=torch.int32, device="tpu")
    _ARMED = True
    logger.info(
        "moe_lowtok fused collectives armed | dp=%d tp=%d | few-token cap %d rows",
        dp,
        tp,
        moe_lowtok_scores.armed_max_tokens(),
    )


def program(mesh: Mesh, topk: int, cap: int) -> Callable[..., jax.Array]:
    """The op's JAX program over ``mesh``, whose axes are (DP_AXIS, TP_AXIS)
    of fused_collectives.

    Operands in :func:`apply`'s order: each engine's rows and router logits,
    then w13, w2, their scales, experts_start and the core's (dp, tp) index,
    by core. A step of at most ``cap`` rows over the DP engines runs the
    fused kernel; a larger one takes the served path inside the op: XLA's DP
    all-gather, ``moe_lowtok_scores.moe_scores_fn``, XLA's DP reduce-scatter
    and TP all-reduce. Returns each engine's rows summed over every rank.
    """
    dp_axis, tp_axis = fused_collectives.DP_AXIS, fused_collectives.TP_AXIS
    dp, tp = mesh.shape[dp_axis], mesh.shape[tp_axis]

    def body(x, scores, w13, w2, s13, s2, experts_start, rank):
        start = experts_start[0]
        rows_l = x.shape[0]
        if 0 < dp * rows_l <= cap:
            # The rows travel as per-row FP8 wherever the kernel takes them.
            return fused_collectives.moe_lowtok_fused(
                x,
                scores,
                w13,
                s13,
                w2,
                s2,
                start,
                rank=rank,
                topk=topk,
                dp=dp,
                tp=tp,
                dispatch_fp8=fused_collectives.fp8_dispatch_fits(rows_l),
            )
        # Above the cap: the served path, inside the op.
        xg = lax.all_gather(x, dp_axis, axis=0, tiled=True)
        lg = lax.all_gather(scores, dp_axis, axis=0, tiled=True)
        out = moe_lowtok_scores.moe_scores_fn(
            xg, lg, w13, w2, s13, s2, start, topk=topk
        )
        out = lax.psum_scatter(out, dp_axis, scatter_dimension=0, tiled=True)
        return lax.psum(out, tp_axis)

    def fn(
        x: jax.Array,
        scores: jax.Array,
        w13: jax.Array,
        w2: jax.Array,
        s13: jax.Array,
        s2: jax.Array,
        experts_start: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=_IN_SPECS,
            out_specs=_OUT_SPEC,
            check_vma=False,
        )(x, scores, w13, w2, s13, s2, experts_start, rank)

    return fn


def _build_op(topk: int) -> torch.library.CustomOpDef:
    op = _OPS.get(topk)
    if op is not None:
        return op
    mesh = build_ep_mesh()
    assert mesh is not None  # prebuild armed the op on this mesh
    mesh2 = Mesh(
        np.asarray(mesh.devices).reshape(_DP, _TP),
        axis_names=(fused_collectives.DP_AXIS, fused_collectives.TP_AXIS),
    )

    def fake(x, *_args):
        return torch.empty_like(x)

    op = sharded_jax_op(
        f"pallas::moe_lowtok_fused_k{topk}",
        program(mesh2, topk, moe_lowtok_scores.armed_max_tokens()),
        mesh=mesh2,
        input_partition_specs=_IN_SPECS,
        output_partition_specs=_OUT_SPEC,
        fake=fake,
    )
    _OPS[topk] = op
    return op


def serves(layer, scores_op, is_sequence_parallel: bool) -> bool:
    """Whether this layer's step takes the fused op: the op is armed, the layer
    is served by the plain scores entry (no copy slots), and the step's rows
    are replicated over TP. ``moe_lowtok_scores.prebuild`` sets
    ``_lowtok_has_copies`` on every layer it returns a scores op for."""
    return (
        _ARMED
        and scores_op is not None
        and not layer._lowtok_has_copies
        and sequence_parallel.rows_are_replicated(is_sequence_parallel)
    )


def retain(out: torch.Tensor) -> torch.Tensor:
    """``out`` plus a zero read from this op's rank buffer
    (``small_all_reduce_op.retain``), for a MoE path that does not take the op
    while another token-count branch of the same model does: a bucket is
    handed over only between traces that lift the same buffers."""
    if _RANK is None:
        return out
    return small_all_reduce_op.retain(out, _RANK)


def apply(layer, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    """This engine's rows summed over every rank."""
    logits = router_logits
    if envs.TPU_MOE_SKIP_PADDED_TOKENS:
        # The op takes this engine's own rows, so the mask is the local one.
        is_padding = token_padding.padding_mask_for(
            logits.shape[0], is_local_tensor=True
        )
        if is_padding is not None:
            logits = torch.where(
                is_padding[:, None], torch.full_like(logits, float("-inf")), logits
            )
    op = _build_op(int(layer.moe_config.experts_per_token))
    return op(
        x,
        logits,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
        layer._experts_start.reshape(1),
        _RANK,
    )
