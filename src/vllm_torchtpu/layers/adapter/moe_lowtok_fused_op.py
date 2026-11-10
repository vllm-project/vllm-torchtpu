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

from typing import Any

import jax
import numpy as np
import torch
from jax import lax
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    get_ep_group,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives
from vllm_torchtpu.layers.adapter import moe_options, token_padding
from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The kernel's forms on the DP x TP decode layout: the hierarchical dispatch
# and combine (half the inter-chip bytes per core over a chip's TP pair), a
# weight ring of one expert, and each expert applied to its routed rows only.
# Every DP x TP decode leg ran these values. Whether the rows travel as
# per-row FP8 is the tpu_moe.lowtok_fused_fp8_dispatch option (on by default,
# as those legs ran it), read when the op is armed.
DISPATCH_FORM = "hier"
COMBINE_FORM = "hier"
RING_DEPTH = 1
GATHER_ROWS = True

_OPS: dict[int, torch.library.CustomOpDef] = {}
# This core's (dp, tp) index on the mesh view, as an int32 [2] device tensor.
_RANK = None
_DP = 0
_TP = 0
_ARMED = False
_CHECKED = False
_FP8_DISPATCH = False


def _groups_match_mesh(mesh_index: int, dp: int, tp: int) -> bool:
    """Whether this core's TP group is its block of `tp` consecutive mesh
    indices and its DP group is the cores with its TP index, agreed over EP."""
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    order = ep_rank_order()
    tp_ranks = tuple(int(r) for r in get_tp_group().ranks)
    dp_ranks = set(int(r) for r in get_dp_group().ranks)
    block = mesh_index - mesh_index % tp
    t = mesh_index % tp
    mine = (
        order is not None
        and len(tp_ranks) == tp
        and all(order[block + j] in tp_ranks for j in range(tp))
        and len(dp_ranks) == dp
        and dp_ranks == {order[j * tp + t] for j in range(dp)}
    )
    group = get_ep_group()
    assert group is not None  # the caller found the EP mesh
    verdicts = [None] * group.world_size
    dist.all_gather_object(verdicts, bool(mine), group=group.cpu_group)
    return all(verdicts)


def prebuild(vllm_config: Any, topk: int) -> None:
    """Arm the op for this deployment, outside any traced region.

    The caller calls it for a layer the few-token kernel serves. Armed when
    the layout is DP x TP over the EP mesh with TP pairs on consecutive cores;
    otherwise a no-op. ``vllm_config`` supplies the tpu_moe options.
    """
    global _RANK, _DP, _TP, _ARMED, _CHECKED, _FP8_DISPATCH
    if _CHECKED:
        return
    _CHECKED = True
    fp8_dispatch = moe_options.get(vllm_config).lowtok_fused_fp8_dispatch
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    mesh = build_ep_mesh()
    if mesh is None:
        return
    cores = mesh.shape[EP_AXIS_NAME]
    tp = int(get_tp_group().world_size)
    dp = int(get_dp_group().world_size)
    forms = (DISPATCH_FORM, COMBINE_FORM)
    index = ep_mesh_index()
    assert index is not None  # build_ep_mesh found the EP group
    if (
        dp * tp != cores
        or dp < 2
        or (tp != 2 and "hier" in forms)
        or not _groups_match_mesh(index, dp, tp)
    ):
        logger.debug(
            "moe_lowtok fused collectives not armed: dp=%d x tp=%d is not the "
            "%d-core EP mesh with TP pairs on consecutive cores (forms %s)",
            dp,
            tp,
            cores,
            forms,
        )
        return
    _DP, _TP = dp, tp
    _FP8_DISPATCH = fp8_dispatch
    _RANK = torch.tensor([index // tp, index % tp], dtype=torch.int32, device="tpu")
    _ARMED = True
    logger.info(
        "moe_lowtok fused collectives armed | dp=%d tp=%d | dispatch=%s combine=%s | "
        "fp8_dispatch=%s | ring_depth=%d | gather_rows=%s | few-token cap %d rows",
        dp,
        tp,
        forms[0],
        forms[1],
        _FP8_DISPATCH,
        RING_DEPTH,
        GATHER_ROWS,
        moe_lowtok_scores.armed_max_tokens(),
    )
    _build_op(topk)


def _build_op(topk: int) -> torch.library.CustomOpDef:
    op = _OPS.get(topk)
    if op is not None:
        return op
    mesh = build_ep_mesh()
    assert mesh is not None  # prebuild armed the op on this mesh
    dp, tp = _DP, _TP
    mesh2 = Mesh(np.asarray(mesh.devices).reshape(dp, tp), axis_names=("dp", "tp"))
    rows = PartitionSpec("dp")
    experts = PartitionSpec(("dp", "tp"))
    cap = moe_lowtok_scores.armed_max_tokens()
    forms = (DISPATCH_FORM, COMBINE_FORM)
    fp8_dispatch = _FP8_DISPATCH
    depth = RING_DEPTH
    gather_rows = GATHER_ROWS

    def body(x, scores, w13, w2, s13, s2, experts_start, rank):
        start = experts_start[0]
        rows_l = x.shape[0]
        if 0 < dp * rows_l <= cap:
            padded = -(-rows_l // 16) * 16
            return fused_collectives.moe_lowtok_fused(
                x,
                scores,
                w13,
                s13,
                w2,
                s2,
                start,
                topk=topk,
                dp=dp,
                tp=tp,
                dispatch="inkernel",
                combine="inkernel",
                dispatch_form=forms[0],
                combine_form=forms[1],
                depth=depth,
                dispatch_fp8=fp8_dispatch and padded % 32 == 0,
                rank=rank,
                gather_rows=gather_rows,
            )
        # Above the cap: the served path, inside the op.
        xg = lax.all_gather(x, "dp", axis=0, tiled=True)
        lg = lax.all_gather(scores, "dp", axis=0, tiled=True)
        out = moe_lowtok_scores._moe_scores_fn(
            xg, lg, w13, w2, s13, s2, start, topk=topk
        )
        out = lax.psum_scatter(out, "dp", scatter_dimension=0, tiled=True)
        return lax.psum(out, "tp")

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
            mesh=mesh2,
            in_specs=(rows, rows, experts, experts, experts, experts, experts, experts),
            out_specs=rows,
            check_vma=False,
        )(x, scores, w13, w2, s13, s2, experts_start, rank)

    def fake(x, *_args):
        return torch.empty_like(x)

    op = sharded_jax_op(
        f"pallas::moe_lowtok_fused_k{topk}",
        fn,
        mesh=mesh2,
        input_partition_specs=(
            rows,
            rows,
            experts,
            experts,
            experts,
            experts,
            experts,
            experts,
        ),
        output_partition_specs=rows,
        fake=fake,
    )
    _OPS[topk] = op
    return op


def serves(layer, scores_op, is_sequence_parallel: bool) -> bool:
    """Whether this layer's step takes the fused op: the op is armed, the layer
    is served by the plain scores entry (no copy slots), and the step's rows
    are replicated over TP."""
    from vllm_torchtpu.layers.adapter import sequence_parallel

    return (
        _ARMED
        and scores_op is not None
        and not getattr(layer, "_lowtok_has_copies", False)
        and sequence_parallel.rows_are_replicated(is_sequence_parallel)
    )


def retain(out: torch.Tensor) -> torch.Tensor:
    """``out`` plus a zero read from the rank buffer, for a MoE path that does
    not take the op while another token-count branch of the same model does:
    a bucket is handed over only between traces that lift the same buffers."""
    if _RANK is None or not shape_variants.in_warmup():
        return out
    value = _RANK[0].to(out.dtype)
    return out + (value - value)


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
