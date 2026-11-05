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
"""PyTorch custom operator wrapper for the low-latency Pallas ICI all-reduce kernel.
"""

import contextlib

import jax
import torch
from jax.sharding import PartitionSpec

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
from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_gather,
    all_reduce,
    reduce_scatter,
)
from vllm_torchtpu.kernels.experimental.small_collectives import common as sc_common
from vllm_torchtpu.kernels.experimental.small_collectives.common import LANES
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Largest TP all-reduce, in rows, the kernel takes: at H=4096 bf16 on 8 cores
# 2-stage hierarchical all-reduce beats XLA through 128 rows (j-c8c95322).
# Configurable via TPU_SMALL_ALL_REDUCE_MAX_ROWS (default 128).
MAX_ROWS = envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS
# Largest DP combine, in rows per engine, the reduce-scatter takes: measured
# crossover between 40 and 48 (j-a8d66cde, j-e6f25e61, j-de6349e8). The win
# relies on XLA lowering psum_scatter to a full all-reduce when rows per engine
# is not a multiple of 16, which depends on libtpu: re-measure on a bump.
COMBINE_MAX_ROWS = 40
# Largest DP dispatch, in rows per engine, the fused all-gather pair takes:
# bounds VMEM scratch (2 x 8 x rows x width) and stays in the latency-bound
# small-batch regime where a single-hop push beats two XLA ring all-gathers.
DISPATCH_MAX_ROWS = 64

_AR_OP = None
_RS_OP = None
_AG_OP = None
# The TP all-reduce over a TP group that is a sub-group of the mesh (DP x TP
# engines): the kernel over the "tp" axis of a (dp, tp) view of the mesh, this
# core's TP index as data, and the TP group's ranks it serves.
_AR_TP_OP = None
_TP_RANK = None
_TP_RANKS: tuple[int, ...] = ()
# This core's mesh index as an int32 [1] device tensor: the kernels take it as
# data (see small_collectives/common.py).
_RANK = None
_CORES = 0
_MESH_RANKS: tuple[int, ...] = ()
# Whether the DP group is the mesh, so its combine may take the kernel.
_DP_COMBINE = False
# True inside a compiled region whose max token bucket is statically <= MAX_ROWS
# (e.g. _MtpFusedLoop when max_loop_bucket <= MAX_ROWS), so the kernel can be
# selected without introducing a dynamic-shape guard outside shape_variants.warmup().
_FORCE_SMALL_AR = False


@contextlib.contextmanager
def force_small_all_reduce(enabled: bool = True):
    """Enable small_all_reduce unconditionally inside a bounded compiled region."""
    global _FORCE_SMALL_AR
    prev = _FORCE_SMALL_AR
    _FORCE_SMALL_AR = bool(enabled)
    try:
        yield
    finally:
        _FORCE_SMALL_AR = prev


def _build_all_reduce(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        if block.shape[0] > MAX_ROWS:
            return jax.lax.psum(block, axis_name=EP_AXIS_NAME)
        return all_reduce.small_all_reduce(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank
        )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return torch.empty_like(x)

    return sharded_jax_op(
        "pallas::small_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def _build_tp_all_reduce(mesh, tp: int, form: str | None):
    """The all-reduce over the TP pairs of the mesh seen as (dp, tp): mesh
    index i is the TP partner of i ^ (tp - 1) group-wise, which `prebuild`
    checks against the torch TP group before arming."""
    import numpy as np
    from jax.sharding import Mesh

    n = mesh.shape[EP_AXIS_NAME]
    mesh2 = Mesh(
        np.asarray(mesh.devices).reshape(n // tp, tp), axis_names=("dp_", "tp_")
    )
    spec = PartitionSpec(("dp_", "tp_"))

    def body(block, rank):
        # A forced region (force_small_all_reduce) reaches here with any row
        # count; blocks above the cap take XLA's all-reduce inside the op.
        if block.shape[0] > MAX_ROWS:
            return jax.lax.psum(block, axis_name="tp_")
        with sc_common.mesh_axes(("dp_", "tp_")):
            return all_reduce.small_all_reduce(
                block, num_cores=tp, axis="tp_", rank=rank, form=form or None
            )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh2, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return torch.empty_like(x)

    return sharded_jax_op(
        "pallas::small_all_reduce_tp",
        fn,
        mesh=mesh2,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def _tp_pairs_match_mesh(mesh_index: int, tp: int) -> bool:
    """Whether every core's TP group is the block of `tp` consecutive mesh
    indices it sits in, agreed over the EP group."""
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_tp_group

    order = ep_rank_order()
    group = get_ep_group()
    ep_ranks = tuple(int(r) for r in group.ranks)
    tp_ranks = tuple(int(r) for r in get_tp_group().ranks)
    block = mesh_index - mesh_index % tp
    mine = order is not None and len(tp_ranks) == tp and all(
        ep_ranks[order[block + j]] in tp_ranks for j in range(tp)
    )
    verdicts = [None] * group.world_size
    dist.all_gather_object(verdicts, bool(mine), group=group.cpu_group)
    return all(verdicts)


def _build_reduce_scatter(mesh, rank_of_core: tuple[int, ...] | None = None):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        return reduce_scatter.small_reduce_scatter(
            block,
            num_cores=n,
            axis=EP_AXIS_NAME,
            rank=rank,
            rank_of_core=rank_of_core,
        )

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, rank)

    def fake(x, rank):
        return x.new_empty((x.shape[0] // n, x.shape[1]))

    return sharded_jax_op(
        "pallas::small_reduce_scatter",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )


def _build_all_gather_pair(mesh, rank_of_core: tuple[int, ...] | None = None):
    """The data-parallel MoE dispatch: each rank's ([rows, hidden],
    [rows, num_experts]) -> ([n * rows, hidden], [n * rows, num_experts])
    with block r = rank r's rows."""
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(x, logits, rank):
        return all_gather.small_all_gather_pair(
            x,
            logits,
            num_cores=n,
            axis=EP_AXIS_NAME,
            rank=rank,
            rank_of_core=rank_of_core,
        )

    def fn(
        x: jax.Array, logits: jax.Array, rank: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=(spec, spec),
            check_vma=False,
        )(x, logits, rank)

    def fake(x, logits, rank):
        return (
            x.new_empty((n * x.shape[0], x.shape[1])),
            logits.new_empty((n * logits.shape[0], logits.shape[1])),
        )

    return sharded_jax_op(
        "pallas::small_all_gather_pair",
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec),
        output_partition_specs=(spec, spec),
        fake=fake,
    )


def prebuild() -> None:
    """Build the ops and this core's rank tensor, outside any traced region:
    the mesh comes from a collective. A no-op unless TPU_SMALL_ALL_REDUCE."""
    global _AR_OP, _RS_OP, _AG_OP, _RANK, _CORES, _MESH_RANKS, _DP_COMBINE, MAX_ROWS
    global _AR_TP_OP, _TP_RANK, _TP_RANKS
    MAX_ROWS = envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS
    if not envs.TPU_SMALL_ALL_REDUCE or _AR_OP is not None:
        return
    mesh = build_ep_mesh()
    if mesh is None:
        raise NotImplementedError(
            "TPU_SMALL_ALL_REDUCE=1 builds its kernels over the expert-parallel "
            "mesh, and this deployment has no expert-parallel group (a dense "
            "model or one core). Unset TPU_SMALL_ALL_REDUCE."
        )
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    _CORES = mesh.shape[EP_AXIS_NAME]
    _MESH_RANKS = tuple(int(r) for r in get_ep_group().ranks)
    _RANK = torch.tensor([ep_mesh_index()], dtype=torch.int32, device="tpu")
    order = ep_rank_order()
    _AR_OP = _build_all_reduce(mesh)
    _RS_OP = _build_reduce_scatter(mesh, order)
    _AG_OP = _build_all_gather_pair(mesh, order)
    # Rows are in DP-rank order while mesh cores sit in device-id order; both
    # DP kernels map mesh core c to DP rank ep_rank_order()[c] when DP == EP.
    _DP_COMBINE = tuple(int(r) for r in get_dp_group().ranks) == _MESH_RANKS
    tp_all_reduce = tuple(int(r) for r in get_tp_group().ranks) == _MESH_RANKS
    tp = int(get_tp_group().world_size)
    if (
        envs.TPU_SMALL_ALL_REDUCE_TP
        and not tp_all_reduce
        and 1 < tp < _CORES
        and _CORES % tp == 0
        and _tp_pairs_match_mesh(ep_mesh_index(), tp)
    ):
        _TP_RANKS = tuple(int(r) for r in get_tp_group().ranks)
        _TP_RANK = torch.tensor([ep_mesh_index() % tp], dtype=torch.int32, device="tpu")
        _AR_TP_OP = _build_tp_all_reduce(mesh, tp, envs.TPU_SMALL_ALL_REDUCE_TP_FORM)
    elif envs.TPU_SMALL_ALL_REDUCE_TP:
        logger.warning(
            "TPU_SMALL_ALL_REDUCE_TP=1 ignored: the TP group (%d ranks) is not a "
            "block of the %d-core EP mesh",
            tp,
            _CORES,
        )
    logger.info(
        "Small collectives armed | cores=%d | max_rows=%d | "
        "combine_max_rows=%d | dispatch_max_rows=%d | one_shot_max_bytes=%d | "
        "tp_all_reduce=%s | dp_combine=%s | tp_pair_all_reduce=%s",
        _CORES,
        MAX_ROWS,
        COMBINE_MAX_ROWS,
        min(MAX_ROWS, DISPATCH_MAX_ROWS),
        all_reduce.ONE_SHOT_MAX_BYTES,
        tp_all_reduce,
        _DP_COMBINE,
        _AR_TP_OP is not None,
    )


def _fits(x: torch.Tensor, width_multiple: int, *, allow_forced: bool = False) -> bool:
    """Static checks: whether this call site can take a kernel at all."""
    return (
        torch.compiler.is_compiling()
        and (shape_variants.in_warmup() or (allow_forced and _FORCE_SMALL_AR))
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] % width_multiple == 0
    )


def retain(out: torch.Tensor, rank: torch.Tensor | None = None) -> torch.Tensor:
    """``out`` plus a zero read from the rank buffer (``rank``, default the
    mesh rank), for a token-count branch that skips the kernel while its other
    side calls it: the warmup hands a bucket over only between traces that
    lift the same buffers. A read whose result is dead would be eliminated, so
    the zero rides on ``out``."""
    rank = _RANK if rank is None else rank
    if rank is None or not shape_variants.in_warmup():
        return out
    value = rank[0].to(out.dtype)
    return out + (value - value)


def all_reduce_or(x: torch.Tensor, ranks, fallback) -> torch.Tensor:
    """The sum of ``x`` over the group ``ranks``: the kernel for a small block
    when the group is the mesh, ``fallback(x)`` otherwise."""
    if _AR_TP_OP is not None and tuple(ranks) == _TP_RANKS:
        if not _fits(x, len(_TP_RANKS) * LANES, allow_forced=True):
            return fallback(x)
        if _FORCE_SMALL_AR or x.shape[0] <= MAX_ROWS:
            return _AR_TP_OP(x, _TP_RANK)
        return retain(fallback(x), _TP_RANK)
    if _AR_OP is None or tuple(ranks) != _MESH_RANKS:
        return fallback(x)
    if not _fits(x, _CORES * LANES, allow_forced=True):
        return fallback(x)
    if _FORCE_SMALL_AR or x.shape[0] <= MAX_ROWS:
        return _AR_OP(x, _RANK)
    return retain(fallback(x))


def dp_combine_or(x: torch.Tensor, fallback) -> torch.Tensor:
    """The DP MoE combine of ``x`` ([dp * rows, hidden], block r for DP rank
    r): the kernel for a small block when the DP group is the mesh,
    ``fallback(x)`` otherwise."""
    if _RS_OP is None or not _DP_COMBINE or not _fits(x, LANES):
        return fallback(x)
    if x.shape[0] <= _CORES * COMBINE_MAX_ROWS:
        return _RS_OP(x, _RANK)
    return retain(fallback(x))


def _fits_dispatch(x: torch.Tensor, router_logits: torch.Tensor) -> bool:
    """Static checks: whether this DP dispatch call site can take the kernel."""
    return (
        torch.compiler.is_compiling()
        and shape_variants.in_warmup()
        and x.dim() == 2
        and router_logits.dim() == 2
        and x.shape[0] == router_logits.shape[0]
        and x.dtype == torch.bfloat16
        and router_logits.dtype in (torch.bfloat16, torch.float32)
        and x.shape[1] % LANES == 0
        and router_logits.shape[1] % LANES == 0
    )


def dp_dispatch_or(
    x: torch.Tensor, router_logits: torch.Tensor, fallback
) -> tuple[torch.Tensor, torch.Tensor]:
    """The DP MoE dispatch of ``(x, router_logits)`` ([rows, hidden] and
    [rows, num_experts] -> [dp * rows, ...], block r for DP rank r): the fused
    kernel for a small block when the DP group is the mesh,
    ``fallback(x, router_logits)`` otherwise."""
    if _AG_OP is None or not _DP_COMBINE or not _fits_dispatch(x, router_logits):
        return fallback(x, router_logits)
    if x.shape[0] <= min(MAX_ROWS, DISPATCH_MAX_ROWS):
        return _AG_OP(x, router_logits, _RANK)
    out_x, out_logits = fallback(x, router_logits)
    return retain(out_x), out_logits
