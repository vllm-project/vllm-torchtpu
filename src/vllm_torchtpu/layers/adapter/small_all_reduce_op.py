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
"""Torch custom ops for the low-latency Pallas ICI small collectives: the TP
all-reduce of a small block, and the reduce-scatter of the DP MoE combine."""

import contextlib
from collections.abc import Collection, Sequence
from typing import Any

import jax
import numpy as np
import torch
import torch.distributed as dist
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import collectives_options
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    get_ep_group,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_reduce,
    reduce_scatter,
)
from vllm_torchtpu.kernels.experimental.small_collectives import common as sc_common
from vllm_torchtpu.kernels.experimental.small_collectives.common import LANES
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Largest TP all-reduce, in rows, the kernel takes. Measured on the 8 cores of
# one TPU7x host, [rows, 4096] bf16, as the marginal time per all-reduce in a
# chain of them: the 2-stage hierarchical form takes 21.3 us at 128 rows
# against XLA's 24.8 us, ties XLA at 160 to 192 rows, and loses at 256 (34.0
# against 33.1 us).
MAX_ROWS = 128
# Largest DP combine, in rows per engine, the reduce-scatter takes: on one
# TPU7x host the kernel was faster than XLA's reduce-scatter at 40 rows per
# engine and slower at 48. The win relies on XLA lowering psum_scatter to a
# full all-reduce when rows per engine is not a multiple of 16, which depends
# on libtpu: re-measure on a bump.
COMBINE_MAX_ROWS = 40

_AR_OP = None
_RS_OP = None
# The TP all-reduce over a TP group that is a sub-group of the mesh (DP x TP
# engines): the kernel over the "tp" axis of a (dp, tp) view of the mesh, this
# core's TP index as data, and the TP group's ranks it serves.
_AR_TP_OP = None
_TP_RANK = None
_TP_RANKS: tuple[int, ...] = ()
# The TP-pair all-reduce serves a TP group of this many cores inside a larger
# mesh (so DP > 1): the two cores of a chip, the only layout it was measured
# on (DP4 x TP2 decode). Its form: every such decode leg ran "oneshot".
TP_PAIR_SIZE = 2
TP_PAIR_FORM = "oneshot"
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


def _build_tp_all_reduce(mesh, tp: int):
    """The all-reduce over the TP pairs of the mesh seen as (dp, tp): mesh
    index i is the TP partner of i ^ (tp - 1) group-wise, which `prebuild`
    checks against the torch TP group before arming."""
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
                block, num_cores=tp, axis="tp_", rank=rank, form=TP_PAIR_FORM
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


def _groups_on_mesh_blocks(
    order: Sequence[int] | None,
    mesh_index: int,
    tp_ranks: Collection[int],
    tp: int,
    dp_ranks: Collection[int] | None = None,
    dp: int = 0,
) -> bool:
    """Whether this core's TP group (``tp_ranks``) is the block of ``tp``
    consecutive mesh indices it sits in and, given ``dp_ranks``, its DP group
    is the ``dp`` cores with its TP index. ``order`` maps mesh index to rank
    (``ep_rank_order``)."""
    if order is None or len(tp_ranks) != tp:
        return False
    block = mesh_index - mesh_index % tp
    if not all(order[block + j] in tp_ranks for j in range(tp)):
        return False
    if dp_ranks is None:
        return True
    t = mesh_index % tp
    return len(dp_ranks) == dp and set(dp_ranks) == {
        order[j * tp + t] for j in range(dp)
    }


def groups_match_mesh(mesh_index: int, tp: int, dp: int = 0) -> bool:
    """Whether every core's TP group is the block of ``tp`` consecutive mesh
    indices it sits in and, when ``dp`` is given, every core's DP group is the
    ``dp`` cores with its TP index; agreed over the EP group (a collective)."""
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    mine = _groups_on_mesh_blocks(
        ep_rank_order(),
        mesh_index,
        tuple(int(r) for r in get_tp_group().ranks),
        tp,
        tuple(int(r) for r in get_dp_group().ranks) if dp else None,
        dp,
    )
    group = get_ep_group()
    assert group is not None  # the caller found the EP mesh
    verdicts = [None] * group.world_size
    dist.all_gather_object(verdicts, bool(mine), group=group.cpu_group)
    return all(verdicts)


def _build_reduce_scatter(mesh, block_order):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        return reduce_scatter.small_reduce_scatter(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank, block_order=block_order
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


def prebuild(vllm_config: Any) -> None:
    """Build the ops and this core's rank tensor, outside any traced region:
    the mesh comes from a collective. A no-op unless the
    ``tpu_collectives.small_all_reduce`` option is on (it is off by default,
    see ``distributed/collectives_options.py``). On a DP x TP layout whose TP
    groups are the chips' core pairs (TP = 2, DP > 1), it also builds the
    TP-pair all-reduce."""
    global _AR_OP, _RS_OP, _RANK, _CORES, _MESH_RANKS, _DP_COMBINE
    global _AR_TP_OP, _TP_RANK, _TP_RANKS
    if _AR_OP is not None or not collectives_options.get(vllm_config).small_all_reduce:
        return
    mesh = build_ep_mesh()
    if mesh is None:
        raise NotImplementedError(
            "tpu_collectives.small_all_reduce builds its kernels over the "
            "expert-parallel mesh, and this deployment has no expert-parallel "
            "group (a dense model or one core). Turn the option off."
        )
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    _CORES = mesh.shape[EP_AXIS_NAME]
    ep_group = get_ep_group()
    assert ep_group is not None  # build_ep_mesh found the group
    _MESH_RANKS = tuple(int(r) for r in ep_group.ranks)
    _RANK = _rank_tensor(ep_mesh_index())
    _AR_OP = _build_all_reduce(mesh)
    # The combine's block r is DP rank r's rows, and when the DP group is the
    # mesh, DP rank r is EP rank r. The mesh core at index i is EP rank
    # ep_rank_order()[i], which is not i when the mesh is in device-id order
    # (ep_mesh._ep_mesh_order), so core i gets block ep_rank_order()[i].
    order = ep_rank_order()
    _RS_OP = _build_reduce_scatter(
        mesh, None if order == tuple(range(_CORES)) else order
    )
    _DP_COMBINE = tuple(int(r) for r in get_dp_group().ranks) == _MESH_RANKS
    tp_all_reduce = tuple(int(r) for r in get_tp_group().ranks) == _MESH_RANKS
    tp = int(get_tp_group().world_size)
    index = ep_mesh_index()
    assert index is not None  # build_ep_mesh found the EP group
    if (
        not tp_all_reduce
        and tp == TP_PAIR_SIZE
        and tp < _CORES
        and _CORES % tp == 0
        and groups_match_mesh(index, tp)
    ):
        _TP_RANKS = tuple(int(r) for r in get_tp_group().ranks)
        _TP_RANK = torch.tensor([index % tp], dtype=torch.int32, device="tpu")
        _AR_TP_OP = _build_tp_all_reduce(mesh, tp)
    logger.info(
        "Small collectives armed | cores=%d | max_rows=%d | "
        "combine_max_rows=%d | one_shot_max_bytes=%d | tp_all_reduce=%s | "
        "dp_combine=%s | tp_pair_all_reduce=%s",
        _CORES,
        MAX_ROWS,
        COMBINE_MAX_ROWS,
        all_reduce.ONE_SHOT_MAX_BYTES,
        tp_all_reduce,
        _DP_COMBINE,
        _AR_TP_OP is not None,
    )


def _rank_tensor(mesh_index: int | None) -> torch.Tensor:
    """This core's mesh index as the kernels' int32 [1] rank operand."""
    return torch.tensor([mesh_index], dtype=torch.int32, device="tpu")


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
    when the group is the mesh or the armed TP pair, ``fallback(x)``
    otherwise."""
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
