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
"""Small decode-step collectives through Pallas kernels instead of XLA's.

At decode a collective moves a few rows, so XLA's collective is bound by its
fixed cost rather than by bandwidth. Two torch custom ops over the
expert-parallel mesh replace it for small blocks:

- ``all_reduce`` (kernels/collectives/small_all_reduce.py): the all-reduce of
  a group whose ranks are the mesh's. Under TP+EP that is the tensor-parallel
  group: the output projections, the MoE output and the embedding.
- ``reduce_scatter`` (kernels/collectives/small_gather_scatter.py): the MoE
  combine of a group whose ranks are the mesh's, in the same order. Under
  DP+EP with TP=1 that is the data-parallel group (the combine after the
  grouped-matmul or dense-expert MoE); under MoE sequence parallelism it is
  the expert-parallel group. A fused EP kernel does its own combine and never
  reaches this.

Both ops are built once before any trace (``prebuild``, from the FP8 MoE
weight-load path) and only looked up by the traced forward. A call takes a
kernel when the block is 2-D bf16, its width splits into whole lane tiles and
each rank holds at most TPU_SMALL_ALL_REDUCE_MAX_ROWS rows; otherwise it keeps
vLLM's collective.

The row check reads the tensor's shape, so the backbone warmup gives the
buckets on each side their own trace (compilation/shape_variants). A module
compiled outside that warmup keeps vLLM's collectives, since a shape guard
there would fail the buckets on the other side. The warmup hands a bucket's
executable between traces only when their inputs agree, so the fallback
reads the same rank buffers as the kernel (``_keep_buffers_live``).
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
    get_ep_group,
    mesh_has_adjacent_chip_mates,
)
from vllm_torchtpu.kernels.collectives import small_all_reduce as sar
from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ALL_REDUCE_OP_NAME = "pallas::small_all_reduce"
REDUCE_SCATTER_OP_NAME = "pallas::small_reduce_scatter"

_ALL_REDUCE_OP: Any = None
_REDUCE_SCATTER_OP: Any = None
_NUM_CORES = 0
# The expert-parallel group's ranks, in rank order: the groups whose ranks
# match may take the kernels.
_MESH_RANKS: tuple[int, ...] = ()
# This core's index in the mesh (ordered by device id, not by rank), as a
# device tensor: the kernels need it as data, see the kernel's note on the
# torch bridge and `lax.axis_index`.
_MESH_INDEX: Any = None
# This core's expert-parallel rank and the mesh core of every rank: the
# reduce-scatter's blocks are in rank order, its peers in mesh order.
_RANK_IN_MESH: Any = None
_CORE_OF_RANK: Any = None


def _build_all_reduce(mesh, form):
    num_cores = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, mesh_index):
        return sar.small_all_reduce(
            block, num_cores=num_cores, axis=EP_AXIS_NAME, rank=mesh_index, form=form
        )

    def fn(x: jax.Array, mesh_index: jax.Array) -> jax.Array:
        # x is the mesh-wide [num_cores * rows, hidden]; each core's block is
        # its shard
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )(x, mesh_index)

    # torch_tpu types the partition specs as tuples of axis names, but accepts
    # jax PartitionSpecs.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        ALL_REDUCE_OP_NAME,
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec),
        output_partition_specs=spec,
    )

    # Replaces the exporting fake jax_op installs, which would
    # specialise the caller's dynamic token dimension to the traced value.
    def fake(x, mesh_index):
        return torch.empty_like(x)

    op.register_fake(fake)
    return op


def _build_reduce_scatter(mesh, hierarchical: bool = False):
    """Each rank's [num_cores * rows, hidden] partial sums (block r = rank r's
    rows) -> its own [rows, hidden] sum."""
    num_cores = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)
    mesh_devices = getattr(mesh, "devices", None)
    use_hierarchical = (
        hierarchical
        and mesh_devices is not None
        and mesh_has_adjacent_chip_mates(mesh_devices.flat)
    )

    def body(block, mesh_index, rank_in_mesh, core_of_rank):
        return sgs.small_reduce_scatter(
            block,
            num_cores=num_cores,
            axis=EP_AXIS_NAME,
            rank=mesh_index,
            my_rank=rank_in_mesh,
            core_of_rank=core_of_rank,
            hierarchical=use_hierarchical,
        )

    def fn(
        x: jax.Array,
        mesh_index: jax.Array,
        rank_in_mesh: jax.Array,
        core_of_rank: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(x, mesh_index, rank_in_mesh, core_of_rank)

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        REDUCE_SCATTER_OP_NAME,
        fn,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec, spec),
        output_partition_specs=spec,
    )

    def fake(x, mesh_index, rank_in_mesh, core_of_rank):
        # in terms of the operands' shapes, symbolic or not: the caller's
        # token dimension stays dynamic through the op
        return x.new_empty((x.shape[0] // num_cores, x.shape[1]))

    op.register_fake(fake)
    return op


def prebuild(device: torch.device) -> bool:
    """Build both ops and this core's rank tensors, outside any traced region.

    Every rank must call this at the same point: resolving the mesh gathers
    over the expert-parallel group. Returns whether the kernels are armed.
    """
    global _ALL_REDUCE_OP, _REDUCE_SCATTER_OP, _NUM_CORES, _MESH_RANKS
    global _MESH_INDEX, _RANK_IN_MESH, _CORE_OF_RANK
    if _ALL_REDUCE_OP is not None:
        return True
    if envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS <= 0:
        return False
    mesh = build_ep_mesh()
    group = get_ep_group()
    if mesh is None or group is None:
        logger.info("Small all-reduce not armed: no expert-parallel group")
        return False
    num_cores = mesh.shape[EP_AXIS_NAME]
    rank_of_core = list(ep_rank_order() or range(num_cores))
    core_of_rank = [rank_of_core.index(r) for r in range(num_cores)]
    form = (envs.TPU_SMALL_ALL_REDUCE_FORM or "").lower() or None
    requested_hierarchical_rs = envs.TPU_SMALL_REDUCE_SCATTER_HIERARCHICAL
    mesh_devices = getattr(mesh, "devices", None)
    use_hierarchical_rs = (
        requested_hierarchical_rs
        and mesh_devices is not None
        and mesh_has_adjacent_chip_mates(mesh_devices.flat)
    )
    if requested_hierarchical_rs and not use_hierarchical_rs:
        logger.warning_once(
            "TPU_SMALL_REDUCE_SCATTER_HIERARCHICAL=1 requested, but the EP mesh "
            "over %d cores does not pair adjacent dual-core chip mates across a "
            "power-of-two chip count >= 2; falling back to one-stage "
            "reduce-scatter.",
            num_cores,
        )

    _MESH_INDEX = torch.tensor([ep_mesh_index()], dtype=torch.int32, device=device)
    _RANK_IN_MESH = torch.tensor(
        [int(group.rank_in_group)], dtype=torch.int32, device=device
    )
    _CORE_OF_RANK = torch.tensor(core_of_rank, dtype=torch.int32, device=device)
    _MESH_RANKS = tuple(int(r) for r in group.ranks)
    _NUM_CORES = num_cores
    _REDUCE_SCATTER_OP = _build_reduce_scatter(mesh, hierarchical=use_hierarchical_rs)
    _ALL_REDUCE_OP = _build_all_reduce(mesh, form)
    logger.info(
        "Small all-reduce armed: up to %d rows per rank over %d cores, form %s, "
        "hierarchical_reduce_scatter=%s",
        envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS,
        num_cores,
        form or f"by size (one-shot up to {sar.ONE_SHOT_MAX_BYTES} bytes per core)",
        use_hierarchical_rs,
    )
    return True


def _rows_fit(rows, ranks_per_row_block: int = 1) -> bool:
    """Whether a block of `rows` rows, holding `ranks_per_row_block` ranks'
    rows, takes the kernel. Outside the backbone warmup a compiled module
    never reads the row count."""
    if torch.compiler.is_compiling() and shape_variants.aot_cache_tag() is None:
        return False
    return rows <= ranks_per_row_block * envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS


def _block_fits(x: torch.Tensor, lane_multiple: int) -> bool:
    return (
        x.dim() == 2 and x.dtype == torch.bfloat16 and x.shape[1] % lane_multiple == 0
    )


def _keep_buffers_live(out: torch.Tensor, buffers: Sequence[torch.Tensor]):
    """`out` plus a zero read from each buffer, so a trace that does not call
    the kernel lifts the same buffers as one that does (see module doc)."""
    if not torch.compiler.is_compiling():
        return out
    for buf in buffers:
        value = buf.reshape(-1)[0].to(out.dtype)
        out = out + (value - value)
    return out


def all_reduce(
    x: torch.Tensor,
    group_ranks: Sequence[int],
    fallback: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """The sum of x over a group: the kernel for a small block when the group
    spans the mesh, `fallback(x)` (vLLM's all-reduce) otherwise."""
    # Keep this compiled-forward path free of logging and global writes.
    if (
        _ALL_REDUCE_OP is None
        or sorted(group_ranks) != sorted(_MESH_RANKS)
        or not _block_fits(x, _NUM_CORES * sar.LANES)
    ):
        return fallback(x)
    if _rows_fit(x.shape[0]):
        return _ALL_REDUCE_OP(x, _MESH_INDEX)
    return _keep_buffers_live(fallback(x), (_MESH_INDEX,))


def reduce_scatter(
    x: torch.Tensor,
    group_ranks: Sequence[int],
    fallback: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """The MoE combine: each rank's [num_cores * rows, hidden] partial sums,
    block r for the group's rank r, to its own [rows, hidden] sum. The kernel
    for a small block when the group is the mesh in rank order,
    `fallback(x)` (vLLM's reduce-scatter along dim 0) otherwise."""
    if (
        _REDUCE_SCATTER_OP is None
        or tuple(group_ranks) != _MESH_RANKS
        or not _block_fits(x, sgs.LANE)
    ):
        return fallback(x)
    buffers = (_MESH_INDEX, _RANK_IN_MESH, _CORE_OF_RANK)
    if _rows_fit(x.shape[0], _NUM_CORES):
        return _REDUCE_SCATTER_OP(x, *buffers)
    return _keep_buffers_live(fallback(x), buffers)
