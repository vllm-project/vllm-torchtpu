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
"""The decode step's all-reduces through the small all-reduce kernel
(kernels/collectives/small_all_reduce.py) instead of XLA's all-reduce.

One torch custom op over the expert-parallel mesh (the tensor-parallel group
of the served layouts), built once before any trace (`prebuild`, from the
weight-load path) and only looked up by the traced forward. `all_reduce`
takes the kernel for a 2-D bf16 block of at most TPU_SMALL_ALL_REDUCE_MAX_ROWS
rows whose width splits into whole lane tiles per core, and vLLM's all-reduce
otherwise. The decision reads the tensor's shape, so the compile flow forks
its traces by token bucket and every call of one trace takes the same form.
"""
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (EP_AXIS_NAME, build_ep_mesh,
                                               ep_mesh_index, ep_rank_order)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.collectives import small_all_reduce as sar
from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

OP_NAME = "pallas::small_all_reduce"

_OP: Any = None
# This core's index in the mesh (ordered by device id, not by rank), as a
# device tensor: the kernel needs it as data, see the kernel's note on the
# torch bridge and `lax.axis_index`.
_RANK: Any = None
_CORES = 0
# Whether the tensor-parallel group is the mesh (the same ranks as the
# expert-parallel group): only then may the TP reductions take the kernel.
_TP_IS_MESH = False
# the data-parallel combine's reduce-scatter: the op and its per-rank operands (this rank and the rank-to-core table)
_RS_OP = None
_MY_RANK = None
_CORE_OF_RANK = None
# whether the data-parallel combine takes the kernel on this deployment (the
# data-parallel group is the mesh); decided at prebuild, read by the traced
# forward, so both sides of the token-count fork see the same answer
_DP_COMBINE = False


def _build(mesh, form=None):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        return sar.small_all_reduce(block,
                                    num_cores=n,
                                    axis=EP_AXIS_NAME,
                                    rank=rank,
                                    form=form)

    def fn(x: jax.Array, rank: jax.Array) -> jax.Array:
        # x is the mesh-wide [n * rows, hidden]; each core's block is its shard
        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(spec, spec),
                             out_specs=spec,
                             check_vma=False)(x, rank)

    op = sharded_jax_op(OP_NAME,
                        fn,
                        mesh=mesh,
                        input_partition_specs=(spec, spec),
                        output_partition_specs=spec)

    def _fake(x, rank):
        return torch.empty_like(x)

    op.register_fake(_fake)
    return op


def _build_reduce_scatter(mesh):
    """The data-parallel combine: each rank's [n * rows, hidden] block of
    partial sums (block r = rank r's rows) -> its own rank's [rows, hidden]
    sum. Rows are in rank order, cores in mesh order; the kernel takes the
    rank-to-core table."""
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank, my_rank, core_of_rank):
        return sgs.small_reduce_scatter(block,
                                        num_cores=n,
                                        axis=EP_AXIS_NAME,
                                        rank=rank,
                                        my_rank=my_rank,
                                        core_of_rank=core_of_rank)

    def fn(x: jax.Array, rank: jax.Array, my_rank: jax.Array,
           core_of_rank: jax.Array) -> jax.Array:
        return jax.shard_map(body,
                             mesh=mesh,
                             in_specs=(spec, spec, spec, spec),
                             out_specs=spec,
                             check_vma=False)(x, rank, my_rank, core_of_rank)

    def fake(x, rank, my_rank, core_of_rank):
        # in terms of the operands' shapes, symbolic or not: the caller's
        # batch dimension stays dynamic through the op
        return torch.empty((x.shape[0] // n, x.shape[1]),
                           dtype=x.dtype,
                           device=x.device)

    return sharded_jax_op(OP_NAME + "_dp_reduce_scatter",
                          fn,
                          mesh=mesh,
                          input_partition_specs=(spec, spec, spec, spec),
                          output_partition_specs=spec,
                          fake=fake)


def prebuild() -> bool:
    """Build the op and this core's rank tensor, outside any traced region.
    Returns whether the path is armed; logs the decision once."""
    global _OP, _RANK, _CORES, _TP_IS_MESH
    if _OP is not None:
        return True
    cap = envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS
    if cap <= 0:
        return False
    mesh = build_ep_mesh()
    if mesh is None:
        logger.info("small all-reduce refused: no expert-parallel group")
        return False
    n = mesh.shape[EP_AXIS_NAME]
    _RANK = torch.tensor([ep_mesh_index()],
                         dtype=torch.int32,
                         device=torch.device("tpu"))
    form = envs.TPU_SMALL_ALL_REDUCE_FORM or None
    _OP = _build(mesh, form)
    _CORES = n
    from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                                 get_tp_group)
    _TP_IS_MESH = tuple(get_tp_group().ranks) == tuple(get_ep_group().ranks)
    # the data-parallel combine's reduce-scatter: this rank and the mesh
    # core of every rank (ep_rank_order gives the rank at each mesh index)
    global _RS_OP, _MY_RANK, _CORE_OF_RANK, _DP_COMBINE
    order = ep_rank_order()
    rank_of_core = list(order) if order is not None else list(range(n))
    core_of_rank = [rank_of_core.index(r) for r in range(n)]
    _MY_RANK = torch.tensor([int(get_dp_group().rank_in_group)],
                            dtype=torch.int32,
                            device=torch.device("tpu"))
    _CORE_OF_RANK = torch.tensor(core_of_rank,
                                 dtype=torch.int32,
                                 device=torch.device("tpu"))
    _RS_OP = _build_reduce_scatter(mesh)
    from vllm.distributed.parallel_state import get_ep_group
    _DP_COMBINE = (int(get_dp_group().world_size) == n and tuple(
        get_dp_group().ranks) == tuple(get_ep_group().ranks))
    logger.info(
        "small all-reduce armed (cap %d rows, %d cores, mesh index %d, "
        "tensor-parallel group is the mesh: %s): form %s", cap, n,
        int(_RANK[0]), _TP_IS_MESH, form
        or "by block size (one-shot up to %d bytes per core, two-shot above)" %
        sar.ONE_SHOT_MAX_BYTES)
    logger.info(
        "data-parallel combine reduce-scatter %s (this rank %d, "
        "mesh core of each rank %s)",
        "armed" if _DP_COMBINE else "built, not this deployment's combine",
        int(_MY_RANK[0]), core_of_rank)
    return True


def eligible(x: torch.Tensor, group_size: int) -> bool:
    return (_OP is not None and group_size == _CORES and x.dim() == 2
            and x.shape[0] <= envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS
            and x.shape[1] % (_CORES * sar.LANES) == 0
            and x.dtype == torch.bfloat16)


def all_reduce_tp(x: torch.Tensor) -> torch.Tensor:
    """The tensor-parallel all-reduce of x: the kernel for a small block when
    the tensor-parallel group is the mesh, vLLM's all-reduce otherwise."""
    from vllm.distributed import (get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_reduce)
    if _TP_IS_MESH and eligible(x, get_tensor_model_parallel_world_size()):
        return _OP(x, _RANK)
    return tensor_model_parallel_all_reduce(x)


def all_reduce_ep(x: torch.Tensor) -> torch.Tensor:
    """The expert-parallel group's all-reduce of x (the MoE combine)."""
    from vllm.distributed.parallel_state import get_ep_group
    group = get_ep_group()
    if eligible(x, int(group.world_size)):
        return _OP(x, _RANK)
    return group.all_reduce(x)


def dp_combine(x: torch.Tensor):
    """The data-parallel MoE combine (a reduce-scatter of the gathered rows'
    partial sums, block r = rank r's rows) through the small reduce-scatter
    kernel, when the data-parallel group is the expert-parallel group (the
    mesh) and the block is small enough; None when the caller should keep
    vLLM's reduce-scatter. The kernel takes the rank-to-core table, so the
    rows land by rank whatever the mesh order."""
    # Keep this compiled-forward path free of logging and global writes.
    # The cap counts output rows per rank. The gathered input has
    # _CORES times as many rows, all held in the kernel's VMEM scratch.
    if (_RS_OP is None or not _DP_COMBINE or x.dim() != 2
            or x.dtype != torch.bfloat16 or x.shape[1] % (_CORES * sar.LANES)
            or x.shape[0] > _CORES * envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS):
        return None
    return _RS_OP(x, _RANK, _MY_RANK, _CORE_OF_RANK)


def zero_from_rank():
    """An int32 zero computed from the rank tensor, for a trace that does not
    otherwise read it: the compile flow hands one bucket's executable to
    another only when both traces lifted the same buffers, so every trace of
    the decoder layer folds this into its positions (the compiler folds the
    multiply by zero), as the layer already does with its own rank buffer."""
    if _RANK is None:
        return 0
    return (_RANK * 0)[0]


def retain_dp_combine_inputs(out: torch.Tensor) -> torch.Tensor:
    """The reduce-scatter's operand buffers read into a live output, for a
    route that never calls the kernel (the fused-EP route above the token
    threshold, vLLM's reduce-scatter when the block is not eligible): the
    compile flow hands a bucket's executable between the two traces of the
    token-count fork only when both lifted the same buffers, and a read whose
    result is dead (a zero on the positions in a layer that does not use
    them) leaves the trace with its buffer. Same form as the fused-EP route's
    own retain: the output plus a zero computed from each buffer."""
    if _RS_OP is None or _RANK is None or not _DP_COMBINE:
        # not this deployment's combine (the tensor-parallel line): nothing to
        # keep in step, and a read here would lift buffers the other route
        # never touches
        return out
    marker = None
    for buf in (_RANK, _MY_RANK, _CORE_OF_RANK):
        value = buf.reshape(-1)[0].to(out.dtype)
        zero = value - value
        marker = zero if marker is None else marker + zero
    return out + marker
