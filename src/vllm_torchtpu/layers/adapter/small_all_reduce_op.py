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
"""Small TP all-reduces through the Pallas two-shot kernel.

With TPU_SMALL_ALL_REDUCE, ``prebuild`` builds one op over the expert-parallel
mesh at model load; ``TpuDeviceCommunicator.all_reduce`` hands it every 2-D
bf16 TP all-reduce whose width splits over the TP cores. The op sums over this
rank's TP group only, so it serves TP groups smaller than the EP mesh (TP8 x
DP2 with EP16): the kernel addresses the group's cores through a per-core peer
table, and torch_tpu wants every Pallas kernel of a module on the one EP mesh.

The kernel/XLA choice is taken inside JAX, where each token bucket is exported
at its concrete shape: up to ``max_rows()`` rows the kernel, above it XLA's
all-reduce over the same groups. No branch on the token count reaches the
torch trace, so the backbone and the drafter take the same op.
"""

import os
import jax
import torch
from jax import lax
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_token_replica_groups,
)
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Largest block, in rows, the kernel takes. On tpu7x 2x2x2 (two TP8 groups in
# a 16-core program), bf16 [rows, 7168]: kernel vs XLA 9.9/22.2 us at 8 rows,
# 11.8/23.7 at 16, 16.2/25.1 at 32, 25.3/29.4 at 64.
MAX_ROWS = 64

_OP = None
# This core's position in its TP group (int32 [1]) and the mesh index of each
# member (int32 [cores]), as device tensors: the kernel takes both as data.
_RANK = None
_PEERS = None
_CORES = 0
_TP_RANKS: tuple[int, ...] = ()


# Default row cap by TP group size, under the DSv4 served LIBTPU flags: the
# kernel saves 4-5 us per call up to 32 rows at TP8 but is slower at 64; at TP4
# it wins only up to 16 (close-gaps decode-step-glue / mixed-step-collectives).
DEFAULT_MAX_ROWS = {8: 32}
DEFAULT_MAX_ROWS_SMALL_TP = 16


def max_rows(cores: int = 0) -> int:
    """The kernel's row cap: TPU_SMALL_ALL_REDUCE_MAX_ROWS, else by the TP
    group's core count (``cores``, or this core's group once prebuilt), else
    MAX_ROWS."""
    if envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS:
        return envs.TPU_SMALL_ALL_REDUCE_MAX_ROWS
    cores = cores or _CORES
    if not cores:
        return MAX_ROWS
    return DEFAULT_MAX_ROWS.get(
        cores, MAX_ROWS if cores > 8 else DEFAULT_MAX_ROWS_SMALL_TP
    )


def group_all_reduce(groups: tuple[tuple[int, ...], ...], cap: int, axis: str):
    """The shard_map body: body(x [rows, hidden], rank int32 [1], peers int32
    [cores]) -> the sum of x over the core's group, by the kernel up to ``cap``
    rows and by XLA above."""
    cores = len(groups[0])
    index_groups = [list(g) for g in groups]
    form, chunks, skip = all_reduce.group_form(groups)

    def body(block, rank, peers):
        if block.shape[0] > cap:
            return lax.psum(block, axis, axis_index_groups=index_groups)
        return all_reduce.small_all_reduce(
            block,
            num_cores=cores,
            axis=axis,
            rank=rank,
            peers=peers,
            form=form,
            chunks=chunks,
            skip_device_barrier=skip,
        )

    return body


def _build_op(mesh, groups: tuple[tuple[int, ...], ...], cap: int):
    """The torch op over ``mesh``: fn(x, rank, peers) per core, see
    ``group_all_reduce``."""
    spec = PartitionSpec(EP_AXIS_NAME)
    body = group_all_reduce(groups, cap, EP_AXIS_NAME)

    # Every operand annotated jax.Array: torch_tpu infers static argnums from
    # the signature and refuses an unannotated one.
    def fn(x: jax.Array, rank: jax.Array, peers: jax.Array) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )(x, rank, peers)

    op = jax_op(
        "pallas::small_tp_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 3,
        output_partition_specs=spec,
    )
    # The default fake exports fn at the operands' shapes, which would pin the
    # caller's dynamic token dimension; the output is the input's shape.
    op.register_fake(lambda x, rank, peers: torch.empty_like(x))
    return op


def prebuild() -> None:
    """Build the op and this core's tables, outside any traced region (the
    groups come from a collective). A no-op unless TPU_SMALL_ALL_REDUCE."""
    global _OP, _RANK, _PEERS, _CORES, _TP_RANKS
    if not envs.TPU_SMALL_ALL_REDUCE or _OP is not None:
        return
    mesh = build_ep_mesh()
    if mesh is None and "TPU_SMALL_ALL_REDUCE" not in os.environ:
        return  # on by default; only an explicit request needs an EP mesh
    if mesh is None:
        raise NotImplementedError(
            "TPU_SMALL_ALL_REDUCE=1 runs its kernel over the expert-parallel "
            "mesh, and this deployment has no expert-parallel group. Unset "
            "TPU_SMALL_ALL_REDUCE."
        )
    from vllm.distributed.parallel_state import get_tp_group

    # The TP groups in mesh coordinates, the same tuple on every rank.
    groups = ep_token_replica_groups()
    me = ep_mesh_index()
    (group,) = [g for g in groups if me in g]
    _CORES = len(group)
    _TP_RANKS = tuple(int(r) for r in get_tp_group().ranks)
    _RANK = torch.tensor([group.index(me)], dtype=torch.int32, device="tpu")
    _PEERS = torch.tensor(group, dtype=torch.int32, device="tpu")
    _OP = _build_op(mesh, groups, max_rows())
    logger.info(
        "Small TP all-reduce armed | cores=%d | groups=%s | max_rows=%d | "
        "one_shot_max_bytes=%d | form=%s",
        _CORES,
        groups,
        max_rows(),
        all_reduce.ONE_SHOT_MAX_BYTES,
        all_reduce.group_form(groups),
    )


def all_reduce_or(x: torch.Tensor, ranks, fallback) -> torch.Tensor:
    """The sum of ``x`` over the group ``ranks``: the op for a 2-D bf16 block
    of the TP group in a compiled trace, ``fallback(x)`` otherwise."""
    if _OP is None or tuple(ranks) != _TP_RANKS:
        return fallback(x)
    if (
        not torch.compiler.is_compiling()
        or x.dim() != 2
        or x.dtype != torch.bfloat16
        or not all_reduce.supports(1, x.shape[1], _CORES)
    ):
        return fallback(x)
    return _OP(x, _RANK, _PEERS)
