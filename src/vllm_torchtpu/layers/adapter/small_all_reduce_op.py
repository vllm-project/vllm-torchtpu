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
"""Small TP all-reduces and DP MoE combines through the Pallas kernels in
``kernels/experimental/small_collectives``.

``prebuild`` builds both ops over the expert-parallel mesh at model load. The
device communicator then hands a 2-D bf16 block of at most ``MAX_ROWS`` rows
(an all-reduce) or ``COMBINE_MAX_ROWS`` rows per engine (a DP combine) to a
kernel and anything else to vLLM's collective.

The choice reads the token count, so it is taken only in a trace that the
backbone warmup forks per token bucket (``compilation.shape_variants``). Any
other trace -- the MTP drafter, eager mode, the profiling run -- keeps vLLM's
collective. Both sides of the fork lift the same rank buffer, as the warmup
requires to hand a bucket's executable from one trace to the other.
"""

import jax
import torch
from jax.sharding import PartitionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    get_ep_group,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_reduce,
    reduce_scatter,
)
from vllm_torchtpu.kernels.experimental.small_collectives.common import LANES
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Largest TP all-reduce, in rows, the kernel takes: at H=4096 bf16 on 8 cores
# two-shot beats XLA through 96 rows, loses from 128 (j-e6f25e61, j-de6349e8).
MAX_ROWS = 96
# Largest DP combine, in rows per engine, the reduce-scatter takes: measured
# crossover between 40 and 48 (j-a8d66cde, j-e6f25e61, j-de6349e8). The win
# relies on XLA lowering psum_scatter to a full all-reduce when rows per engine
# is not a multiple of 16, which depends on libtpu: re-measure on a bump.
COMBINE_MAX_ROWS = 40

_AR_OP = None
_RS_OP = None
# This core's mesh index as an int32 [1] device tensor: the kernels take it as
# data (see small_collectives/common.py).
_RANK = None
_CORES = 0
_MESH_RANKS: tuple[int, ...] = ()
# Whether the DP group is the mesh, so its combine may take the kernel.
_DP_COMBINE = False


def _build_all_reduce(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
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


def _build_reduce_scatter(mesh):
    n = mesh.shape[EP_AXIS_NAME]
    spec = PartitionSpec(EP_AXIS_NAME)

    def body(block, rank):
        return reduce_scatter.small_reduce_scatter(
            block, num_cores=n, axis=EP_AXIS_NAME, rank=rank
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


def prebuild() -> None:
    """Build the ops and this core's rank tensor, outside any traced region:
    the mesh comes from a collective. A no-op unless TPU_SMALL_ALL_REDUCE."""
    global _AR_OP, _RS_OP, _RANK, _CORES, _MESH_RANKS, _DP_COMBINE
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
    _AR_OP = _build_all_reduce(mesh)
    _RS_OP = _build_reduce_scatter(mesh)
    # The mesh is rank-ordered (ep_mesh.build_ep_mesh), so when the DP group
    # is the mesh, DP rank r's rows go to mesh core r.
    _DP_COMBINE = tuple(int(r) for r in get_dp_group().ranks) == _MESH_RANKS
    tp_all_reduce = tuple(int(r) for r in get_tp_group().ranks) == _MESH_RANKS
    logger.info(
        "Small collectives armed | cores=%d | max_rows=%d | "
        "combine_max_rows=%d | one_shot_max_bytes=%d | tp_all_reduce=%s | "
        "dp_combine=%s | "
        "backbone warmup traces only; the MTP drafter and eager steps keep "
        "vLLM's collectives",
        _CORES,
        MAX_ROWS,
        COMBINE_MAX_ROWS,
        all_reduce.ONE_SHOT_MAX_BYTES,
        tp_all_reduce,
        _DP_COMBINE,
    )


def _fits(x: torch.Tensor, width_multiple: int) -> bool:
    """Static checks: whether this call site can take a kernel at all."""
    return (
        torch.compiler.is_compiling()
        and shape_variants.in_warmup()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape[1] % width_multiple == 0
    )


def retain(out: torch.Tensor) -> torch.Tensor:
    """``out`` plus a zero read from the rank buffer, for a token-count branch
    that skips the kernel while its other side calls it: the warmup hands a
    bucket over only between traces that lift the same buffers. A read whose
    result is dead would be eliminated, so the zero rides on ``out``."""
    if _RANK is None or not shape_variants.in_warmup():
        return out
    value = _RANK[0].to(out.dtype)
    return out + (value - value)


def all_reduce_or(x: torch.Tensor, ranks, fallback) -> torch.Tensor:
    """The sum of ``x`` over the group ``ranks``: the kernel for a small block
    when the group is the mesh, ``fallback(x)`` otherwise."""
    if _AR_OP is None or tuple(ranks) != _MESH_RANKS:
        return fallback(x)
    if not _fits(x, _CORES * LANES):
        return fallback(x)
    if x.shape[0] <= MAX_ROWS:
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
