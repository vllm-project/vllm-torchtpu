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
"""Token-dimension collectives for GLM-5.3-Flash sequence-parallel prefill.

vLLM's ``GroupCoordinator.all_gather`` allocates its output with
``torch.empty(input_size[0] * world_size, ...)``; under ``torch.compile`` the
token count is symbolic and TorchTPU's export rejects the allocation
("SymIntArrayRef expected to contain only concrete integers"), so the compiled
backbone cannot trace the SP prefill path through it. These ops run the gather
and the reduce-scatter as one SPMD JAX program over a TP-ordered mesh instead,
with a fake that only derives shapes. Same approach as
``models/vllm/kimi_k3/collective_ops.py``, minus K3's ring tables: a plain
tiled ``all_gather`` / ``psum_scatter`` over a TP-ordered mesh is the TP
collective.

Eager calls keep vLLM's own collectives. Inside a trace, when no TP mesh is
available (see `_mesh`), an all-reduce formulation that never allocates from
a symbolic size serves instead.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

import jax
import torch
from jax.sharding import PartitionSpec as P
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce

from vllm_torchtpu.logger import init_logger

_AXIS = "glm53_sp"
logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _mesh() -> Any | None:
    """A one-axis mesh over the TP group in TP rank order, or None.

    `get_pallas_mesh` numbers mesh positions by torch.distributed rank and
    spans the whole world, so it is the TP-ordered mesh exactly when the TP
    group is the whole world in rank order (no PP/DP/EP split of the world).
    Otherwise the all-reduce formulations below serve the trace.
    """
    tp = get_tp_group()
    if tp.world_size <= 1:
        return None
    import torch.distributed as dist

    world_size = int(dist.get_world_size())
    if tuple(int(r) for r in tp.ranks) != tuple(range(world_size)):
        logger.warning_once(
            "GLM-5.3 SP collectives: the TP group %s is not the whole world "
            "of %d ranks in order; using all-reduce.",
            tuple(tp.ranks),
            world_size,
        )
        return None
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(axis_names=(_AXIS,))


def has_tp_mesh() -> bool:
    """Whether the world-ordered TP JAX mesh is available for SP collectives."""
    return _mesh() is not None


@lru_cache(maxsize=4)
def _build(gather: bool, ndim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    from torch_tpu._internal.pallas import jax_op

    mesh = _mesh()
    assert mesh is not None
    world = mesh.devices.size
    spec = P(_AXIS, *([None] * (ndim - 1)))

    def collective(x: jax.Array) -> jax.Array:
        def local(part):
            if gather:
                return jax.lax.all_gather(part, _AXIS, axis=0, tiled=True)
            return jax.lax.psum_scatter(part, _AXIS, scatter_dimension=0, tiled=True)

        return jax.shard_map(
            local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
        )(x)

    op = cast(Any, jax_op)(
        f"pallas::glm53_token_{'ag' if gather else 'rs'}_{ndim}d",
        collective,
        mesh=mesh,
        input_partition_specs=(spec,),
        # One spec per output tensor so torch_tpu sizes each rank's shard.
        output_partition_specs=(spec,),
    )

    def fake(x):
        rows = x.shape[0] * world if gather else x.shape[0] // world
        return x.new_empty((rows, *x.shape[1:]))

    op.register_fake(fake)
    return op


@lru_cache(maxsize=2)
def _build_vocab_gather(ndim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    from torch_tpu._internal.pallas import jax_op

    mesh = _mesh()
    assert mesh is not None
    world = mesh.devices.size
    spec = P(*([None] * (ndim - 1)), _AXIS)

    def collective(x: jax.Array) -> jax.Array:
        def local(part):
            return jax.lax.all_gather(part, _AXIS, axis=ndim - 1, tiled=True)

        return jax.shard_map(
            local, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
        )(x)

    op = cast(Any, jax_op)(
        f"pallas::glm53_vocab_ag_{ndim}d",
        collective,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=(spec,),
    )

    def fake(x):
        return x.new_empty((*x.shape[:-1], x.shape[-1] * world))

    op.register_fake(fake)
    return op


# Built by `prepare()` before tracing; the traced code only reads these dicts,
# so Dynamo never steps into mesh construction (which calls pybind device
# queries).
_OPS: dict[tuple[bool, int], Callable[[torch.Tensor], torch.Tensor]] = {}
_VOCAB_OPS: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}


def prepare() -> None:
    """Build the mesh and ops eagerly.

    The mesh needs a collective, which must not run for the first time inside
    a traced region.
    """
    if _mesh() is not None:
        for gather in (True, False):
            for ndim in (2, 3):
                _OPS[(gather, ndim)] = _build(gather, ndim)
        for ndim in (2, 3):
            _VOCAB_OPS[ndim] = _build_vocab_gather(ndim)


def _gather_by_all_reduce(x: torch.Tensor) -> torch.Tensor:
    # Place the local shard at this rank's offset among zero shards and sum:
    # every tensor is derived from `x`, so no factory call sees a symbolic size.
    tp = get_tp_group()
    zero = torch.zeros_like(x)
    parts = [x if r == tp.rank_in_group else zero for r in range(tp.world_size)]
    return tensor_model_parallel_all_reduce(torch.cat(parts, dim=0))


def _vocab_gather_by_all_reduce(x: torch.Tensor) -> torch.Tensor:
    tp = get_tp_group()
    zero = torch.zeros_like(x)
    parts = [x if r == tp.rank_in_group else zero for r in range(tp.world_size)]
    return tensor_model_parallel_all_reduce(torch.cat(parts, dim=-1))


def _scatter_by_all_reduce(x: torch.Tensor) -> torch.Tensor:
    tp = get_tp_group()
    total = tensor_model_parallel_all_reduce(x)
    return total.unflatten(0, (tp.world_size, -1))[tp.rank_in_group]


def token_all_gather(x: torch.Tensor) -> torch.Tensor:
    """All-gather token shards along dim 0 over the TP group."""
    if not torch.compiler.is_compiling():
        return get_tp_group().all_gather(x, dim=0)
    op = _OPS.get((True, x.dim()))
    return op(x) if op is not None else _gather_by_all_reduce(x)


def vocab_all_gather(x: torch.Tensor) -> torch.Tensor:
    """All-gather vocabulary shards along the last dim over the TP group."""
    if not torch.compiler.is_compiling():
        return get_tp_group().all_gather(x, dim=-1)
    op = _VOCAB_OPS.get(x.dim())
    return op(x) if op is not None else _vocab_gather_by_all_reduce(x)


def token_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Sum partial results over the TP group and keep this rank's token shard."""
    if not torch.compiler.is_compiling():
        return get_tp_group().reduce_scatter(x, dim=0)
    op = _OPS.get((False, x.dim()))
    return op(x) if op is not None else _scatter_by_all_reduce(x)
