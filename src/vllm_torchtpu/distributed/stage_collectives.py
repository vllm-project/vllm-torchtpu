# SPDX-License-Identifier: Apache-2.0
"""TP and PCP device collectives as stage-wide JAX programs.

Under PP x TP x PCP a fused-eager traversal can hold a TP collective (device
assignment = the TP group, e.g. devices [0, 4]) next to a PCP collective
(devices [0, 1]). A torch_tpu that compiles subgroup programs for the
subgroup alone (#3931) resolves the traversal to one of the two and rejects
the other ("Disjoint or conflicting device assignments"). Running the two
groups' collectives as JAX programs over the stage's 2D ``(stage_tp, pcp)``
mesh gives every such op the whole stage as its device assignment, which the
stage's other kernels (PCP attention, EP MoE) already use, so any mix of them
resolves to the stage.
"""

import os
from typing import Any

import jax
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_OPS: dict[tuple, Any] = {}
_STAGE_TP_AXIS = "stage_tp"  # mesh_utils.STAGE_TP_AXIS_NAME
_SIZES: dict[str, int] = {}
_ACTIVE = False


def init_stage_collectives() -> None:
    """Decide once, eagerly (the worker calls it right after the stage table
    is gathered), whether TP/PCP collectives run stage-wide, and build their
    ops: the collective itself is traced by Dynamo, so it may only read this
    state."""
    global _ACTIVE
    from vllm.distributed.parallel_state import (get_pcp_group, get_pp_group,
                                                 get_tp_group)
    from vllm_torchtpu.distributed.mesh_utils import (
        STAGE_TP_AXIS_NAME, _uses_mesh_device_assignment,
        get_stage_device_table)
    active = (int(get_pp_group().world_size) > 1
              and int(get_tp_group().world_size) > 1
              and int(get_pcp_group().world_size) > 1
              and _uses_mesh_device_assignment()
              and get_stage_device_table() is not None)
    override = os.environ.get("VLLM_TPU_STAGE_COLLECTIVES")
    if override is not None:
        active = active and override == "1"
    if active:
        _SIZES[STAGE_TP_AXIS_NAME] = int(get_tp_group().world_size)
        _SIZES["pcp"] = int(get_pcp_group().world_size)
        for axis in (STAGE_TP_AXIS_NAME, "pcp"):
            for ndim in range(1, 5):
                _op("all_reduce", axis, 0, ndim)
                for dim in range(ndim):
                    _op("all_gather", axis, dim, ndim)
                    _op("reduce_scatter", axis, dim, ndim)
    _ACTIVE = active
    if stage_programs_unserializable():
        _compile_stage_programs_unserialized()
    logger.info("Stage-wide TP/PCP collectives: %s (%d ops)", active,
                len(_OPS))


def stage_programs_unserializable() -> bool:
    """True when compiled stage graphs must not round-trip through a
    serialized executable (VLLM_TPU_STAGE_SERIALIZABLE=1 overrides)."""
    return _ACTIVE and os.environ.get("VLLM_TPU_STAGE_SERIALIZABLE") != "1"


def _compile_stage_programs_unserialized() -> None:
    """Make every torch_tpu ``torch.compile`` build unserializable.

    A serializable build copies its executable through pickle while saving
    the AOTAutograd entry, which reloads it in-process; libtpu cannot load a
    program for a stage off the slice origin (tpu_topology.cc:66 in
    GetAddressableDevices). With stage-wide TP/PCP collectives every stage
    graph is such a program. The cost is that these graphs are not cached
    across restarts.
    """
    from torch_tpu._internal.compile._backend import TpuBackend
    if getattr(TpuBackend.__call__, "_vllm_stage_unserialized", False):
        return
    original = TpuBackend.__call__

    def call(self, graph_module, example_inputs, **kwargs):
        options = dict(kwargs.pop("options", None) or {})
        options["serializable"] = False
        return original(self, graph_module, example_inputs, options=options,
                        **kwargs)

    call._vllm_stage_unserialized = True
    TpuBackend.__call__ = call


def _mesh():
    from vllm.distributed.parallel_state import get_pcp_group
    from vllm_torchtpu.distributed.mesh_utils import get_or_create_stage_mesh
    return get_or_create_stage_mesh("pcp", int(get_pcp_group().world_size))


def _axis_for(unique_name: str) -> str | None:
    name = unique_name.split(":")[0]
    if name == "tp":
        return _STAGE_TP_AXIS
    if name == "pcp":
        return "pcp"
    return None


def _op(kind: str, axis: str, dim: int, ndim: int):
    key = (kind, axis, dim, ndim)
    op = _OPS.get(key)
    if op is not None:
        return op
    import jax
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec

    from vllm_torchtpu.distributed.mesh_utils import STAGE_TP_AXIS_NAME
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
    mesh = _mesh()
    # Each device's tensor is its own block of the global view.
    # torch_tpu's placeholders take one mesh axis per dim, so the op is called
    # on x[None, None] with the two leading unit dims split over the mesh.
    spec = PartitionSpec(STAGE_TP_AXIS_NAME, "pcp", *([None] * ndim))
    jdim = dim + 2

    if kind == "all_reduce":

        def body(x):
            return jax.lax.psum(x, axis)
    elif kind == "all_gather":

        def body(x):
            return jax.lax.all_gather(x, axis, axis=jdim, tiled=True)
    elif kind == "reduce_scatter":

        def body(x):
            return jax.lax.psum_scatter(x,
                                        axis,
                                        scatter_dimension=jdim,
                                        tiled=True)
    else:
        raise ValueError(kind)

    def fn(x: jax.Array) -> jax.Array:
        return shard_map(body,
                         mesh=mesh,
                         in_specs=(spec, ),
                         out_specs=spec,
                         check_rep=False)(x)

    name = f"vllm_torchtpu::stage_{kind}_{axis}_d{dim}_n{ndim}"
    op = sharded_jax_op(name,
                        fn,
                        mesh=mesh,
                        input_partition_specs=(spec, ),
                        output_partition_specs=spec)
    _OPS[key] = op
    return op


def stage_collective(kind: str, unique_name: str, x: torch.Tensor,
                     dim: int = 0) -> torch.Tensor | None:
    """Run ``kind`` for the group named ``unique_name`` as a stage-wide JAX
    program, or return None when it does not apply."""
    if not _ACTIVE:
        return None
    axis = _axis_for(unique_name)
    if axis is None:
        return None
    squeeze = x.dim() == 0
    if squeeze:
        x = x.reshape(1)
    if dim < 0:
        dim += x.dim()
    if not x.is_contiguous():
        x = x.contiguous()
    if (kind, axis, dim, x.dim()) not in _OPS:
        return None
    out = torch.ops.vllm_torchtpu.stage_collective(x, kind, axis, dim)
    return out.reshape(()) if squeeze else out


# The JAX op's own fake runs a concrete export, whose output sizes would pin
# the caller's dynamic token dimension under torch.compile; this wrapper's
# fake keeps the sizes symbolic.
@torch.library.custom_op("vllm_torchtpu::stage_collective", mutates_args=())
def _stage_collective_op(x: torch.Tensor, kind: str, axis: str,
                         dim: int) -> torch.Tensor:
    return _OPS[(kind, axis, dim, x.dim())](x[None, None])[0, 0]


@_stage_collective_op.register_fake
def _(x: torch.Tensor, kind: str, axis: str, dim: int) -> torch.Tensor:
    shape = list(x.shape)
    if kind == "all_gather":
        shape[dim] = shape[dim] * _SIZES[axis]
    elif kind == "reduce_scatter":
        shape[dim] = shape[dim] // _SIZES[axis]
    return x.new_empty(shape)
