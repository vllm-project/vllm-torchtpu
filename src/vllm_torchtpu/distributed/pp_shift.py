# SPDX-License-Identifier: Apache-2.0
"""Hand one pipeline stage's activations to the next over ICI.

The op is an XLA collective-permute over a mesh of the slice's devices: every
stage's block lands on the stage after it, and the last stage's block lands on
the first stage, which ignores it, so the permutation is one closed cycle. A
stage of several devices (a context-parallel group) forms one cycle per
device column: each device hands its rows to the device holding the same rows
in the next stage. Every device must call it at the same point in its launch
sequence; ``pp_wave`` arranges that.
"""
import jax
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from vllm.logger import init_logger

from vllm_torchtpu.distributed.mesh_utils import (StageDeviceTable,
                                                  _uses_mesh_device_assignment,
                                                  gather_stage_device_table)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

logger = init_logger(__name__)

AXIS = "pp"
_MESH: Mesh | None = None
_TABLE: StageDeviceTable | None = None
_OPS: dict[tuple, object] = {}


def get_or_create_pp_mesh() -> Mesh:
    """A JAX mesh over the slice's devices in world-rank order.

    torch_tpu numbers the slice's devices by world rank, so mesh position r
    is rank r; ``pipeline_permutation`` maps stages onto the positions.
    """
    global _MESH, _TABLE
    if _MESH is not None:
        return _MESH
    _TABLE = gather_stage_device_table()
    devices_by_id = {int(device.id): device for device in jax.devices()}
    mesh_dev_ids = (list(_TABLE.device_ids) if _uses_mesh_device_assignment()
                    else list(range(len(_TABLE.stages))))
    _MESH = Mesh(np.asarray([devices_by_id[d] for d in mesh_dev_ids]),
                 axis_names=(AXIS, ))
    logger.info(
        "PP hand-off mesh (rank order): %s",
        ", ".join(f"stage{s}/col{c}->rank{r}/dev{d}"
                  for r, (s, c, d) in enumerate(
                      zip(_TABLE.stages, _TABLE.columns, _TABLE.device_ids))))
    return _MESH


def pipeline_permutation(
        rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, int], ...]:
    """Source/target rank pairs of the hand-off for stage-major ``rows`` of
    ranks: stage s's column j sends to stage s+1's column j, the last stage
    back to stage 0, one closed cycle per column."""
    size = len(rows)
    return tuple((rows[s][j], rows[(s + 1) % size][j])
                 for j in range(len(rows[0])) for s in range(size))


def pp_permute_op(mesh: Mesh):
    """Torch op ``x_out = pp_permute(x)``: one collective-permute carrying a
    [tokens, hidden] tensor one stage forward around its column's cycle,
    routed by mesh position."""
    key = ("permute", ) + tuple(int(d.id) for d in mesh.devices)
    op = _OPS.get(key)
    if op is not None:
        return op
    assert _TABLE is not None
    perm = list(pipeline_permutation(_TABLE.rows()))
    spec = PartitionSpec(AXIS)

    def shift(x: jax.Array) -> jax.Array:
        return shard_map(lambda xb: jax.lax.ppermute(xb, AXIS, perm),
                         mesh=mesh,
                         in_specs=(spec, ),
                         out_specs=spec,
                         check_rep=False)(x)

    op = sharded_jax_op("vllm_torchtpu::pp_permute",
                        shift,
                        mesh=mesh,
                        input_partition_specs=(spec, ),
                        output_partition_specs=spec)
    _OPS[key] = op
    return op
