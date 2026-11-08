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
"""Torch bridge for the hierarchical TP4 all-reduce (``TPU_TP4_HIER_ALL_REDUCE``).

Row-parallel linears hand their partial output here instead of to vLLM's TP
all-reduce when the TP group is four cores on two chips, ordered
``2 * chip + core``. Shapes the kernel does not take fall back to XLA's
all-reduce inside the same op.
"""

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.collectives.tp4_all_reduce import tp4_all_reduce
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DP_AXIS = "dp"
TP_AXIS = "tp"
_STATE: dict[str, object] = {}


def _tp_group_placement(dp_size: int, tp_size: int):
    """(chip, core) of every rank, per TP group: from device coordinates when JAX
    exposes them, else from the global device ids (two cores per chip, numbered
    consecutively). A wrong guess costs speed only: the kernel addresses its
    peers by TP rank."""
    from vllm.distributed.parallel_state import get_world_group

    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.mesh_utils import (
        _collect_rank_to_device_id,
        _get_current_global_rank,
        _get_tpu_global_device_id,
    )

    world = dp_size * tp_size
    topology = get_chip_topology()
    if topology is not None and len(topology.device_ids) == world:
        source = "coords"
        place = [topology.chip_of(r) for r in range(world)]
    else:
        source = "device ids"
        ids = _collect_rank_to_device_id(
            get_world_group(), _get_current_global_rank(), _get_tpu_global_device_id()
        )
        everyone = list(range(world))
        if sorted(ids) != everyone or sorted(ids.values()) != everyone:
            return None
        place = [(ids[r] // 2, ids[r] % 2) for r in range(world)]
    logger.info("TP4 hierarchical all-reduce placement from %s: %s", source, place)
    return [place[d * tp_size : (d + 1) * tp_size] for d in range(dp_size)]


def _build():
    """The (dp, tp) mesh and the op, or None when the layout does not fit."""
    import torch.distributed as dist
    from torch_tpu._internal.pallas import get_pallas_mesh
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    from vllm_torchtpu.distributed.mesh_utils import get_pipeline_stage

    dp, tp = get_dp_group(), get_tp_group()
    dp_size, tp_size = int(dp.world_size), int(tp.world_size)
    rank, world = int(dist.get_rank()), int(dist.get_world_size())
    _, stages = get_pipeline_stage()
    dp_rank, tp_rank = int(dp.rank_in_group), int(tp.rank_in_group)
    expected = {
        "tp size": (tp_size, 4),
        "world": (world, dp_size * tp_size),
        "pipeline stages": (stages, 1),
        "rank": (rank, dp_rank * tp_size + tp_rank),
        "tp group": (
            tuple(tp.ranks),
            tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size)),
        ),
    }
    bad = {k: v for k, v in expected.items() if v[0] != v[1]}
    if bad:
        logger.warning("TPU_TP4_HIER_ALL_REDUCE off: layout %s", bad)
        return None
    placement = _tp_group_placement(dp_size, tp_size)
    for d, placed in enumerate(placement or [None] * dp_size):
        if not (
            placed is not None
            and placed[0][0] == placed[1][0]
            and placed[2][0] == placed[3][0]
            and placed[0][0] != placed[2][0]
            and [core for _, core in placed] == [0, 1, 0, 1]
        ):
            logger.warning(
                "TPU_TP4_HIER_ALL_REDUCE off: TP group %d (chip, core) = %s", d, placed
            )
            return None
    mesh = get_pallas_mesh(axis_names=(DP_AXIS, TP_AXIS), mesh_shape=(dp_size, tp_size))
    num_blocks = envs.TPU_TP4_HIER_ALL_REDUCE_BLOCKS
    max_rows = envs.TPU_TP4_HIER_ALL_REDUCE_MAX_ROWS

    def fn(x: jax.Array) -> jax.Array:
        return tp4_all_reduce(
            x,
            mesh=mesh,
            dp_axis=DP_AXIS,
            tp_axis=TP_AXIS,
            num_blocks=num_blocks,
            max_rows=max_rows,
        )

    spec = PartitionSpec(DP_AXIS, TP_AXIS)
    op = pallas.jax_op(
        "pallas::tp4_hier_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=(spec,),
    )
    op.register_fake(lambda x: torch.empty_like(x))
    logger.info(
        "TP4 hierarchical all-reduce armed | dp=%d tp=4 | %d column blocks per "
        "quarter, up to %d rows (XLA all-reduce above)",
        dp_size,
        num_blocks,
        max_rows,
    )
    return op


def tp4_hier_all_reduce_op():
    """The op when ``TPU_TP4_HIER_ALL_REDUCE`` is on and the layout fits, else None."""
    if not envs.TPU_TP4_HIER_ALL_REDUCE:
        return None
    if "op" not in _STATE:
        _STATE["op"] = _build()
    return _STATE["op"]


def tp4_hier_all_reduce(op, x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    out = op(x.reshape(1, 1, -1, shape[-1]))
    return out.reshape(shape)
