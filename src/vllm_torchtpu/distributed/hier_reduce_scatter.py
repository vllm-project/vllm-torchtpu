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
"""Arms the EP MoE combine's reduce-scatter on the hierrs_tc kernels.

The op runs on the EP mesh, which must be in ascending device id: arming
refuses the rank-ordered mesh (`MOE_FUSED_EP_SORTED_GATHER`), because the
kernel takes its neighbour chips from the mesh order. `group.reduce_scatter`
hands rank r the r-th row block, so mesh index i receives the block of the
rank there (`ep_rank_order`). Kernel selection lives in
`kernels/collectives/hierrs_tc/dispatch.py`.
"""

from typing import Any, NamedTuple

import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

OP_NAME = "pallas::hier_reduce_scatter"


class _Armed(NamedTuple):
    op: Any
    ranks: tuple[int, ...]


_ARMED: _Armed | None = None


def _unsupported_world_reason() -> str | None:
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group
    from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
        supports_chip_topology,
        supports_num_devices,
    )

    ep = get_ep_group()
    if ep is None:
        return "expert parallelism is off"
    if not supports_num_devices(int(ep.world_size)):
        return f"the kernel does not support an EP group of {ep.world_size}"
    if int(ep.world_size) != int(dist.get_world_size()):
        return "the EP group does not span the whole torch.distributed world"
    if list(get_dp_group().ranks) != list(ep.ranks):
        return "the DP group's ranks differ from the EP group's"
    topology = get_chip_topology()
    if topology is None:
        return "the chip topology is unknown"
    if not supports_chip_topology(topology.num_chips, topology.cores_per_chip):
        return (
            f"the kernel needs 4 chips of 2 TensorCores, not {topology.num_chips} "
            f"of {topology.cores_per_chip}"
        )
    if envs.MOE_FUSED_EP_SORTED_GATHER:
        return "MOE_FUSED_EP_SORTED_GATHER keeps the EP mesh in EP-rank order"
    return None


def _unordered_mesh_reason(mesh: Any) -> str | None:
    ids = [int(d.id) for d in mesh.devices.flat]
    if ids != sorted(ids):
        return f"the EP mesh is not in ascending device id: {ids}"
    return None


def _build_op(
    mesh: Any,
    order: tuple[int, ...],
    axis_name: str,
    *,
    fp8_comm: bool = False,
    fp8_static_scale: float | None = None,
) -> Any:
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.collectives.hierrs_tc.dispatch import (
        make_reduce_scatter_fn,
    )

    reduce_scatter = make_reduce_scatter_fn(
        mesh,
        order,
        axis_name=axis_name,
        fp8_comm=fp8_comm,
        fp8_static_scale=fp8_static_scale,
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(x: jax.Array) -> jax.Array:
        return reduce_scatter(x)

    spec = P(axis_name, None)
    # torch_tpu annotates the partition specs as tuples of axis-name tuples, but
    # jax_op takes PartitionSpecs at runtime, as other jax_op call sites pass them.
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        OP_NAME,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=spec,
    )
    num_devices = len(order)
    # vLLM traces with a dynamic token dimension.
    op.register_fake(lambda x: x.new_empty((x.shape[0] // num_devices, x.shape[1])))
    return op


def prebuild_hier_reduce_scatter() -> bool:
    """Returns whether the op is armed. Call outside Dynamo: it runs a collective."""
    global _ARMED
    if _ARMED is not None or envs.TPU_MOE_EP_HIER_RS == "off":
        return _ARMED is not None
    fp8 = envs.TPU_MOE_EP_HIER_RS == "fp8"
    scale = envs.TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE
    if scale is not None and scale <= 0:
        raise ValueError(
            f"TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE must be a positive number, got "
            f"{scale}."
        )
    if fp8 and scale is None:
        raise ValueError(
            "TPU_MOE_EP_HIER_RS=fp8 needs TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE: the "
            "row-streaming kernel has no dynamic scale. See "
            "docs/reference/env_vars.md for how to choose it."
        )

    from vllm_torchtpu.distributed.ep_mesh import (
        EP_AXIS_NAME,
        build_ep_mesh,
        ep_rank_order,
        get_ep_group,
    )

    reason = _unsupported_world_reason()
    mesh = order = ep = None
    if reason is None:
        mesh = build_ep_mesh(axis_name=EP_AXIS_NAME)
        order = ep_rank_order()
        ep = get_ep_group()
        assert mesh is not None and order is not None and ep is not None
        reason = _unordered_mesh_reason(mesh)
    if reason is not None:
        logger.warning_once(
            "TPU_MOE_EP_HIER_RS=%s is set, but the MoE combine keeps the group's "
            "reduce-scatter: %s.",
            envs.TPU_MOE_EP_HIER_RS,
            reason,
        )
        return False
    assert mesh is not None and order is not None and ep is not None

    op = _build_op(
        mesh,
        order,
        EP_AXIS_NAME,
        fp8_comm=fp8,
        fp8_static_scale=scale if fp8 else None,
    )
    _ARMED = _Armed(op=op, ranks=tuple(int(r) for r in ep.ranks))
    logger.info_once(
        "MoE combine reduce-scatter runs on hierrs_tc: %d devices, ici_dtype=%s%s, "
        "rank at each mesh index=%s.",
        len(order),
        envs.TPU_MOE_EP_HIER_RS,
        f", static scale {scale}" if fp8 else "",
        order,
    )
    return True


def moe_combine_reduce_scatter(group: Any, x: torch.Tensor) -> torch.Tensor:
    """`group.reduce_scatter(x, dim=0)`, on hierrs_tc when armed and eligible."""
    from vllm_torchtpu.kernels.collectives.hierrs_tc.wrapper import (
        is_supported_by_hierrs_tc,
    )

    armed = _ARMED
    if (
        armed is not None
        and tuple(int(r) for r in group.ranks) == armed.ranks
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and is_supported_by_hierrs_tc(len(armed.ranks), x.shape[0], x.shape[1])
    ):
        return armed.op(x)
    return group.reduce_scatter(x, dim=0)
