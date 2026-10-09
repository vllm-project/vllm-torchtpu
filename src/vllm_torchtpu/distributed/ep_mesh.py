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
"""An op-local JAX mesh over the expert-parallel group.

The runner's model mesh is deliberately single-device (`TPUModelRunner.
_create_mesh_for_parallelism`): the torch-native model path runs one JAX device
per worker process and crosses ranks through torch.distributed. That is not a
limit on what a *single op* can do. `distributed.pcp.get_or_create_pcp_mesh`
already builds a multi-device mesh for the PCP group and runs a `shard_map`
Pallas kernel with device-to-device remote DMAs over it, in the serving path
(`kernels/experimental/pcp_streaming_rpa`). This is the same construction for
the EP group, so an MoE op can fuse its dispatch/combine transport into the
kernel instead of paying a separate, fully exposed reduce-scatter.

The one runtime precondition is that every worker's `jax.devices()` exposes the
whole slice, not just its own chip. `build_ep_mesh` checks that and fails with
the ids it could and could not see, rather than letting a caller build a mesh
that silently addresses the wrong device.
"""

from typing import Any

import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed.dcp import get_dcp_world_size
from vllm_torchtpu.distributed.mesh_utils import (
    _collect_rank_to_device_id,
    _get_current_global_rank,
    _get_tpu_global_device_id,
    get_pipeline_stage,
    get_stage_mesh,
)
from vllm_torchtpu.distributed.pcp import get_pcp_world_size
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The fused MoE kernel body calls `lax.axis_index` on its own module constant
# (`kernels.fused_moe.v2.AXIS`) rather than on a name from the mesh it is
# handed, so a mesh axis named anything else traces the whole kernel and then
# dies on an unbound axis name. Any consumer that cares asserts against that
# constant; this default is what makes it work.
EP_AXIS_NAME = "d"

_MESH_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}
_TOKEN_GROUP_CACHE: dict[tuple, tuple[tuple[int, ...], ...]] = {}


def ep_token_replica_groups(
    *, is_sequence_parallel: bool = False
) -> tuple[tuple[int, ...], ...]:
    """Native token replica groups expressed in the EP mesh's coordinates.

    TP inputs are replicated unless MoE sequence parallelism partitions them.
    Gather the actual group memberships rather than deriving them from rank
    arithmetic: vLLM ranks and mesh positions may differ (see `_ep_mesh_order`).
    Called during weight loading, never inside the compiled forward.
    """
    group = get_ep_group()
    if group is None:
        raise RuntimeError("Token replica groups require an initialized EP group")
    rank = _get_current_global_rank()
    ranks = tuple(int(r) for r in group.ranks)
    replicas = (
        (rank,)
        if is_sequence_parallel
        else tuple(int(r) for r in parallel_state.get_tp_group().ranks)
    )
    device_ids = ep_device_ids()
    assert device_ids is not None
    key = (ranks, device_ids, replicas)
    cached = _TOKEN_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    gathered: list[Any] = [None] * len(ranks)
    torch.distributed.all_gather_object(
        gathered, (rank, replicas), group=group.cpu_group
    )
    if any(item is None for item in gathered):
        raise RuntimeError("EP token replica gather returned an empty entry")
    reported = {int(r): tuple(int(p) for p in members) for r, members in gathered}
    if set(reported) != set(ranks):
        raise RuntimeError("EP token replica gather does not cover the EP ranks")
    for r, members in reported.items():
        if (
            r not in members
            or len(set(members)) != len(members)
            or any(reported.get(peer) != members for peer in members)
        ):
            raise RuntimeError(f"Inconsistent token replica group: {r}: {members}")
    if len({len(members) for members in reported.values()}) != 1:
        raise RuntimeError("EP token replica groups must have equal sizes")
    mesh_index = {ranks[e]: i for i, e in enumerate(_ep_mesh_order(device_ids))}
    groups = tuple(
        sorted(
            {tuple(mesh_index[r] for r in members) for members in reported.values()},
            key=min,
        )
    )
    _TOKEN_GROUP_CACHE[key] = groups
    return groups


def get_ep_group() -> Any | None:
    """vLLM's expert-parallel group, or None when EP is off."""
    try:
        from vllm.distributed.parallel_state import get_ep_group as _get
    except ImportError:
        return None
    try:
        group = _get()
    except (AssertionError, AttributeError, RuntimeError):
        return None
    return group if group is not None and int(group.world_size) > 1 else None


def ep_device_ids() -> tuple[int, ...] | None:
    """Global TPU device ids of the EP group, in EP-rank order.

    This is the natural order -- mesh index i would be EP rank i -- but it is
    NOT the order `build_ep_mesh` uses; see `ep_rank_order` for why. Returns
    None when EP is off.
    """
    group = get_ep_group()
    if group is None:
        return None
    ranks = tuple(int(r) for r in group.ranks)
    rank_to_device_id = _collect_rank_to_device_id(
        group, _get_current_global_rank(), _get_tpu_global_device_id()
    )
    missing = [r for r in ranks if r not in rank_to_device_id]
    if missing:
        raise RuntimeError(
            "Could not map all EP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}"
        )
    return tuple(rank_to_device_id[r] for r in ranks)


def _ep_mesh_order(device_ids: tuple[int, ...]) -> tuple[int, ...]:
    """EP rank at each index of the mesh `build_ep_mesh` builds.

    `device_ids` is `ep_device_ids()`, in EP-rank order. This is the one place
    the mesh order is decided: `build_ep_mesh`, `ep_rank_order`,
    `ep_mesh_index` and `ep_token_replica_groups` all take it from here.

    Since google-pytorch/torch_tpu#3522 the mesh's device order is the kernel's
    DeviceAssignment, so any order addresses the right peers. Ascending device
    id is preferred because XLA's SparseCore-offloaded all-gathers inside the
    fused MoE are much slower over the rank-ordered one: on a tpu7x 2x2x2
    DSv4-Pro run the fp8 activation all-gather (identical HLO and bytes) went
    from 178 to 321 us and end-to-end throughput dropped 4.5%.

    With PCP or DCP on, the mesh stays in rank order. torch_tpu requires every
    Pallas kernel in one compiled module to share a single DeviceAssignment,
    and both CP meshes are rank-ordered: the PCP mesh
    (`pcp.get_or_create_pcp_mesh`) because the PCP kernels index token shards
    and state by PCP rank, the DCP mesh (`dcp.get_or_create_dcp_mesh`, used by
    the MLA attention op) because it is built by walking the group's ranks. An
    EP mesh in device-id order beside either fails to compile with "Disjoint or
    conflicting device assignments" -- for DCP whenever a DCP group is not a
    contiguous tile of the sorted EP mesh, e.g. TP4/DCP4 on tpu7x, whose DCP
    groups are devices [0, 1, 4, 5] and [6, 7, 2, 3].
    """
    if len(set(device_ids)) != len(device_ids):
        raise RuntimeError(
            f"Two EP ranks report the same TPU global device id: {device_ids}"
        )
    ep_ranks = range(len(device_ids))
    if get_pcp_world_size() > 1 or get_dcp_world_size() > 1:
        return tuple(ep_ranks)
    return tuple(sorted(ep_ranks, key=device_ids.__getitem__))


def ep_rank_order() -> tuple[int, ...] | None:
    """EP rank at each mesh index, or None when EP is off.

    With the mesh in ascending device-id order (PCP and DCP off; see
    `_ep_mesh_order`)
    mesh index i is not EP rank i on a host whose PCI enumeration differs from
    its device ids -- every tpu7x host, where local ranks 0..7 land on devices
    `[0, 1, 4, 5, 6, 7, 2, 3]` -- so the shard at mesh index i owns the experts
    of EP rank `ep_rank_order()[i]`. The caller has to say so -- see
    `fused_moe_ep._mesh_expert_order` -- because the kernel reads a token's
    expert id as `mesh_index * experts_per_shard + j`. With PCP or DCP on it is
    the identity.
    """
    device_ids = ep_device_ids()
    if device_ids is None:
        return None
    return _ep_mesh_order(device_ids)


def ep_mesh_index() -> int | None:
    """This rank's index into the mesh `build_ep_mesh` builds, or None."""
    group = get_ep_group()
    device_ids = ep_device_ids()
    if group is None or device_ids is None:
        return None
    ranks = tuple(int(r) for r in group.ranks)
    return _ep_mesh_order(device_ids).index(ranks.index(_get_current_global_rank()))


def build_ep_mesh(axis_name: str = EP_AXIS_NAME) -> Any | None:
    """A one-axis JAX mesh over the EP group, or None when EP is off.

    The devices are Torchtpu's ``get_pallas_mesh`` ones, put in the order
    `_ep_mesh_order` picks: ascending device id, or rank order when PCP or DCP
    is on.
    `ep_rank_order` has the permutation this introduces and the consequence
    for the caller.

    Cached per (axis, device ids in mesh order): the ids come from a
    collective, so this must not be called for the first time inside a traced
    region.
    """
    unordered = ep_device_ids()
    if unordered is None:
        return None
    order = _ep_mesh_order(unordered)
    device_ids = tuple(unordered[e] for e in order)

    cache_key = (axis_name, device_ids)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # No jax.devices() lookup: get_pallas_mesh builds abstract devices from the
    # process group, so this worker no longer needs JAX to expose every peer.
    import torch.distributed as dist

    world_size = int(dist.get_world_size())
    _, stages = get_pipeline_stage()
    if len(device_ids) * stages != world_size:
        raise NotImplementedError(
            f"The EP group spans {len(device_ids)} devices inside a "
            f"torch.distributed world of {world_size}. get_pallas_mesh builds "
            "a mesh over the whole world, so EP alongside another parallelism "
            "axis needs a mesh_shape factorisation that has not been worked "
            "out."
        )

    import numpy as np
    from jax.sharding import Mesh
    from torch_tpu._internal.pallas import get_pallas_mesh

    if stages > 1:
        # The EP group is one pipeline stage.
        ranked = get_stage_mesh(
            (axis_name,), (len(device_ids),), get_ep_group(), unordered
        )
    else:
        ranked = get_pallas_mesh(axis_names=(axis_name,))
    by_id = {int(d.id): d for d in ranked.devices.flat}
    if sorted(by_id) != sorted(device_ids):
        # The ids become the DeviceAssignment, and ep_rank_order/ep_mesh_index
        # index into them assuming they are the ids gathered above.
        raise RuntimeError(
            "get_pallas_mesh device ids do not match the EP group's: "
            f"pallas={[int(d.id) for d in ranked.devices.flat]}, "
            f"ep_device_ids={unordered}"
        )
    mesh = Mesh(np.asarray([by_id[d] for d in device_ids]), axis_names=(axis_name,))
    logger.info(
        "Built EP mesh | axis=%s | size=%d | device_ids=%s | ep_rank_order=%s",
        axis_name,
        len(device_ids),
        list(device_ids),
        order,
    )
    _MESH_CACHE[cache_key] = mesh
    return mesh
