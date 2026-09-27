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

from vllm_torchtpu.distributed.mesh_utils import (
    _collect_rank_to_device_id,
    _get_current_global_rank,
    _get_tpu_global_device_id,
)
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
    arithmetic: vLLM ranks and ascending-device mesh positions may differ.
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
    key = (ranks, device_ids, replicas)
    cached = _TOKEN_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    gathered = [None] * len(ranks)
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
    if len(set(device_ids)) != len(ranks):
        raise RuntimeError("EP token replica ranks must use distinct devices")
    mesh_index = {r: i for i, (_, r) in enumerate(sorted(zip(device_ids, ranks)))}
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


def ep_rank_order() -> tuple[int, ...] | None:
    """EP rank at each mesh index -- the identity, or None when EP is off.

    It was not always the identity. Before google-pytorch/torch_tpu#3522 a
    Mosaic kernel resolved a `DeviceIdType.MESH` coordinate positionally
    against the runtime device list rather than through the device array the
    mesh was built with, so the mesh had to be built in ascending device-id
    order to address the right peer, and mesh index i then held the experts of
    a different EP rank. #3522 makes partition index p bind to PyTorch rank p,
    `build_ep_mesh` orders the mesh by rank to match, and mesh index i holds EP
    rank i's experts again.

    Kept rather than deleted so `fused_moe_ep._mesh_expert_order` keeps a
    single place to ask, and so a future ordering change has somewhere to live.
    """
    group = get_ep_group()
    if group is None:
        return None
    return tuple(range(len(tuple(group.ranks))))


def ep_mesh_index() -> int | None:
    """This rank's index into the mesh `build_ep_mesh` builds, or None.

    The mesh is rank-ordered since #3522, so the index is this worker's EP
    rank rather than a position reconstructed from device ids.
    """
    group = get_ep_group()
    if group is None:
        return None
    ranks = tuple(int(r) for r in group.ranks)
    return ranks.index(_get_current_global_rank())


def build_ep_mesh(axis_name: str = EP_AXIS_NAME) -> Any | None:
    """A one-axis JAX mesh over the EP group, or None when EP is off.

    Torchtpu's ``get_pallas_mesh`` orders the mesh by PyTorch rank, which is
    what google-pytorch/torch_tpu#3522 binds partition indices to. Mesh index
    i therefore owns EP rank i's experts, with no permutation for the caller
    to carry.

    Cached per (axis, device ids): the ids come from a collective, so this must
    not be called for the first time inside a traced region.
    """
    unordered = ep_device_ids()
    if unordered is None:
        return None
    device_ids = tuple(sorted(unordered))

    cache_key = (axis_name, device_ids)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # No jax.devices() lookup: get_pallas_mesh builds abstract devices from the
    # process group, so this worker no longer needs JAX to expose every peer.
    import torch.distributed as dist

    world_size = int(dist.get_world_size())
    if len(device_ids) != world_size:
        raise NotImplementedError(
            f"The EP group spans {len(device_ids)} devices inside a "
            f"torch.distributed world of {world_size}. get_pallas_mesh builds "
            "a mesh over the whole world, so EP alongside another parallelism "
            "axis needs a mesh_shape factorisation that has not been worked "
            "out."
        )

    from torch_tpu._internal.pallas import get_pallas_mesh

    mesh = get_pallas_mesh(axis_names=(axis_name,))
    logger.info(
        "Built EP mesh | axis=%s | size=%d | rank-ordered", axis_name, world_size
    )
    _MESH_CACHE[cache_key] = mesh
    return mesh
