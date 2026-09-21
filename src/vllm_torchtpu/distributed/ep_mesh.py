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

from vllm_torchtpu.distributed.mesh_utils import (_collect_rank_to_device_id,
                                                  _get_current_global_rank,
                                                  _get_tpu_global_device_id)
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


def ep_token_replica_groups(*,
                            is_sequence_parallel: bool = False
                            ) -> tuple[tuple[int, ...], ...]:
    """Native token replica groups expressed in the EP mesh's coordinates.

    TP inputs are replicated unless MoE sequence parallelism partitions them.
    Gather the actual group memberships rather than deriving them from rank
    arithmetic: vLLM ranks and ascending-device mesh positions may differ.
    Called during weight loading, never inside the compiled forward.
    """
    group = get_ep_group()
    if group is None:
        raise RuntimeError(
            "Token replica groups require an initialized EP group")
    rank = _get_current_global_rank()
    ranks = tuple(int(r) for r in group.ranks)
    replicas = ((rank, ) if is_sequence_parallel else tuple(
        int(r) for r in parallel_state.get_tp_group().ranks))
    device_ids = ep_device_ids()
    key = (ranks, device_ids, replicas)
    cached = _TOKEN_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    gathered = [None] * len(ranks)
    torch.distributed.all_gather_object(gathered, (rank, replicas),
                                        group=group.cpu_group)
    if any(item is None for item in gathered):
        raise RuntimeError("EP token replica gather returned an empty entry")
    reported = {
        int(r): tuple(int(p) for p in members)
        for r, members in gathered
    }
    if set(reported) != set(ranks):
        raise RuntimeError(
            "EP token replica gather does not cover the EP ranks")
    for r, members in reported.items():
        if (r not in members or len(set(members)) != len(members)
                or any(reported.get(peer) != members for peer in members)):
            raise RuntimeError(
                f"Inconsistent token replica group: {r}: {members}")
    if len({len(members) for members in reported.values()}) != 1:
        raise RuntimeError("EP token replica groups must have equal sizes")
    if len(set(device_ids)) != len(ranks):
        raise RuntimeError("EP token replica ranks must use distinct devices")
    mesh_index = {
        r: i
        for i, (_, r) in enumerate(sorted(zip(device_ids, ranks)))
    }
    groups = tuple(
        sorted(
            {
                tuple(mesh_index[r] for r in members)
                for members in reported.values()
            },
            key=min))
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
    rank_to_device_id = _collect_rank_to_device_id(group,
                                                   _get_current_global_rank(),
                                                   _get_tpu_global_device_id())
    missing = [r for r in ranks if r not in rank_to_device_id]
    if missing:
        raise RuntimeError(
            "Could not map all EP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}")
    return tuple(rank_to_device_id[r] for r in ranks)


def ep_rank_order() -> tuple[int, ...] | None:
    """EP rank at each mesh index, for a mesh in ascending device-id order.

    A Mosaic kernel that addresses a peer with `DeviceIdType.MESH` resolves the
    mesh coordinate positionally against the device list, not through the
    device array the mesh was built with: an eight-process run whose mesh is in
    EP-rank order (device ids `[0, 1, 4, 5, 6, 7, 2, 3]` on a tpu7x-8) sends
    every routed row to the wrong peer and deadlocks on the first call, and so
    does any other permuted order. Ascending device-id order is the one that
    works, and it is also what a single-process mesh gets from `jax.devices()`,
    which is why the kernel's own device tests never saw this.

    Ordering by device id means mesh index i is no longer EP rank i, so the
    shard sitting at mesh index i owns the experts of EP rank
    `ep_rank_order()[i]` rather than of rank i. The caller has to say so -- see
    `fused_moe_ep._mesh_expert_order` -- because the kernel reads a
    token's expert id as `mesh_index * experts_per_shard + j`.

    Returns None when EP is off.
    """
    group = get_ep_group()
    device_ids = ep_device_ids()
    if group is None or device_ids is None:
        return None
    ranks = tuple(int(r) for r in group.ranks)
    by_device = dict(zip(device_ids, ranks))
    if len(by_device) != len(ranks):
        raise RuntimeError(
            "Two EP ranks report the same TPU global device id: "
            f"ranks={ranks}, device_ids={device_ids}")
    ep_rank_of_group_rank = {r: i for i, r in enumerate(ranks)}
    return tuple(ep_rank_of_group_rank[by_device[d]]
                 for d in sorted(device_ids))


def ep_mesh_index() -> int | None:
    """This rank's index into the mesh `build_ep_mesh` builds, or None."""
    device_ids = ep_device_ids()
    if device_ids is None:
        return None
    return sorted(device_ids).index(_get_tpu_global_device_id())


def build_ep_mesh(axis_name: str = EP_AXIS_NAME) -> Any | None:
    """A one-axis JAX mesh over the EP group, or None when EP is off.

    Devices go in ascending id order rather than EP-rank order, which is what
    the kernels' peer addressing requires; `ep_rank_order` has the detail and
    the consequence for the caller.

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

    import jax
    import numpy as np
    from jax.sharding import Mesh

    visible = {
        int(getattr(device, "id", idx)): device
        for idx, device in enumerate(jax.devices())
    }
    missing = [d for d in device_ids if d not in visible]
    if missing:
        raise RuntimeError(
            "Cannot build the EP mesh: JAX does not expose every EP device in "
            f"this worker. missing={missing}, ep_device_ids={device_ids}, "
            f"jax_visible_ids={sorted(visible)}. A multi-device op needs the "
            "whole slice visible in each process; check the chip binding "
            "(TPU_VISIBLE_CHIPS/TPU_VISIBLE_DEVICES) the worker was started "
            "with.")

    mesh = Mesh(np.asarray([visible[d] for d in device_ids]),
                axis_names=(axis_name, ))
    logger.info("Built EP mesh | axis=%s | size=%d | device_ids=%s", axis_name,
                len(device_ids), list(device_ids))
    _MESH_CACHE[cache_key] = mesh
    return mesh
