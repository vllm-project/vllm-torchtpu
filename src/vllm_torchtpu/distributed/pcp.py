# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.distributed.mesh_utils import (CpGroupLayout,
                                                  get_cp_group_layout,
                                                  get_or_create_cp_mesh)

PcpGroupLayout = CpGroupLayout


def get_pcp_group() -> Any | None:
    """Return vLLM's native PCP group, or None before distributed init."""
    try:
        from vllm.distributed.parallel_state import get_pcp_group as _get_group

        return _get_group()
    except (AssertionError, ImportError):
        return None


def get_pcp_rank() -> int:
    group = get_pcp_group()
    return 0 if group is None else int(group.rank_in_group)


def get_pcp_world_size() -> int:
    group = get_pcp_group()
    return 1 if group is None else int(group.world_size)


def all_gather_equal_tokens(tensor: torch.Tensor,
                            dim: int = 0) -> torch.Tensor:
    """Gather equal-shaped tensors across the native PCP group."""
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    gathered = group.all_gather(tensor, dim=dim)
    if isinstance(gathered, torch.Tensor):
        return gathered
    return torch.cat(list(gathered), dim=dim)


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across the native PCP group."""
    group = get_pcp_group()
    if group is None or int(group.world_size) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return group.all_reduce(tensor)


def _get_current_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _get_tpu_global_device_id() -> int:
    try:
        from torch_tpu._internal.distributed import tpu_distributed

        return int(tpu_distributed.global_device_id())
    except Exception:
        # Single-process tests may exercise layout code before TorchTPU/PJRT is
        # initialized. Use JAX's local device id when available, otherwise the
        # only sensible single-device fallback is 0.
        try:
            import jax

            local_devices = jax.local_devices()
            if local_devices:
                return int(getattr(local_devices[0], "id", 0))
        except Exception:
            pass
        return 0


def _collect_rank_to_device_id(group: Any, global_rank: int,
                               device_id: int) -> dict[int, int]:
    gathered: list[tuple[int, int] | None] = [None] * int(group.world_size)
    torch.distributed.all_gather_object(gathered, (global_rank, device_id),
                                        group=group.cpu_group)
    rank_to_device_id: dict[int, int] = {}
    for item in gathered:
        if item is None:
            raise RuntimeError("PCP device id gather returned an empty entry.")
        rank, gathered_device_id = item
        rank_to_device_id[int(rank)] = int(gathered_device_id)
    return rank_to_device_id


def get_pcp_group_layout() -> PcpGroupLayout:
    """Return the current PCP group's rank order and TPU device-id order.

    The order of ``device_ids`` mirrors ``get_pcp_group().ranks``. This is the
    order a PCP Pallas mesh should use so TP+PCP layouts form one PCP ring per
    TP lane.
    """
    return get_cp_group_layout(get_pcp_group())


def get_or_create_pcp_mesh(axis_name: str = "pcp") -> Any:
    """Build a JAX mesh ordered by the current vLLM PCP group.

    This helper is intentionally op-local: it does not replace the runner's
    normal single-device model mesh.

    The mesh is built by Torchtpu's ``get_pallas_mesh``, which orders devices
    by the ranks of the process group it is handed. It does not consult
    ``jax.devices()``, so it neither requires JAX to have a view of every
    peer's device nor forces JAX to initialize a hardware runtime just to
    compile a kernel. Partition ``i`` inside the kernel is
    ``mesh.devices.flat[i]``, so ordering by rank is what makes
    ``lax.axis_index(axis_name)`` agree with ``rank_in_group``.
    """

    layout = get_pcp_group_layout()
    cache_key = (axis_name, layout.device_ids)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    group = get_pcp_group()
    if group is None:
        # No PCP group: a one-device mesh, where there is no order to get
        # wrong. get_pallas_mesh needs a ProcessGroup, so build it directly.
        import jax
        import numpy as np
        from jax.sharding import Mesh

        mesh = Mesh(np.asarray(jax.local_devices()[:1]),
                    axis_names=(axis_name, ))
        _MESH_CACHE[cache_key] = mesh
        return mesh

    from torch_tpu._internal.pallas import get_pallas_mesh

    mesh = get_pallas_mesh(group.device_group,
                           axis_names=(axis_name, ),
                           mesh_shape=(layout.world_size, ))

    # get_pallas_mesh reads the rank -> device mapping out of Torchtpu's own
    # process-group state, and falls back to using the rank *as* the device id
    # when that state is missing for this group -- which on a host where rank
    # order and device order differ is silently the wrong mesh, and therefore
    # silently the wrong DeviceAssignment. layout.device_ids is the same
    # mapping gathered independently over the group, so comparing them turns
    # that fallback into a startup failure instead of wrong numbers.
    mesh_device_ids = tuple(
        int(getattr(d, "id", -1)) for d in mesh.devices.flat)
    if mesh_device_ids != layout.device_ids:
        raise RuntimeError(
            "PCP mesh device order does not match the PCP group, so kernel "
            "partition indices would not be PCP ranks and collectives would "
            f"move data to the wrong worker. Torchtpu built "
            f"{mesh_device_ids}; the group gathered {layout.device_ids} for "
            f"ranks {layout.ranks}.")

    # Log rank and device id for each position in the mesh. Logs once per
    # worker.
    rank_and_device = zip(layout.ranks, layout.device_ids)
    positions = ", ".join(f"pos{i}=rank{rank}/dev{dev}"
                          for i, (rank, dev) in enumerate(rank_and_device))
    logger.info(
        "PCP device mesh | axis=%s world_size=%d my_rank_in_group=%d | %s",
        axis_name, layout.world_size, layout.rank_in_group, positions)

    _MESH_CACHE[cache_key] = mesh
    return mesh
