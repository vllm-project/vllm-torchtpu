# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PcpGroupLayout:
    """PCP process group order plus the matching TPU global device ids."""

    ranks: tuple[int, ...]
    device_ids: tuple[int, ...]
    rank_in_group: int
    world_size: int

    @property
    def device_id(self) -> int:
        return self.device_ids[self.rank_in_group]

    @property
    def prev_rank_in_group(self) -> int:
        return (self.rank_in_group - 1) % self.world_size

    @property
    def next_rank_in_group(self) -> int:
        return (self.rank_in_group + 1) % self.world_size


_LAYOUT_CACHE: dict[tuple[int, tuple[int, ...], int, int], PcpGroupLayout] = {}
_MESH_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}


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

    group = get_pcp_group()
    global_rank = _get_current_global_rank()
    device_id = _get_tpu_global_device_id()
    if group is None or int(group.world_size) == 1:
        return PcpGroupLayout(
            ranks=(global_rank, ),
            device_ids=(device_id, ),
            rank_in_group=0,
            world_size=1,
        )

    ranks = tuple(int(rank) for rank in group.ranks)
    rank_in_group = int(group.rank_in_group)
    cache_key = (global_rank, ranks, rank_in_group, device_id)
    cached = _LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rank_to_device_id = _collect_rank_to_device_id(group, global_rank,
                                                   device_id)
    missing = [rank for rank in ranks if rank not in rank_to_device_id]
    if missing:
        raise RuntimeError(
            "Could not map all PCP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}")
    device_ids = tuple(rank_to_device_id[rank] for rank in ranks)
    layout = PcpGroupLayout(
        ranks=ranks,
        device_ids=device_ids,
        rank_in_group=rank_in_group,
        world_size=int(group.world_size),
    )
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def get_or_create_pcp_mesh(axis_name: str = "pcp") -> Any:
    """Build a JAX mesh ordered by the current vLLM PCP group.

    This helper is intentionally op-local: it does not replace the runner's
    normal single-device model mesh.
    """

    layout = get_pcp_group_layout()
    cache_key = (axis_name, layout.device_ids)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import jax
    import numpy as np
    from jax.sharding import Mesh

    devices_by_id = {
        int(getattr(device, "id", idx)): device
        for idx, device in enumerate(jax.devices())
    }
    missing = [
        device_id for device_id in layout.device_ids
        if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(
            "Cannot build PCP mesh because JAX does not expose all PCP TPU "
            f"device ids: missing={missing}, layout={layout}, "
            f"jax_device_ids={sorted(devices_by_id)}")

    mesh_devices = np.asarray(
        [devices_by_id[device_id] for device_id in layout.device_ids])
    mesh = Mesh(mesh_devices, axis_names=(axis_name, ))
    _MESH_CACHE[cache_key] = mesh
    return mesh
