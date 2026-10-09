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
"""Group layout shared by the context-parallel axes.

`pcp.py` and `dcp.py` each need the same thing: the order of the ranks in
their process group paired with the TPU device each one sits on. Neither is
specific to which axis asked -- a layout is just "this group's ranks, in
order, and where they live"

Every axis builds its own Pallas mesh, because torch_tpu requires one that
spans the whole world and the axes factor the world differently. `ep_mesh.py`
takes the rank/device helpers from here too, which is why this module is
named for the job rather than for an axis.

Under pipeline parallelism a group is one pipeline stage, a part of the slice.
Its mesh then holds the stage's devices alone (`get_stage_mesh`): torch_tpu
compiles a program whose kernels and collectives all name the devices of one
process group for those devices, so a stage launches its programs while the
other stages launch theirs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CpGroupLayout:
    """A CP process group's rank order plus the matching TPU device ids."""

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


_LAYOUT_CACHE: dict[tuple[int, tuple[int, ...], int, int], CpGroupLayout] = {}


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
        except Exception:  # noqa: S110
            pass
        return 0


def _collect_rank_to_device_id(
    group: Any, global_rank: int, device_id: int
) -> dict[int, int]:
    gathered: list[tuple[int, int] | None] = [None] * int(group.world_size)
    torch.distributed.all_gather_object(
        gathered, (global_rank, device_id), group=group.cpu_group
    )
    rank_to_device_id: dict[int, int] = {}
    for item in gathered:
        if item is None:
            raise RuntimeError("CP device id gather returned an empty entry.")
        rank, gathered_device_id = item
        rank_to_device_id[int(rank)] = int(gathered_device_id)
    return rank_to_device_id


def get_cp_group_layout(group: Any | None) -> CpGroupLayout:
    """Return `group`'s rank order and the matching TPU device-id order.

    The order of ``device_ids`` mirrors ``group.ranks``. This is the order a CP
    Pallas mesh should use, so that a TP+CP layout forms one CP ring per TP
    lane.

    Args:
      group: the process group to describe, or None before distributed init
        (in which case the layout describes this worker alone).
    """

    global_rank = _get_current_global_rank()
    device_id = _get_tpu_global_device_id()
    if group is None or int(group.world_size) == 1:
        return CpGroupLayout(
            ranks=(global_rank,),
            device_ids=(device_id,),
            rank_in_group=0,
            world_size=1,
        )

    ranks = tuple(int(rank) for rank in group.ranks)
    rank_in_group = int(group.rank_in_group)
    cache_key = (global_rank, ranks, rank_in_group, device_id)
    cached = _LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rank_to_device_id = _collect_rank_to_device_id(group, global_rank, device_id)
    missing = [rank for rank in ranks if rank not in rank_to_device_id]
    if missing:
        raise RuntimeError(
            "Could not map all CP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}"
        )
    device_ids = tuple(rank_to_device_id[rank] for rank in ranks)
    layout = CpGroupLayout(
        ranks=ranks,
        device_ids=device_ids,
        rank_in_group=rank_in_group,
        world_size=int(group.world_size),
    )
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def get_pipeline_stage() -> tuple[int, int]:
    """This worker's pipeline stage and the number of stages.

    (0, 1) without pipeline parallelism or before distributed init.
    """
    try:
        from vllm.distributed.parallel_state import get_pp_group

        group = get_pp_group()
    except (AssertionError, ImportError):
        return 0, 1
    return int(group.rank_in_group), int(group.world_size)


def get_stage_mesh(
    axis_names: tuple[str, ...],
    mesh_shape: tuple[int, ...],
    group: Any,
    device_ids: tuple[int, ...],
) -> Any:
    """Build a rank-aligned JAX mesh over `group`, one pipeline stage.

    Args:
      axis_names: the mesh axes.
      mesh_shape: the mesh shape; its product is the size of `group`.
      group: the stage's process group.
      device_ids: the TPU device ids of `group`, in rank order.
    """
    from torch_tpu._internal.pallas import get_pallas_mesh

    # TODO(yinlin): torch_tpu releases refuse a mesh smaller than the world
    # here, and pp_validation turns such stages away; make the stage group
    # standalone once the pinned torch_tpu has google-pytorch/torch_tpu#4387.
    mesh = get_pallas_mesh(
        axis_names=axis_names,
        mesh_shape=mesh_shape,
        process_group=group.device_group,
    )
    mesh_device_ids = tuple(int(device.id) for device in mesh.devices.flat)
    if mesh_device_ids != tuple(device_ids):
        raise RuntimeError(
            "The stage mesh does not hold the stage's devices in rank order: "
            f"mesh={mesh_device_ids}, stage={tuple(device_ids)}, "
            f"ranks={tuple(group.ranks)}."
        )
    return mesh
