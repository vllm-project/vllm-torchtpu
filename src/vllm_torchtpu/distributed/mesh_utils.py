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
"""Group layout and JAX mesh construction shared by the context-parallel axes.

`pcp.py` and `dcp.py` each need the same two things: the order of the ranks in
their process group paired with the TPU device each one sits on, and a
one-axis JAX mesh laid out in that order. Neither is specific to which axis
asked -- a layout is just "this group's ranks, in order, and where they live"

`ep_mesh.py` builds its own mesh but takes the rank/device helpers from here
too, which is why this module is named for the job rather than for an axis.

Under pipeline parallelism a context-parallel group is one pipeline stage, and
a multi-device kernel of that group must run on the stage's devices alone: the
other stages run other layers at other times, so a slice-wide launch would
wait for devices that never join. torch_tpu compiles a program for a proper
subgroup of the slice when every collective and kernel in it is tagged with
that group (``collective_devices``), so the stage meshes here span the whole
slice, shaped (stage, group), and every op built over one carries the stage's
devices as its tag. torch_tpu numbers the slice's devices by world rank, and a
collective a kernel bakes in names its peers by mesh position, so the two must
coincide: stage s is the ranks s*K .. s*K+K-1 in that order
(``stage_group_ranks`` builds the model-parallel groups to make it so), and
the stage mesh is the slice's devices in rank order, shaped (stage, group).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


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
_MESH_CACHE: dict[tuple[str, tuple[int, ...]], Any] = {}

# The mesh axis a pipeline stage's group mesh gets for the stage dimension.
STAGE_AXIS_NAME = "stage"
# The mesh axis over a stage's TP lanes when its PCP group is one lane.
STAGE_TP_AXIS_NAME = "stage_tp"


@dataclass(frozen=True)
class StageDeviceTable:
    """Where every rank of the slice sits: its pipeline stage, its position in
    the stage's context-parallel group, and its TPU device id, by world rank.
    torch_tpu numbers the slice's devices by world rank, so a rank is also the
    logical id of its device."""

    stages: tuple[int, ...]
    columns: tuple[int, ...]
    device_ids: tuple[int, ...]
    stage: int
    column: int

    @property
    def num_stages(self) -> int:
        return max(self.stages) + 1

    @property
    def num_columns(self) -> int:
        return max(self.columns) + 1

    def rows(self) -> tuple[tuple[int, ...], ...]:
        """World ranks of every stage's group, stage-major and in group order."""
        table: dict[tuple[int, int], int] = {}
        for rank, (stage, column) in enumerate(zip(self.stages, self.columns)):
            table[(stage, column)] = rank
        return tuple(
            tuple(table[(stage, column)] for column in range(self.num_columns))
            for stage in range(self.num_stages))

    def stage_device_ids(self, stage: int) -> tuple[int, ...]:
        """TPU device ids of one stage's group, in group order."""
        return tuple(self.device_ids[rank] for rank in self.rows()[stage])


_STAGE_TABLE: StageDeviceTable | None = None


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
            raise RuntimeError("CP device id gather returned an empty entry.")
        rank, gathered_device_id = item
        rank_to_device_id[int(rank)] = int(gathered_device_id)
    return rank_to_device_id


def _stage_rows(ranks: list[int], stages: int, stage_size: int,
                device_ids: dict[int, int] | None) -> list[list[int]]:
    """Stage s holds the world ranks s*K .. s*K+K-1. Within a stage the ranks
    are put in ascending device-id order: a subgroup program is loaded for
    its device assignment, and libtpu (tpu_topology.cc:66) rejects an
    assignment whose first device is not the block's origin, e.g. [6, 7, 2,
    3] (VLLM_TPU_STAGE_ORDER_BY_DEVICE=0 keeps rank order)."""
    import os
    rows = [ranks[s * stage_size:(s + 1) * stage_size] for s in range(stages)]
    if (device_ids is not None
            and os.environ.get("VLLM_TPU_STAGE_ORDER_BY_DEVICE", "1") == "1"):
        rows = [sorted(row, key=lambda r: device_ids[r]) for row in rows]
    return rows


def stage_group_ranks(
        stages: int,
        group_size: int,
        tp_size: int = 1,
        device_ids: dict[int, int] | None = None) -> dict[str, list[list[int]]]:
    """The model-parallel groups that put pipeline stage s on the world ranks
    s*K .. s*K+K-1 in rank order: one ``pcp`` (and ``ep``) group per stage,
    one ``pp`` group per column of the stages."""
    from vllm.distributed.parallel_state import get_world_group
    ranks = sorted(int(rank) for rank in get_world_group().ranks)
    stage_size = group_size * tp_size
    width = stages * stage_size
    if ranks != list(range(width)):
        raise RuntimeError(
            f"pipeline stages of {stage_size} devices need the ranks "
            f"0..{width - 1}, got {ranks}")
    rows = _stage_rows(ranks, stages, stage_size, device_ids)
    columns = [[rows[s][k] for s in range(stages)] for k in range(stage_size)]
    if tp_size == 1:
        return {"pcp": rows, "pp": columns, "ep": rows}
    # TP lanes inside a stage: the PCP index is innermost (rank = s*K +
    # t*pcp + p), so a lane's PCP group is a contiguous tile of the stage and
    # a TP group is a strided coset of it; torch_tpu accepts both in a module
    # whose device assignment is the whole stage.
    pcp = [[rows[s][t * group_size + p] for p in range(group_size)]
           for s in range(stages) for t in range(tp_size)]
    tp = [[rows[s][t * group_size + p] for t in range(tp_size)]
          for s in range(stages) for p in range(group_size)]
    return {"pcp": pcp, "tp": tp, "pp": columns, "ep": rows}


def gather_stage_device_table() -> StageDeviceTable:
    """Gather every rank's stage, group position and device id over the world
    group. A collective: every rank calls it once the parallel groups exist
    (the worker does, right after building them)."""
    global _STAGE_TABLE
    if _STAGE_TABLE is not None:
        return _STAGE_TABLE
    from vllm.distributed.parallel_state import (get_pcp_group, get_pp_group,
                                                 get_world_group)
    world = get_world_group()
    pp = get_pp_group()
    stage_size = int(world.world_size) // int(pp.world_size)
    del get_pcp_group
    my_rank = int(world.rank_in_group)
    pairs: list[tuple[int, int] | None] = [None] * int(world.world_size)
    torch.distributed.all_gather_object(
        pairs, (int(pp.rank_in_group), _get_tpu_global_device_id()),
        group=world.cpu_group)
    if any(entry is None for entry in pairs):
        raise RuntimeError("stage device table gather returned an empty entry")
    by_rank = {r: int(p[1]) for r, p in enumerate(pairs)}
    num_stages = int(pp.world_size)
    rows = _stage_rows(list(range(len(pairs))), num_stages, stage_size,
                       by_rank)
    column_of = {r: j for row in rows for j, r in enumerate(row)}
    for r, p in enumerate(pairs):
        if int(p[0]) != r // stage_size:
            raise RuntimeError(
                f"rank {r} is in pipeline stage {p[0]}, not {r // stage_size}"
                "; the model-parallel groups were not built by "
                "stage_group_ranks")
    entries = [(int(p[0]), column_of[r], int(p[1]))
               for r, p in enumerate(pairs)]
    mine = entries[my_rank]
    _STAGE_TABLE = StageDeviceTable(
        stages=tuple(entry[0] for entry in entries),
        columns=tuple(entry[1] for entry in entries),
        device_ids=tuple(entry[2] for entry in entries),
        stage=mine[0],
        column=mine[1],
    )
    logger.info(
        "Stage device table | stage=%d column=%d | rows=%s | device ids=%s",
        _STAGE_TABLE.stage, _STAGE_TABLE.column, _STAGE_TABLE.rows(),
        _STAGE_TABLE.device_ids)
    return _STAGE_TABLE


def get_stage_device_table() -> StageDeviceTable | None:
    """The gathered table, or None outside pipeline parallelism."""
    if _STAGE_TABLE is not None:
        return _STAGE_TABLE
    try:
        from vllm.distributed.parallel_state import get_pp_group
        if int(get_pp_group().world_size) <= 1:
            return None
    except (AssertionError, ImportError):
        return None
    raise RuntimeError(
        "the stage device table was not gathered; the worker gathers it "
        "after the parallel groups are built")


def stage_collective_devices() -> list[int] | None:
    """World ranks of this rank's pipeline stage, in group order: the
    ``collective_devices`` tag of every multi-device op the stage runs. None
    outside pipeline parallelism."""
    table = get_stage_device_table()
    if table is None:
        return None
    return list(table.rows()[table.stage])


def _uses_mesh_device_assignment() -> bool:
    """True when torch_tpu derives a custom kernel's XLA DeviceAssignment from
    ``mesh.devices.flat`` (#4059) rather than ``JaxCallable(collective_devices=...)``
    (#3931)."""
    import inspect
    try:
        from torch_tpu._internal.pallas import pallas as pallas_impl
        if "collective_devices" in inspect.signature(
                pallas_impl.JaxCallable.__init__).parameters:
            return False
        return "process_group" in inspect.signature(
            pallas_impl.get_pallas_mesh).parameters
    except Exception:
        return False


def get_or_create_stage_mesh(axis_name: str,
                             group_size: int | None = None) -> Any:
    """Build the JAX mesh for a pipeline stage's context-parallel or EP group.

    - Under ``torch_tpu#3931`` (``JaxCallable(collective_devices=...)``), returns
      a 2D ``(stage, axis_name)`` mesh over the whole slice in world-rank order;
      C++ ``RewriteCollectivesForDeviceGroup`` projects ``mhlo.sharding`` onto
      this stage's ``collective_devices``.
    - Under ``torch_tpu#4059`` (``device_assignment`` derived from
      ``mesh.devices.flat``), returns a 1D ``(axis_name,)`` mesh over this
      stage's ``K`` physical PJRT device IDs (``table.stage_device_ids``),
      matching ``ProcessGroupTpu::device_assignment()`` for the stage's group.
    """
    table = get_stage_device_table()
    assert table is not None
    rows = table.rows()
    width = len(rows[0])
    for s, row in enumerate(rows):
        if sorted(row) != list(range(s * width, (s + 1) * width)):
            raise RuntimeError(
                f"pipeline stage {s} must hold the consecutive ranks "
                f"{s * width}..{(s + 1) * width - 1}, got {row}; the "
                "model-parallel groups were not built by stage_group_ranks")

    use_1d_stage_mesh = _uses_mesh_device_assignment()
    stage_dev_ids = table.stage_device_ids(table.stage)
    lanes = 1
    if group_size is not None and int(group_size) < width:
        if width % int(group_size) != 0:
            raise RuntimeError(f"a stage of {width} devices cannot hold "
                               f"groups of {group_size}")
        if not use_1d_stage_mesh:
            raise NotImplementedError(
                "TP lanes inside a pipeline stage need a torch_tpu that "
                "derives device_assignment from mesh.devices (#4059)")
        lanes = width // int(group_size)
    cache_key = ((axis_name, "1d", lanes, stage_dev_ids) if use_1d_stage_mesh
                 else (axis_name, tuple(d for row in rows for d in row)))
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
    if use_1d_stage_mesh:
        missing = [d for d in stage_dev_ids if d not in devices_by_id]
        if missing:
            try:
                from torch_tpu._internal.pallas._mock_device import \
                    _PallasDevice
                for d in missing:
                    devices_by_id[d] = _PallasDevice(id=int(d),
                                                     process_index=int(d))
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot build the '{axis_name}' 1D stage mesh because JAX "
                    f"does not expose stage device ids: missing={missing}, "
                    f"stage_dev_ids={stage_dev_ids}, "
                    f"jax_device_ids={sorted(devices_by_id)}") from exc
        flat = np.asarray([devices_by_id[d] for d in stage_dev_ids])
        if lanes > 1:
            mesh = Mesh(flat.reshape(lanes, width // lanes),
                        axis_names=(STAGE_TP_AXIS_NAME, axis_name))
        else:
            mesh = Mesh(flat, axis_names=(axis_name, ))
        logger.info(
            "Stage 1D mesh (#4059) | axis=%s | stage %d column %d | device_ids=%s",
            axis_name, table.stage, table.column, stage_dev_ids)
    else:
        missing = [d for row in rows for d in row if d not in devices_by_id]
        if missing:
            raise RuntimeError(
                f"Cannot build the '{axis_name}' stage mesh because JAX does not "
                f"expose a device per rank of the slice: missing={missing}, "
                f"rows={rows}, jax_device_ids={sorted(devices_by_id)}")
        mesh = Mesh(
            np.asarray([[devices_by_id[d] for d in row] for row in rows]),
            axis_names=(STAGE_AXIS_NAME, axis_name),
        )
        logger.info("Stage mesh | axis=%s | this stage %d column %d | rows=%s",
                    axis_name, table.stage, table.column, rows)
    _MESH_CACHE[cache_key] = mesh
    return mesh


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
            "Could not map all CP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}")
    device_ids = tuple(rank_to_device_id[rank] for rank in ranks)
    layout = CpGroupLayout(
        ranks=ranks,
        device_ids=device_ids,
        rank_in_group=rank_in_group,
        world_size=int(group.world_size),
    )
    _LAYOUT_CACHE[cache_key] = layout
    return layout


def get_or_create_cp_mesh(axis_name: str, group: Any | None) -> Any:
    """Build a one-axis JAX mesh ordered by `group`.

    This helper is intentionally op-local: it does not replace the runner's
    normal single-device model mesh.

    The cache is keyed on `(axis_name, device_ids)`, so two axes coexist even
    when they cover the same devices -- which they do under `tp=N, dcp=N`.

    Under pipeline parallelism the group is one stage and the mesh spans the
    slice as (stage, group); see `get_or_create_stage_mesh`.
    """

    if get_stage_device_table() is not None:
        return get_or_create_stage_mesh(
            axis_name,
            None if group is None else int(group.world_size))
    layout = get_cp_group_layout(group)
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
            f"Cannot build the '{axis_name}' mesh because JAX does not "
            f"expose all of the group's TPU device ids: missing={missing}, "
            f"layout={layout}, "
            f"jax_device_ids={sorted(devices_by_id)}")

    mesh_devices = np.asarray(
        [devices_by_id[device_id] for device_id in layout.device_ids])
    mesh = Mesh(mesh_devices, axis_names=(axis_name, ))

    # This is the mesh the CP kernel runs on, so it is the only place that
    # sees the rank -> device correspondence the ring actually uses.
    # ``lax.axis_index(axis_name)`` inside the kernel returns a *position* in
    # this array, not a rank; it coincides with rank_in_group only because
    # device_ids above is built by walking layout.ranks in order. Logging the
    # two side by side makes a divergence visible instead of silent.
    # Cached per layout, so this logs once per worker rather than per step.
    positions = ", ".join(
        f"pos{position}=rank{rank}/dev{device_id}"
        f"@{tuple(getattr(device, 'coords', ()))}"
        f"c{getattr(device, 'core_on_chip', '?')}"
        for position, (rank, device_id, device) in enumerate(
            zip(layout.ranks, layout.device_ids, mesh_devices)))
    logger.info(
        "CP device mesh | axis=%s world_size=%d my_rank_in_group=%d | %s",
        axis_name, layout.world_size, layout.rank_in_group, positions)

    _MESH_CACHE[cache_key] = mesh
    return mesh
