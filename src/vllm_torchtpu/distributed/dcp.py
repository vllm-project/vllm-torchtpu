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
"""DCP (Decode Context Parallelism) process-group accessors.

Mirrors the role of distributed/pcp.py for PCP: a narrow, safe accessor
layer every call site shares, rather than each re-deriving group state
independently. The group layout the two axes have in common lives in
mesh_utils.py.

"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.distributed.mesh_utils import CpGroupLayout, get_cp_group_layout
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The JAX mesh axis the DCP kernels shard the KV cache over.
DCP_AXIS_NAME = "dcp"
# The TP lanes DCP does not shard over. No kernel collects over it; it is
# there so the mesh spans the world, which torch_tpu requires of a Pallas
# kernel mesh.
DCP_LANE_AXIS_NAME = "dcp_lane"

# DCP builds its own mesh rather than going through Torchtpu-free helpers, so
# it keeps its own cache for the same reason pcp.py does: the key is the mesh
# shape and the group, not a device list.
_MESH_CACHE: dict[tuple, Any] = {}


def get_dcp_group() -> Any | None:
    """Return vLLM's DCP GroupCoordinator, or None if not initialized."""
    try:
        from vllm.distributed.parallel_state import get_dcp_group as _get_group

        return _get_group()
    except (AssertionError, ImportError):
        return None


def get_dcp_rank() -> int:
    group = get_dcp_group()
    return 0 if group is None else int(group.rank_in_group)


_RANK_OPERANDS: dict[str, Any] = {}


def dcp_rank_operand(device: Any = None) -> Any:
    """``TPU_DCP_RANK_OPERAND``: this rank's DCP index as an int32 ``[1]`` tensor.

    Passed to a DCP Pallas op under ``PartitionSpec(DCP_AXIS_NAME)``, the
    per-rank tensors form the global ``i32[dcp_size]`` with entry r = r, so
    each shard reads its own rank as data. That replaces
    ``cp_rank_as_data``'s ``all_to_all`` of an iota (a collective once per
    program, ~8 us on 2x2x2 TP16xDCP16) without ``lax.axis_index``, whose
    ``partition-id`` the torch_tpu export path refuses. Every rank compiles
    the same program; only the operand's value differs. Built once per device
    (call it at op-build time, outside Dynamo).
    """
    import torch

    if device is None:
        device = torch.device("tpu")
    key = str(device)
    t = _RANK_OPERANDS.get(key)
    if t is None:
        t = torch.tensor([get_dcp_rank()], dtype=torch.int32, device=device)
        _RANK_OPERANDS[key] = t
    return t


def get_dcp_world_size() -> int:
    group = get_dcp_group()
    return 1 if group is None else int(group.world_size)


def get_dcp_group_layout() -> CpGroupLayout:
    """The DCP group's rank order and the TPU device each rank sits on."""
    return get_cp_group_layout(get_dcp_group())


def _validate_dcp_rank_alignment(group: Any, dcp_size: int) -> None:
    """Check DCP is the fastest-varying factor of the global rank.

    ``get_pallas_mesh`` numbers mesh positions by PyTorch rank, so a
    ``(lane, dcp)`` mesh only describes the DCP grouping when
    ``global_rank == lane * dcp_size + dcp_rank``. If it does not, the mesh
    still builds and the kernels still run -- they collect over the wrong
    partners, which is a wrong answer rather than an error.
    """
    import torch.distributed as dist

    rank = int(dist.get_rank())
    lane, dcp_rank = divmod(rank, dcp_size)
    expected = tuple(range(lane * dcp_size, (lane + 1) * dcp_size))
    actual = tuple(int(value) for value in group.ranks)
    if actual != expected or int(group.rank_in_group) != dcp_rank:
        raise RuntimeError(
            "DCP groups must be contiguous runs of ranks so the mesh's last "
            f"axis is the DCP axis: global rank {rank} with "
            f"dcp_size={dcp_size} expects ranks {expected} at position "
            f"{dcp_rank}, got {actual} at position {group.rank_in_group}."
        )


def _single_device_dcp_mesh(axis_names: tuple[str, str]) -> Any:
    """The mesh before distributed init: one device, both axes length 1."""
    import jax
    import numpy as np
    from jax.sharding import Mesh

    devices = np.asarray(jax.local_devices()[:1]).reshape(1, 1)
    return Mesh(devices, axis_names=axis_names)


def get_or_create_dcp_mesh(axis_name: str | None = None) -> Any:
    """Build the full-world ``(lane, DCP)`` mesh the DCP Pallas ops run on.

    torch_tpu requires a Pallas kernel's mesh to span the whole world, so the
    mesh carries a leading axis for the TP lanes DCP does not shard over. The
    kernels still collect only over `axis_name`, which gives one independent
    DCP group per lane. Under `dcp == tp` the shape is `(1, world)` and the
    device order is the one the single-axis builder used to produce.

    The axis name defaults to `DCP_AXIS_NAME`; the kernels take the same
    constant from here, so a caller that overrides it has to override it in
    both places or the mesh and the partition specs will not agree.
    """
    import torch.distributed as dist

    if axis_name is None:
        axis_name = DCP_AXIS_NAME
    axis_names = (DCP_LANE_AXIS_NAME, axis_name)
    group = get_dcp_group()
    if group is None or not dist.is_initialized():
        return _single_device_dcp_mesh(axis_names)

    dcp_size = int(group.world_size)
    world_size = int(dist.get_world_size())
    if dcp_size <= 0 or world_size % dcp_size:
        raise NotImplementedError(
            f"dcp_size={dcp_size} must divide the torch.distributed world of "
            f"{world_size}."
        )
    _validate_dcp_rank_alignment(group, dcp_size)

    shape = (world_size // dcp_size, dcp_size)
    ranks = tuple(int(value) for value in group.ranks)
    cache_key = (axis_names, shape, ranks)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from torch_tpu._internal.pallas import get_pallas_mesh

    # No device array: get_pallas_mesh numbers mesh positions by PyTorch rank,
    # which is the binding torch_tpu#3522 guarantees.
    mesh = get_pallas_mesh(axis_names=axis_names, mesh_shape=shape)
    logger.info(
        "DCP device mesh | axes=%s shape=%s rank_in_group=%d ranks=%s",
        axis_names,
        shape,
        int(group.rank_in_group),
        ranks,
    )
    _MESH_CACHE[cache_key] = mesh
    return mesh
