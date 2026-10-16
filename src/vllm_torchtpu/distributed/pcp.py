# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.distributed.mesh_utils import (
    CpGroupLayout,
    get_cp_group_layout,
    get_pipeline_stage,
    get_stage_mesh,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PcpGroupLayout = CpGroupLayout

# Each CP axis builds its own Pallas mesh and keys the cache on its own
# factorization of the world, so each keeps its own dict rather than sharing
# one that would hand a caller whichever axis ran first.
_MESH_CACHE: dict[tuple, Any] = {}


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


def all_gather_equal_tokens(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
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


def get_pcp_group_layout() -> PcpGroupLayout:
    """Return the current PCP group's rank order and TPU device-id order.

    The order of ``device_ids`` mirrors ``get_pcp_group().ranks``. This is the
    order a PCP Pallas mesh should use so TP+PCP layouts form one PCP ring per
    TP lane.
    """
    return get_cp_group_layout(get_pcp_group())


def get_or_create_pcp_mesh(
    axis_name: str = "pcp", *, tp_axis_name: str | None = None
) -> Any:
    """Build a rank-aligned PCP x TP mesh over the TorchTPU world.

    Partition ``pcp_rank * tp_size + tp_rank`` is the global PyTorch rank.
    Validate the actual vLLM groups rather than silently accepting reordered
    groups that disagree with that runtime assignment.

    Under pipeline parallelism the mesh covers this worker's stage, whose
    ranks follow those of the stages before it.
    """
    import torch.distributed as dist
    from torch_tpu._internal.pallas import get_pallas_mesh

    layout = get_pcp_group_layout()
    group = get_pcp_group()
    axis_names = (axis_name,) if tp_axis_name is None else (axis_name, tp_axis_name)
    if len(set(axis_names)) != len(axis_names):
        raise ValueError("PCP and TP mesh axis names must be distinct.")
    tp_size = 1
    tp_rank = 0
    tp_group = None
    if tp_axis_name is not None and group is not None:
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        tp_rank = int(tp_group.rank_in_group)
    shape = (
        (layout.world_size,) if tp_axis_name is None else (layout.world_size, tp_size)
    )
    cache_key = (
        axis_names,
        shape,
        layout.ranks,
        layout.device_ids,
        () if tp_group is None else tuple(tp_group.ranks),
    )
    if cache_key in _MESH_CACHE:
        return _MESH_CACHE[cache_key]
    if group is None:
        import jax
        import numpy as np
        from jax.sharding import Mesh

        mesh = Mesh(
            np.asarray(jax.local_devices()[:1]).reshape(shape), axis_names=axis_names
        )
    else:
        world_size = int(dist.get_world_size())
        stage, stages = get_pipeline_stage()
        stage_size = layout.world_size * tp_size
        if stage_size * stages != world_size:
            raise NotImplementedError(
                f"PCP world_size={layout.world_size} x TP={tp_size} does not "
                f"cover the torch.distributed world={world_size}."
            )
        if stages > 1 and tp_size > 1:
            raise NotImplementedError(
                "A pipeline stage spans a PCP group or a TP group, not both: "
                f"PCP world_size={layout.world_size}, TP={tp_size}."
            )
        stage_start = stage * stage_size
        expected_pcp_ranks = tuple(
            range(stage_start + tp_rank, stage_start + stage_size, tp_size)
        )
        if layout.ranks != expected_pcp_ranks:
            raise RuntimeError(
                "PCP group must follow ranks 0..N-1 reshaped as PCP x TP: "
                f"expected={expected_pcp_ranks}, actual={layout.ranks}."
            )
        if tp_group is not None:
            first = stage_start + layout.rank_in_group * tp_size
            expected_tp_ranks = tuple(range(first, first + tp_size))
            if (
                tuple(tp_group.ranks) != expected_tp_ranks
                or int(dist.get_rank()) != first + tp_rank
            ):
                raise RuntimeError(
                    "TP group coordinates disagree with the rank-aligned "
                    f"PCP x TP mesh: expected={expected_tp_ranks}, "
                    f"actual={tuple(tp_group.ranks)}."
                )
        if stages > 1:
            mesh = get_stage_mesh(axis_names, shape, group, layout.device_ids)
        else:
            mesh = (
                get_pallas_mesh(axis_names=axis_names)
                if tp_axis_name is None
                else get_pallas_mesh(axis_names=axis_names, mesh_shape=shape)
            )
    logger.info(
        "PCP device mesh | axes=%s shape=%s PCP ranks=%s device_ids=%s",
        axis_names,
        shape,
        layout.ranks,
        layout.device_ids,
    )
    _MESH_CACHE[cache_key] = mesh
    return mesh
