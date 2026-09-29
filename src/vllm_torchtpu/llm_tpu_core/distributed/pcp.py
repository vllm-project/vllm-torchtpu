# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Decoupled PCP distributed primitives and rank-aligned device mesh.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.llm_tpu_core.distributed.collectives import (
    all_gather_tensor as _core_all_gather_tensor,
)
from vllm_torchtpu.llm_tpu_core.distributed.collectives import (
    all_reduce_sum as _core_all_reduce_sum,
)
from vllm_torchtpu.llm_tpu_core.distributed.group_provider import (
    get_pcp_group as _get_pcp_group_from_provider,
)
from vllm_torchtpu.llm_tpu_core.distributed.group_provider import (
    get_tp_group as _get_tp_group_from_provider,
)
from vllm_torchtpu.llm_tpu_core.distributed.mesh_utils import (
    CpGroupLayout,
    get_cp_group_layout,
)
from vllm_torchtpu.llm_tpu_core.logger import init_logger

logger = init_logger(__name__)

PcpGroupLayout = CpGroupLayout

_MESH_CACHE: dict[tuple, Any] = {}


def get_pcp_group() -> Any | None:
    """Return active PCP process group, or None before distributed init."""
    return _get_pcp_group_from_provider()


def get_pcp_rank() -> int:
    group = get_pcp_group()
    return 0 if group is None else int(getattr(group, "rank_in_group", 0))


def get_pcp_world_size() -> int:
    group = get_pcp_group()
    return 1 if group is None else int(getattr(group, "world_size", 1))


def all_gather_equal_tokens(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Gather equal-shaped tensors across the active PCP group."""
    group = get_pcp_group()
    if group is None or int(getattr(group, "world_size", 1)) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return _core_all_gather_tensor(tensor, dim=dim, group=group)


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across the active PCP group."""
    group = get_pcp_group()
    if group is None or int(getattr(group, "world_size", 1)) == 1:
        return tensor
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return _core_all_reduce_sum(tensor, group=group)


def get_pcp_group_layout() -> PcpGroupLayout:
    """Return the current PCP group's rank order and TPU device-id order."""
    return get_cp_group_layout(get_pcp_group())


def get_or_create_pcp_mesh(
    axis_name: str = "pcp", *, tp_axis_name: str | None = None
) -> Any:
    """Build a rank-aligned PCP x TP mesh over the TorchTPU world."""
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
        tp_group = _get_tp_group_from_provider()
        if tp_group is not None:
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
        if layout.world_size * tp_size != world_size:
            raise NotImplementedError(
                f"PCP world_size={layout.world_size} x TP={tp_size} does not "
                f"cover the torch.distributed world={world_size}."
            )
        expected_pcp_ranks = tuple(range(tp_rank, world_size, tp_size))
        if layout.ranks != expected_pcp_ranks:
            raise RuntimeError(
                "PCP group must follow ranks 0..N-1 reshaped as PCP x TP: "
                f"expected={expected_pcp_ranks}, actual={layout.ranks}."
            )
        if tp_group is not None:
            first = layout.rank_in_group * tp_size
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


__all__ = [
    "PcpGroupLayout",
    "all_gather_equal_tokens",
    "all_reduce_sum",
    "get_or_create_pcp_mesh",
    "get_pcp_group",
    "get_pcp_group_layout",
    "get_pcp_rank",
    "get_pcp_world_size",
]
