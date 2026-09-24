# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.distributed.mesh_utils import CpGroupLayout, get_cp_group_layout
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PcpGroupLayout = CpGroupLayout

# PCP builds its own mesh rather than going through
# ``mesh_utils.get_or_create_cp_mesh``, so it needs its own cache: the two
# builders produce different objects for the same key -- abstract
# ``_PallasDevice`` positions here, real TPU devices there -- and sharing one
# dict between them would hand a caller whichever ran first.
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


def get_or_create_pcp_mesh(axis_name: str = "pcp") -> Any:
    """Build a JAX mesh ordered by the current vLLM PCP group.

    This helper is intentionally op-local: it does not replace the runner's
    normal single-device model mesh.

    The mesh is built by Torchtpu's ``get_pallas_mesh``, which orders devices
    by PyTorch rank. It does not consult ``jax.devices()``, so it neither
    requires JAX to have a view of every peer's device nor forces JAX to
    initialize a hardware runtime just to compile a kernel. Partition ``i``
    inside the kernel is rank ``i`` -- that is what google-pytorch/torch_tpu#3522
    guarantees -- so ``lax.axis_index(axis_name)`` is ``rank_in_group`` as long
    as the PCP group is the whole world.
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

        mesh = Mesh(np.asarray(jax.local_devices()[:1]), axis_names=(axis_name,))
        _MESH_CACHE[cache_key] = mesh
        return mesh

    import torch.distributed as dist

    # get_pallas_mesh spans the whole torch.distributed world; it has no way to
    # describe a sub-group. Every supported deployment runs PCP as the only
    # axis in its process, so the two coincide -- but say so, because a silent
    # mismatch would build a mesh of the wrong size.
    world_size = int(dist.get_world_size())
    if layout.world_size != world_size:
        raise NotImplementedError(
            f"The PCP group has world_size={layout.world_size} inside a "
            f"torch.distributed world of {world_size}. get_pallas_mesh builds "
            "a mesh over the whole world, so PCP alongside another "
            "parallelism axis needs a mesh_shape factorisation that has not "
            "been worked out."
        )

    from torch_tpu._internal.pallas import get_pallas_mesh

    mesh = get_pallas_mesh(axis_names=(axis_name,))

    # The mesh holds abstract _PallasDevice objects numbered by rank, not TPU
    # devices, so there is no device order here to check: position i is rank i
    # by construction of PallasMesh. What the kernels additionally need is that
    # the group's own rank order is the identity, because the host shards by
    # rank_in_group while the kernel indexes by rank.
    if layout.ranks != tuple(range(layout.world_size)):
        raise RuntimeError(
            "The PCP group is not made of ranks 0..N-1 in order "
            f"({layout.ranks}), so rank_in_group is not the PyTorch rank that "
            "torch_tpu#3522 binds partition indices to, and the host's token "
            "split would not match the kernel's view."
        )

    # Log rank and device id for each position in the mesh. Logs once per
    # worker.
    rank_and_device = zip(layout.ranks, layout.device_ids)
    positions = ", ".join(
        f"pos{i}=rank{rank}/dev{dev}" for i, (rank, dev) in enumerate(rank_and_device)
    )
    logger.info(
        "PCP device mesh | axis=%s world_size=%d my_rank_in_group=%d | %s",
        axis_name,
        layout.world_size,
        layout.rank_in_group,
        positions,
    )

    _MESH_CACHE[cache_key] = mesh
    return mesh
