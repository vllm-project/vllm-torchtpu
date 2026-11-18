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
"""Rank-ordered 2D ``('dp', 'tp')`` JAX mesh for multi-device SPMD ops.

``ep_mesh.build_ep_mesh`` sorts devices by hardware ``d.id`` by default, which
does not match ``global_rank = dp_rank * tp_size + tp_rank`` on multi-host TPU
slices. Reshaping a hardware-sorted 1D mesh to ``(dp_size, tp_size)`` would
scramble ``('dp', 'tp')`` coordinates relative to vLLM's ``dp_group`` and
``tp_group``.

Instead, ``build_dp_tp_mesh`` builds a PyTorch-rank-ordered 2D mesh with
``torch_tpu._internal.pallas.get_pallas_mesh(axis_names=('dp', 'tp'),
mesh_shape=(dp_size, tp_size))`` (as ``dcp.get_or_create_dcp_mesh`` does) and
checks on every worker that ``dp_group.ranks == mesh_ranks[:, tp_rank]`` and
``tp_group.ranks == mesh_ranks[dp_rank, :]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from jax.sharding import Mesh
from vllm.distributed import parallel_state as ps

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DP_AXIS_NAME = "dp"
TP_AXIS_NAME = "tp"
DP_TP_MESH_AXES: tuple[str, str] = (DP_AXIS_NAME, TP_AXIS_NAME)

_MESH_2D_CACHE: dict[tuple[int, int, tuple[int, ...]], Mesh] = {}


def _resolve_vllm_dp_tp_groups() -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Resolve ``(dp_size, tp_size, dp_ranks, tp_ranks)`` from vLLM parallel state."""
    if not ps.model_parallel_is_initialized():
        raise RuntimeError(
            "vLLM parallel_state is not initialized; pass explicit dp_size/tp_size "
            "(and dp_ranks/tp_ranks) when calling outside vLLM."
        )
    dp_ranks = tuple(int(r) for r in ps.get_dp_group().ranks)
    tp_ranks = tuple(int(r) for r in ps.get_tp_group().ranks)
    return len(dp_ranks), len(tp_ranks), dp_ranks, tp_ranks


def validate_dp_tp_group_ranks(
    dp_size: int,
    tp_size: int,
    *,
    dp_ranks: Sequence[int] | None = None,
    tp_ranks: Sequence[int] | None = None,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> tuple[int, int]:
    """Check that ``dp_ranks`` and ``tp_ranks`` match a ``(dp_size, tp_size)`` grid.

    Position ``(d, t)`` of the 2D mesh is PyTorch global rank
    ``d * tp_size + t``. Consequently:
      - ``dp_group.ranks`` must be the column
        ``[d * tp_size + tp_rank for d in range(dp_size)]``;
      - ``tp_group.ranks`` must be the row
        ``[dp_rank * tp_size + t for t in range(tp_size)]``.

    ``global_rank`` and ``world_size`` default to the torch.distributed values.

    Returns:
        ``(dp_rank, tp_rank)`` of ``global_rank`` in the 2D mesh.
    """
    if dp_size < 1 or tp_size < 1:
        raise ValueError(
            "dp_size and tp_size must be >= 1, got "
            f"dp_size={dp_size}, tp_size={tp_size}"
        )

    if (global_rank is None or world_size is None) and not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized; pass explicit global_rank "
            "and world_size."
        )
    rank = dist.get_rank() if global_rank is None else global_rank
    world = dist.get_world_size() if world_size is None else world_size

    expected_world = dp_size * tp_size
    if world != expected_world:
        raise ValueError(
            f"2D ('dp', 'tp') mesh requires world_size == dp_size * tp_size "
            f"({dp_size} * {tp_size} = {expected_world}), got world_size={world}."
        )
    if not (0 <= rank < world):
        raise ValueError(f"global_rank={rank} out of bounds for world_size={world}.")

    dp_rank, tp_rank = divmod(rank, tp_size)
    expected_dp_ranks = tuple(d * tp_size + tp_rank for d in range(dp_size))
    expected_tp_ranks = tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))

    if dp_ranks is not None:
        got_dp = tuple(int(r) for r in dp_ranks)
        if got_dp != expected_dp_ranks:
            raise ValueError(
                f"dp_group.ranks mismatch on global_rank={rank} "
                f"(dp_rank={dp_rank}, tp_rank={tp_rank}): got {got_dp}, "
                f"expected mesh_ranks[:, {tp_rank}]={expected_dp_ranks}."
            )

    if tp_ranks is not None:
        got_tp = tuple(int(r) for r in tp_ranks)
        if got_tp != expected_tp_ranks:
            raise ValueError(
                f"tp_group.ranks mismatch on global_rank={rank} "
                f"(dp_rank={dp_rank}, tp_rank={tp_rank}): got {got_tp}, "
                f"expected mesh_ranks[{dp_rank}, :]={expected_tp_ranks}."
            )

    return dp_rank, tp_rank


def _resolve_sizes_and_ranks(
    dp_size: int | None,
    tp_size: int | None,
    dp_ranks: Sequence[int] | None,
    tp_ranks: Sequence[int] | None,
) -> tuple[int, int, Sequence[int] | None, Sequence[int] | None]:
    """Fill unspecified sizes and group ranks from vLLM parallel state.

    Missing sizes require vLLM parallel state. With explicit sizes and no
    ranks, the ranks come from vLLM parallel state when it is initialized
    and are left unchecked otherwise.
    """
    if dp_size is None or tp_size is None:
        v_dp, v_tp, v_dp_ranks, v_tp_ranks = _resolve_vllm_dp_tp_groups()
        return (
            v_dp if dp_size is None else dp_size,
            v_tp if tp_size is None else tp_size,
            v_dp_ranks if dp_ranks is None else dp_ranks,
            v_tp_ranks if tp_ranks is None else tp_ranks,
        )
    if dp_ranks is None and tp_ranks is None and ps.model_parallel_is_initialized():
        _, _, dp_ranks, tp_ranks = _resolve_vllm_dp_tp_groups()
    return dp_size, tp_size, dp_ranks, tp_ranks


def build_dp_tp_mesh(
    dp_size: int | None = None,
    tp_size: int | None = None,
    *,
    dp_ranks: Sequence[int] | None = None,
    tp_ranks: Sequence[int] | None = None,
) -> Mesh:
    """Build (or return cached) rank-ordered 2D ``('dp', 'tp')`` JAX mesh.

    Must be pre-built outside ``torch.compile`` (e.g., during model/weight
    initialization) so ``jax.sharding.Mesh`` construction never enters a
    Dynamo-traced frame.
    """
    dp_size, tp_size, dp_ranks, tp_ranks = _resolve_sizes_and_ranks(
        dp_size, tp_size, dp_ranks, tp_ranks
    )

    if torch.compiler.is_compiling():
        for (cached_dp, cached_tp, _), cached_mesh in _MESH_2D_CACHE.items():
            if cached_dp == dp_size and cached_tp == tp_size:
                return cached_mesh
        raise RuntimeError(
            "build_dp_tp_mesh must be called before torch.compile tracing "
            f"(no cached mesh found for dp_size={dp_size}, tp_size={tp_size})."
        )

    dp_rank, tp_rank = validate_dp_tp_group_ranks(
        dp_size,
        tp_size,
        dp_ranks=dp_ranks,
        tp_ranks=tp_ranks,
    )
    for (cached_dp, cached_tp, _), cached_mesh in _MESH_2D_CACHE.items():
        if cached_dp == dp_size and cached_tp == tp_size:
            return cached_mesh

    from torch_tpu._internal.pallas import get_pallas_mesh

    mesh = get_pallas_mesh(axis_names=DP_TP_MESH_AXES, mesh_shape=(dp_size, tp_size))
    if tuple(mesh.axis_names) != DP_TP_MESH_AXES:
        raise RuntimeError(
            f"Unexpected mesh axis_names={mesh.axis_names}, expected {DP_TP_MESH_AXES}."
        )
    if tuple(mesh.devices.shape) != (dp_size, tp_size):
        raise RuntimeError(
            f"Unexpected 2D mesh shape={mesh.devices.shape}, "
            f"expected {(dp_size, tp_size)}."
        )

    flat_device_ids = tuple(int(d.id) for d in mesh.devices.flat)
    if len(set(flat_device_ids)) != len(flat_device_ids):
        raise RuntimeError(f"Duplicate TPU device IDs in 2D mesh: {flat_device_ids}")

    cache_key = (dp_size, tp_size, flat_device_ids)
    if cache_key not in _MESH_2D_CACHE:
        logger.info(
            "Built (dp, tp) mesh | shape=(dp=%d, tp=%d) | rank=%d -> (dp=%d, tp=%d) "
            "| dev_id=%d | flat_device_ids=%s",
            dp_size,
            tp_size,
            dp_rank * tp_size + tp_rank,
            dp_rank,
            tp_rank,
            int(mesh.devices[dp_rank, tp_rank].id),
            list(flat_device_ids),
        )
        _MESH_2D_CACHE[cache_key] = mesh

    return _MESH_2D_CACHE[cache_key]
