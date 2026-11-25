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

Instead, ``build_dp_tp_mesh`` relies on
``torch.tpu.pallas.get_pallas_mesh(axis_names=('dp', 'tp'),
mesh_shape=(dp_size, tp_size))`` for PyTorch-rank-ordered device placement (as
``dcp.get_or_create_dcp_mesh`` does); this module only checks on every worker
that vLLM's ``dp_group`` and ``tp_group`` match that layout
(``dp_group.ranks == mesh_ranks[:, tp_rank]`` and
``tp_group.ranks == mesh_ranks[dp_rank, :]``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
from jax.sharding import Mesh
from vllm.distributed import parallel_state

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DP_AXIS_NAME = "dp"
TP_AXIS_NAME = "tp"
DP_TP_MESH_AXES: tuple[str, str] = (DP_AXIS_NAME, TP_AXIS_NAME)

_MESH_2D_CACHE: dict[tuple[int, int], Mesh] = {}


def _validate_dp_tp_group_ranks(
    dp_size: int,
    tp_size: int,
    dp_ranks: Sequence[int],
    tp_ranks: Sequence[int],
) -> tuple[int, int]:
    """Check that ``dp_ranks`` and ``tp_ranks`` match a ``(dp_size, tp_size)`` grid.

    Position ``(d, t)`` of the 2D mesh is PyTorch global rank
    ``d * tp_size + t``. Consequently:
      - ``dp_group.ranks`` must be the column
        ``[d * tp_size + tp_rank for d in range(dp_size)]``;
      - ``tp_group.ranks`` must be the row
        ``[dp_rank * tp_size + t for t in range(tp_size)]``.

    Returns:
        ``(dp_rank, tp_rank)`` of ``global_rank`` in the 2D mesh.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized.")

    global_rank = int(dist.get_rank())
    global_world = int(dist.get_world_size())

    expected_world = dp_size * tp_size
    if global_world != expected_world:
        raise ValueError(
            "2D ('dp', 'tp') mesh requires world_size == dp_size * tp_size "
            f"({dp_size} * {tp_size} = {expected_world}), "
            f"got world_size={global_world}."
        )

    dp_rank, tp_rank = divmod(global_rank, tp_size)
    expected_dp_ranks = tuple(d * tp_size + tp_rank for d in range(dp_size))
    expected_tp_ranks = tuple(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))

    got_dp = tuple(int(r) for r in dp_ranks)
    if got_dp != expected_dp_ranks:
        raise ValueError(
            f"dp_group.ranks mismatch on global_rank={global_rank} "
            f"(dp_rank={dp_rank}, tp_rank={tp_rank}): got {got_dp}, "
            f"expected mesh_ranks[:, {tp_rank}]={expected_dp_ranks}."
        )

    got_tp = tuple(int(r) for r in tp_ranks)
    if got_tp != expected_tp_ranks:
        raise ValueError(
            f"tp_group.ranks mismatch on global_rank={global_rank} "
            f"(dp_rank={dp_rank}, tp_rank={tp_rank}): got {got_tp}, "
            f"expected mesh_ranks[{dp_rank}, :]={expected_tp_ranks}."
        )

    return dp_rank, tp_rank


def build_dp_tp_mesh() -> Mesh:
    """Build (or return cached) rank-ordered 2D ``('dp', 'tp')`` JAX mesh.

    Must be pre-built outside ``torch.compile`` (e.g., during model/weight
    initialization) so ``jax.sharding.Mesh`` construction never enters a
    Dynamo-traced frame.
    """
    if not parallel_state.model_parallel_is_initialized():
        raise RuntimeError("vLLM parallel_state is not initialized.")

    try:
        dp_group = parallel_state.get_dp_group()
        tp_group = parallel_state.get_tp_group()
    except AssertionError as exc:
        raise RuntimeError("vLLM DP/TP process groups are not initialized.") from exc

    dp_ranks = tuple(int(r) for r in dp_group.ranks)
    tp_ranks = tuple(int(r) for r in tp_group.ranks)
    dp_size = len(dp_ranks)
    tp_size = len(tp_ranks)
    cache_key = (dp_size, tp_size)

    if torch.compiler.is_compiling():
        cached = _MESH_2D_CACHE.get(cache_key)
        if cached is not None:
            return cached
        raise RuntimeError(
            "build_dp_tp_mesh must be called before torch.compile tracing "
            f"(no cached mesh found for dp_size={dp_size}, tp_size={tp_size})."
        )

    dp_rank, tp_rank = _validate_dp_tp_group_ranks(dp_size, tp_size, dp_ranks, tp_ranks)
    cached = _MESH_2D_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from torch.tpu.pallas import get_pallas_mesh

    mesh = get_pallas_mesh(axis_names=DP_TP_MESH_AXES, mesh_shape=(dp_size, tp_size))
    flat_device_ids = [int(d.id) for d in mesh.devices.flat]
    logger.info(
        "Built (dp, tp) mesh | shape=(dp=%d, tp=%d) | rank=%d -> (dp=%d, tp=%d) "
        "| dev_id=%d | flat_device_ids=%s",
        dp_size,
        tp_size,
        dp_rank * tp_size + tp_rank,
        dp_rank,
        tp_rank,
        int(mesh.devices[dp_rank, tp_rank].id),
        flat_device_ids,
    )
    _MESH_2D_CACHE[cache_key] = mesh
    return mesh
