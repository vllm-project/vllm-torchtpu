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
"""Op-local JAX mesh over the tensor-parallel group in ascending device-id order.

Mirrors `ep_mesh.build_ep_mesh` for dense models where vLLM creates no EP
group. On dual-core tpu7x hosts, local ranks 0..7 map to global TPU device ids
`[0, 1, 4, 5, 6, 7, 2, 3]`, so `build_tp_mesh` returns both the ascending
device-id `Mesh` and `tp_rank_order`, where `tp_rank_order[i]` is the TP rank
residing at mesh index `i`.
"""

from typing import Any

import torch.distributed as dist
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed.mesh_utils import (
    _collect_rank_to_device_id,
    _get_current_global_rank,
    _get_tpu_global_device_id,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TP_AXIS_NAME = "tp"

_MESH_CACHE: dict[tuple[str, tuple[int, ...]], tuple[Any, tuple[int, ...]]] = {}
# TP group ranks -> the device ids `build_tp_mesh` gathered for them, in
# TP-rank order.
_GATHERED_DEVICE_IDS: dict[tuple[int, ...], tuple[int, ...]] = {}
# Set once `tp_device_placement_key` has been read.
_PLACEMENT_KEY_READ = False


def tp_device_placement_key() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Every (TP ranks, device ids) pair `build_tp_mesh` has gathered here.

    Programs built over the TP mesh compile `tp_rank_order` in as a constant,
    which can differ across dense DP replicas that share a cache directory or
    across restarts even when sorted mesh device ids are identical (`0..N-1`).
    The compile cache keys on this so each placement compiles into its own
    cache entry. Mirrors `ep_mesh.ep_device_placement_key`.
    """
    global _PLACEMENT_KEY_READ
    _PLACEMENT_KEY_READ = True
    return tuple(sorted(_GATHERED_DEVICE_IDS.items()))


def build_tp_mesh(
    axis_name: str = TP_AXIS_NAME,
) -> tuple[Any, tuple[int, ...]] | None:
    """Returns `(mesh, tp_rank_order)`, or `None` if TP does not span the world.

    Runs a CPU-group collective on cache miss, so callers must invoke this
    outside Dynamo/JAX tracing (for example, during model construction) and
    before `tp_device_placement_key()` is read by the compile cache.
    """
    if not dist.is_available() or not dist.is_initialized():
        return None
    group = parallel_state.get_tp_group()
    ranks = tuple(int(r) for r in group.ranks)
    if len(ranks) != int(dist.get_world_size()):
        # get_pallas_mesh spans the entire world; TP alongside DP/PP would
        # require a multi-axis mesh_shape factorization.
        return None

    cache_key = (axis_name, ranks)
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rank_to_device_id = _collect_rank_to_device_id(
        group, _get_current_global_rank(), _get_tpu_global_device_id()
    )
    missing = [r for r in ranks if r not in rank_to_device_id]
    if missing:
        raise RuntimeError(
            "Could not map all TP ranks to TPU global device ids: "
            f"missing={missing}, ranks={ranks}, gathered={rank_to_device_id}"
        )
    unordered = tuple(rank_to_device_id[r] for r in ranks)
    if len(set(unordered)) != len(unordered):
        raise RuntimeError(
            f"Multiple TP ranks report the same TPU global device id: {unordered}"
        )
    if _PLACEMENT_KEY_READ and _GATHERED_DEVICE_IDS.get(ranks) != unordered:
        raise RuntimeError(
            "TP device placement gathered after the compile cache key was taken, "
            "so the key does not cover it. Either a TP mesh is built after "
            "model load, or the compile cache key is read before model load. "
            f"ranks={ranks}, device_ids={unordered}"
        )
    order = tuple(sorted(range(len(ranks)), key=unordered.__getitem__))
    device_ids = tuple(unordered[r] for r in order)

    import numpy as np
    from jax.sharding import Mesh
    from torch_tpu._internal.pallas import get_pallas_mesh

    by_id = {
        int(d.id): d for d in get_pallas_mesh(axis_names=(axis_name,)).devices.flat
    }
    if sorted(by_id) != list(device_ids):
        raise RuntimeError(
            "get_pallas_mesh device ids do not match the TP group's: "
            f"pallas={sorted(by_id)}, tp_device_ids={list(device_ids)}"
        )
    mesh = Mesh(np.asarray([by_id[d] for d in device_ids]), axis_names=(axis_name,))
    logger.info(
        "Built TP mesh | axis=%s | device_ids=%s | tp_rank_order=%s",
        axis_name,
        list(device_ids),
        order,
    )
    _GATHERED_DEVICE_IDS[ranks] = unordered
    _MESH_CACHE[cache_key] = (mesh, order)
    return mesh, order
