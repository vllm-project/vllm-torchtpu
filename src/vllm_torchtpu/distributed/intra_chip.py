# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""A process group spanning the cores of one physical chip.

Collectives over this group stay on-package, so they are far cheaper than the
same collective over the full TP group. That makes it worth sharding a weight
across a chip's cores even when the resulting exchange happens every layer:
the halved HBM read beats the added on-chip hop.

The group is deliberately separate from vLLM's TP group and from
:mod:`vllm_torchtpu.layers.adapter.moe_hierarchical`, which only rewrites the
*numbers* in ``FusedMoEParallelConfig`` and keeps reducing over all of TP.
"""

from __future__ import annotations

import collections

from vllm_torchtpu.distributed.chip_topology import get_chip_topology
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_group = None
_initialized = False


def _chip_rank_groups() -> list[list[int]] | None:
    """Global ranks grouped by the chip they sit on, or ``None`` if unusable.

    Returns ``None`` rather than raising whenever the grouping cannot be
    trusted; every caller's fallback is to leave the weight replicated, which
    is always correct.
    """
    topology = get_chip_topology()
    if topology is None:
        return None
    if topology.cores_per_chip < 2:
        logger.warning(
            "Only %d core(s) per chip; nothing to split within a chip.",
            topology.cores_per_chip,
        )
        return None

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None

    return _group_ranks_by_chip(topology, _rank_to_device())


def _rank_to_device() -> dict[int, int]:
    """Global rank -> the TPU device id it runs on. Collective over the world.

    Global rank need not follow the runtime's device order: under DP each
    engine's workers are placed separately (Ray hands out whole chips in any
    order), so each rank reports its own device.
    """
    from vllm.distributed.parallel_state import get_world_group

    from vllm_torchtpu.distributed.mesh_utils import (
        _collect_rank_to_device_id,
        _get_current_global_rank,
        _get_tpu_global_device_id,
    )

    return _collect_rank_to_device_id(
        get_world_group(), _get_current_global_rank(), _get_tpu_global_device_id()
    )


def _group_ranks_by_chip(
    topology, rank_to_device: dict[int, int]
) -> list[list[int]] | None:
    """Ranks grouped by the chip their device sits on, or ``None`` if unusable."""
    chip_of_device = {d: chip for chip, ids in enumerate(topology.chips) for d in ids}
    # One rank per device of the topology, no more and no fewer.
    if sorted(rank_to_device.values()) != sorted(chip_of_device):
        logger.warning(
            "Ranks run on devices %s, not exactly the devices of %s; refusing to "
            "guess the intra-chip grouping.",
            sorted(rank_to_device.values()),
            topology.describe(),
        )
        return None

    by_chip: dict[int, list[int]] = collections.defaultdict(list)
    for rank, device in rank_to_device.items():
        by_chip[chip_of_device[device]].append(rank)

    groups = [sorted(by_chip[chip]) for chip in sorted(by_chip)]
    sizes = {len(g) for g in groups}
    if sizes != {topology.cores_per_chip}:
        logger.warning("Chip groups are ragged (%s); not usable.", sorted(sizes))
        return None
    return groups


def get_intra_chip_group():
    """The ``GroupCoordinator`` over this rank's chip, or ``None``.

    Built once, on first call. Every rank must reach this at the same point in
    model construction -- creating a process group is collective -- which holds
    because all ranks build the same layers in the same order.
    """
    global _group, _initialized
    if _initialized:
        return _group
    _initialized = True

    groups = _chip_rank_groups()
    if groups is None:
        return None

    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_model_parallel_group,
    )

    world = get_world_group()
    backend = dist.get_backend(world.device_group)
    _group = init_model_parallel_group(
        groups,
        world.local_rank,
        backend,
        group_name="intra_chip",
    )
    logger.info(
        "Intra-chip group ready: %d chip(s) x %d core(s), this rank is "
        "%d/%d within its chip; rank groups %s.",
        len(groups),
        len(groups[0]),
        _group.rank_in_group,
        _group.world_size,
        groups,
    )
    return _group
