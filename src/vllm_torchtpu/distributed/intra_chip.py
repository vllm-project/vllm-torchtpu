# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A process group spanning the cores of one physical chip.

Collectives over this group stay on-package, so they are far cheaper than the
same collective over the full TP group. That makes it worth sharding a weight
across a chip's cores even when the resulting exchange happens every layer:
the halved HBM read beats the added on-chip hop.

The group is deliberately separate from vLLM's TP group and from
:mod:`vllm_torchtpu.layers.vllm.moe_hierarchical`, which only rewrites the
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
            topology.cores_per_chip)
        return None

    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return None

    world_size = dist.get_world_size()
    if world_size != len(topology.device_ids):
        # `chip_of` indexes the runtime's device order, which only coincides
        # with the global rank when TP spans every visible device (no DP/PP).
        # Anything else would silently pair up the wrong cores.
        logger.warning(
            "World size %d does not match the %d visible devices; refusing to "
            "guess the intra-chip grouping.", world_size,
            len(topology.device_ids))
        return None

    by_chip: dict[int, list[int]] = collections.defaultdict(list)
    for rank in range(world_size):
        chip_index, _ = topology.chip_of(rank)
        by_chip[chip_index].append(rank)

    groups = [sorted(by_chip[chip]) for chip in sorted(by_chip)]
    sizes = {len(g) for g in groups}
    if sizes != {topology.cores_per_chip}:
        logger.warning("Chip groups are ragged (%s); not usable.",
                       sorted(sizes))
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
    from vllm.distributed.parallel_state import (get_world_group,
                                                 init_model_parallel_group)

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
        "%d/%d within its chip.", len(groups), len(groups[0]),
        _group.rank_in_group, _group.world_size)
    return _group
