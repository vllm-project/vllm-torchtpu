# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functions for finding which devices share a physical chip."""

from __future__ import annotations

import collections
import functools

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class ChipTopology:
    """The chip/chiplet grouping of the devices this process can see."""

    def __init__(self, chips: list[list[int]], device_ids: list[int]) -> None:
        self.chips = chips  # chip index -> device ids on it, sorted
        self.device_ids = device_ids  # global device order
        self.num_chips = len(chips)
        self.cores_per_chip = len(chips[0]) if chips else 0

    def chip_of(self, rank: int) -> tuple[int, int]:
        """``(chip_index, core_on_chip)`` for a global device rank.

        ``rank`` indexes ``device_ids`` -- i.e. it is the position in the
        runtime's own device order, which is what a vLLM TP rank corresponds
        to.
        """
        if not 0 <= rank < len(self.device_ids):
            raise ValueError(
                f"rank {rank} outside the {len(self.device_ids)} visible devices"
            )
        device_id = self.device_ids[rank]
        for chip_index, ids in enumerate(self.chips):
            if device_id in ids:
                return chip_index, ids.index(device_id)
        raise RuntimeError(f"device {device_id} is on no chip; chips={self.chips}")

    def describe(self) -> str:
        return (
            f"{self.num_chips} chip(s) x {self.cores_per_chip} core(s): "
            + ", ".join(f"chip{i}={ids}" for i, ids in enumerate(self.chips))
        )


@functools.lru_cache(maxsize=1)
def get_chip_topology() -> ChipTopology | None:
    """Group the visible devices by physical chip, or ``None`` if unknowable.

    Returns ``None`` rather than raising when the devices do not expose
    ``coords``: the caller's job is then to fall back to flat parallelism, not
    to fail. Cached because it initializes the runtime's device list.
    """
    try:
        import jax

        devices = jax.devices()
    except Exception as error:  # noqa: BLE001 - optional capability probe
        logger.warning("Cannot enumerate devices for chip topology: %s", error)
        return None

    if not devices:
        return None

    grouped: dict[tuple, list[int]] = collections.defaultdict(list)
    for device in devices:
        coords = getattr(device, "coords", None)
        if coords is None:
            logger.warning(
                "Devices expose no `coords`; chip grouping unavailable, so "
                "hierarchical MoE parallelism cannot be enabled."
            )
            return None
        grouped[tuple(coords)].append(int(device.id))

    # Order chips by their mesh coordinates so the grouping is deterministic
    # across ranks -- every rank must agree on which chip is EP rank 0, and
    # dict insertion order follows device enumeration, which need not match.
    chips = [sorted(grouped[key]) for key in sorted(grouped)]

    sizes = {len(ids) for ids in chips}
    if len(sizes) != 1:
        logger.warning(
            "Chips hold differing core counts (%s); chip grouping "
            "unusable for hierarchical parallelism.",
            sorted(sizes),
        )
        return None

    topology = ChipTopology(chips, [int(d.id) for d in devices])
    logger.info("Chip topology: %s", topology.describe())
    return topology


def hierarchical_moe_split(
    world_size: int, rank: int
) -> tuple[int, int, int, int] | None:
    """``(ep_size, ep_rank, tp_size, tp_rank)`` for hierarchical MoE.

    Expert parallelism runs *between* chips and tensor parallelism *within*
    one, so ``ep_size`` is the chip count and ``tp_size`` the cores per chip.
    Returns ``None`` when the split does not apply -- a topology that cannot be
    read, a single core per chip, or a world size that is not the full visible
    device set (a partial world would make "which chip" ambiguous).
    """
    topology = get_chip_topology()
    if topology is None:
        return None
    if topology.cores_per_chip < 2:
        logger.warning(
            "Only %d core(s) per chip; hierarchical MoE has "
            "nothing to split within a chip.",
            topology.cores_per_chip,
        )
        return None
    if world_size != len(topology.device_ids):
        logger.warning(
            "World size %d does not match the %d visible devices; refusing to "
            "guess the chip split.",
            world_size,
            len(topology.device_ids),
        )
        return None

    chip_index, core_on_chip = topology.chip_of(rank)
    return (topology.num_chips, chip_index, topology.cores_per_chip, core_on_chip)
