# SPDX-License-Identifier: Apache-2.0
"""Eligibility and physical rank mapping for split-SparseCore preprocessing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessTopology:
    masks: tuple[int, int, int]
    wire_map: tuple[int, int, int, int]


def topology_for_coordinates(
    coordinates: tuple[tuple[int, int, int, int], ...],
) -> PreprocessTopology | None:
    """Require an affine rank map over two adjacent X/Y chips and both dies."""
    if len(coordinates) != 8 or len(set(coordinates)) != 8:
        return None
    minima = tuple(min(c[a] for c in coordinates) for a in range(3))
    keys = tuple(
        tuple(c[a] - minima[a] for a in range(3)) + (c[3],) for c in coordinates
    )
    if set(keys) != {
        (x, y, 0, d) for x in range(2) for y in range(2) for d in range(2)
    }:
        return None
    masks = []
    for axis in (0, 1, 3):
        peers = set()
        for rank, key in enumerate(keys):
            target = list(key)
            target[axis] ^= 1
            peers.add(rank ^ keys.index(tuple(target)))
        if len(peers) != 1:
            return None
        masks.append(peers.pop())
    wire = tuple(d + 2 * x + 4 * y for x, y, _, d in keys)
    origin = wire[0]
    deltas = tuple(wire[i] ^ origin for i in (1, 2, 4))
    if any(
        value
        != origin
        ^ ((i & 1) * deltas[0])
        ^ (((i >> 1) & 1) * deltas[1])
        ^ (((i >> 2) & 1) * deltas[2])
        for i, value in enumerate(wire)
    ):
        return None
    return PreprocessTopology(
        (masks[0], masks[1], masks[2]), (origin, deltas[0], deltas[1], deltas[2])
    )


def supports_activation_gather(local_bytes: int) -> bool:
    # Four colors hold all eight ranks, with half the local input per batch.
    # Bound explicit SPMEM to the 32 MiB scoped allocation on TPU7x. Each
    # chunk must also satisfy the packed 1D HBM layout's 1024-byte tiling.
    return 0 < 4 * local_bytes <= 32 * 1024 * 1024 and local_bytes % (8 * 1024) == 0
