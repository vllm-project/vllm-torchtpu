# SPDX-License-Identifier: Apache-2.0
"""Eligibility and physical rank mapping for split-SparseCore preprocessing."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessTopology:
    masks: tuple[int, int, int]
    wire_map: tuple[int, int, int, int]
    # False: the activation all-gather is XLA's own SparseCore collective
    # pinned to SC1, not the 8-device topology kernel (masks/wire_map unused).
    custom_gather: bool = True


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


def xla_gather_topology() -> PreprocessTopology:
    """Split-SC preprocessing whose activation gather is XLA's collective."""
    return PreprocessTopology((0, 0, 0), (0, 0, 0, 0), custom_gather=False)


def supports_activation_gather(local_bytes: int) -> bool:
    # Four colors hold all eight ranks, with half the local input per batch.
    # Bound explicit SPMEM to the 32 MiB scoped allocation on TPU7x. Each
    # chunk must also satisfy the packed 1D HBM layout's 1024-byte tiling.
    return 0 < 4 * local_bytes <= 32 * 1024 * 1024 and local_bytes % (8 * 1024) == 0


# The 16-device path (XLA activation gather on SC1) is validated at 288 local
# tokens, the DSV4.1 decode step of 48 requests x (5 speculative + 1), with
# XLA's scatter. It has no scratch guard of its own, so larger prefill buckets
# inside a decode server are bounded here and keep the native path, as the
# 8-device guards do for them.
XLA_GATHER_MAX_LOCAL_TOKENS = 512


def split_sc_plan(
    topology: PreprocessTopology | None,
    *,
    fused: bool,
    fp4: bool,
    t_local: int,
    hidden: int,
    scatter_fits: Callable[[], bool],
) -> tuple[PreprocessTopology | None, bool, bool]:
    """The split-SC pieces one FP4 MoE call uses.

    Returns (topology, use_sc_scatter, use_split_sc). The topology is None where
    the call keeps native preprocessing. The 8-device topology kernel needs both
    its SPMEM scratch and the SC scatter to fit. The 16-device XLA gather only
    needs the SC placement and early submission and always keeps XLA's scatter:
    the SC scatter's scratch grows with the global routed rows and does not fit
    at the decode step (4608 tokens x top-k 6), and the shapes where it would
    fit were not measured with it.
    """
    if topology is None or not fused or not fp4:
        return None, False, False
    if topology.custom_gather:
        if not supports_activation_gather(t_local * hidden):
            return None, False, False
        if not scatter_fits():
            return None, False, False
        return topology, True, True
    if t_local > XLA_GATHER_MAX_LOCAL_TOKENS:
        return None, False, False
    return topology, False, True
