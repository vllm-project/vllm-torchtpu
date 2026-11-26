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
"""Wire bytes, serial hops, and 4-axis torus floors for collectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from .constants import (
    AR_PHASES,
    DEFAULT_CORES_PER_CHIP,
    WRAPPED_LANES_PER_AXIS,
)

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology

Kind = Literal["ar", "ag", "rs", "push", "a2a", "p2p"]
Algo = Literal["one_shot", "two_shot", "ring"]

_MIN_BISECT_EXTENT: int = 2
_BISECT_DIVISOR: int = 2


@dataclass(frozen=True, slots=True)
class CollectiveConfig:
    """Algorithm and per-chip rank packing parameters for a collective."""

    algo: Algo = "one_shot"
    ranks_per_chip: int | None = None


@dataclass(frozen=True, slots=True)
class Collective:
    """One collective as seen by one rank."""

    name: str
    group: int
    wire_bytes: float
    hops: int
    offchip: float
    kind: Kind
    payload: float


def offchip_share(group: int, ranks_per_chip: int) -> float:
    """Share of peers outside this rank's chip, assuming adjacent ranks share a chip."""
    if group <= 1:
        return 0.0
    return max(0, group - min(group, ranks_per_chip)) / (group - 1)


def make(
    name: str,
    kind: Kind,
    payload: float,
    group: int,
    config: CollectiveConfig | None = None,
) -> Collective:
    """Price one collective; payload is AR/RS input, AG output, or push/p2p wire."""
    collective_config = config or CollectiveConfig()
    if group <= 1:
        return Collective(name, group, 0.0, 0, 0.0, kind, payload)
    total_ranks = group
    if kind in ("push", "p2p"):
        wire, hops = payload, 1
    elif kind == "ar":
        if collective_config.algo == "one_shot":
            wire, hops = (total_ranks - 1) * payload, 1
        elif collective_config.algo == "two_shot":
            wire, hops = 2 * (total_ranks - 1) / total_ranks * payload, AR_PHASES
        else:
            wire, hops = (
                2 * (total_ranks - 1) / total_ranks * payload,
                AR_PHASES * (total_ranks - 1),
            )
    else:
        wire = (total_ranks - 1) / total_ranks * payload
        hops = total_ranks - 1 if collective_config.algo == "ring" else 1
    return Collective(
        name,
        total_ranks,
        wire,
        hops,
        offchip_share(
            total_ranks,
            collective_config.ranks_per_chip
            if collective_config.ranks_per_chip is not None
            else DEFAULT_CORES_PER_CHIP,
        ),
        kind,
        payload,
    )


@lru_cache(maxsize=2048)
def _pack_group_within_bounds(
    group: int,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Pack group ranks inner-to-outer within 4-axis bounds."""
    remaining_ranks = group
    packed: list[int] = []
    for axis_limit in bounds:
        take = math.gcd(remaining_ranks, max(1, axis_limit))
        packed.append(take)
        remaining_ranks //= take
    if remaining_ranks > 1:
        msg = f"Collective group {group} cannot pack within bounds {bounds}"
        raise ValueError(msg)
    return (packed[0], packed[1], packed[2], packed[3])


def submesh_for_group(
    group: int,
    topology: SliceTopology | None,
    ranks_per_chip: int | None = None,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pack `group` ranks across `(intra_chip_cores, chip_x, chip_y, chip_z)`."""
    eff_ranks = ranks_per_chip if ranks_per_chip is not None else DEFAULT_CORES_PER_CHIP
    if group <= 1:
        return (1, 1, 1, 1)
    if topology is not None and group > topology.num_cores:
        msg = (
            f"Collective group {group} exceeds topology {topology.name} "
            f"({topology.num_cores} cores)"
        )
        raise ValueError(msg)
    if mesh_axes is not None:
        capacity = mesh_axes[0] * mesh_axes[1] * mesh_axes[2] * mesh_axes[3]
        if group <= capacity and capacity % group == 0:
            bounds = mesh_axes
        elif topology is not None:
            bounds = (topology.cores_per_chip, *topology.chip_axes)
        else:
            msg = f"Collective group {group} exceeds submesh bounds {mesh_axes}"
            raise ValueError(msg)
    elif topology is not None:
        bounds = (topology.cores_per_chip, *topology.chip_axes)
    else:
        intra_chip_degree = math.gcd(group, eff_ranks)
        return (intra_chip_degree, group // intra_chip_degree, 1, 1)

    return _pack_group_within_bounds(group, bounds)


def outer_submesh_for_group(
    group: int,
    inner_group: int,
    topology: SliceTopology | None,
    ranks_per_chip: int | None = None,
) -> tuple[int, int, int, int]:
    """Pack `group` outer ranks across `(d2d, x, y, z)` after `inner_group` ranks."""
    eff_ranks = ranks_per_chip if ranks_per_chip is not None else DEFAULT_CORES_PER_CHIP
    if group <= 1:
        return (1, 1, 1, 1)
    eff_inner = max(1, inner_group)
    if topology is not None:
        bounds = (topology.cores_per_chip, *topology.chip_axes)
    else:
        intra_chip_degree = math.gcd(eff_inner * group, eff_ranks)
        bounds = (
            intra_chip_degree,
            (eff_inner * group) // max(1, intra_chip_degree),
            1,
            1,
        )
    rem_inner = eff_inner
    rem_outer = group
    out_axes: list[int] = []
    for axis_limit in bounds:
        avail = max(1, axis_limit)
        skip = math.gcd(rem_inner, avail)
        rem_inner //= skip
        free_on_axis = avail // skip
        take = math.gcd(rem_outer, free_on_axis)
        out_axes.append(take)
        rem_outer //= take
    if rem_inner > 1 or rem_outer > 1:
        msg = (
            f"Collective outer group {group} after inner_group={inner_group} "
            f"cannot pack within bounds {bounds}"
        )
        raise ValueError(msg)
    return (out_axes[0], out_axes[1], out_axes[2], out_axes[3])


def collective_ici_hops(
    collective: Collective,
    topology: SliceTopology | None = None,
    ranks_per_chip: int | None = None,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> int:
    """Unloaded one-way or two-phase physical ICI hops traversed by a collective."""
    if collective.group <= 1:
        return 0
    if topology is None and mesh_axes is None:
        if collective.offchip <= 0.0:
            return 0
        return collective.hops
    eff_ranks = ranks_per_chip if ranks_per_chip is not None else DEFAULT_CORES_PER_CHIP
    _, axis_x, axis_y, axis_z = submesh_for_group(
        collective.group, topology, eff_ranks, mesh_axes
    )
    if axis_x * axis_y * axis_z <= 1:
        return 0
    hops_one_phase = sum(
        (
            extent // 2
            if (topology is not None and topology.axis_wrapped(axis_idx, extent))
            else (extent - 1)
        )
        for axis_idx, extent in enumerate((axis_x, axis_y, axis_z))
        if extent > 1
    )
    phases = AR_PHASES if collective.kind == "ar" else 1
    return phases * hops_one_phase


def _has_finite_ici(chip: ChipSpec) -> bool:
    """Return whether chip specifies finite positive ICI bandwidths."""
    return (
        chip.ici_bw > 0.0
        and math.isfinite(chip.ici_bw)
        and chip.ici_link_bw > 0.0
        and math.isfinite(chip.ici_link_bw)
    )


def torus_collective_s(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    topology: SliceTopology | None = None,
) -> float:
    """Analytical torus floor across `(intra_chip_cores, chip_x, chip_y, chip_z)`.

    Models one isolated collective group: when `intra_chip < chip.cores_per_chip`,
    `cap_bw` scales to full chip ICI ingress (`chip.ici_bw * cores_per_chip`)
    assuming sibling cores on the chip are idle rather than running a
    concurrent collective. For `a2a`, enforces `lat_1p + collective_cut_bound_s`
    so torus latency never falls below the physical bisection cut bound.
    """
    intra_chip, axis_x, axis_y, axis_z = submesh_axes
    group = intra_chip * axis_x * axis_y * axis_z
    if group <= 1 or payload_bytes <= 0.0 or not _has_finite_ici(chip):
        return 0.0

    # Pure intra-chip collective (zero external ICI hops).
    # Google Cloud TPU7x doc (https://docs.cloud.google.com/tpu/docs/tpu7x):
    # D2D interconnect is six times faster than one ICI link (600 GB/s per chip).
    if axis_x * axis_y * axis_z == 1:
        intra_bw = (
            chip.d2d_bw
            if (math.isfinite(chip.d2d_bw) and chip.d2d_bw > 0.0)
            else chip.ici_bw * chip.cores_per_chip
        )
        xfer = (
            payload_bytes
            if kind in ("push", "p2p")
            else payload_bytes * (group - 1) / group
        )
        phase_s = xfer / intra_bw
        return float(AR_PHASES) * phase_s if kind == "ar" else phase_s

    ici_hops = 0
    lanes_sum = 0
    for axis_idx, extent in enumerate((axis_x, axis_y, axis_z)):
        if extent <= 1:
            continue
        wrapped = (
            topology.axis_wrapped(axis_idx, extent) if topology is not None else False
        )
        if wrapped:
            ici_hops += extent // 2
            lanes_sum += WRAPPED_LANES_PER_AXIS
        else:
            ici_hops += extent - 1
            lanes_sum += 1

    contention = float(max(1, intra_chip))
    cap_bw = (
        chip.ici_bw
        if intra_chip >= chip.cores_per_chip
        else chip.ici_bw * float(chip.cores_per_chip) / contention
    )
    raw_bw = lanes_sum * chip.ici_link_bw / contention
    eff_bw = min(raw_bw, cap_bw) if math.isfinite(cap_bw) and cap_bw > 0.0 else raw_bw

    hop_lat = (
        chip.ici_hop_latency_s
        if chip.ici_hop_latency_s > 0.0 and math.isfinite(chip.ici_hop_latency_s)
        else 0.0
    )
    lat_1p = ici_hops * hop_lat

    if kind == "p2p":
        return lat_1p + payload_bytes / eff_bw

    xfer = payload_bytes if kind == "push" else payload_bytes * (group - 1) / group
    phase_s = lat_1p + xfer / eff_bw
    if kind in ("a2a", "push"):
        return max(
            phase_s,
            lat_1p
            + collective_cut_bound_s(kind, payload_bytes, submesh_axes, chip, topology),
        )
    return float(AR_PHASES) * phase_s if kind == "ar" else phase_s


@lru_cache(maxsize=2048)
def _count_cut_links(
    axes: tuple[int, int, int],
    topology: SliceTopology | None,
) -> int:
    """Count unidirectional ICI cut links per direction across active mesh axes."""
    links_in = 0
    for axis_idx, extent in enumerate(axes):
        if extent <= 1:
            continue
        if topology is not None and topology.axis_wrapped(axis_idx, extent):
            links_in += WRAPPED_LANES_PER_AXIS
        else:
            links_in += 1
    return max(1, links_in)


def _a2a_interchip_cut_s(
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    topology: SliceTopology | None,
) -> float:
    """Compute vertex and hyperplane bisection lower bound for inter-chip a2a."""
    intra_chip, axis_x, axis_y, axis_z = submesh_axes
    group_size = intra_chip * axis_x * axis_y * axis_z
    local_ranks = max(1, min(group_size, intra_chip))
    axes = (axis_x, axis_y, axis_z)
    cut_links = _count_cut_links(axes, topology)
    vertex_s = (
        float(local_ranks * (group_size - local_ranks))
        / float(group_size)
        * payload_bytes
    ) / (float(cut_links) * chip.ici_link_bw)
    bisect_s = 0.0
    for axis_idx, extent in enumerate(axes):
        if extent < _MIN_BISECT_EXTENT:
            continue
        half_lo = extent // _BISECT_DIVISOR
        half_hi = extent - half_lo
        wrapped = (
            topology.axis_wrapped(axis_idx, extent) if topology is not None else False
        )
        lanes = WRAPPED_LANES_PER_AXIS if wrapped else 1
        axis_bisect_s = (
            (float(half_lo * half_hi) / float(extent))
            * float(local_ranks)
            * payload_bytes
            / (float(lanes) * chip.ici_link_bw)
        )
        bisect_s = max(bisect_s, axis_bisect_s)
    return max(vertex_s, bisect_s)


_ICI_SUBSET_COUNT = 7
_ZERO_MASK_BYTES: tuple[float, float, float, float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def _merge_partition_axis_bisect(
    left: tuple[tuple[str, tuple[float, float, float]], ...],
    right: tuple[tuple[str, tuple[float, float, float]], ...],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Sum per-axis bisection bytes only within identical non-empty partition keys."""
    if not left:
        return right
    if not right:
        return left
    merged: dict[str, tuple[float, float, float]] = dict(left)
    for partition_key, (bx, by, bz) in right:
        prev = merged.get(partition_key)
        if prev is None:
            merged[partition_key] = (bx, by, bz)
        else:
            merged[partition_key] = (prev[0] + bx, prev[1] + by, prev[2] + bz)
    return tuple(sorted(merged.items()))


@dataclass(frozen=True, slots=True)
class LinkCutLoad:
    """Additive parent-topology cut loads, partition-keyed bisection, and isolated max.

    Each 7-tuple stores byte loads across non-empty subsets ``mask in 1..7`` of the
    3 ICI axes (bit 0 = x, bit 1 = y, bit 2 = z):
    - ``vertex_mask_bytes``: additive per-chip slab ingress bytes (uninflated).
    - ``edge_mask_bytes``: additive per-chip directed byte-hops on active axes.
    - ``partition_axis_bisect``: per-link bisection bytes keyed by parallel partition;
      summed only within identical non-empty keys and maxed across distinct keys.
    - ``isolated_cut_s``: single-collective cut bound combined via ``max`` on add.
    """

    d2d_bytes: float = 0.0
    vertex_mask_bytes: tuple[float, float, float, float, float, float, float] = (
        _ZERO_MASK_BYTES
    )
    edge_mask_bytes: tuple[float, float, float, float, float, float, float] = (
        _ZERO_MASK_BYTES
    )
    partition_axis_bisect: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    isolated_cut_s: float = 0.0

    def scaled(self, factor: float) -> LinkCutLoad:
        """Return a copy with all byte loads and repeated cuts scaled by factor."""
        if factor == 1.0:
            return self
        return LinkCutLoad(
            d2d_bytes=self.d2d_bytes * factor,
            vertex_mask_bytes=(
                self.vertex_mask_bytes[0] * factor,
                self.vertex_mask_bytes[1] * factor,
                self.vertex_mask_bytes[2] * factor,
                self.vertex_mask_bytes[3] * factor,
                self.vertex_mask_bytes[4] * factor,
                self.vertex_mask_bytes[5] * factor,
                self.vertex_mask_bytes[6] * factor,
            ),
            edge_mask_bytes=(
                self.edge_mask_bytes[0] * factor,
                self.edge_mask_bytes[1] * factor,
                self.edge_mask_bytes[2] * factor,
                self.edge_mask_bytes[3] * factor,
                self.edge_mask_bytes[4] * factor,
                self.edge_mask_bytes[5] * factor,
                self.edge_mask_bytes[6] * factor,
            ),
            partition_axis_bisect=tuple(
                (key, (bx * factor, by * factor, bz * factor))
                for key, (bx, by, bz) in self.partition_axis_bisect
            ),
            isolated_cut_s=self.isolated_cut_s * factor,
        )

    def add(self, other: LinkCutLoad) -> LinkCutLoad:
        """Combine two LinkCutLoad records soundly across all cut families."""
        return LinkCutLoad(
            d2d_bytes=self.d2d_bytes + other.d2d_bytes,
            vertex_mask_bytes=(
                self.vertex_mask_bytes[0] + other.vertex_mask_bytes[0],
                self.vertex_mask_bytes[1] + other.vertex_mask_bytes[1],
                self.vertex_mask_bytes[2] + other.vertex_mask_bytes[2],
                self.vertex_mask_bytes[3] + other.vertex_mask_bytes[3],
                self.vertex_mask_bytes[4] + other.vertex_mask_bytes[4],
                self.vertex_mask_bytes[5] + other.vertex_mask_bytes[5],
                self.vertex_mask_bytes[6] + other.vertex_mask_bytes[6],
            ),
            edge_mask_bytes=(
                self.edge_mask_bytes[0] + other.edge_mask_bytes[0],
                self.edge_mask_bytes[1] + other.edge_mask_bytes[1],
                self.edge_mask_bytes[2] + other.edge_mask_bytes[2],
                self.edge_mask_bytes[3] + other.edge_mask_bytes[3],
                self.edge_mask_bytes[4] + other.edge_mask_bytes[4],
                self.edge_mask_bytes[5] + other.edge_mask_bytes[5],
                self.edge_mask_bytes[6] + other.edge_mask_bytes[6],
            ),
            partition_axis_bisect=_merge_partition_axis_bisect(
                self.partition_axis_bisect, other.partition_axis_bisect
            ),
            isolated_cut_s=max(self.isolated_cut_s, other.isolated_cut_s),
        )


@lru_cache(maxsize=2048)
def _directed_edge_count(
    axes: tuple[int, int, int],
    topology: SliceTopology | None,
) -> float:
    """Count total directed ICI edges inside an isolated (x, y, z) chip submesh."""
    num_chips = axes[0] * axes[1] * axes[2]
    if num_chips <= 1:
        return 0.0
    total_directed = 0.0
    for axis_idx, extent in enumerate(axes):
        if extent <= 1:
            continue
        wrapped = (
            topology.axis_wrapped(axis_idx, extent) if topology is not None else False
        )
        if wrapped:
            total_directed += float(WRAPPED_LANES_PER_AXIS * num_chips)
        else:
            total_directed += float(
                WRAPPED_LANES_PER_AXIS * num_chips * (extent - 1)
            ) / float(extent)
    return total_directed


@lru_cache(maxsize=2048)
def _axis_lane_counts(
    axes: tuple[int, int, int],
    topology: SliceTopology | None,
) -> tuple[int, int, int]:
    """Return unidirectional link lanes per axis (1 unwrapped, 2 wrapped, 0 unused)."""
    lanes: list[int] = []
    for axis_idx, extent in enumerate(axes):
        if extent <= 1:
            lanes.append(0)
        elif topology is not None and topology.axis_wrapped(axis_idx, extent):
            lanes.append(WRAPPED_LANES_PER_AXIS)
        else:
            lanes.append(1)
    return (lanes[0], lanes[1], lanes[2])


@lru_cache(maxsize=2048)
def _valid_submesh_factorizations(
    group: int,
    bounds: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate all (intra_chip, x, y, z) factorizations fitting within bounds."""
    if group <= 1:
        return ((1, 1, 1, 1),)
    bound_intra, bound_x, bound_y, bound_z = bounds
    factorizations: list[tuple[int, int, int, int]] = []
    for intra in range(1, min(group, bound_intra) + 1):
        if group % intra != 0:
            continue
        rem_x = group // intra
        for axis_x in range(1, min(rem_x, bound_x) + 1):
            if rem_x % axis_x != 0:
                continue
            rem_y = rem_x // axis_x
            for axis_y in range(1, min(rem_y, bound_y) + 1):
                if rem_y % axis_y != 0:
                    continue
                axis_z = rem_y // axis_y
                if axis_z <= bound_z:
                    factorizations.append((intra, axis_x, axis_y, axis_z))
    return tuple(factorizations)


def _intra_chip_bw(chip: ChipSpec) -> float:
    """Return intra-chip D2D bandwidth in bytes/s."""
    return (
        chip.d2d_bw
        if (math.isfinite(chip.d2d_bw) and chip.d2d_bw > 0.0)
        else chip.ici_bw * float(chip.cores_per_chip)
    )


@dataclass(frozen=True, slots=True)
class CutPlacementOptions:
    """Topology, rank packing, partition identity, and concurrent-group context."""

    topology: SliceTopology | None = None
    ranks_per_chip: int | None = None
    mesh_axes: tuple[int, int, int, int] | None = None
    concurrent_groups_per_chip: int = 1
    partition_key: str = ""


def collective_cut_bound_s(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    topology: SliceTopology | None = None,
) -> float:
    """Physical cut-set lower bound in seconds for one collective."""
    intra_chip, axis_x, axis_y, axis_z = submesh_axes
    group_size = intra_chip * axis_x * axis_y * axis_z
    if group_size <= 1 or payload_bytes <= 0.0 or not _has_finite_ici(chip):
        return 0.0
    if axis_x * axis_y * axis_z == 1:
        intra_bw = _intra_chip_bw(chip)
        if kind == "ar":
            cut_bytes = (
                float(AR_PHASES)
                * payload_bytes
                * float(group_size - 1)
                / float(group_size)
            )
        elif kind in ("ag", "rs", "a2a"):
            cut_bytes = payload_bytes * float(group_size - 1) / float(group_size)
        else:
            cut_bytes = payload_bytes
        return cut_bytes / intra_bw

    local_ranks = max(1, min(group_size, intra_chip))
    axes = (axis_x, axis_y, axis_z)
    if kind == "a2a":
        return _a2a_interchip_cut_s(payload_bytes, submesh_axes, chip, topology)
    if kind == "push":
        # Per-rank off-self push payload P maps identically to a2a with full
        # payload P * G / (G - 1), accounting for local_ranks and bisections.
        equiv_a2a_payload = (
            payload_bytes * float(group_size) / float(max(1, group_size - 1))
        )
        return _a2a_interchip_cut_s(equiv_a2a_payload, submesh_axes, chip, topology)
    cut_links = _count_cut_links(axes, topology)
    if kind == "ar":
        num_chips = axis_x * axis_y * axis_z
        vertex_s = payload_bytes / (float(cut_links) * chip.ici_link_bw)
        directed_edges = _directed_edge_count(axes, topology)
        edge_s = (
            (float(AR_PHASES * (num_chips - 1)) * payload_bytes)
            / (directed_edges * chip.ici_link_bw)
            if directed_edges > 0.0
            else 0.0
        )
        return max(vertex_s, edge_s)
    if kind in ("ag", "rs"):
        cut_bytes = (
            payload_bytes
            * float(max(group_size - local_ranks, local_ranks))
            / float(group_size)
        )
    else:
        cut_bytes = payload_bytes
    return cut_bytes / (float(cut_links) * chip.ici_link_bw)


def _scaled_cut_bound_s(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    options: CutPlacementOptions,
) -> float:
    """Scale inter-chip cut bound by concurrent independent groups per chip."""
    base_s = collective_cut_bound_s(
        kind, payload_bytes, submesh_axes, chip, options.topology
    )
    if submesh_axes[1] * submesh_axes[2] * submesh_axes[3] <= 1:
        return base_s
    return float(max(1, options.concurrent_groups_per_chip)) * base_s


def min_collective_cut_bound_s(
    kind: Kind,
    payload_bytes: float,
    group: int,
    chip: ChipSpec,
    options: CutPlacementOptions | None = None,
) -> tuple[float, tuple[int, int, int, int]]:
    """Return minimum cut bound and submesh over all valid rank placements."""
    if group <= 1 or payload_bytes <= 0.0 or not _has_finite_ici(chip):
        return (0.0, (1, 1, 1, 1))
    opts = options or CutPlacementOptions()
    eff_ranks = (
        opts.ranks_per_chip if opts.ranks_per_chip is not None else chip.cores_per_chip
    )
    if opts.mesh_axes is not None:
        capacity = (
            opts.mesh_axes[0]
            * opts.mesh_axes[1]
            * opts.mesh_axes[2]
            * opts.mesh_axes[3]
        )
        if group <= capacity and capacity % group == 0:
            bounds = opts.mesh_axes
        elif opts.topology is not None:
            bounds = (opts.topology.cores_per_chip, *opts.topology.chip_axes)
        else:
            bounds = opts.mesh_axes
    elif opts.topology is not None:
        bounds = (opts.topology.cores_per_chip, *opts.topology.chip_axes)
    else:
        submesh = submesh_for_group(group, None, eff_ranks, None)
        return (_scaled_cut_bound_s(kind, payload_bytes, submesh, chip, opts), submesh)
    candidates = _valid_submesh_factorizations(group, bounds)
    if not candidates:
        submesh = submesh_for_group(group, opts.topology, eff_ranks, opts.mesh_axes)
        return (_scaled_cut_bound_s(kind, payload_bytes, submesh, chip, opts), submesh)
    best_s = math.inf
    best_submesh = candidates[0]
    for cand in candidates:
        cand_s = _scaled_cut_bound_s(kind, payload_bytes, cand, chip, opts)
        if cand_s < best_s:
            best_s = cand_s
            best_submesh = cand
    return (best_s, best_submesh)


@lru_cache(maxsize=256)
def _parent_axis_lanes(
    topology: SliceTopology | None,
) -> tuple[int, int, int]:
    """Return per-axis ICI slab lane divisors for the parent slice topology."""
    if topology is None:
        return (1, 1, 1)
    return _axis_lane_counts(topology.chip_axes, topology)


@lru_cache(maxsize=256)
def _parent_axis_directed_edges_per_chip(
    topology: SliceTopology | None,
) -> tuple[float, float, float]:
    """Return average directed ICI edges per chip along each parent axis."""
    if topology is None:
        return (
            float(WRAPPED_LANES_PER_AXIS),
            float(WRAPPED_LANES_PER_AXIS),
            float(WRAPPED_LANES_PER_AXIS),
        )
    edges: list[float] = []
    for axis_idx, extent in enumerate(topology.chip_axes):
        if extent <= 1:
            edges.append(0.0)
        elif topology.axis_wrapped(axis_idx, extent):
            edges.append(float(WRAPPED_LANES_PER_AXIS))
        else:
            edges.append(float(WRAPPED_LANES_PER_AXIS * (extent - 1)) / float(extent))
    return (edges[0], edges[1], edges[2])


def _mask_lane_sum(lanes: tuple[int, int, int], mask: int) -> float:
    """Sum lane counts across the active bits of a 3-axis bitmask."""
    return float(
        (lanes[0] if (mask & 1) else 0)
        + (lanes[1] if (mask & 2) else 0)
        + (lanes[2] if (mask & 4) else 0)
    )


def _mask_float_sum(values: tuple[float, float, float], mask: int) -> float:
    """Sum floating-point per-axis capacities across active bits of a bitmask."""
    return (
        (values[0] if (mask & 1) else 0.0)
        + (values[1] if (mask & 2) else 0.0)
        + (values[2] if (mask & 4) else 0.0)
    )


def _expand_mask_bytes(
    mask: int,
    amount: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Place amount into every axis subset mask that contains all bits of mask."""
    return (
        amount if (mask & ~1) == 0 else 0.0,
        amount if (mask & ~2) == 0 else 0.0,
        amount if (mask & ~3) == 0 else 0.0,
        amount if (mask & ~4) == 0 else 0.0,
        amount if (mask & ~5) == 0 else 0.0,
        amount if (mask & ~6) == 0 else 0.0,
        amount if (mask & ~7) == 0 else 0.0,
    )


def _compute_partition_axis_bisect(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    conc_scale: float,
    cut_s: float,
    ici_link_bw: float,
    topology: SliceTopology | None,
    partition_key: str,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Build partition-keyed per-link bisection bytes for non-empty partition keys."""
    if not partition_key:
        return ()
    intra_chip, axis_x, axis_y, axis_z = submesh_axes
    group_size = intra_chip * axis_x * axis_y * axis_z
    local_ranks = max(1, min(group_size, intra_chip))
    axes = (axis_x, axis_y, axis_z)
    mask = (
        (1 if axis_x > 1 else 0) | (2 if axis_y > 1 else 0) | (4 if axis_z > 1 else 0)
    )
    axis_bytes = [0.0, 0.0, 0.0]
    if kind in ("a2a", "push"):
        equiv_payload = (
            payload_bytes
            if kind == "a2a"
            else payload_bytes * float(group_size) / float(max(1, group_size - 1))
        )
        for axis_idx, extent in enumerate(axes):
            if extent < _MIN_BISECT_EXTENT:
                continue
            half_lo = extent // _BISECT_DIVISOR
            half_hi = extent - half_lo
            wrapped = (
                topology.axis_wrapped(axis_idx, extent)
                if topology is not None
                else False
            )
            lanes = WRAPPED_LANES_PER_AXIS if wrapped else 1
            axis_bytes[axis_idx] = (
                conc_scale
                * (float(half_lo * half_hi) / float(extent))
                * float(local_ranks)
                * equiv_payload
                / float(lanes)
            )
    elif mask in (1, 2, 4):
        axis_idx = 0 if mask == 1 else (1 if mask == 2 else 2)
        axis_bytes[axis_idx] = cut_s * ici_link_bw
    if max(axis_bytes) <= 0.0:
        return ()
    return ((partition_key, (axis_bytes[0], axis_bytes[1], axis_bytes[2])),)


@lru_cache(maxsize=8192)
def _collective_axis_cut_cached(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    topology: SliceTopology | None,
    concurrent_groups_per_chip: int,
    partition_key: str,
) -> LinkCutLoad:
    """Cached implementation of collective_axis_cut_bytes over value-tuple keys."""
    intra_chip, axis_x, axis_y, axis_z = submesh_axes
    group_size = intra_chip * axis_x * axis_y * axis_z
    num_chips = axis_x * axis_y * axis_z
    if num_chips == 1:
        cut_s = collective_cut_bound_s(
            kind, payload_bytes, submesh_axes, chip, topology
        )
        return LinkCutLoad(d2d_bytes=cut_s * _intra_chip_bw(chip))

    opts = CutPlacementOptions(
        topology=topology,
        concurrent_groups_per_chip=concurrent_groups_per_chip,
        partition_key=partition_key,
    )
    cut_s = _scaled_cut_bound_s(kind, payload_bytes, submesh_axes, chip, opts)
    local_ranks = max(1, min(group_size, intra_chip))
    conc_scale = float(max(1, concurrent_groups_per_chip))
    mask = (
        (1 if axis_x > 1 else 0) | (2 if axis_y > 1 else 0) | (4 if axis_z > 1 else 0)
    )
    if kind in ("a2a", "push"):
        equiv_payload = (
            payload_bytes
            if kind == "a2a"
            else payload_bytes * float(group_size) / float(max(1, group_size - 1))
        )
        vertex_bytes = (
            conc_scale
            * float(local_ranks * (group_size - local_ranks))
            / float(group_size)
            * equiv_payload
        )
        edge_bytes = vertex_bytes
    elif kind == "ar":
        vertex_bytes = conc_scale * payload_bytes
        edge_bytes = (
            conc_scale
            * float(AR_PHASES * (num_chips - 1))
            / float(num_chips)
            * payload_bytes
        )
    elif kind in ("ag", "rs"):
        vertex_bytes = (
            conc_scale
            * float(max(group_size - local_ranks, local_ranks))
            / float(group_size)
            * payload_bytes
        )
        edge_bytes = vertex_bytes
    else:
        vertex_bytes = conc_scale * payload_bytes
        edge_bytes = vertex_bytes

    bisect = _compute_partition_axis_bisect(
        kind,
        payload_bytes,
        submesh_axes,
        conc_scale,
        cut_s,
        chip.ici_link_bw,
        topology,
        partition_key,
    )
    return LinkCutLoad(
        d2d_bytes=0.0,
        vertex_mask_bytes=_expand_mask_bytes(mask, vertex_bytes),
        edge_mask_bytes=_expand_mask_bytes(mask, edge_bytes),
        partition_axis_bisect=bisect,
        isolated_cut_s=cut_s,
    )


def collective_axis_cut_bytes(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    options: CutPlacementOptions | None = None,
) -> LinkCutLoad:
    """Return sound multi-family link cut loads for one collective."""
    intra_chip, axis_x, axis_y, axis_z = submesh_axes
    group_size = intra_chip * axis_x * axis_y * axis_z
    if group_size <= 1 or payload_bytes <= 0.0 or not _has_finite_ici(chip):
        return LinkCutLoad()
    opts = options or CutPlacementOptions()
    return _collective_axis_cut_cached(
        kind,
        payload_bytes,
        submesh_axes,
        chip,
        opts.topology,
        max(1, opts.concurrent_groups_per_chip),
        opts.partition_key,
    )


def eval_link_subset_bound_s(
    load: LinkCutLoad,
    chip: ChipSpec,
    topology: SliceTopology | None = None,
) -> float:
    """Evaluate sound collective lower bound by taking max across all cut families."""
    if not _has_finite_ici(chip):
        return 0.0
    best_s = max(
        load.isolated_cut_s,
        load.d2d_bytes / _intra_chip_bw(chip) if load.d2d_bytes > 0.0 else 0.0,
    )
    parent_lanes = _parent_axis_lanes(topology)
    parent_edges = _parent_axis_directed_edges_per_chip(topology)
    link_bw = chip.ici_link_bw
    for subset in range(1, _ICI_SUBSET_COUNT + 1):
        vertex_bytes = load.vertex_mask_bytes[subset - 1]
        if vertex_bytes > 0.0:
            vertex_links = _mask_lane_sum(parent_lanes, subset)
            if vertex_links > 0.0:
                best_s = max(best_s, vertex_bytes / (vertex_links * link_bw))
        edge_bytes = load.edge_mask_bytes[subset - 1]
        if edge_bytes > 0.0:
            edge_links = _mask_float_sum(parent_edges, subset)
            if edge_links > 0.0:
                best_s = max(best_s, edge_bytes / (edge_links * link_bw))
    for _, (bx, by, bz) in load.partition_axis_bisect:
        peak_bisect = max(bx, by, bz)
        if peak_bisect > 0.0:
            best_s = max(best_s, peak_bisect / link_bw)
    return best_s
