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
from typing import TYPE_CHECKING, Literal

from .constants import (
    AR_PHASES,
    DEFAULT_CORES_PER_CHIP,
    MIN_TORUS_DIM,
    WRAPPED_LANES_PER_AXIS,
)

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology

Kind = Literal["ar", "ag", "rs", "push", "a2a", "p2p"]
Algo = Literal["one_shot", "two_shot", "ring"]


@dataclass(frozen=True, slots=True)
class CollectiveConfig:
    """Algorithm and per-chip rank packing parameters for a collective."""

    algo: Algo = "one_shot"
    ranks_per_chip: int = DEFAULT_CORES_PER_CHIP


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
        offchip_share(total_ranks, collective_config.ranks_per_chip),
        kind,
        payload,
    )


def padded_rows(rows: int, tp: int, residual: str) -> int:
    """Rows a collective carries: sequence parallelism pads to a multiple of tp."""
    return rows if residual == "replicated" else math.ceil(rows / tp) * tp


def submesh_for_group(
    group: int,
    topology: SliceTopology | None,
    ranks_per_chip: int = DEFAULT_CORES_PER_CHIP,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pack `group` ranks inner-to-outer across (d2d, x, y, z) physical bounds."""
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
        intra_chip_degree = math.gcd(group, ranks_per_chip)
        return (intra_chip_degree, group // intra_chip_degree, 1, 1)

    remaining_ranks = group
    submesh_axes: list[int] = []
    for axis_limit in bounds:
        take = math.gcd(remaining_ranks, max(1, axis_limit))
        submesh_axes.append(take)
        remaining_ranks //= take
    if remaining_ranks > 1:
        msg = f"Collective group {group} cannot pack within bounds {bounds}"
        raise ValueError(msg)
    return (submesh_axes[0], submesh_axes[1], submesh_axes[2], submesh_axes[3])


def collective_ici_hops(
    collective: Collective,
    topology: SliceTopology | None = None,
    ranks_per_chip: int = DEFAULT_CORES_PER_CHIP,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> int:
    """Unloaded one-way or two-phase physical ICI hops traversed by a collective."""
    if collective.group <= 1:
        return 0
    if topology is None and mesh_axes is None:
        if collective.offchip <= 0.0:
            return 0
        return collective.hops
    _, axis_x, axis_y, axis_z = submesh_for_group(
        collective.group, topology, ranks_per_chip, mesh_axes
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
    """Analytical two-phase torus alpha-beta floor across (d2d, a, b, c) axes."""
    intra_chip, axis_x, axis_y, axis_z = submesh_axes
    group = intra_chip * axis_x * axis_y * axis_z
    if group <= 1 or payload_bytes <= 0.0 or not _has_finite_ici(chip):
        return 0.0

    # Pure intra-chip collective (zero external ICI hops).
    # Google Cloud TPU7x doc (https://cloud.google.com/tpu/docs/tpu7x):
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
        else chip.ici_bw * float(chip.cores_per_chip)
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
    return float(AR_PHASES) * phase_s if kind == "ar" else phase_s


def _count_cut_links(
    axes: tuple[int, int, int],
    topology: SliceTopology | None,
) -> int:
    """Count unidirectional ICI cut links per direction across active mesh axes."""
    links_in = 0
    for axis_idx, ext in enumerate(axes):
        if ext <= 1:
            continue
        if (
            ext >= MIN_TORUS_DIM
            and topology is not None
            and topology.axis_wrapped(axis_idx, ext)
        ):
            links_in += WRAPPED_LANES_PER_AXIS
        else:
            links_in += 1
    return max(1, links_in)


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
        intra_bw = (
            chip.d2d_bw
            if (math.isfinite(chip.d2d_bw) and chip.d2d_bw > 0.0)
            else chip.ici_bw * float(chip.cores_per_chip)
        )
        cut_bytes = (
            payload_bytes * float(group_size - 1) / float(group_size)
            if kind in ("ag", "rs")
            else payload_bytes
        )
        return cut_bytes / intra_bw
    local_ranks = max(1, min(group_size, intra_chip))
    if kind in ("ag", "rs"):
        # AG imports (group_size - local_ranks) / group_size across the cut;
        # RS exports (group_size - local_ranks) / group_size symmetrically
        # across the same cut (max with local shard local_ranks / group_size).
        cut_bytes = (
            payload_bytes
            * float(max(group_size - local_ranks, local_ranks))
            / float(group_size)
        )
    else:
        cut_bytes = payload_bytes
    cut_links = _count_cut_links((axis_x, axis_y, axis_z), topology)
    return cut_bytes / (float(cut_links) * chip.ici_link_bw)
