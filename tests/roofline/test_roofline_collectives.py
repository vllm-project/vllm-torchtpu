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
"""Unit tests for 3D torus collective floors and CollectiveTable interpolation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from vllm_torchtpu.roofline import CHIPS, TOPOLOGIES, SliceTopology, torus_collective_s
from vllm_torchtpu.roofline import collectives as rf_collectives
from vllm_torchtpu.roofline.collectives import (
    CollectiveConfig,
    collective_cut_bound_s,
    make,
    submesh_for_group,
)
from vllm_torchtpu.roofline.constants import (
    DEFAULT_ICI_BW,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    WRAPPED_LANES_PER_AXIS,
)


def test_torus_wraparound_and_mesh_packing() -> None:
    """Verify torus wraparound detection and submesh packing across slices."""
    payload_bytes = 64.0 * 1024.0 * 1024.0
    assert not TOPOLOGIES["2x2x4"].has_torus_wraparound
    assert TOPOLOGIES["4x4x4"].has_torus_wraparound
    chip = CHIPS["tpu7x"]
    assert torus_collective_s(
        "ar", payload_bytes, (2, 1, 1, 4), chip, TOPOLOGIES["4x4x4"]
    ) < torus_collective_s("ar", payload_bytes, (2, 1, 1, 4), chip, TOPOLOGIES["2x2x4"])
    assert submesh_for_group(128, TOPOLOGIES["4x4x4"], 2, (2, 4, 1, 1)) == (2, 4, 4, 4)


def test_spec_cited_chip_and_unloaded_hop_latency() -> None:
    """Verify chip spec fields match documented citations and unloaded hops."""
    # Chip spec defaults imported directly from constants for exact equality check.
    expected_link_bw = DEFAULT_ICI_LINK_BW
    expected_ici_bw = DEFAULT_ICI_BW
    expected_hop_latency = DEFAULT_ICI_HOP_LATENCY_S
    chip = CHIPS["tpu7x"]
    assert chip.d2d_bw == pytest.approx(expected_ici_bw * chip.cores_per_chip)
    assert chip.ici_link_bw == expected_link_bw
    assert chip.ici_bw == expected_ici_bw
    assert chip.ici_hop_latency_s == expected_hop_latency
    # 2 phases * (2 hops * latency + (payload * (group - 1) / group) / link_bw)
    transfer_bytes = 16.0 * 7.0 / 8.0
    expected_phase_time = 2.0 * expected_hop_latency + transfer_bytes / expected_link_bw
    assert torus_collective_s(
        "ar", 16.0, (2, 2, 2, 1), chip, TOPOLOGIES["2x2x1"]
    ) == pytest.approx(
        2.0 * expected_phase_time,
        rel=1e-12,
    )


def test_torus_collective_adds_hop_latency_to_bandwidth_phase() -> None:
    """Verify torus_collective_s adds per-phase hop latency to wire transfer time."""
    # Derived from constants: 100 GB/s link bandwidth and 1.0 us hop latency.
    expected_link_bw = DEFAULT_ICI_LINK_BW
    expected_hop_latency = DEFAULT_ICI_HOP_LATENCY_S
    payload_bytes = 64.0 * 1024.0 * 1024.0
    chip = CHIPS["tpu7x"]
    topo = TOPOLOGIES["2x2x1"]
    submesh = (2, 2, 2, 1)
    xfer_per_phase = payload_bytes * 7.0 / 8.0
    expected_s = 2.0 * (2.0 * expected_hop_latency + xfer_per_phase / expected_link_bw)
    actual_s = torus_collective_s("ar", payload_bytes, submesh, chip, topo)
    assert actual_s == pytest.approx(expected_s, rel=1e-12)
    assert (
        torus_collective_s("ar", 1024.0, (2, 1, 1, 1), replace(chip, ici_bw=0.0)) == 0.0
    )
    assert (
        torus_collective_s(
            "ar", 1024.0, (2, 2, 2, 1), replace(chip, ici_bw=math.inf), topo
        )
        == 0.0
    )
    assert (
        torus_collective_s("ar", 1024.0, (2, 1, 1, 1), replace(chip, ici_bw=-1.0))
        == 0.0
    )
    # Synthetic high-ICI test configuration with 500 GB/s per core.
    high_ici_bw = 500.0e9
    high_ici = replace(chip, ici_bw=high_ici_bw)
    assert rf_collectives.collective_cut_bound_s(
        "ar", 1024.0, (2, 1, 1, 1), high_ici
    ) <= torus_collective_s("ar", 1024.0, (2, 1, 1, 1), high_ici)


def test_collective_bound_at_or_below_feasible_schedule() -> None:
    """Verify 2x2x1 collective cut bound stays below feasible schedules."""
    payload_bytes = 64.0 * 1024.0 * 1024.0
    chip = CHIPS["tpu7x"]
    topo = TOPOLOGIES["2x2x1"]
    submesh = submesh_for_group(8, topo, 2)
    feasible_s = (
        payload_bytes / chip.d2d_bw
        + payload_bytes / chip.ici_link_bw
        + payload_bytes / chip.d2d_bw
    )
    # Cut bound across 2 external links: payload / (2 * link_bw)
    cut_s = payload_bytes / (2.0 * chip.ici_link_bw)
    bound_s = rf_collectives.collective_cut_bound_s(
        "ar", payload_bytes, submesh, chip, topo
    )
    est_s = torus_collective_s("ar", payload_bytes, submesh, chip, topo)
    # Expected analytical bound derived from 64 MiB / (2 * 100 GB/s).
    assert bound_s == pytest.approx(
        (64.0 * 1024.0 * 1024.0) / (2.0 * DEFAULT_ICI_LINK_BW), rel=1e-12
    )
    assert bound_s == pytest.approx(cut_s, rel=1e-12)
    assert bound_s <= feasible_s
    assert feasible_s <= est_s


def test_collective_cut_bound_rs_symmetric_with_ag() -> None:
    """Verify Reduce-Scatter cut bound matches All-Gather cut traffic on multichip."""
    payload_bytes = 64.0 * 1024.0 * 1024.0
    chip = CHIPS["tpu7x"]
    topo = TOPOLOGIES["2x2x1"]
    submesh = (2, 2, 2, 1)
    ag_s = rf_collectives.collective_cut_bound_s(
        "ag", payload_bytes, submesh, chip, topo
    )
    rs_s = rf_collectives.collective_cut_bound_s(
        "rs", payload_bytes, submesh, chip, topo
    )
    assert rs_s == pytest.approx(ag_s, rel=1e-12)
    # Reduce-Scatter exports 6/8 of the payload across 2 links of link_bw.
    expected_rs_s = (payload_bytes * 6.0 / 8.0) / (2.0 * chip.ici_link_bw)
    assert rs_s == pytest.approx(expected_rs_s, rel=1e-12)


def test_collective_make_uses_typed_config() -> None:
    """Verify make() accepts CollectiveConfig without vararg positional shims."""
    coll = make(
        "ag_tp", "ag", 1024.0, 8, CollectiveConfig(algo="ring", ranks_per_chip=2)
    )
    expected_hops = 7
    expected_offchip = 6.0 / 7.0
    assert coll.hops == expected_hops
    assert coll.offchip == pytest.approx(expected_offchip)


def test_a2a_cut_never_exceeds_torus_and_enforces_bisection() -> None:
    """Verify a2a cut bound matches intra-chip D2D, enforces bisection, and <= torus."""
    chip = CHIPS["tpu7x"]
    payload_bytes = 1.0e9
    intra_submesh = (2, 1, 1, 1)
    intra_group = 2
    expected_intra_cut = (
        float(intra_group - 1) / float(intra_group) * payload_bytes / chip.d2d_bw
    )
    intra_cut = collective_cut_bound_s(
        "a2a", payload_bytes, intra_submesh, chip, TOPOLOGIES["2x2x1"]
    )
    intra_torus = torus_collective_s(
        "a2a", payload_bytes, intra_submesh, chip, TOPOLOGIES["2x2x1"]
    )
    assert intra_cut == pytest.approx(expected_intra_cut, rel=1e-12)
    assert intra_cut <= intra_torus

    # 2x2x4 wrapped topology on (2, 2, 2, 4): Z-axis bisection uses 2 wrapped lanes.
    wrapped_2x2x4 = SliceTopology(
        "wrapped_2x2x4", (2, 2, 4), wrapped_axes=(True, True, True)
    )
    submesh_2x2x4 = (2, 2, 2, 4)
    local_ranks = 2
    axis_z = 4
    group_2x2x4 = 32
    bisect_divisor = 2
    expected_wrapped_bisect = (
        (
            float(
                math.floor(axis_z / bisect_divisor) * math.ceil(axis_z / bisect_divisor)
            )
            / float(axis_z)
        )
        * float(local_ranks)
        * payload_bytes
        / (float(WRAPPED_LANES_PER_AXIS) * chip.ici_link_bw)
    )
    wrapped_cut_links = 1 + 1 + WRAPPED_LANES_PER_AXIS
    naive_vertex_cut = (
        float(local_ranks * (group_2x2x4 - local_ranks))
        / float(group_2x2x4)
        * payload_bytes
        / (float(wrapped_cut_links) * chip.ici_link_bw)
    )
    wrapped_cut = collective_cut_bound_s(
        "a2a", payload_bytes, submesh_2x2x4, chip, wrapped_2x2x4
    )
    wrapped_torus = torus_collective_s(
        "a2a", payload_bytes, submesh_2x2x4, chip, wrapped_2x2x4
    )
    assert wrapped_cut == pytest.approx(expected_wrapped_bisect, rel=1e-12)
    assert wrapped_cut > naive_vertex_cut
    assert wrapped_cut <= wrapped_torus

    # Unwrapped 2x2x4 bisection exceeds the naive 3-link single-payload cut by 6.0x.
    unwrapped_cut_links = 3
    old_naive_cut = payload_bytes / (float(unwrapped_cut_links) * chip.ici_link_bw)
    expected_unwrapped_bisect = (
        (
            float(
                math.floor(axis_z / bisect_divisor) * math.ceil(axis_z / bisect_divisor)
            )
            / float(axis_z)
        )
        * float(local_ranks)
        * payload_bytes
        / chip.ici_link_bw
    )
    unwrapped_scale = 6.0
    unwrapped_cut = collective_cut_bound_s(
        "a2a", payload_bytes, submesh_2x2x4, chip, TOPOLOGIES["2x2x4"]
    )
    unwrapped_torus = torus_collective_s(
        "a2a", payload_bytes, submesh_2x2x4, chip, TOPOLOGIES["2x2x4"]
    )
    assert unwrapped_cut == pytest.approx(expected_unwrapped_bisect, rel=1e-12)
    assert unwrapped_cut == pytest.approx(unwrapped_scale * old_naive_cut, rel=1e-12)
    assert unwrapped_cut <= unwrapped_torus

    sweep_shapes = ((1, 1, 1), (2, 2, 1), (2, 2, 2), (2, 2, 4), (4, 4, 4))
    for shape in sweep_shapes:
        for wrapped_flag in (False, True):
            topo = SliceTopology(
                f"sweep_{shape}_{wrapped_flag}",
                shape,
                wrapped_axes=(wrapped_flag, wrapped_flag, wrapped_flag),
            )
            for intra in (1, chip.cores_per_chip):
                for ax, ay, az in ((1, 1, 1), shape):
                    submesh = (intra, ax, ay, az)
                    cut_s = collective_cut_bound_s(
                        "a2a", payload_bytes, submesh, chip, topo
                    )
                    torus_s = torus_collective_s(
                        "a2a", payload_bytes, submesh, chip, topo
                    )
                    assert cut_s <= torus_s
                    assert cut_s >= 0.0
