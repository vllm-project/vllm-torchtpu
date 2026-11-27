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

from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    TPU6E_TOPOLOGIES,
    SliceTopology,
    torus_collective_s,
)
from vllm_torchtpu.roofline import collectives as rf_collectives
from vllm_torchtpu.roofline.collectives import (
    CollectiveConfig,
    CutPlacementOptions,
    collective_axis_cut_bytes,
    collective_cut_bound_s,
    collective_ici_hops,
    eval_link_subset_bound_s,
    make,
    min_collective_cut_bound_s,
    outer_submesh_for_group,
    submesh_for_group,
)
from vllm_torchtpu.roofline.constants import (
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
    expected_link_bw = 100.0e9
    expected_ici_bw = 300.0e9
    expected_hop_latency = 1.0e-6
    chip = CHIPS["tpu7x"]
    assert chip.d2d_bw == pytest.approx(expected_ici_bw * chip.cores_per_chip)
    assert chip.ici_link_bw == expected_link_bw
    assert chip.ici_bw == expected_ici_bw
    assert chip.ici_hop_latency_s == expected_hop_latency
    transfer_bytes = 16.0 * 7.0 / 8.0
    expected_phase_time = 2.0 * expected_hop_latency + transfer_bytes / expected_link_bw
    assert torus_collective_s(
        "ar", 16.0, (2, 2, 2, 1), chip, TOPOLOGIES["2x2x1"]
    ) == pytest.approx(2.0 * expected_phase_time, rel=1e-12)


def test_torus_collective_adds_hop_latency_to_bandwidth_phase() -> None:
    """Verify torus_collective_s adds per-phase hop latency to wire transfer time."""
    expected_link_bw = DEFAULT_ICI_LINK_BW
    expected_hop_latency = DEFAULT_ICI_HOP_LATENCY_S
    payload_bytes = 64.0 * 1024.0 * 1024.0
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    submesh = (2, 2, 2, 1)
    xfer_per_phase = payload_bytes * 7.0 / 8.0
    expected_s = 2.0 * (2.0 * expected_hop_latency + xfer_per_phase / expected_link_bw)
    actual_s = torus_collective_s("ar", payload_bytes, submesh, chip, topology)
    assert actual_s == pytest.approx(expected_s, rel=1e-12)
    assert (
        torus_collective_s("ar", 1024.0, (2, 1, 1, 1), replace(chip, ici_bw=0.0)) == 0.0
    )
    assert (
        torus_collective_s(
            "ar", 1024.0, (2, 2, 2, 1), replace(chip, ici_bw=math.inf), topology
        )
        == 0.0
    )
    assert (
        torus_collective_s("ar", 1024.0, (2, 1, 1, 1), replace(chip, ici_bw=-1.0))
        == 0.0
    )
    high_ici_bw = 500000000000.0
    high_ici = replace(chip, ici_bw=high_ici_bw)
    assert rf_collectives.collective_cut_bound_s(
        "ar", 1024.0, (2, 1, 1, 1), high_ici
    ) <= torus_collective_s("ar", 1024.0, (2, 1, 1, 1), high_ici)


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
            topology = SliceTopology(
                f"sweep_{shape}_{wrapped_flag}",
                shape,
                wrapped_axes=(wrapped_flag, wrapped_flag, wrapped_flag),
            )
            for intra in (1, chip.cores_per_chip):
                for ax, ay, az in ((1, 1, 1), shape):
                    submesh = (intra, ax, ay, az)
                    cut_s = collective_cut_bound_s(
                        "a2a", payload_bytes, submesh, chip, topology
                    )
                    torus_s = torus_collective_s(
                        "a2a", payload_bytes, submesh, chip, topology
                    )
                    assert cut_s <= torus_s
                    assert cut_s >= 0.0


def test_allreduce_cut_bound_sound_on_unwrapped_line() -> None:
    """Verify AR cut bound stays exact on unwrapped lines and tightens 2D/3D meshes."""
    chip = CHIPS["tpu7x"]
    payload_bytes = 1.0e9
    link_bw = 100.0e9

    # 4x1x1 unwrapped line: E_dir = 6, cut_links = 1, 2*(4-1)/6 == 1.0 (not 1.5).
    line_topo = SliceTopology("4x1x1", (4, 1, 1), wrapped_axes=(False, False, False))
    line_cut = collective_cut_bound_s(
        "ar", payload_bytes, (2, 4, 1, 1), chip, line_topo
    )
    assert line_cut == pytest.approx(payload_bytes / link_bw, rel=1e-12)

    # 2x2x1 unwrapped mesh: C=4, E_dir=8 -> 2*(4-1)/8 = 0.75 * P / link_bw.
    mesh_2x2x1_cut = collective_cut_bound_s(
        "ar", payload_bytes, (2, 2, 2, 1), chip, TOPOLOGIES["2x2x1"]
    )
    assert mesh_2x2x1_cut == pytest.approx(
        (2.0 * (4 - 1) * payload_bytes) / (8.0 * link_bw), rel=1e-12
    )

    # 2x2x2 unwrapped mesh: C=8, E_dir=24 -> 2*(8-1)/24 = 7/12 * P / link_bw.
    mesh_2x2x2_cut = collective_cut_bound_s(
        "ar", payload_bytes, (2, 2, 2, 2), chip, TOPOLOGIES["2x2x2"]
    )
    assert mesh_2x2x2_cut == pytest.approx(
        (2.0 * (8 - 1) * payload_bytes) / (24.0 * link_bw), rel=1e-12
    )


def test_partial_chip_cap_divided_by_contention() -> None:
    """Verify partial-chip torus cap divides chip ICI bandwidth by intra_chip."""
    base_chip = CHIPS["tpu7x"]
    four_core_chip = replace(
        base_chip,
        cores_per_chip=4,
        ici_link_bw=100.0e9,
        ici_bw=25.0e9,
        ici_hop_latency_s=0.0,
    )
    payload_bytes = 1.0e9
    # intra_chip=2 < cores_per_chip=4 on a 2x2x1 inter-chip submesh:
    # 2 unwrapped lanes give raw_bw = 100 GB/s, so chip ICI cap
    # 4 * 25 / 2 = 50 GB/s binds.
    topo_2d = SliceTopology(
        "2x2x1_4c", (2, 2, 1), cores_per_chip=4, wrapped_axes=(False, False, False)
    )
    submesh_2d = (2, 2, 2, 1)
    actual_s = torus_collective_s(
        "ag", payload_bytes, submesh_2d, four_core_chip, topo_2d
    )
    expected_xfer = payload_bytes * (8 - 1) / 8
    expected_cap_bw = four_core_chip.ici_bw * 4.0 / 2.0
    assert actual_s == pytest.approx(expected_xfer / expected_cap_bw, rel=1e-12)


def test_outer_submesh_offsets_pp_p2p_outside_pcp() -> None:
    """Verify outer_submesh_for_group places outer ranks after inner ranks."""
    topology = TOPOLOGIES["2x2x2"]
    assert outer_submesh_for_group(2, 8, topology) == (1, 1, 1, 2)
    assert outer_submesh_for_group(4, 2, topology) == (1, 2, 2, 1)


def test_min_collective_cut_bound_beats_greedy_placement() -> None:
    """Verify minimizing over submesh factorizations beats greedy placement."""
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    payload_bytes = 1.0e9
    greedy_submesh = submesh_for_group(4, topology, chip.cores_per_chip)
    greedy_s = collective_cut_bound_s(
        "ag", payload_bytes, greedy_submesh, chip, topology
    )
    min_s, best_submesh = min_collective_cut_bound_s(
        "ag", payload_bytes, 4, chip, CutPlacementOptions(topology=topology)
    )
    assert best_submesh == (1, 2, 2, 1)
    assert min_s < greedy_s
    assert min_s == pytest.approx((0.75 * payload_bytes) / (2.0 * 100.0e9), rel=1e-12)
    # Two concurrent independent groups on the same chip double the inter-chip cut.
    conc_s, _ = min_collective_cut_bound_s(
        "ag",
        payload_bytes,
        4,
        chip,
        CutPlacementOptions(topology=topology, concurrent_groups_per_chip=2),
    )
    assert conc_s == pytest.approx(2.0 * min_s, rel=1e-12)


def test_push_cut_bound_scales_with_local_ranks_and_bisection() -> None:
    """Verify multi-rank push accounts for all local ranks leaving the chip."""
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    payload_bytes = 1.0e9
    # On 2x1x1 (G=4, L=2), 2 cores on a chip send 2/3 of payload off-chip across 1 link.
    cut_s = collective_cut_bound_s("push", payload_bytes, (2, 2, 1, 1), chip, topology)
    expected_s = (2.0 * (2.0 / 3.0) * payload_bytes) / 100.0e9
    assert cut_s == pytest.approx(expected_s, rel=1e-12)


def test_link_subset_bound_does_not_sum_disjoint_axes() -> None:
    """Verify disjoint collectives on x and y take max rather than summing."""
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x1"]
    payload_bytes = 1.0e9
    opts = CutPlacementOptions(topology=topology)
    load_x = collective_axis_cut_bytes("p2p", payload_bytes, (1, 2, 1, 1), chip, opts)
    load_y = collective_axis_cut_bytes(
        "p2p", 2.0 * payload_bytes, (1, 1, 2, 1), chip, opts
    )
    combined = load_x.add(load_y)
    bound_s = eval_link_subset_bound_s(combined, chip, topology)
    expected_max_s = (2.0 * payload_bytes) / 100.0e9
    assert bound_s == pytest.approx(expected_max_s, rel=1e-12)


def test_push_and_a2a_link_subset_bound_no_double_counting() -> None:
    """Verify a2a/push LinkCutLoad matches independent closed-form arithmetic."""
    chip = CHIPS["tpu7x"]
    topology = TOPOLOGIES["2x2x2"]
    submesh = (2, 2, 2, 2)
    payload = 1_000_000.0
    link_bw = 100.0e9
    eps = 1e-12
    opts = CutPlacementOptions(topology=topology)
    # 2x2x2 unwrapped: 1 lane/axis (3 vertex links), local_ranks=2, group=16.
    # Axis bisection (1/2)*2*equiv_payload/(1*100e9) dominates vertex cut.
    expected_by_kind = (
        ("a2a", payload / link_bw),
        ("push", (payload * 16.0 / 15.0) / link_bw),
    )
    for kind, expected_s in expected_by_kind:
        cut_s = collective_cut_bound_s(kind, payload, submesh, chip, topology)
        torus_s = torus_collective_s(kind, payload, submesh, chip, topology)
        load = collective_axis_cut_bytes(kind, payload, submesh, chip, opts)
        subset_s = eval_link_subset_bound_s(load, chip, topology)
        assert cut_s == pytest.approx(expected_s, rel=eps)
        assert subset_s == pytest.approx(expected_s, rel=eps)
        assert cut_s <= torus_s + eps


def _iter_interchip_submeshes(
    topology: SliceTopology,
) -> list[tuple[int, int, int, int]]:
    """Return representative inter-chip submesh factorizations within topology."""
    max_x, max_y, max_z = topology.chip_axes
    x_exts = sorted({e for e in (1, 2, max_x) if e <= max_x})
    y_exts = sorted({e for e in (1, 2, max_y) if e <= max_y})
    z_exts = sorted({e for e in (1, 2, max_z) if e <= max_z})
    return [
        (intra, ax, ay, az)
        for intra in (1, topology.cores_per_chip)
        for ax in x_exts
        for ay in y_exts
        for az in z_exts
        if ax * ay * az > 1
    ]


def test_link_subset_bound_exhaustive_topology_and_nonwrapped_submesh_tightness() -> (
    None
):
    """Verify subset_s == cut_s <= torus_s across all topologies, submeshes, kinds."""
    chip = CHIPS["tpu7x"]
    payload = 1.0e9
    link_bw = 100.0e9
    eps = 1e-12
    kinds = ("ar", "ag", "rs", "p2p", "a2a", "push")
    for topology in TOPOLOGIES.values():
        opts = CutPlacementOptions(topology=topology)
        for submesh in _iter_interchip_submeshes(topology):
            for kind in kinds:
                cut_s = collective_cut_bound_s(kind, payload, submesh, chip, topology)
                torus_s = torus_collective_s(kind, payload, submesh, chip, topology)
                load = collective_axis_cut_bytes(kind, payload, submesh, chip, opts)
                subset_s = eval_link_subset_bound_s(load, chip, topology)
                assert cut_s <= torus_s + eps
                assert subset_s <= torus_s + eps
                assert subset_s == pytest.approx(cut_s, rel=eps)
    # Explicit closed-form check: unwrapped 2-chip submesh inside wrapped 4x4x4
    # must charge 1 lane (not 2 parent wrapped lanes).
    topo_4x4x4 = TOPOLOGIES["4x4x4"]
    opts_4x4x4 = CutPlacementOptions(topology=topo_4x4x4)
    ag_load = collective_axis_cut_bytes("ag", payload, (2, 2, 1, 1), chip, opts_4x4x4)
    expected_ag_s = (0.5 * payload) / link_bw
    assert eval_link_subset_bound_s(ag_load, chip, topo_4x4x4) == pytest.approx(
        expected_ag_s, rel=eps
    )
    push_load = collective_axis_cut_bytes(
        "push", payload, (2, 2, 1, 1), chip, opts_4x4x4
    )
    expected_push_s = (2.0 * (2.0 / 3.0) * payload) / link_bw
    assert eval_link_subset_bound_s(push_load, chip, topo_4x4x4) == pytest.approx(
        expected_push_s, rel=eps
    )


def test_multi_collective_link_cut_load_sound_across_submeshes_and_partitions() -> None:
    """Verify multi-collective LinkCutLoad stays sound on wrapped rings and meshes."""
    chip = CHIPS["tpu7x"]
    topo_4x4x4 = TOPOLOGIES["4x4x4"]
    eps = 1e-12

    # Unwrapped 2-chip AR + full 4-chip AG on wrapped ring: completes in 2.0 ns.
    opts_unkeyed = CutPlacementOptions(topology=topo_4x4x4)
    ar_2 = collective_axis_cut_bytes("ar", 100.0, (1, 2, 1, 1), chip, opts_unkeyed)
    ag_4 = collective_axis_cut_bytes("ag", 400.0, (1, 4, 1, 1), chip, opts_unkeyed)
    assert eval_link_subset_bound_s(ar_2.add(ag_4), chip, topo_4x4x4) == pytest.approx(
        2.0e-9, rel=eps
    )

    # Unwrapped 2-chip AR + full 4-chip AR on wrapped ring: completes in 2.0 ns.
    ar_4 = collective_axis_cut_bytes("ar", 200.0, (1, 4, 1, 1), chip, opts_unkeyed)
    assert eval_link_subset_bound_s(ar_2.add(ar_4), chip, topo_4x4x4) == pytest.approx(
        2.0e-9, rel=eps
    )

    # Orthogonal pair partitions on wrapped 4-ring finish in 1.0 ns on direct links.
    opts_tp = CutPlacementOptions(topology=topo_4x4x4, partition_key="tp")
    opts_outer = CutPlacementOptions(topology=topo_4x4x4, partition_key="pcp")
    ar_tp = collective_axis_cut_bytes("ar", 100.0, (1, 2, 1, 1), chip, opts_tp)
    ar_outer = collective_axis_cut_bytes("ar", 100.0, (1, 2, 1, 1), chip, opts_outer)
    assert eval_link_subset_bound_s(
        ar_tp.add(ar_outer), chip, topo_4x4x4
    ) == pytest.approx(1.0e-9, rel=eps)
    assert eval_link_subset_bound_s(ar_2.add(ar_2), chip, topo_4x4x4) == pytest.approx(
        1.0e-9, rel=eps
    )

    # Repeating the identical partition sums its bisection cut cleanly (2.0 ns).
    assert eval_link_subset_bound_s(
        ar_tp.add(ar_tp), chip, topo_4x4x4
    ) == pytest.approx(2.0e-9, rel=eps)
    assert eval_link_subset_bound_s(
        ar_tp.scaled(2.0), chip, topo_4x4x4
    ) == pytest.approx(2.0e-9, rel=eps)

    # Mixed AR + AG on 2x2x1 accumulates soundly across both vertex and edge families.
    topo_2x2x1 = TOPOLOGIES["2x2x1"]
    opts_2x2x1 = CutPlacementOptions(topology=topo_2x2x1, partition_key="tp")
    ar_2x2 = collective_axis_cut_bytes("ar", 200.0, (2, 2, 2, 1), chip, opts_2x2x1)
    ag_2x2 = collective_axis_cut_bytes("ag", 200.0, (2, 2, 2, 1), chip, opts_2x2x1)
    # AR edge bytes/chip = 300; AG edge bytes/chip = 150 -> 450 / (2 * 100e9) = 2.25 ns.
    assert eval_link_subset_bound_s(
        ar_2x2.add(ag_2x2), chip, topo_2x2x1
    ) == pytest.approx(2.25e-9, rel=eps)


def test_tpu6e_single_core_collectives_and_2d_torus_wraparound() -> None:
    """Verify 1-core packing, external hops, and 16-axis 2D torus wraparound."""
    payload_bytes = 64.0 * 1024.0 * 1024.0
    chip = CHIPS["tpu6e"]
    expected_hop_latency = chip.ici_hop_latency_s
    topo_2x2 = TPU6E_TOPOLOGIES["2x2"]
    assert submesh_for_group(2, topo_2x2, 1) == (1, 2, 1, 1)

    coll_1c = make(
        "tp_ar", "ar", 16.0, 2, CollectiveConfig(algo="one_shot", ranks_per_chip=1)
    )
    coll_default = make("tp_ar", "ar", 16.0, 2)
    assert coll_1c.offchip == pytest.approx(1.0)
    assert collective_ici_hops(coll_1c, topo_2x2, 1) == 2
    assert collective_ici_hops(coll_default, topo_2x2, 1) == 2
    transfer_bytes = 16.0 * 1.0 / 2.0
    expected_phase_time = expected_hop_latency + transfer_bytes / chip.ici_link_bw
    assert torus_collective_s(
        "ar", 16.0, (1, 2, 1, 1), chip, topo_2x2
    ) == pytest.approx(2.0 * expected_phase_time, rel=1e-12)

    for open_name in ("1x1", "2x2", "2x4", "4x4", "4x8", "8x8"):
        assert not TPU6E_TOPOLOGIES[open_name].has_torus_wraparound

    topo_8x16 = TPU6E_TOPOLOGIES["8x16"]
    assert topo_8x16.has_torus_wraparound
    assert not topo_8x16.axis_wrapped(0, 8)
    assert topo_8x16.axis_wrapped(1, 16)
    assert torus_collective_s(
        "ar", payload_bytes, (1, 8, 16, 1), chip, topo_8x16
    ) < torus_collective_s("ar", payload_bytes, (1, 8, 16, 1), chip)

    topo_16x16 = TPU6E_TOPOLOGIES["16x16"]
    assert topo_16x16.has_torus_wraparound
    assert topo_16x16.axis_wrapped(0, 16)
    assert topo_16x16.axis_wrapped(1, 16)
    assert torus_collective_s(
        "ar", payload_bytes, (1, 16, 16, 1), chip, topo_16x16
    ) < torus_collective_s("ar", payload_bytes, (1, 16, 16, 1), chip)
