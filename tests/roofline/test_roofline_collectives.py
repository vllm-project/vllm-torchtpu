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
    torus_collective_s,
)
from vllm_torchtpu.roofline import collectives as rf_collectives
from vllm_torchtpu.roofline.collectives import (
    CollectiveConfig,
    collective_ici_hops,
    make,
    submesh_for_group,
)
from vllm_torchtpu.roofline.constants import (
    DEFAULT_ICI_BW,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
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
    expected_link_bw = DEFAULT_ICI_LINK_BW
    expected_ici_bw = DEFAULT_ICI_BW
    expected_hop_latency = DEFAULT_ICI_HOP_LATENCY_S
    chip = CHIPS["tpu7x"]
    assert "ici_startup_hops" not in dir(chip)
    assert "ici_transit_hops" not in dir(chip)
    assert "contraction_floor_s" not in dir(chip)
    assert "d2d_latency_s" not in dir(chip)
    assert "usable_hbm_frac" not in dir(chip)
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


def test_tpu6e_single_core_collectives_and_2d_torus_wraparound() -> None:
    """Verify 1-core packing, external hops, and 16-axis 2D torus wraparound."""
    expected_hop_latency = DEFAULT_ICI_HOP_LATENCY_S
    payload_bytes = 64.0 * 1024.0 * 1024.0
    chip = CHIPS["tpu6e"]
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
    expected_phase_time = expected_hop_latency + transfer_bytes / DEFAULT_ICI_LINK_BW
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
