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

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    TPU6E_TOPOLOGIES,
    collective_ici_hops,
    submesh_for_group,
    torus_collective_s,
)
from vllm_torchtpu.roofline.collectives import make

_EXPECTED_LINK_BW: float = 100.0e9
_EXPECTED_ICI_BW: float = 300.0e9
_EXPECTED_HOP_LATENCY: float = 1.0e-6
_PAYLOAD_BYTES: float = 64.0 * 1024.0 * 1024.0


def test_torus_wraparound_and_mesh_packing() -> None:
    """Verify torus wraparound detection and submesh packing across slices."""
    assert not TOPOLOGIES["2x2x4"].has_torus_wraparound
    assert TOPOLOGIES["4x4x4"].has_torus_wraparound
    chip = CHIPS["tpu7x"]
    assert torus_collective_s(
        "ar", _PAYLOAD_BYTES, (2, 1, 1, 4), chip, TOPOLOGIES["4x4x4"]
    ) < torus_collective_s(
        "ar", _PAYLOAD_BYTES, (2, 1, 1, 4), chip, TOPOLOGIES["2x2x4"]
    )
    assert submesh_for_group(128, TOPOLOGIES["4x4x4"], 2, (2, 4, 1, 1)) == (2, 4, 4, 4)


def test_spec_cited_chip_and_unloaded_hop_latency() -> None:
    """Verify chip spec fields match documented citations and unloaded hops."""
    chip = CHIPS["tpu7x"]
    assert "ici_startup_hops" not in dir(chip)
    assert "ici_transit_hops" not in dir(chip)
    assert "contraction_floor_s" not in dir(chip)
    assert "d2d_latency_s" not in dir(chip)
    assert "usable_hbm_frac" not in dir(chip)
    assert "d2d_bw" not in dir(chip)
    assert chip.ici_link_bw == _EXPECTED_LINK_BW
    assert chip.ici_bw == _EXPECTED_ICI_BW
    assert chip.ici_hop_latency_s == _EXPECTED_HOP_LATENCY
    assert torus_collective_s(
        "ar", 16.0, (2, 2, 2, 1), chip, TOPOLOGIES["2x2x1"]
    ) == pytest.approx(2 * 2 * _EXPECTED_HOP_LATENCY)


def test_tpu6e_single_core_collectives_and_2d_torus_wraparound() -> None:
    """Verify 1-core packing, external hops, and 16-axis 2D torus wraparound."""
    chip = CHIPS["tpu6e"]
    topo_2x2 = TPU6E_TOPOLOGIES["2x2"]
    assert submesh_for_group(2, topo_2x2, 1) == (1, 2, 1, 1)

    coll_1c = make("tp_ar", "ar", 16.0, 2, "one_shot", 1)
    coll_default = make("tp_ar", "ar", 16.0, 2)
    assert coll_1c.offchip == pytest.approx(1.0)
    assert collective_ici_hops(coll_1c, topo_2x2, 1) == 2
    assert collective_ici_hops(coll_default, topo_2x2, 1) == 2
    assert torus_collective_s(
        "ar", 16.0, (1, 2, 1, 1), chip, topo_2x2
    ) == pytest.approx(2 * _EXPECTED_HOP_LATENCY)

    for open_name in ("1x1", "2x2", "2x4", "4x4", "4x8", "8x8"):
        assert not TPU6E_TOPOLOGIES[open_name].has_torus_wraparound

    topo_8x16 = TPU6E_TOPOLOGIES["8x16"]
    assert topo_8x16.has_torus_wraparound
    assert not topo_8x16.axis_wrapped(0, 8)
    assert topo_8x16.axis_wrapped(1, 16)
    assert torus_collective_s(
        "ar", _PAYLOAD_BYTES, (1, 8, 16, 1), chip, topo_8x16
    ) < torus_collective_s("ar", _PAYLOAD_BYTES, (1, 8, 16, 1), chip)

    topo_16x16 = TPU6E_TOPOLOGIES["16x16"]
    assert topo_16x16.has_torus_wraparound
    assert topo_16x16.axis_wrapped(0, 16)
    assert topo_16x16.axis_wrapped(1, 16)
    assert torus_collective_s(
        "ar", _PAYLOAD_BYTES, (1, 16, 16, 1), chip, topo_16x16
    ) < torus_collective_s("ar", _PAYLOAD_BYTES, (1, 16, 16, 1), chip)
