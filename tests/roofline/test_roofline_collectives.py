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
    submesh_for_group,
    torus_collective_s,
)

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
