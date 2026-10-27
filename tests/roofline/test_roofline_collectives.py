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
    Collective,
    CollectiveTable,
    submesh_for_group,
    torus_collective_s,
)

_KIB = 1024

_FIXTURE_32R = CollectiveTable(
    rows={
        ("dispatch_ag", 32): (
            (256 * _KIB, (32.65e-6, 37.17e-6, 35.39e-6)),
            (512 * _KIB, (34.64e-6,)),
            (2048 * _KIB, (28.16e-6,)),
        ),
        ("route_ag", 32): (
            (96 * _KIB, (29.11e-6, 33.58e-6, 32.34e-6)),
            (192 * _KIB, (30.10e-6,)),
            (768 * _KIB, (36.56e-6,)),
        ),
        ("restore_ag", 32): (
            (512 * _KIB, (31.55e-6, 33.67e-6, 34.13e-6)),
            (1024 * _KIB, (31.15e-6,)),
            (4096 * _KIB, (42.19e-6,)),
        ),
        ("ar", 32): (
            (256 * _KIB, (29.41e-6, 25.51e-6)),
            (512 * _KIB, (30.77e-6,)),
            (1024 * _KIB, (38.63e-6,)),
            (4096 * _KIB, (95.09e-6,)),
        ),
    }
)
KIND = {"dispatch_ag": "ag", "route_ag": "ag", "restore_ag": "ag", "ar": "ar"}


def _query(key: tuple[str, int], payload: float) -> Collective:
    return Collective(key[0], key[1], 0.0, 0, 0.0, KIND[key[0]], payload)


def _max_spread(table: CollectiveTable) -> float:
    return max(
        max(s) - min(s) for pts in table.rows.values() for _, s in pts if len(s) > 1
    )


def test_table_loo_interior_within_pooled_spread() -> None:
    tol = _max_spread(_FIXTURE_32R)
    checked = 0
    for key, pts in _FIXTURE_32R.rows.items():
        for i, (payload, samples) in enumerate(pts):
            for j, held_out in enumerate(samples):
                rest = samples[:j] + samples[j + 1 :]
                kept = pts[:i] + (((payload, rest),) if rest else ()) + pts[i + 1 :]
                if not kept[0][0] < payload < kept[-1][0]:
                    continue
                t, _ = CollectiveTable({key: kept}).time_s(_query(key, payload))
                assert abs(t - held_out) <= tol, (key, payload, t, held_out, tol)
                checked += 1
    assert checked > 0


def test_context_invariance_same_payload() -> None:
    tol = _max_spread(_FIXTURE_32R)
    for key, d1_s in ((("restore_ag", 32), 42.35e-6), (("ar", 32), 95.43e-6)):
        t, src = _FIXTURE_32R.time_s(_query(key, 4096 * 1024))
        assert src == "measured"
        assert abs(d1_s - t) <= tol
    assert not TOPOLOGIES["2x2x4"].has_torus_wraparound
    assert TOPOLOGIES["4x4x4"].has_torus_wraparound
    chip = CHIPS["tpu7x"]
    assert torus_collective_s(
        "ar", 64 * 1024 * 1024, (2, 1, 1, 4), chip, TOPOLOGIES["4x4x4"]
    ) < torus_collective_s(
        "ar", 64 * 1024 * 1024, (2, 1, 1, 4), chip, TOPOLOGIES["2x2x4"]
    )
    assert submesh_for_group(128, TOPOLOGIES["4x4x4"], 2, (2, 4, 1, 1)) == (2, 4, 4, 4)


def test_spec_cited_chip_and_unloaded_hop_latency() -> None:
    chip = CHIPS["tpu7x"]
    assert not hasattr(chip, "ici_startup_hops")
    assert not hasattr(chip, "ici_transit_hops")
    assert not hasattr(chip, "contraction_floor_s")
    assert not hasattr(chip, "d2d_latency_s")
    assert not hasattr(chip, "usable_hbm_frac")
    assert not hasattr(chip, "d2d_bw")
    assert chip.ici_link_bw == 100.0e9
    assert chip.ici_bw == 300.0e9
    assert chip.ici_hop_latency_s == 1.0e-6
    assert torus_collective_s(
        "ar", 16.0, (2, 2, 2, 1), chip, TOPOLOGIES["2x2x1"]
    ) == pytest.approx(2 * 2 * 1.0e-6)
