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
"""Structural survival and floor-bound checks for representative public layouts."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import FIXTURES_DIR, load_locked_config
from vllm_torchtpu.roofline.constants import BF16_BYTES, FP8_BYTES
from vllm_torchtpu.roofline.solver import SolveOptions

_REP_FIXTURE_PATH = FIXTURES_DIR / "representative_layouts.json"
_REP_SPEC: dict[str, Any] = json.loads(_REP_FIXTURE_PATH.read_text(encoding="utf-8"))
_CONCURRENCIES: tuple[int, ...] = tuple(int(c) for c in _REP_SPEC["concurrencies"])
_EPS: float = 1e-6


def _pool_matches(
    pool: rf.PoolMeshPlan, expected: dict[str, Any], topology: str
) -> bool:
    """Return True when a solved pool matches the expected structural axes."""
    return (
        pool.topology == topology
        and pool.tp == expected["tp"]
        and pool.dp == expected["dp"]
        and pool.ep == expected["ep"]
        and pool.pp == expected["pp"]
        and pool.pcp == expected["pcp"]
        and pool.kv_split == expected["kv_split"]
    )


def _assert_mixed_layout_present(
    serving_plan: rf.ServingPlan,
    expected: dict[str, Any],
    topology: str,
    concurrency: int,
) -> None:
    """Assert a mixed layout survives enumeration with positive finite floors."""
    matches = [
        point
        for point in serving_plan.candidates
        if point.concurrency == concurrency
        and _pool_matches(point.decode_pool, expected, topology)
    ]
    assert matches, f"Missing mixed layout {expected['label']} at c={concurrency}"
    for point in matches:
        assert math.isfinite(point.bound_tpot_ms)
        assert point.bound_tpot_ms > 0.0
        assert math.isfinite(point.bound_ttft_ms)
        assert point.bound_ttft_ms > 0.0
        assert math.isfinite(point.floor_throughput_per_chip)
        assert point.floor_throughput_per_chip > 0.0
        assert point.bound_tpot_ms <= point.est_tpot_ms + _EPS
        assert point.bound_ttft_ms <= point.est_ttft_ms + _EPS
        assert point.decode_pool.hbm_used_gb <= point.decode_pool.hbm_limit_gb + _EPS


def _assert_pool_layout_present(
    records: tuple[rf.PoolFloorRecord, ...],
    expected: dict[str, Any],
    topology: str,
    concurrency: int,
) -> None:
    """Assert a prefill or decode pool record survives with positive finite floors."""
    matches = [
        record
        for record in records
        if record.concurrency == concurrency
        and _pool_matches(record.pool, expected, topology)
    ]
    assert matches, (
        f"Missing {expected['role']} pool layout {expected['label']} at c={concurrency}"
    )
    for record in matches:
        assert record.pool.hbm_used_gb <= record.pool.hbm_limit_gb + _EPS
        assert math.isfinite(record.token_throughput_per_chip)
        assert record.token_throughput_per_chip > 0.0
        if expected["role"] == "prefill":
            assert record.ttft_floor_ms is not None
            assert record.ttft_ms is not None
            assert math.isfinite(record.ttft_floor_ms)
            assert record.ttft_floor_ms > 0.0
            assert record.ttft_floor_ms <= record.ttft_ms + _EPS
        else:
            assert record.tpot_floor_ms is not None
            assert record.tpot_ms is not None
            assert math.isfinite(record.tpot_floor_ms)
            assert record.tpot_floor_ms > 0.0
            assert record.tpot_floor_ms <= record.tpot_ms + _EPS


@pytest.mark.parametrize(
    "case",
    _REP_SPEC["cases"],
    ids=[case["id"] for case in _REP_SPEC["cases"]],
)
def test_representative_public_layouts_survive_with_valid_floors(
    case: dict[str, Any],
) -> None:
    """Verify representative parallelism layouts survive across round concurrencies."""
    config = load_locked_config(case["checkpoint"])
    geometry = rf.from_hf_config(config)
    base_dtypes = rf.dtypes_from_hf_config(config)
    layer_counts = rf.count_layer_types(geometry)
    is_hybrid = (layer_counts.gdn + layer_counts.kda) > 0
    dtypes = replace(
        base_dtypes,
        kv=FP8_BYTES,
        kv_dtype="fp8",
        kv_source="flag",
        **(
            {
                "state": BF16_BYTES,
                "state_dtype": "bf16",
                "state_source": "flag",
            }
            if is_hybrid
            else {}
        ),
    )
    chip = rf.CHIPS["tpu7x"]
    topology_name = str(case["topology"])
    topology = rf.TOPOLOGIES[topology_name]
    scenario = rf.TrafficScenario(
        name=str(case["id"]),
        isl=int(case["isl"]),
        osl=int(case["osl"]),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=None,
        concurrencies=_CONCURRENCIES,
        kv_working_set_turns=1.0,
    )
    mode_val = str(case["mode"])
    assert mode_val in ("mixed", "disagg", "auto")
    serving_plan = rf.solve_serving_plan(
        geometry,
        dtypes,
        chip,
        scenario,
        SolveOptions(
            mode="mixed"
            if mode_val == "mixed"
            else ("disagg" if mode_val == "disagg" else "auto"),
            max_chips=int(case["max_chips"]),
            topologies=(topology,),
            include_pairs=False,
        ),
    )
    for expected in case["expected_layouts"]:
        assert expected["label"] == rf.format_parallelism(expected)
        for concurrency in _CONCURRENCIES:
            if case["mode"] == "mixed":
                _assert_mixed_layout_present(
                    serving_plan, expected, topology_name, concurrency
                )
            elif expected["role"] == "prefill":
                _assert_pool_layout_present(
                    serving_plan.prefill_pools, expected, topology_name, concurrency
                )
            else:
                _assert_pool_layout_present(
                    serving_plan.decode_pools, expected, topology_name, concurrency
                )
