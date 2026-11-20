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
"""Unit tests for routed MoE activation envelopes, hulls, and CLI wiring."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pytest

from tests.roofline.conftest import QWEN35_CFG, SMALL_HYBRID_MOE_CFG
from vllm_torchtpu import roofline
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    Dtypes,
    FloorOptions,
    Parallel,
    SolveOptions,
    SpecShape,
    StepShape,
    TrafficScenario,
    cli,
    from_hf_config,
    layer_ffn_kinds,
    pareto_skyline_by_concurrency,
    pass_costs,
    render_html,
    solve_serving_plan,
    spec_cycle_floor,
    step_floor,
)

if TYPE_CHECKING:
    from pathlib import Path

_CLI_USAGE_EXIT_CODE = 2
_DEFAULT_DECODE_GRID = [1, 4, 16, 64]
_DEFAULT_DECODE_ACTIVE = [10.0, 28.0, 80.0, 200.0]
_DEFAULT_PREFILL_GRID = [1, 64, 512]
_DEFAULT_PREFILL_ACTIVE = [10.0, 200.0, 440.0]


def _write_hull_json(
    path: Path,
    geom: roofline.ModelGeometry,
    *,
    overrides: dict[str, object] | None = None,
    decode_curve: tuple[list[int], list[float]] | None = None,
    prefill_curve: tuple[list[int], list[float]] | None = None,
) -> Path:
    """Write a v1 MoE activation hull JSON file for testing."""
    decode_grid, decode_active = (
        (_DEFAULT_DECODE_GRID, [float(geom.top_k), *_DEFAULT_DECODE_ACTIVE[1:]])
        if decode_curve is None
        else decode_curve
    )
    prefill_grid, prefill_active = (
        (_DEFAULT_PREFILL_GRID, [float(geom.top_k), *_DEFAULT_PREFILL_ACTIVE[1:]])
        if prefill_curve is None
        else prefill_curve
    )
    payload: dict[str, object] = {
        "schema_version": roofline.ACTIVATION_HULL_SCHEMA,
        "num_experts": geom.num_experts,
        "experts_per_token": geom.top_k,
        "moe_layers": layer_ffn_kinds(geom).count("moe"),
        "decode_hull": {
            "token_grid": decode_grid,
            "active_experts": decode_active,
        },
        "prefill_hull": {
            "token_grid": prefill_grid,
            "active_experts": prefill_active,
        },
    }
    if overrides:
        payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_empirical_activation_hull_sits_between_pigeonhole_and_bernoulli(
    tmp_path: Path,
) -> None:
    """Empirical hull sits between pigeonhole and Bernoulli across decode/spec/mixed."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_path = _write_hull_json(tmp_path / "qwen35_hull.json", geom)

    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    hull_model = roofline.load_activation_hull(hull_path, geom)
    bernoulli_model = roofline.ExpertActivationModel(regime="bernoulli")

    dispatch_ep = roofline.ExpertDispatch(
        num_experts=geom.num_experts,
        experts_per_token=geom.top_k,
        dispatched_tokens=_DEFAULT_DECODE_GRID[2],
        ep_shards=8,
        is_prefill=False,
        tokens_per_seq=1,
    )
    pigeon_shard = roofline.min_experts_per_ep_shard(pigeonhole_model, dispatch_ep)
    hull_shard = roofline.min_experts_per_ep_shard(hull_model, dispatch_ep)
    bernoulli_shard = roofline.min_experts_per_ep_shard(bernoulli_model, dispatch_ep)
    assert pigeon_shard < hull_shard < bernoulli_shard

    # Non-EP tp_slice uses ep_shards=1 (full routed experts across TP).
    dispatch_tp = roofline.ExpertDispatch(
        num_experts=geom.num_experts,
        experts_per_token=geom.top_k,
        dispatched_tokens=_DEFAULT_DECODE_GRID[2],
        ep_shards=1,
        is_prefill=False,
        tokens_per_seq=1,
    )
    assert roofline.min_experts_per_ep_shard(hull_model, dispatch_tp) == pytest.approx(
        _DEFAULT_DECODE_ACTIVE[2]
    )

    # Origin segment interpolation when first knot > 1 (chord from (0, 0.0)).
    origin_model = roofline.ExpertActivationModel(
        regime="empirical_hull",
        decode_token_grid=(4, 16),
        decode_active_experts=(28.0, 80.0),
        prefill_token_grid=(4, 64),
        prefill_active_experts=(28.0, 200.0),
    )
    origin_dispatch = roofline.ExpertDispatch(
        num_experts=geom.num_experts,
        experts_per_token=geom.top_k,
        dispatched_tokens=2,
        ep_shards=1,
        is_prefill=False,
        tokens_per_seq=1,
    )
    expected_origin = (2.0 / 4.0) * 28.0
    assert roofline.min_experts_per_ep_shard(
        origin_model, origin_dispatch
    ) == pytest.approx(expected_origin)

    # Speculative verify uses independent sequence count (set-union monotonicity).
    dispatch_verify = roofline.ExpertDispatch(
        num_experts=geom.num_experts,
        experts_per_token=geom.top_k,
        dispatched_tokens=_DEFAULT_DECODE_GRID[2],
        ep_shards=1,
        is_prefill=False,
        tokens_per_seq=4,
    )
    assert roofline.min_experts_per_ep_shard(
        hull_model, dispatch_verify
    ) == pytest.approx(_DEFAULT_DECODE_ACTIVE[1])

    par_ep = Parallel(tp=8, ep=8, moe="ep", moe_dispatch="a2a_push")
    shape = StepShape(batch=_DEFAULT_DECODE_GRID[2], rows_per_seq=1, context=4096)
    sf_pigeon = step_floor(
        geom,
        chip,
        par_ep,
        shape,
        FloorOptions(dt=dt, activation_model=pigeonhole_model),
    )
    sf_hull = step_floor(
        geom,
        chip,
        par_ep,
        shape,
        FloorOptions(dt=dt, activation_model=hull_model),
    )
    sf_bern = step_floor(
        geom,
        chip,
        par_ep,
        shape,
        FloorOptions(dt=dt, activation_model=bernoulli_model),
    )
    assert sf_pigeon.hard_floor_s == pytest.approx(sf_hull.hard_floor_s, rel=0, abs=0)
    assert sf_pigeon.envelope_floor_s < sf_hull.envelope_floor_s
    assert sf_hull.envelope_floor_s < sf_bern.envelope_floor_s
    assert sf_bern.envelope_floor_s <= sf_bern.serial_estimate_s


def test_empirical_hull_spec_verify_and_mixed_step_set_union_bounds(
    tmp_path: Path,
) -> None:
    """Speculative verify and mixed steps enforce set-union monotonicity."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_path = _write_hull_json(tmp_path / "qwen35_hull_set_union.json", geom)
    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    hull_model = roofline.load_activation_hull(hull_path, geom)
    bernoulli_model = roofline.ExpertActivationModel(regime="bernoulli")

    # Speculative verify step path: verify MoE module matches 4-seq decode, not 16-seq.
    par_tp = Parallel(tp=8, ep=1, moe="tp_slice")
    sf_spec_pigeon = spec_cycle_floor(
        geom,
        chip,
        par_tp,
        SpecShape(batch=4, context=4096, k=3),
        FloorOptions(dt=dt, activation_model=pigeonhole_model),
    )
    sf_spec_hull = spec_cycle_floor(
        geom,
        chip,
        par_tp,
        SpecShape(batch=4, context=4096, k=3),
        FloorOptions(dt=dt, activation_model=hull_model),
    )
    sf_spec_bern = spec_cycle_floor(
        geom,
        chip,
        par_tp,
        SpecShape(batch=4, context=4096, k=3),
        FloorOptions(dt=dt, activation_model=bernoulli_model),
    )
    assert sf_spec_pigeon.envelope_floor_s < sf_spec_hull.envelope_floor_s
    assert sf_spec_hull.envelope_floor_s < sf_spec_bern.envelope_floor_s
    verify_moe = next(m for m in sf_spec_hull.modules if m.name == "moe")
    decode_4_moe = next(
        m
        for m in pass_costs(
            geom,
            par_tp,
            StepShape(batch=4, rows_per_seq=1, context=4096),
            dt,
            activation_model=hull_model,
        )
        if m.name == "moe"
    )
    decode_16_moe = next(
        m
        for m in pass_costs(
            geom,
            par_tp,
            StepShape(batch=16, rows_per_seq=1, context=4096),
            dt,
            activation_model=hull_model,
        )
        if m.name == "moe"
    )
    assert verify_moe.envelope_hbm_bytes == pytest.approx(
        decode_4_moe.envelope_hbm_bytes
    )
    assert verify_moe.envelope_hbm_bytes < decode_16_moe.envelope_hbm_bytes

    # Mixed step uses set-union lower bound max(decode_hull(8), prefill_hull(8)).
    mixed_moe = next(
        m
        for m in pass_costs(
            geom,
            par_tp,
            StepShape(batch=8, rows_per_seq=1, context=4096, prefill_tokens=8),
            dt,
            activation_model=hull_model,
        )
        if m.name == "moe"
    )
    decode_8_moe = next(
        m
        for m in pass_costs(
            geom,
            par_tp,
            StepShape(batch=8, rows_per_seq=1, context=4096),
            dt,
            activation_model=hull_model,
        )
        if m.name == "moe"
    )
    assert mixed_moe.envelope_hbm_bytes == pytest.approx(
        decode_8_moe.envelope_hbm_bytes
    )
    assert mixed_moe.envelope_hbm_bytes < decode_16_moe.envelope_hbm_bytes


def test_latent_projection_flops_remain_unpadded_by_moe_envelope(
    tmp_path: Path,
) -> None:
    """Envelope avoids grouped-matmul padding on latent FLOPs; strict stays intact."""
    geom = from_hf_config(QWEN35_CFG)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_path = _write_hull_json(tmp_path / "qwen35_hull_latent.json", geom)
    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    hull_model = roofline.load_activation_hull(hull_path, geom)
    par_tp = Parallel(tp=8, ep=1, moe="tp_slice")
    batch_rows = _DEFAULT_DECODE_GRID[2]
    lhs_floor = 32
    latent_hidden = 512

    latent_geom = replace(geom, routed_expert_hidden=latent_hidden)
    latent_moe = next(
        m
        for m in pass_costs(
            latent_geom,
            par_tp,
            StepShape(batch=batch_rows, rows_per_seq=1, context=4096),
            dt,
            mxu_lhs_rows_fp8=lhs_floor,
            activation_model=hull_model,
        )
        if m.name == "moe"
    )
    pigeon_latent_moe = next(
        m
        for m in pass_costs(
            latent_geom,
            par_tp,
            StepShape(batch=batch_rows, rows_per_seq=1, context=4096),
            dt,
            mxu_lhs_rows_fp8=lhs_floor,
            activation_model=pigeonhole_model,
        )
        if m.name == "moe"
    )
    moe_layers = layer_ffn_kinds(latent_geom).count("moe")
    routed_base_fp8 = float(
        2
        * batch_rows
        * moe_layers
        * latent_geom.top_k
        * 3
        * latent_hidden
        * (latent_geom.moe_inter / par_tp.tp)
    )
    latent_fp8 = float(
        2 * batch_rows * moe_layers * 2 * latent_geom.hidden * latent_hidden / par_tp.tp
    )
    shared_fp8 = float(
        2
        * batch_rows
        * moe_layers
        * 3
        * latent_geom.hidden
        * (latent_geom.shared_inter // par_tp.tp)
    )
    strict_mult = float(math.ceil(batch_rows / lhs_floor) * lhs_floor) / float(
        batch_rows
    )
    hull_active = _DEFAULT_DECODE_ACTIVE[2]
    total_assignments = float(batch_rows * latent_geom.top_k)
    envelope_mult = (
        max(float(math.ceil(total_assignments / lhs_floor)), hull_active) * lhs_floor
    ) / total_assignments
    expected_strict_fp8 = routed_base_fp8 * strict_mult + latent_fp8 + shared_fp8
    expected_envelope_fp8 = routed_base_fp8 * envelope_mult + latent_fp8 + shared_fp8
    assert pigeon_latent_moe.strict_fp8_flops == pytest.approx(
        expected_strict_fp8, rel=1e-12
    )
    assert pigeon_latent_moe.envelope_fp8_flops == pytest.approx(
        expected_strict_fp8, rel=1e-12
    )
    assert latent_moe.strict_fp8_flops == pytest.approx(expected_strict_fp8, rel=1e-12)
    assert latent_moe.envelope_fp8_flops == pytest.approx(
        expected_envelope_fp8, rel=1e-12
    )


def test_pigeonhole_regime_matches_hard_floor() -> None:
    """Pigeonhole activation regime matches hard_floor_s identically on EP and TP."""
    geom = from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    for par in (
        Parallel(tp=8, ep=8, moe="ep", moe_dispatch="a2a_push"),
        Parallel(tp=8, ep=1, moe="tp_slice"),
    ):
        sf = step_floor(
            geom,
            chip,
            par,
            StepShape(batch=16, rows_per_seq=1, context=2048),
            FloorOptions(dt=dt, activation_model=pigeonhole_model),
        )
        assert sf.envelope_floor_s == pytest.approx(sf.hard_floor_s, rel=0, abs=0)


def test_activation_hull_mismatched_experts_per_token_is_rejected(
    tmp_path: Path,
) -> None:
    """Hull loader rejects mismatched experts_per_token."""
    geom = from_hf_config(QWEN35_CFG)
    bad_topk = _write_hull_json(
        tmp_path / "bad_topk.json",
        geom,
        overrides={"experts_per_token": geom.top_k + 2},
    )
    with pytest.raises(ValueError, match="experts_per_token"):
        roofline.load_activation_hull(bad_topk, geom)


@pytest.mark.parametrize(
    "case",
    [
        (
            "bad_schema",
            {"schema_version": "vllm_torchtpu.roofline.moe_activation_hull/v99"},
            None,
            ValueError,
            "schema_version",
        ),
        (
            "bad_num_experts",
            {"num_experts": 128},
            None,
            ValueError,
            "num_experts",
        ),
        (
            "bad_moe_layers",
            {"moe_layers": 1},
            None,
            ValueError,
            "moe_layers",
        ),
        (
            "bool_field",
            {"num_experts": True},
            None,
            TypeError,
            "integer",
        ),
        (
            "non_object_curve",
            {"decode_hull": [1, 2, 3]},
            None,
            TypeError,
            "JSON object",
        ),
        (
            "unequal_arrays",
            None,
            ([1, 4], [10.0]),
            ValueError,
            "equal-length",
        ),
        (
            "non_positive_grid",
            None,
            ([0, 4], [10.0, 20.0]),
            ValueError,
            "positive ints",
        ),
        (
            "non_increasing_grid",
            None,
            ([4, 4], [20.0, 25.0]),
            ValueError,
            "strictly increase",
        ),
        (
            "decreasing_active",
            None,
            ([1, 4], [10.0, 9.0]),
            ValueError,
            "non-decreasing",
        ),
        (
            "below_top_k",
            None,
            ([1, 4], [5.0, 15.0]),
            ValueError,
            ">= 10",
        ),
        (
            "non_concave",
            None,
            ([1, 4, 16], [10.0, 15.0, 110.0]),
            ValueError,
            "concave",
        ),
    ],
)
def test_activation_hull_loader_rejects_invalid_inputs(
    tmp_path: Path,
    case: tuple[
        str,
        dict[str, object] | None,
        tuple[list[int], list[float]] | None,
        type[Exception],
        str,
    ],
) -> None:
    """Hull loader rejects malformed schemas, mismatched dims, and invalid curves."""
    case_name, overrides, decode_curve, exc_type, match = case
    geom = from_hf_config(QWEN35_CFG)
    hull_path = _write_hull_json(
        tmp_path / f"{case_name}.json",
        geom,
        overrides=overrides,
        decode_curve=decode_curve,
    )
    with pytest.raises(exc_type, match=match):
        roofline.load_activation_hull(hull_path, geom)


def test_activation_hull_rejects_cap_root_nan_dense_empty_spec_and_constructor(
    tmp_path: Path,
) -> None:
    """Reject over-cap, list root, NaN, dense model, empty hull:, bad constructor."""
    geom = from_hf_config(QWEN35_CFG)
    cap_at_4 = roofline.bernoulli_expected_experts(geom.num_experts, geom.top_k, 4)
    over_cap = _write_hull_json(
        tmp_path / "over_bernoulli.json",
        geom,
        decode_curve=([1, 4], [float(geom.top_k), cap_at_4 + 1.0]),
    )
    with pytest.raises(ValueError, match="Bernoulli"):
        roofline.load_activation_hull(over_cap, geom)

    non_obj_path = tmp_path / "list_root.json"
    non_obj_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        roofline.load_activation_hull(non_obj_path, geom)

    nan_path = tmp_path / "nan_val.json"
    nan_path.write_text(
        json.dumps(
            {
                "schema_version": roofline.ACTIVATION_HULL_SCHEMA,
                "num_experts": geom.num_experts,
                "experts_per_token": geom.top_k,
                "moe_layers": layer_ffn_kinds(geom).count("moe"),
                "decode_hull": {"token_grid": [1], "active_experts": [math.nan]},
                "prefill_hull": {"token_grid": [1], "active_experts": [10.0]},
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        roofline.load_activation_hull(nan_path, geom)

    dense_geom = replace(geom, num_experts=0, top_k=0, moe_inter=0, dense_inter=1024)
    valid_path = _write_hull_json(tmp_path / "valid.json", geom)
    with pytest.raises(ValueError, match="routed MoE"):
        roofline.load_activation_hull(valid_path, dense_geom)

    with pytest.raises(ValueError, match="non-empty file path"):
        roofline.parse_activation_spec("hull:", geom)

    with pytest.raises(ValueError, match="concave"):
        roofline.ExpertActivationModel(
            regime="empirical_hull",
            decode_token_grid=(1, 4, 16),
            decode_active_experts=(10.0, 15.0, 110.0),
            prefill_token_grid=(1, 16),
            prefill_active_experts=(10.0, 80.0),
        )


def test_activation_hull_rejects_boolean_active_experts(
    tmp_path: Path,
) -> None:
    """Hull loader rejects booleans in active_experts arrays."""
    geom = from_hf_config(QWEN35_CFG)
    moe_layers = layer_ffn_kinds(geom).count("moe")
    decode_tokens = [1, 4]
    prefill_tokens = [1]
    prefill_active = [10.0]
    for bool_val in (True, False):
        bool_path = tmp_path / f"bool_{str(bool_val).lower()}.json"
        bool_path.write_text(
            json.dumps(
                {
                    "schema_version": roofline.ACTIVATION_HULL_SCHEMA,
                    "num_experts": geom.num_experts,
                    "experts_per_token": geom.top_k,
                    "moe_layers": moe_layers,
                    "decode_hull": {
                        "token_grid": decode_tokens,
                        "active_experts": [bool_val, 4.0],
                    },
                    "prefill_hull": {
                        "token_grid": prefill_tokens,
                        "active_experts": prefill_active,
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(TypeError, match="finite"):
            roofline.load_activation_hull(bool_path, geom)
        with pytest.raises(ValueError, match="finite"):
            roofline.ExpertActivationModel(
                regime="empirical_hull",
                decode_token_grid=(1, 4),
                decode_active_experts=(bool_val, 4.0),
                prefill_token_grid=(1,),
                prefill_active_experts=(10.0,),
            )


def test_default_bernoulli_regime_leaves_hard_floor_and_serial_estimate_unchanged() -> (
    None
):
    """Default bernoulli keeps hard_floor_s, serial_estimate_s, and verdicts fixed."""
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    par = Parallel(tp=8, ep=8, moe="ep", moe_dispatch="a2a_push")

    sf_default = step_floor(
        geom,
        chip,
        par,
        StepShape(batch=8, rows_per_seq=1, context=1024),
        FloorOptions(dt=dt),
    )
    sf_pigeon = step_floor(
        geom,
        chip,
        par,
        StepShape(batch=8, rows_per_seq=1, context=1024),
        FloorOptions(
            dt=dt,
            activation_model=roofline.ExpertActivationModel(regime="pigeonhole"),
        ),
    )
    assert sf_default.hard_floor_s == pytest.approx(
        sf_pigeon.hard_floor_s, rel=0, abs=0
    )
    assert sf_default.serial_estimate_s == pytest.approx(
        sf_pigeon.serial_estimate_s, rel=0, abs=0
    )
    assert sf_default.hard_floor_s < sf_default.envelope_floor_s
    assert sf_default.envelope_floor_s <= sf_default.serial_estimate_s

    probe_scenario = TrafficScenario(
        name="bernoulli_probe",
        isl=512,
        osl=256,
        concurrencies=(16,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        kv_working_set_turns=1.0,
        ttft_slo_ms=2000.0,
        tpot_slo_ms=None,
        think_time_s=0.0,
    )
    probe_plan = solve_serving_plan(
        geom,
        dt,
        chip,
        probe_scenario,
        SolveOptions(mode="mixed", max_chips=4, topologies=(TOPOLOGIES["2x2x1"],)),
    )
    separated = [
        pt for pt in probe_plan.candidates if pt.bound_tpot_ms < pt.envelope_tpot_ms
    ]
    assert separated
    mid_slo = 0.5 * (separated[0].bound_tpot_ms + separated[0].envelope_tpot_ms)
    scenario = replace(probe_scenario, name="bernoulli_invariance", tpot_slo_ms=mid_slo)
    plan_default = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4, topologies=(TOPOLOGIES["2x2x1"],)),
    )
    plan_pigeon = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(
            mode="mixed",
            max_chips=4,
            topologies=(TOPOLOGIES["2x2x1"],),
            activation_model=roofline.ExpertActivationModel(regime="pigeonhole"),
        ),
    )
    assert len(plan_default.candidates) == len(plan_pigeon.candidates)
    for pt_def, pt_pig in zip(
        plan_default.candidates, plan_pigeon.candidates, strict=True
    ):
        assert pt_def.bound_tpot_ms == pytest.approx(pt_pig.bound_tpot_ms, rel=0, abs=0)
        assert pt_def.bound_ttft_ms == pytest.approx(pt_pig.bound_ttft_ms, rel=0, abs=0)
        assert pt_def.est_tpot_ms == pytest.approx(pt_pig.est_tpot_ms, rel=0, abs=0)
        assert pt_def.slo_verdict == pt_pig.slo_verdict
    assert plan_default.search is not None
    assert plan_default.search.envelope_feasible < plan_default.search.not_ruled_out
    assert (
        plan_default.search.enumerated
        >= plan_default.search.hbm_fit
        >= plan_default.search.not_ruled_out
        >= plan_default.search.envelope_feasible
        >= plan_default.search.slo_met
        >= 0
    )


@pytest.mark.parametrize("mode", ["mixed", "disagg"])
@pytest.mark.parametrize("think_time_s", [0.0, 2.0])
def test_empirical_hull_prunes_across_modes_and_think_times(
    tmp_path: Path,
    mode: Literal["mixed", "disagg"],
    think_time_s: float,
) -> None:
    """Hull prunes ruled-out points and tracks envelope residency for mixed & disagg."""
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_path = _write_hull_json(
        tmp_path / f"small_hull_{mode}_{think_time_s}.json",
        geom,
        decode_curve=([1, 4, 16, 64], [float(geom.top_k), 6.0, 12.0, 15.0]),
        prefill_curve=([1, 16, 128], [float(geom.top_k), 12.0, 15.5]),
    )
    hull_model = roofline.load_activation_hull(hull_path, geom)
    max_chips = 4 if mode == "mixed" else 8
    topos = (TOPOLOGIES["2x2x1"],)
    base_scenario = TrafficScenario(
        name=f"hull_{mode}_{think_time_s}",
        isl=256,
        osl=1024,
        concurrencies=(64,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        kv_working_set_turns=1.0,
        ttft_slo_ms=50.0,
        tpot_slo_ms=None,
        think_time_s=think_time_s,
    )
    unconstrained = solve_serving_plan(
        geom,
        dt,
        chip,
        base_scenario,
        SolveOptions(
            mode=mode,
            max_chips=max_chips,
            topologies=topos,
            activation_model=hull_model,
        ),
    )
    separated = [
        pt for pt in unconstrained.candidates if pt.bound_tpot_ms < pt.envelope_tpot_ms
    ]
    assert separated
    assert all(
        1.0 <= pt.min_decode_seqs <= pt.envelope_decode_seqs
        for pt in unconstrained.candidates
    )
    target_pt = max(separated, key=lambda pt: pt.envelope_tpot_ms - pt.bound_tpot_ms)
    mid_slo = 0.5 * (target_pt.bound_tpot_ms + target_pt.envelope_tpot_ms)
    constrained = replace(base_scenario, tpot_slo_ms=mid_slo)
    plan_bern = solve_serving_plan(
        geom,
        dt,
        chip,
        constrained,
        SolveOptions(mode=mode, max_chips=max_chips, topologies=topos),
    )
    plan_hull = solve_serving_plan(
        geom,
        dt,
        chip,
        constrained,
        SolveOptions(
            mode=mode,
            max_chips=max_chips,
            topologies=topos,
            activation_model=hull_model,
        ),
    )
    assert plan_bern.search is not None
    assert plan_hull.search is not None
    assert plan_hull.search.not_ruled_out < plan_bern.search.not_ruled_out
    assert plan_hull.search.not_ruled_out == plan_hull.search.envelope_feasible


def test_pareto_skyline_uses_envelope_tpot_under_empirical_hull() -> None:
    """Hull skyline uses envelope_tpot_ms across family, anchor, slot fill, and sort."""
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_model = roofline.ExpertActivationModel(
        regime="empirical_hull",
        decode_token_grid=(1, 4, 16),
        decode_active_experts=(float(geom.top_k), 6.0, 12.0),
        prefill_token_grid=(1, 16),
        prefill_active_experts=(float(geom.top_k), 12.0),
    )
    bern_model = roofline.ExpertActivationModel(regime="bernoulli")
    scenario = TrafficScenario(
        name="skyline_envelope_order",
        isl=256,
        osl=128,
        concurrencies=(16,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        kv_working_set_turns=1.0,
        think_time_s=0.0,
    )
    base_plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(
            mode="mixed",
            max_chips=4,
            topologies=(TOPOLOGIES["2x2x1"],),
            activation_model=hull_model,
        ),
    )
    template = base_plan.candidates[0]

    def _make_group(
        tp: int,
        metrics: tuple[float, float, float, float],
        *,
        kv_split: int = 1,
    ) -> roofline.ParetoPoint:
        """Build synthetic ParetoPoint with controlled TP group and TPOT floors."""
        tput, inter, bound_tpot, envelope_tpot = metrics
        pool = replace(template.decode_pool, tp=tp, kv_split=kv_split)
        return replace(
            template,
            decode_pool=pool,
            slo_met=True,
            token_throughput_per_chip=tput,
            est_interactivity_tok_s=inter,
            bound_tpot_ms=bound_tpot,
            envelope_tpot_ms=envelope_tpot,
        )

    # 1. Family winner (_skyline_preference): same mesh family, different kv_split.
    kv_split_hull_winner = 2
    fam_v1 = _make_group(1, (1000.0, 100.0, 1.0, 5.0), kv_split=1)
    fam_v2 = _make_group(1, (1000.0, 100.0, 2.0, 3.0), kv_split=kv_split_hull_winner)
    hull_fam = pareto_skyline_by_concurrency(
        replace(base_plan, candidates=(fam_v1, fam_v2), activation_model=hull_model),
        k=1,
    )[16][0]
    bern_fam = pareto_skyline_by_concurrency(
        replace(base_plan, candidates=(fam_v1, fam_v2), activation_model=bern_model),
        k=1,
    )[16][0]
    assert hull_fam.decode_pool.kv_split == kv_split_hull_winner
    assert bern_fam.decode_pool.kv_split == 1

    # 2 & 3. Anchor 1 (k=1) and final sort (k=2) on tied-throughput groups.
    tp_hull_winner = 4
    grp_a = _make_group(2, (1000.0, 100.0, 1.0, 5.0))
    grp_b = _make_group(tp_hull_winner, (1000.0, 100.0, 2.0, 3.0))
    hull_pair = pareto_skyline_by_concurrency(
        replace(base_plan, candidates=(grp_a, grp_b), activation_model=hull_model),
        k=2,
    )[16]
    bern_pair = pareto_skyline_by_concurrency(
        replace(base_plan, candidates=(grp_a, grp_b), activation_model=bern_model),
        k=2,
    )[16]
    assert [pt.decode_pool.tp for pt in hull_pair] == [tp_hull_winner, 2]
    assert [pt.decode_pool.tp for pt in bern_pair] == [2, tp_hull_winner]
    assert (
        pareto_skyline_by_concurrency(
            replace(base_plan, candidates=(grp_a, grp_b), activation_model=hull_model),
            k=1,
        )[16][0].decode_pool.tp
        == tp_hull_winner
    )

    # 4. Slot fill (_fill_skyline_slots): 3 anchors take slots 1..3; slot 4 fills.
    fill_candidates = (
        _make_group(1, (500.0, 50.0, 0.5, 0.5)),
        _make_group(2, (2000.0, 50.0, 4.0, 4.0)),
        _make_group(4, (500.0, 200.0, 4.0, 4.0)),
        _make_group(8, (900.0, 90.0, 1.5, 3.5)),
        _make_group(16, (900.0, 90.0, 2.5, 2.0)),
    )
    hull_fill_tps = {
        pt.decode_pool.tp
        for pt in pareto_skyline_by_concurrency(
            replace(base_plan, candidates=fill_candidates, activation_model=hull_model),
            k=4,
        )[16]
    }
    bern_fill_tps = {
        pt.decode_pool.tp
        for pt in pareto_skyline_by_concurrency(
            replace(base_plan, candidates=fill_candidates, activation_model=bern_model),
            k=4,
        )[16]
    }
    assert hull_fill_tps == {1, 2, 4, 16}
    assert bern_fill_tps == {1, 2, 4, 8}


def _assert_legacy_html_omits_absent_activation_fields(
    out_hull: dict[str, object],
) -> None:
    """Verify older solve/v2 payloads lacking activation keys render cleanly."""
    legacy_payload = dict(out_hull)
    legacy_payload.pop("activation_regime", None)
    raw_search = legacy_payload["search"]
    assert isinstance(raw_search, dict)
    legacy_search = dict(raw_search)
    legacy_search.pop("envelope_feasible", None)
    legacy_payload["search"] = legacy_search
    legacy_html = render_html(legacy_payload)
    assert "MoE Activation:" not in legacy_html
    assert "Envelope Feasible" not in legacy_html


def test_cli_moe_activation_flag_sets_payload_activation_regime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI --moe-activation sets activation_regime across solve/emit/HTML and prunes."""
    cfg_path = tmp_path / "small_moe_config.json"
    cfg_path.write_text(json.dumps(SMALL_HYBRID_MOE_CFG), encoding="utf-8")
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    hull_path = _write_hull_json(
        tmp_path / "small_moe_hull.json",
        geom,
        decode_curve=([1, 4, 16], [float(geom.top_k), 6.0, 12.0]),
        prefill_curve=([1, 16, 128], [float(geom.top_k), 12.0, 15.5]),
    )
    shared_args = [
        "--config",
        str(cfg_path),
        "--chip",
        "tpu7x",
        "--mode",
        "mixed",
        "--max-chips",
        "4",
        "--isl",
        "256",
        "--osl",
        "128",
        "--concurrency",
        "16",
        "--prefix-hit-rate",
        "0.0",
        "--accept-len",
        "1.0",
        "--think-time-s",
        "0.0",
        "--json",
    ]
    solve_args = ["solve", *shared_args]

    assert cli.main([*solve_args, "--moe-activation", "pigeonhole"]) == 0
    out_pigeon = json.loads(capsys.readouterr().out)
    assert out_pigeon["activation_regime"] == "pigeonhole"
    assert "envelope_feasible" in out_pigeon["search"]

    assert cli.main([*solve_args, "--moe-activation", "bernoulli"]) == 0
    out_bern = json.loads(capsys.readouterr().out)
    assert out_bern["activation_regime"] == "bernoulli"
    assert "envelope_*" in out_bern["provenance"]["basis"]

    # emit JSON also records activation_regime inside serving_plan.
    assert (
        cli.main(["emit", *shared_args, "--moe-activation", f"hull:{hull_path}"]) == 0
    )
    out_emit = json.loads(capsys.readouterr().out)
    assert out_emit["serving_plan"]["activation_regime"] == "empirical_hull"

    html_out = tmp_path / "report_hull.html"
    assert (
        cli.main(
            [
                *solve_args,
                "--moe-activation",
                f"hull:{hull_path}",
                "--html",
                str(html_out),
            ]
        )
        == 0
    )
    out_hull = json.loads(capsys.readouterr().out)
    assert out_hull["activation_regime"] == "empirical_hull"
    html_text = html_out.read_text(encoding="utf-8")
    assert "empirical_hull" in html_text
    assert "No SLO set" in html_text
    assert "Envelope Feasible" not in html_text
    _assert_legacy_html_omits_absent_activation_fields(out_hull)

    # With bounded TTFT SLO ensuring sound batch >= 2, pick TPOT SLO in the hull gap.
    slo_solve_args = [
        "solve",
        "--config",
        str(cfg_path),
        "--chip",
        "tpu7x",
        "--mode",
        "mixed",
        "--max-chips",
        "4",
        "--isl",
        "256",
        "--osl",
        "1024",
        "--concurrency",
        "64",
        "--prefix-hit-rate",
        "0.0",
        "--accept-len",
        "1.0",
        "--think-time-s",
        "0.0",
        "--ttft-slo-ms",
        "50.0",
        "--json",
    ]
    assert cli.main([*slo_solve_args, "--moe-activation", f"hull:{hull_path}"]) == 0
    out_hull_slo = json.loads(capsys.readouterr().out)
    envelope_tpots = sorted(pt["envelope_tpot_ms"] for pt in out_hull_slo["candidates"])
    bound_tpots = sorted(pt["bound_tpot_ms"] for pt in out_hull_slo["candidates"])
    assert bound_tpots[-1] < envelope_tpots[-1]
    mid_slo = 0.5 * (max(bound_tpots[-1], envelope_tpots[0]) + envelope_tpots[-1])
    assert (
        cli.main(
            [
                *slo_solve_args,
                "--tpot-slo-ms",
                f"{mid_slo:.6f}",
                "--moe-activation",
                "bernoulli",
            ]
        )
        == 0
    )
    out_prune_bern = json.loads(capsys.readouterr().out)
    html_slo = tmp_path / "report_hull_slo.html"
    assert (
        cli.main(
            [
                *slo_solve_args,
                "--tpot-slo-ms",
                f"{mid_slo:.6f}",
                "--moe-activation",
                f"hull:{hull_path}",
                "--html",
                str(html_slo),
            ]
        )
        == 0
    )
    assert "Envelope Feasible" in html_slo.read_text(encoding="utf-8")
    out_prune_hull = json.loads(capsys.readouterr().out)
    assert (
        out_prune_hull["search"]["not_ruled_out"]
        < out_prune_bern["search"]["not_ruled_out"]
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main([*solve_args, "--moe-activation", "invalid_regime"])
    assert exc_info.value.code == _CLI_USAGE_EXIT_CODE
