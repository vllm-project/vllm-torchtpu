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
from typing import TYPE_CHECKING

import pytest

from tests.roofline.conftest import load_locked_config
from vllm_torchtpu import roofline
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    FloorOptions,
    Parallel,
    SpecShape,
    StepShape,
    from_hf_config,
    pass_costs,
    spec_cycle_floor,
    step_floor,
)
from vllm_torchtpu.roofline.geometry import layer_ffn_kinds

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_DECODE_GRID = [1, 4, 16, 64]
_DEFAULT_DECODE_ACTIVE = [10.0, 28.0, 80.0, 200.0]
_DEFAULT_PREFILL_GRID = [1, 64, 512]
_DEFAULT_PREFILL_ACTIVE = [10.0, 200.0, 440.0]


def _write_hull_json(
    path: Path,
    geometry: roofline.ModelGeometry,
    *,
    overrides: dict[str, object] | None = None,
    decode_curve: tuple[list[int], list[float]] | None = None,
    prefill_curve: tuple[list[int], list[float]] | None = None,
) -> Path:
    """Write a v1 MoE activation hull JSON file for testing."""
    decode_grid, decode_active = (
        (_DEFAULT_DECODE_GRID, [float(geometry.top_k), *_DEFAULT_DECODE_ACTIVE[1:]])
        if decode_curve is None
        else decode_curve
    )
    prefill_grid, prefill_active = (
        (_DEFAULT_PREFILL_GRID, [float(geometry.top_k), *_DEFAULT_PREFILL_ACTIVE[1:]])
        if prefill_curve is None
        else prefill_curve
    )
    payload: dict[str, object] = {
        "schema_version": roofline.ACTIVATION_HULL_SCHEMA,
        "num_experts": geometry.num_experts,
        "experts_per_token": geometry.top_k,
        "moe_layers": layer_ffn_kinds(geometry).count("moe"),
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
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    chip = CHIPS["tpu7x"]
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_path = _write_hull_json(tmp_path / "qwen35_hull.json", geometry)

    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    hull_model = roofline.load_activation_hull(hull_path, geometry)
    bernoulli_model = roofline.ExpertActivationModel(regime="bernoulli")

    dispatch_ep = roofline.ExpertDispatch(
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
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
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
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
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
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
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
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
        geometry,
        chip,
        par_ep,
        shape,
        FloorOptions(dtypes=dtypes, activation_model=pigeonhole_model),
    )
    sf_hull = step_floor(
        geometry,
        chip,
        par_ep,
        shape,
        FloorOptions(dtypes=dtypes, activation_model=hull_model),
    )
    sf_bern = step_floor(
        geometry,
        chip,
        par_ep,
        shape,
        FloorOptions(dtypes=dtypes, activation_model=bernoulli_model),
    )
    assert sf_pigeon.hard_floor_s == pytest.approx(sf_hull.hard_floor_s, rel=0, abs=0)
    assert sf_pigeon.envelope_floor_s < sf_hull.envelope_floor_s
    assert sf_hull.envelope_floor_s < sf_bern.envelope_floor_s
    assert sf_bern.envelope_floor_s <= sf_bern.serial_estimate_s


def test_empirical_hull_spec_verify_and_mixed_step_set_union_bounds(
    tmp_path: Path,
) -> None:
    """Speculative verify and mixed steps enforce set-union monotonicity."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    chip = CHIPS["tpu7x"]
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_path = _write_hull_json(tmp_path / "qwen35_hull_set_union.json", geometry)
    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    hull_model = roofline.load_activation_hull(hull_path, geometry)
    bernoulli_model = roofline.ExpertActivationModel(regime="bernoulli")

    # Speculative verify step path: verify MoE module matches 4-seq decode, not 16-seq.
    par_tp = Parallel(tp=8, ep=1, moe="tp_slice")
    sf_spec_pigeon = spec_cycle_floor(
        geometry,
        chip,
        par_tp,
        SpecShape(batch=4, context=4096, spec_k=3),
        FloorOptions(dtypes=dtypes, activation_model=pigeonhole_model),
    )
    sf_spec_hull = spec_cycle_floor(
        geometry,
        chip,
        par_tp,
        SpecShape(batch=4, context=4096, spec_k=3),
        FloorOptions(dtypes=dtypes, activation_model=hull_model),
    )
    sf_spec_bern = spec_cycle_floor(
        geometry,
        chip,
        par_tp,
        SpecShape(batch=4, context=4096, spec_k=3),
        FloorOptions(dtypes=dtypes, activation_model=bernoulli_model),
    )
    assert sf_spec_pigeon.envelope_floor_s < sf_spec_hull.envelope_floor_s
    assert sf_spec_hull.envelope_floor_s < sf_spec_bern.envelope_floor_s
    verify_moe = next(m for m in sf_spec_hull.modules if m.name == "moe")
    decode_4_moe = next(
        m
        for m in pass_costs(
            geometry,
            par_tp,
            StepShape(batch=4, rows_per_seq=1, context=4096),
            dtypes,
            activation_model=hull_model,
        )
        if m.name == "moe"
    )
    decode_16_moe = next(
        m
        for m in pass_costs(
            geometry,
            par_tp,
            StepShape(batch=16, rows_per_seq=1, context=4096),
            dtypes,
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
            geometry,
            par_tp,
            StepShape(batch=8, rows_per_seq=1, context=4096, prefill_tokens=8),
            dtypes,
            activation_model=hull_model,
        )
        if m.name == "moe"
    )
    decode_8_moe = next(
        m
        for m in pass_costs(
            geometry,
            par_tp,
            StepShape(batch=8, rows_per_seq=1, context=4096),
            dtypes,
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
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    hull_path = _write_hull_json(tmp_path / "qwen35_hull_latent.json", geometry)
    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    hull_model = roofline.load_activation_hull(hull_path, geometry)
    par_tp = Parallel(tp=8, ep=1, moe="tp_slice")
    batch_rows = _DEFAULT_DECODE_GRID[2]
    lhs_floor = 32
    latent_hidden = 512

    latent_geom = replace(geometry, routed_expert_hidden=latent_hidden)
    latent_moe = next(
        m
        for m in pass_costs(
            latent_geom,
            par_tp,
            StepShape(batch=batch_rows, rows_per_seq=1, context=4096),
            dtypes,
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
            dtypes,
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
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    chip = CHIPS["tpu7x"]
    dtypes = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    pigeonhole_model = roofline.ExpertActivationModel(regime="pigeonhole")
    for parallel in (
        Parallel(tp=8, ep=8, moe="ep", moe_dispatch="a2a_push"),
        Parallel(tp=8, ep=1, moe="tp_slice"),
    ):
        floor_result = step_floor(
            geometry,
            chip,
            parallel,
            StepShape(batch=16, rows_per_seq=1, context=2048),
            FloorOptions(dtypes=dtypes, activation_model=pigeonhole_model),
        )
        assert floor_result.envelope_floor_s == pytest.approx(
            floor_result.hard_floor_s, rel=0, abs=0
        )


def test_activation_hull_mismatched_experts_per_token_is_rejected(
    tmp_path: Path,
) -> None:
    """Hull loader rejects mismatched experts_per_token."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    bad_topk = _write_hull_json(
        tmp_path / "bad_topk.json",
        geometry,
        overrides={"experts_per_token": geometry.top_k + 2},
    )
    with pytest.raises(ValueError, match="experts_per_token"):
        roofline.load_activation_hull(bad_topk, geometry)


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
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    hull_path = _write_hull_json(
        tmp_path / f"{case_name}.json",
        geometry,
        overrides=overrides,
        decode_curve=decode_curve,
    )
    with pytest.raises(exc_type, match=match):
        roofline.load_activation_hull(hull_path, geometry)


def test_activation_hull_rejects_cap_root_nan_dense_empty_spec_and_constructor(
    tmp_path: Path,
) -> None:
    """Reject over-cap, list root, NaN, dense model, empty hull:, bad constructor."""
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    cap_at_4 = roofline.bernoulli_expected_experts(
        geometry.num_experts, geometry.top_k, 4
    )
    over_cap = _write_hull_json(
        tmp_path / "over_bernoulli.json",
        geometry,
        decode_curve=([1, 4], [float(geometry.top_k), cap_at_4 + 1.0]),
    )
    with pytest.raises(ValueError, match="Bernoulli"):
        roofline.load_activation_hull(over_cap, geometry)

    non_obj_path = tmp_path / "list_root.json"
    non_obj_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        roofline.load_activation_hull(non_obj_path, geometry)

    nan_path = tmp_path / "nan_val.json"
    nan_path.write_text(
        json.dumps(
            {
                "schema_version": roofline.ACTIVATION_HULL_SCHEMA,
                "num_experts": geometry.num_experts,
                "experts_per_token": geometry.top_k,
                "moe_layers": layer_ffn_kinds(geometry).count("moe"),
                "decode_hull": {"token_grid": [1], "active_experts": [math.nan]},
                "prefill_hull": {"token_grid": [1], "active_experts": [10.0]},
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        roofline.load_activation_hull(nan_path, geometry)

    dense_geom = replace(
        geometry, num_experts=0, top_k=0, moe_inter=0, dense_inter=1024
    )
    valid_path = _write_hull_json(tmp_path / "valid.json", geometry)
    with pytest.raises(ValueError, match="routed MoE"):
        roofline.load_activation_hull(valid_path, dense_geom)

    with pytest.raises(ValueError, match="non-empty file path"):
        roofline.parse_activation_spec("hull:", geometry)

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
    geometry = from_hf_config(load_locked_config("qwen3_5_397b_fp8.json"))
    moe_layers = layer_ffn_kinds(geometry).count("moe")
    decode_tokens = [1, 4]
    prefill_tokens = [1]
    prefill_active = [10.0]
    for bool_val in (True, False):
        bool_path = tmp_path / f"bool_{str(bool_val).lower()}.json"
        bool_path.write_text(
            json.dumps(
                {
                    "schema_version": roofline.ACTIVATION_HULL_SCHEMA,
                    "num_experts": geometry.num_experts,
                    "experts_per_token": geometry.top_k,
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
            roofline.load_activation_hull(bool_path, geometry)
        with pytest.raises(ValueError, match="finite"):
            roofline.ExpertActivationModel(
                regime="empirical_hull",
                decode_token_grid=(1, 4),
                decode_active_experts=(bool_val, 4.0),
                prefill_token_grid=(1,),
                prefill_active_experts=(10.0,),
            )


def test_bernoulli_per_shard_uses_ep_busiest_experts() -> None:
    """Verify bernoulli uses busiest-rank while pruning keeps the lower bound."""
    dispatch = roofline.ExpertDispatch(
        num_experts=256,
        experts_per_token=8,
        dispatched_tokens=32,
        ep_shards=8,
    )
    pigeon = roofline.min_experts_per_ep_shard(
        roofline.ExpertActivationModel(regime="pigeonhole"), dispatch
    )
    bern = roofline.min_experts_per_ep_shard(
        roofline.ExpertActivationModel(regime="bernoulli"), dispatch
    )
    avg = roofline.active_experts_envelope(
        roofline.ExpertActivationModel(regime="bernoulli"), dispatch
    ) / float(dispatch.ep_shards)
    assert bern > avg
    assert bern > pigeon
