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
"""Soundness envelope, verdict ranking, and empirical validation tests."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from tests.megakernel.test_advisor_bounds import (
    case_chips,
    case_layouts,
    case_workload,
    iter_case_grid,
)
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    specs,
    verdict,
)
from vllm_torchtpu.megakernel.advisor.roofline import (
    _evaluate_scopes,
    _evaluate_scopes_with_context,
    _EvaluatedScopes,
)
from vllm_torchtpu.megakernel.core import errors
from vllm_torchtpu.megakernel.core.exec_spec import Tiles

MEASURED_CORPUS_PATH = (
    Path(__file__).resolve().parent / "data" / "advisor" / "measured_points.json"
)
REQUIRED_MEASURED_KEYS = frozenset(
    {
        "id",
        "provenance_kind",
        "model",
        "chip",
        "scope",
        "workload",
        "layout",
        "step_time_us",
        "n",
    }
)
REQUIRED_PUBLIC_SOURCE_KEYS = frozenset(
    {"source_url", "source_quote", "approximate", "assumptions", "rows_quote"}
)
PHASES: dict[str, advisor.Phase] = {
    "decode": "decode",
    "prefill": "prefill",
    "verify": "verify",
    "mixed": "mixed",
}
MOE_MODES: dict[str, advisor.MoEMode] = {
    "masked_on_chip": "masked_on_chip",
    "grouped_on_chip": "grouped_on_chip",
    "grouped_in_hbm": "grouped_in_hbm",
}
VALID_BINDING_UNITS = frozenset({"hbm", "matmul", "vector", "comm", "launch"})
VALID_SCOPE_ASSUMPTIONS = frozenset(
    {
        "launch_gap_us",
        "collective_latency_us",
        "prefill_attn_dot_dtype",
        "boundary_sync_hop_factor",
        "per_op_collective_call_factor",
    }
)
ALL_SCOPES = ("per_op", "single_layer", "layer_group", "whole_step")
COMPARISON_WORKLOAD = advisor.WorkloadSpec(
    phase="decode", batch=16, context=4096, objective="both"
)


@dataclasses.dataclass(frozen=True)
class _Qwen35OracleTotals:
    """Independent Qwen3.5 HBM byte and padded FLOP totals."""

    step_weights_bytes: float
    state_act_bytes: float
    dense_flops: float
    moe_padded_flops: float


def _build_workload_from_point(point: dict[str, object]) -> advisor.WorkloadSpec:
    """Construct a typed `WorkloadSpec` from a validation point dictionary."""
    workload_raw = point["workload"]
    if type(workload_raw) is not dict:
        msg = f"Expected dict workload, got {type(workload_raw).__name__}"
        raise TypeError(msg)
    defaults = advisor.WorkloadSpec()
    phase_str = str(workload_raw.get("phase", defaults.phase))
    if phase_str not in PHASES:
        msg = f"Unsupported phase: {phase_str!r}"
        raise ValueError(msg)
    moe_str = str(workload_raw.get("moe_mode", defaults.moe_mode))
    if moe_str not in MOE_MODES:
        msg = f"Unsupported moe_mode: {moe_str!r}"
        raise ValueError(msg)
    return advisor.WorkloadSpec(
        phase=PHASES[phase_str],
        batch=int(str(workload_raw.get("batch", defaults.batch))),
        isl=int(str(workload_raw.get("isl", defaults.isl))),
        osl=int(str(workload_raw.get("osl", defaults.osl))),
        context=int(str(workload_raw.get("context", defaults.context))),
        spec_tokens=int(str(workload_raw.get("spec_tokens", defaults.spec_tokens))),
        chunk_size=int(str(workload_raw.get("chunk_size", defaults.chunk_size))),
        tp=int(str(workload_raw.get("tp", defaults.tp))),
        pp=int(str(workload_raw.get("pp", defaults.pp))),
        dp=int(str(workload_raw.get("dp", defaults.dp))),
        ep=int(str(workload_raw.get("ep", defaults.ep))),
        dcp=int(str(workload_raw.get("dcp", defaults.dcp))),
        pcp=int(str(workload_raw.get("pcp", defaults.pcp))),
        moe_mode=MOE_MODES[moe_str],
    )


def _scope_by_name(
    report: advisor.AdvisorReport, name: advisor.FusionScope
) -> advisor.ScopeEstimate:
    """Return the `ScopeEstimate` matching `name` from `report`."""
    return next(scope_est for scope_est in report.scopes if scope_est.scope == name)


def _small_dense_profile() -> advisor.ModelProfile:
    """Build a small hybrid GQA+GDN+dense-MLP profile for fast unit tests."""
    return models.finalize_model_profile(
        dataclasses.replace(
            advisor.resolve_model_profile("qwen3.5-397b"),
            name="dense-small-gdn",
            num_layers=24,
            num_attention_layers=6,
            num_linear_layers=18,
            hidden_size=1024,
            intermediate_size=3584,
            num_attention_heads=8,
            num_kv_heads=2,
            linear_key_heads=16,
            linear_value_heads=16,
            num_experts=0,
            top_k=0,
            num_moe_layers=0,
            moe_intermediate_size=0,
            shared_expert_intermediate_size=0,
            has_attn_output_gate=False,
            weight_block_k=0,
            state_dtype="fp32",
            layer_families=("gqa", "gdn", "dense_mlp"),
        )
    )


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
def test_floor_ordering_and_monotonicity_across_models(model_slug: str) -> None:
    """Verify band ordering and batch/context monotonicity across shipped presets."""
    decode_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    small_batch = 4
    large_batch = 32
    short_context = 8192
    long_context = 32768
    workload_small = advisor.WorkloadSpec(
        phase="decode",
        batch=small_batch,
        context=short_context,
        osl=1,
        tp=decode_layout.tp,
        ep=decode_layout.ep,
        dp=decode_layout.dp,
        pp=decode_layout.pp,
        dcp=decode_layout.dcp,
        pcp=decode_layout.pcp,
    )
    workload_large_batch = dataclasses.replace(workload_small, batch=large_batch)
    workload_long_context = dataclasses.replace(workload_small, context=long_context)

    report_small = advisor.advise(model_slug, workload_small, "tpu7x")
    report_large_batch = advisor.advise(model_slug, workload_large_batch, "tpu7x")
    report_long_context = advisor.advise(model_slug, workload_long_context, "tpu7x")

    for report in (report_small, report_large_batch, report_long_context):
        for scope_est in report.scopes:
            if not scope_est.feasible:
                continue
            assert (
                scope_est.step_floor_strict_us
                <= scope_est.step_bound_us
                <= scope_est.step_upper_us
            )
            assert scope_est.step_achievable_us <= scope_est.step_achievable_upper_us

    small_per_op = _scope_by_name(report_small, "per_op")
    large_per_op = _scope_by_name(report_large_batch, "per_op")
    long_per_op = _scope_by_name(report_long_context, "per_op")
    assert small_per_op.step_floor_strict_us < large_per_op.step_floor_strict_us
    assert small_per_op.step_floor_strict_us <= long_per_op.step_floor_strict_us


@pytest.mark.parametrize("model_slug", sorted(advisor.MODEL_PRESETS))
@pytest.mark.parametrize("phase", ["decode", "prefill", "verify", "mixed"])
def test_all_scopes_satisfy_band_ordering_and_empirical_bracket(
    model_slug: str,
    phase: advisor.Phase,
) -> None:
    """Verify full 5-band ordering across all shipped presets, phases, and scopes."""
    profile = advisor.resolve_model_profile(model_slug)
    decode_layout = advisor.DEPLOYMENT_PRESETS[model_slug].decode
    spec_tokens = (
        (decode_layout.spec_tokens or 3) if phase in ("verify", "mixed") else 0
    )
    chunk_size = 1024 if phase in ("prefill", "mixed") else 0
    pcp = 2 if phase in ("prefill", "mixed") else 1
    workload = advisor.WorkloadSpec(
        phase=phase,
        batch=1 if phase == "prefill" else 16,
        isl=4096,
        chunk_size=chunk_size,
        spec_tokens=spec_tokens,
        tp=decode_layout.tp,
        ep=decode_layout.ep,
        dp=decode_layout.dp,
        pp=decode_layout.pp,
        dcp=decode_layout.dcp,
        pcp=pcp,
        layer_groups=(profile.num_layers,),
    )
    report = advisor.advise(profile, workload, "tpu7x")
    assert report.ranking_metric == "step_bound_us"
    assert tuple(scope_est.scope for scope_est in report.scopes) == ALL_SCOPES
    assert all(scope_est.feasible for scope_est in report.scopes)
    for scope_est in report.scopes:
        assert (
            scope_est.step_floor_strict_us
            <= scope_est.step_bound_us
            <= scope_est.step_achievable_us
            <= scope_est.step_achievable_upper_us
        )
        assert (
            scope_est.step_bound_us
            <= scope_est.step_upper_us
            <= scope_est.step_achievable_upper_us
        )
        assert scope_est.binding_unit in VALID_BINDING_UNITS


def test_measured_corpus_empirical_bracket() -> None:
    """Verify every public corpus point satisfies active rows and 5-point bracket."""
    document = json.loads(MEASURED_CORPUS_PATH.read_text(encoding="utf-8"))
    public_points = [
        point
        for point in document["points"]
        if point.get("provenance_kind") == "public_source"
    ]
    assert public_points
    for point in public_points:
        report = advisor.advise(
            str(point["model"]),
            _build_workload_from_point(point),
            str(point["chip"]),
        )
        assert report.ranking_metric == "step_bound_us"
        assert report.active_tokens == int(str(point["rows"]))
        target_scope = _scope_by_name(report, point["scope"])
        measured_step_us = float(str(point["step_time_us"]))
        assert (
            target_scope.step_floor_strict_us
            <= target_scope.step_bound_us
            <= target_scope.step_achievable_us
            <= measured_step_us
            <= target_scope.step_achievable_upper_us
        )
        assert (
            target_scope.step_bound_us
            <= target_scope.step_upper_us
            <= target_scope.step_achievable_upper_us
        )


def test_parallelism_objective_split_and_workload_fidelity() -> None:
    """Verify objective splits, empty layout verdict, and base_workload preservation."""
    split_result = advisor.compare_parallelism_candidates(
        "kimi-k3",
        [
            advisor.ParallelLayout(tp=1, ep=32, dp=1, dcp=32),
            advisor.ParallelLayout(tp=1, ep=32, dp=2, dcp=16),
        ],
        chip="tpu7x",
        base_workload=advisor.WorkloadSpec(batch=16, context=8192, objective="both"),
    )
    assert str(split_result["verdict"]).startswith("tie(objective_split:")

    empty_res = advisor.compare_parallelism_candidates("qwen3.5-397b", ())
    assert empty_res["verdict"] == "no_layouts_provided"
    assert empty_res["candidates"] == []

    custom_vmem_mib = 48.0
    custom_gap_us = 5.0
    spec_tokens = 3
    meas = specs.load_measurements(
        {
            "schema": constants.MEASUREMENTS_SCHEMA,
            "source": {"kind": "xprof_xplane", "desc": "verify", "file": "v.pb"},
            "launch_gap_us": {"p50": 4.2, "p90": 6.4, "n": 20},
        },
        chip_name="tpu7x",
    )
    layout_a = advisor.ParallelLayout(tp=8, ep=8, dcp=4)
    layout_b = advisor.ParallelLayout(tp=16, ep=16, dcp=4)
    base_wl = advisor.WorkloadSpec(
        phase="verify",
        batch=8,
        context=4096,
        spec_tokens=spec_tokens,
        vmem_budget_mib=custom_vmem_mib,
        op_overhead_us=custom_gap_us,
        objective="both",
    )
    comp = advisor.compare_parallelism_candidates(
        "qwen3.5-397b",
        (layout_a, layout_b),
        chip="tpu7x",
        base_workload=base_wl,
        measurements=meas,
    )
    rows = comp["candidates"]
    assert isinstance(rows, list)
    for row, layout in zip(rows, (layout_a, layout_b), strict=True):
        direct_wl = dataclasses.replace(
            base_wl,
            batch=base_wl.batch * layout.dp,
            isl=base_wl.context,
            osl=0,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
            layer_groups=layout.layer_groups,
            objective="step",
        )
        direct_report = advisor.advise(
            "qwen3.5-397b", direct_wl, "tpu7x", measurements=meas
        )
        expected_step = float(direct_report.metrics["step_floor_us"]["value"])
        expected_tput = float(
            direct_report.metrics["throughput_tok_s_per_chip"]["value"]
        )
        assert row["step_floor_us"] == pytest.approx(expected_step, abs=0.01)
        assert row["throughput_tok_s_per_chip"] == pytest.approx(
            expected_tput, abs=0.01
        )


@pytest.mark.parametrize("source_mode", ["bundled", "inline_json"])
def test_public_calibration_loader_and_validation(
    tmp_path: Path, source_mode: str
) -> None:
    """Verify bundled prior loader semantics, confidence, and corrupt-JSON errors."""
    calibration_path = specs.public_calibration_path("tpu7x")
    assert calibration_path is not None
    assert calibration_path.is_file()
    raw_json = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert raw_json["schema"] == constants.MEASUREMENTS_SCHEMA
    assert "launch_gap_us" in raw_json
    assert "step_time_us" not in raw_json
    for entry in raw_json.get("collectives", []):
        assert isinstance(entry, dict)
        assert "step_time_us" not in entry

    meas_arg: str | dict[str, object] = (
        "bundled" if source_mode == "bundled" else raw_json
    )
    meas_data = specs.load_measurements(meas_arg, chip_name="tpu7x")
    assert meas_data.source_kind == "architectural_prior"
    assert meas_data.step_time_us is None

    calibrated_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=meas_arg,
    )
    assert calibrated_report.confidence == "prior_only"
    assert calibrated_report.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert calibrated_report.metrics["collective_latency_us"]["tag"] == "ASSUMED"

    bad_calibration_file = tmp_path / "tpu7x_public.json"
    bad_calibration_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(errors.MegakernelConfigError, match="JSON object"):
        specs.load_measurements(bad_calibration_file, chip_name="tpu7x")


def test_measured_corpus_schema() -> None:
    """Verify measured_points.json exists and enforces advisor.measured_points.v2."""
    assert MEASURED_CORPUS_PATH.is_file()
    document = json.loads(MEASURED_CORPUS_PATH.read_text(encoding="utf-8"))
    assert document["schema"] == "advisor.measured_points.v2"
    points = document["points"]
    assert type(points) is list
    for point in points:
        assert REQUIRED_MEASURED_KEYS.issubset(point.keys())
        assert bool(str(point["id"]).strip())
        if point["provenance_kind"] == "public_source":
            assert REQUIRED_PUBLIC_SOURCE_KEYS.issubset(point.keys())
            assert str(point["source_url"]).startswith("https://")
            assert isinstance(point["source_quote"], str)
            assert point["source_quote"].strip()
            assert isinstance(point["rows_quote"], str)
            assert point["rows_quote"].strip()
            assert type(point["approximate"]) is bool
            assumptions = point["assumptions"]
            assert type(assumptions) is list
            assert assumptions
            assert all(isinstance(item, str) and item.strip() for item in assumptions)
        else:
            assert bool(str(point.get("job_id", "")).strip())
        assert int(str(point["n"])) >= 1
        assert float(str(point["step_time_us"])) > 0.0


def _assert_prefill_pcp_chips(
    chip: str,
    reference: advisor.ParallelLayout,
    layout: advisor.ParallelLayout,
) -> None:
    """Verify PCP ranks multiply physical chip count in prefill."""
    prefill_batch = 1
    prefill_isl = 1024
    expected_prefill_chips = 8
    prefill_wl = advisor.WorkloadSpec(
        phase="prefill", batch=prefill_batch, isl=prefill_isl, objective="both"
    )
    prefill_res = advisor.compare_parallelism_candidates(
        "qwen3.5-397b", (reference, layout), chip=chip, base_workload=prefill_wl
    )
    prefill_rows = prefill_res["candidates"]
    assert isinstance(prefill_rows, list)
    prefill_row = prefill_rows[1]
    assert prefill_row["chips"] == expected_prefill_chips
    expected_prefill_tput = (
        prefill_wl.batch * prefill_isl * layout.dp / expected_prefill_chips
    ) / (prefill_row["step_floor_us"] * 1e-6)
    assert prefill_row["throughput_tok_s_per_chip"] == pytest.approx(
        expected_prefill_tput, rel=1e-3
    )


@pytest.mark.parametrize(
    ("chip", "layout", "expected_chips"),
    [
        ("tpu7x", advisor.ParallelLayout(tp=8, ep=8, dcp=4), 4),
        ("tpu7x", advisor.ParallelLayout(tp=2, ep=8, dp=4), 4),
        ("tpu7x", advisor.ParallelLayout(tp=8, ep=8, dp=2, dcp=4), 8),
        ("tpu7x", advisor.ParallelLayout(tp=16, ep=16), 8),
        ("tpu7x", advisor.ParallelLayout(tp=1, ep=32, dcp=32), 16),
        ("tpu7x", advisor.ParallelLayout(tp=1, ep=1, pp=8), 4),
        ("tpu7x", advisor.ParallelLayout(tp=8, ep=8, pcp=2), 4),
        ("v6e", advisor.ParallelLayout(tp=8, ep=8), 8),
    ],
)
def test_layout_comparison_rows_print_chips(
    chip: str, layout: advisor.ParallelLayout, expected_chips: int
) -> None:
    """Verify layout rows carry physical chip count and exact chip throughput."""
    reference = advisor.ParallelLayout(tp=8, ep=8)
    result = advisor.compare_parallelism_candidates(
        "qwen3.5-397b",
        (reference, layout),
        chip=chip,
        base_workload=COMPARISON_WORKLOAD,
    )
    rows = result["candidates"]
    assert isinstance(rows, list)
    assert [row["layout"] for row in rows] == [reference.label(), layout.label()]
    cand_row = rows[1]
    assert cand_row["chips"] == expected_chips
    assert all(isinstance(row["chips"], int) and row["chips"] >= 1 for row in rows)
    if cand_row["hbm_fits"] and cand_row["valid"]:
        total_batch = COMPARISON_WORKLOAD.batch * layout.dp
        expected_tput = (total_batch / expected_chips) / (
            cand_row["step_floor_us"] * 1e-6
        )
        assert cand_row["throughput_tok_s_per_chip"] == pytest.approx(
            expected_tput, rel=1e-3
        )
    else:
        assert cand_row["throughput_tok_s_per_chip"] == 0.0
    if layout.pcp > 1:
        _assert_prefill_pcp_chips(chip, reference, layout)


@pytest.mark.parametrize("objective", ["step", "both", "throughput"])
def test_step_ranking_rejects_unequal_chip_layouts(
    objective: advisor.ObjectiveMode,
) -> None:
    """Step time is never ranked across layouts that use different chip counts."""
    base = dataclasses.replace(COMPARISON_WORKLOAD, objective=objective)
    four_chips = advisor.ParallelLayout(tp=8, ep=8)
    eight_chips = advisor.ParallelLayout(tp=16, ep=16)
    unequal = advisor.compare_parallelism_candidates(
        "qwen3.5-397b", (four_chips, eight_chips), chip="tpu7x", base_workload=base
    )
    rows = unequal["candidates"]
    assert isinstance(rows, list)
    assert all(bool(row["hbm_fits"]) and bool(row["valid"]) for row in rows)
    assert unequal["best_step_layout"] == ""
    best_per_chip = max(rows, key=lambda row: row["throughput_tok_s_per_chip"])
    assert unequal["best_throughput_layout"] == best_per_chip["layout"]
    if objective == "throughput":
        assert unequal["verdict"] == unequal["best_throughput_layout"]
    else:
        assert unequal["verdict"] == "not comparable"
    expected_pair = [4, 8]
    assert [row["chips"] for row in rows] == expected_pair

    equal = advisor.compare_parallelism_candidates(
        "qwen3.5-397b",
        (
            advisor.ParallelLayout(tp=8, ep=8, dcp=4),
            advisor.ParallelLayout(tp=2, ep=8, dp=4),
        ),
        chip="tpu7x",
        base_workload=base,
    )
    equal_rows = equal["candidates"]
    assert isinstance(equal_rows, list)
    assert all(bool(row["hbm_fits"]) and bool(row["valid"]) for row in equal_rows)
    fastest = min(equal_rows, key=lambda row: row["step_floor_us"])
    assert equal["best_step_layout"] == fastest["layout"]
    assert equal["verdict"] != "not comparable"


def test_asymptotic_decode_intensity_and_vmem_fit_headroom() -> None:
    """Verify asymptotic intensity, VMEM fit headroom, critical weights, and ridge."""
    small_state = _small_dense_profile()
    r_b16 = advisor.advise(
        small_state,
        advisor.WorkloadSpec(phase="verify", batch=16, spec_tokens=3, tp=8),
        "tpu7x",
    )
    r_b128 = advisor.advise(
        small_state,
        advisor.WorkloadSpec(phase="verify", batch=128, spec_tokens=3, tp=8),
        "tpu7x",
    )
    assert r_b16.asymptotic_decode_intensity is not None
    assert r_b16.asymptotic_decode_intensity == r_b128.asymptotic_decode_intensity
    assert r_b16.asymptotic_decode_intensity < r_b16.ridge_dense_flop_per_byte
    assert r_b16.step_arithmetic_intensity < r_b16.asymptotic_decode_intensity

    assert r_b16.ridge_tokens == r_b128.ridge_tokens

    # Weight-only ridge_tokens tracks w_step without inflating with KV/state bytes.
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    q_b16 = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, context=4096, tp=8, ep=8),
        "tpu7x",
    )
    q_b16_long = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, context=32768, tp=8, ep=8),
        "tpu7x",
    )
    q_b128 = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=128, context=4096, tp=8, ep=8),
        "tpu7x",
    )
    assert q_b16_long.ridge_tokens <= q_b16.ridge_tokens
    b16_weight_gb = _scope_by_name(q_b16, "per_op").step_weight_gb
    b128_weight_gb = _scope_by_name(q_b128, "per_op").step_weight_gb
    routed_weight_ratio = b128_weight_gb / b16_weight_gb
    assert q_b128.ridge_tokens <= round(q_b16.ridge_tokens * routed_weight_ratio) + 1

    r_alias = advisor.advise(
        small_state,
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8),
        "tpu7x",
    )
    r_no_alias = advisor.advise(
        small_state,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
        ),
        "tpu7x",
    )
    r_no_alias_disp = advisor.advise(
        small_state,
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=2048,
            tp=8,
            vmem_alias=False,
            vmem_fit_headroom=True,
            op_overhead_us=5.0,
        ),
        "tpu7x",
    )
    expected_vmem_mib = float(
        constants.DEFAULT_USABLE_VMEM_MIB - constants.CORE_LIMITS.fit_vmem_headroom_mib
    )
    assert r_alias.vmem_budget_mib == expected_vmem_mib
    assert r_no_alias.vmem_budget_mib == advisor.fit_vmem_budget_bytes() / (1 << 20)
    whole_alias = _scope_by_name(r_alias, "whole_step")
    whole_no_alias = _scope_by_name(r_no_alias, "whole_step")
    whole_dispatched = _scope_by_name(r_no_alias_disp, "whole_step")
    assert whole_alias.num_vmem_splits == 1
    assert whole_no_alias.num_vmem_splits > 1
    assert whole_no_alias.critical_weight_gb is not None
    assert whole_dispatched.critical_weight_gb is not None
    assert whole_dispatched.critical_weight_gb > whole_no_alias.critical_weight_gb
    assert whole_dispatched.step_bound_us > whole_no_alias.step_bound_us


def _check_qwen_core_and_dense_scopes(
    qwen_bf16: advisor.ModelProfile,
    qwen_fp32: advisor.ModelProfile,
    small_dense: advisor.ModelProfile,
) -> None:
    """Assert core-gated vs physics-winner scopes across Qwen3.5 and small dense."""
    # Explicit state_dtype=fp32 stays outside_core_limits while whole_step remains
    # the physics winner; default state_dtype=bf16 satisfies core limits.
    r_fp32 = advisor.advise(
        qwen_fp32, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8), "tpu7x"
    )
    assert r_fp32.fastest_scope == "whole_step"
    assert r_fp32.recommended_scope == "whole_step"
    assert r_fp32.core_support == "outside_core_limits"
    assert r_fp32.core_ready is False
    assert r_fp32.core_ready_scope == "per_op"
    assert "kv_or_state_dtype_not_in_core" in r_fp32.core_blockers
    assert r_fp32.build_megakernel is True
    assert "hypothetical_best" not in r_fp32.to_dict()
    whole_fp32 = _scope_by_name(r_fp32, "whole_step")
    assert whole_fp32.support == "outside_core_limits"
    assert "kv_or_state_dtype_not_in_core" in whole_fp32.blockers

    r_dec = advisor.advise(
        qwen_bf16, advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8), "tpu7x"
    )
    assert r_dec.recommended_scope == "whole_step"
    assert r_dec.core_support == "within_core_limits"
    assert r_dec.core_ready is True
    assert r_dec.core_blockers == ()
    assert r_dec.core_ready_scope == "whole_step"
    assert r_dec.recommended_moe_mode == "masked_on_chip"
    assert r_dec.build_megakernel is True
    assert _scope_by_name(r_dec, "whole_step").support == "within_core_limits"

    r_b32 = advisor.advise(
        qwen_bf16, advisor.WorkloadSpec(phase="decode", batch=32, tp=8, ep=8), "tpu7x"
    )
    assert r_b32.recommended_scope == "whole_step"
    assert r_b32.core_ready is True
    assert r_b32.recommended_moe_mode == "grouped_on_chip"
    r_08_pre = advisor.advise(
        small_dense,
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=2048, tp=8),
        "tpu7x",
    )
    assert r_08_pre.fastest_scope == "whole_step"
    assert r_08_pre.recommended_scope == "whole_step"
    assert r_08_pre.core_support == "outside_core_limits"
    assert r_08_pre.core_ready_scope == "per_op"
    assert r_08_pre.recommended_moe_mode == "dense_mlp"


def _check_cliff_single_and_group_scopes(
    qwen_bf16: advisor.ModelProfile, large_dense: advisor.ModelProfile
) -> None:
    """Assert VMEM-cliff per_op, single_layer, and layer_group scopes are reachable."""
    # Derive VMEM-cliff batches dynamically from max_vmem_tokens + 1.
    cliff_base = advisor.WorkloadSpec(
        phase="decode",
        batch=16,
        context=1024,
        tp=8,
        moe_mode="masked_on_chip",
        moe_mode_explicit=True,
    )
    cliff_max_tok = _scope_by_name(
        advisor.advise(qwen_bf16, cliff_base, "tpu7x"), "whole_step"
    ).max_vmem_tokens
    r_cliff_7x = advisor.advise(
        qwen_bf16, dataclasses.replace(cliff_base, batch=cliff_max_tok + 1), "tpu7x"
    )
    assert r_cliff_7x.fastest_scope == "per_op"
    assert r_cliff_7x.recommended_scope == "per_op"
    assert _scope_by_name(r_cliff_7x, "whole_step").num_vmem_splits > 1

    sat_base = advisor.WorkloadSpec(
        phase="decode", batch=16, tp=2, dp=4, ep=8, tiles=Tiles()
    )
    sat_max_tok = _scope_by_name(
        advisor.advise(qwen_bf16, sat_base, "tpu7x"), "whole_step"
    ).max_vmem_tokens
    r_ep8_sat = advisor.advise(
        qwen_bf16,
        dataclasses.replace(sat_base, batch=(sat_max_tok + 1) * sat_base.dp),
        "tpu7x",
    )
    assert r_ep8_sat.fastest_scope == "single_layer"
    assert r_ep8_sat.recommended_scope == "single_layer"
    assert r_ep8_sat.core_support == "outside_core_limits"
    assert r_ep8_sat.core_ready is False
    assert r_ep8_sat.core_ready_scope == "per_op"
    assert r_ep8_sat.recommended_moe_mode == "grouped_on_chip"

    r_llama_pre = advisor.advise(
        large_dense,
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=8),
        "tpu7x",
    )
    assert r_llama_pre.recommended_scope == "per_op"
    assert r_llama_pre.core_support == "baseline"
    assert r_llama_pre.core_ready is False
    assert r_llama_pre.recommended_moe_mode == "dense_mlp"
    assert r_llama_pre.build_megakernel is False

    r_stg = advisor.advise(
        qwen_bf16,
        advisor.WorkloadSpec(
            phase="prefill", batch=1, isl=4096, chunk_size=64, tp=1, pp=8
        ),
        "tpu7x",
    )
    assert r_stg.fastest_scope == "layer_group"
    assert r_stg.recommended_scope == "layer_group"
    assert r_stg.core_support == "outside_core_limits"
    r_seg = advisor.advise(
        qwen_bf16,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, segment_layers=10),
        "tpu7x",
    )
    assert r_seg.fastest_scope == "layer_group"
    assert r_seg.recommended_scope == "layer_group"
    expected_ops_per_step = (qwen_bf16.num_layers // 10) * 3
    assert _scope_by_name(r_seg, "layer_group").ops_per_step == expected_ops_per_step
    r_16k = advisor.advise(
        qwen_bf16,
        advisor.WorkloadSpec(phase="prefill", batch=8, isl=16384, tp=1, pp=8),
        "tpu7x",
    )
    assert r_16k.recommended_scope == "per_op"
    assert r_16k.build_megakernel is False


def test_all_four_fusion_scopes_reachable() -> None:
    """Prove `whole_step`, `single_layer`, `layer_group`, and `per_op` each win."""
    qwen_bf16 = advisor.resolve_model_profile("qwen3.5-397b")
    qwen_fp32 = dataclasses.replace(qwen_bf16, state_dtype="fp32")
    small_dense = _small_dense_profile()
    large_dense = models.finalize_model_profile(
        advisor.ModelProfile(
            name="dense-large-gqa",
            num_layers=126,
            hidden_size=16384,
            intermediate_size=53248,
            num_attention_heads=128,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="fp8",
            expert_dtype="fp8",
        )
    )
    _check_qwen_core_and_dense_scopes(qwen_bf16, qwen_fp32, small_dense)
    _check_cliff_single_and_group_scopes(qwen_bf16, large_dense)


def test_gain_split_sums_to_total_and_fact_part_ignores_priors() -> None:
    """Verify gain_fact_us invariance, per_op zero band, and signed boundary_cost."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, dcp=4, ep=8)
    report = advisor.advise(model_profile, wl, "tpu7x")
    gain_fact_us = float(report.metrics["gain_fact_us"]["value"])
    gain_assumed_us = float(report.metrics["gain_assumed_us"]["value"])
    total_gain = float(report.metrics["compiler_proof_gain_us"]["value"])
    assert gain_fact_us > 0.0
    assert gain_assumed_us > 0.0
    assert total_gain == pytest.approx(gain_fact_us + gain_assumed_us, abs=0.01)
    assert report.metrics["compiler_proof_gain_us"]["tag"] == "ASSUMED"

    # Perturbing collective latency and launch gap must leave gain_fact_us identical.
    wl_no_dcp = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    zero_coll_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"), collective_latency_s=0.0
    )
    zero_coll = advisor.advise(model_profile, wl_no_dcp, zero_coll_chip)
    perturbed_meas = {
        "schema": constants.MEASUREMENTS_SCHEMA,
        "source": {"kind": "xprof_xplane", "desc": "perturbed", "file": "p.pb"},
        "launch_gap_us": {"p50": 15.0, "p90": 25.0, "n": 10},
    }
    perturbed = advisor.advise(
        model_profile, wl_no_dcp, "tpu7x", measurements=perturbed_meas
    )
    assert float(perturbed.metrics["gain_fact_us"]["value"]) == float(
        zero_coll.metrics["gain_fact_us"]["value"]
    )
    assert float(perturbed.metrics["gain_assumed_us"]["value"]) > float(
        zero_coll.metrics["gain_assumed_us"]["value"]
    )
    assert float(perturbed.metrics["boundary_cost_us"]["value"]) > float(
        zero_coll.metrics["boundary_cost_us"]["value"]
    )

    # When per_op wins, fusion gain bands must collapse strictly to [0.0, 0.0].
    large_dense = models.finalize_model_profile(
        advisor.ModelProfile(
            name="dense-large-gqa",
            num_layers=126,
            hidden_size=16384,
            intermediate_size=53248,
            num_attention_heads=128,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="fp8",
            expert_dtype="fp8",
        )
    )
    per_op_win = advisor.advise(
        large_dense,
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=8),
        "tpu7x",
    )
    assert per_op_win.fastest_scope == "per_op"
    assert per_op_win.recommended_scope == "per_op"
    for metric_key in (
        "boundary_cost_us",
        "gain_assumed_us",
        "compiler_proof_gain_us",
        "fusion_gain_max",
    ):
        assert per_op_win.metrics[metric_key]["value"] == 0.0
        assert per_op_win.metrics[metric_key]["band"] == [0.0, 0.0]

    # When fused scope has higher roof_floor_us yet wins via host_head_per_op_us,
    # boundary_cost_us preserves the signed negative physics delta.
    signed_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=256,
        tp=8,
        ep=8,
        vmem_budget_mib=20,
        dispatch_regime="zero_floor",
        host_head_per_op_us=5000.0,
        host_head_fused_us=0.0,
    )
    signed_report = advisor.advise(model_profile, signed_wl, "tpu7x")
    per_op_est = _scope_by_name(signed_report, "per_op")
    fastest_est = _scope_by_name(signed_report, signed_report.fastest_scope)
    assert fastest_est.roof_floor_us > per_op_est.roof_floor_us
    assert fastest_est.step_bound_us < per_op_est.step_bound_us
    assert float(signed_report.metrics["boundary_cost_us"]["value"]) == pytest.approx(
        per_op_est.step_bound_us - fastest_est.step_bound_us, abs=0.05
    )


@pytest.mark.parametrize(
    "bad_layout",
    [
        advisor.ParallelLayout(tp=64, ep=64),
        advisor.ParallelLayout(tp=16, ep=16, pp=3),
        advisor.ParallelLayout(tp=4, ep=4),
    ],
)
def test_invalid_sharding_returns_invalid_config(
    bad_layout: advisor.ParallelLayout,
) -> None:
    """Verify HBM-fitting invalid meshes pin fastest_scope and search to per_op/None."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    wl = advisor.WorkloadSpec(
        phase="decode",
        batch=16,
        tp=bad_layout.tp,
        ep=bad_layout.ep,
        dp=bad_layout.dp,
        pp=bad_layout.pp,
        objective="both",
    )
    report = advisor.advise(model_profile, wl, "tpu7x")
    assert report.hbm_fits is True
    assert report.verdict.startswith("invalid_config(")
    assert report.fastest_verdict == report.verdict
    assert report.fastest_scope == "per_op"
    assert report.recommended_scope == "per_op"
    assert report.build_megakernel is False

    search_res = advisor.search_layer_groups(model_profile, wl, "tpu7x")
    assert search_res.best is None
    assert all(not cand.feasible for cand in search_res.candidates)

    with pytest.raises(errors.MegakernelConfigError, match="dp_not_power_of_two"):
        advisor.advise(
            model_profile,
            advisor.WorkloadSpec(phase="decode", batch=24, tp=8, ep=8, dp=3),
            "tpu7x",
        )


def test_recommended_scope_metrics_and_hbm_oom_zero_tps() -> None:
    """Verify summary metrics follow recommended scope and zero out on HBM OOM."""
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    invalid_report = advisor.advise(
        qwen_profile, advisor.WorkloadSpec(phase="decode", batch=16, tp=2), "tpu7x"
    )
    assert invalid_report.verdict == "invalid_config(slice_topology_unsupported)"
    assert invalid_report.fastest_scope == "per_op"
    assert invalid_report.recommended_scope == "per_op"
    assert invalid_report.core_support == "baseline"
    assert invalid_report.core_ready is False
    assert invalid_report.core_blockers == ()
    assert invalid_report.core_ready_scope == "per_op"
    per_op_bad = _scope_by_name(invalid_report, "per_op")
    assert float(invalid_report.metrics["step_floor_us"]["value"]) == pytest.approx(
        per_op_bad.step_bound_us, abs=0.01
    )
    assert float(invalid_report.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0
    comparison_result = advisor.compare_parallelism_candidates(
        "qwen3.5-397b",
        (advisor.ParallelLayout(tp=2, ep=1), advisor.ParallelLayout(tp=8, ep=8)),
        chip="tpu7x",
    )
    assert comparison_result["best_step_layout"] == "TP8 x EP8"
    assert comparison_result["best_throughput_layout"] == "TP8 x EP8"
    assert comparison_result["verdict"] == "TP8 x EP8"

    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    rep_k3 = advisor.advise(
        kimi_profile,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=32, ep=32),
        "tpu7x",
    )
    assert rep_k3.fastest_scope == "whole_step"
    assert rep_k3.recommended_scope == "whole_step"
    assert rep_k3.core_support == "outside_core_limits"
    assert rep_k3.core_ready is False
    assert rep_k3.core_ready_scope == "per_op"
    assert len(rep_k3.core_blockers) >= 1
    whole_k3 = _scope_by_name(rep_k3, "whole_step")
    assert float(rep_k3.metrics["step_floor_us"]["value"]) == pytest.approx(
        whole_k3.step_bound_us, abs=0.01
    )
    assert float(rep_k3.metrics["throughput_tok_s_per_chip"]["value"]) > 0.0

    rep_oom = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=4, tp=2, ep=2, dp=2),
        "tpu7x",
    )
    assert rep_oom.hbm_fits is False
    assert rep_oom.hbm_pool_overflow is True
    assert rep_oom.recommended_scope == "per_op"
    assert rep_oom.core_support == "baseline"
    assert rep_oom.core_ready is False
    assert rep_oom.core_blockers == ()
    assert rep_oom.core_ready_scope == "per_op"
    assert float(rep_oom.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0


def test_hbm_fits_physical_vs_pool_overflow_cap() -> None:
    """Verify hbm_fits (1.00 OOM), pool overflow (0.92), and custom cap exposure."""
    profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8)
    chip = advisor.resolve_chip_spec("tpu7x")
    base_report = advisor.advise(profile, workload, chip)
    used_bytes = int(base_report.hbm_used_gb * 1e9)
    assert base_report.hbm_fits is True
    assert base_report.hbm_pool_overflow is False
    assert "hbm_pool_overflow" not in base_report.flags
    hbm_cap = constants.HBM_UTILIZATION_CAP
    assert hbm_cap == pytest.approx(0.92)
    assert workload.hbm_utilization_cap == pytest.approx(0.92)

    tight_cap_bytes = int(used_bytes / 0.95)
    tight_chip = dataclasses.replace(chip, hbm_capacity_bytes=tight_cap_bytes)
    overflow_report = advisor.advise(profile, workload, tight_chip)
    assert overflow_report.hbm_fits is True
    assert overflow_report.hbm_pool_overflow is True
    assert "hbm_pool_overflow" in overflow_report.flags
    assert float(overflow_report.metrics["throughput_tok_s_per_chip"]["value"]) > 0.0

    oom_cap_bytes = int(used_bytes * 0.95)
    oom_chip = dataclasses.replace(chip, hbm_capacity_bytes=oom_cap_bytes)
    oom_report = advisor.advise(profile, workload, oom_chip)
    assert oom_report.hbm_fits is False
    assert oom_report.hbm_pool_overflow is True
    assert "hbm_pool_overflow" in oom_report.flags
    assert oom_report.recommended_scope == "per_op"
    assert float(oom_report.metrics["throughput_tok_s_per_chip"]["value"]) == 0.0

    custom_cap = 0.45
    custom_report = advisor.advise(
        profile, dataclasses.replace(workload, hbm_utilization_cap=custom_cap), chip
    )
    assert custom_report.hbm_utilization_cap == pytest.approx(custom_cap)


def _qwen35_oracle_bytes_and_flops(
    model_profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
    chip_spec: advisor.ChipSpec,
    *,
    num_vmem_splits: int = 1,
) -> _Qwen35OracleTotals:
    """Compute independent single-read Qwen3.5 HBM bytes and grouped MoE FLOPs."""
    active_tokens = workload.batch
    eff_tp = workload.tp
    eff_ctx = workload.isl + workload.osl // 2
    nh_local = math.ceil(model_profile.num_attention_heads / eff_tp)
    kvh_local = math.ceil(model_profile.num_kv_heads / eff_tp)
    q_dim = nh_local * model_profile.head_dim
    attn_params = (
        model_profile.hidden_size * 2 * (q_dim + kvh_local * model_profile.head_dim)
        + model_profile.hidden_size * q_dim
    )
    vh_local = math.ceil(model_profile.linear_value_heads / eff_tp)
    kh_local = math.ceil(model_profile.linear_key_heads / eff_tp)
    z_dim = vh_local * model_profile.linear_value_dim
    qkv_lin = 2 * kh_local * model_profile.linear_key_dim + z_dim
    lin_params = model_profile.hidden_size * (qkv_lin + 2 * z_dim)
    shared_inter_local = math.ceil(
        model_profile.shared_expert_intermediate_size / eff_tp
    )
    shared_params = 3 * model_profile.hidden_size * shared_inter_local
    dense_params = (
        model_profile.num_attention_layers * attn_params
        + model_profile.num_linear_layers * lin_params
        + model_profile.num_moe_layers * shared_params
    )
    router_params = (
        model_profile.num_moe_layers
        * model_profile.hidden_size
        * model_profile.num_experts
    )
    weight_bytes_per_elem = constants.DTYPE_BYTES[model_profile.weight_dtype.lower()]
    expert_bytes_per_elem = constants.DTYPE_BYTES[
        (model_profile.expert_dtype or model_profile.weight_dtype).lower()
    ]
    kv_bytes_per_elem = constants.DTYPE_BYTES[model_profile.kv_dtype.lower()]
    state_bytes_per_elem = constants.DTYPE_BYTES[model_profile.state_dtype.lower()]
    w_static = (
        dense_params * weight_bytes_per_elem + router_params * constants.BF16_BYTES
    )

    experts_local = model_profile.num_experts // workload.ep
    ep_tp = math.gcd(workload.ep, eff_tp)
    i_exp_local = math.ceil(
        model_profile.moe_intermediate_size / max(1, eff_tp // ep_tp)
    )
    exp_params = 3 * model_profile.hidden_size * i_exp_local
    miss_prob = (1.0 - model_profile.top_k / model_profile.num_experts) ** active_tokens
    experts_hit = max(1.0, experts_local * (1.0 - miss_prob))
    w_routed = (
        model_profile.num_moe_layers * experts_hit * exp_params * expert_bytes_per_elem
    )
    step_weights_bytes = w_static + w_routed

    kv_bytes = (
        model_profile.num_attention_layers
        * active_tokens
        * eff_ctx
        * 2.0
        * kvh_local
        * model_profile.head_dim
        * kv_bytes_per_elem
    )
    lin_state_bytes = (
        2.0
        * model_profile.num_linear_layers
        * active_tokens
        * (
            z_dim * model_profile.linear_key_dim
            + qkv_lin * model_profile.linear_conv_kernel
        )
        * state_bytes_per_elem
    )
    act_bytes = (
        2.0
        * max(1, model_profile.residual_streams)
        * model_profile.num_layers
        * active_tokens
        * model_profile.hidden_size
        * constants.BF16_BYTES
    )
    tile_tokens = math.ceil(active_tokens / max(1, num_vmem_splits))
    tile_miss = (1.0 - model_profile.top_k / model_profile.num_experts) ** tile_tokens
    experts_hit_tile = max(1.0, experts_local * (1.0 - tile_miss))
    pairs_tile = (
        tile_tokens * model_profile.top_k * experts_local / model_profile.num_experts
    )
    avg_m_tile = pairs_tile / experts_hit_tile
    row_tile = chip_spec.matmul_row_tile(
        model_profile.expert_dtype or model_profile.weight_dtype
    )
    pad_factor = math.ceil(avg_m_tile / row_tile) * row_tile / avg_m_tile
    moe_padded_flops = (
        2.0
        * model_profile.num_moe_layers
        * active_tokens
        * model_profile.top_k
        * exp_params
        * (experts_local / model_profile.num_experts)
        * pad_factor
    )
    flops_kv = (
        2.0 * model_profile.num_attention_layers * eff_ctx * 2.0 * active_tokens * q_dim
    )
    flops_dense = 2.0 * active_tokens * (dense_params + router_params) + flops_kv
    return _Qwen35OracleTotals(
        step_weights_bytes=step_weights_bytes,
        state_act_bytes=act_bytes + kv_bytes + lin_state_bytes,
        dense_flops=flops_dense,
        moe_padded_flops=moe_padded_flops,
    )


def test_fastest_tiled_scope_arithmetic_intensity_streams_weights_once() -> None:
    """Verify fastest tiled scope arithmetic intensity streams weights once."""
    model_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=128,
        tp=8,
        ep=8,
        vmem_budget_mib=20,
        dispatch_regime="unfused_eager",
    )
    chip_tpu7x = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(model_profile, workload, chip_tpu7x)
    assert report.fastest_scope == "whole_step"
    best = _scope_by_name(report, report.fastest_scope)
    assert best.num_vmem_splits > 1
    oracle = _qwen35_oracle_bytes_and_flops(
        model_profile, workload, chip_tpu7x, num_vmem_splits=best.num_vmem_splits
    )
    total_flops = oracle.dense_flops + oracle.moe_padded_flops
    single_read_ai = total_flops / (oracle.step_weights_bytes + oracle.state_act_bytes)
    multi_read_ai = total_flops / (
        best.num_vmem_splits * oracle.step_weights_bytes + oracle.state_act_bytes
    )
    ai_round_tol = 0.02
    assert report.step_arithmetic_intensity == pytest.approx(
        round(single_read_ai, 2), abs=ai_round_tol
    )
    assert report.step_arithmetic_intensity > 1.5 * multi_read_ai


def test_layer_group_search_returns_lowest_bound_feasible_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify search_layer_groups ranks partitions and skips saturation overhead."""
    prefill_role = advisor.DEPLOYMENT_PRESETS["kimi-k3"].prefill
    prefill_window_tokens = 512
    workload = advisor.WorkloadSpec(
        phase="prefill",
        batch=1,
        isl=prefill_window_tokens,
        osl=0,
        context=prefill_window_tokens,
        tp=prefill_role.tp,
        ep=prefill_role.ep,
        dp=prefill_role.dp,
        pp=prefill_role.pp,
        dcp=prefill_role.dcp,
        pcp=prefill_role.pcp,
        layer_groups=prefill_role.layer_groups,
        objective="both",
    )
    calls = 0

    def _counting_eval(
        profile: models.ModelProfile,
        eval_wl: specs.WorkloadSpec,
        chip_spec: advisor.ChipSpec,
        *,
        plan_error: str = "",
        measurements: specs.MeasurementsData | None = None,
    ) -> _EvaluatedScopes:
        """Count _evaluate_scopes_with_context calls in search_layer_groups."""
        nonlocal calls
        calls += 1
        return _evaluate_scopes_with_context(
            profile,
            eval_wl,
            chip_spec,
            plan_error=plan_error,
            measurements=measurements,
        )

    monkeypatch.setattr(verdict, "_evaluate_scopes_with_context", _counting_eval)
    search_res = advisor.search_layer_groups("kimi-k3", workload, "tpu7x")
    assert calls <= len(search_res.candidates) + 1

    # On decode with saturation preset + objective="both", skip saturation compare.
    calls = 0
    dec_both = advisor.WorkloadSpec(
        phase="decode", batch=16, tp=8, ep=8, objective="both"
    )
    q35_search = advisor.search_layer_groups("qwen3.5-397b", dec_both, "tpu7x")
    assert calls <= len(q35_search.candidates) + 1
    assert search_res.best is not None
    assert search_res.best.num_groups == 1
    assert search_res.listed_step_bound_us is not None
    kimi_num_layers = advisor.resolve_model_profile("kimi-k3").num_layers
    per_layer = next(
        cand.step_bound_us
        for cand in search_res.candidates
        if cand.num_groups == kimi_num_layers
    )
    assert (
        search_res.best.step_bound_us
        <= search_res.listed_step_bound_us
        <= per_layer
        < search_res.per_op_step_bound_us
    )
    assert search_res.gap_vs_per_op_us > 0.0
    assert "one fused kernel per group" in search_res.note
    assert "per_op" in search_res.note


def test_measured_per_op_step_and_coverage_set_whole_server_gain_basis() -> None:
    """Verify gain_basis requires per_op_step_us and step_coverage to be MEASURED."""
    base_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
    )
    uncalibrated_report = advisor.advise("qwen3.5-397b", base_workload, "tpu7x")
    expected_estimate_basis = "fused_scope_device_estimate"
    expected_e2e_basis = "calibrated_whole_server_e2e"
    assert uncalibrated_report.gain_basis == expected_estimate_basis
    assert uncalibrated_report.metrics["gain_basis"]["value"] == expected_estimate_basis
    assert uncalibrated_report.metrics["per_op_step_us"]["tag"] == "ASSUMED"

    scope_affecting = {
        str(item["name"])
        for item in uncalibrated_report.unresolved_assumptions
        if item["affects"] == "scope"
    }
    assert scope_affecting <= VALID_SCOPE_ASSUMPTIONS
    unresolved_names = {
        str(item["name"]) for item in uncalibrated_report.unresolved_assumptions
    }
    for expected_name in (
        "per_op_step_us",
        "step_coverage",
        "host_head_us",
        "eta_hbm",
        "eta_dense",
        "decoding_sequences",
    ):
        assert expected_name in unresolved_names
    for item in uncalibrated_report.unresolved_assumptions:
        for required_key in (
            "name",
            "value",
            "band",
            "unit",
            "sensitivity_us",
            "affects",
            "how_to_settle",
        ):
            assert required_key in item

    only_step_workload = dataclasses.replace(base_workload, per_op_step_us=9500.0)
    only_step_report = advisor.advise("qwen3.5-397b", only_step_workload, "tpu7x")
    assert only_step_report.gain_basis == expected_estimate_basis
    assert only_step_report.metrics["per_op_step_us"]["tag"] == "MEASURED"

    calibrated_measurements = {
        "schema": constants.MEASUREMENTS_SCHEMA,
        "source": {
            "kind": "xprof_xplane",
            "file": "baseline_trace.xplane.pb",
            "image": "vllm-tpu:baseline-v1",
            "scope": "per_op",
        },
        "step_time_us": {"p50": 9500.0},
        "step_coverage": {"fused_eligible_fraction": 0.85},
        "host_head_us": {"p50": 4800.0, "p90": 5100.0, "n": 12},
    }
    calibrated_report = advisor.advise(
        "qwen3.5-397b",
        base_workload,
        "tpu7x",
        measurements=calibrated_measurements,
    )
    assert calibrated_report.gain_basis == expected_e2e_basis
    assert calibrated_report.metrics["gain_basis"]["value"] == expected_e2e_basis
    assert calibrated_report.metrics["per_op_step_us"]["tag"] == "MEASURED"
    assert (
        "vllm-tpu:baseline-v1" in calibrated_report.metrics["per_op_step_us"]["source"]
    )
    assert calibrated_report.metrics["calibrated_host_head_us"][
        "value"
    ] == pytest.approx(4800.0)


def test_asymptotic_decode_intensity_invariant_to_dcp_and_caps_dsa() -> None:
    """Verify asymptotic intensity DCP physics, DSA capping, and phase/spec rules."""
    # KV cache read bytes stay invariant to DCP when num_kv_heads >= tp.
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_tpu7x = advisor.resolve_chip_spec("tpu7x")
    wl_norep_d1 = advisor.WorkloadSpec(
        phase="decode", batch=16, context=4096, tp=2, ep=8, dp=4, dcp=1
    )
    wl_norep_d2 = advisor.WorkloadSpec(
        phase="decode", batch=16, context=4096, tp=2, ep=8, dp=4, dcp=2
    )
    ev_norep_d1 = _evaluate_scopes_with_context(qwen_profile, wl_norep_d1, chip_tpu7x)
    ev_norep_d2 = _evaluate_scopes_with_context(qwen_profile, wl_norep_d2, chip_tpu7x)
    r_norep_d1 = advisor.advise(qwen_profile, wl_norep_d1, chip_tpu7x)
    r_norep_d2 = advisor.advise(qwen_profile, wl_norep_d2, chip_tpu7x)
    assert (
        ev_norep_d2.scope_context.kv_cache_read_bytes
        == ev_norep_d1.scope_context.kv_cache_read_bytes
    )
    norep_state_bytes = (
        ev_norep_d1.scope_context.kv_cache_read_bytes
        + ev_norep_d1.scope_context.recurrent_state_rw_bytes
    )
    kv_proj_ai_delta = (
        abs(
            ev_norep_d2.scope_context.flops_dense
            - ev_norep_d1.scope_context.flops_dense
        )
        / norep_state_bytes
    )
    assert r_norep_d1.asymptotic_decode_intensity is not None
    assert r_norep_d2.asymptotic_decode_intensity == pytest.approx(
        r_norep_d1.asymptotic_decode_intensity,
        abs=round(kv_proj_ai_delta, 2) + 0.02,
    )

    # Scales UP with DCP when DCP eliminates KV or MLA replication.
    r_q35_d1 = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, context=4096, tp=8, ep=8, dcp=1),
        "tpu7x",
    )
    r_q35_d4 = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, context=4096, tp=8, ep=8, dcp=4),
        "tpu7x",
    )
    assert r_q35_d1.asymptotic_decode_intensity is not None
    assert r_q35_d4.asymptotic_decode_intensity is not None
    assert r_q35_d4.asymptotic_decode_intensity > r_q35_d1.asymptotic_decode_intensity

    kimi_profile = advisor.resolve_model_profile("kimi-k3")
    long_context = 100_000_000
    kimi_dcp1_report = advisor.advise(
        kimi_profile,
        advisor.WorkloadSpec(
            phase="decode", batch=4, context=long_context, tp=32, ep=32, dcp=1
        ),
        "tpu7x",
    )
    r_k3_d32 = advisor.advise(
        kimi_profile,
        advisor.WorkloadSpec(
            phase="decode", batch=4, context=long_context, tp=32, ep=32, dcp=32
        ),
        "tpu7x",
    )
    assert kimi_dcp1_report.asymptotic_decode_intensity is not None
    assert r_k3_d32.asymptotic_decode_intensity is not None
    assert (
        r_k3_d32.asymptotic_decode_intensity
        > kimi_dcp1_report.asymptotic_decode_intensity
    )

    # Caps DSA span at index_topk and satisfies step_AI <= asymp_AI at ctx=1e6.
    glm_profile = advisor.resolve_model_profile("glm-5.2")
    r_glm_1 = advisor.advise(
        glm_profile,
        advisor.WorkloadSpec(
            phase="decode", batch=4, context=32768, tp=16, ep=16, dcp=4
        ),
        "tpu7x",
    )
    r_glm_2 = advisor.advise(
        glm_profile,
        advisor.WorkloadSpec(
            phase="decode", batch=4, context=65536, tp=16, ep=16, dcp=4
        ),
        "tpu7x",
    )
    assert r_glm_1.asymptotic_decode_intensity is not None
    assert r_glm_2.asymptotic_decode_intensity is not None
    assert r_glm_2.asymptotic_decode_intensity < r_glm_1.asymptotic_decode_intensity

    r_glm_long = advisor.advise(
        glm_profile,
        advisor.WorkloadSpec(
            phase="decode", batch=4096, context=1_000_000, tp=16, ep=16, dcp=16
        ),
        "tpu7x",
    )
    assert r_glm_long.asymptotic_decode_intensity is not None
    assert (
        r_glm_long.step_arithmetic_intensity <= r_glm_long.asymptotic_decode_intensity
    )
    idx_score_ai = (
        2.0 * glm_profile.index_n_heads * glm_profile.index_head_dim
    ) / models.indexer_record_bytes(glm_profile.index_head_dim)
    assert r_glm_long.asymptotic_decode_intensity > idx_score_ai

    # Returns None for prefill/mixed; decode ignores spec_tokens.
    r_prefill = advisor.advise(
        glm_profile,
        advisor.WorkloadSpec(phase="prefill", batch=1, isl=4096, tp=16, ep=16, dcp=16),
        "tpu7x",
    )
    r_mixed = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(
            phase="mixed", batch=8, isl=4096, chunk_size=512, tp=8, ep=8
        ),
        "tpu7x",
    )
    assert r_prefill.asymptotic_decode_intensity is None
    assert r_mixed.asymptotic_decode_intensity is None

    r_dec_s0 = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, spec_tokens=0, tp=8, ep=8),
        "tpu7x",
    )
    r_dec_s3 = advisor.advise(
        qwen_profile,
        advisor.WorkloadSpec(phase="decode", batch=16, spec_tokens=3, tp=8, ep=8),
        "tpu7x",
    )
    assert r_dec_s3.asymptotic_decode_intensity == r_dec_s0.asymptotic_decode_intensity


def test_moe_mode_selection_falls_back_when_expert_dtype_empty() -> None:
    """Verify MoE mode falls back to weight_dtype and matches explicit expert_dtype."""
    base_qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    no_exp_dt = dataclasses.replace(base_qwen_profile, expert_dtype="")
    workload = advisor.WorkloadSpec(phase="decode", batch=64, tp=8, ep=8)
    report = advisor.advise(no_exp_dt, workload, "tpu7x")
    explicit_report = advisor.advise(
        dataclasses.replace(base_qwen_profile, expert_dtype="fp8"), workload, "tpu7x"
    )
    assert report.recommended_moe_mode == "grouped_on_chip"
    assert report.recommended_moe_mode == explicit_report.recommended_moe_mode


def test_mixed_phase_throughput_prices_decode_once_and_divides_by_pcp() -> None:
    """Verify mixed divides by active PCP while decode keeps effective_pcp=1."""
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    chip_tpu7x = advisor.resolve_chip_spec("tpu7x")
    mixed_wl = advisor.WorkloadSpec(
        phase="mixed",
        batch=8,
        isl=1024,
        chunk_size=512,
        tp=8,
        ep=8,
        pcp=4,
    )
    report = advisor.advise(qwen_profile, mixed_wl, chip_tpu7x)
    recommended_estimate = _scope_by_name(report, report.recommended_scope)
    decode_tokens = mixed_wl.batch
    chunk_tokens = mixed_wl.chunk_size // mixed_wl.pcp
    expected_tokens_per_chip = (
        (decode_tokens + chunk_tokens * mixed_wl.pcp)
        * chip_tpu7x.devices_per_chip
        / (mixed_wl.tp * mixed_wl.pp * mixed_wl.pcp)
    )
    expected_high = round(
        expected_tokens_per_chip / (recommended_estimate.step_bound_us * 1e-6), 2
    )
    tput_hi = float(report.metrics["throughput_tok_s_per_chip"]["band"][1])
    assert tput_hi == pytest.approx(expected_high, abs=0.02)

    decode_pcp_wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8, pcp=2)
    dec_report = advisor.advise(qwen_profile, decode_pcp_wl, chip_tpu7x)
    dec_rec = _scope_by_name(dec_report, dec_report.recommended_scope)
    dec_tokens_per_chip = (
        decode_pcp_wl.batch
        * chip_tpu7x.devices_per_chip
        / (decode_pcp_wl.tp * decode_pcp_wl.pp)
    )
    expected_dec_high = round(dec_tokens_per_chip / (dec_rec.step_bound_us * 1e-6), 2)
    dec_tput_hi = float(dec_report.metrics["throughput_tok_s_per_chip"]["value"])
    assert dec_tput_hi == pytest.approx(expected_dec_high, abs=0.02)


def test_search_layer_groups_marks_hbm_oom_candidates_infeasible() -> None:
    """Verify search_layer_groups clears bound on HBM-OOM candidates."""
    tiny_hbm_chip = dataclasses.replace(
        advisor.resolve_chip_spec("tpu7x"), hbm_capacity_bytes=1 << 20
    )
    workload = advisor.WorkloadSpec(
        phase="decode", batch=16, tp=8, ep=8, layer_groups=(60,)
    )
    search_res = advisor.search_layer_groups("qwen3.5-397b", workload, tiny_hbm_chip)
    assert search_res.best is None
    assert search_res.listed_step_bound_us is None
    assert search_res.gap_vs_per_op_us == 0.0
    assert all(not cand.feasible for cand in search_res.candidates)


def test_uncalibrated_how_to_settle_mentions_no_unlanded_cli_flags() -> None:
    """Verify verdict how_to_settle and step_coverage sources avoid CLI flags."""
    report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
    )
    for item in report.unresolved_assumptions:
        hint = str(item["how_to_settle"])
        assert "--" not in hint
        assert "advisor calibrate" not in hint
    step_cov_source = str(report.metrics["step_coverage"]["source"])
    assert "--" not in step_cov_source
    assert "advisor calibrate" not in step_cov_source


def test_advise_matches_evaluate_scopes_with_measurements_ring_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify advise shares _evaluate_scopes_with_context once with evaluate_scopes."""
    qwen_profile = advisor.resolve_model_profile("qwen3.5-397b")
    workload = advisor.WorkloadSpec(
        phase="decode", batch=16, tp=8, ep=8, moe_mode_explicit=True
    )
    meas = advisor.load_measurements(
        {
            "schema": constants.MEASUREMENTS_SCHEMA,
            "source": {"kind": "xprof_xplane", "desc": "ring", "file": "ring.pb"},
            "ring_plan": {"dense_slots": 4, "expert_slots": 3},
        },
        chip_name="tpu7x",
    )
    expected_scopes = advisor.evaluate_scopes(
        qwen_profile, workload, "tpu7x", measurements=meas
    )

    calls = 0

    def _counting_eval(
        profile: models.ModelProfile,
        eval_wl: specs.WorkloadSpec,
        chip_spec: advisor.ChipSpec,
        *,
        plan_error: str = "",
        measurements: specs.MeasurementsData | None = None,
    ) -> _EvaluatedScopes:
        """Count _evaluate_scopes_with_context invocations inside advise."""
        nonlocal calls
        calls += 1
        return _evaluate_scopes_with_context(
            profile,
            eval_wl,
            chip_spec,
            plan_error=plan_error,
            measurements=measurements,
        )

    monkeypatch.setattr(verdict, "_evaluate_scopes_with_context", _counting_eval)
    report = advisor.advise(qwen_profile, workload, "tpu7x", measurements=meas)
    assert calls == 1
    assert report.scopes == expected_scopes
    whole = _scope_by_name(report, "whole_step")
    expected_dense_slots = 4
    expected_expert_slots = 3
    assert whole.selected_dense_slots == expected_dense_slots
    assert whole.selected_expert_slots == expected_expert_slots


@pytest.mark.parametrize(
    ("model_slug", "mode"),
    [
        (slug, mode)
        for slug in sorted(advisor.MODEL_PRESETS)
        for mode in ("decode", "prefill", "pp_decode")
    ],
)
def test_scope_assumption_sensitivities_match_roofline_upper_deltas(
    model_slug: str,
    mode: str,
) -> None:
    """Verify boundary hop and collective call sensitivities match roofline."""
    profile = advisor.resolve_model_profile(model_slug)
    preset = advisor.DEPLOYMENT_PRESETS[model_slug]
    chip_tpu7x = advisor.resolve_chip_spec("tpu7x")
    if mode == "pp_decode":
        fitting_pp = next(
            pp
            for pp in (8, 16, 32)
            if pp <= profile.num_layers
            and advisor.advise(
                profile,
                advisor.WorkloadSpec(phase="decode", batch=16, tp=1, ep=1, pp=pp),
                chip_tpu7x,
            ).metrics["fits_hbm"]["value"]
        )
        workload = advisor.WorkloadSpec(
            phase="decode", batch=16, tp=1, ep=1, pp=fitting_pp
        )
    elif mode == "prefill":
        layout = preset.prefill
        workload = advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=4096,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
        )
    else:
        layout = preset.decode
        workload = advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            tp=layout.tp,
            ep=layout.ep,
            dp=layout.dp,
            pp=layout.pp,
            dcp=layout.dcp,
            pcp=layout.pcp,
        )

    report = advisor.advise(profile, workload, chip_tpu7x)
    by_name = {
        str(item["name"]): float(str(item["sensitivity_us"]))
        for item in report.unresolved_assumptions
    }
    evaluated = _evaluate_scopes_with_context(profile, workload, chip_tpu7x)
    # Zero first_slot_bytes to isolate boundary-sync hop delta from DMA fill/drain.
    zero_slot_ctx = dataclasses.replace(evaluated.scope_context, first_slot_bytes=0.0)
    est_with_factor, _ = _evaluate_scopes(
        zero_slot_ctx,
        profile,
        evaluated.eval_workload,
        upper_priors=(True, True),
    )
    est_without_factor, _ = _evaluate_scopes(
        zero_slot_ctx,
        profile,
        evaluated.eval_workload,
        upper_priors=(False, True),
    )
    fill_deltas = {
        est_w.scope: est_w.step_upper_us - est_wo.step_upper_us
        for est_w, est_wo in zip(est_with_factor, est_without_factor, strict=True)
    }
    per_op_est = report.scopes[0]
    candidates = [
        est
        for est in report.scopes
        if est.scope == "per_op"
        or (
            est.feasible
            and est.step_bound_us <= per_op_est.step_bound_us
            and est.step_achievable_us <= per_op_est.step_achievable_us
        )
    ]
    fastest = next(est for est in candidates if est.scope == report.fastest_scope)
    upper_winner = min(
        candidates,
        key=lambda est: (est.step_upper_us, est.step_bound_us, est.ops_per_step),
    )
    if upper_winner.scope != fastest.scope:
        challenger = upper_winner
    else:
        others = [est for est in candidates if est.scope != fastest.scope]
        challenger = (
            min(others, key=lambda est: est.step_bound_us)
            if others
            else report.scopes[0]
        )
    expected_sync_delta = round(
        abs(fill_deltas[challenger.scope] - fill_deltas[fastest.scope]), 2
    )
    assert by_name["boundary_sync_hop_factor"] == pytest.approx(
        expected_sync_delta, abs=0.05
    )
    if mode == "pp_decode":
        assert by_name["boundary_sync_hop_factor"] > 0.0

    est_with_coll, _ = _evaluate_scopes(
        evaluated.scope_context,
        profile,
        evaluated.eval_workload,
        upper_priors=(True, True),
    )
    est_without_coll, _ = _evaluate_scopes(
        evaluated.scope_context,
        profile,
        evaluated.eval_workload,
        upper_priors=(True, False),
    )
    expected_coll_delta = round(
        est_with_coll[0].step_upper_us - est_without_coll[0].step_upper_us, 2
    )
    if expected_coll_delta > 0.0:
        assert by_name["per_op_collective_call_factor"] == pytest.approx(
            expected_coll_delta, abs=0.05
        )
    else:
        assert by_name.get("per_op_collective_call_factor", 0.0) == 0.0


_ROUND_HALF_UNIT_US = 0.005
_ROUND_EPS_US = 1e-9
_TWO_FIELD_ROUND_TOL_US = 2 * _ROUND_HALF_UNIT_US + _ROUND_EPS_US
_THREE_FIELD_ROUND_TOL_US = 3 * _ROUND_HALF_UNIT_US + _ROUND_EPS_US
_FOUR_FIELD_ROUND_TOL_US = 4 * _ROUND_HALF_UNIT_US + _ROUND_EPS_US
_BAND_LEN = 2


def _try_advise(
    profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
    chip: advisor.ChipSpec,
) -> advisor.AdvisorReport | None:
    """Run `advisor.advise` and return `None` when `MegakernelConfigError` is raised."""
    try:
        return advisor.advise(profile, workload, chip)
    except errors.MegakernelConfigError:
        return None


def _collect_band_violations(label: str, report: advisor.AdvisorReport) -> list[str]:
    """Return descriptions of any numeric metric lying outside its stated band."""
    issues: list[str] = []
    for name, metric in report.metrics.items():
        band = metric.get("band")
        val = metric.get("value")
        if (
            isinstance(band, Sequence)
            and not isinstance(band, (str, bytes))
            and len(band) == _BAND_LEN
            and all(
                isinstance(bound, (int, float)) and not isinstance(bound, bool)
                for bound in band
            )
            and isinstance(val, (int, float))
            and not isinstance(val, bool)
        ):
            lo = float(band[0])
            hi = float(band[1])
            lo_bound, hi_bound = min(lo, hi), max(lo, hi)
            if not (
                lo_bound - _TWO_FIELD_ROUND_TOL_US
                <= float(val)
                <= hi_bound + _TWO_FIELD_ROUND_TOL_US
            ):
                issues.append(f"{label} {name}: value={val} outside band={[lo, hi]}")
    return issues


def _collect_invalid_config_issues(
    case_label: str,
    rep: advisor.AdvisorReport,
    expected_moe: str,
) -> list[str]:
    """Return any scope, rationale, or non-zero gain leakage on an invalid config."""
    zero_metrics = (
        "compiler_proof_gain_us",
        "compiler_proof_gain_pct",
        "gain_fact_us",
        "gain_assumed_us",
        "fusion_gain_max",
        "expected_gain_max",
        "achievable_gain_pct",
        "boundary_cost_us",
        "dispatch_bubble_gain_us",
        "throughput_tok_s_per_chip",
    )
    disallowed_tags = ("whole_step (", "layer_group (", "single_layer (")
    issues: list[str] = []
    if rep.recommended_scope != "per_op" or rep.fastest_scope != "per_op":
        issues.append(
            f"{case_label}: scopes=({rep.recommended_scope}, {rep.fastest_scope})"
        )
    if rep.build_megakernel is not False:
        issues.append(f"{case_label}: build={rep.build_megakernel}")
    if "hypothetical_best" in rep.to_dict():
        issues.append(f"{case_label}: unexpected hypothetical_best key")
    if (
        rep.core_support != "baseline"
        or rep.core_ready is not False
        or rep.core_blockers != ()
        or rep.core_ready_scope != "per_op"
    ):
        issues.append(
            f"{case_label}: core=({rep.core_support}, {rep.core_ready}, "
            f"{rep.core_blockers}, {rep.core_ready_scope})"
        )
    if rep.recommended_moe_mode != expected_moe:
        issues.append(
            f"{case_label}: moe_mode={rep.recommended_moe_mode} != {expected_moe}"
        )
    if any(tag in rep.rationale for tag in disallowed_tags):
        issues.append(f"{case_label}: rationale={rep.rationale!r}")
    for metric_name in zero_metrics:
        val = abs(float(rep.metrics[metric_name]["value"]))
        if val > _TWO_FIELD_ROUND_TOL_US:
            issues.append(f"{case_label} {metric_name}={val}")
    return issues


def test_gain_split_within_total() -> None:
    """Verify zero-hop factual gain never exceeds total gain and sums cleanly."""
    chip_default = case_chips()[0][1]
    cases: list[tuple[str, advisor.ModelProfile, advisor.WorkloadSpec]] = [
        (label, profile, wl) for label, _, profile, _, _, wl in iter_case_grid()
    ]
    tiled_q35_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=128,
        tp=8,
        ep=8,
        vmem_budget_mib=20.0,
        dispatch_regime="unfused_eager",
    )
    cases.append(
        (
            "qwen3.5-397b/decode/tiled_vmem20",
            advisor.resolve_model_profile("qwen3.5-397b"),
            tiled_q35_wl,
        )
    )
    violations: list[str] = []
    for label, profile, wl in cases:
        rep = advisor.advise(profile, wl, chip_default)
        fact = float(rep.metrics["gain_fact_us"]["value"])
        assumed = float(rep.metrics["gain_assumed_us"]["value"])
        total = float(rep.metrics["compiler_proof_gain_us"]["value"])
        if not (-_TWO_FIELD_ROUND_TOL_US <= fact <= total + _TWO_FIELD_ROUND_TOL_US):
            violations.append(f"{label}: fact={fact} not in [0, {total}]")
        if assumed < -_TWO_FIELD_ROUND_TOL_US:
            violations.append(f"{label}: assumed={assumed} < 0")
        if abs((fact + assumed) - total) > _THREE_FIELD_ROUND_TOL_US:
            violations.append(f"{label}: fact={fact} + assumed={assumed} != {total}")
    assert not violations, (
        f"Gain split violations ({len(violations)}): {violations[:5]}"
    )


def test_invalid_config_reports_no_gain() -> None:
    """Verify invalid meshes report per_op scope, zero gains, and clear rationale."""
    chip_default = case_chips()[0][1]
    invalid_meshes: tuple[tuple[int, int], ...] = (
        (3, 1),
        (7, 1),
        (64, 1),
        (64, 64),
        (4, 4),
    )
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        expected_moe = "dense_mlp" if profile.num_moe_layers == 0 else "grouped_in_hbm"
        checked = 0
        for tp_val, ep_val in invalid_meshes:
            wl = advisor.WorkloadSpec(phase="decode", batch=16, tp=tp_val, ep=ep_val)
            rep = _try_advise(profile, wl, chip_default)
            if rep is None or not rep.verdict.startswith("invalid_config"):
                continue
            checked += 1
            violations.extend(
                _collect_invalid_config_issues(
                    f"{slug} tp={tp_val} ep={ep_val}", rep, expected_moe
                )
            )
        if checked < 1:
            violations.append(f"{slug}: no invalid_config case exercised")
    assert not violations, (
        f"Invalid config gain/scope violations ({len(violations)}): {violations[:5]}"
    )


def test_compare_parallelism_matches_advise() -> None:
    """Verify compare_parallelism_candidates matches direct advise across presets."""
    chip_default = case_chips()[0][1]
    prefill_like = frozenset({"prefill", "mixed"})
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        for phase in ("decode", "prefill", "verify", "mixed"):
            preset_axes = dict(case_layouts(slug, phase))["preset"]
            ctx_val = 16384 if phase in prefill_like else 0
            template = case_workload(phase, preset_axes, context=ctx_val, osl=2048)
            layout = advisor.ParallelLayout(
                tp=template.tp,
                ep=template.ep,
                dp=template.dp,
                pp=template.pp,
                dcp=template.dcp,
                pcp=template.pcp,
            )
            per_replica = dataclasses.replace(
                template, batch=max(1, template.batch // template.dp)
            )
            res = advisor.compare_parallelism_candidates(
                profile, [layout], chip=chip_default, base_workload=per_replica
            )
            direct = advisor.advise(
                profile, dataclasses.replace(template, objective="step"), chip_default
            )
            rows = res["candidates"]
            assert isinstance(rows, list)
            row = rows[0]
            label = f"{slug}/{phase}"
            cmp_step = float(row["step_floor_us"])
            dir_step = float(direct.metrics["step_floor_us"]["value"])
            if abs(cmp_step - dir_step) > _TWO_FIELD_ROUND_TOL_US:
                violations.append(f"{label} step_floor_us: {cmp_step} vs {dir_step}")
            cmp_tput = float(row["throughput_tok_s_per_chip"])
            dir_tput = float(direct.metrics["throughput_tok_s_per_chip"]["value"])
            if abs(cmp_tput - dir_tput) > _TWO_FIELD_ROUND_TOL_US:
                violations.append(f"{label} throughput: {cmp_tput} vs {dir_tput}")
            if row["recommended_scope"] != direct.recommended_scope:
                violations.append(
                    f"{label} recommended_scope: "
                    f"{row['recommended_scope']} vs {direct.recommended_scope}"
                )
            if row["fastest_scope"] != direct.fastest_scope:
                violations.append(
                    f"{label} fastest_scope: "
                    f"{row['fastest_scope']} vs {direct.fastest_scope}"
                )
    assert not violations, (
        f"Parallelism comparison mismatches ({len(violations)}): {violations[:5]}"
    )


def test_entry_points_reject_same_inputs() -> None:
    """Verify advise and search_layer_groups reject invalid layer_groups identically."""
    chip_default = case_chips()[0][1]
    max_core_rank = 16
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        dec = advisor.DEPLOYMENT_PRESETS[slug].decode
        core_axes = {
            "tp": min(dec.tp, max_core_rank),
            "ep": min(dec.ep, max_core_rank),
            "dp": 1,
            "pp": 1,
            "dcp": 1,
            "pcp": 1,
        }
        wl_bad = dataclasses.replace(
            case_workload("decode", core_axes), layer_groups=(10, 10)
        )
        with pytest.raises(errors.MegakernelConfigError):
            advisor.advise(profile, wl_bad, chip_default)
        with pytest.raises(errors.MegakernelConfigError):
            advisor.search_layer_groups(profile, wl_bad, chip_default)


def test_unused_axis_keeps_throughput_per_chip() -> None:
    """Verify decode/verify ignore pcp for both advise and parallelism comparison."""
    chip_default = case_chips()[0][1]
    max_core_rank = 16
    violations: list[str] = []
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        dec = advisor.DEPLOYMENT_PRESETS[slug].decode
        core_axes = {
            "tp": min(dec.tp, max_core_rank),
            "ep": min(dec.ep, max_core_rank),
            "dp": 1,
            "pp": 1,
            "dcp": 1,
            "pcp": 1,
        }
        for phase in ("decode", "verify"):
            wl_pcp1 = case_workload(phase, core_axes, pcp=1)
            wl_pcp2 = dataclasses.replace(wl_pcp1, pcp=2)
            rep_pcp1 = advisor.advise(profile, wl_pcp1, chip_default)
            rep_pcp2 = advisor.advise(profile, wl_pcp2, chip_default)
            label = f"{slug}/{phase}"
            if rep_pcp2.verdict.startswith("invalid_config"):
                violations.append(f"{label}: pcp=2 verdict={rep_pcp2.verdict}")
                continue
            if rep_pcp1.recommended_scope != rep_pcp2.recommended_scope:
                violations.append(
                    f"{label} recommended_scope: {rep_pcp1.recommended_scope} vs "
                    f"{rep_pcp2.recommended_scope}"
                )
            if rep_pcp1.fastest_scope != rep_pcp2.fastest_scope:
                violations.append(
                    f"{label} fastest_scope: {rep_pcp1.fastest_scope} vs "
                    f"{rep_pcp2.fastest_scope}"
                )
            tput1 = float(rep_pcp1.metrics["throughput_tok_s_per_chip"]["value"])
            tput2 = float(rep_pcp2.metrics["throughput_tok_s_per_chip"]["value"])
            if abs(tput1 - tput2) > _TWO_FIELD_ROUND_TOL_US:
                violations.append(f"{label} advise tput: {tput1} vs {tput2}")
            lay1 = advisor.ParallelLayout(tp=core_axes["tp"], ep=core_axes["ep"], pcp=1)
            lay2 = advisor.ParallelLayout(tp=core_axes["tp"], ep=core_axes["ep"], pcp=2)
            comp = advisor.compare_parallelism_candidates(
                profile, [lay1, lay2], chip=chip_default, base_workload=wl_pcp1
            )
            rows = comp["candidates"]
            assert isinstance(rows, list)
            row1, row2 = rows
            if row1["chips"] != row2["chips"]:
                violations.append(
                    f"{label} compare chips: {row1['chips']} vs {row2['chips']}"
                )
            cmp_tput1 = float(row1["throughput_tok_s_per_chip"])
            cmp_tput2 = float(row2["throughput_tok_s_per_chip"])
            if abs(cmp_tput1 - cmp_tput2) > _TWO_FIELD_ROUND_TOL_US:
                violations.append(f"{label} compare tput: {cmp_tput1} vs {cmp_tput2}")
            wl_single_p2 = case_workload(
                phase,
                {"tp": 1, "ep": 1, "dp": 1, "pp": 1, "dcp": 1, "pcp": 1},
                pcp=2,
            )
            rep_single_p2 = advisor.advise(profile, wl_single_p2, chip_default)
            if any(
                row.get("name") == "eta_ici"
                for row in rep_single_p2.unresolved_assumptions
            ):
                violations.append(f"{label} single-rank pcp=2 listed eta_ici")
    assert not violations, (
        f"Unused axis throughput violations ({len(violations)}): {violations[:5]}"
    )


def test_metric_values_inside_bands() -> None:
    """Verify all metric values sit inside bands and OOM throughput avoids blanks."""
    chip_default = case_chips()[0][1]
    violations: list[str] = []
    for label, _, profile, _, _, wl in iter_case_grid():
        rep = advisor.advise(profile, wl, chip_default)
        violations.extend(_collect_band_violations(label, rep))

    custom_eff = advisor.EfficiencySpec(
        eta_hbm=0.95, eta_dense=0.95, eta_moe_grouped=0.95, eta_ici=0.95
    )
    custom_chips = (
        ("lat12", dataclasses.replace(chip_default, collective_latency_s=12.0e-6)),
        ("lat0", dataclasses.replace(chip_default, collective_latency_s=0.0)),
    )
    tiny_chip = dataclasses.replace(chip_default, hbm_capacity_bytes=1 << 30)
    huge_batch = 100000
    for slug in sorted(advisor.MODEL_PRESETS):
        profile = advisor.resolve_model_profile(slug)
        preset_axes = dict(case_layouts(slug, "decode"))["preset"]
        wl_eff = case_workload("decode", preset_axes, efficiency=custom_eff)
        violations.extend(
            _collect_band_violations(
                f"{slug}/custom_eff", advisor.advise(profile, wl_eff, chip_default)
            )
        )
        wl_dec = case_workload("decode", preset_axes)
        for chip_label, custom_chip in custom_chips:
            violations.extend(
                _collect_band_violations(
                    f"{slug}/{chip_label}",
                    advisor.advise(profile, wl_dec, custom_chip),
                )
            )
        for oom_label, oom_wl, oom_chip in (
            (
                "tiny_hbm",
                case_workload("decode", preset_axes, objective="throughput"),
                tiny_chip,
            ),
            (
                "huge_batch",
                case_workload(
                    "decode", preset_axes, batch=huge_batch, objective="throughput"
                ),
                chip_default,
            ),
        ):
            rep_oom = advisor.advise(profile, oom_wl, oom_chip)
            if rep_oom.rationale.rstrip().endswith("best_throughput_layout=") or (
                "; best_throughput_layout=;" in rep_oom.rationale
            ):
                violations.append(
                    f"{slug}/{oom_label} empty best_throughput_layout: "
                    f"{rep_oom.rationale!r}"
                )
    assert not violations, (
        f"Metric band / OOM rationale violations ({len(violations)}): {violations[:5]}"
    )


def test_sensitivities_match_finite_differences() -> None:
    """Verify boundary sync and collective sensitivities match roofline finite diffs."""
    chip_default = case_chips()[0][1]
    sync_cases = (
        (
            "glm-5.2/mixed",
            advisor.resolve_model_profile("glm-5.2"),
            advisor.WorkloadSpec(
                phase="mixed", tp=16, ep=16, dcp=16, batch=8, isl=1024, chunk_size=512
            ),
        ),
        (
            "qwen3.5-397b/decode_tiled",
            advisor.resolve_model_profile("qwen3.5-397b"),
            advisor.WorkloadSpec(
                phase="decode", tp=8, ep=8, batch=128, vmem_budget_mib=20.0
            ),
        ),
        (
            "qwen3.5-397b/pp8_tiled",
            advisor.resolve_model_profile("qwen3.5-397b"),
            advisor.WorkloadSpec(
                phase="decode", tp=1, pp=8, ep=1, batch=512, vmem_budget_mib=16.0
            ),
        ),
    )
    for label, profile, wl in sync_cases:
        rep = advisor.advise(profile, wl, chip_default)
        evaluated = _evaluate_scopes_with_context(profile, wl, chip_default)
        zero_slot_ctx = dataclasses.replace(
            evaluated.scope_context, first_slot_bytes=0.0
        )
        est_with, _ = _evaluate_scopes(
            zero_slot_ctx, profile, evaluated.eval_workload, upper_priors=(True, True)
        )
        est_without, _ = _evaluate_scopes(
            zero_slot_ctx, profile, evaluated.eval_workload, upper_priors=(False, True)
        )
        deltas = {
            est_w.scope: est_w.step_upper_us - est_wo.step_upper_us
            for est_w, est_wo in zip(est_with, est_without, strict=True)
        }
        per_op_est = rep.scopes[0]
        candidates = [
            est
            for est in rep.scopes
            if est.scope == "per_op"
            or (
                est.feasible
                and est.step_bound_us <= per_op_est.step_bound_us
                and est.step_achievable_us <= per_op_est.step_achievable_us
            )
        ]
        fastest = next(est for est in candidates if est.scope == rep.fastest_scope)
        upper_winner = min(
            candidates,
            key=lambda est: (est.step_upper_us, est.step_bound_us, est.ops_per_step),
        )
        if upper_winner.scope != fastest.scope:
            compare_est = upper_winner
        else:
            others = [est for est in candidates if est.scope != fastest.scope]
            compare_est = (
                min(others, key=lambda est: est.step_bound_us)
                if others
                else rep.scopes[0]
            )
        expected_sync = abs(deltas[compare_est.scope] - deltas[fastest.scope])
        by_name = {
            str(item["name"]): float(str(item["sensitivity_us"]))
            for item in rep.unresolved_assumptions
        }
        assert "boundary_sync_hop_factor" in by_name, label
        assert (
            abs(by_name["boundary_sync_hop_factor"] - expected_sync)
            <= _FOUR_FIELD_ROUND_TOL_US
        ), f"{label}: {by_name['boundary_sync_hop_factor']} vs {expected_sync}"

    lo_us, hi_us = chip_default.collective_latency_us_bracket
    chip_lo = dataclasses.replace(chip_default, collective_latency_s=lo_us * 1e-6)
    chip_hi = dataclasses.replace(chip_default, collective_latency_s=hi_us * 1e-6)
    coll_cases = (
        (
            "qwen3.5-397b/dp8_ep8",
            advisor.resolve_model_profile("qwen3.5-397b"),
            advisor.WorkloadSpec(phase="decode", tp=1, ep=8, dp=8, batch=16, isl=4096),
            False,
            False,
        ),
        (
            "qwen3.5-397b/pp8_prefill",
            advisor.resolve_model_profile("qwen3.5-397b"),
            advisor.WorkloadSpec(phase="prefill", tp=1, ep=1, pp=8, batch=1, isl=1024),
            True,
            False,
        ),
        (
            "glm-5.2/tp16_ep16_dcp16",
            advisor.resolve_model_profile("glm-5.2"),
            advisor.WorkloadSpec(
                phase="decode", tp=16, ep=16, dcp=16, batch=16, isl=4096
            ),
            True,
            True,
        ),
    )
    for label, profile, wl, allow_call_factor, require_above_ar_only in coll_cases:
        rep = advisor.advise(profile, wl, chip_default)
        by_name = {
            str(item["name"]): float(str(item["sensitivity_us"]))
            for item in rep.unresolved_assumptions
        }
        assert "collective_latency_us" in by_name, label
        assert by_name["collective_latency_us"] > 0.0, label
        if not allow_call_factor:
            assert "per_op_collective_call_factor" not in by_name, label
        ev_lo = _evaluate_scopes_with_context(profile, wl, chip_lo)
        ev_hi = _evaluate_scopes_with_context(profile, wl, chip_hi)
        s_lo, _ = _evaluate_scopes(
            ev_lo.scope_context,
            profile,
            ev_lo.eval_workload,
            upper_priors=(False, False),
        )
        s_hi, _ = _evaluate_scopes(
            ev_hi.scope_context,
            profile,
            ev_hi.eval_workload,
            upper_priors=(False, False),
        )
        fd_total = (s_hi[0].comm_us + s_hi[0].fill_drain_us + s_hi[0].chain_us) - (
            s_lo[0].comm_us + s_lo[0].fill_drain_us + s_lo[0].chain_us
        )
        assert (
            abs(by_name["collective_latency_us"] - fd_total) <= _FOUR_FIELD_ROUND_TOL_US
        ), f"{label}: {by_name['collective_latency_us']} vs {fd_total}"
        if require_above_ar_only:
            ar_only_us = (
                ev_lo.scope_context.num_ar
                * ev_lo.scope_context.ar_hops
                * (hi_us - lo_us)
            )
            assert by_name["collective_latency_us"] > ar_only_us, label
