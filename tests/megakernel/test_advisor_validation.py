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
"""Soundness envelope and provenance tests for the megakernel advisor."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tests.megakernel.test_advisor_regimes import (
    LAUNCH_GAP_PRE_TIE_BATCH,
    LAUNCH_GAP_TIE_BATCH,
)
from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import chips
from vllm_torchtpu.megakernel.core import errors

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
    {"source_url", "source_quote", "approximate", "assumptions"}
)
PHASES: dict[str, advisor.Phase] = {
    "decode": "decode",
    "prefill": "prefill",
    "verify": "verify",
}
MOE_MODES: dict[str, advisor.MoEMode] = {
    "masked_on_chip": "masked_on_chip",
    "grouped_on_chip": "grouped_on_chip",
    "grouped_in_hbm": "grouped_in_hbm",
}


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


@pytest.fixture
def without_declared_megakernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip deployment `megakernel` blocks to test core-limit mechanics."""
    for model_key, preset in list(advisor.DEPLOYMENT_PRESETS.items()):
        stripped_sat = (
            dataclasses.replace(preset.saturation, megakernel=None)
            if preset.saturation is not None
            else None
        )
        monkeypatch.setitem(
            advisor.DEPLOYMENT_PRESETS,
            model_key,
            dataclasses.replace(
                preset,
                decode=dataclasses.replace(preset.decode, megakernel=None),
                prefill=dataclasses.replace(preset.prefill, megakernel=None),
                saturation=stripped_sat,
                megakernel_layouts=(),
            ),
        )


@pytest.mark.parametrize(
    ("model_slug", "tp_size", "ep_size", "dcp_size"),
    [
        ("qwen3.5-397b", 8, 8, 4),
        ("kimi-k3", 32, 32, 32),
        # INFERRED layout matching kimi-k3 decode until kimi-k3.1-pro has a recipe.
        ("kimi-k3.1-pro", 32, 32, 32),
        ("glm-5.2", 16, 16, 16),
    ],
)
def test_floor_ordering_and_monotonicity_across_models(
    model_slug: str,
    tp_size: int,
    ep_size: int,
    dcp_size: int,
) -> None:
    """Verify band ordering and batch/context monotonicity across submitted models."""
    small_batch = 4
    large_batch = 32
    short_context = 8192
    long_context = 32768
    workload_small = advisor.WorkloadSpec(
        phase="decode",
        batch=small_batch,
        context=short_context,
        osl=1,
        tp=tp_size,
        ep=ep_size,
        dcp=dcp_size,
    )
    workload_large_batch = dataclasses.replace(workload_small, batch=large_batch)
    workload_long_context = dataclasses.replace(workload_small, context=long_context)

    report_small = advisor.advise(model_slug, workload_small, "tpu7x")
    report_large_batch = advisor.advise(model_slug, workload_large_batch, "tpu7x")
    report_long_context = advisor.advise(model_slug, workload_long_context, "tpu7x")

    for report in (report_small, report_large_batch, report_long_context):
        for scope in report.scopes:
            if not scope.feasible:
                continue
            assert (
                scope.step_floor_strict_us <= scope.step_bound_us <= scope.step_upper_us
            )
            assert scope.step_achievable_us <= scope.step_achievable_upper_us

    small_per_op = next(s for s in report_small.scopes if s.scope == "per_op")
    large_per_op = next(s for s in report_large_batch.scopes if s.scope == "per_op")
    long_per_op = next(s for s in report_long_context.scopes if s.scope == "per_op")
    assert small_per_op.step_floor_strict_us < large_per_op.step_floor_strict_us
    assert small_per_op.step_floor_strict_us <= long_per_op.step_floor_strict_us


@pytest.mark.usefixtures("without_declared_megakernels")
def test_parallelism_objective_split_and_launch_gap_tie() -> None:
    """Verify objective splits across parallelism layouts and paired-corner ties."""
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

    regimes_sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    saturated_row = next(
        r for r in regimes_sweep.regimes if r.label.startswith("Decode Saturated")
    )
    assert saturated_row.fastest_scope == "whole_step"
    assert saturated_row.num_vmem_splits == 1

    # Reuse LAUNCH_GAP_PRE_TIE_BATCH and LAUNCH_GAP_TIE_BATCH with isl=16384.
    below_tie_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=LAUNCH_GAP_PRE_TIE_BATCH,
            tp=8,
            dp=2,
            ep=16,
            dcp=4,
            isl=16384,
            context=16384,
        ),
        "tpu7x",
    )
    assert below_tie_report.fastest_verdict == "megakernel_likely"
    assert below_tie_report.fastest_scope == "whole_step"

    tie_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=LAUNCH_GAP_TIE_BATCH,
            tp=8,
            dp=2,
            ep=16,
            dcp=4,
            isl=16384,
            context=16384,
        ),
        "tpu7x",
    )
    assert tie_report.fastest_verdict == "tie(measure: launch_gap_us)"
    assert tie_report.fastest_scope == "single_layer"
    assert any(
        unresolved["affects"] == "scope"
        for unresolved in tie_report.unresolved_assumptions
    )


def test_public_calibration_loader_and_validation(tmp_path: Path) -> None:
    """Verify public calibration loader semantics and corrupt-JSON rejection."""
    calibration_path = chips.public_calibration_path("tpu7x")
    calibration_data = chips.load_public_calibration("tpu7x")
    assert calibration_path is not None
    assert calibration_path.is_file()
    assert calibration_data is not None
    assert calibration_data["schema"] == "advisor.measurements.v1"
    assert "launch_gap_us" in calibration_data
    assert "step_time_us" not in calibration_data
    collectives_block = calibration_data.get("collectives")
    assert type(collectives_block) is list
    for entry in collectives_block:
        assert type(entry) is dict
        assert "step_time_us" not in entry

    calibrated_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=calibration_data,
    )
    assert calibrated_report.confidence == "prior_only"
    assert calibrated_report.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert calibrated_report.metrics["collective_latency_us"]["tag"] == "ASSUMED"

    bad_calibration_file = tmp_path / "tpu7x_public.json"
    bad_calibration_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(errors.MegakernelConfigError, match="Expected JSON object"):
        chips.load_public_calibration("tpu7x", calibration_dir=tmp_path)


def test_advisor_cli_with_bundled_measurements(tmp_path: Path) -> None:
    """Verify CLI `estimate` with `--measurements bundled` and missing-chip error."""
    out_json = tmp_path / "bundled_out.json"
    base_argv = [
        "estimate",
        "--model",
        "qwen3.5-397b",
        "--phase",
        "decode",
        "--batch",
        "16",
        "--tp",
        "8",
        "--ep",
        "8",
        "--measurements",
        "bundled",
    ]
    rc = advisor.main([*base_argv, "--chip", "tpu7x", "--json", str(out_json)])
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["confidence"] == "prior_only"
    assert payload["metrics"]["collective_latency_us"]["tag"] == "ASSUMED"
    assert payload["metrics"]["launch_gap_us"]["tag"] == "ASSUMED"

    with pytest.raises(
        errors.MegakernelConfigError, match="No bundled architectural priors"
    ):
        advisor.main([*base_argv, "--chip", "tpu6e"])


def test_qwen38_b256_directional_megakernel_win() -> None:
    """Verify single-pass adaptive ring whole_step beats per_op at B=64 and B=256."""
    tp_size = 32
    ep_size = 32
    dcp_size = 8
    isl_tokens = 1024
    osl_tokens = 1024
    context_tokens = 1536
    for batch_size in (64, 256):
        workload = advisor.WorkloadSpec(
            phase="decode",
            batch=batch_size,
            isl=isl_tokens,
            osl=osl_tokens,
            context=context_tokens,
            tp=tp_size,
            ep=ep_size,
            dcp=dcp_size,
            moe_mode="grouped_on_chip",
        )
        report = advisor.advise("qwen3.8-2.4t", workload, "tpu7x")
        whole_step_scope = next(s for s in report.scopes if s.scope == "whole_step")
        per_op_scope = next(s for s in report.scopes if s.scope == "per_op")
        assert whole_step_scope.n_splits == 1
        assert whole_step_scope.step_bound_us < per_op_scope.step_bound_us
        assert report.fastest_scope == "whole_step"


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
            assert bool(str(point["source_url"]).strip())
            assert isinstance(point["source_quote"], str)
            assert point["source_quote"].strip()
            assert type(point["approximate"]) is bool
            assumptions = point["assumptions"]
            assert type(assumptions) is list
            assert assumptions
        else:
            assert bool(str(point.get("job_id", "")).strip())
        assert int(str(point["n"])) >= 1
        assert float(str(point["step_time_us"])) > 0.0


def test_public_measured_point_respects_floor() -> None:
    """Verify public measured points do not fall below advisor step_bound_us."""
    document = json.loads(MEASURED_CORPUS_PATH.read_text(encoding="utf-8"))
    public_points = [
        point
        for point in document["points"]
        if point.get("provenance_kind") == "public_source"
    ]
    assert public_points
    for point in public_points:
        assert str(point["source_url"]).startswith("https://")
        assert isinstance(point["source_quote"], str)
        assert point["source_quote"].strip()
        assert isinstance(point["rows_quote"], str)
        assert point["rows_quote"].strip()
        assert point["approximate"] is True
        assumptions = point["assumptions"]
        assert type(assumptions) is list
        assert assumptions
        for item in assumptions:
            assert isinstance(item, str)
            assert item.strip()
        workload = _build_workload_from_point(point)
        report = advisor.advise(str(point["model"]), workload, str(point["chip"]))
        assert report.active_tokens == int(str(point["rows"]))
        scope_name = str(point["scope"])
        scopes_by_name = {item.scope: item for item in report.scopes}
        scope_estimate = scopes_by_name[scope_name]
        assert scope_estimate.step_bound_us <= float(str(point["step_time_us"]))
