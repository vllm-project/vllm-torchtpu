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

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import chips
from vllm_torchtpu.megakernel.core import errors

VALIDATION_CORPUS_PATH = (
    Path(__file__).resolve().parent / "data" / "advisor" / "validation_points.json"
)
MEASURED_CORPUS_PATH = (
    Path(__file__).resolve().parent / "data" / "advisor" / "measured_points.json"
)
ALLOWED_PROVENANCE_KINDS = frozenset({"synthetic_envelope_fixture"})
REQUIRED_MEASURED_KEYS = frozenset(
    {"job_id", "model", "chip", "scope", "workload", "layout", "step_time_us", "n"}
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


def _load_validation_corpus() -> list[dict[str, object]]:
    """Load and validate the top-level schema of `validation_points.json`."""
    assert VALIDATION_CORPUS_PATH.is_file(), f"Missing {VALIDATION_CORPUS_PATH}"
    document = json.loads(VALIDATION_CORPUS_PATH.read_text(encoding="utf-8"))
    assert document["schema"] == "advisor.validation.v1"
    points = document["points"]
    min_corpus_points = 2 * 7  # 14 validation points across models and regimes
    assert len(points) >= min_corpus_points
    return points


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


def _evaluate_validation_points() -> list[
    tuple[dict[str, object], advisor.ScopeEstimate]
]:
    """Evaluate every point in the validation corpus against `advisor.advise`."""
    evaluated: list[tuple[dict[str, object], advisor.ScopeEstimate]] = []
    for point in _load_validation_corpus():
        workload = _build_workload_from_point(point)
        report = advisor.advise(str(point["model"]), workload, str(point["chip"]))
        scope_name = str(point["scope"])
        estimate = next(s for s in report.scopes if s.scope == scope_name)
        evaluated.append((point, estimate))
    return evaluated


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
            ),
        )


def test_no_estimate_exceeds_measured_beyond_spread() -> None:
    """Verify every soundness fixture satisfies floor <= measured + spread."""
    models_seen: set[str] = set()
    for point, estimate in _evaluate_validation_points():
        models_seen.add(str(point["model"]))
        assert estimate.feasible, (
            f"{point['id']} scope {point['scope']} marked infeasible"
        )
        measured_us = float(str(point["fixture_step_us"]))
        spread_us = float(str(point["measured_spread_us"]))
        assert estimate.step_bound_us <= measured_us + spread_us, (
            f"{point['id']}: floor {estimate.step_bound_us:.2f} us exceeds "
            f"measured {measured_us:.2f} + spread {spread_us:.2f} us"
        )
    min_distinct_models = 3  # 3 distinct models tested in validation corpus
    assert len(models_seen) >= min_distinct_models


def test_validation_corpus_zero_lower_bound_violations() -> None:
    """Verify every soundness fixture satisfies floor <= measured step."""
    evaluated = _evaluate_validation_points()
    min_corpus_points = 2 * 7  # 14 validation points across models and regimes
    assert len(evaluated) >= min_corpus_points
    for point, estimate in evaluated:
        measured_us = float(str(point["fixture_step_us"]))
        assert estimate.step_upper_us >= estimate.step_bound_us
        assert estimate.step_bound_us <= measured_us, (
            f"{point['id']}: floor {estimate.step_bound_us:.2f} us exceeds "
            f"measured {measured_us:.2f} us"
        )


@pytest.mark.usefixtures("without_declared_megakernels")
def test_validation_corpus_provenance_and_tie_overlap() -> None:
    """Verify corpus provenance metadata, objective splits, and paired-corner ties."""
    for point in _load_validation_corpus():
        assert point["provenance_kind"] in ALLOWED_PROVENANCE_KINDS
        desc = str(point["workload_description"]).strip()
        assert desc
        assert "sha256:" not in desc
        assert "gs://" not in desc

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

    tie_report = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(
            phase="decode",
            batch=512,
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
        "Qwen/Qwen3-30B-A3B-FP8",
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
        "Qwen/Qwen3-30B-A3B-FP8",
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
    corpus_points = {str(point["id"]): point for point in _load_validation_corpus()}
    for batch_tag in ("b64", "b256"):
        mk_point = corpus_points[f"qwen38_2_4t_tp32_dcp8_mk_{batch_tag}"]
        per_op_point = corpus_points[f"qwen38_2_4t_tp32_dcp8_per_op_{batch_tag}"]
        assert float(str(mk_point["fixture_step_us"])) < float(
            str(per_op_point["fixture_step_us"])
        )
        workload = _build_workload_from_point(mk_point)
        report = advisor.advise("qwen3.8-2.4t", workload, "tpu7x")
        whole_step_scope = next(s for s in report.scopes if s.scope == "whole_step")
        per_op_scope = next(s for s in report.scopes if s.scope == "per_op")
        assert whole_step_scope.n_splits == 1
        assert whole_step_scope.step_bound_us < per_op_scope.step_bound_us
        assert report.fastest_scope == "whole_step"


def test_fixture_corpus_is_labelled_fixture() -> None:
    """Verify validation_points.json uses fixture_step_us and synthetic provenance."""
    raw_text = VALIDATION_CORPUS_PATH.read_text(encoding="utf-8")
    assert "measured_step_us" not in raw_text
    document = json.loads(raw_text)
    assert "synthetic" in str(document.get("description", "")).lower()
    for point in document["points"]:
        assert "fixture_step_us" in point
        assert "measured_step_us" not in point
        assert point["provenance_kind"] == "synthetic_envelope_fixture"


def test_measured_corpus_schema() -> None:
    """Verify measured_points.json exists and enforces advisor.measured_points.v1."""
    assert MEASURED_CORPUS_PATH.is_file()
    document = json.loads(MEASURED_CORPUS_PATH.read_text(encoding="utf-8"))
    assert document["schema"] == "advisor.measured_points.v1"
    points = document["points"]
    assert type(points) is list
    for point in points:
        assert REQUIRED_MEASURED_KEYS.issubset(point.keys())
        assert bool(str(point["job_id"]).strip())
        assert int(str(point["n"])) >= 1
        assert float(str(point["step_time_us"])) > 0.0
