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
"""Validation corpus and provenance invariant tests for the megakernel advisor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import advisor, errors
from vllm_torchtpu.megakernel.advisor import chips

_VALIDATION_CORPUS_PATH = (
    Path(__file__).resolve().parent / "data" / "advisor" / "validation_points.json"
)
_MIN_CORPUS_POINTS = 14
_MIN_DISTINCT_MODELS = 3
_ALLOWED_PROVENANCE_KINDS = frozenset({"public_calibration_run", "oss_tpu_measurement"})
_PHASES: dict[str, advisor.Phase] = {
    "decode": "decode",
    "prefill": "prefill",
    "verify": "verify",
}
_MOE_MODES: dict[str, advisor.MoEMode] = {
    "masked_on_chip": "masked_on_chip",
    "grouped_on_chip": "grouped_on_chip",
    "grouped_in_hbm": "grouped_in_hbm",
}


def _load_validation_corpus() -> list[dict[str, object]]:
    """Load and validate the top-level schema of `validation_points.json`."""
    assert _VALIDATION_CORPUS_PATH.is_file(), f"Missing {_VALIDATION_CORPUS_PATH}"
    doc = json.loads(_VALIDATION_CORPUS_PATH.read_text(encoding="utf-8"))
    assert doc["schema"] == "advisor.validation.v1"
    points = doc["points"]
    assert len(points) >= _MIN_CORPUS_POINTS
    return points


def _build_workload_from_point(pt: dict[str, object]) -> advisor.WorkloadSpec:
    """Construct a typed `WorkloadSpec` from a validation point dictionary."""
    raw = pt["workload"]
    if type(raw) is not dict:
        msg = f"Expected dict workload, got {type(raw).__name__}"
        raise TypeError(msg)
    defaults = advisor.WorkloadSpec()
    phase_str = str(raw.get("phase", defaults.phase))
    if phase_str not in _PHASES:
        msg = f"Unsupported phase: {phase_str!r}"
        raise ValueError(msg)
    moe_str = str(raw.get("moe_mode", defaults.moe_mode))
    if moe_str not in _MOE_MODES:
        msg = f"Unsupported moe_mode: {moe_str!r}"
        raise ValueError(msg)
    return advisor.WorkloadSpec(
        phase=_PHASES[phase_str],
        batch=int(str(raw.get("batch", defaults.batch))),
        isl=int(str(raw.get("isl", defaults.isl))),
        osl=int(str(raw.get("osl", defaults.osl))),
        context=int(str(raw.get("context", defaults.context))),
        spec_tokens=int(str(raw.get("spec_tokens", defaults.spec_tokens))),
        chunk_size=int(str(raw.get("chunk_size", defaults.chunk_size))),
        tp=int(str(raw.get("tp", defaults.tp))),
        pp=int(str(raw.get("pp", defaults.pp))),
        dp=int(str(raw.get("dp", defaults.dp))),
        ep=int(str(raw.get("ep", defaults.ep))),
        dcp=int(str(raw.get("dcp", defaults.dcp))),
        pcp=int(str(raw.get("pcp", defaults.pcp))),
        moe_mode=_MOE_MODES[moe_str],
    )


def test_no_estimate_exceeds_measured_beyond_spread() -> None:
    """Verify every silicon validation point satisfies floor <= measured + spread."""
    points = _load_validation_corpus()
    models_seen: set[str] = set()

    for pt in points:
        model_name = str(pt["model"])
        models_seen.add(model_name)
        chip_name = str(pt["chip"])
        scope_name = str(pt["scope"])
        workload = _build_workload_from_point(pt)
        rep = advisor.advise(model_name, workload, chip_name)
        est = next(s for s in rep.scopes if s.scope == scope_name)
        assert est.feasible, f"{pt['id']} scope {scope_name} marked infeasible"

        measured_us = float(str(pt["measured_step_us"]))
        spread_us = float(str(pt["measured_spread_us"]))
        assert est.step_bound_us <= measured_us + spread_us, (
            f"{pt['id']}: floor {est.step_bound_us:.2f} us exceeds "
            f"measured {measured_us:.2f} + spread {spread_us:.2f} us"
        )

    assert len(models_seen) >= _MIN_DISTINCT_MODELS


def test_validation_corpus_provenance_and_tie_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify corpus provenance metadata, tie overlap, and public calibration loader."""
    points = _load_validation_corpus()
    for pt in points:
        assert pt["provenance_kind"] in _ALLOWED_PROVENANCE_KINDS
        desc = str(pt["workload_description"]).strip()
        assert desc
        assert "sha256:" not in desc
        assert "gs://" not in desc

    # Verify objective-split tie reporting between step-latency and throughput layouts
    split_res = advisor.compare_parallelism_candidates(
        "kimi-k3",
        [
            advisor.ParallelLayout(tp=1, ep=32, dp=1, dcp=32),
            advisor.ParallelLayout(tp=1, ep=32, dp=2, dcp=16),
        ],
        batch=16,
        chip="tpu7x",
        context=8192,
        objective="both",
    )
    assert str(split_res["verdict"]).startswith("tie(objective_split:")

    # Verify a paired-corner tie names an assumed input that changes the winner
    sweep = advisor.advise_regimes("qwen3.5-397b", "tpu7x")
    tie_row = next(r for r in sweep.regimes if r.label.startswith("Decode Saturated"))
    assert tie_row.verdict == "tie(measure: collective_latency_us)"
    assert any(u["affects"] == "scope" for u in tie_row.unresolved_assumptions)

    # Verify public calibration loader API on chips module
    cal_path = chips.public_calibration_path("tpu7x")
    cal_data = chips.load_public_calibration("tpu7x")
    assert cal_path is not None
    assert cal_path.is_file()
    assert cal_data is not None
    assert cal_data["schema"] == "advisor.measurements.v1"
    cal_rep = advisor.advise(
        "Qwen/Qwen3-30B-A3B-FP8",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=cal_data,
    )
    # Bundled calibration carries collective priors only.
    assert "launch_gap_us" not in cal_data
    assert "step_time_us" not in cal_data
    for sub in cal_data.values():
        if isinstance(sub, dict):
            assert "launch_gap_us" not in sub
            assert "step_time_us" not in sub
    assert cal_rep.confidence == "bounded"
    assert cal_rep.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert cal_rep.metrics["collective_latency_us"]["tag"] == "MEASURED"

    # Non-dict JSON in calibration directory raises MegakernelConfigError
    bad_cal = tmp_path / "tpu7x_public.json"
    bad_cal.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(chips, "_CALIBRATION_DIR", tmp_path)
    with pytest.raises(errors.MegakernelConfigError, match="Expected JSON object"):
        chips.load_public_calibration("tpu7x")


def test_advisor_cli_with_bundled_measurements(tmp_path: Path) -> None:
    """Verify CLI with --measurements bundled resolves public priors.

    Also asserts that the resulting report produces bounded confidence.
    """
    out_json = tmp_path / "bundled_out.json"
    rc = advisor.main(
        [
            "estimate",
            "--model",
            "Qwen/Qwen3-30B-A3B-FP8",
            "--chip",
            "tpu7x",
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
            "--json",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["confidence"] == "bounded"
    assert payload["metrics"]["collective_latency_us"]["tag"] == "MEASURED"
    assert payload["metrics"]["launch_gap_us"]["tag"] == "ASSUMED"
