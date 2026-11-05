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

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import chips

_VALIDATION_CORPUS_PATH = (
    Path(__file__).resolve().parent / "data" / "advisor" / "validation_points.json"
)
_MIN_CORPUS_POINTS = 14
_MIN_DISTINCT_MODELS = 3
_ALLOWED_PROVENANCE_KINDS = frozenset({"public_calibration_run", "oss_tpu_measurement"})


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
    wl: dict[str, object] = raw if type(raw) is dict else {}
    phase_str = str(wl.get("phase", "decode"))
    phase_val: advisor.Phase = (
        "prefill"
        if phase_str == "prefill"
        else ("verify" if phase_str == "verify" else "decode")
    )
    moe_str = str(wl.get("moe_mode", "masked_on_chip"))
    moe_val: advisor.MoEMode = (
        "grouped_on_chip"
        if moe_str == "grouped_on_chip"
        else ("grouped_in_hbm" if moe_str == "grouped_in_hbm" else "masked_on_chip")
    )
    return advisor.WorkloadSpec(
        phase=phase_val,
        batch=int(str(wl.get("batch", 16))),
        isl=int(str(wl.get("isl", 4096))),
        osl=int(str(wl.get("osl", 1024))),
        context=int(str(wl.get("context", 0))),
        spec_tokens=int(str(wl.get("spec_tokens", 0))),
        chunk_size=int(str(wl.get("chunk_size", 0))),
        tp=int(str(wl.get("tp", 8))),
        pp=int(str(wl.get("pp", 1))),
        dp=int(str(wl.get("dp", 1))),
        ep=int(str(wl.get("ep", 1))),
        dcp=int(str(wl.get("dcp", 1))),
        pcp=int(str(wl.get("pcp", 1))),
        moe_mode=moe_val,
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


def test_validation_corpus_provenance_and_tie_overlap() -> None:
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
    # The bundled trace carries no host launch gap; that input stays a prior.
    assert "launch_gap_us" not in cal_data
    for sub in cal_data.values():
        if isinstance(sub, dict):
            assert "launch_gap_us" not in sub
    assert cal_rep.metrics["launch_gap_us"]["tag"] == "ASSUMED"
    assert cal_rep.metrics["collective_latency_us"]["tag"] == "MEASURED"
