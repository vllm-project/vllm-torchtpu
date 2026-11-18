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
"""TPU nightly test for compare_diffusion_trajectory_similarity tool."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from vllm_omni.quantization.tools import compare_diffusion_trajectory_similarity

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "compare_diffusion_trajectory_similarity._build_omni_kwargs passes "
        "unsupported 'mode' kwarg to Omni(), raising Unknown diffusion config "
        "field(s) for stage 0: 'mode' in vllm-omni 0.32.0rc1"
    ),
)
def test_diffusion_trajectory_similarity_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diffusion Trajectory Similarity Tool.

    - TPU path: Run ``compare_diffusion_trajectory_similarity.main()`` on
      ``tiny_flux_schnell()`` (256x256, 2 steps, seed 42, ``measure_runs=1``)
      comparing two inline single-device ``Omni`` runs on TPU.
    - Setup check: Assert output JSON and saved reference/candidate PNG files
      exist, ``variant_configs["reference"]["model"] == model_dir``, and both
      variants record 1 positive ``generation_time_s`` sample.
    - Outcome check: Assert exact self-similarity in
      ``output_metrics["all_images_metrics"]`` (``mae == 0.0``,
      ``relative_l2 == 0.0``, ``cosine_similarity`` within ``1e-6`` of ``1.0``,
      and ``math.isinf(psnr_db)``).
    """
    model_dir = tiny_flux_schnell()
    output_json = tmp_path / "similarity.json"
    save_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diffusion_trajectory_similarity",
            "--task",
            "t2i",
            "--model",
            model_dir,
            "--width",
            "256",
            "--height",
            "256",
            "--num-inference-steps",
            "2",
            "--guidance-scale",
            "0.0",
            "--seed",
            "42",
            "--measure-runs",
            "1",
            "--output-json",
            str(output_json),
            "--save-output-dir",
            str(save_dir),
        ],
    )
    compare_diffusion_trajectory_similarity.main()

    assert output_json.exists(), f"Output JSON not written at {output_json}"
    summary = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary["variant_configs"]["reference"]["model"] == model_dir, (
        f"Expected reference model {model_dir!r}, "
        f"got {summary['variant_configs']['reference']['model']!r}"
    )
    ref_times = summary["reference_generation"]["per_run_generation_time_s"]
    cand_times = summary["candidate_generation"]["per_run_generation_time_s"]
    assert len(ref_times) == 1, f"Expected 1 reference time, got {len(ref_times)}"
    assert ref_times[0] > 0.0, f"Expected positive reference time, got {ref_times[0]}"
    assert len(cand_times) == 1, f"Expected 1 candidate time, got {len(cand_times)}"
    assert cand_times[0] > 0.0, f"Expected positive candidate time, got {cand_times[0]}"

    ref_saved = Path(summary["saved_outputs"]["reference"][0])
    cand_saved = Path(summary["saved_outputs"]["candidate"][0])
    assert ref_saved.exists(), f"Missing saved reference output {ref_saved}"
    assert cand_saved.exists(), f"Missing saved candidate output {cand_saved}"

    metrics = summary["output_metrics"]["all_images_metrics"]
    assert metrics["mae"] == 0.0, (
        f"Expected self-similarity mae == 0.0, got {metrics['mae']}"
    )
    assert metrics["relative_l2"] == 0.0, (
        f"Expected self-similarity relative_l2 == 0.0, got {metrics['relative_l2']}"
    )
    assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected cosine_similarity ~= 1.0, got {metrics['cosine_similarity']}"
    )
    assert math.isinf(metrics["psnr_db"]), (
        f"Expected infinite psnr_db for identical outputs, got {metrics['psnr_db']}"
    )
