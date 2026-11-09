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
"""Nightly build tests for FLUX models on TPU v7x.

Focused integration and numerical correctness tests for nightly builds (~5 minutes).
Focuses on accuracy of common paths: tri-model matrix validation (FLUX.1-schnell,
FLUX.1-dev, FLUX.2-klein-4B) with pixel-statistics sanity checks.
All tests run in-memory without PNG writes.
"""

import numpy as np
import pytest
from PIL import Image
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


def compute_distribution_metrics(img: Image.Image) -> dict[str, float]:
    """Calculate scalar distribution moments and percentiles from output pixels."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "var": float(np.var(arr)),
    }


class TestFluxNightlyOnTpu:
    """Nightly regression test suite for FLUX models on TPU v7x."""

    @pytest.mark.parametrize(
        "omni_runner,steps,guidance",
        [
            ("black-forest-labs/FLUX.1-schnell", 4, 0.0),
            ("black-forest-labs/FLUX.1-dev", 10, 3.5),
            ("black-forest-labs/FLUX.2-klein-4B", 4, 1.0),
        ],
        indirect=["omni_runner"],
    )
    def test_flux_tri_model_matrix_tpu(
        self, omni_runner: OmniRunner, steps: int, guidance: float
    ):
        """Validate common path execution across the three FLUX variants on TPU v7x.

        Covers distilled guidance-free Schnell, 10-step guided Dev, and Klein-4B.
        """
        prompt = (
            "A futuristic glowing crystal monolith in a misty desert at night, "
            "neon reflections"
        )

        sampling_params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=42,
        )

        outputs = omni_runner.generate(prompt, sampling_params)
        assert outputs, f"Failed generation for model {omni_runner.model_name}"

        img = _extract_image(outputs[0])
        assert isinstance(img, Image.Image)
        assert img.size == (512, 512)

        metrics = compute_distribution_metrics(img)
        assert 0.10 < metrics["mean"] < 0.90, (
            f"Unnatural pixel mean on {omni_runner.model_name}: {metrics['mean']:.4f}"
        )
        assert metrics["var"] > 0.01, (
            f"Image variance too low on {omni_runner.model_name} ({metrics['var']:.4f})"
        )
        assert metrics["p90"] > metrics["p10"], (
            f"Collapsed dynamic range on {omni_runner.model_name}"
        )
