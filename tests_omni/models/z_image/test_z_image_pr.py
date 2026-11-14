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
"""Smoke tests for the Z-Image model family on TPU (temporarily marked ``nightly``).

Single-engine end-to-end smoke test for Z-Image-Turbo (512x512, 4 steps) plus
deterministic seed parity, without file writes. The model is resolved through
the HF cache (``$HF_HOME/hub``) like the FLUX PR tests.

Temporarily marked ``nightly`` so the per-PR omni step (``-m 'not nightly'``)
does not download Z-Image-Turbo from Hugging Face on every PR.
"""

import numpy as np
import pytest
from PIL import Image
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner

pytestmark = [
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
    # TODO(rishabhmanoj): drop once Tongyi-MAI/Z-Image-Turbo is cached in
    # $HF_HOME/hub on CI runners so this can run in the per-PR omni step.
    pytest.mark.nightly,
]

Z_IMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


@pytest.mark.parametrize("omni_runner", [Z_IMAGE_TURBO], indirect=True)
class TestZImagePrCiOnTpu:
    """Fast PR CI functional and numerical tests for Z-Image-Turbo on TPU."""

    def test_z_image_turbo_fast_t2i_tpu(self, omni_runner: OmniRunner):
        """Test Z-Image-Turbo fast generation and deterministic seed parity on TPU."""
        prompt = (
            "A cinematic portrait of a red panda astronaut in space, "
            "detailed fur, soft rim light, 4k"
        )
        params1 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            seed=42,
        )
        out1 = omni_runner.generate(prompt, params1)
        assert out1, "Omni returned empty output list"
        img1 = _extract_image(out1[0])
        assert isinstance(img1, Image.Image)
        assert img1.size == (512, 512)

        arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
        assert 0.05 < float(np.mean(arr1)) < 0.95, (
            f"Suspicious image mean: {np.mean(arr1):.4f}"
        )
        assert float(np.var(arr1)) > 0.005, (
            f"Suspiciously low variance ({np.var(arr1):.4f}), likely blank"
        )

        # Replay with identical seed must match bitwise.
        out2 = omni_runner.generate(prompt, params1)
        img2 = _extract_image(out2[0])
        arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0
        np.testing.assert_array_equal(
            arr1,
            arr2,
            err_msg="Identical seed 42 on TPU must produce a bitwise pixel match",
        )

        # Distinct seed must diverge.
        params3 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            seed=99999,
        )
        out3 = omni_runner.generate(prompt, params3)
        img3 = _extract_image(out3[0])
        arr3 = np.asarray(img3.convert("RGB"), dtype=np.float32) / 255.0
        assert not np.array_equal(arr1, arr3), (
            "Different seeds should produce divergent pixel values"
        )
