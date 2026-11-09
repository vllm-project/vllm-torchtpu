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
"""PR CI tests for Wan 2.1 model family on TPU v7x."""

import numpy as np
import pytest

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Wan tests")

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.models.wan.utils import (
    WAN2_1_T2V_1_3B,
    WAN2_1_T2V_1_3B_EULER,
    WanTestConfig,
    assert_wan_golden_parity,
    build_wan_golden_request,
    extract_wan_video,
)

pytestmark = [
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]


def _build_diffusion_sampling_params(seed: int) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_frames=5,
        num_inference_steps=2,
        guidance_scale=5.0,
        seed=seed,
        extra_args={"sample_solver": "unipc"},
    )


@pytest.mark.parametrize(
    "omni_runner",
    [WAN2_1_T2V_1_3B.path],
    indirect=True,
)
def test_wan_deterministic_seed_parity_tpu(omni_runner: OmniRunner):
    """Test that identical seeds produce bitwise identical videos on TPU v7x."""
    prompt = "A serene mountain lake at sunrise with reflections"
    fixed_seed = 42
    # Wan requires 4k + 1 frames (VAE temporal compression factor 4).
    num_frames = 5
    expected_shape = (num_frames, 256, 256, 3)

    # Run 1 with fixed seed
    params1 = _build_diffusion_sampling_params(fixed_seed)
    out1 = omni_runner.generate(prompt, params1)
    video1 = extract_wan_video(out1, expected_shape)

    # Run 2 with identical seed
    params2 = _build_diffusion_sampling_params(fixed_seed)
    out2 = omni_runner.generate(prompt, params2)
    video2 = extract_wan_video(out2, expected_shape)

    # Run 3 with distinct seed
    params3 = _build_diffusion_sampling_params(99999)
    out3 = omni_runner.generate(prompt, params3)
    video3 = extract_wan_video(out3, expected_shape)

    # Identical seeds must match exactly bitwise across all frames
    np.testing.assert_array_equal(
        video1,
        video2,
        err_msg="Identical seed 42 on TPU v7x must produce exact bitwise video match",
    )

    # Distinct seed must diverge
    assert not np.array_equal(video1, video3), (
        "Different seeds should produce divergent videos"
    )


@pytest.mark.parametrize(
    "omni_runner,config",
    [
        pytest.param(c.path, c, id=c.config_id)
        for c in (WAN2_1_T2V_1_3B, WAN2_1_T2V_1_3B_EULER)
    ],
    indirect=["omni_runner"],
)
def test_wan_golden_answers_tpu(omni_runner: OmniRunner, config: WanTestConfig):
    """Verify Wan 2.1 1.3B output against CPU fp32 golden reference videos."""
    inputs, sampling_params = build_wan_golden_request(config)
    outputs = list(omni_runner.generate(inputs, sampling_params))
    video = extract_wan_video(
        outputs, (config.num_frames, config.height, config.width, 3)
    )
    assert_wan_golden_parity(video, config)
