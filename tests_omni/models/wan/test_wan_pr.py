# SPDX-License-Identifier: Apache-2.0
"""PR CI tests for Wan 2.1 model family on TPU v7x."""

import numpy as np
import pytest

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Wan tests")

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

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


def _extract_video(output: OmniRequestOutput) -> np.ndarray:
    """Return a Wan video output as a float32 (F, H, W, C) array in [0, 1].

    The Wan post-process func returns diffusers
    VideoProcessor.postprocess_video(output_type="np"), which vllm-omni stores
    as the single entry of `images`: a float32 array of shape
    (batch=1, F, H, W, C) in [0, 1].
    """
    assert output.final_output_type == "video", output.final_output_type
    assert len(output.images) == 1, f"Expected 1 video, got {len(output.images)}"
    video: np.ndarray = output.images[0]
    assert video.ndim == 5 and video.shape[0] == 1, f"Bad shape {video.shape}"
    return video[0]


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
    video1 = _extract_video(out1[0])

    # Run 2 with identical seed
    params2 = _build_diffusion_sampling_params(fixed_seed)
    out2 = omni_runner.generate(prompt, params2)
    video2 = _extract_video(out2[0])

    # Run 3 with distinct seed
    params3 = _build_diffusion_sampling_params(99999)
    out3 = omni_runner.generate(prompt, params3)
    video3 = _extract_video(out3[0])

    for video in (video1, video2, video3):
        assert video.shape == expected_shape, f"Unexpected shape {video.shape}"
        assert np.all(np.isfinite(video)), "Video contains NaN or Inf values"

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
    video = extract_wan_video(outputs, config)
    assert_wan_golden_parity(video, config)
