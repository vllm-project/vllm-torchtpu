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
"""Helper utilities for Wan model family tests."""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


@dataclass(frozen=True)
class WanTestConfig:
    """A Wan test config and its CPU golden wan_<config_id>_cpu_golden.npz."""

    config_id: str
    model_id: str  # recorded in the golden meta
    hf_id: str
    task: str
    limits: tuple[float, float, float]  # (MAE max, SSIM min, mean shift max)
    solver: str = "unipc"
    steps: int = 2
    guidance: float = 5.0
    num_frames: int = 17
    height: int = 480
    width: int = 832
    flow_shift: float = 3.0

    @property
    def uses_image(self) -> bool:
        return any(t in self.task for t in ("i2v", "ti2v", "vace"))

    @property
    def path(self) -> str:
        """Grabs local model directory from environment variable or falls back to
        Hugging Face ID.

        CI always uses raw HF IDs via cached $HF_HOME.
        """
        environ_suffix = self.model_id.upper().replace(".", "_").replace("-", "_")
        env_var = f"VLLM_OMNI_TEST_WAN_{environ_suffix}_DIR"
        model_dir_str = os.environ.get(env_var)
        if model_dir_str is None:
            return self.hf_id

        model_dir = Path(model_dir_str)
        if not model_dir.exists():
            raise FileNotFoundError(f"{env_var}={model_dir_str} does not exist")

        return str(model_dir.resolve())


# ==============================================================================
# Golden references: CPU fp32 diffusers videos staged from
# gs://tpu-commons-ci/golden/wan/wan_<config id>_cpu_golden.npz (uint8 video +
# JSON meta). Image-conditioned configs read their input frame
# wan_ref_input_<H>x<W>.png from the same prefix. Each golden uses this test's
# prompts, guidance and flow_shift, seed 42 on a CPU torch.Generator and the
# vllm-omni Wan scheduler (build_wan_scheduler) at 2 steps.
#
# Calibration (TPU v7x vs CPU golden; b1 = dev image 57305db2d,
# b2 = main image 0fcc1f69):
#   config            MAE b1/b2      SSIM b1/b2     mean shift b1/b2
#   t2v-1.3b          .0148/.0165    .9962/.9960    .0046/.0106
#   t2v-14b           .0073/.0075    .9994/.9994    .0051/.0049
#   t2v-1.3b-euler    .0150/.0158    .9957/.9956    .0015/.0061
#   vace-1.3b         .0135/.0138    .9894/.9891    .0040/.0053
#   i2v-14b-480p      .0097/.0094    .9965/.9966    .0018/.0008
#   i2v-14b-720p      .0178/.0158    .9940/.9951    .0164/.0146
#   ti2v-5b           .0080/.0080    .9977/.9977    .0013/.0005
#   i2v-a14b-moe      .0125/.0120    .9947/.9948    .0029/.0020
# HF energy ratio 0.82-1.01 and motion ratio 1.03-1.21 across both builds.
# Limits: MAE and 1 - SSIM at 2x the b1 error, mean shift at b1 + 0.015,
# rounded outward. Both builds pass. Caught: seed 43, a different prompt,
# max_sequence_length=256 and a solid-colour input image (thinnest catch:
# max_sequence_length=256 on ti2v-5b, SSIM 0.9946), and synthetic blur, noise,
# frozen video, +0.03 brightness and R/B swap.
# ==============================================================================
WAN2_1_T2V_1_3B = WanTestConfig(
    "wan2.1-t2v-1.3b",
    "2.1-t2v-1.3b",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "t2v",
    (0.030, 0.992, 0.020),
)
WAN2_1_T2V_14B = WanTestConfig(
    "wan2.1-t2v-14b",
    "2.1-t2v-14b",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "t2v",
    (0.015, 0.998, 0.021),
)
WAN2_1_T2V_1_3B_EULER = replace(
    WAN2_1_T2V_1_3B,
    config_id="wan2.1-t2v-1.3b-euler",
    task="t2v_euler",
    solver="euler",
    limits=(0.030, 0.991, 0.017),
)
WAN2_1_VACE_1_3B = WanTestConfig(
    "wan2.1-vace-1.3b",
    "2.1-vace-1.3b",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "vace",
    (0.027, 0.978, 0.019),
)
WAN2_1_I2V_14B_480P = WanTestConfig(
    "wan2.1-i2v-14b-480p",
    "2.1-i2v-14b-480p",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "i2v",
    (0.020, 0.992, 0.017),
)
WAN2_1_I2V_14B_720P = WanTestConfig(
    "wan2.1-i2v-14b-720p",
    "2.1-i2v-14b-720p",
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    "i2v",
    (0.036, 0.987, 0.032),
    height=720,
    width=1280,
    flow_shift=5.0,
)
WAN2_2_TI2V_5B = WanTestConfig(
    "wan2.2-ti2v-5b",
    "2.2-ti2v-5b",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "ti2v",
    (0.017, 0.995, 0.017),
)
WAN2_2_I2V_A14B_MOE = WanTestConfig(
    "wan2.2-i2v-a14b-moe",
    "2.2-i2v-a14b",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "i2v_moe",
    (0.025, 0.989, 0.018),
)
HF_ENERGY_RATIO_RANGE = (0.70, 1.40)
MOTION_RATIO_RANGE = (0.80, 1.40)


def get_wan_golden_path(filename: str) -> str:
    """Path of a Wan golden file (CPU golden video or reference input frame).

    CI stages gs://tpu-commons-ci/golden/ into $HF_HOME/golden, so the default
    directory is $HF_HOME/golden/wan. Set WAN_GOLDEN_DIR to override it.
    """
    golden_dir = os.environ.get("WAN_GOLDEN_DIR") or os.path.join(
        os.environ.get("HF_HOME", "/local_hf_cache"), "golden", "wan"
    )
    return os.path.join(golden_dir, filename)


def compare_wan_video_to_golden(
    video: np.ndarray, golden: np.ndarray
) -> dict[str, float]:
    """Compare two uint8 [F, H, W, 3] videos with the Flux golden-test metrics.

    Returns the mean per-frame MAE and SSIM (single window, per channel), the
    largest per-channel mean shift, and the video-to-golden ratios of
    high-frequency energy (mean absolute Laplacian: drops with blur, rises with
    noise) and of motion (mean absolute difference between consecutive frames).
    """
    if video.shape != golden.shape:
        raise ValueError(f"video shape {video.shape} != golden {golden.shape}")
    x = video.astype(np.float64) / 255.0
    y = golden.astype(np.float64) / 255.0
    hw = (1, 2)
    mu_x, mu_y = x.mean(hw, keepdims=True), y.mean(hw, keepdims=True)
    cov = ((x - mu_x) * (y - mu_y)).mean(hw, keepdims=True)
    var_x, var_y = x.var(hw, keepdims=True), y.var(hw, keepdims=True)
    c1, c2 = 0.01**2, 0.03**2
    ssim = (
        (2 * mu_x * mu_y + c1)
        * (2 * cov + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    )
    hf_x, hf_y = (
        np.abs(
            4 * v[:, 1:-1, 1:-1]
            - v[:, :-2, 1:-1]
            - v[:, 2:, 1:-1]
            - v[:, 1:-1, :-2]
            - v[:, 1:-1, 2:]
        ).mean()
        for v in (x, y)
    )
    motion_x, motion_y = (np.abs(np.diff(v, axis=0)).mean() for v in (x, y))
    return {
        "mae": float(np.abs(x - y).mean()),
        "ssim": float(ssim.mean()),
        "mean_shift": float(np.abs(mu_x.mean(0) - mu_y.mean(0)).max()),
        "hf_ratio": float(hf_x / max(hf_y, 1e-9)),
        "motion_ratio": float(motion_x / max(motion_y, 1e-9)),
    }


GOLDEN_PROMPT = (
    "A serene cinematic drone shot gliding smoothly over ancient "
    "snow-covered pagodas in the mist, golden sunrise reflections on river ice, "
    "photorealistic, 4k"
)
GOLDEN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, "
    "low quality, JPEG compression residue, ugly, messy background"
)
GOLDEN_SEED = 42


def _get_ref_image_path(config: WanTestConfig) -> str:
    return get_wan_golden_path(f"wan_ref_input_{config.height}x{config.width}.png")


def build_wan_golden_request(
    config: WanTestConfig,
) -> tuple[dict[str, object], OmniDiffusionSamplingParams]:
    """Build inputs and sampling params for a Wan golden test config."""
    inputs: dict[str, object] = {
        "prompt": GOLDEN_PROMPT,
        "negative_prompt": GOLDEN_NEGATIVE_PROMPT,
    }
    if config.uses_image:
        input_image = Image.open(_get_ref_image_path(config)).convert("RGB")
        inputs["multi_modal_data"] = {"image": input_image}

    sampling_params = OmniDiffusionSamplingParams(
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        fps=16.0,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance,
        seed=GOLDEN_SEED,
        generator_device="cpu",  # CPU generator for reproducibility
        extra_args={
            "sample_solver": config.solver,
            "flow_shift": config.flow_shift,
        },
    )
    return inputs, sampling_params


def extract_wan_video(
    outputs: list[OmniRequestOutput],
    expected_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract and validate the (F, H, W, 3) float32 video from Omni outputs.

    One prompt produces one OmniRequestOutput. The Wan post-process function
    returns diffusers VideoProcessor.postprocess_video(output_type="np"), which
    vllm-omni stores as the single entry of `images`: a float32 array of shape
    (batch=1, F, H, W, C) in [0, 1].
    """
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    output = outputs[0]
    assert output.final_output_type == "video", output.final_output_type
    assert len(output.images) == 1, f"Expected 1 video, got {len(output.images)}"
    video: np.ndarray = output.images[0]
    batch_shape = (1, *expected_shape)
    assert video.shape == batch_shape, (
        f"Expected shape {batch_shape}, got {video.shape}"
    )
    video_arr: np.ndarray = video[0]  # (F, H, W, C)
    assert np.all(np.isfinite(video_arr)), "Video contains NaN or Inf values"
    return video_arr


def load_and_verify_wan_golden(config: WanTestConfig) -> np.ndarray:
    """Load the CPU fp32 golden video and verify its metadata matches `config`."""
    golden_path = get_wan_golden_path(f"wan_{config.config_id}_cpu_golden.npz")
    with np.load(golden_path) as data:
        golden, golden_meta = data["video"], json.loads(str(data["meta"]))

    input_image_sha256: str | None = None
    if config.uses_image:
        with open(_get_ref_image_path(config), "rb") as f:
            input_image_sha256 = hashlib.sha256(f.read()).hexdigest()

    golden_config = golden_meta["config"]
    expected_meta = {
        "model": config.model_id,
        "task": config.task,
        "steps": config.steps,
        "guidance": config.guidance,
        "solver": config.solver,
        "num_frames": config.num_frames,
        "height": config.height,
        "width": config.width,
        "flow_shift": config.flow_shift,
        "seed": GOLDEN_SEED,
        "generator_device": "cpu",
        "input_image_sha256": input_image_sha256,
        "prompt": GOLDEN_PROMPT,
        "negative_prompt": GOLDEN_NEGATIVE_PROMPT,
    }
    actual_meta = {
        "model": golden_config["model"],
        "task": golden_config["task"],
        "steps": golden_config["steps"],
        "guidance": golden_config["guidance"],
        "solver": golden_config["solver"],
        "num_frames": golden_config["num_frames"],
        "height": golden_config["height"],
        "width": golden_config["width"],
        "flow_shift": golden_meta["flow_shift"],
        "seed": golden_meta["seed"],
        "generator_device": golden_meta["generator_device"],
        "input_image_sha256": golden_meta["input_image_sha256"],
        "prompt": golden_meta.get("prompt"),
        "negative_prompt": golden_meta.get("negative_prompt"),
    }
    for key, expected_val in expected_meta.items():
        actual_val = actual_meta[key]
        assert actual_val == expected_val, (
            f"Golden for {config.config_id} was generated with {key}={actual_val!r}; "
            f"the test runs {key}={expected_val!r}"
        )
    return golden


def assert_wan_golden_parity(video_arr: np.ndarray, config: WanTestConfig) -> None:
    """Compare `video_arr` against the CPU golden video and assert metric limits."""
    golden = load_and_verify_wan_golden(config)
    # Same uint8 rounding as the golden generator.
    video_u8 = np.round(np.clip(video_arr, 0, 1) * 255).astype(np.uint8)
    golden_metrics = compare_wan_video_to_golden(video_u8, golden)
    mae_max, ssim_min, shift_max = config.limits
    hf_lo, hf_hi = HF_ENERGY_RATIO_RANGE
    motion_lo, motion_hi = MOTION_RATIO_RANGE
    assert golden_metrics["mae"] <= mae_max, (
        f"{config.config_id} MAE {golden_metrics['mae']:.4f} exceeds max {mae_max}"
    )
    assert golden_metrics["ssim"] >= ssim_min, (
        f"{config.config_id} SSIM {golden_metrics['ssim']:.4f} is below min {ssim_min}"
    )
    assert golden_metrics["mean_shift"] <= shift_max, (
        f"{config.config_id} mean shift {golden_metrics['mean_shift']:.4f} "
        f"exceeds max {shift_max}"
    )
    assert hf_lo <= golden_metrics["hf_ratio"] <= hf_hi, (
        f"{config.config_id} HF energy ratio {golden_metrics['hf_ratio']:.3f} "
        f"outside [{hf_lo}, {hf_hi}]"
    )
    assert motion_lo <= golden_metrics["motion_ratio"] <= motion_hi, (
        f"{config.config_id} motion ratio {golden_metrics['motion_ratio']:.3f} "
        f"outside [{motion_lo}, {motion_hi}]"
    )
