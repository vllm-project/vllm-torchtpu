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
"""Shared helpers for the LTX-2 component and nightly test suites."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

# Default error-ratio bounds (omni_rel_l2 / diffusers_cpu_bf16_rel_l2), set ~3%
# above the worst observed ratios across LTX-2 DiT and connector checks
# (fp32: 0.524x-0.651x; bf16: 1.056x-1.135x).
FP32_ERR_RATIO = 0.67
BF16_ERR_RATIO = 1.17
BF16_ERR_FLOOR = 1e-3


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compute cosine similarity, relative L2 error, and max absolute error in fp64."""
    actual_flat = actual.detach().cpu().double().flatten()
    expected_flat = expected.detach().cpu().double().flatten()
    diff = actual_flat - expected_flat
    norm_product = actual_flat.norm() * expected_flat.norm()
    cos = torch.dot(actual_flat, expected_flat) / norm_product.clamp_min(1e-30)
    return {
        "cos": float(cos),
        "rel_l2": float(diff.norm() / expected_flat.norm().clamp_min(1e-30)),
        "max_abs": float(diff.abs().max()),
    }


def assert_parity(
    *,
    ref: torch.Tensor,
    ref_bf16: torch.Tensor,
    tpu_fp32: torch.Tensor,
    tpu_bf16: torch.Tensor,
    fp32_ratio: float = FP32_ERR_RATIO,
    bf16_ratio: float = BF16_ERR_RATIO,
    floor: float = BF16_ERR_FLOOR,
) -> dict[str, object]:
    """Assert TPU fp32 and bf16 outputs match the CPU fp32 reference.

    Both TPU error budgets are normalized against the CPU bf16 reference error
    so depth and conditioning sensitivity cancel out across components.
    """
    ref_bf16_metrics = metrics(ref_bf16, ref)
    tpu_fp32_metrics = metrics(tpu_fp32, ref)
    tpu_bf16_metrics = metrics(tpu_bf16, ref)
    ref_bf16_err = ref_bf16_metrics["rel_l2"]
    denom = max(ref_bf16_err, 1e-30)
    results: dict[str, object] = {
        "diffusers_cpu_bf16": ref_bf16_metrics,
        "omni_tpu_fp32": tpu_fp32_metrics,
        "omni_tpu_bf16": tpu_bf16_metrics,
        "fp32_ratio": tpu_fp32_metrics["rel_l2"] / denom,
        "bf16_ratio": tpu_bf16_metrics["rel_l2"] / denom,
    }
    fp32_budget = max(fp32_ratio * ref_bf16_err, floor)
    bf16_budget = max(bf16_ratio * ref_bf16_err, floor)
    assert tpu_fp32_metrics["rel_l2"] <= fp32_budget, results
    assert tpu_bf16_metrics["rel_l2"] <= bf16_budget, results
    return results


def assert_all_loaded(loaded: set[str], module: torch.nn.Module) -> None:
    """Assert that every parameter of ``module`` was populated by ``load_weights``."""
    missing = {name for name, _ in module.named_parameters()} - loaded
    assert not missing, (
        f"vllm-omni parameters not loaded from checkpoint: {sorted(missing)[:10]}"
    )


LTX2_MODEL_ID = "Lightricks/LTX-2"
LTX2_PROMPT = (
    "A golden retriever runs across a sunny meadow toward the camera and barks "
    "twice, birds chirping in the background, cinematic lighting."
)
LTX2_DEFAULT_FRAMES = 25
LTX2_DEFAULT_HEIGHT = 512
LTX2_DEFAULT_WIDTH = 768
LTX2_DEFAULT_FRAME_RATE = 24.0
LTX2_DEFAULT_STEPS = 2
LTX2_DEFAULT_SEED = 42
LTX2_AUDIO_SAMPLE_RATE = 24000
# 25 video frames at 24 fps -> 26 audio latent frames -> 4 * (26 - 1) + 1 = 101
# causal Audio VAE mel frames -> 101 * 240 vocoder upsample = 24240 samples/channel.
LTX2_DEFAULT_AUDIO_SAMPLES = 24240


@dataclass(frozen=True)
class LTX2Outputs:
    """Validated video and audio outputs from a single LTX-2 generation request."""

    video: np.ndarray
    audio: np.ndarray
    sample_rate: int
    fps: float


def get_ltx2_model() -> str:
    """Resolve the LTX-2 checkpoint path from VLLM_OMNI_TEST_LTX2_DIR or HF Hub."""
    model_dir = os.environ.get("VLLM_OMNI_TEST_LTX2_DIR")
    if model_dir is None:
        return LTX2_MODEL_ID
    model_path = Path(model_dir).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"VLLM_OMNI_TEST_LTX2_DIR={model_dir} does not exist or is not a directory"
        )
    return str(model_path)


def build_ltx2_request(
    seed: int = LTX2_DEFAULT_SEED,
    **overrides: object,
) -> OmniDiffusionSamplingParams:
    """Build the canonical LTX-2 one-stage T2V sampling parameters."""
    params: dict[str, object] = {
        "height": LTX2_DEFAULT_HEIGHT,
        "width": LTX2_DEFAULT_WIDTH,
        "num_frames": LTX2_DEFAULT_FRAMES,
        "frame_rate": LTX2_DEFAULT_FRAME_RATE,
        "num_inference_steps": LTX2_DEFAULT_STEPS,
        "seed": seed,
        "generator_device": "cpu",
    }
    params.update(overrides)
    return OmniDiffusionSamplingParams(**params)


def extract_ltx2_outputs(
    outputs: list[OmniRequestOutput],
    expected_video_shape: tuple[int, int, int, int] = (
        LTX2_DEFAULT_FRAMES,
        LTX2_DEFAULT_HEIGHT,
        LTX2_DEFAULT_WIDTH,
        3,
    ),
    expected_audio_samples: int = LTX2_DEFAULT_AUDIO_SAMPLES,
    expected_fps: float = LTX2_DEFAULT_FRAME_RATE,
) -> LTX2Outputs:
    """Extract and validate video and audio arrays from LTX-2 Omni outputs.

    ``get_ltx2_post_process_func`` returns a 5D ``np.uint8`` video array of
    shape ``(batch=1, F, H, W, 3)`` (stored as ``output.images[0]``), a 3D CPU
    ``torch.bfloat16`` audio tensor of shape ``(batch=1, 2, N)`` in
    ``output.multimodal_output["audio"]``, and
    ``output.multimodal_output["audio_sample_rate"] == 24000``.
    """
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    output = outputs[0]
    assert output.final_output_type == "video", output.final_output_type
    assert len(output.images) == 1, f"Expected 1 video, got {len(output.images)}"

    video_batch: np.ndarray = output.images[0]
    assert video_batch.dtype == np.uint8, (
        f"Expected uint8 video, got {video_batch.dtype}"
    )
    batch_video_shape = (1, *expected_video_shape)
    assert video_batch.shape == batch_video_shape, (
        f"Expected video shape {batch_video_shape}, got {video_batch.shape}"
    )
    video: np.ndarray = video_batch[0]
    assert np.all(np.isfinite(video)), "Video contains non-finite values"

    mm_output = output.multimodal_output
    audio_tensor: torch.Tensor = mm_output["audio"]
    assert audio_tensor.device.type == "cpu", (
        f"Expected CPU audio tensor, got {audio_tensor.device}"
    )
    batch_audio_shape = (1, 2, expected_audio_samples)
    assert audio_tensor.shape == batch_audio_shape, (
        f"Expected audio shape {batch_audio_shape}, got {tuple(audio_tensor.shape)}"
    )
    audio: np.ndarray = audio_tensor[0].to(torch.float32).numpy()
    assert audio.dtype == np.float32, f"Expected float32 audio, got {audio.dtype}"
    assert audio.shape == (2, expected_audio_samples), (
        f"Expected audio shape (2, {expected_audio_samples}), got {audio.shape}"
    )
    assert np.all(np.isfinite(audio)), "Audio contains non-finite values"

    sample_rate: int = mm_output["audio_sample_rate"]
    assert sample_rate == LTX2_AUDIO_SAMPLE_RATE, (
        f"Expected sample_rate {LTX2_AUDIO_SAMPLE_RATE}, got {sample_rate}"
    )
    fps = float(mm_output.get("fps", expected_fps))
    assert fps == expected_fps, f"Expected fps {expected_fps}, got {fps}"

    return LTX2Outputs(
        video=video,
        audio=audio,
        sample_rate=sample_rate,
        fps=fps,
    )
