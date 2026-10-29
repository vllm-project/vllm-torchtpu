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
"""Shared helpers for the LTX-2 nightly test suite."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTX_DEFAULT_NEGATIVE_PROMPT
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

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


# Hugging Face revision of the Lightricks/LTX-2 checkpoint the goldens use.
LTX2_REVISION = "12beb2ad8ac6fb6fb73efbdd6dc2eb8ce3dc44ef"
# Guidance of the golden request: vllm-omni's default 4-pass guidance (CFG, STG
# at block 29, modality isolation, rescale), sent explicitly so the golden does
# not depend on vllm-omni's defaults.
LTX2_GOLDEN_VARIANT = "full_default"
LTX2_GOLDEN_EXTRA_ARGS = {
    "video_cfg_scale": 3.0,
    "audio_cfg_scale": 7.0,
    "video_stg_scale": 1.0,
    "audio_stg_scale": 1.0,
    "video_modality_scale": 3.0,
    "audio_modality_scale": 3.0,
    "video_rescale_scale": 0.7,
    "audio_rescale_scale": 0.7,
}
# Metric -> (min, max) for TPU output vs the CPU fp32 golden. Three TPU v7x runs
# were bitwise identical: video MAE 0.0077, SSIM 0.9903, mean shift 0.0045, HF
# ratio 1.015, motion ratio 0.964, audio log-spectrogram L1 0.563, RMS ratio
# 1.092. MAE, 1 - SSIM and log-spectrogram L1 allow 2x that error, mean shift
# +0.015. Must-fail goldens (seed 43, 3 steps, another prompt, CFG-only
# guidance) fail on MAE (>= 0.0237), SSIM (<= 0.926) and the RMS ratio.
LTX2_GOLDEN_LIMITS = {
    "video_mae": (0.0, 0.0155),
    "video_ssim": (0.980, 1.0),
    "video_mean_shift": (0.0, 0.0195),
    "video_hf_ratio": (0.70, 1.40),
    "video_motion_ratio": (0.80, 1.40),
    "audio_logspec_l1": (0.0, 1.13),
    "audio_rms_ratio": (0.75, 1.35),
}


def get_ltx2_golden_path() -> Path:
    """Path of the CPU fp32 golden for ``LTX2_GOLDEN_VARIANT``.

    CI stages gs://tpu-commons-ci/golden/ into $HF_HOME/golden, so the default
    directory is $HF_HOME/golden/ltx2. Set LTX2_GOLDEN_DIR to override it.
    """
    golden_dir = os.environ.get("LTX2_GOLDEN_DIR") or os.path.join(
        os.environ.get("HF_HOME", "/local_hf_cache"), "golden", "ltx2"
    )
    return Path(golden_dir) / f"ltx2_t2v_{LTX2_GOLDEN_VARIANT}_cpu_golden.npz"


def load_and_verify_ltx2_golden() -> tuple[np.ndarray, np.ndarray]:
    """Load the golden video and audio and check its metadata matches the test."""
    with np.load(get_ltx2_golden_path()) as data:
        video, audio = data["video"], data["audio"]
        meta = json.loads(str(data["meta"]))
    expected_meta = {
        "model_id": LTX2_MODEL_ID,
        "revision": LTX2_REVISION,
        "variant": LTX2_GOLDEN_VARIANT,
        "omni_extra_args": LTX2_GOLDEN_EXTRA_ARGS,
        "prompt": LTX2_PROMPT,
        "negative_prompt_sha256": hashlib.sha256(
            LTX_DEFAULT_NEGATIVE_PROMPT.encode()
        ).hexdigest(),
        "seed": LTX2_DEFAULT_SEED,
        "generator_device": "cpu",
        "height": LTX2_DEFAULT_HEIGHT,
        "width": LTX2_DEFAULT_WIDTH,
        "num_frames": LTX2_DEFAULT_FRAMES,
        "frame_rate": LTX2_DEFAULT_FRAME_RATE,
        "steps": LTX2_DEFAULT_STEPS,
    }
    for key, expected_val in expected_meta.items():
        assert meta[key] == expected_val, (
            f"Golden was generated with {key}={meta[key]!r}; "
            f"the test runs {key}={expected_val!r}. Regenerate the golden."
        )
    return video, audio


def compare_ltx2_to_golden(
    out: LTX2Outputs, golden_video: np.ndarray, golden_audio: np.ndarray
) -> dict[str, float]:
    """Compare LTX-2 outputs with a golden video and audio.

    Video (uint8 [F, H, W, 3]): mean per-frame MAE and SSIM (single window, per
    channel), the largest per-channel mean shift, and the output-to-golden
    ratios of high-frequency energy (mean absolute Laplacian: drops with blur,
    rises with noise) and of motion (mean absolute frame-to-frame difference).
    Audio (float32 [2, N]): mean absolute log-spectrogram difference and RMS
    ratio. Waveform metrics are not used: bf16 alone (CPU bf16 vs CPU fp32)
    moves the waveform by rel-L2 ~1.0 (cosine ~0.4) at 2 steps.
    """
    assert out.video.shape == golden_video.shape, (out.video.shape, golden_video.shape)
    assert out.audio.shape == golden_audio.shape, (out.audio.shape, golden_audio.shape)
    x = out.video.astype(np.float64) / 255.0
    y = golden_video.astype(np.float64) / 255.0
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
    logspec_x, logspec_y = (_log_spectrogram(a) for a in (out.audio, golden_audio))
    rms_x, rms_y = (
        np.sqrt(np.mean(a.astype(np.float64) ** 2)) for a in (out.audio, golden_audio)
    )
    return {
        "video_mae": float(np.abs(x - y).mean()),
        "video_ssim": float(ssim.mean()),
        "video_mean_shift": float(np.abs(mu_x.mean(0) - mu_y.mean(0)).max()),
        "video_hf_ratio": float(hf_x / max(hf_y, 1e-9)),
        "video_motion_ratio": float(motion_x / max(motion_y, 1e-9)),
        "audio_logspec_l1": float((logspec_x - logspec_y).abs().mean()),
        "audio_rms_ratio": float(rms_x / max(rms_y, 1e-12)),
    }


def _log_spectrogram(audio: np.ndarray) -> torch.Tensor:
    """Per-channel log-magnitude STFT (1024-sample Hann window, hop 256)."""
    spec = torch.stft(
        torch.from_numpy(audio).double(),
        n_fft=1024,
        hop_length=256,
        window=torch.hann_window(1024, dtype=torch.float64),
        return_complex=True,
    )
    return torch.log(spec.abs() + 1e-5)


def assert_ltx2_golden_parity(out: LTX2Outputs) -> None:
    """Assert every golden metric of ``out`` is within ``LTX2_GOLDEN_LIMITS``."""
    golden_video, golden_audio = load_and_verify_ltx2_golden()
    results = compare_ltx2_to_golden(out, golden_video, golden_audio)
    for name, (low, high) in LTX2_GOLDEN_LIMITS.items():
        assert low <= results[name] <= high, (
            f"{name}={results[name]:.4f} outside [{low}, {high}]; metrics: {results}"
        )
