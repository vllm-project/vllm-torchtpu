# SPDX-License-Identifier: Apache-2.0
"""Nightly build tests for Qwen-Image models on TPU v7x.

Focused integration and numerical correctness tests for nightly builds.
Focuses on accuracy of common paths: Qwen-Image-Edit, Qwen-Image-Edit-2511,
and Qwen-Image.
All tests run in-memory without PNG writes.
"""

import os

import numpy as np
import pytest
from PIL import Image
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_image_prompt, get_model_class_name

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

# ==============================================================================
# Pixel statistics on TPU v7x, torch_tpu ...release.2026.9.22 (CI pin),
# 512x512, 10 steps, seed 42. HF = mean abs Laplacian.
# Pearson = 8x8 thumbnail vs the input image. Red = fraction of pixels with
# R > 0.5 and R - max(G, B) > 0.25. Pearson and Red are computed for edit rows.
# Model                            Mean   Var    HF      Pearson  Red
# Qwen/Qwen-Image-Edit             0.211  0.037  0.010   0.600    0.028
# Qwen/Qwen-Image-Edit-2511        0.248  0.021  0.010   0.569    0.031
# Qwen/Qwen-Image                  0.130  0.033  0.010   N/A      N/A
#
# Offline must-fail controls:
# All-black image                  0.000  0.000  0.000   NaN      0.000
# Uniform gray 0.5                 0.500  0.000  0.000   NaN      0.000
# U(0,1) noise                     0.500  0.083  1.084  -0.048    0.135
# Edit output + N(0,0.1) clipped   0.265  0.041  0.343   0.551    0.046
# Input passthrough                0.233  0.021  0.010   1.000    0.000
# T2I scored as edit               0.130  0.033  0.010  -0.083    0.000
# Old torch_tpu (ignores input)    0.903  0.009  0.004  -0.120    0.029
#
# Catches: all-black, uniform gray 0.5, U(0,1) noise, older torch_tpu
MEAN_RANGE = (0.05, 0.40)
# Catches: all-black, uniform gray 0.5, older torch_tpu
VAR_MIN = 0.01
# Catches: U(0,1) noise, Edit output + N(0,0.1) clipped
HF_MAX = 0.03
# Catches: U(0,1) noise, T2I scored as edit, older torch_tpu
EDIT_PEARSON_MIN = 0.30
# Catches: all-black, uniform gray 0.5, input passthrough, T2I scored as edit
EDIT_RED_MIN = 0.01
# ==============================================================================


def _load_input_image() -> Image.Image:
    """Load the Qwen Image Edit input image staged by CI."""
    path = os.path.join(
        os.environ.get("HF_HOME", "/local_hf_cache"),
        "golden",
        "qwen_image",
        "qwen_image_edit_input.png",
    )
    return Image.open(path).convert("RGB")


def _high_freq_energy(img: np.ndarray) -> float:
    """Mean absolute Laplacian: drops with blur, rises with noise or blocking."""
    laplacian = (
        4 * img[1:-1, 1:-1]
        - img[:-2, 1:-1]
        - img[2:, 1:-1]
        - img[1:-1, :-2]
        - img[1:-1, 2:]
    )
    return float(np.abs(laplacian).mean())


def compute_distribution_metrics(
    img: Image.Image, input_img: Image.Image | None = None
) -> dict[str, float]:
    """Calculate scalar distribution moments and metrics from output pixels."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    metrics = {
        "mean": float(np.mean(arr)),
        "var": float(np.var(arr)),
        "hf": _high_freq_energy(arr),
    }
    if input_img is not None:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        is_red = (r > 0.5) & (r - np.maximum(g, b) > 0.25)
        metrics["red"] = float(is_red.mean())

        img_thumb = img.resize((8, 8), Image.Resampling.LANCZOS)
        in_thumb = input_img.resize((8, 8), Image.Resampling.LANCZOS)

        img_arr = np.asarray(img_thumb.convert("RGB"), dtype=np.float32)
        img_arr = img_arr.flatten() / 255.0

        in_arr = np.asarray(in_thumb.convert("RGB"), dtype=np.float32)
        in_arr = in_arr.flatten() / 255.0

        metrics["pearson"] = float(np.corrcoef(img_arr, in_arr)[0, 1])
    return metrics


class TestQwenImageNightlyOnTpu:
    """Nightly regression test suite for Qwen-Image models on TPU v7x."""

    @pytest.mark.parametrize(
        "model_id,mode",
        [
            ("Qwen/Qwen-Image-Edit", "edit"),
            ("Qwen/Qwen-Image-Edit-2511", "edit"),
            ("Qwen/Qwen-Image", "t2i"),
        ],
    )
    def test_qwen_image_matrix_tpu(self, model_id: str, mode: str):
        """Validate common path execution across the three Qwen variants on TPU v7x."""
        input_image = _load_input_image()

        omni = Omni(model_id, init_timeout=600, stage_init_timeout=600)
        try:
            if mode == "edit":
                prompt = "Change the latte art into a red heart"
                prompt_obj = build_image_to_image_prompt(
                    model_class_name=get_model_class_name(omni),
                    prompt=prompt,
                    negative_prompt=" ",
                    input_image=input_image,
                    height=512,
                    width=512,
                )
            else:
                prompt = (
                    "A futuristic glowing crystal monolith in a misty desert at night, "
                    "neon reflections"
                )
                prompt_obj = {"prompt": prompt, "negative_prompt": " "}

            sampling_params = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=10,
                true_cfg_scale=4.0,
                seed=42,
            )

            outputs = omni.generate(prompt_obj, [sampling_params])
            assert outputs, f"Failed generation for model {model_id}"

            img = outputs[0].images[0]
            assert isinstance(img, Image.Image)
            assert img.size == (512, 512)

            metrics = compute_distribution_metrics(
                img, input_image if mode == "edit" else None
            )

            # Format and print the summary exactly once
            lo, hi = MEAN_RANGE
            if mode == "edit":
                summary = (
                    f"[{model_id}] Mean={metrics['mean']:.3f} (range {lo}-{hi}), "
                    f"Var={metrics['var']:.3f} (min {VAR_MIN}), "
                    f"HF={metrics['hf']:.3f} (max {HF_MAX}), "
                    f"Pearson={metrics['pearson']:.3f} (min {EDIT_PEARSON_MIN}), "
                    f"Red={metrics['red']:.3f} (min {EDIT_RED_MIN})"
                )
            else:
                summary = (
                    f"[{model_id}] Mean={metrics['mean']:.3f} (range {lo}-{hi}), "
                    f"Var={metrics['var']:.3f} (min {VAR_MIN}), "
                    f"HF={metrics['hf']:.3f} (max {HF_MAX})"
                )
            print(summary)

            assert lo <= metrics["mean"] <= hi, f"Mean outside tolerance: {summary}"
            assert metrics["var"] >= VAR_MIN, f"Variance below minimum: {summary}"
            assert metrics["hf"] <= HF_MAX, f"HF energy too high: {summary}"

            if mode == "edit":
                msg_p = f"Thumbnail Pearson too low: {summary}"
                assert metrics["pearson"] >= EDIT_PEARSON_MIN, msg_p
                msg_r = f"Red presence too low: {summary}"
                assert metrics["red"] >= EDIT_RED_MIN, msg_r

        finally:
            omni.shutdown()
