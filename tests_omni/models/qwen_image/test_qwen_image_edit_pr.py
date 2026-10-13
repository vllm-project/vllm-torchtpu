# SPDX-License-Identifier: Apache-2.0
"""PR CI running tests for Qwen-Image-Edit models on TPU v7x.

End-to-end tests designed to run in PR CI on TPU v7x machines.
Covers fast inference with Qwen-Image-Edit, golden-image parity with a CPU diffusers
reference, and deterministic seed parity, without file writes.
"""

import os

import numpy as np
import pytest
from PIL import Image
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import build_image_to_image_prompt, get_model_class_name

from tests_omni.omni_test_utils import compute_ssim, high_freq_energy

pytestmark = [
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

# ==============================================================================
# Golden reference: full 512x512 CPU diffusers reference image staged from
# gs://tpu-commons-ci/golden/qwen_image/qwen_image_edit_cpu_golden_img.npy.
# Regenerate with diffusers 0.40.0 on CPU:
#
#   import torch
#   import numpy as np
#   from PIL import Image
#   from diffusers import QwenImageEditPipeline
#
#   torch.set_num_threads(112)
#   pipe = QwenImageEditPipeline.from_pretrained(
#       "Qwen/Qwen-Image-Edit",
#       revision="ac7f9318f633fc4b5778c59367c8128225f1e3de",
#       torch_dtype=torch.float32,
#   )
#   pipe.to("cpu")
#
#   input_image = Image.open("qwen_image_edit_input.png").convert("RGB")
#   generator = torch.Generator(device="cpu").manual_seed(0)
#
#   result = pipe(
#       prompt="Change the latte art into a red heart and make the table white marble",
#       image=input_image,
#       negative_prompt=" ",
#       true_cfg_scale=4.0,
#       height=512,
#       width=512,
#       num_inference_steps=4,
#       generator=generator,
#       output_type="np",
#   )
#   np.save("qwen_image_edit_cpu_golden_img.npy", result.images[0], allow_pickle=False)
# ==============================================================================

# Full-image comparison metrics (calibrated against CPU reference across 512x512x3):
# Image vs CPU ref                               MAE    SSIM  Pearson  HF     MeanDif
# TPU v7x, torch_tpu ...release.2026.9.22 (CI)  0.0493 0.9527  0.9717  1.041  0.0380
# TPU v7x, ...dev20260912 (ignores input img)   0.1719 0.1957  0.2187  0.522  0.1523
# CPU same-recipe rerun (112 threads)           0.0020 0.9998  0.9998  0.999  0.0003
# CPU 56 vs 112 threads                         0.0014 0.9999  0.9998  1.001  0.0001
# 2-step output vs 4-step reference             0.0808 0.8969  0.9188  1.175  0.0269
# Different seed                                0.1810 0.6287  0.6435  0.955  0.1494
# Different prompt                              0.4454 -0.003 -0.1155  1.902  0.4354
# CFG off                                       0.2878 0.6819  0.7895  1.739  0.2846
# Brightness +0.03                              0.0300 0.9992  0.9999  0.999  0.0300
# Contrast 0.8                                  0.0570 0.9736  1.0000  0.839  0.0502
# R/B channels swapped                          0.0366 0.9225  0.9103  1.000  0.0000
# 3x3 box blur                                  0.0016 0.9998  0.9997  0.569  0.0000
# Gaussian noise, sigma 0.02                    0.0159 0.9950  0.9937 10.682  0.0000
#
# ~1.5x margin over CI. Catches dev pin, 2-step, seed, prompt, CFG off.
FULL_IMAGE_MAE_MAX = 0.074
# Margin over 0.9527. Catches dev pin, 2-step, seed, prompt, CFG off, R/B swap.
FULL_IMAGE_SSIM_MIN = 0.930
# Margin over 0.9717. Catches dev pin, 2-step, seed, prompt, CFG off, R/B swap.
FULL_IMAGE_PEARSON_MIN = 0.950
# Margin over 1.041. Catches dev pin, contrast, blur, noise, prompt, CFG off.
# Note: Contrast 0.8 is caught only by the HF energy ratio (0.839 < 0.85).
HF_ENERGY_RATIO_RANGE = (0.85, 1.20)
# Margin over 0.0380. Catches dev pin, seed, prompt, CFG off.
# Note: Brightness +0.03 not caught (smaller than TPU-vs-CPU mean shift of 0.0380).
MEAN_SHIFT_MAX = 0.060

PROMPT = "Change the latte art into a red heart and make the table white marble"
NEGATIVE_PROMPT = " "

# ==============================================================================


def _load_golden_reference_image() -> np.ndarray:
    """Load the CPU diffusers golden reference image staged by CI.
    Quantize the reference to uint8 to match vllm-omni PIL output.
    """
    path = os.path.join(
        os.environ.get("HF_HOME", "/local_hf_cache"),
        "golden",
        "qwen_image",
        "qwen_image_edit_cpu_golden_img.npy",
    )
    ref = np.load(path).astype(np.float32)
    return np.round(ref * 255.0) / 255.0


def _load_input_image() -> Image.Image:
    """Load the Qwen Image Edit input image staged by CI."""
    path = os.path.join(
        os.environ.get("HF_HOME", "/local_hf_cache"),
        "golden",
        "qwen_image",
        "qwen_image_edit_input.png",
    )
    return Image.open(path).convert("RGB")


def _build_qwen_prompt_dict(edit_engine, input_image):
    return build_image_to_image_prompt(
        model_class_name=get_model_class_name(edit_engine),
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        input_image=input_image,
        height=512,
        width=512,
    )


class TestQwenImageEditPrCiOnTpu:
    """Fast PR CI functional and numerical tests for Qwen-Image-Edit on TPU v7x."""

    @pytest.fixture(scope="class")
    def edit_engine(self):
        """Omni engine fixture: loads Qwen/Qwen-Image-Edit once per class."""
        omni = Omni(
            "Qwen/Qwen-Image-Edit",
            init_timeout=600,
            stage_init_timeout=600,
        )
        try:
            yield omni
        finally:
            omni.shutdown()

    def test_qwen_image_edit_fast_tpu(self, edit_engine):
        """Test Qwen-Image-Edit 2-step fast generation without file writes."""
        input_image = _load_input_image()
        prompt_dict = _build_qwen_prompt_dict(edit_engine, input_image)

        sampling_params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            true_cfg_scale=4.0,
            seed=42,
        )
        outputs = edit_engine.generate(prompt_dict, [sampling_params])
        assert outputs, "Omni returned empty output list"

        img = outputs[0].images[0]
        # FLUX-style PIL-to-float32 extraction
        arr = np.array(img.convert("RGB"), dtype=np.float32)

        # Sanity: image must not be blank / collapsed
        variance = float(np.var(arr))
        assert variance > 10.0, (
            f"Generated image has suspiciously low variance ({variance:.2f}), "
            "likely blank"
        )

    def test_qwen_image_edit_seed_parity_tpu(self, edit_engine):
        """Test identical seeds produce bitwise identical output on TPU v7x.

        Note: The test runs after the 2-step smoke test on the shared engine,
        so it checks repeatability of warm requests,
        not first-request-after-startup identity.
        """
        input_image = _load_input_image()
        prompt_dict = _build_qwen_prompt_dict(edit_engine, input_image)

        def run_with_seed(seed: int) -> np.ndarray:
            params = OmniDiffusionSamplingParams(
                height=512,
                width=512,
                num_inference_steps=4,
                true_cfg_scale=4.0,
                seed=seed,
            )
            out = edit_engine.generate(prompt_dict, [params])
            return np.asarray(out[0].images[0].convert("RGB"), dtype=np.float32) / 255.0

        arr1 = run_with_seed(42)
        arr2 = run_with_seed(42)
        arr3 = run_with_seed(99999)

        # Identical seeds must match exactly bitwise
        np.testing.assert_array_equal(
            arr1,
            arr2,
            err_msg="Identical seed 42 on TPU v7x must produce a bitwise pixel match",
        )

        # Distinct seed must diverge
        assert not np.array_equal(arr1, arr3), (
            "Different seeds should produce divergent pixel values"
        )

    def test_qwen_image_edit_golden_parity_tpu(self, edit_engine):
        """Compare full 512x512 image against CPU diffusers golden reference.

        The reference image is staged from gs://tpu-commons-ci/golden/.
        Asserts full-image Mean Absolute Error (MAE), Structural Similarity (SSIM),
        per-channel Pearson correlation, high-frequency Laplacian energy ratio,
        and absolute mean shift.
        """

        ref_img = _load_golden_reference_image()
        input_image = _load_input_image()
        prompt_dict = _build_qwen_prompt_dict(edit_engine, input_image)

        params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=4,
            true_cfg_scale=4.0,
            seed=0,
            generator_device="cpu",
        )
        outputs = edit_engine.generate(prompt_dict, [params])
        assert outputs, "Omni returned empty output list"

        arr = np.asarray(outputs[0].images[0].convert("RGB"), dtype=np.float32) / 255.0

        mae = float(np.mean(np.abs(arr - ref_img)))
        ssim_val = compute_ssim(arr, ref_img)
        pearsons = [
            float(np.corrcoef(arr[..., c].ravel(), ref_img[..., c].ravel())[0, 1])
            for c in range(3)
        ]
        min_pearson = min(pearsons)
        ref_hf = high_freq_energy(ref_img)
        hf_ratio = high_freq_energy(arr) / ref_hf
        mean_shift = float(abs(np.mean(arr) - np.mean(ref_img)))

        lo, hi = HF_ENERGY_RATIO_RANGE

        summary = (
            f"MAE={mae:.4f} (max {FULL_IMAGE_MAE_MAX}), "
            f"SSIM={ssim_val:.4f} (min {FULL_IMAGE_SSIM_MIN}), "
            f"Pearson={min_pearson:.4f} (min {FULL_IMAGE_PEARSON_MIN}), "
            f"HF ratio={hf_ratio:.3f} (range {lo}-{hi}), "
            f"Mean shift={mean_shift:.4f} (max {MEAN_SHIFT_MAX})"
        )
        print(summary)
        assert mae <= FULL_IMAGE_MAE_MAX, f"MAE exceeded tolerance: {summary}"
        assert ssim_val >= FULL_IMAGE_SSIM_MIN, f"SSIM below minimum: {summary}"
        assert min_pearson >= FULL_IMAGE_PEARSON_MIN, (
            f"Pearson below minimum: {summary}"
        )
        assert lo <= hf_ratio <= hi, f"HF ratio outside allowed range: {summary}"
        assert mean_shift <= MEAN_SHIFT_MAX, f"Mean shift exceeded tolerance: {summary}"
