# SPDX-License-Identifier: Apache-2.0
"""PR CI running tests for FLUX models on TPU v7x.

Fast end-to-end tests designed to run in PR CI (< 3 minutes) on TPU v7x machines.
Covers fast inference with FLUX.1-schnell, golden-image parity with a CPU diffusers
reference, and deterministic seed parity, without file writes.
"""

import os

import numpy as np
import pytest
from PIL import Image
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import compute_ssim, high_freq_energy
from ...conftest import OmniRunner

pytestmark = [
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

# ==============================================================================
# Golden reference: full 512x512 CPU diffusers reference image staged from
# gs://tpu-commons-ci/golden/flux_schnell_cpu_golden_img.npy.
# Regenerate with diffusers 0.40.0 on CPU:
#
#   import torch
#   from diffusers import FluxPipeline
#
#   torch.set_num_threads(8)
#   pipe = FluxPipeline.from_pretrained(
#       "black-forest-labs/FLUX.1-schnell",
#       revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
#       torch_dtype=torch.bfloat16,
#   )
#   # float32 VAE (bf16 convolutions are very slow on CPU); cast the latents.
#   pipe.vae.to(torch.float32)
#   decode = pipe.vae.decode
#   pipe.vae.decode = lambda z, *args, **kwargs: decode(z.float(), *args, **kwargs)
#   img = pipe(
#       prompt="A vibrant green leaf with glistening morning water droplets",
#       height=512,
#       width=512,
#       num_inference_steps=2,
#       guidance_scale=0.0,
#       max_sequence_length=512,
#       generator=torch.Generator(device="cpu").manual_seed(0),
#       output_type="np",
#   ).images[0]
#   np.save("flux_schnell_cpu_golden_img.npy", img)
# ==============================================================================

# Full-image comparison metrics (calibrated against CPU reference across 512x512x3):
#   Image vs CPU ref                 MAE    SSIM   min Pearson  HF ratio
#   TPU v7x (dev20260912)           0.0181 0.9616    0.9389      0.982
#   TPU v7x (dev20260924)           0.0184 0.9618    0.9373      1.049
#   Same recipe, different CPU      0.0164 0.9698    0.9462      1.002
#   max_sequence_length=256         0.1021 0.6240    0.5376      0.970
#   Different seed                  0.2041 0.1066    0.0430      1.395
#   Different prompt                0.1947 0.2174    0.1371      0.676
#   Brightness +0.03                0.0300 0.9659    1.0000      0.999
#   R/B channels swapped            0.0843 0.4825    0.4802      1.000
#   5x5 box blur                    0.0063 0.9908    0.9817      0.438
#   Gaussian noise, sigma 0.02      0.0150 0.9916    0.9838      3.818
FULL_IMAGE_MAE_MAX = 0.025
FULL_IMAGE_SSIM_MIN = 0.940
FULL_IMAGE_PEARSON_MIN = 0.900
HF_ENERGY_RATIO_RANGE = (0.80, 1.25)

# ==============================================================================


def _load_golden_reference_image() -> np.ndarray:
    """Load the CPU diffusers golden reference image staged by CI."""
    path = os.path.join(
        os.environ.get("HF_HOME", "/local_hf_cache"),
        "golden",
        "flux_schnell_cpu_golden_img.npy",
    )
    return np.load(path).astype(np.float32)


def _extract_image(output) -> Image.Image:
    """Extract PIL Image from Omni generate output."""
    return output.images[0]


@pytest.mark.parametrize(
    "omni_runner",
    ["black-forest-labs/FLUX.1-schnell"],
    indirect=True,
)
class TestFluxPrCiOnTpu:
    """Fast PR CI functional and numerical tests for FLUX on TPU v7x."""

    def test_flux1_schnell_fast_t2i_tpu(self, omni_runner: OmniRunner):
        """Test FLUX.1-schnell 2-step fast generation on TPU v7x without file writes."""

        prompt = (
            "A photo of a majestic snow leopard perched atop a mountain peak at "
            "golden hour, 8k"
        )
        sampling_params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=42,
        )
        outputs = omni_runner.generate(prompt, sampling_params)
        assert outputs, "Omni returned empty output list"

        img = _extract_image(outputs[0])
        assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
        assert img.size == (512, 512), f"Expected (512, 512), got {img.size}"

        # In-memory numerical sanity: image must not be blank / collapsed
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        variance = float(np.var(arr))
        assert variance > 10.0, (
            f"Generated image has suspiciously low variance ({variance:.2f}), "
            "likely blank"
        )

    def test_flux_deterministic_seed_parity_tpu(self, omni_runner: OmniRunner):
        """Test that identical seeds produce bitwise identical output on TPU v7x."""

        prompt = "A red flower in an open sunny green meadow"
        fixed_seed = 42

        # Run 1 with fixed seed
        params1 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=fixed_seed,
        )
        out1 = omni_runner.generate(prompt, params1)
        img1 = _extract_image(out1[0])

        # Run 2 with identical seed
        params2 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=fixed_seed,
        )
        out2 = omni_runner.generate(prompt, params2)
        img2 = _extract_image(out2[0])

        # Run 3 with distinct seed
        params3 = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            seed=99999,
        )
        out3 = omni_runner.generate(prompt, params3)
        img3 = _extract_image(out3[0])

        arr1 = np.array(img1.convert("RGB"), dtype=np.float32)
        arr2 = np.array(img2.convert("RGB"), dtype=np.float32)
        arr3 = np.array(img3.convert("RGB"), dtype=np.float32)

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

    def test_flux_golden_full_image_parity_tpu(self, omni_runner: OmniRunner):
        """Compare full 512x512 image against CPU diffusers golden reference.

        The reference image is staged from gs://tpu-commons-ci/golden/.
        Asserts full-image Mean Absolute Error (MAE), Structural Similarity (SSIM),
        per-channel Pearson correlation, and high-frequency Laplacian energy ratio.
        """

        ref_img = _load_golden_reference_image()

        prompt = "A vibrant green leaf with glistening morning water droplets"
        params = OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_inference_steps=2,
            guidance_scale=0.0,
            # T5 pads the prompt to this length without a mask, so it changes the
            # image and must match the reference.
            max_sequence_length=512,
            seed=0,
            generator_device="cpu",
        )
        outputs = omni_runner.generate(prompt, params)
        assert outputs, "Omni returned empty output list"

        img = _extract_image(outputs[0])
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

        assert 0.10 < float(np.mean(arr)) < 0.35, (
            f"Suspicious image mean: {np.mean(arr):.4f}"
        )
        assert float(np.var(arr)) > 0.01, (
            f"Suspicious image variance: {np.var(arr):.4f}"
        )

        mae = float(np.mean(np.abs(arr - ref_img)))
        ssim_val = compute_ssim(arr, ref_img)
        pearsons = [
            float(np.corrcoef(arr[..., c].ravel(), ref_img[..., c].ravel())[0, 1])
            for c in range(3)
        ]
        min_pearson = min(pearsons)
        ref_hf = high_freq_energy(ref_img)
        hf_ratio = high_freq_energy(arr) / ref_hf
        lo, hi = HF_ENERGY_RATIO_RANGE

        summary = (
            f"full image MAE={mae:.4f} (max {FULL_IMAGE_MAE_MAX}), "
            f"SSIM={ssim_val:.4f} (min {FULL_IMAGE_SSIM_MIN}), "
            f"min channel Pearson={min_pearson:.4f} (min {FULL_IMAGE_PEARSON_MIN}), "
            f"high-frequency energy ratio={hf_ratio:.3f} (range {lo}-{hi})"
        )
        print(summary)
        assert (
            mae <= FULL_IMAGE_MAE_MAX
            and ssim_val >= FULL_IMAGE_SSIM_MIN
            and min_pearson >= FULL_IMAGE_PEARSON_MIN
            and lo <= hf_ratio <= hi
        ), f"TPU image diverged from the CPU diffusers reference: {summary}"
