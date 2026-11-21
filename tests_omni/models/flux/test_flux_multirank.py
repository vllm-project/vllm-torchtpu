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
"""Nightly multi-chip parity tests for FLUX.1-dev on TPU v7x."""

import math
import os
import time

import numpy as np
import pytest
from PIL import Image
from torch_tpu._internal.utils import hardware

pytest.importorskip("vllm_omni", reason="vllm_omni is required for FLUX tests")

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import compute_ssim

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.multichip,
]

FLUX1_DEV = "black-forest-labs/FLUX.1-dev"
PROMPT = (
    "A futuristic glowing crystal monolith in a misty desert at night, neon reflections"
)
NEGATIVE_PROMPT = "blurry, low quality, distorted, oversaturated"
HEIGHT = 512
WIDTH = 512
STEPS = 4
GUIDANCE_SCALE = 3.5
TRUE_CFG_SCALE = 2.0
SEED = 42

# PSNR floor calibrated from TPU v7x run-to-run floor and bf16 TP all-reduce.
FLUX_MULTIRANK_MIN_PSNR_DB = 30.0
FLUX_RUN_TO_RUN_MIN_PSNR_DB = 60.0

CASES = [
    ({"tensor_parallel_size": 8}, "tp8"),
    ({"cfg_parallel_size": 2, "tensor_parallel_size": 4}, "cfg2xtp4"),
]


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR (dB) between two RGB images in [0, 255]."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((255.0**2) / mse))


def _maybe_save_output(tag: str, arr: np.ndarray) -> None:
    out_dir = os.environ.get("VLLM_FLUX_PARITY_OUT_DIR")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    u8 = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    np.save(os.path.join(out_dir, f"{tag}.npy"), u8)
    Image.fromarray(u8).save(os.path.join(out_dir, f"{tag}.png"))


def _generate_flux_image(runner: OmniRunner) -> tuple[np.ndarray, float]:
    inputs = {"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT}
    params = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        true_cfg_scale=TRUE_CFG_SCALE,
        max_sequence_length=512,
        seed=SEED,
        generator_device="cpu",
    )
    t0 = time.perf_counter()
    outputs = list(runner.generate(inputs, params))
    elapsed = time.perf_counter() - t0
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    img = outputs[0].images[0]
    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
    assert img.size == (WIDTH, HEIGHT), f"Expected {(WIDTH, HEIGHT)}, got {img.size}"
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    assert np.all(np.isfinite(arr)), "Image contains NaN or Inf values"
    assert float(np.var(arr)) > 10.0, f"Low image variance: {float(np.var(arr)):.2f}"
    return arr, elapsed


@pytest.fixture(scope="module", autouse=True)
def _check_tpu_devices() -> None:
    max_world_size = max(
        DiffusionParallelConfig(**degrees).world_size for degrees, _ in CASES
    )
    device_count = hardware.get_tpu_device_count()
    assert device_count >= max_world_size, (
        f"Test requires {max_world_size} TPU devices, found {device_count}"
    )


@pytest.fixture(scope="module", autouse=True)
def single_device_reference() -> np.ndarray:
    """Generate 1-device FLUX.1-dev baseline and verify its run-to-run PSNR floor."""
    t_load0 = time.perf_counter()
    # Use mp executor so the 1-device worker exits and releases /dev/vfio/*.
    with OmniRunner(FLUX1_DEV, distributed_executor_backend="mp") as runner:
        load_s = time.perf_counter() - t_load0
        img1, cold_s = _generate_flux_image(runner)
        img2, warm_s = _generate_flux_image(runner)
    _maybe_save_output("flux_1dev_run1", img1)
    _maybe_save_output("flux_1dev_run2", img2)
    r2r_psnr = compute_psnr(img1, img2)
    print(
        f"FLUX.1-dev 1-device: load_s={load_s:.3f}s, cold_s={cold_s:.3f}s, "
        f"warm_s={warm_s:.3f}s ({warm_s / STEPS:.4f}s/step), "
        f"r2r_psnr={r2r_psnr:.2f} dB, mean={float(np.mean(img1)):.4f}, "
        f"std={float(np.std(img1)):.4f}"
    )
    assert r2r_psnr >= FLUX_RUN_TO_RUN_MIN_PSNR_DB, (
        f"1-device run-to-run PSNR {r2r_psnr:.2f} dB < {FLUX_RUN_TO_RUN_MIN_PSNR_DB} dB"
    )
    return img1


@pytest.mark.parametrize(
    "omni_runner,degrees,case_id",
    [
        pytest.param((FLUX1_DEV, None, degrees), degrees, case_id, id=case_id)
        for degrees, case_id in CASES
    ],
    indirect=["omni_runner"],
)
def test_flux_multirank_parity_tpu(
    single_device_reference: np.ndarray,
    omni_runner: OmniRunner,
    degrees: dict[str, int],
    case_id: str,
) -> None:
    """Verify TP8 and CFG2xTP4 FLUX.1-dev outputs match 1-device output."""
    stage_parallel = omni_runner.stage_configs[0].parallel_config
    expected = {
        "cfg_parallel_size": 1,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "tensor_parallel_size": 1,
        **degrees,
    }
    actual = {name: getattr(stage_parallel, name) for name in expected}
    assert actual == expected, (
        f"Engine parallel degrees {actual} != expected {expected}"
    )
    img1, cold_s = _generate_flux_image(omni_runner)
    img2, warm_s = _generate_flux_image(omni_runner)
    _maybe_save_output(f"flux_{case_id}_run1", img1)
    _maybe_save_output(f"flux_{case_id}_run2", img2)
    r2r_psnr = compute_psnr(img1, img2)
    parity_psnr = compute_psnr(img1, single_device_reference)
    ssim_val = compute_ssim(img1 / 255.0, single_device_reference / 255.0)
    mae_val = float(np.mean(np.abs(img1 - single_device_reference)) / 255.0)
    print(
        f"FLUX.1-dev {case_id} ({degrees}): cold_s={cold_s:.3f}s, "
        f"warm_s={warm_s:.3f}s ({warm_s / STEPS:.4f}s/step), "
        f"r2r_psnr={r2r_psnr:.2f} dB, parity_psnr={parity_psnr:.2f} dB, "
        f"ssim={ssim_val:.4f}, mae={mae_val:.4f}, "
        f"mean={float(np.mean(img1)):.4f}, std={float(np.std(img1)):.4f}"
    )
    assert r2r_psnr >= FLUX_RUN_TO_RUN_MIN_PSNR_DB, (
        f"Run-to-run PSNR {r2r_psnr:.2f} dB < {FLUX_RUN_TO_RUN_MIN_PSNR_DB} dB"
    )
    assert parity_psnr >= FLUX_MULTIRANK_MIN_PSNR_DB, (
        f"Multi-rank PSNR {parity_psnr:.2f} dB < {FLUX_MULTIRANK_MIN_PSNR_DB} dB"
    )
