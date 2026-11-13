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
"""Nightly build tests for Z-Image models on TPU.

Integration and numerical correctness tests across the Z-Image model family:
- Tongyi-MAI/Z-Image-Turbo (distilled 8-step guidance-free)
- Tongyi-MAI/Z-Image (base 28-step with Classifier-Free Guidance)

Models are resolved through the HF cache (``HF_HOME``) like the FLUX tests.

Every generation runs in a fresh interpreter (see ``_generate_once``): the
single-chip uniproc executor opens the TPU *in-process* and PJRT keeps
``/dev/vfio/0`` until that process exits, so a multi-chip run launched
afterwards from the same pytest process cannot acquire chip 0.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch_tpu._internal.utils import hardware

from tests.tpu_release import wait_for_tpu_release

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

Z_IMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_BASE = "Tongyi-MAI/Z-Image"

PROMPT = (
    "A photorealistic landscape of majestic mountains and a serene lake "
    "at sunset, golden hour reflections"
)

# Same-seed parity of multi-chip Ulysses (sp=8 on 8-chip slices, sp=4 on 4-chip
# slices) against the single-chip image. "advanced_uaa" pads 30 heads to 32 and
# runs bf16 all-to-all collectives, so bit-exactness is not expected. Measured
# on v6e-8 (Turbo, sp=8, 8 steps, seed 42): PSNR 42.14 dB, MAE 0.0028.
# Thresholds leave >10 dB / ~3x headroom.
SP_PARITY_PSNR_MIN_DB = 30.0
SP_PARITY_MAE_MAX = 0.01

# Executed with ``python -c`` in a child interpreter; argv[1] is a JSON config.
_GENERATE_CHILD_SCRIPT = """
import json, sys
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

cfg = json.loads(sys.argv[1])
omni = Omni(
    cfg["model_id"],
    init_timeout=900,
    stage_init_timeout=900,
    **cfg["omni_kwargs"],
)
try:
    out = omni.generate(cfg["prompt"], OmniDiffusionSamplingParams(**cfg["params"]))
    out[0].images[0].save(cfg["out_path"])
finally:
    omni.shutdown()
"""
_GENERATE_TIMEOUT_S = 600


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def compute_distribution_metrics(img: Image.Image) -> dict[str, float]:
    """Calculate scalar distribution moments and percentiles from output pixels."""
    arr = _to_array(img)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "var": float(np.var(arr)),
    }


def _assert_plausible_image(img: Image.Image, label: str) -> None:
    assert isinstance(img, Image.Image)
    assert img.size == (1024, 1024)
    metrics = compute_distribution_metrics(img)
    assert 0.10 < metrics["mean"] < 0.90, (
        f"Unnatural pixel mean on {label}: {metrics['mean']:.4f}"
    )
    assert metrics["var"] > 0.01, (
        f"Image variance too low on {label} ({metrics['var']:.4f})"
    )
    assert metrics["p90"] > metrics["p10"], f"Collapsed dynamic range on {label}"


def _sampling_params(steps: int, guidance: float) -> dict:
    return {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": 42,
    }


def _generate_once(
    model_id: str, params: dict, out_path: Path, **omni_kwargs
) -> Image.Image:
    """Generate one image in a fresh interpreter and return it.

    A child process (rather than ``Omni`` in-process) is required so that the
    TPU device nodes opened by a single-chip run are released before the next
    run starts; see the module docstring.
    """
    cfg = {
        "model_id": model_id,
        "prompt": PROMPT,
        "params": params,
        "omni_kwargs": omni_kwargs,
        "out_path": str(out_path),
    }
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", _GENERATE_CHILD_SCRIPT, json.dumps(cfg)],
        start_new_session=True,
    )
    try:
        returncode = proc.wait(timeout=_GENERATE_TIMEOUT_S)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        wait_for_tpu_release()
    assert returncode == 0, (
        f"Generation for {model_id} ({omni_kwargs}) exited with {returncode}"
    )
    assert out_path.is_file(), f"Generation for {model_id} produced no image"
    with Image.open(out_path) as img:
        return img.convert("RGB")


class TestZImageNightlyOnTpu:
    """Nightly regression test suite for Z-Image models on TPU."""

    @pytest.mark.parametrize(
        "model_id,steps,guidance",
        [
            (Z_IMAGE_TURBO, 8, 0.0),
            (Z_IMAGE_BASE, 28, 3.5),
        ],
        ids=["z-image-turbo", "z-image-base"],
    )
    def test_z_image_model_matrix_tpu(
        self, tmp_path: Path, model_id: str, steps: int, guidance: float
    ):
        """Validate common path execution across Z-Image-Turbo and Base on TPU."""
        img = _generate_once(
            model_id, _sampling_params(steps, guidance), tmp_path / "img.png"
        )
        _assert_plausible_image(img, model_id)

    @pytest.mark.multichip
    def test_z_image_turbo_ulysses_parity_tpu(self, tmp_path: Path):
        """Same-seed parity of multi-chip Ulysses against a single-chip run.

        Exercises the Z-Image multi-chip path that the TPU platform enables:
        30 heads are not divisible by 4 or 8, so strict Ulysses mode would
        raise and the run only succeeds because
        ``OmniTpuPlatform.init_diffusion_model_runner_runtime`` promoted
        ``ulysses_mode`` to ``"advanced_uaa"`` (``ulysses_degree=2`` divides 30
        and would silently stay in strict mode). Uses ``ulysses_degree=8`` on
        8-chip slices (since a 4-chip ``2x2`` sub-slice of a ``2x4`` v6e-8
        fails in libtpu ``ConstructSlice2D``) and ``4`` on 4-chip slices.
        """
        # PCI scan: counts TPU devices without starting the TPU runtime here.
        device_count = hardware.get_tpu_device_count()
        if device_count < 4:
            pytest.skip("Requires at least 4 TPU chips for multi-chip Ulysses")
        ulysses_degree = 8 if device_count >= 8 else 4

        params = _sampling_params(steps=8, guidance=0.0)
        ref = _to_array(_generate_once(Z_IMAGE_TURBO, params, tmp_path / "sp1.png"))
        sp_img = _generate_once(
            Z_IMAGE_TURBO,
            params,
            tmp_path / f"sp{ulysses_degree}.png",
            ulysses_degree=ulysses_degree,
        )
        _assert_plausible_image(sp_img, f"{Z_IMAGE_TURBO} (sp={ulysses_degree})")
        sp = _to_array(sp_img)

        mae = float(np.mean(np.abs(sp - ref)))
        mse = float(np.mean((sp - ref) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * float(np.log10(1.0 / mse))
        # Printed (not just asserted) so nightly logs record the measured margin.
        print(
            f"[z-image sp={ulysses_degree} parity] PSNR={psnr:.2f} dB MAE={mae:.4f}",
            flush=True,
        )
        assert psnr >= SP_PARITY_PSNR_MIN_DB, (
            f"sp={ulysses_degree} vs sp=1 PSNR {psnr:.2f} dB below "
            f"{SP_PARITY_PSNR_MIN_DB} dB (MAE={mae:.4f})"
        )
        assert mae <= SP_PARITY_MAE_MAX, (
            f"sp={ulysses_degree} vs sp=1 MAE {mae:.4f} above {SP_PARITY_MAE_MAX} "
            f"(PSNR={psnr:.2f} dB)"
        )
