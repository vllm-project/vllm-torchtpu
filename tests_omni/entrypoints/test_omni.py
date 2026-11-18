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
"""TPU engine test for the synchronous Omni offline inference API."""

from __future__ import annotations

from collections.abc import Generator
from typing import cast

import numpy as np
import pytest
from PIL import Image
from vllm_omni.diffusion.inline_stage_diffusion_client import (
    InlineStageDiffusionClient,
)
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

IMAGE_SIZE = 256
STEPS = 2
SEED_PRIMARY = 42
SEED_CONTROL = 99
STAGE_INIT_TIMEOUT_S = 300
PROMPT = "a geometric cube on a plain table"


def _extract_rgb_image(
    outputs: list[OmniRequestOutput],
    label: str,
) -> np.ndarray:
    """Validate and extract a single ``(H, W, 3)`` uint8 RGB image."""
    assert len(outputs) == 1, f"{label}: output count {len(outputs)} != 1"
    output = outputs[0]
    assert output.finished is True, f"{label}: output finished={output.finished}"
    assert len(output.images) == 1, f"{label}: image count {len(output.images)} != 1"
    pil_img = cast(Image.Image, output.images[0]).convert("RGB")
    arr = np.asarray(pil_img, dtype=np.uint8)
    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert arr.shape == expected_shape, (
        f"{label}: image shape {arr.shape} != {expected_shape}"
    )
    return arr


def test_omni_offline_generate() -> None:
    """Programmatic Offline Inference API (Omni).

    - TPU path: constructs inline single-device ``Omni`` on ``tiny_flux_schnell()``
      with ``stage_overrides={"0": {"vae_use_slicing": True}}`` and generates
      256x256 images in both batch (``py_generator=False``) and streaming
      generator (``py_generator=True``) modes.
    - Setup check: ``omni.engine`` is an ``AsyncOmniEngine`` with 1 stage and
      ``vae_use_slicing`` enabled on stage 0.
    - Outcome check: ``py_generator=True`` returns a ``Generator`` whose output
      image matches ``py_generator=False`` bitwise at ``seed=42`` and differs
      from a ``seed=99`` control run.
    """
    model_dir = tiny_flux_schnell()
    omni = Omni(
        model=model_dir,
        stage_overrides={"0": {"vae_use_slicing": True}},
        stage_init_timeout=STAGE_INIT_TIMEOUT_S,
    )
    try:
        assert isinstance(omni.engine, AsyncOmniEngine), (
            f"Omni engine type {type(omni.engine).__name__} != AsyncOmniEngine"
        )
        assert omni.engine.num_stages == 1, (
            f"Omni num_stages {omni.engine.num_stages} != 1"
        )
        stage_client = cast(InlineStageDiffusionClient, omni.engine.stage_clients[0])
        assert stage_client.od_config.vae_use_slicing is True, (
            "Omni stage_overrides vae_use_slicing="
            f"{stage_client.od_config.vae_use_slicing} != True"
        )

        params_batch = OmniDiffusionSamplingParams(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            seed=SEED_PRIMARY,
            generator_device="cpu",
        )
        params_stream = OmniDiffusionSamplingParams(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            seed=SEED_PRIMARY,
            generator_device="cpu",
        )
        params_control = OmniDiffusionSamplingParams(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            seed=SEED_CONTROL,
            generator_device="cpu",
        )

        batch_outputs = omni.generate(
            PROMPT,
            params_batch,
            py_generator=False,
            use_tqdm=False,
        )
        stream_gen = omni.generate(
            PROMPT,
            params_stream,
            py_generator=True,
            use_tqdm=False,
        )
        assert isinstance(stream_gen, Generator), (
            f"py_generator=True returned {type(stream_gen).__name__} != Generator"
        )
        stream_outputs = list(stream_gen)
        control_outputs = omni.generate(
            PROMPT,
            params_control,
            py_generator=False,
            use_tqdm=False,
        )
    finally:
        omni.close()

    batch_img = _extract_rgb_image(batch_outputs, "batch_seed42")
    stream_img = _extract_rgb_image(stream_outputs, "stream_seed42")
    control_img = _extract_rgb_image(control_outputs, "batch_seed99")

    assert np.array_equal(stream_img, batch_img), (
        "Omni py_generator=True output diverged from py_generator=False at seed=42: "
        f"max_abs={int(np.abs(stream_img.astype(int) - batch_img.astype(int)).max())}"
    )
    assert not np.array_equal(control_img, batch_img), (
        "Omni seed=99 output unexpectedly matched seed=42 bitwise"
    )
