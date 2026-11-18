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
"""TPU engine test for the asynchronous AsyncOmni entrypoint."""

from __future__ import annotations

import asyncio
from typing import cast

import numpy as np
import pytest
from PIL import Image
from vllm.engine.protocol import EngineClient
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.async_omni import AsyncOmni
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
SEED = 42
STAGE_INIT_TIMEOUT_S = 300
REQUEST_ID = "req-async-tpu"
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


def test_async_omni_generate() -> None:
    """Programmatic Asynchronous Engine API (AsyncOmni).

    - TPU path: constructs ``AsyncOmni`` on ``tiny_flux_schnell()`` inside an
      ``asyncio`` loop, streams ``async_omni.generate(...)`` on TPU, shuts down
      ``AsyncOmni``, and runs synchronous ``Omni`` on the same model and seed.
    - Setup check: ``async_omni`` implements ``EngineClient``, wraps an
      ``AsyncOmniEngine``, and passes ``await async_omni.check_health()``.
    - Outcome check: the streamed ``OmniRequestOutput`` preserves the external
      ``request_id``, marks ``finished=True``, and matches the synchronous
      ``Omni.generate`` RGB output bitwise.
    """
    model_dir = tiny_flux_schnell()

    async def _run_async() -> list[OmniRequestOutput]:
        async_omni = AsyncOmni(
            model=model_dir,
            stage_init_timeout=STAGE_INIT_TIMEOUT_S,
        )
        try:
            assert isinstance(async_omni, EngineClient), (
                f"AsyncOmni {type(async_omni).__name__} is not an EngineClient"
            )
            assert isinstance(async_omni.engine, AsyncOmniEngine), (
                "AsyncOmni engine type "
                f"{type(async_omni.engine).__name__} != AsyncOmniEngine"
            )
            await async_omni.check_health()

            params = OmniDiffusionSamplingParams(
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=STEPS,
                guidance_scale=0.0,
                seed=SEED,
                generator_device="cpu",
            )
            collected: list[OmniRequestOutput] = []
            async for out in async_omni.generate(
                prompt=PROMPT,
                request_id=REQUEST_ID,
                sampling_params_list=[params],
            ):
                collected.append(out)
            return collected
        finally:
            async_omni.shutdown()

    async_outputs = asyncio.run(_run_async())
    assert len(async_outputs) == 1, (
        f"AsyncOmni yielded {len(async_outputs)} outputs, expected 1"
    )
    assert async_outputs[0].request_id == REQUEST_ID, (
        f"AsyncOmni request_id {async_outputs[0].request_id!r} != {REQUEST_ID!r}"
    )

    sync_omni = Omni(
        model=model_dir,
        stage_init_timeout=STAGE_INIT_TIMEOUT_S,
    )
    try:
        sync_params = OmniDiffusionSamplingParams(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            seed=SEED,
            generator_device="cpu",
        )
        sync_outputs = sync_omni.generate(PROMPT, sync_params, use_tqdm=False)
    finally:
        sync_omni.close()

    async_img = _extract_rgb_image(async_outputs, "async_omni")
    sync_img = _extract_rgb_image(sync_outputs, "sync_omni")
    assert np.array_equal(async_img, sync_img), (
        "AsyncOmni output diverged from synchronous Omni baseline: "
        f"max_abs={int(np.abs(async_img.astype(int) - sync_img.astype(int)).max())}"
    )
