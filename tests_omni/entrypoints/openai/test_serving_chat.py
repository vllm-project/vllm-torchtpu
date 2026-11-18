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
"""TPU end-to-end test for the OpenAI Chat Completions API on diffusion models."""

from __future__ import annotations

import asyncio
import base64
import io
from argparse import Namespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state, router
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

MODEL_NAME = "tiny-flux-schnell"
IMAGE_SIZE = 256
NUM_STEPS = 2
SEED = 42
PROMPT = "A geometric prism on a sunlit desk"


def _server_args(model_dir: str) -> Namespace:
    return Namespace(
        model=model_dir,
        served_model_name=[MODEL_NAME],
        enable_log_requests=False,
        max_log_len=None,
        disable_log_stats=True,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_server_load_tracking=False,
        default_sampling_params=None,
        max_generated_image_size=None,
    )


def _decode_chat_png(choice: dict[str, object]) -> np.ndarray:
    message = choice["message"]
    assert isinstance(message, dict), f"Expected dict message, got {type(message)}"
    content = message["content"]
    assert isinstance(content, list), (
        f"Expected content block list, got {type(content)}"
    )
    assert len(content) == 1, f"Expected 1 image block, got {len(content)}"
    block = content[0]
    assert block["type"] == "image_url", f"Unexpected block type: {block['type']}"
    url = block["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix), f"Expected PNG data URL, got {url[:32]!r}"
    raw_bytes = base64.b64decode(url[len(prefix) :])
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _encode_decode_png(image: Image.Image) -> np.ndarray:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


async def _run_chat_completions(
    model_dir: str,
) -> tuple[
    str,
    bool,
    bool,
    int,
    str,
    np.ndarray,
    int,
    np.ndarray,
    np.ndarray,
]:
    engine = AsyncOmni(model=model_dir, log_stats=False)
    try:
        app = FastAPI()
        app.include_router(router)
        await omni_init_app_state(engine, app.state, _server_args(model_dir))

        chat_mode = bool(app.state.openai_serving_chat._diffusion_mode)
        batch_mode = bool(app.state.openai_serving_chat_batch._diffusion_mode)

        extra_body = {
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "num_inference_steps": NUM_STEPS,
            "guidance_scale": 0.0,
            "seed": SEED,
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            chat_resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "extra_body": extra_body,
                },
            )
            batch_resp = await client.post(
                "/v1/chat/completions/batch",
                json={
                    "model": MODEL_NAME,
                    "messages": [[{"role": "user", "content": PROMPT}]],
                    "extra_body": extra_body,
                },
            )

        chat_body = chat_resp.json()
        chat_img = _decode_chat_png(chat_body["choices"][0])
        batch_body = batch_resp.json()
        batch_img = _decode_chat_png(batch_body["choices"][0])

        ref_params = OmniDiffusionSamplingParams(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=NUM_STEPS,
            guidance_scale=0.0,
            seed=SEED,
        )
        ref_output = None
        async for item in engine.generate(
            prompt=OmniTextPrompt(prompt=PROMPT, modalities=["image"]),
            sampling_params_list=[ref_params],
            request_id="direct-ref",
        ):
            ref_output = item
        assert ref_output is not None, "Direct AsyncOmni.generate produced no output"
        ref_img = _encode_decode_png(ref_output.images[0])

        return (
            current_omni_platform.device_type,
            chat_mode,
            batch_mode,
            chat_resp.status_code,
            str(chat_body["model"]),
            chat_img,
            batch_resp.status_code,
            batch_img,
            ref_img,
        )
    finally:
        engine.shutdown()


def test_chat_completions_api() -> None:
    """Chat Completions API on TPU via pure-diffusion serving.

    - TPU path: inline ``AsyncOmni`` on ``tiny_flux_schnell()`` mounted behind
      ``/v1/chat/completions`` and ``/v1/chat/completions/batch``.
    - Setup check: TPU omni platform active and both chat handlers initialized in
      diffusion mode via ``omni_init_app_state``.
    - Outcome check: HTTP 200 PNG responses match direct ``AsyncOmni.generate``
      pixels bitwise for identical sampling parameters.
    """
    model_dir = tiny_flux_schnell()
    (
        device_type,
        chat_mode,
        batch_mode,
        chat_status,
        served_model,
        chat_img,
        batch_status,
        batch_img,
        ref_img,
    ) = asyncio.run(_run_chat_completions(model_dir))

    assert device_type == "tpu", f"Expected TPU omni platform, got {device_type!r}"
    assert chat_mode is True, "OmniOpenAIServingChat did not enter diffusion mode"
    assert batch_mode is True, "OmniOpenAIServingChatBatch did not enter diffusion mode"
    assert chat_status == 200, f"POST /v1/chat/completions returned HTTP {chat_status}"
    assert served_model == MODEL_NAME, (
        f"Expected served model {MODEL_NAME!r}, got {served_model!r}"
    )
    assert batch_status == 200, (
        f"POST /v1/chat/completions/batch returned HTTP {batch_status}"
    )
    assert chat_img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), (
        f"Unexpected chat image shape {chat_img.shape}"
    )
    assert batch_img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), (
        f"Unexpected batch chat image shape {batch_img.shape}"
    )
    np.testing.assert_array_equal(
        chat_img,
        ref_img,
        err_msg="/v1/chat/completions PNG pixels diverged from direct AsyncOmni output",
    )
    np.testing.assert_array_equal(
        batch_img,
        ref_img,
        err_msg="/v1/chat/completions/batch PNG pixels diverged from AsyncOmni output",
    )
