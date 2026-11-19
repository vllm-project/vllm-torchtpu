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
"""TPU test for the JoyVL Interaction Server fronting the JoyAI AR pipeline."""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.experimental.fullduplex.joyvl.serving.server import create_app
from vllm_omni.model_executor.models.joyai_vl_interaction.pipeline import (
    JOYAI_VL_INTERACTION_PIPELINE,
)
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MODEL_NAME = "JoyAI-VL-Interaction-Preview"
SERVER_START_TIMEOUT_S = 10.0


def _write_tiny_ar_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 512,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _png_data_url() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(40, 120, 200)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@contextmanager
def _serve_joyvl_model_backend(reply_content: str) -> Iterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    backend_app = FastAPI()

    @backend_app.post("/v1/chat/completions")
    async def completions(request: Request) -> dict[str, Any]:
        payload = await request.json()
        return {
            "id": "joyvl-backend-0",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
        }

    server = uvicorn.Server(
        uvicorn.Config(backend_app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert server.started, (
            f"fake JoyVL backend not started after {SERVER_START_TIMEOUT_S}s"
        )
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        sock.close()


def test_joyvl_interaction_server(tmp_path: Path) -> None:
    """JoyVL Interaction Server on TPU fronting the JoyAI AR/TTS pipeline.

    - TPU path: ``joyvl.serving.server.create_app`` paired with
      ``JOYAI_VL_INTERACTION_PIPELINE`` AR/connector ``OmniEngineArgs`` on TPU.
    - Setup check: TPU omni platform active, ``get_omni_ar_worker_cls()`` resolved,
      and ``GET /health`` and ``GET /v1/models`` succeed on the interaction app.
    - Outcome check: once ``OmniEngineArgs`` accepts the TPU AR worker,
      ``POST /v1/chat/completions`` returns parsed JoyVL interaction actions and
      ``POST /v1/streaming/reset`` clears the session.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_type!r}"
    )
    ar_worker_cls = current_omni_platform.get_omni_ar_worker_cls()
    assert isinstance(ar_worker_cls, str), (
        f"Expected str AR worker class path, got {type(ar_worker_cls).__name__}"
    )
    assert bool(ar_worker_cls), (
        f"Expected non-empty AR worker class path, got {ar_worker_cls!r}"
    )

    tiny_model_dir = tmp_path / "joyvl_ar_stage"
    _write_tiny_ar_config(tiny_model_dir)
    talker_stage = JOYAI_VL_INTERACTION_PIPELINE.stages[1]

    with _serve_joyvl_model_backend(
        "<speak>Blue square visible.</speak>"
    ) as backend_url:
        config = InteractionConfig(
            main_backend_url=backend_url,
            main_model=MODEL_NAME,
            enable_memory=False,
            enable_delegation=False,
        )
        app = create_app(config)

        with TestClient(app) as client:
            health = client.get("/health")
            models = client.get("/v1/models")
            assert health.status_code == 200, (
                f"GET /health returned {health.status_code}"
            )
            assert health.json() == {"status": "ok"}, (
                f"Unexpected /health payload: {health.json()}"
            )
            assert models.status_code == 200, (
                f"GET /v1/models returned {models.status_code}"
            )
            assert models.json()["data"][0]["id"] == MODEL_NAME, (
                f"Unexpected served model id: {models.json()}"
            )

            # Build the backing JoyAI AR connector stage on the active TPU platform.
            stage_args = OmniEngineArgs(
                model=str(tiny_model_dir),
                worker_type="ar",
                model_stage=talker_stage.model_stage,
                custom_process_next_stage_input_func=(
                    talker_stage.custom_process_next_stage_input_func
                ),
            )
            assert stage_args.worker_cls == ar_worker_cls, (
                f"Expected stage worker {ar_worker_cls!r}, "
                f"got {stage_args.worker_cls!r}"
            )

            chat_resp = client.post(
                "/v1/chat/completions",
                headers={"x-session-id": "sess-0"},
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _png_data_url()},
                                },
                                {"type": "text", "text": "What is on screen?"},
                            ],
                        }
                    ],
                },
            )
            assert chat_resp.status_code == 200, (
                f"POST /v1/chat/completions failed ({chat_resp.status_code}): "
                f"{chat_resp.text}"
            )
            body = chat_resp.json()
            assert body["model"] == MODEL_NAME, (
                f"Expected model {MODEL_NAME!r}, got {body['model']!r}"
            )
            assert body["interaction"]["spoke"] is True, (
                f"Expected spoke=True for <speak> reply, got {body['interaction']}"
            )
            assert "Blue square visible." in body["interaction"]["text"], (
                f"Unexpected interaction text: {body['interaction']['text']!r}"
            )

            reset_resp = client.post(
                "/v1/streaming/reset", headers={"x-session-id": "sess-0"}
            )
            assert reset_resp.status_code == 200, (
                f"POST /v1/streaming/reset returned {reset_resp.status_code}"
            )
            assert reset_resp.json() == {"status": "reset"}, (
                f"Unexpected reset response: {reset_resp.json()}"
            )
