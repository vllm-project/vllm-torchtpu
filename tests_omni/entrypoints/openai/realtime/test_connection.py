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
"""TPU feature test for turn-based OpenAI Realtime audio WebSocket serving."""

from __future__ import annotations

import base64
import io
import json
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.duplex.openai import OpenAIRealtimeHandler
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

REALTIME_SAMPLE_RATE = 24000
REALTIME_MAX_TOKENS = 8


def _parse_serve_args(model_dir: Path) -> Any:
    parser = TrackingArgumentParser(description="vLLM OMNI CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    return parser.parse_args(
        ["serve", "--model", str(model_dir), "--omni", "--disable-log-stats"]
    )


def _build_tiny_qwen3_omni_dir(root: Path) -> Path:
    model_dir = root / "tiny-qwen3-omni-turn"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = Qwen3OmniMoeConfig(
        enable_audio_output=True,
        thinker_config={
            "text_config": {
                "vocab_size": 256,
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "num_experts": 2,
                "num_experts_per_tok": 1,
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 128,
                "num_heads": 2,
                "intermediate_size": 256,
                "out_hidden_size": 128,
            },
            "audio_config": {
                "d_model": 128,
                "encoder_layers": 2,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 256,
                "output_dim": 128,
            },
        },
        talker_config={
            "text_config": {
                "vocab_size": 256,
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "num_experts": 2,
                "num_experts_per_tok": 1,
            }
        },
        code2wav_config={
            "codebook_size": 256,
            "hidden_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_quantizers": 16,
            "decoder_dim": 128,
        },
    )
    cfg.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    cfg.save_pretrained(str(model_dir))

    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {tok: idx for idx, tok in enumerate(special + ["user", "assistant"])}
    for idx in range(len(vocab), 256):
        vocab[f"tok_{idx}"] = idx
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(special)
    tok.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "chat_template": (
                    "{% for m in messages %}{{ m['role'] }}: "
                    "{% if m['content'] is string %}{{ m['content'] }}{% endif %}"
                    "{% endfor %}assistant:"
                ),
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {"dummy.weight": torch.zeros(1, dtype=torch.bfloat16)},
        str(model_dir / "model.safetensors"),
    )
    return model_dir


def _make_turn_audio_b64() -> tuple[str, str]:
    pcm_bytes = np.zeros(2400, dtype="<i2").tobytes()
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(REALTIME_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return (
        base64.b64encode(pcm_bytes).decode("ascii"),
        base64.b64encode(wav_buf.getvalue()).decode("ascii"),
    )


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "TPUWorker lacks an Omni connector model runner for turn-based realtime "
        "stages (vllm-torchtpu#2065)"
    ),
)
def test_realtime_audio_websocket(tmp_path: Path) -> None:
    """Realtime audio WebSocket (OpenAI Realtime, turn-based): WS /v1/realtime.

    - TPU path: turn-based AsyncOmni Qwen3-Omni-MoE pipeline driven through
      OpenAIRealtimeHandler / OpenAIFullDuplexConnection on TPU.
    - Setup check: active omni platform is TPU, _should_serve_duplex is False,
      openai_serving_duplex is None, and OpenAIRealtimeHandler is registered.
    - Outcome check: client-VAD input_audio_buffer commit + response.create
      emits response.output_text.done matching POST /v1/chat/completions.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"realtime audio setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    model_dir = _build_tiny_qwen3_omni_dir(tmp_path)
    should_duplex = api_server._should_serve_duplex(str(model_dir), {})
    assert should_duplex is False, (
        f"realtime audio setup: _should_serve_duplex={should_duplex!r}, expected False"
    )
    args = _parse_serve_args(model_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = AsyncOmni(
            model=str(model_dir),
            stage_init_timeout=120,
            init_timeout=180,
        )
        try:
            await api_server.omni_init_app_state(engine, app.state, args)
            app.state.api_server_count = 1
            yield
        finally:
            engine.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def stamp_request_time(request: Request, call_next: Any) -> Any:
        request.state.request_timestamp = time.time()
        return await call_next(request)

    app.include_router(api_server.router)

    pcm_b64, wav_b64 = _make_turn_audio_b64()

    with TestClient(app) as client:
        assert app.state.openai_serving_duplex is None, (
            "realtime audio setup: expected openai_serving_duplex to be None"
        )
        assert isinstance(app.state.openai_realtime_handler, OpenAIRealtimeHandler), (
            "realtime audio setup: openai_realtime_handler="
            f"{type(app.state.openai_realtime_handler).__name__}, "
            "expected OpenAIRealtimeHandler"
        )
        assert app.state.openai_serving_chat is not None, (
            "realtime audio setup: expected openai_serving_chat to be initialized"
        )

        ws_text: str | None = None
        with client.websocket_connect("/v1/realtime") as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created", (
                f"realtime handshake: got {created['type']!r}, "
                "expected 'session.created'"
            )

            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["text"],
                        "audio": {"input": {"turn_detection": None}},
                    },
                }
            )
            updated = ws.receive_json()
            assert updated["type"] == "session.updated", (
                f"realtime session.update: got {updated['type']!r}, "
                "expected 'session.updated'"
            )

            ws.send_json({"type": "input_audio_buffer.append", "audio": pcm_b64})
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["text"],
                        "max_output_tokens": REALTIME_MAX_TOKENS,
                    },
                }
            )

            saw_done = False
            while not saw_done:
                event = ws.receive_json()
                assert event["type"] != "error", (
                    f"realtime turn returned error event: {event}"
                )
                if event["type"] == "response.output_text.done":
                    ws_text = event["text"]
                if event["type"] == "response.done":
                    assert event["response"]["status"] == "completed", (
                        f"realtime response.done status: got "
                        f"{event['response']['status']!r}, expected 'completed'"
                    )
                    saw_done = True

        http_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": str(model_dir),
                "modalities": ["text"],
                "temperature": 0.0,
                "max_tokens": REALTIME_MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_b64, "format": "wav"},
                            }
                        ],
                    }
                ],
            },
        )
        assert http_resp.status_code == 200, (
            f"reference POST /v1/chat/completions status: got {http_resp.status_code}, "
            f"expected 200 ({http_resp.text})"
        )
        http_text = http_resp.json()["choices"][0]["message"]["content"]
        assert ws_text == http_text, (
            f"WS /v1/realtime text {ws_text!r} != "
            f"POST /v1/chat/completions text {http_text!r}"
        )
