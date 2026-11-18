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
"""TPU feature test for streaming video input chat over WebSocket."""

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
from PIL import Image
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.serving_video_stream import (
    QwenOmniStreamingVideoHandler,
)
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

VIDEO_AUDIO_SAMPLE_RATE = 16000
VIDEO_MAX_TOKENS = 8


def _parse_serve_args(model_dir: Path) -> Any:
    parser = TrackingArgumentParser(description="vLLM OMNI CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    return parser.parse_args(
        ["serve", "--model", str(model_dir), "--omni", "--disable-log-stats"]
    )


def _build_tiny_qwen3_omni_dir(root: Path) -> Path:
    model_dir = root / "tiny-qwen3-omni"
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
    words = ["user", "assistant", "Describe"]
    vocab = {tok: idx for idx, tok in enumerate(special + words)}
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


def _make_png_b64() -> str:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :16, 0] = 200
    arr[:, 16:, 2] = 180
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_pcm16_and_wav_b64() -> tuple[str, str]:
    samples = np.zeros(1600, dtype="<i2")
    pcm_bytes = samples.tobytes()
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(VIDEO_AUDIO_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return (
        base64.b64encode(pcm_bytes).decode("ascii"),
        base64.b64encode(wav_buf.getvalue()).decode("ascii"),
    )


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "TPUWorker lacks an Omni connector model runner for multi-stage omni chat "
        "(vllm-torchtpu#2065)"
    ),
)
def test_streaming_video_input_chat_websocket(tmp_path: Path) -> None:
    """Streaming Video Input Chat WebSocket: WS /v1/video/chat/stream.

    - TPU path: multi-stage Qwen3-Omni-MoE pipeline buffering streamed video
      frames and PCM16 audio chunks and running Thinker AR inference on TPU.
    - Setup check: active omni platform is TPU and app state registers
      QwenOmniStreamingVideoHandler at /v1/video/chat/stream.
    - Outcome check: greedy response.text.done text from /v1/video/chat/stream
      matches greedy POST /v1/chat/completions on the same frame and audio turn.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"streaming video setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    model_dir = _build_tiny_qwen3_omni_dir(tmp_path)
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

    frame_b64 = _make_png_b64()
    pcm_b64, wav_b64 = _make_pcm16_and_wav_b64()
    query_text = "Describe the frame."

    with TestClient(app) as client:
        assert isinstance(
            app.state.openai_streaming_video, QwenOmniStreamingVideoHandler
        ), (
            "streaming video setup: openai_streaming_video="
            f"{type(app.state.openai_streaming_video).__name__}, "
            "expected QwenOmniStreamingVideoHandler"
        )

        ws_text: str | None = None
        with client.websocket_connect("/v1/video/chat/stream") as ws:
            ws.send_json(
                {
                    "type": "session.config",
                    "model": str(model_dir),
                    "modalities": ["text"],
                    "sampling_params_list": [
                        {"temperature": 0.0, "max_tokens": VIDEO_MAX_TOKENS}
                    ],
                }
            )
            ws.send_json(
                {
                    "type": "video.frame",
                    "data": frame_b64,
                    "frame_id": "f0",
                    "pts_ms": 0,
                }
            )
            ws.send_json({"type": "audio.chunk", "data": pcm_b64})
            ws.send_json({"type": "video.query", "text": query_text})

            while ws_text is None:
                msg = ws.receive_json()
                assert msg["type"] != "error", (
                    f"streaming video WebSocket returned error: {msg}"
                )
                if msg["type"] == "response.text.done":
                    ws_text = msg["text"]

            ws.send_json({"type": "video.done"})
            done_msg = ws.receive_json()
            assert done_msg["type"] == "session.done", (
                f"streaming video close: got {done_msg['type']!r}, "
                "expected 'session.done'"
            )

        http_resp = client.post(
            "/v1/chat/completions",
            json={
                "model": str(model_dir),
                "modalities": ["text"],
                "temperature": 0.0,
                "max_tokens": VIDEO_MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{frame_b64}",
                                },
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_b64, "format": "wav"},
                            },
                            {"type": "text", "text": query_text},
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
            f"WS /v1/video/chat/stream text {ws_text!r} != "
            f"POST /v1/chat/completions text {http_text!r}"
        )
