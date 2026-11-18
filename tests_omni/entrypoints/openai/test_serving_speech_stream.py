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
"""TPU feature test for streaming text-to-speech over WebSocket."""

from __future__ import annotations

import json
import time
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
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.serving_speech_stream import (
    OmniStreamingSpeechHandler,
)
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (  # noqa: E501
    Qwen3TTSTokenizerV2Config,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

STREAM_TTS_SEED = 42
PCM16_MAX_ABS = 1.0 / 32767.0


def _parse_serve_args(model_dir: Path) -> Any:
    parser = TrackingArgumentParser(description="vLLM OMNI CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    return parser.parse_args(
        ["serve", "--model", str(model_dir), "--omni", "--disable-log-stats"]
    )


def _write_minimal_tokenizer(target_dir: Path, vocab_size: int = 256) -> None:
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    words = ["Hello", "world", "world.", "vivian"]
    vocab = {tok: idx for idx, tok in enumerate(special + words)}
    for idx in range(len(vocab), vocab_size):
        vocab[f"tok_{idx}"] = idx
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(special)
    tok.save(str(target_dir / "tokenizer.json"))
    (target_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )


def _build_tiny_qwen3_tts_dir(root: Path) -> Path:
    model_dir = root / "tiny-qwen3-tts"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = Qwen3TTSConfig(
        tts_model_type="custom_voice",
        tts_model_size="0b6",
        tokenizer_type="qwen3_tts_tokenizer_12hz",
        im_start_token_id=1,
        im_end_token_id=2,
        tts_pad_token_id=3,
        tts_bos_token_id=4,
        tts_eos_token_id=5,
        talker_config={
            "vocab_size": 3072,
            "text_vocab_size": 256,
            "hidden_size": 256,
            "text_hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "num_code_groups": 16,
            "spk_id": {"vivian": 10},
            "code_predictor_config": {
                "vocab_size": 2048,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 128,
            },
        },
    )
    cfg.architectures = ["Qwen3TTSForConditionalGeneration"]
    cfg.save_pretrained(str(model_dir))
    _write_minimal_tokenizer(model_dir, vocab_size=256)
    save_file(
        {"dummy.weight": torch.zeros(1, dtype=torch.bfloat16)},
        str(model_dir / "model.safetensors"),
    )

    speech_tok_dir = model_dir / "speech_tokenizer"
    speech_tok_dir.mkdir(parents=True, exist_ok=True)
    tok_v2_cfg = Qwen3TTSTokenizerV2Config(
        decoder_config={
            "codebook_size": 2048,
            "hidden_size": 128,
            "latent_dim": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_quantizers": 16,
            "decoder_dim": 128,
        }
    )
    tok_v2_cfg.save_pretrained(str(speech_tok_dir))
    save_file(
        {"decoder.dummy.weight": torch.zeros(1, dtype=torch.bfloat16)},
        str(speech_tok_dir / "model.safetensors"),
    )
    return model_dir


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "TPUWorker lacks an Omni connector model runner for multi-stage TTS "
        "(vllm-torchtpu#2065)"
    ),
)
def test_streaming_text_to_speech_websocket(tmp_path: Path) -> None:
    """Streaming Text-to-Speech WebSocket: WS /v1/audio/speech/stream.

    - TPU path: multi-stage Qwen3-TTS pipeline (Talker LLM_AR -> Code2Wav
      LLM_GENERATION) serving incremental text-to-PCM chunks on TPU.
    - Setup check: active omni platform is TPU and app state registers
      OmniStreamingSpeechHandler at /v1/audio/speech/stream.
    - Outcome check: concatenated binary PCM16 chunks from incremental
      input.text messages match greedy POST /v1/audio/speech on the full text.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"streaming TTS setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    model_dir = _build_tiny_qwen3_tts_dir(tmp_path)
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

    with TestClient(app) as client:
        assert isinstance(
            app.state.openai_streaming_speech, OmniStreamingSpeechHandler
        ), (
            "streaming TTS setup: openai_streaming_speech="
            f"{type(app.state.openai_streaming_speech).__name__}, "
            "expected OmniStreamingSpeechHandler"
        )

        pcm_chunks: list[bytes] = []
        saw_audio_done = False
        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json(
                {
                    "type": "session.config",
                    "model": str(model_dir),
                    "voice": "vivian",
                    "response_format": "pcm",
                    "stream_audio": True,
                    "seed": STREAM_TTS_SEED,
                    "max_new_tokens": 8,
                }
            )
            ws.send_json({"type": "input.text", "text": "Hello "})
            ws.send_json({"type": "input.text", "text": "world."})
            ws.send_json({"type": "input.done"})

            while True:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    pcm_chunks.append(frame["bytes"])
                    continue
                payload = json.loads(frame["text"])
                assert payload["type"] != "error", (
                    f"streaming TTS WebSocket returned error: {payload}"
                )
                if payload["type"] == "audio.done":
                    saw_audio_done = True
                if payload["type"] == "session.done":
                    break
            ws.send_json({"type": "session.close"})

        assert saw_audio_done is True, (
            "streaming TTS outcome: did not receive audio.done event"
        )
        stream_pcm = b"".join(pcm_chunks)
        assert len(stream_pcm) > 0, (
            "streaming TTS outcome: received 0 PCM bytes over WebSocket"
        )

        http_resp = client.post(
            "/v1/audio/speech",
            json={
                "model": str(model_dir),
                "input": "Hello world.",
                "voice": "vivian",
                "response_format": "pcm",
                "seed": STREAM_TTS_SEED,
                "max_new_tokens": 8,
                "extra_params": {"temperature": 0.0},
            },
        )
        assert http_resp.status_code == 200, (
            f"reference POST /v1/audio/speech status: got {http_resp.status_code}, "
            f"expected 200 ({http_resp.text})"
        )
        ws_wave = torch.from_numpy(
            np.frombuffer(stream_pcm, dtype="<i2").astype(np.float32) / 32768.0
        )
        ref_wave = torch.from_numpy(
            np.frombuffer(http_resp.content, dtype="<i2").astype(np.float32) / 32768.0
        )
        assert_parity(
            ws_wave,
            ref_wave,
            name="WS /v1/audio/speech/stream vs POST /v1/audio/speech",
            max_abs=PCM16_MAX_ABS,
        )
