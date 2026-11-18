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
"""TPU feature test for full-duplex Realtime WebSocket serving."""

from __future__ import annotations

import base64
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
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni
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

DUPLEX_DEPLOY_CONFIG = "qwen3_omni_duplex.yaml"
DUPLEX_MAX_TOKENS = 8


def _parse_serve_args(model_dir: Path) -> Any:
    parser = TrackingArgumentParser(description="vLLM OMNI CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    return parser.parse_args(
        [
            "serve",
            "--model",
            str(model_dir),
            "--omni",
            "--deploy-config",
            DUPLEX_DEPLOY_CONFIG,
            "--disable-log-stats",
        ]
    )


def _build_tiny_qwen3_omni_dir(root: Path) -> Path:
    model_dir = root / "tiny-qwen3-omni-duplex"
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
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {"dummy.weight": torch.zeros(1, dtype=torch.bfloat16)},
        str(model_dir / "model.safetensors"),
    )
    return model_dir


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "TPUWorker lacks an Omni connector model runner for duplex omni stages "
        "(vllm-torchtpu#2065)"
    ),
)
def test_realtime_duplex_websocket(tmp_path: Path) -> None:
    """Realtime Duplex WebSocket: WS /v1/realtime?duplex=1 and /v1/duplex.

    - TPU path: DuplexOmni running Qwen3-Omni-MoE (session_mode=duplex) for
      bidirectional audio turns and resume_token session re-attachment on TPU.
    - Setup check: active omni platform is TPU, _should_serve_duplex selects
      duplex mode, and app state mounts OmniDuplexSessionHandler.
    - Outcome check: /v1/realtime?duplex=1 emits session.created with
      resume_token, completes a greedy audio turn, and resumes on /v1/duplex.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"duplex realtime setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    model_dir = _build_tiny_qwen3_omni_dir(tmp_path)
    should_duplex = api_server._should_serve_duplex(
        str(model_dir),
        {"deploy_config": DUPLEX_DEPLOY_CONFIG},
    )
    assert should_duplex is True, (
        f"duplex realtime setup: _should_serve_duplex={should_duplex!r}, expected True"
    )
    args = _parse_serve_args(model_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = DuplexOmni(
            model=str(model_dir),
            deploy_config=DUPLEX_DEPLOY_CONFIG,
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

    pcm_b64 = base64.b64encode(np.zeros(2400, dtype="<i2").tobytes()).decode("ascii")

    with TestClient(app) as client:
        assert isinstance(app.state.openai_serving_duplex, OmniDuplexSessionHandler), (
            "duplex realtime setup: openai_serving_duplex="
            f"{type(app.state.openai_serving_duplex).__name__}, "
            "expected OmniDuplexSessionHandler"
        )

        with client.websocket_connect("/v1/realtime?duplex=1") as ws:
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "turn_detection": None,
                    },
                }
            )
            created = ws.receive_json()
            assert created["type"] == "session.created", (
                f"duplex handshake: got {created['type']!r}, expected 'session.created'"
            )
            session_id = created["session"]["id"]
            resume_token = created.get("resume_token")
            assert isinstance(resume_token, str), (
                f"duplex handshake: expected str resume_token, got "
                f"{type(resume_token).__name__} in {created}"
            )
            assert len(resume_token) > 0, (
                f"duplex handshake: expected non-empty resume_token in {created}"
            )

            ws.send_json({"type": "input_audio_buffer.append", "audio": pcm_b64})
            ws.send_json({"type": "input_audio_buffer.commit"})
            ws.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "temperature": 0.0,
                        "max_output_tokens": DUPLEX_MAX_TOKENS,
                    },
                }
            )
            done_event = None
            while done_event is None:
                event = ws.receive_json()
                assert event["type"] != "error", (
                    f"duplex turn returned error event: {event}"
                )
                if event["type"] == "response.done":
                    done_event = event
            assert done_event["response"]["status"] == "completed", (
                f"duplex response status: got {done_event['response']['status']!r}, "
                "expected 'completed'"
            )

        with client.websocket_connect("/v1/duplex") as ws_resume:
            ws_resume.send_json(
                {
                    "type": "session.resume",
                    "session_id": session_id,
                    "resume_token": resume_token,
                    "last_received_server_event_seq": 0,
                }
            )
            resumed = ws_resume.receive_json()
            assert resumed["type"] == "session.resumed", (
                f"duplex resume: got {resumed['type']!r}, expected 'session.resumed'"
            )
            assert resumed["session_id"] == session_id, (
                f"duplex resume session_id: got {resumed['session_id']!r}, "
                f"expected {session_id!r}"
            )
