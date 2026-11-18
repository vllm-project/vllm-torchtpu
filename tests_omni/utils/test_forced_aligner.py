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
"""TPU feature test for streaming TTS forced-aligner word timestamps."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (  # noqa: E501
    Qwen3TTSTokenizerV2Config,
)
from vllm_omni.model_executor.stage_input_processors.forced_aligner import (
    POOLING_OUTPUT_DECODER_PATH,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.forced_aligner import extract_word_timestamps
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

ALIGNER_SEED = 42
EXPECTED_WORDS = ["Hello", "world"]

_AlignerCliBuilder = Callable[[Path, Path], list[str]]


def _cli_flag_args(aligner_dir: Path, _tmp_path: Path) -> list[str]:
    return [
        "--forced-aligner",
        str(aligner_dir),
        "--forced-aligner-device",
        "0",
    ]


def _config_yaml_args(aligner_dir: Path, tmp_path: Path) -> list[str]:
    cfg_path = tmp_path / "forced_aligner_override.yaml"
    cfg_path.write_text(
        "forced_aligner:\n"
        f"  model: {aligner_dir}\n"
        "  runner: pooling\n"
        "  architecture: Qwen3ASRForcedAlignerForTokenClassification\n"
        "  pooling_task: token_classify\n"
        "  gpu_memory_utilization: 0.30\n"
        "  dtype: bfloat16\n"
        "  max_model_len: 512\n",
        encoding="utf-8",
    )
    return [
        "--forced-aligner-config",
        str(cfg_path),
        "--forced-aligner-device",
        "0",
    ]


def _parse_serve_args(model_dir: Path, extra_flags: list[str]) -> Any:
    parser = TrackingArgumentParser(description="vLLM OMNI CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    args = parser.parse_args(
        [
            "serve",
            str(model_dir),
            "--omni",
            "--disable-log-stats",
            *extra_flags,
        ]
    )
    if args.model_tag is not None:
        args.model = args.model_tag
    return args


def _write_minimal_tokenizer(target_dir: Path, vocab_size: int = 256) -> None:
    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|audio_start|>",
        "<|audio_pad|>",
        "<|audio_end|>",
        "<timestamp>",
    ]
    words = ["Hello", "world", "vivian"]
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


def _build_tiny_qwen3_tts_and_aligner_dirs(root: Path) -> tuple[Path, Path]:
    tts_dir = root / "tiny-qwen3-tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    tts_cfg = Qwen3TTSConfig(
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
    tts_cfg.architectures = ["Qwen3TTSForConditionalGeneration"]
    tts_cfg.save_pretrained(str(tts_dir))
    _write_minimal_tokenizer(tts_dir, vocab_size=256)
    save_file(
        {"dummy.weight": torch.zeros(1, dtype=torch.bfloat16)},
        str(tts_dir / "model.safetensors"),
    )

    speech_tok_dir = tts_dir / "speech_tokenizer"
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

    aligner_dir = root / "tiny-qwen3-forced-aligner"
    aligner_dir.mkdir(parents=True, exist_ok=True)
    aligner_cfg = {
        "model_type": "qwen2",
        "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1024,
        "timestamp_token_id": 6,
        "timestamp_segment_time": 80.0,
        "classify_num": 128,
    }
    (aligner_dir / "config.json").write_text(
        json.dumps(aligner_cfg),
        encoding="utf-8",
    )
    _write_minimal_tokenizer(aligner_dir, vocab_size=256)
    save_file(
        {"dummy.weight": torch.zeros(1, dtype=torch.bfloat16)},
        str(aligner_dir / "model.safetensors"),
    )
    return tts_dir, aligner_dir


@pytest.mark.parametrize(
    "cli_builder",
    [_cli_flag_args, _config_yaml_args],
    ids=["cli_flag", "config_yaml"],
)
@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "TPUWorker lacks an Omni connector model runner for forced-aligner TTS "
        "stages (vllm-torchtpu#2065)"
    ),
)
def test_streaming_tts_forced_aligner_word_timestamps(
    tmp_path: Path,
    cli_builder: _AlignerCliBuilder,
) -> None:
    """Streaming TTS Forced Aligner Word Timestamps: --forced-aligner on TTS.

    - TPU path: 3-stage Qwen3-TTS + pooling Qwen3-ForcedAligner pipeline
      emitting per-word [start_ms, end_ms] timestamps on TPU.
    - Setup check: active omni platform is TPU, resolve_omni_config appends a
      pooling forced_aligner LLM_AR stage, and forced_aligner_enabled is True.
    - Outcome check: WS /v1/audio/speech/stream (word_timestamps=True) emits a
      trailing alignment chunk matching extract_word_timestamps in [0, end_ms].
    """
    assert current_omni_platform.device_type == "tpu", (
        f"forced aligner setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    tts_dir, aligner_dir = _build_tiny_qwen3_tts_and_aligner_dirs(tmp_path)
    extra_flags = cli_builder(aligner_dir, tmp_path)
    args = _parse_serve_args(tts_dir, extra_flags)

    resolution = resolve_omni_config(
        args.model,
        trust_remote_code=args.trust_remote_code,
        deploy_config_path=args.deploy_config,
        cli_overrides={
            "forced_aligner": args.forced_aligner,
            "forced_aligner_config": args.forced_aligner_config,
            "forced_aligner_device": args.forced_aligner_device,
        },
        stage_overrides=args.stage_overrides,
        strategy_config_path=args.strategy_config,
    )
    assert resolution.pipeline_config is not None, (
        "forced aligner setup: expected resolved pipeline_config"
    )
    assert len(resolution.stage_configs) == 3, (
        f"forced aligner setup: stage count={len(resolution.stage_configs)}, expected 3"
    )
    tail_pipeline_stage = resolution.pipeline_config.stages[-1]
    assert tail_pipeline_stage.model_stage == "forced_aligner", (
        f"forced aligner setup: tail model_stage={tail_pipeline_stage.model_stage!r}, "
        "expected 'forced_aligner'"
    )
    assert tail_pipeline_stage.execution_type == StageExecutionType.LLM_AR, (
        "forced aligner setup: tail execution_type="
        f"{tail_pipeline_stage.execution_type!r}, expected StageExecutionType.LLM_AR"
    )
    tail_pooling = resolution.stage_configs[-1].pooling_config
    assert tail_pooling.runner == "pooling", (
        f"forced aligner setup: runner={tail_pooling.runner!r}, expected 'pooling'"
    )
    assert tail_pooling.pooling_output_decoder == POOLING_OUTPUT_DECODER_PATH, (
        "forced aligner setup: pooling_output_decoder="
        f"{tail_pooling.pooling_output_decoder!r}, "
        f"expected {POOLING_OUTPUT_DECODER_PATH!r}"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = AsyncOmni(
            model=args.model,
            forced_aligner=args.forced_aligner,
            forced_aligner_config=args.forced_aligner_config,
            forced_aligner_device=args.forced_aligner_device,
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

    utterance_text = "Hello world"
    with TestClient(app) as client:
        assert app.state.openai_serving_speech.forced_aligner_enabled is True, (
            "forced aligner setup: expected forced_aligner_enabled=True on speech "
            "serving handler"
        )

        audio_chunks: list[dict[str, Any]] = []
        with client.websocket_connect("/v1/audio/speech/stream") as ws:
            ws.send_json(
                {
                    "type": "session.config",
                    "model": str(tts_dir),
                    "voice": "vivian",
                    "response_format": "pcm",
                    "stream_audio": True,
                    "word_timestamps": True,
                    "seed": ALIGNER_SEED,
                    "max_new_tokens": 8,
                }
            )
            ws.send_json({"type": "input.text", "text": utterance_text})
            ws.send_json({"type": "input.done"})

            while True:
                msg = ws.receive_json()
                assert msg["type"] != "error", (
                    f"forced aligner WebSocket returned error: {msg}"
                )
                if msg["type"] == "audio.chunk":
                    audio_chunks.append(msg)
                if msg["type"] == "session.done":
                    break
            ws.send_json({"type": "session.close"})

        assert len(audio_chunks) >= 2, (
            f"forced aligner outcome: expected PCM + trailing alignment chunks, "
            f"got {len(audio_chunks)}"
        )
        trailing = audio_chunks[-1]
        assert trailing["audio_b64"] == "", (
            f"forced aligner trailing chunk audio_b64={trailing['audio_b64']!r}, "
            "expected ''"
        )
        timestamps = trailing["timestamps"]
        assert isinstance(timestamps, list), (
            f"forced aligner timestamps={timestamps!r}, expected list"
        )
        assert [entry["word"] for entry in timestamps] == EXPECTED_WORDS, (
            f"forced aligner words={[entry['word'] for entry in timestamps]}, "
            f"expected {EXPECTED_WORDS}"
        )
        duration_ms = int(trailing["chunk_end_ms"])
        prev_end = 0
        for entry in timestamps:
            assert prev_end <= entry["start_ms"], (
                "forced aligner non-monotonic "
                f"start_ms={entry['start_ms']} < {prev_end}"
            )
            assert entry["start_ms"] <= entry["end_ms"], (
                f"forced aligner inverted interval: {entry}"
            )
            assert entry["end_ms"] <= duration_ms, (
                "forced aligner "
                f"end_ms={entry['end_ms']} exceeds duration {duration_ms}"
            )
            prev_end = entry["end_ms"]

        ms_tensor = torch.tensor(
            [[entry["start_ms"], entry["end_ms"]] for entry in timestamps],
            dtype=torch.int32,
        )
        ref_timestamps = extract_word_timestamps(
            SimpleNamespace(outputs=SimpleNamespace(data=ms_tensor)),
            utterance_text,
        )
        assert timestamps == ref_timestamps, (
            f"forced aligner wire timestamps {timestamps} != "
            f"extract_word_timestamps reference {ref_timestamps}"
        )
