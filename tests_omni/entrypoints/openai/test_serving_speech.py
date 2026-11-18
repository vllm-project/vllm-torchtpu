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
"""TPU feature tests for OpenAI speech synthesis and voice management routes."""

from __future__ import annotations

import copy
import io
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.omnivoice import OmniVoiceAdapter
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni]

SPEECH_SAMPLE_RATE = 24000
SPEECH_STEPS = 2
SPEECH_SEED = 42
PCM16_MAX_ABS = 1.0 / 32767.0
VOICE_PARITY_MAX_ABS = 1e-6


@dataclass(frozen=True)
class _VoiceUploadCase:
    name: str
    post_kwargs: dict[str, Any]
    tensor_key: str
    expected_tensor: torch.Tensor
    expected_source: str


def _make_audio_sample_case() -> _VoiceUploadCase:
    t = np.linspace(0.0, 1.2, int(SPEECH_SAMPLE_RATE * 1.2), endpoint=False)
    waveform = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, SPEECH_SAMPLE_RATE, format="WAV", subtype="FLOAT")
    wav_bytes = buf.getvalue()
    return _VoiceUploadCase(
        name="tpu_wav_voice",
        post_kwargs={
            "data": {
                "name": "tpu_wav_voice",
                "consent": "consent_wav_01",
                "ref_text": "hello from tpu",
            },
            "files": {"audio_sample": ("sample.wav", wav_bytes, "audio/wav")},
        },
        tensor_key="audio",
        expected_tensor=torch.from_numpy(waveform),
        expected_source="audio",
    )


def _make_speaker_embedding_case() -> _VoiceUploadCase:
    values = [float((idx % 17) - 8) / 16.0 for idx in range(64)]
    return _VoiceUploadCase(
        name="tpu_emb_voice",
        post_kwargs={
            "data": {
                "name": "tpu_emb_voice",
                "consent": "consent_emb_01",
                "speaker_embedding": json.dumps(values),
            },
        },
        tensor_key="speaker_embedding",
        expected_tensor=torch.tensor(values, dtype=torch.float32),
        expected_source="direct",
    )


def _parse_serve_args(model_dir: Path) -> Any:
    parser = TrackingArgumentParser(description="vLLM OMNI CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    return parser.parse_args(
        ["serve", "--model", str(model_dir), "--omni", "--disable-log-stats"]
    )


def _build_tiny_omnivoice_dir(root: Path) -> Path:
    model_dir = root / "tiny-omnivoice"
    model_dir.mkdir(parents=True, exist_ok=True)
    hidden = 256
    layers = 2
    heads = 2
    kv_heads = 1
    head_dim = 128
    inter = 512
    vocab_size = 128
    audio_vocab = 65
    num_cb = 8

    cfg = {
        "model_type": "omnivoice",
        "architectures": ["OmniVoiceModel"],
        "audio_vocab_size": audio_vocab,
        "audio_mask_id": audio_vocab - 1,
        "num_audio_codebook": num_cb,
        "num_step": SPEECH_STEPS,
        "sample_rate": SPEECH_SAMPLE_RATE,
        "frame_rate": 25,
        "enable_cuda_graph": False,
        "llm_config": {
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "intermediate_size": inter,
            "vocab_size": vocab_size,
            "head_dim": head_dim,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    special_tokens = [
        "<unk>",
        "<|denoise|>",
        "<|lang_start|>",
        "<|lang_end|>",
        "<|instruct_start|>",
        "<|instruct_end|>",
        "<|text_start|>",
        "<|text_end|>",
    ]
    words = ["None", "Hello", "world.", "Nice", "to", "meet", "you."]
    vocab = {tok: idx for idx, tok in enumerate(special_tokens + words)}
    for idx in range(len(vocab), vocab_size):
        vocab[f"tok_{idx}"] = idx
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(special_tokens)
    tok.save(str(model_dir / "tokenizer.json"))

    gen = torch.Generator(device="cpu").manual_seed(0)

    def _w(*shape: int) -> torch.Tensor:
        return (torch.randn(shape, generator=gen, dtype=torch.bfloat16) * 0.02).clone()

    weights: dict[str, torch.Tensor] = {
        "llm.embed_tokens.weight": _w(vocab_size, hidden),
        "llm.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "audio_embeddings.weight": _w(num_cb * audio_vocab, hidden),
        "audio_heads.weight": _w(num_cb * audio_vocab, hidden),
    }
    for idx in range(layers):
        p = f"llm.layers.{idx}"
        weights[f"{p}.input_layernorm.weight"] = torch.ones(
            hidden, dtype=torch.bfloat16
        )
        weights[f"{p}.post_attention_layernorm.weight"] = torch.ones(
            hidden, dtype=torch.bfloat16
        )
        weights[f"{p}.self_attn.q_proj.weight"] = _w(heads * head_dim, hidden)
        weights[f"{p}.self_attn.k_proj.weight"] = _w(kv_heads * head_dim, hidden)
        weights[f"{p}.self_attn.v_proj.weight"] = _w(kv_heads * head_dim, hidden)
        weights[f"{p}.self_attn.o_proj.weight"] = _w(hidden, heads * head_dim)
        weights[f"{p}.self_attn.q_norm.weight"] = torch.ones(
            head_dim, dtype=torch.bfloat16
        )
        weights[f"{p}.self_attn.k_norm.weight"] = torch.ones(
            head_dim, dtype=torch.bfloat16
        )
        weights[f"{p}.mlp.gate_proj.weight"] = _w(inter, hidden)
        weights[f"{p}.mlp.up_proj.weight"] = _w(inter, hidden)
        weights[f"{p}.mlp.down_proj.weight"] = _w(hidden, inter)
    save_file(weights, str(model_dir / "model.safetensors"))

    audio_tok_cfg = HiggsAudioV2TokenizerConfig(
        sample_rate=SPEECH_SAMPLE_RATE,
        target_bandwidths=[1.2],
        codebook_size=audio_vocab - 1,
        codebook_dim=16,
        acoustic_model_config={
            "model_type": "dac",
            "encoder_hidden_size": 8,
            "downsampling_ratios": [8, 5, 4, 6],
            "decoder_hidden_size": 32,
            "upsampling_ratios": [8, 5, 4, 6],
            "hidden_size": 16,
        },
        semantic_model_config={
            "model_type": "hubert",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "conv_dim": [16, 16],
            "conv_stride": [2, 2],
            "conv_kernel": [3, 3],
        },
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        audio_tok = HiggsAudioV2TokenizerModel(audio_tok_cfg)
        with torch.no_grad():
            for param in audio_tok.parameters():
                param.mul_(0.05)
    audio_tok.save_pretrained(str(model_dir / "audio_tokenizer"))
    return model_dir


async def _generate_direct_omnivoice(
    engine: AsyncOmni, prompt_text: str, seed: int, steps: int
) -> torch.Tensor:
    sampling_params = copy.deepcopy(engine.default_sampling_params_list)
    sampling_params[0].num_inference_steps = steps
    sampling_params[0].extra_args = {"seed": seed, "num_inference_steps": steps}
    final_output = None
    async for out in engine.generate(
        prompt={"input": prompt_text},
        request_id="direct-omnivoice-ref",
        sampling_params_list=sampling_params,
        output_modalities=["audio"],
    ):
        final_output = out
    assert final_output is not None, "Direct AsyncOmni.generate returned no output"
    mm = final_output.multimodal_output
    audio = mm.get("audio", mm.get("model_outputs"))
    return torch.as_tensor(audio, dtype=torch.float32).squeeze()


@pytest.mark.nightly
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="OmniVoice load_weights passes device='tpu' to safetensors.torch.load_file",
)
def test_audio_speech_synthesis_api(tmp_path: Path) -> None:
    """Audio Speech Synthesis API: POST /v1/audio/speech on pure-diffusion TTS.

    - TPU path: single-stage inline AsyncOmni running OmniVoicePipeline
      iterative unmasking and DAC waveform decode on TPU.
    - Setup check: active omni platform is TPU and app state wires
      OmniOpenAIServingSpeech in diffusion mode with OmniVoiceAdapter.
    - Outcome check: HTTP 200 audio/wav waveform matches direct
      AsyncOmni.generate output for the same seed and step count.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"speech synthesis setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    model_dir = _build_tiny_omnivoice_dir(tmp_path)
    args = _parse_serve_args(model_dir)
    engines: list[AsyncOmni] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = AsyncOmni(
            model=str(model_dir),
            stage_init_timeout=120,
            init_timeout=180,
        )
        engines.append(engine)
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

    prompt_text = "Hello world."
    with TestClient(app) as client:
        speech_handler = app.state.openai_serving_speech
        assert speech_handler._diffusion_mode is True, (
            "speech synthesis setup: _diffusion_mode="
            f"{speech_handler._diffusion_mode!r}, expected True"
        )
        assert isinstance(speech_handler._adapter, OmniVoiceAdapter), (
            "speech synthesis setup: adapter="
            f"{type(speech_handler._adapter).__name__}, expected OmniVoiceAdapter"
        )

        resp = client.post(
            "/v1/audio/speech",
            json={
                "model": str(model_dir),
                "input": prompt_text,
                "response_format": "wav",
                "seed": SPEECH_SEED,
                "extra_params": {"num_inference_steps": SPEECH_STEPS},
            },
        )
        assert resp.status_code == 200, (
            f"speech synthesis HTTP status: got {resp.status_code}, "
            f"expected 200 ({resp.text})"
        )
        assert resp.headers["content-type"].startswith("audio/wav"), (
            f"speech synthesis content-type: got {resp.headers['content-type']!r}, "
            "expected 'audio/wav'"
        )
        wav_np, wav_sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        assert wav_sr == SPEECH_SAMPLE_RATE, (
            f"speech synthesis sample_rate: got {wav_sr}, expected {SPEECH_SAMPLE_RATE}"
        )
        assert client.portal is not None, "TestClient portal was not initialized"
        ref_audio = client.portal.call(
            _generate_direct_omnivoice,
            engines[0],
            prompt_text,
            SPEECH_SEED,
            SPEECH_STEPS,
        )
        assert_parity(
            torch.from_numpy(wav_np),
            ref_audio.clamp(-1.0, 32767.0 / 32768.0),
            name="POST /v1/audio/speech vs direct AsyncOmni.generate",
            max_abs=PCM16_MAX_ABS,
        )


@pytest.mark.timeout(PR_TIMEOUT_S)
@pytest.mark.parametrize(
    "voice_case",
    [_make_audio_sample_case(), _make_speaker_embedding_case()],
    ids=["audio_sample", "speaker_embedding"],
)
def test_audio_voices_management_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    voice_case: _VoiceUploadCase,
) -> None:
    """Audio Voices Management API: GET, POST, and DELETE /v1/audio/voices.

    - TPU path: host-only voice registry and safetensors persistence routes
      executed with the TPU omni platform active.
    - Setup check: active omni platform is TPU, uploaded_speakers_dir resolves
      to SPEAKER_SAMPLES_DIR, and initial GET /v1/audio/voices is empty.
    - Outcome check: POST persists a safetensors tensor matching the uploaded
      waveform or speaker embedding, GET lists metadata, and DELETE removes it.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"{voice_case.name} setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path))
    handler = OmniOpenAIServingSpeech.for_diffusion(
        diffusion_engine=None,
        model_name="omnivoice",
    )
    assert handler.uploaded_speakers_dir == tmp_path, (
        f"{voice_case.name} setup: uploaded_speakers_dir="
        f"{handler.uploaded_speakers_dir}, expected {tmp_path}"
    )

    app = FastAPI()
    app.state.openai_serving_speech = handler
    app.state.api_server_count = 1
    app.include_router(api_server.router)

    with TestClient(app) as client:
        init_resp = client.get("/v1/audio/voices")
        assert init_resp.status_code == 200, (
            f"{voice_case.name} initial GET status: got {init_resp.status_code}, "
            "expected 200"
        )
        assert init_resp.json()["voices"] == ["default"], (
            f"{voice_case.name} initial voices: got {init_resp.json()['voices']}, "
            "expected ['default']"
        )
        assert init_resp.json()["uploaded_voices"] == [], (
            f"{voice_case.name} initial uploaded_voices: got "
            f"{init_resp.json()['uploaded_voices']}, expected []"
        )

        post_resp = client.post("/v1/audio/voices", **voice_case.post_kwargs)
        assert post_resp.status_code == 200, (
            f"{voice_case.name} POST status: got {post_resp.status_code}, "
            f"expected 200 ({post_resp.text})"
        )
        assert post_resp.json()["success"] is True, (
            f"{voice_case.name} POST success: got {post_resp.json()['success']!r}, "
            "expected True"
        )

        saved_files = list(tmp_path.glob(f"{voice_case.name}_*.safetensors"))
        assert len(saved_files) == 1, (
            f"{voice_case.name} persisted files: count={len(saved_files)}, expected 1"
        )
        persisted = load_file(str(saved_files[0]))[voice_case.tensor_key]
        assert_parity(
            persisted,
            voice_case.expected_tensor,
            name=f"{voice_case.name} persisted {voice_case.tensor_key}",
            max_abs=VOICE_PARITY_MAX_ABS,
        )

        list_resp = client.get("/v1/audio/voices")
        assert list_resp.status_code == 200, (
            f"{voice_case.name} GET after upload status: got {list_resp.status_code}, "
            "expected 200"
        )
        expected_voices = sorted(["default", voice_case.name])
        assert list_resp.json()["voices"] == expected_voices, (
            f"{voice_case.name} listed voices: got {list_resp.json()['voices']}, "
            f"expected {expected_voices}"
        )
        uploaded_entries = list_resp.json()["uploaded_voices"]
        assert len(uploaded_entries) == 1, (
            f"{voice_case.name} uploaded_voices count: got {len(uploaded_entries)}, "
            "expected 1"
        )
        assert uploaded_entries[0]["embedding_source"] == voice_case.expected_source, (
            f"{voice_case.name} embedding_source: got "
            f"{uploaded_entries[0]['embedding_source']!r}, "
            f"expected {voice_case.expected_source!r}"
        )

        del_resp = client.delete(f"/v1/audio/voices/{voice_case.name}")
        assert del_resp.status_code == 200, (
            f"{voice_case.name} DELETE status: got {del_resp.status_code}, expected 200"
        )
        assert not saved_files[0].exists(), (
            f"{voice_case.name} file after DELETE: {saved_files[0]} still exists"
        )

        after_resp = client.get("/v1/audio/voices")
        assert after_resp.json()["voices"] == ["default"], (
            f"{voice_case.name} voices after DELETE: "
            f"got {after_resp.json()['voices']}, expected ['default']"
        )
        repeat_del = client.delete(f"/v1/audio/voices/{voice_case.name}")
        assert repeat_del.status_code == 404, (
            f"{voice_case.name} repeat DELETE status: got {repeat_del.status_code}, "
            "expected 404"
        )
