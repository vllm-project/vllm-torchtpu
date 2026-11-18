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
"""TPU feature test for the OpenAI audio generation route."""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import diffusers
import pytest
import soundfile as sf
import torch
from diffusers import (
    AutoencoderOobleck,
    CosineDPMSolverMultistepScheduler,
    StableAudioDiTModel,
)
from diffusers.pipelines.stable_audio.modeling_stable_audio import (
    StableAudioProjectionModel,
)
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from transformers import T5Config, T5EncoderModel, T5TokenizerFast
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.serving_audio_generate import (
    OmniOpenAIServingAudioGenerate,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

AUDIO_GEN_SAMPLE_RATE = 44100
AUDIO_GEN_CHANNELS = 2
AUDIO_GEN_LENGTH_S = 0.01
AUDIO_GEN_STEPS = 2
AUDIO_GEN_GUIDANCE_SCALE = 3.0
AUDIO_GEN_SEED = 42
PCM16_MAX_ABS = 1.0 / 32767.0


def _parse_serve_args(model_dir: Path) -> Any:
    parser = TrackingArgumentParser(description="vLLM OMNI CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    args = parser.parse_args(["serve", str(model_dir), "--omni", "--disable-log-stats"])
    # OmniServeCommand.cmd copies the positional model onto args.model, which
    # omni_init_app_state uses as the served model name.
    args.model = args.model_tag
    return args


def _build_tiny_stable_audio_dir(root: Path) -> Path:
    model_dir = root / "tiny-stable-audio"
    if model_dir.is_dir():
        return model_dir

    tokenizer = T5TokenizerFast.from_pretrained(
        tiny_flux_schnell(),
        subfolder="tokenizer_2",
    )
    tokenizer.model_max_length = 126

    text_hidden = 256
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        text_encoder = T5EncoderModel(
            T5Config(
                vocab_size=tokenizer.vocab_size,
                d_model=text_hidden,
                d_kv=64,
                d_ff=512,
                num_layers=2,
                num_decoder_layers=2,
                num_heads=4,
            )
        ).to(torch.bfloat16)
        vae = AutoencoderOobleck(
            encoder_hidden_size=16,
            downsampling_ratios=[2, 2],
            channel_multiples=[1, 2],
            decoder_channels=16,
            decoder_input_channels=8,
            audio_channels=AUDIO_GEN_CHANNELS,
            sampling_rate=AUDIO_GEN_SAMPLE_RATE,
        )
        with torch.no_grad():
            for param in vae.parameters():
                param.mul_(0.05)
        projection_model = StableAudioProjectionModel(
            text_encoder_dim=text_hidden,
            conditioning_dim=text_hidden,
            min_value=0,
            max_value=512,
        ).to(torch.bfloat16)
        # sample_size=255 gives 1 + 255 = 256 DiT query tokens and 126 + 2 = 128
        # cross-attention tokens, both exact multiples of TPU lane width 128.
        transformer = StableAudioDiTModel(
            sample_size=255,
            in_channels=8,
            out_channels=8,
            num_layers=2,
            attention_head_dim=128,
            num_attention_heads=4,
            num_key_value_attention_heads=2,
            cross_attention_dim=text_hidden,
            cross_attention_input_dim=text_hidden,
            time_proj_dim=64,
            global_states_input_dim=2 * text_hidden,
        ).to(torch.bfloat16)
    scheduler = CosineDPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="v_prediction",
        sigma_data=1.0,
        sigma_schedule="exponential",
    )
    pipeline = diffusers.StableAudioPipeline(
        vae=vae,
        text_encoder=text_encoder,
        projection_model=projection_model,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.save_pretrained(str(model_dir))
    return model_dir


async def _generate_direct_stable_audio(
    engine: AsyncOmni,
    prompt_text: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    audio_length: float,
) -> torch.Tensor:
    rng = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    sampling = OmniDiffusionSamplingParams(
        num_outputs_per_prompt=1,
        generator=rng,
        guidance_scale=guidance_scale,
        guidance_scale_provided=True,
        num_inference_steps=steps,
        extra_args={
            "audio_length": audio_length,
            "audio_start_in_s": 0.0,
            "audio_end_in_s": audio_length,
        },
    )
    final_output = None
    async for out in engine.generate(
        prompt={"prompt": prompt_text},
        request_id="direct-stable-audio-ref",
        sampling_params_list=[sampling],
        output_modalities=["audio"],
    ):
        final_output = out
    assert final_output is not None, "Direct AsyncOmni.generate returned no output"
    mm = final_output.multimodal_output
    raw = mm.get("audio", mm.get("model_outputs"))
    waveform = torch.as_tensor(raw, dtype=torch.float32).squeeze()
    return waveform.transpose(0, 1).contiguous()


def test_audio_generation_api(tmp_path: Path) -> None:
    """Audio Generation API: POST /v1/audio/generate on StableAudioPipeline.

    - TPU path: single-stage inline AsyncOmni running T5 text encoding, 2-layer
      StableAudioDiTModel (head_dim=128), and AutoencoderOobleck decode on TPU.
    - Setup check: active omni platform is TPU and app state registers
      OmniOpenAIServingAudioGenerate with diffusion_mode=True.
    - Outcome check: HTTP 200 stereo 44.1 kHz WAV matches direct
      AsyncOmni.generate output within 16-bit PCM quantization tolerance.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"audio generation setup: device_type={current_omni_platform.device_type!r}, "
        "expected 'tpu'"
    )
    model_dir = _build_tiny_stable_audio_dir(tmp_path)
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

    prompt_text = "warm ambient synth pad"
    with TestClient(app) as client:
        handler = app.state.openai_serving_audio_generate
        assert isinstance(handler, OmniOpenAIServingAudioGenerate), (
            f"audio generation setup: handler={type(handler).__name__}, "
            "expected OmniOpenAIServingAudioGenerate"
        )
        assert handler.diffusion_mode is True, (
            f"audio generation setup: diffusion_mode={handler.diffusion_mode!r}, "
            "expected True"
        )

        resp = client.post(
            "/v1/audio/generate",
            json={
                "model": str(model_dir),
                "input": prompt_text,
                "num_inference_steps": AUDIO_GEN_STEPS,
                "guidance_scale": AUDIO_GEN_GUIDANCE_SCALE,
                "audio_length": AUDIO_GEN_LENGTH_S,
                "seed": AUDIO_GEN_SEED,
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200, (
            f"audio generation HTTP status: got {resp.status_code}, "
            f"expected 200 ({resp.text})"
        )
        assert resp.headers["content-type"].startswith("audio/wav"), (
            f"audio generation content-type: got {resp.headers['content-type']!r}, "
            "expected 'audio/wav'"
        )
        wav_np, wav_sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        assert wav_sr == AUDIO_GEN_SAMPLE_RATE, (
            f"audio generation sample_rate: got {wav_sr}, "
            f"expected {AUDIO_GEN_SAMPLE_RATE}"
        )
        expected_samples = int(AUDIO_GEN_LENGTH_S * AUDIO_GEN_SAMPLE_RATE)
        assert wav_np.shape == (expected_samples, AUDIO_GEN_CHANNELS), (
            f"audio generation shape: got {wav_np.shape}, "
            f"expected {(expected_samples, AUDIO_GEN_CHANNELS)}"
        )

        assert client.portal is not None, "TestClient portal was not initialized"
        ref_audio = client.portal.call(
            _generate_direct_stable_audio,
            engines[0],
            prompt_text,
            AUDIO_GEN_SEED,
            AUDIO_GEN_STEPS,
            AUDIO_GEN_GUIDANCE_SCALE,
            AUDIO_GEN_LENGTH_S,
        )
        assert_parity(
            torch.from_numpy(wav_np),
            ref_audio.clamp(-1.0, 32767.0 / 32768.0),
            name="POST /v1/audio/generate vs direct AsyncOmni.generate",
            max_abs=PCM16_MAX_ABS,
        )
