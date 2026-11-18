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
"""TPU test for the SeedVR2 Long Video Restoration API (/v1/seedvr2/restore-long)."""

from __future__ import annotations

import io
import socket
import subprocess
import threading
import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import imageio_ffmpeg
import numpy as np
import pytest
import requests
import torch
import uvicorn
from fastapi import FastAPI
from torch import nn
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.models.seedvr2.long_video import register_routes
from vllm_omni.diffusion.models.seedvr2.nadit import SeedVR2NaDiT
from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import (
    MAX_CLIP_PIXELS,
    MAX_FRAME_PIXELS,
    SeedVR2Pipeline,
    _seedvr2_post_process,
)
from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, compare_video_to_golden

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

VIDEO_SIZE = 32
FPS = 24
WINDOW_FRAMES = 9
TOTAL_FRAMES = 14
VID_DIM = 64
TXT_DIM = 64
NUM_HEADS = 2
HEAD_DIM = 32
TEXT_TOKENS = 4

# MP4 output uses CRF 18 (yuv420p); gates account for 4:2:0 subsampling + fp16 vs fp32.
MAX_MAE = 0.08
MIN_SSIM = 0.85
SERVER_START_TIMEOUT_S = 15.0
JOB_TIMEOUT_S = 90.0


class _SeedVR2Engine:
    """Single-worker diffusion engine adapter driving a real ``SeedVR2Pipeline``."""

    def __init__(self, pipeline: SeedVR2Pipeline) -> None:
        self.pipeline = pipeline
        self._lock = threading.Lock()
        self.stage_configs = [
            SimpleNamespace(
                stage_type="diffusion",
                final_output=True,
                final_output_type="video",
                diffusion_config=SimpleNamespace(model_class_name="SeedVR2Pipeline"),
            )
        ]
        self.default_sampling_params_list = [OmniDiffusionSamplingParams()]
        self._od_config = SimpleNamespace(
            model_class_name="SeedVR2Pipeline",
            supports_multimodal_inputs=True,
        )

    def get_diffusion_od_config(self) -> Any:
        return self._od_config

    async def generate(
        self,
        prompt: Any,
        request_id: str,
        sampling_params_list: list[OmniDiffusionSamplingParams],
    ):
        params = sampling_params_list[0]
        params.generator = torch.Generator(device="cpu").manual_seed(
            int(params.seed or 0)
        )
        req = OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=params,
            request_id=request_id,
        )
        with self._lock:
            outputs = self.pipeline.forward(DiffusionRequestBatch(requests=[req]))
        post = _seedvr2_post_process(outputs[0].output)
        payload = post["payload"]
        yield SimpleNamespace(
            images=[payload["video"]],
            multimodal_output={**payload, "metadata": post["metadata"]},
            stage_durations={},
            peak_memory_mb=0.0,
            custom_output={},
        )


def _build_tiny_seedvr2_components() -> tuple[SeedVR2NaDiT, SeedVR2VAE, torch.Tensor]:
    transformer = SeedVR2NaDiT(
        vid_in_channels=33,
        vid_out_channels=16,
        vid_dim=VID_DIM,
        txt_in_dim=TXT_DIM,
        heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        expand_ratio=2,
        num_layers=2,
        mm_layers=1,
        rope_dim=HEAD_DIM,
        use_varlen_kernel=False,
    )
    vae = SeedVR2VAE(channels=(16, 16, 32, 32), groups=8, latent_channels=16)
    text_emb = torch.randn(TEXT_TOKENS, TXT_DIM, dtype=torch.float32)
    return transformer, vae, text_emb


def _make_seedvr2_pipeline(
    dit_state: dict[str, torch.Tensor],
    vae_state: dict[str, torch.Tensor],
    text_emb: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
) -> SeedVR2Pipeline:
    transformer, vae, _ = _build_tiny_seedvr2_components()
    transformer.load_state_dict(dit_state, strict=True)
    vae.load_state_dict(vae_state, strict=True)

    pipe = object.__new__(SeedVR2Pipeline)
    nn.Module.__init__(pipe)
    pipe.device = torch.device(device)
    pipe.od_config = SimpleNamespace(
        dtype=dtype,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(),
    )
    pipe.frame_pixels = MAX_FRAME_PIXELS
    pipe.clip_pixels = MAX_CLIP_PIXELS
    pipe.transformer = transformer.to(device=device, dtype=dtype).eval()
    pipe.vae = vae.to(device=device, dtype=dtype).eval()
    pipe.register_buffer(
        "text", text_emb.to(device=device, dtype=dtype), persistent=False
    )
    return pipe


def _write_source_mp4_with_audio(path: Path) -> None:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={VIDEO_SIZE}x{VIDEO_SIZE}:rate={FPS}",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            "-frames:v",
            str(TOTAL_FRAMES),
            "-t",
            str((TOTAL_FRAMES + 2) / FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


@contextmanager
def _serve_seedvr2_app(pipeline: SeedVR2Pipeline) -> Iterator[tuple[FastAPI, str]]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    engine = _SeedVR2Engine(pipeline)
    app = FastAPI()
    app.state.api_server_count = 1
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = engine.stage_configs
    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="seedvr2-tiny", model_path="seedvr2")]
    )
    app.state.args = Namespace(
        default_sampling_params=None,
        max_generated_image_size=None,
        enable_server_load_tracking=False,
    )
    app.state.server_load_metrics = 0
    video_handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine,  # type: ignore[arg-type]
        model_name="seedvr2-tiny",
        stage_configs=engine.stage_configs,
    )
    app.state.openai_serving_video = video_handler
    app.include_router(router)
    register_routes(app, port)

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
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
            f"SeedVR2 video server not started after {SERVER_START_TIMEOUT_S}s"
        )
        yield app, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        sock.close()
        video_handler.shutdown()


def _run_long_video_job(
    base_url: str, source_mp4: Path
) -> tuple[int, dict[str, Any], bytes]:
    endpoint = f"{base_url}/v1/seedvr2/restore-long"
    with source_mp4.open("rb") as upload:
        submit = requests.post(
            endpoint,
            data={
                "prompt": " ",
                "size": f"{VIDEO_SIZE}x{VIDEO_SIZE}",
                "num_frames": str(TOTAL_FRAMES),
                "seed": "7723",
                "extra_params": '{"color_correction_method":"wavelet"}',
            },
            files={"input_references": ("input.mp4", upload, "video/mp4")},
            timeout=60,
        )
    if submit.status_code != 202:
        return submit.status_code, {"error": submit.text}, b""
    job_id = submit.json()["id"]
    deadline = time.monotonic() + JOB_TIMEOUT_S
    record: dict[str, Any] = {}
    while time.monotonic() < deadline:
        record = requests.get(f"{endpoint}/{job_id}", timeout=10).json()
        if record["status"] not in {"queued", "running"}:
            break
        time.sleep(0.1)
    assert record["status"] not in {"queued", "running"}, (
        f"Long-video job {job_id} still {record['status']} after {JOB_TIMEOUT_S}s"
    )
    content = requests.get(f"{endpoint}/{job_id}/content", timeout=30)
    assert content.status_code == 200, (
        f"Download failed ({content.status_code}): {record}"
    )
    return submit.status_code, record, content.content


def _decode_mp4(mp4_bytes: bytes) -> tuple[np.ndarray, Fraction, list[Fraction], bool]:
    with av.open(io.BytesIO(mp4_bytes)) as container:
        vstream = container.streams.video[0]
        rate = Fraction(vstream.average_rate)
        has_audio = bool(container.streams.audio)
        frames = list(container.decode(video=0))
        pts = [Fraction(f.pts * f.time_base) for f in frames]
        rgb = np.stack([f.to_ndarray(format="rgb24") for f in frames])
    return rgb, rate, pts, has_audio


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "SeedVR2VAE group_norm fails on TPU with non-contiguous view error, "
        "causing window restoration job to fail"
    ),
)
def test_seedvr2_long_video_restoration_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SeedVR2 Long Video Restoration API on TPU across two overlapping windows.

    - TPU path: ``POST /v1/seedvr2/restore-long`` splitting a 14-frame clip into
      two 9-frame windows restored on ``"tpu"`` via ``/v1/videos/sync`` +
      ``SeedVR2Pipeline`` (``SeedVR2NaDiT`` + ``SeedVR2VAE`` in fp16).
    - Setup check: ``pipeline.device.type == "tpu"``, fp16 weights,
      ``api_server_count == 1``, and HTTP 202 job admission.
    - Outcome check: completed 14-frame 24-FPS MP4 preserves source audio and
      matches the CPU reference ``SeedVR2Pipeline`` restoration on MAE and SSIM.
    """
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW", str(WINDOW_FRAMES))
    tpu_jobs = tmp_path / "tpu_jobs"
    cpu_jobs = tmp_path / "cpu_jobs"
    tpu_jobs.mkdir()
    cpu_jobs.mkdir()

    source_mp4 = tmp_path / "source.mp4"
    _write_source_mp4_with_audio(source_mp4)

    torch.manual_seed(0)
    dit_init, vae_init, text_emb = _build_tiny_seedvr2_components()
    dit_state = {k: v.detach().clone() for k, v in dit_init.state_dict().items()}
    vae_state = {k: v.detach().clone() for k, v in vae_init.state_dict().items()}

    tpu_pipe = _make_seedvr2_pipeline(
        dit_state, vae_state, text_emb, device="tpu", dtype=torch.float16
    )
    cpu_pipe = _make_seedvr2_pipeline(
        dit_state, vae_state, text_emb, device="cpu", dtype=torch.float16
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_type!r}"
    )
    assert tpu_pipe.device.type == "tpu", (
        f"Expected SeedVR2Pipeline on tpu, got {tpu_pipe.device}"
    )
    assert next(tpu_pipe.transformer.parameters()).dtype == torch.float16, (
        "Expected SeedVR2NaDiT weights in float16"
    )

    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR", str(tpu_jobs))
    with _serve_seedvr2_app(tpu_pipe) as (tpu_app, tpu_url):
        assert tpu_app.state.api_server_count == 1, (
            f"Expected api_server_count=1, got {tpu_app.state.api_server_count}"
        )
        tpu_status, tpu_record, tpu_mp4 = _run_long_video_job(tpu_url, source_mp4)

    assert tpu_status == 202, f"Expected HTTP 202 on submit, got {tpu_status}"
    assert tpu_record["status"] == "completed", (
        f"TPU SeedVR2 long-video job did not complete: {tpu_record}"
    )
    assert tpu_record["frames"] == TOTAL_FRAMES, (
        f"Expected {TOTAL_FRAMES} restored frames, got {tpu_record['frames']}"
    )

    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR", str(cpu_jobs))
    with _serve_seedvr2_app(cpu_pipe) as (_cpu_app, cpu_url):
        cpu_status, cpu_record, cpu_mp4 = _run_long_video_job(cpu_url, source_mp4)
    assert cpu_status == 202, f"CPU reference submit returned {cpu_status}"
    assert cpu_record["status"] == "completed", (
        f"CPU reference SeedVR2 job failed: {cpu_record}"
    )

    tpu_rgb, tpu_rate, tpu_pts, tpu_has_audio = _decode_mp4(tpu_mp4)
    cpu_rgb, _, _, _ = _decode_mp4(cpu_mp4)

    assert tpu_rgb.shape == (TOTAL_FRAMES, VIDEO_SIZE, VIDEO_SIZE, 3), (
        f"Unexpected output video shape {tpu_rgb.shape}"
    )
    assert tpu_rate == Fraction(FPS), f"Expected {FPS} FPS, got {tpu_rate}"
    assert tpu_pts == [Fraction(i, FPS) for i in range(TOTAL_FRAMES)], (
        f"Unexpected frame timestamps: {tpu_pts}"
    )
    assert tpu_has_audio is True, "Restored MP4 lost the source audio track"

    metrics = compare_video_to_golden(tpu_rgb, cpu_rgb)
    assert metrics["mae"] <= MAX_MAE, (
        f"seedvr2_long_video MAE {metrics['mae']:.4f} exceeds max {MAX_MAE}"
    )
    assert metrics["ssim"] >= MIN_SSIM, (
        f"seedvr2_long_video SSIM {metrics['ssim']:.4f} is below min {MIN_SSIM}"
    )
