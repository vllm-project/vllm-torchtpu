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
"""TPU server test for vLLM-Omni video generation and CRUD HTTP endpoints."""

from __future__ import annotations

import contextlib
import errno
import io
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import av
import filelock
import numpy as np
import psutil
import pytest
import requests
import torch
from vllm_omni.entrypoints.openai.video_api_utils import (
    _encode_video_bytes,
    _PlanarFrameConverter,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import (
    NIGHTLY_TIMEOUT_S,
    assert_parity,
    compare_video_to_golden,
)
from tests_omni.tiny_models import tiny_wan2_1_t2v

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

SERVED_VIDEO_MODEL = "wan-tiny"
VIDEO_PROMPT = "smooth waves rolling onto a sandy shore"
VIDEO_WIDTH = 256
VIDEO_HEIGHT = 256
VIDEO_NUM_FRAMES = 5
VIDEO_FPS = 8
VIDEO_STEPS = 2
VIDEO_GUIDANCE_SCALE = 1.0
VIDEO_SEED = 42
SERVER_STARTUP_TIMEOUT_S = 240.0
REQUEST_TIMEOUT_S = 180.0
JOB_POLL_TIMEOUT_S = 180.0
VIDEO_MAX_REL_L2 = 5e-2
VIDEO_MAX_ABS = 12.0
VIDEO_MIN_SSIM = 0.98
# OmniOpenAIServingVideo's MP4 encoder settings; H.264 is lossy, so the offline
# reference must go through the same encode as the server response.
SERVER_VIDEO_CODEC_OPTIONS = {"preset": "ultrafast", "threads": "0"}


@dataclass(frozen=True)
class _WanVideoServerBundle:
    """Running Wan ``OmniServer`` paired with its offline MP4-encoded TPU reference."""

    base_url: str
    model_dir: str
    proc: subprocess.Popen[bytes]
    ref_frames: np.ndarray


def _decode_mp4_bytes(mp4_bytes: bytes) -> np.ndarray:
    """Decode MP4 bytes via PyAV into a uint8 ``(F, H, W, 3)`` RGB array."""
    with av.open(io.BytesIO(mp4_bytes)) as container:
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    return np.stack(frames, axis=0)


def _reserve_local_port(host: str = "127.0.0.1") -> tuple[int, filelock.FileLock]:
    """Reserve a localhost TCP port with an advisory file lock."""
    for _ in range(128):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])
        lock_path = (
            Path(tempfile.gettempdir())
            / f"vllm-omni-test-port-{os.getuid()}-{port}.lock"
        )
        lock = filelock.FileLock(str(lock_path))
        try:
            lock.acquire(timeout=0)
        except filelock.Timeout:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((host, port))
        except OSError as exc:
            lock.release()
            if exc.errno != errno.EADDRINUSE:
                raise
            continue
        return port, lock
    raise RuntimeError("Could not reserve a free localhost port after 128 attempts")


def _stop_server_process(
    proc: subprocess.Popen[bytes], lock: filelock.FileLock
) -> None:
    """Terminate the ``vllm serve`` process tree and release its port lock."""
    try:
        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.kill()
        with contextlib.suppress(ChildProcessError):
            os.waitpid(proc.pid, os.WNOHANG)
    finally:
        if lock.is_locked:
            lock.release()
        wait_for_tpu_release()


def _start_omni_server(
    model_dir: str, serve_args: list[str]
) -> tuple[str, subprocess.Popen[bytes], filelock.FileLock]:
    """Start ``vllm serve --omni`` in a child process and poll ``/health``."""
    wait_for_tpu_release()
    host = "127.0.0.1"
    port, lock = _reserve_local_port(host)
    base_url = f"http://{host}:{port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model_dir,
        "--host",
        host,
        "--port",
        str(port),
        "--omni",
        *serve_args,
    ]
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    try:
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            ret = proc.poll()
            if ret is not None:
                raise RuntimeError(f"OmniServer exited early with code {ret} for {cmd}")
            try:
                resp = requests.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return base_url, proc, lock
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(
            f"OmniServer did not pass /health within {SERVER_STARTUP_TIMEOUT_S}s"
        )
    except BaseException:
        _stop_server_process(proc, lock)
        raise


def _wan_offline_worker(model_dir: str, output_mp4: str) -> None:
    """Run offline ``Omni`` Wan generation and mux through the server MP4 encoder."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=model_dir, log_stats=False)
    try:
        prompt = OmniTextPrompt(prompt=VIDEO_PROMPT, modalities=["video"])
        params = OmniDiffusionSamplingParams(
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            num_frames=VIDEO_NUM_FRAMES,
            fps=VIDEO_FPS,
            frame_rate=float(VIDEO_FPS),
            num_inference_steps=VIDEO_STEPS,
            guidance_scale=VIDEO_GUIDANCE_SCALE,
            seed=VIDEO_SEED,
            num_outputs_per_prompt=1,
        )
        outputs = list(omni.generate(prompt, [params]))
        raw_video = outputs[0].images[0]
        if isinstance(raw_video, np.ndarray) and raw_video.ndim == 5:
            raw_video = raw_video[0]
        mp4_bytes = _encode_video_bytes(
            raw_video,
            fps=VIDEO_FPS,
            video_codec_options=SERVER_VIDEO_CODEC_OPTIONS,
            frame_converter=_PlanarFrameConverter(max_workers=1),
        )
        Path(output_mp4).write_bytes(mp4_bytes)
    finally:
        omni.close()


def _run_wan_offline_reference(model_dir: str, tmp_dir: Path) -> np.ndarray:
    """Spawn and join the offline Wan TPU reference worker, returning RGB frames."""
    wait_for_tpu_release()
    out_mp4 = tmp_dir / "wan_ref.mp4"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_wan_offline_worker, args=(model_dir, str(out_mp4)))
    proc.start()
    proc.join(timeout=SERVER_STARTUP_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        raise TimeoutError("Offline Wan reference worker timed out")
    assert proc.exitcode == 0, (
        f"Offline Wan reference worker exited with code {proc.exitcode}"
    )
    wait_for_tpu_release()
    return _decode_mp4_bytes(out_mp4.read_bytes())


def _poll_video_until_completed(base_url: str, video_id: str) -> dict[str, object]:
    """Poll ``GET /v1/videos/{video_id}`` until ``status`` leaves queued/in_progress."""
    deadline = time.monotonic() + JOB_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = requests.get(
            f"{base_url}/v1/videos/{video_id}",
            timeout=REQUEST_TIMEOUT_S,
        )
        assert resp.status_code == 200, (
            f"GET /v1/videos/{video_id} returned {resp.status_code}: {resp.text}"
        )
        payload = resp.json()
        if payload["status"] not in ("queued", "in_progress"):
            return payload
        time.sleep(0.5)
    raise TimeoutError(
        f"Video job {video_id} did not finish within {JOB_POLL_TIMEOUT_S}s"
    )


@pytest.fixture(scope="module")
def _wan_video_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_WanVideoServerBundle, None, None]:
    model_dir = tiny_wan2_1_t2v()
    tmp_dir = tmp_path_factory.mktemp("wan_video_server")
    ref_frames = _run_wan_offline_reference(model_dir, tmp_dir)
    base_url, proc, lock = _start_omni_server(
        model_dir,
        ["--served-model-name", SERVED_VIDEO_MODEL],
    )
    try:
        yield _WanVideoServerBundle(
            base_url=base_url,
            model_dir=model_dir,
            proc=proc,
            ref_frames=ref_frames,
        )
    finally:
        _stop_server_process(proc, lock)


def test_videos_generation_and_crud_routes(
    _wan_video_bundle: _WanVideoServerBundle,
) -> None:
    """Videos Generation and CRUD API.

    - TPU path: Exercises ``POST /v1/videos``, ``GET /v1/videos``,
      ``GET /v1/videos/{id}``, ``GET /v1/videos/{id}/content``,
      ``DELETE /v1/videos/{id}``, and ``POST /v1/videos/sync`` on ``OmniServer``
      serving ``tiny_wan2_1_t2v()`` on TPU.
    - Setup check: Confirms async job creation, store listing, completion status,
      ``X-Stage-Durations`` sync header, deletion response, and 404 after delete.
    - Outcome check: Decodes async and sync MP4 streams to ``(5, 256, 256, 3)``
      frames and checks ``assert_parity`` and SSIM against the offline TPU reference.
    """
    assert _wan_video_bundle.proc.poll() is None, (
        f"OmniServer exited unexpectedly with code {_wan_video_bundle.proc.poll()}"
    )
    form_fields = {
        "model": SERVED_VIDEO_MODEL,
        "prompt": VIDEO_PROMPT,
        "width": str(VIDEO_WIDTH),
        "height": str(VIDEO_HEIGHT),
        "num_frames": str(VIDEO_NUM_FRAMES),
        "fps": str(VIDEO_FPS),
        "num_inference_steps": str(VIDEO_STEPS),
        "guidance_scale": str(VIDEO_GUIDANCE_SCALE),
        "seed": str(VIDEO_SEED),
    }

    create_resp = requests.post(
        f"{_wan_video_bundle.base_url}/v1/videos",
        data=form_fields,
        timeout=REQUEST_TIMEOUT_S,
    )
    assert create_resp.status_code == 200, (
        f"POST /v1/videos returned {create_resp.status_code}: {create_resp.text}"
    )
    created = create_resp.json()
    video_id = created["id"]
    assert created["status"] in ("queued", "in_progress", "completed"), (
        f"Unexpected initial job status {created['status']!r}"
    )

    list_resp = requests.get(
        f"{_wan_video_bundle.base_url}/v1/videos",
        timeout=REQUEST_TIMEOUT_S,
    )
    assert list_resp.status_code == 200, (
        f"GET /v1/videos returned {list_resp.status_code}: {list_resp.text}"
    )
    listed_ids = [item["id"] for item in list_resp.json()["data"]]
    assert video_id in listed_ids, (
        f"Created video job {video_id!r} missing from GET /v1/videos: {listed_ids}"
    )

    completed = _poll_video_until_completed(_wan_video_bundle.base_url, video_id)
    assert completed["status"] == "completed", (
        f"Video job {video_id} ended in status {completed['status']!r}: {completed}"
    )

    content_resp = requests.get(
        f"{_wan_video_bundle.base_url}/v1/videos/{video_id}/content",
        timeout=REQUEST_TIMEOUT_S,
    )
    assert content_resp.status_code == 200, (
        f"GET /v1/videos/{video_id}/content returned {content_resp.status_code}"
    )
    assert content_resp.headers["content-type"].startswith("video/mp4"), (
        f"Expected video/mp4 content-type, got {content_resp.headers['content-type']!r}"
    )

    sync_resp = requests.post(
        f"{_wan_video_bundle.base_url}/v1/videos/sync",
        data=form_fields,
        timeout=REQUEST_TIMEOUT_S,
    )
    assert sync_resp.status_code == 200, (
        f"POST /v1/videos/sync returned {sync_resp.status_code}: {sync_resp.text}"
    )
    assert sync_resp.headers["content-type"].startswith("video/mp4"), (
        f"Sync video content-type {sync_resp.headers['content-type']!r} != video/mp4"
    )
    stage_durations = json.loads(sync_resp.headers["X-Stage-Durations"])
    assert len(stage_durations) >= 1, (
        f"Expected non-empty X-Stage-Durations header, got {stage_durations}"
    )

    delete_resp = requests.delete(
        f"{_wan_video_bundle.base_url}/v1/videos/{video_id}",
        timeout=REQUEST_TIMEOUT_S,
    )
    assert delete_resp.status_code == 200, (
        f"DELETE /v1/videos/{video_id} returned {delete_resp.status_code}: "
        f"{delete_resp.text}"
    )
    assert delete_resp.json()["deleted"] is True, (
        f"Expected deleted=True on DELETE, got {delete_resp.json()}"
    )

    missing_resp = requests.get(
        f"{_wan_video_bundle.base_url}/v1/videos/{video_id}",
        timeout=REQUEST_TIMEOUT_S,
    )
    assert missing_resp.status_code == 404, (
        f"Expected 404 after DELETE for {video_id}, got {missing_resp.status_code}"
    )

    expected_shape = (VIDEO_NUM_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 3)
    async_frames = _decode_mp4_bytes(content_resp.content)
    sync_frames = _decode_mp4_bytes(sync_resp.content)
    assert async_frames.shape == expected_shape, (
        f"Async MP4 shape {async_frames.shape} != {expected_shape}"
    )
    assert sync_frames.shape == expected_shape, (
        f"Sync MP4 shape {sync_frames.shape} != {expected_shape}"
    )

    ref_tensor = torch.from_numpy(_wan_video_bundle.ref_frames).to(torch.float32)
    assert_parity(
        torch.from_numpy(async_frames).to(torch.float32),
        ref_tensor,
        name="GET /v1/videos/{id}/content vs offline Wan TPU",
        max_rel_l2=VIDEO_MAX_REL_L2,
        max_abs=VIDEO_MAX_ABS,
    )
    assert_parity(
        torch.from_numpy(sync_frames).to(torch.float32),
        ref_tensor,
        name="POST /v1/videos/sync vs offline Wan TPU",
        max_rel_l2=VIDEO_MAX_REL_L2,
        max_abs=VIDEO_MAX_ABS,
    )
    async_metrics = compare_video_to_golden(async_frames, _wan_video_bundle.ref_frames)
    sync_metrics = compare_video_to_golden(sync_frames, _wan_video_bundle.ref_frames)
    assert async_metrics["ssim"] >= VIDEO_MIN_SSIM, (
        f"Async video SSIM {async_metrics['ssim']:.4f} below min {VIDEO_MIN_SSIM}"
    )
    assert sync_metrics["ssim"] >= VIDEO_MIN_SSIM, (
        f"Sync video SSIM {sync_metrics['ssim']:.4f} below min {VIDEO_MIN_SSIM}"
    )
