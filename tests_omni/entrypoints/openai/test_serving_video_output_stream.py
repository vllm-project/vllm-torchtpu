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
"""TPU server test for vLLM-Omni WebSocket chunked streaming video output."""

from __future__ import annotations

import contextlib
import errno
import io
import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import filelock
import numpy as np
import psutil
import pytest
import requests
import torch
from diffusers import HeliosTransformer3DModel
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from websockets.sync import client as ws_client

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_wan2_1_t2v

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

SERVED_STREAM_MODEL = "helios-tiny"
STREAM_PROMPT = "a glowing lantern drifting across calm water"
STREAM_WIDTH = 256
STREAM_HEIGHT = 256
STREAM_NUM_FRAMES = 66  # 2 chunks of 33 frames
# Helios's default; the non-streaming path ignores any other value.
STREAM_LATENT_FRAMES_PER_CHUNK = 9
# Non-default chunk size for the auxiliary test: 2 chunks of 9 frames.
CUSTOM_CHUNK_LATENT_FRAMES = 3
CUSTOM_CHUNK_NUM_FRAMES = 18
STREAM_FPS = 8
STREAM_STEPS = 2
STREAM_GUIDANCE_SCALE = 1.0
STREAM_SEED = 42
# The stream handler's default x264 options, but with CRF 0 (lossless for 8-bit
# video) instead of the encoder's CRF 18. Both videos are encoded with them.
LOSSLESS_VIDEO_CODEC_OPTIONS = {
    "preset": "ultrafast",
    "threads": "0",
    "tune": "zerolatency",
    "crf": "0",
}
TEXT_HIDDEN_SIZE = 256
SERVER_STARTUP_TIMEOUT_S = 240.0
WS_RECV_TIMEOUT_S = 180.0


@dataclass(frozen=True)
class _HeliosStreamBundle:
    """Running Helios streaming ``OmniServer`` and non-streaming TPU reference."""

    ws_url: str
    model_dir: str
    proc: subprocess.Popen[bytes]
    ref_frames: np.ndarray


def _decode_video_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode MP4 or concatenated fMP4 bytes into uint8 ``(F, H, W, 3)`` RGB frames."""
    with av.open(io.BytesIO(raw_bytes)) as container:
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
                    return f"ws://{host}:{port}", proc, lock
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(
            f"OmniServer did not pass /health within {SERVER_STARTUP_TIMEOUT_S}s"
        )
    except BaseException:
        _stop_server_process(proc, lock)
        raise


def _helios_checkpoint_worker(wan_dir: str, target_dir: str) -> None:
    """Build a 2-layer bf16 ``HeliosPipeline`` directory in an isolated CPU process."""
    # CPU generator only: torch.manual_seed() also seeds, and so opens, the TPU.
    torch.default_generator.manual_seed(0)
    transformer = HeliosTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=TEXT_HIDDEN_SIZE,
        freq_dim=256,
        ffn_dim=512,
        num_layers=2,
        rope_dim=(44, 42, 42),
        # HeliosPipeline always feeds short/mid/long history latents.
        has_multi_term_memory_patch=True,
    )

    out_root = Path(target_dir)
    for sub in ("tokenizer", "text_encoder", "vae"):
        shutil.copytree(Path(wan_dir) / sub, out_root / sub)
    transformer.to(torch.bfloat16).save_pretrained(out_root / "transformer")

    sched_dir = out_root / "scheduler"
    sched_dir.mkdir(parents=True, exist_ok=True)
    (sched_dir / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "HeliosScheduler",
                "num_train_timesteps": 1000,
                "shift": 3.0,
                "stages": 2,
                "stage_range": [0.0, 0.5, 1.0],
                "gamma": 0.333333,
                "prediction_type": "flow_prediction",
                "solver_order": 1,
            },
            indent=2,
        )
    )
    (out_root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "HeliosPipeline",
                "_diffusers_version": "0.33.0",
            },
            indent=2,
        )
    )


def _build_tiny_helios() -> str:
    """Return a cached tiny ``HeliosPipeline`` checkpoint built from local Wan files."""
    root = os.path.join(tempfile.gettempdir(), "vllm-torchtpu-tiny-models")
    model_dir = os.path.join(root, "tiny-helios-streaming")
    if os.path.isdir(model_dir):
        return model_dir

    wan_dir = tiny_wan2_1_t2v()
    os.makedirs(root, exist_ok=True)
    staging_dir = tempfile.mkdtemp(dir=root)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_helios_checkpoint_worker, args=(wan_dir, staging_dir))
    proc.start()
    proc.join(timeout=120.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        raise TimeoutError("Tiny Helios checkpoint builder timed out")
    assert proc.exitcode == 0, (
        f"Tiny Helios checkpoint builder failed with exit code {proc.exitcode}"
    )
    os.rename(staging_dir, model_dir)
    return model_dir


def _helios_offline_worker(
    model_dir: str,
    output_mp4: str,
    num_frames: int,
    latent_frames_per_chunk: int,
) -> None:
    """Run non-streaming ``Omni`` on ``HeliosPipeline`` and save a lossless MP4."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=model_dir, log_stats=False)
    try:
        prompt = OmniTextPrompt(prompt=STREAM_PROMPT, modalities=["video"])
        params = OmniDiffusionSamplingParams(
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            num_frames=num_frames,
            fps=STREAM_FPS,
            frame_rate=float(STREAM_FPS),
            num_inference_steps=STREAM_STEPS,
            guidance_scale=STREAM_GUIDANCE_SCALE,
            seed=STREAM_SEED,
            extra_args={
                "num_latent_frames_per_chunk": latent_frames_per_chunk,
                "is_enable_stage2": False,
            },
        )
        outputs = list(omni.generate(prompt, [params]))
        raw_video = outputs[0].images[0]
        if isinstance(raw_video, np.ndarray) and raw_video.ndim == 5:
            raw_video = raw_video[0]
        mp4_bytes = _encode_video_bytes(
            raw_video,
            fps=STREAM_FPS,
            video_codec_options=LOSSLESS_VIDEO_CODEC_OPTIONS,
        )
        Path(output_mp4).write_bytes(mp4_bytes)
    finally:
        omni.close()


def _run_helios_offline_reference(
    model_dir: str,
    tmp_dir: Path,
    num_frames: int,
    latent_frames_per_chunk: int,
) -> np.ndarray:
    """Spawn and join the non-streaming Helios TPU reference worker."""
    wait_for_tpu_release()
    out_mp4 = tmp_dir / "helios_ref.mp4"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_helios_offline_worker,
        args=(model_dir, str(out_mp4), num_frames, latent_frames_per_chunk),
    )
    proc.start()
    proc.join(timeout=SERVER_STARTUP_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        raise TimeoutError("Offline Helios reference worker timed out")
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Offline Helios reference worker exited with code {proc.exitcode}"
        )
    wait_for_tpu_release()
    return _decode_video_bytes(out_mp4.read_bytes())


@pytest.fixture(scope="module")
def _shared_compile_cache(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """Share one on-disk TorchTPU compile cache across this module's processes.

    - The reference engines and the server are separate processes; the cache
      lets later ones reuse programs that earlier ones compiled.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
            str(tmp_path_factory.mktemp("tpu_compile_cache")),
        )
        yield


# Function scope: the server must stop before the next test opens the TPU.
@pytest.fixture
def _helios_stream_bundle(
    _shared_compile_cache: None,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_HeliosStreamBundle, None, None]:
    model_dir = _build_tiny_helios()
    tmp_dir = tmp_path_factory.mktemp("helios_stream_server")
    ref_frames = _run_helios_offline_reference(
        model_dir,
        tmp_dir,
        num_frames=STREAM_NUM_FRAMES,
        latent_frames_per_chunk=STREAM_LATENT_FRAMES_PER_CHUNK,
    )
    ws_url, proc, lock = _start_omni_server(
        model_dir,
        [
            "--served-model-name",
            SERVED_STREAM_MODEL,
            "--diffusion-streaming-output",
        ],
    )
    try:
        yield _HeliosStreamBundle(
            ws_url=ws_url,
            model_dir=model_dir,
            proc=proc,
            ref_frames=ref_frames,
        )
    finally:
        _stop_server_process(proc, lock)


def test_streaming_video_output_websocket(
    _helios_stream_bundle: _HeliosStreamBundle,
) -> None:
    """Streaming video output (chunked diffusion output).

    - TPU path: Connects to ``WS /v1/realtime/video`` on ``OmniServer`` started
      with ``--diffusion-streaming-output`` and streams a 2-chunk ``HeliosPipeline``
      video on TPU.
    - Setup check: Confirms ``video.start`` handshake, ``session.pong`` control
      reply, 2 ordered ``video.chunk_metadata`` + fMP4 binary payloads, and
      ``session.done``.
    - Outcome check: Concatenates streamed fMP4 fragments, decodes RGB frames via
      PyAV, and requires them to equal the non-streaming ``HeliosPipeline`` TPU
      output. Both videos use the same lossless x264 options.
    """
    assert _helios_stream_bundle.proc.poll() is None, (
        f"OmniServer exited unexpectedly with code {_helios_stream_bundle.proc.poll()}"
    )

    media_metadatas: list[dict[str, Any]] = []
    binary_fragments: list[bytes] = []
    received_pong = False
    done_msg: dict[str, Any] | None = None

    with ws_client.connect(
        f"{_helios_stream_bundle.ws_url}/v1/realtime/video",
        max_size=64 * 1024 * 1024,
    ) as websocket:
        websocket.send(
            json.dumps(
                {
                    "type": "session.start",
                    "model": SERVED_STREAM_MODEL,
                    "prompt": STREAM_PROMPT,
                    "width": STREAM_WIDTH,
                    "height": STREAM_HEIGHT,
                    "num_frames": STREAM_NUM_FRAMES,
                    "fps": STREAM_FPS,
                    "num_inference_steps": STREAM_STEPS,
                    "guidance_scale": STREAM_GUIDANCE_SCALE,
                    "seed": STREAM_SEED,
                    "extra_params": {
                        "num_latent_frames_per_chunk": STREAM_LATENT_FRAMES_PER_CHUNK,
                        "is_enable_stage2": False,
                        "video_codec_options": LOSSLESS_VIDEO_CODEC_OPTIONS,
                    },
                }
            )
        )
        start_raw = websocket.recv(timeout=WS_RECV_TIMEOUT_S)
        assert isinstance(start_raw, str), (
            f"Expected JSON text for video.start, got {type(start_raw).__name__}"
        )
        start_msg = json.loads(start_raw)
        assert start_msg["type"] == "video.start", (
            f"Expected video.start handshake, got {start_msg}"
        )

        websocket.send(json.dumps({"type": "session.ping"}))

        while done_msg is None:
            raw_msg = websocket.recv(timeout=WS_RECV_TIMEOUT_S)
            assert isinstance(raw_msg, str), (
                f"Expected JSON control/metadata frame, got {type(raw_msg).__name__}"
            )
            msg = json.loads(raw_msg)
            msg_type = msg["type"]
            if msg_type == "session.pong":
                received_pong = True
            elif msg_type == "video.chunk_metadata":
                chunk_bytes = websocket.recv(timeout=WS_RECV_TIMEOUT_S)
                assert isinstance(chunk_bytes, bytes), (
                    "Expected binary fMP4 payload immediately after chunk metadata"
                )
                assert len(chunk_bytes) == msg["byte_length"], (
                    f"Chunk byte length {len(chunk_bytes)} != "
                    f"metadata {msg['byte_length']}"
                )
                binary_fragments.append(chunk_bytes)
                if msg["kind"] == "media":
                    media_metadatas.append(msg)
            elif msg_type == "session.done":
                done_msg = msg
            else:
                raise AssertionError(f"Unexpected WebSocket message: {msg}")

    assert received_pong is True, "Expected session.pong reply to session.ping"
    assert done_msg is not None, "Expected session.done message"
    assert done_msg["stopped"] is False, (
        f"Expected unstopped session.done, got {done_msg}"
    )
    assert len(media_metadatas) == 2, (
        f"Expected 2 media chunks, got {len(media_metadatas)}: {media_metadatas}"
    )
    chunk_indices = [m["generation_chunk_index"] for m in media_metadatas]
    assert chunk_indices == [0, 1], (
        f"Expected generation_chunk_index [0, 1], got {chunk_indices}"
    )
    assert done_msg["chunks"] == len(binary_fragments), (
        f"session.done chunks {done_msg['chunks']} != "
        f"received fragments {len(binary_fragments)}"
    )

    streamed_frames = _decode_video_bytes(b"".join(binary_fragments))
    expected_shape = (STREAM_NUM_FRAMES, STREAM_HEIGHT, STREAM_WIDTH, 3)
    assert streamed_frames.shape == expected_shape, (
        f"Streamed fMP4 video shape {streamed_frames.shape} != {expected_shape}"
    )

    np.testing.assert_array_equal(
        streamed_frames,
        _helios_stream_bundle.ref_frames,
        err_msg="Lossless streamed frames differ from non-streaming Helios frames",
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Non-streaming HeliosPipeline.forward() does not read "
        "num_latent_frames_per_chunk from extra_args (only the streaming step "
        "path does), so it always uses 9 latent frames per chunk"
    ),
)
def test_non_streaming_num_latent_frames_per_chunk(
    _shared_compile_cache: None,
    tmp_path: Path,
) -> None:
    """Non-streaming Helios honours a non-default ``num_latent_frames_per_chunk``.

    - TPU path: Runs non-streaming ``Omni`` on ``HeliosPipeline`` with 3 latent
      frames per chunk for an 18-frame video.
    - Outcome check: Expects 2 chunks of 9 frames (18 frames), which the
      streaming path produces for the same request.
    """
    model_dir = _build_tiny_helios()
    frames = _run_helios_offline_reference(
        model_dir,
        tmp_path,
        num_frames=CUSTOM_CHUNK_NUM_FRAMES,
        latent_frames_per_chunk=CUSTOM_CHUNK_LATENT_FRAMES,
    )
    expected_shape = (CUSTOM_CHUNK_NUM_FRAMES, STREAM_HEIGHT, STREAM_WIDTH, 3)
    assert frames.shape == expected_shape, (
        f"Non-streaming Helios video shape {frames.shape} != {expected_shape}"
    )
