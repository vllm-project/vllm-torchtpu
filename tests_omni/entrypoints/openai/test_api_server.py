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
"""TPU tests for vLLM-Omni OpenAI API server thread cap, health/models, and images."""

from __future__ import annotations

import base64
import contextlib
import errno
import io
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
from typing import Any

import filelock
import numpy as np
import psutil
import pytest
import requests
import torch
from PIL import Image
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_flux_schnell
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [pytest.mark.omni]

SERVED_MODEL_NAME = "flux-tiny"
PROMPT = "a red cube on a gray table"
IMAGE_SIZE = 256
NUM_STEPS = 2
GUIDANCE_SCALE = 0.0
SEED = 42
SERVER_STARTUP_TIMEOUT_S = 240.0
REQUEST_TIMEOUT_S = 120.0
IMAGE_MAX_REL_L2 = 2e-2
IMAGE_MAX_ABS = 4.0


@dataclass(frozen=True)
class _FluxServerBundle:
    """Running Flux ``OmniServer`` handle paired with its offline TPU reference."""

    base_url: str
    model_dir: str
    proc: subprocess.Popen[bytes]
    ref_image: torch.Tensor


def _reserve_local_port(host: str = "127.0.0.1") -> tuple[int, filelock.FileLock]:
    """Reserve a localhost TCP port with an advisory file lock.

    Adapted from ``OmniServer._reserve_port`` in vLLM-Omni's
    ``tests/helpers/runtime.py``.
    """
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


def _flux_offline_worker(model_dir: str, output_npy: str) -> None:
    """Run offline ``Omni`` in an isolated process so pytest stays TPU-free."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=model_dir, log_stats=False)
    try:
        prompt = OmniTextPrompt(prompt=PROMPT, modalities=["image"])
        params = OmniDiffusionSamplingParams(
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=SEED,
            num_outputs_per_prompt=1,
        )
        outputs = list(omni.generate(prompt, [params]))
        pil_img = outputs[0].images[0]
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        np.save(output_npy, arr)
    finally:
        omni.close()


def _run_flux_offline_reference(model_dir: str, tmp_dir: Path) -> torch.Tensor:
    """Spawn and join the offline TPU reference worker, then wait for TPU release."""
    wait_for_tpu_release()
    out_path = tmp_dir / "flux_ref.npy"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_flux_offline_worker, args=(model_dir, str(out_path)))
    proc.start()
    proc.join(timeout=SERVER_STARTUP_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        raise TimeoutError("Offline Flux reference worker timed out")
    assert proc.exitcode == 0, (
        f"Offline Flux reference worker exited with code {proc.exitcode}"
    )
    wait_for_tpu_release()
    return torch.from_numpy(np.load(out_path)).to(torch.float32)


def _decode_png_bytes(raw_bytes: bytes) -> torch.Tensor:
    """Decode raw PNG bytes into a float32 ``(H, W, 3)`` RGB tensor in ``[0, 255]``."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return torch.from_numpy(np.asarray(img, dtype=np.uint8)).to(torch.float32)


def _configure_threads_env(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    """Set or clear ``VLLM_OMNI_API_SERVER_TORCH_THREADS`` for one case."""
    monkeypatch.delenv("VLLM_OMNI_API_SERVER_TORCH_THREADS", raising=False)
    if env_value is not None:
        monkeypatch.setenv("VLLM_OMNI_API_SERVER_TORCH_THREADS", env_value)


@pytest.fixture(scope="module")
def _flux_server_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_FluxServerBundle, None, None]:
    model_dir = tiny_flux_schnell()
    tmp_dir = tmp_path_factory.mktemp("flux_api_server")
    ref_image = _run_flux_offline_reference(model_dir, tmp_dir)
    base_url, proc, lock = _start_omni_server(
        model_dir,
        ["--served-model-name", SERVED_MODEL_NAME],
    )
    try:
        yield _FluxServerBundle(
            base_url=base_url,
            model_dir=model_dir,
            proc=proc,
            ref_image=ref_image,
        )
    finally:
        _stop_server_process(proc, lock)


@pytest.mark.timeout(PR_TIMEOUT_S)
@pytest.mark.parametrize(
    ("env_value", "initial_threads", "expected_threads", "expectation"),
    [
        (None, 8, 4, contextlib.nullcontext()),
        ("2", 8, 2, contextlib.nullcontext()),
        ("8", 3, 3, contextlib.nullcontext()),
        ("0", 8, 8, contextlib.nullcontext()),
        ("-1", 8, 8, pytest.raises(ValueError, match="non-negative integer")),
    ],
    ids=[
        "default_cap_4",
        "explicit_cap_2",
        "below_cap_preserved",
        "zero_disables_cap",
        "negative_raises",
    ],
)
def test_api_server_torch_threads_cap(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    initial_threads: int,
    expected_threads: int,
    expectation: Any,
) -> None:
    """API Server Intra-Op Torch Threads Cap.

    - TPU path: Calls ``_limit_api_server_torch_threads()`` in-process while
      ``OmniTpuPlatform`` is active across default, explicit, uncapped, and
      invalid ``VLLM_OMNI_API_SERVER_TORCH_THREADS`` settings.
    - Setup check: Verifies ``current_omni_platform`` is ``OmniTpuPlatform`` on
      ``"tpu"`` and pre-seeds ``torch.set_num_threads(initial_threads)``.
    - Outcome check: Asserts ``torch.get_num_threads()`` equals the expected
      capped count (or raises ``ValueError`` for negative values).
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )
    assert current_omni_platform.device_type == "tpu", (
        f"Expected platform device_type 'tpu', got {current_omni_platform.device_type}"
    )

    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(initial_threads)
        assert torch.get_num_threads() == initial_threads, (
            f"Setup torch.set_num_threads({initial_threads}) produced "
            f"{torch.get_num_threads()}"
        )
        _configure_threads_env(monkeypatch, env_value)
        with expectation:
            api_server._limit_api_server_torch_threads()
        actual_threads = torch.get_num_threads()
        assert actual_threads == expected_threads, (
            f"env={env_value!r} from initial={initial_threads}: expected "
            f"{expected_threads} threads, got {actual_threads}"
        )
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
def test_health_and_models_routes(_flux_server_bundle: _FluxServerBundle) -> None:
    """Health and model-list routes.

    - TPU path: Queries ``GET /health`` and ``GET /v1/models`` against a live
      ``vllm serve --omni`` subprocess running ``tiny_flux_schnell()`` on TPU.
    - Setup check: Confirms the tiny Flux directory has ``model_index.json`` and
      the server subprocess is running.
    - Outcome check: Asserts HTTP 200 ``{"status": "healthy"}`` from ``/health``
      and the served model list ``["flux-tiny"]`` from ``/v1/models``.
    """
    index_path = os.path.join(_flux_server_bundle.model_dir, "model_index.json")
    assert os.path.isfile(index_path), (
        f"Expected tiny Flux model_index.json at {index_path}"
    )
    assert _flux_server_bundle.proc.poll() is None, (
        f"OmniServer exited unexpectedly with code {_flux_server_bundle.proc.poll()}"
    )

    health_resp = requests.get(
        f"{_flux_server_bundle.base_url}/health",
        timeout=REQUEST_TIMEOUT_S,
    )
    models_resp = requests.get(
        f"{_flux_server_bundle.base_url}/v1/models",
        timeout=REQUEST_TIMEOUT_S,
    )

    assert health_resp.status_code == 200, (
        f"GET /health returned {health_resp.status_code}: {health_resp.text}"
    )
    assert health_resp.json() == {"status": "healthy"}, (
        f"GET /health payload {health_resp.json()} != {{'status': 'healthy'}}"
    )
    assert models_resp.status_code == 200, (
        f"GET /v1/models returned {models_resp.status_code}: {models_resp.text}"
    )
    models_body = models_resp.json()
    assert models_body["object"] == "list", (
        f"GET /v1/models object {models_body['object']!r} != 'list'"
    )
    model_ids = [entry["id"] for entry in models_body["data"]]
    assert model_ids == [SERVED_MODEL_NAME], (
        f"GET /v1/models IDs {model_ids} != [{SERVED_MODEL_NAME!r}]"
    )


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
def test_images_generations_route(_flux_server_bundle: _FluxServerBundle) -> None:
    """Image Generation API.

    - TPU path: Sends ``POST /v1/images/generations`` in both ``b64_json`` and
      ``file`` response formats to ``OmniServer`` running ``FluxPipeline`` on TPU.
    - Setup check: Asserts HTTP 200, ``output_format == "png"``,
      ``size == "256x256"``, and non-empty ``metrics["stage_durations"]``.
    - Outcome check: Compares decoded PNG pixels against the offline TPU ``Omni``
      reference via ``assert_parity`` and checks exact ``b64_json`` vs ``file``
      equality.
    """
    assert _flux_server_bundle.proc.poll() is None, (
        f"OmniServer exited unexpectedly with code {_flux_server_bundle.proc.poll()}"
    )
    common_payload = {
        "model": SERVED_MODEL_NAME,
        "prompt": PROMPT,
        "size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "num_inference_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "output_format": "png",
        "return_stage_metrics": True,
    }

    b64_resp = requests.post(
        f"{_flux_server_bundle.base_url}/v1/images/generations",
        json={**common_payload, "response_format": "b64_json"},
        timeout=REQUEST_TIMEOUT_S,
    )
    file_resp = requests.post(
        f"{_flux_server_bundle.base_url}/v1/images/generations",
        json={**common_payload, "response_format": "file"},
        timeout=REQUEST_TIMEOUT_S,
    )

    assert b64_resp.status_code == 200, (
        f"POST /v1/images/generations (b64_json) returned {b64_resp.status_code}: "
        f"{b64_resp.text}"
    )
    body = b64_resp.json()
    assert body["output_format"] == "png", (
        f"Expected output_format 'png', got {body['output_format']!r}"
    )
    assert body["size"] == f"{IMAGE_SIZE}x{IMAGE_SIZE}", (
        f"Expected size '{IMAGE_SIZE}x{IMAGE_SIZE}', got {body['size']!r}"
    )
    assert body["metrics"]["stage_durations"] is not None, (
        f"Expected non-null stage_durations in metrics, got {body['metrics']}"
    )
    assert len(body["data"]) == 1, (
        f"Expected 1 generated image item, got {len(body['data'])}"
    )

    assert file_resp.status_code == 200, (
        f"POST /v1/images/generations (file) returned {file_resp.status_code}: "
        f"{file_resp.text}"
    )
    assert file_resp.headers["content-type"].startswith("image/png"), (
        f"Expected image/png content-type, got {file_resp.headers['content-type']!r}"
    )

    b64_img = _decode_png_bytes(base64.b64decode(body["data"][0]["b64_json"]))
    file_img = _decode_png_bytes(file_resp.content)

    assert b64_img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), (
        f"Decoded b64_json image shape {tuple(b64_img.shape)} != "
        f"({IMAGE_SIZE}, {IMAGE_SIZE}, 3)"
    )
    assert torch.equal(b64_img, file_img), (
        "Expected exact PNG pixel match between b64_json and file response formats"
    )
    assert_parity(
        b64_img,
        _flux_server_bundle.ref_image,
        name="POST /v1/images/generations vs offline Flux TPU",
        max_rel_l2=IMAGE_MAX_REL_L2,
        max_abs=IMAGE_MAX_ABS,
    )
