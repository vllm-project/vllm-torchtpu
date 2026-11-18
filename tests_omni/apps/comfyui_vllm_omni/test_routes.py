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
"""TPU server test for ComfyUI-vLLM-Omni model-list proxy route."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import time
from collections.abc import Iterator

import psutil
import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

SERVER_READY_TIMEOUT_S = 240.0
POLL_INTERVAL_S = 0.5
REQUEST_TIMEOUT_S = 15.0
SHUTDOWN_TIMEOUT_S = 15.0


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    """Terminate a spawned server process and its child workers."""
    children: list[psutil.Process] = []
    with contextlib.suppress(psutil.NoSuchProcess):
        parent = psutil.Process(proc.pid)
        children = [
            child
            for child in parent.children(recursive=True)
            if "resource_tracker" not in " ".join(child.cmdline())
        ]
    if proc.poll() is None:
        proc.terminate()
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.TimeoutExpired):
            child.wait(timeout=5.0)
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()


@contextlib.contextmanager
def _running_omni_server(model_dir: str) -> Iterator[tuple[subprocess.Popen[str], str]]:
    """Launch ``vllm serve --omni`` in a subprocess and yield ``(proc, base_url)``."""
    wait_for_tpu_release()
    port = get_open_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model_dir,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while True:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout is not None else ""
                raise RuntimeError(
                    f"Omni server exited early ({proc.poll()}):\n{output[-4000:]}"
                )
            try:
                resp = requests.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Omni server did not pass /health within {SERVER_READY_TIMEOUT_S}s"
                )
            time.sleep(POLL_INTERVAL_S)
        yield proc, base_url
    finally:
        _terminate_process_tree(proc)
        wait_for_tpu_release()


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason=(
        "comfyui_vllm_omni lives under apps/ComfyUI-vLLM-Omni and is not "
        "installed by the vllm-omni wheel in the TPU image"
    ),
)
def test_comfyui_models_proxy() -> None:
    """ComfyUI integration (custom nodes).

    - TPU path: imports ``comfyui_vllm_omni.routes``, launches ``vllm serve --omni``
      on ``tiny_flux_schnell()`` in a TPU subprocess, and queries
      ``models_proxy_response``.
    - Setup check: ``MODELS_ROUTE`` equals ``/vllm_omni/models``, the server
      subprocess stays alive, and ``GET /health`` returns HTTP 200.
    - Outcome check: ``models_proxy_response`` returns HTTP 200 and the exact
      model list served by ``GET /v1/models``.
    """
    from comfyui_vllm_omni.routes import MODELS_ROUTE, models_proxy_response

    assert MODELS_ROUTE == "/vllm_omni/models", (
        f"comfyui MODELS_ROUTE {MODELS_ROUTE!r} != '/vllm_omni/models'"
    )
    model_dir = tiny_flux_schnell()
    with _running_omni_server(model_dir) as (proc, base_url):
        assert proc.poll() is None, (
            f"comfyui Omni server exited prematurely with code {proc.poll()}"
        )
        health_resp = requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_S)
        assert health_resp.status_code == 200, (
            f"comfyui GET /health status {health_resp.status_code} != 200"
        )

        direct_resp = requests.get(f"{base_url}/v1/models", timeout=REQUEST_TIMEOUT_S)
        assert direct_resp.status_code == 200, (
            f"comfyui GET /v1/models status {direct_resp.status_code} != 200"
        )
        expected_models = [entry["id"] for entry in direct_resp.json()["data"]]
        assert expected_models == [model_dir], (
            f"comfyui direct /v1/models {expected_models} != {[model_dir]}"
        )

        status, payload = asyncio.run(models_proxy_response(f"{base_url}/v1"))
        assert status == 200, (
            f"comfyui models_proxy_response status {status} != 200: {payload}"
        )
        assert payload == {"models": expected_models}, (
            f"comfyui models_proxy_response payload {payload} != "
            f"{{'models': {expected_models}}}"
        )
