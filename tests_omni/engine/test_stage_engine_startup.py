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
"""TPU tests for headless stage execution and per-stage device placement."""

from __future__ import annotations

import base64
import contextlib
import io
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import psutil
import pytest
import requests
from PIL import Image
from torch_tpu._internal.utils import hardware
from vllm.utils.network_utils import get_open_port
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests.tpu_release import tpu_device_owners, wait_for_tpu_release
from tests_omni.conftest import OmniRunner
from tests_omni.models.wan.utils import extract_wan_video
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell, tiny_wan2_1_t2v

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

IMAGE_SIZE = 256
IMAGE_STEPS = 2
WAN_FRAMES = 5
WAN_STEPS = 1
SEED_PRIMARY = 42
SEED_CONTROL = 99
MIN_TPU_DEVICES = 4
SERVER_READY_TIMEOUT_S = 240.0
POLL_INTERVAL_S = 0.5
REQUEST_TIMEOUT_S = 15.0
GENERATION_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 15.0
PROMPT = "a geometric cube on a plain table"


def _terminate_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a spawned CLI process and its child workers."""
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
def _running_headless_topology(
    model_dir: str,
) -> Iterator[tuple[subprocess.Popen[bytes], subprocess.Popen[bytes], str]]:
    """Start a ``--stage-id none`` head and a ``--headless --stage-id 0`` worker."""
    wait_for_tpu_release()
    master_port = get_open_port()
    http_port = get_open_port()
    base_url = f"http://127.0.0.1:{http_port}"
    head_cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model_dir,
        "--omni",
        "--stage-id",
        "none",
        "--omni-master-address",
        "127.0.0.1",
        "--omni-master-port",
        str(master_port),
        "--host",
        "127.0.0.1",
        "--port",
        str(http_port),
    ]
    headless_cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model_dir,
        "--omni",
        "--headless",
        "--stage-id",
        "0",
        "--omni-master-address",
        "127.0.0.1",
        "--omni-master-port",
        str(master_port),
    ]
    # Inherit stdout/stderr: nothing reads a PIPE until exit, and once it fills,
    # the process blocks on its next log write.
    head_proc = subprocess.Popen(head_cmd)
    headless_proc = subprocess.Popen(headless_cmd)
    try:
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while True:
            for label, proc in (("head", head_proc), ("headless", headless_proc)):
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{label} process exited early with code {proc.poll()}"
                    )
            try:
                resp = requests.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Headless topology did not pass /health within "
                    f"{SERVER_READY_TIMEOUT_S}s"
                )
            time.sleep(POLL_INTERVAL_S)
        yield head_proc, headless_proc, base_url
    finally:
        _terminate_process_tree(head_proc)
        _terminate_process_tree(headless_proc)
        wait_for_tpu_release()


def _image_generation_payload(seed: int) -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "num_inference_steps": IMAGE_STEPS,
        "guidance_scale": 0.0,
        "seed": seed,
        "response_format": "b64_json",
    }


def _decode_b64_image_response(resp: requests.Response, label: str) -> np.ndarray:
    assert resp.status_code == 200, (
        f"{label}: HTTP status {resp.status_code} != 200: {resp.text[:500]}"
    )
    body = resp.json()
    assert len(body["data"]) == 1, f"{label}: data length {len(body['data'])} != 1"
    raw = base64.b64decode(body["data"][0]["b64_json"])
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert arr.shape == expected_shape, (
        f"{label}: decoded shape {arr.shape} != {expected_shape}"
    )
    return arr


def _wan_sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_frames=WAN_FRAMES,
        num_inference_steps=WAN_STEPS,
        guidance_scale=1.0,
        seed=SEED_PRIMARY,
        generator_device="cpu",
    )


def test_headless_stage_execution() -> None:
    """Headless Stage Execution.

    - TPU path: launches a CPU-only ``--stage-id none`` frontend head and a TPU
      ``--headless --stage-id 0`` diffusion worker for ``tiny_flux_schnell()``,
      routing ``POST /v1/images/generations`` over ZMQ.
    - Setup check: both head and headless subprocesses stay alive and head
      ``GET /health`` returns HTTP 200 after remote stage-0 registration.
    - Outcome check: repeated ``seed=42`` image requests return identical
      ``(256, 256, 3)`` RGB outputs while differing from ``seed=99``.
    """
    model_dir = tiny_flux_schnell()
    with _running_headless_topology(model_dir) as (head_proc, headless_proc, base_url):
        assert head_proc.poll() is None, (
            f"Head process exited prematurely with code {head_proc.poll()}"
        )
        assert headless_proc.poll() is None, (
            f"Headless stage exited prematurely with code {headless_proc.poll()}"
        )
        health_resp = requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_S)
        assert health_resp.status_code == 200, (
            f"Headless head GET /health status {health_resp.status_code} != 200"
        )

        resp_a = requests.post(
            f"{base_url}/v1/images/generations",
            json=_image_generation_payload(SEED_PRIMARY),
            timeout=GENERATION_TIMEOUT_S,
        )
        resp_b = requests.post(
            f"{base_url}/v1/images/generations",
            json=_image_generation_payload(SEED_PRIMARY),
            timeout=GENERATION_TIMEOUT_S,
        )
        resp_ctrl = requests.post(
            f"{base_url}/v1/images/generations",
            json=_image_generation_payload(SEED_CONTROL),
            timeout=GENERATION_TIMEOUT_S,
        )

    img_a = _decode_b64_image_response(resp_a, "headless_seed42_a")
    img_b = _decode_b64_image_response(resp_b, "headless_seed42_b")
    img_ctrl = _decode_b64_image_response(resp_ctrl, "headless_seed99")

    assert np.array_equal(img_b, img_a), (
        "Headless stage repeated seed=42 outputs diverged: "
        f"max_abs={int(np.abs(img_b.astype(int) - img_a.astype(int)).max())}"
    )
    assert not np.array_equal(img_ctrl, img_a), (
        "Headless stage seed=99 output unexpectedly matched seed=42 bitwise"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Multi-rank TPU diffusion workers bind chips 0..N-1 by local rank "
        "instead of honoring stage_overrides devices (vllm-torchtpu PR #2620)"
    ),
)
def test_per_stage_device_placement() -> None:
    """Per-Stage Device Placement.

    - TPU path: runs two sequential 2-rank (``ulysses_degree=2``,
      ``distributed_executor_backend="mp"``) ``OmniRunner`` engines on
      ``tiny_wan2_1_t2v()`` with ``stage_overrides`` pinning stage 0 first to
      ``devices="0,1"`` and then to ``devices="2,3"``.
    - Setup check: host has at least 4 TPU chips, ``runtime_config.devices``
      equals ``"0,1"`` / ``"2,3"``, and ``parallel_config.ulysses_degree == 2``.
    - Outcome check: live VFIO device nodes held by ``devices="2,3"`` are
      disjoint from ``devices="0,1"``, and the generated ``(5, 256, 256, 3)``
      videos match bitwise.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= MIN_TPU_DEVICES, (
        f"Per-Stage Device Placement requires >= {MIN_TPU_DEVICES} TPU chips, "
        f"found {device_count}"
    )

    model_dir = tiny_wan2_1_t2v()
    video_shape = (WAN_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)

    runner_01 = OmniRunner(
        model_dir,
        ulysses_degree=2,
        distributed_executor_backend="mp",
        stage_overrides={"0": {"devices": "0,1"}},
    )
    try:
        stage_cfg_01 = runner_01.stage_configs[0]
        assert stage_cfg_01.runtime_config.devices == "0,1", (
            f"devices='0,1' resolved to {stage_cfg_01.runtime_config.devices!r}"
        )
        assert stage_cfg_01.parallel_config.ulysses_degree == 2, (
            "devices='0,1' ulysses_degree "
            f"{stage_cfg_01.parallel_config.ulysses_degree} != 2"
        )
        owners_01 = set(tpu_device_owners())
        assert len(owners_01) == 2, (
            f"devices='0,1' held {len(owners_01)} device nodes ({sorted(owners_01)}), "
            "expected 2"
        )
        outputs_01 = list(
            runner_01.generate(PROMPT, _wan_sampling_params(), use_tqdm=False)
        )
    finally:
        runner_01.close()
    video_01 = extract_wan_video(outputs_01, video_shape)

    runner_23 = OmniRunner(
        model_dir,
        ulysses_degree=2,
        distributed_executor_backend="mp",
        stage_overrides={"0": {"devices": "2,3"}},
    )
    try:
        stage_cfg_23 = runner_23.stage_configs[0]
        assert stage_cfg_23.runtime_config.devices == "2,3", (
            f"devices='2,3' resolved to {stage_cfg_23.runtime_config.devices!r}"
        )
        assert stage_cfg_23.parallel_config.ulysses_degree == 2, (
            "devices='2,3' ulysses_degree "
            f"{stage_cfg_23.parallel_config.ulysses_degree} != 2"
        )
        owners_23 = set(tpu_device_owners())
        assert len(owners_23) == 2, (
            f"devices='2,3' held {len(owners_23)} device nodes ({sorted(owners_23)}), "
            "expected 2"
        )
        assert owners_23.isdisjoint(owners_01), (
            f"devices='2,3' held {sorted(owners_23)}, overlapping "
            f"devices='0,1' nodes {sorted(owners_01)}"
        )
        outputs_23 = list(
            runner_23.generate(PROMPT, _wan_sampling_params(), use_tqdm=False)
        )
    finally:
        runner_23.close()
    video_23 = extract_wan_video(outputs_23, video_shape)
    np.testing.assert_array_equal(video_23, video_01)
