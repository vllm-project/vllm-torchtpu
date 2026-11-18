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
"""TPU server test for vLLM-Omni independent stage execution (/v1/run)."""

from __future__ import annotations

import contextlib
import errno
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import filelock
import numpy as np
import psutil
import pytest
import requests
import torch
import yaml
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_wan2_1_t2v

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

RUN_PROMPT = "a paper boat floating down a sunlit stream"
RUN_WIDTH = 256
RUN_HEIGHT = 256
RUN_NUM_FRAMES = 5
RUN_STEPS = 2
RUN_GUIDANCE_SCALE = 1.0
RUN_SEED = 42
SERVER_STARTUP_TIMEOUT_S = 240.0
REQUEST_TIMEOUT_S = 180.0
RUN_MAX_REL_L2 = 2e-2
RUN_MAX_ABS = 2e-2


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
    model_dir: str,
    serve_args: list[str],
    env_overrides: dict[str, str],
) -> tuple[str, subprocess.Popen[bytes], filelock.FileLock]:
    """Start ``vllm serve --omni`` with ``env_overrides`` and poll ``/health``."""
    wait_for_tpu_release()
    host = "127.0.0.1"
    port, lock = _reserve_local_port(host)
    base_url = f"http://{host}:{port}"
    env = {**os.environ, **env_overrides}
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
    proc = subprocess.Popen(cmd, env=env)
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


def _wan_single_stage_worker(model_dir: str, out_npy: str) -> None:
    """Run single-stage offline ``Omni`` Wan generation for parity comparison."""
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=model_dir, log_stats=False)
    try:
        prompt = OmniTextPrompt(prompt=RUN_PROMPT, modalities=["video"])
        params = OmniDiffusionSamplingParams(
            width=RUN_WIDTH,
            height=RUN_HEIGHT,
            num_frames=RUN_NUM_FRAMES,
            num_inference_steps=RUN_STEPS,
            guidance_scale=RUN_GUIDANCE_SCALE,
            seed=RUN_SEED,
        )
        outputs = list(omni.generate(prompt, [params]))
        video = np.asarray(outputs[0].images[0], dtype=np.float32)
        if video.ndim == 5:
            video = video[0]
        np.save(out_npy, video)
    finally:
        omni.close()


def _run_single_stage_reference(model_dir: str, tmp_dir: Path) -> torch.Tensor:
    """Spawn and join single-stage Wan TPU reference worker."""
    wait_for_tpu_release()
    out_npy = tmp_dir / "wan_single_stage_ref.npy"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_wan_single_stage_worker,
        args=(model_dir, str(out_npy)),
    )
    proc.start()
    proc.join(timeout=SERVER_STARTUP_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        raise TimeoutError("Single-stage Wan reference worker timed out")
    assert proc.exitcode == 0, (
        f"Single-stage Wan reference worker exited with code {proc.exitcode}"
    )
    wait_for_tpu_release()
    return torch.from_numpy(np.load(out_npy)).to(torch.float32)


def _write_wan_disagg_deploy_yaml(yaml_path: Path) -> None:
    """Write a 2-stage disaggregated ``wan2_2_eg`` deploy config on chips 0 and 1."""
    deploy_cfg = {
        "pipeline": "wan2_2_eg",
        "async_chunk": False,
        "distributed_executor_backend": "mp",
        "connectors": {
            "wan_encode_connector": {"name": "SharedMemoryConnector"},
        },
        "stages": [
            {
                "stage_id": 0,
                "devices": "0",
                "max_num_seqs": 1,
                "enforce_eager": True,
                "tensor_parallel_size": 1,
                "output_connectors": {"to_stage_1": "wan_encode_connector"},
                "default_sampling_params": {"seed": RUN_SEED},
            },
            {
                "stage_id": 1,
                "devices": "1",
                "max_num_seqs": 1,
                "enforce_eager": True,
                "tensor_parallel_size": 1,
                "input_connectors": {"from_stage_0": "wan_encode_connector"},
                "default_sampling_params": {"seed": RUN_SEED},
            },
        ],
    }
    yaml_path.write_text(yaml.safe_dump(deploy_cfg, sort_keys=False))


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_independent_stage_execution_run_route(tmp_path: Path) -> None:
    """Independent Stage Execution API.

    - TPU path: Imports ``ServingRun`` and ``decode_output`` inside the test body
      (strict xfail while ``serving_run`` is absent in vllm-omni 0.32.0rc1), then
      runs stage 0 and stage 1 of a 2-stage disaggregated Wan pipeline over
      ``POST /v1/run`` with ``VLLM_OMNI_ENABLE_RUN_ENDPOINT=1``.
    - Setup check: Asserts stage 0 ``POST /v1/run`` returns HTTP 200 (not 400
      disabled) with ``stage_id == 1`` and an encoded ``stage_input`` payload.
    - Outcome check: Decodes stage 1 ``output`` via ``decode_output`` and checks
      ``assert_parity`` against single-stage ``tiny_wan2_1_t2v()`` TPU output.
    """
    from vllm_omni.entrypoints.openai.serving_run import ServingRun, decode_output

    assert callable(ServingRun.run), "Expected ServingRun.run to be callable"

    model_dir = tiny_wan2_1_t2v()
    ref_video = _run_single_stage_reference(model_dir, tmp_path)

    deploy_yaml = tmp_path / "wan_eg_disagg.yaml"
    _write_wan_disagg_deploy_yaml(deploy_yaml)

    sampling_dict = {
        "width": RUN_WIDTH,
        "height": RUN_HEIGHT,
        "num_frames": RUN_NUM_FRAMES,
        "num_inference_steps": RUN_STEPS,
        "guidance_scale": RUN_GUIDANCE_SCALE,
        "seed": RUN_SEED,
    }
    base_url, proc, lock = _start_omni_server(
        model_dir,
        ["--deploy-config", str(deploy_yaml)],
        {
            "VLLM_OMNI_ENABLE_RUN_ENDPOINT": "1",
            "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
        },
    )
    try:
        stage0_resp = requests.post(
            f"{base_url}/v1/run",
            json={
                "stage_id": 0,
                "stage_input": {"prompt": RUN_PROMPT},
                "sampling_params": [sampling_dict, sampling_dict],
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        assert stage0_resp.status_code == 200, (
            f"Stage 0 POST /v1/run returned {stage0_resp.status_code}: "
            f"{stage0_resp.text}"
        )
        stage0_body = stage0_resp.json()
        assert stage0_body["stage_id"] == 1, (
            f"Expected next stage_id 1, got {stage0_body['stage_id']!r}"
        )
        assert len(stage0_body["stage_input"]) > 0, (
            "Expected non-empty base64 stage_input from stage 0"
        )

        stage1_resp = requests.post(
            f"{base_url}/v1/run",
            json={
                "stage_id": 1,
                "stage_input": stage0_body["stage_input"],
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        assert stage1_resp.status_code == 200, (
            f"Stage 1 POST /v1/run returned {stage1_resp.status_code}: "
            f"{stage1_resp.text}"
        )
        stage1_body = stage1_resp.json()
        final_output = decode_output(stage1_body["output"])
    finally:
        _stop_server_process(proc, lock)

    # The Wan post-process returns one (batch=1, F, H, W, 3) array per request.
    raw_video = np.asarray(final_output.images[0], dtype=np.float32)
    expected_shape = (1, RUN_NUM_FRAMES, RUN_HEIGHT, RUN_WIDTH, 3)
    assert raw_video.shape == expected_shape, (
        f"Disaggregated /v1/run video shape {raw_video.shape} != {expected_shape}"
    )
    actual_video = torch.from_numpy(raw_video[0]).to(torch.float32)

    assert_parity(
        actual_video,
        ref_video,
        name="POST /v1/run 2-stage Wan vs single-stage Wan TPU",
        max_rel_l2=RUN_MAX_REL_L2,
        max_abs=RUN_MAX_ABS,
    )
