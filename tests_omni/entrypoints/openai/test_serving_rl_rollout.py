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
"""TPU server test for vLLM-Omni RL rollout stepwise world-model sessions API."""

from __future__ import annotations

import base64
import contextlib
import errno
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

import filelock
import numpy as np
import psutil
import pytest
import requests
import safetensors.torch
import torch
import yaml

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

SERVED_ROLLOUT_MODEL = "dreamzero-tiny"
ROLLOUT_PROMPT = "pick up the red block"
OBS_HEIGHT = 180
OBS_WIDTH = 320
NUM_FRAME_PER_BLOCK = 2
ACTION_HORIZON = 4
ACTION_DIM = 8
MAX_STATE_DIM = 8
SERVER_STARTUP_TIMEOUT_S = 240.0
REQUEST_TIMEOUT_S = 180.0


@dataclass(frozen=True)
class _DreamZeroServerBundle:
    """Running DreamZero ``OmniServer`` exposing ``/v1/realtime/sessions``."""

    base_url: str
    model_dir: str
    proc: subprocess.Popen[bytes]


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


def _dreamzero_checkpoint_worker(wan_dir: str, target_dir: str, port: int) -> None:
    """Build a tiny 2-layer bf16 ``DreamZeroPipeline`` checkpoint on CPU."""
    import torch.distributed as dist
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm_omni.diffusion.models.dreamzero.causal_wan_model import CausalWanModel
    from vllm_omni.platforms import current_omni_platform

    current_platform.device_name = "cpu"
    current_omni_platform.get_diffusion_attn_backend_cls = classmethod(  # type: ignore[method-assign]
        lambda cls, selected_backend, head_size, allow_trtllm_default=False: (
            "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
        )
    )
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(tensor_parallel_size=1, pipeline_parallel_size=1)
    try:
        diffusion_model_cfg: dict[str, Any] = {
            "_target_": (
                "groot.vla.model.dreamzero.modules."
                "wan_video_dit_action_casual_chunk.CausalWanModel"
            ),
            "model_type": "i2v",
            "patch_size": [1, 2, 2],
            "text_len": 512,
            "in_dim": 36,
            "dim": 256,
            "ffn_dim": 512,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 2,
            "num_layers": 2,
            "local_attn_size": 6,
            "sink_size": 1,
            "qk_norm": True,
            "cross_attn_norm": True,
            "eps": 1e-6,
        }
        ctor_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k != "_target_"}
        torch.default_generator.manual_seed(0)
        transformer = CausalWanModel(
            **ctor_kwargs,
            action_dim=ACTION_DIM,
            max_state_dim=MAX_STATE_DIM,
            num_frame_per_block=NUM_FRAME_PER_BLOCK,
        ).to(torch.bfloat16)

        weights: dict[str, torch.Tensor] = {}
        for name, tensor in transformer.state_dict().items():
            prefixed = f"action_head.model.{name}"
            if ".self_attn.qkv." in prefixed:
                q_t, k_t, v_t = tensor.chunk(3, dim=0)
                weights[prefixed.replace(".self_attn.qkv.", ".self_attn.q.")] = (
                    q_t.contiguous()
                )
                weights[prefixed.replace(".self_attn.qkv.", ".self_attn.k.")] = (
                    k_t.contiguous()
                )
                weights[prefixed.replace(".self_attn.qkv.", ".self_attn.v.")] = (
                    v_t.contiguous()
                )
            else:
                weights[prefixed] = tensor.contiguous()

        out_root = Path(target_dir)
        shutil.copytree(Path(wan_dir) / "vae", out_root / "vae")
        shutil.copytree(Path(wan_dir) / "tokenizer", out_root / "tokenizer")

        root_cfg = {
            "model_type": "vla",
            "action_head_cfg": {
                "_target_": (
                    "groot.vla.model.dreamzero.action_head."
                    "wan_flow_matching_action_tf.WANPolicyHead"
                ),
                "config": {
                    "action_dim": ACTION_DIM,
                    "max_action_dim": ACTION_DIM,
                    "max_state_dim": MAX_STATE_DIM,
                    "action_horizon": ACTION_HORIZON,
                    "num_frames": 9,
                    "num_frame_per_block": NUM_FRAME_PER_BLOCK,
                    "decouple_inference_noise": False,
                    "video_inference_final_noise": 0.0,
                    "diffusion_model_cfg": diffusion_model_cfg,
                },
            },
        }
        (out_root / "config.json").write_text(json.dumps(root_cfg, indent=2))

        exp_dir = out_root / "experiment_cfg"
        exp_dir.mkdir(parents=True, exist_ok=True)
        stats_entry = {
            "statistics": {
                "action": {
                    "joint_position": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
                    "gripper_position": {"q01": [0.0], "q99": [1.0]},
                },
                "state": {
                    "joint_position": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
                    "gripper_position": {"q01": [0.0], "q99": [1.0]},
                },
            }
        }
        (exp_dir / "metadata.json").write_text(
            json.dumps({"roboarena": stats_entry, "oxe_droid": stats_entry}, indent=2)
        )
        safetensors.torch.save_file(weights, str(out_root / "model.safetensors"))
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
        if dist.is_initialized():
            dist.destroy_process_group()


def _build_tiny_dreamzero() -> str:
    """Return a cached tiny ``DreamZeroPipeline`` checkpoint built locally."""
    root = os.path.join(tempfile.gettempdir(), "vllm-torchtpu-tiny-models")
    model_dir = os.path.join(root, "tiny-dreamzero-rollout")
    if os.path.isdir(model_dir):
        return model_dir

    wan_dir = tiny_wan2_1_t2v()
    os.makedirs(root, exist_ok=True)
    staging_dir = tempfile.mkdtemp(dir=root)
    port, lock = _reserve_local_port("127.0.0.1")
    try:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_dreamzero_checkpoint_worker,
            args=(wan_dir, staging_dir, port),
        )
        proc.start()
        proc.join(timeout=120.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
            raise TimeoutError("Tiny DreamZero checkpoint builder timed out")
        assert proc.exitcode == 0, (
            f"Tiny DreamZero checkpoint builder failed with exit code {proc.exitcode}"
        )
    finally:
        if lock.is_locked:
            lock.release()
    os.rename(staging_dir, model_dir)
    return model_dir


def _write_dreamzero_deploy_yaml(model_dir: str, yaml_path: Path) -> None:
    """Write a single-device ``ARDiffusionEngine`` DreamZero deploy YAML."""
    deploy_cfg = {
        "pipeline": "dreamzero",
        "async_chunk": False,
        "dtype": "bfloat16",
        "stages": [
            {
                "stage_id": 0,
                "devices": "0",
                "max_num_seqs": 1,
                "enforce_eager": True,
                "model_class_name": "DreamZeroPipeline",
                "engine_backend": (
                    "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
                ),
                "model_paths": {
                    "tokenizer": os.path.join(model_dir, "tokenizer"),
                    "vae": os.path.join(model_dir, "vae"),
                },
                "model_config": {
                    "default_robot_embodiment": "roboarena",
                    "num_inference_steps": 1,
                    "cfg_scale": 1.0,
                    "seed": 42,
                    "ar_diffusion_kv_config": {"warmup_cudagraph": False},
                    "policy_server_config": {
                        "image_resolution": [OBS_HEIGHT, OBS_WIDTH],
                        "n_external_cameras": 2,
                        "needs_wrist_camera": True,
                        "needs_stereo_camera": False,
                        "needs_session_id": True,
                        "action_space": "joint_position",
                    },
                },
            }
        ],
    }
    yaml_path.write_text(yaml.safe_dump(deploy_cfg, sort_keys=False))


def _decode_latent_observation(next_obs: dict[str, Any]) -> torch.Tensor:
    """Decode ``base64_raw_tensor`` ``video_latent`` into a float32 tensor."""
    raw = base64.b64decode(next_obs["video_latent"])
    np_dtype = np.dtype(next_obs["dtype"])
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(next_obs["shape"]).copy()
    return torch.from_numpy(arr).to(torch.float32)


@pytest.fixture(scope="module")
def _dreamzero_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_DreamZeroServerBundle, None, None]:
    model_dir = _build_tiny_dreamzero()
    tmp_dir = tmp_path_factory.mktemp("dreamzero_rollout_server")
    deploy_yaml = tmp_dir / "dreamzero_tpu.yaml"
    _write_dreamzero_deploy_yaml(model_dir, deploy_yaml)
    base_url, proc, lock = _start_omni_server(
        model_dir,
        [
            "--deploy-config",
            str(deploy_yaml),
            "--served-model-name",
            SERVED_ROLLOUT_MODEL,
        ],
    )
    try:
        yield _DreamZeroServerBundle(
            base_url=base_url,
            model_dir=model_dir,
            proc=proc,
        )
    finally:
        _stop_server_process(proc, lock)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Interactive world models (DreamZero on ARDiffusionEngine) fail on TPU: "
        "CausalWanModel complex128 RoPE views under inference mode and causal SDPA "
        "fallback return an inference_error or non-finite latents"
    ),
)
def test_rl_rollout_stepwise_sessions_routes(
    _dreamzero_bundle: _DreamZeroServerBundle,
) -> None:
    """RL Rollout Stepwise Sessions API.

    - TPU path: Starts ``OmniServer`` on a 2-layer bf16 ``DreamZeroPipeline``
      backed by ``ARDiffusionEngine`` and runs ``POST /v1/realtime/sessions``,
      ``GET .../status``, ``POST .../step``, ``POST .../reset``, and
      ``POST .../close``.
    - Setup check: Asserts session creation returns HTTP 200 (not 501) and initial
      status reports ``committed_step_id == -1`` and ``closed is False``.
    - Outcome check: Asserts ``POST .../step`` succeeds with ``error is None``,
      finite ``base64_raw_tensor`` ``video_latent``, step-count advance on status,
      reset back to ``-1``, and clean session close.
    """
    assert _dreamzero_bundle.proc.poll() is None, (
        f"OmniServer exited unexpectedly with code {_dreamzero_bundle.proc.poll()}"
    )

    create_resp = requests.post(
        f"{_dreamzero_bundle.base_url}/v1/realtime/sessions",
        json={"model": SERVED_ROLLOUT_MODEL, "mode": "world_model_env"},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert create_resp.status_code == 200, (
        f"POST /v1/realtime/sessions returned {create_resp.status_code}: "
        f"{create_resp.text}"
    )
    session_id = create_resp.json()["session_id"]

    status_before = requests.get(
        f"{_dreamzero_bundle.base_url}/v1/realtime/sessions/{session_id}/status",
        timeout=REQUEST_TIMEOUT_S,
    )
    assert status_before.status_code == 200, (
        f"GET status returned {status_before.status_code}: {status_before.text}"
    )
    status_before_body = status_before.json()
    assert status_before_body["committed_step_id"] == -1, (
        f"Initial committed_step_id {status_before_body['committed_step_id']} != -1"
    )
    assert status_before_body["closed"] is False, (
        f"Expected open session on create, got {status_before_body}"
    )

    rng = np.random.default_rng(0)
    cam_img = rng.integers(
        0, 255, size=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8
    ).tolist()
    step_resp = requests.post(
        f"{_dreamzero_bundle.base_url}/v1/realtime/sessions/{session_id}/step",
        json={
            "step_id": 0,
            "use_session_context": True,
            "observation": {
                "prompt": ROLLOUT_PROMPT,
                "images": {
                    "observation/exterior_image_0_left": cam_img,
                    "observation/exterior_image_1_left": cam_img,
                    "observation/wrist_image_left": cam_img,
                },
                "state": [0.0] * 8,
            },
            "action": {
                "joint_positions": [0.05] * 7,
                "gripper_position": [0.0],
            },
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    assert step_resp.status_code == 200, (
        f"POST step returned {step_resp.status_code}: {step_resp.text}"
    )
    step_body = step_resp.json()
    assert step_body["error"] is None, (
        f"Rollout step returned error on TPU: {step_body['error']}"
    )
    next_obs = step_body["next_observation"]
    assert next_obs["encoding"] == "base64_raw_tensor", (
        f"Expected base64_raw_tensor encoding, got {next_obs}"
    )
    latent = _decode_latent_observation(next_obs)
    expected_latent_frames = 1 + NUM_FRAME_PER_BLOCK
    expected_latent_shape = (
        1,
        16,
        expected_latent_frames,
        OBS_HEIGHT // 8,
        OBS_WIDTH // 8,
    )
    assert tuple(latent.shape) == expected_latent_shape, (
        f"Rollout latent shape {tuple(latent.shape)} != {expected_latent_shape}"
    )
    assert bool(torch.all(torch.isfinite(latent))), (
        "Rollout next_observation video_latent contained non-finite values"
    )

    status_after = requests.get(
        f"{_dreamzero_bundle.base_url}/v1/realtime/sessions/{session_id}/status",
        timeout=REQUEST_TIMEOUT_S,
    ).json()
    assert status_after["committed_step_id"] == 0, (
        f"Expected committed_step_id 0 after step 0, got {status_after}"
    )

    reset_resp = requests.post(
        f"{_dreamzero_bundle.base_url}/v1/realtime/sessions/{session_id}/reset",
        timeout=REQUEST_TIMEOUT_S,
    )
    assert reset_resp.status_code == 200, (
        f"POST reset returned {reset_resp.status_code}: {reset_resp.text}"
    )
    assert reset_resp.json()["committed_step_id"] == -1, (
        f"Expected committed_step_id -1 after reset, got {reset_resp.json()}"
    )

    close_resp = requests.post(
        f"{_dreamzero_bundle.base_url}/v1/realtime/sessions/{session_id}/close",
        timeout=REQUEST_TIMEOUT_S,
    )
    assert close_resp.status_code == 200, (
        f"POST close returned {close_resp.status_code}: {close_resp.text}"
    )
    assert close_resp.json()["closed"] is True, (
        f"Expected closed=True on close, got {close_resp.json()}"
    )
