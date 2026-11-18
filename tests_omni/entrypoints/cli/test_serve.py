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
"""TPU tests for the vllm-omni serve CLI, --api-server-count, and --stage-overrides."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import io
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest
import requests
from PIL import Image
from transformers import AutoTokenizer, Qwen3OmniMoeConfig
from vllm.entrypoints.openai.api_server import setup_server
from vllm.utils.network_utils import get_open_port
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
from vllm_omni.config.resolver import OmniConfigResolution, resolve_omni_config
from vllm_omni.entrypoints.api_server_shared_dir import (
    create_api_server_shared_dir,
    remove_api_server_shared_dir,
)
from vllm_omni.entrypoints.cli.serve import (
    OmniServeCommand,
    _build_multi_api_stage_runtime,
    _EngineAddressListener,
    _parse_stage_overrides,
    _start_api_server_process_manager,
    run_omni_api_server_worker_proc,
)
from vllm_omni.entrypoints.utils import prepare_stage_config_inputs
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, PR_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell, tiny_qwen_image

pytestmark = [pytest.mark.omni]

IMAGE_SIZE = 256
IMAGE_STEPS = 2
SEED_PRIMARY = 42
SEED_CONTROL = 99
NUM_API_SERVERS = 2
SERVER_READY_TIMEOUT_S = 240.0
POLL_INTERVAL_S = 0.5
REQUEST_TIMEOUT_S = 15.0
GENERATION_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 15.0
PROMPT = "a geometric cube on a plain table"


def _build_serve_parser() -> tuple[TrackingArgumentParser, OmniServeCommand]:
    parser = TrackingArgumentParser(prog="vllm")
    subparsers = parser.add_subparsers(dest="subparser")
    cmd = OmniServeCommand()
    cmd.subparser_init(subparsers)
    return parser, cmd


def _resolve_from_cli_args(args: TrackingNamespace) -> OmniConfigResolution:
    kwargs = args.get_explicit_kwargs_dict()
    model = kwargs.pop("model", None) or args.model_tag or args.model
    trust_remote_code = kwargs.get("trust_remote_code")
    if trust_remote_code is False:
        trust_remote_code = None
    inputs = prepare_stage_config_inputs(
        model,
        kwargs,
        trust_remote_code=trust_remote_code,
        snapshot_model=False,
    )
    return resolve_omni_config(
        inputs.model,
        trust_remote_code=inputs.trust_remote_code,
        deploy_config_path=inputs.deploy_config_path,
        cli_overrides=inputs.kwargs,
        stage_overrides=inputs.stage_overrides,
        strategy_config_path=inputs.strategy_config_path,
    )


def _write_tiny_qwen3_omni_config(target_dir: Path, *, with_tokenizer: bool) -> str:
    target_dir.mkdir(parents=True, exist_ok=True)
    cfg = Qwen3OmniMoeConfig(
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
    )
    cfg.thinker_config.text_config.num_hidden_layers = 2
    cfg.talker_config.text_config.num_hidden_layers = 2
    cfg.code2wav_config.num_hidden_layers = 2
    cfg.save_pretrained(target_dir)
    if with_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            tiny_qwen_image(),
            subfolder="tokenizer",
        )
        tokenizer.save_pretrained(target_dir)
    return str(target_dir)


def _write_minimal_flux_index(target_dir: Path) -> str:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}),
        encoding="utf-8",
    )
    return str(target_dir)


def _terminate_process_tree(proc: subprocess.Popen[bytes]) -> None:
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
def _running_serve_cli(
    model_dir: str,
) -> Iterator[tuple[subprocess.Popen[bytes], str]]:
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
    # Inherit stdout/stderr: nothing reads a PIPE until exit, and once it fills,
    # the server blocks on its next log write.
    proc = subprocess.Popen(cmd)
    try:
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve --omni exited early with code {proc.poll()}"
                )
            try:
                resp = requests.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"vllm serve --omni did not pass /health within "
                    f"{SERVER_READY_TIMEOUT_S}s"
                )
            time.sleep(POLL_INTERVAL_S)
        yield proc, base_url
    finally:
        _terminate_process_tree(proc)
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


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
def test_serve_cli_diffusion() -> None:
    """Serve CLI (vllm serve --omni / vllm-omni serve).

    - TPU path: spawns
      ``python -m vllm_omni.entrypoints.cli.main serve <model_dir> --omni``
      on ``tiny_flux_schnell()`` in a TPU subprocess and serves HTTP image
      generation requests.
    - Setup check: the server subprocess stays alive, ``GET /health`` returns
      HTTP 200, and ``GET /v1/models`` lists ``model_dir``.
    - Outcome check: repeated ``seed=42`` ``POST /v1/images/generations`` calls
      return bitwise-identical ``(256, 256, 3)`` RGB images while differing
      from a ``seed=99`` control request.
    """
    model_dir = tiny_flux_schnell()
    with _running_serve_cli(model_dir) as (proc, base_url):
        assert proc.poll() is None, (
            f"Serve CLI subprocess exited prematurely with code {proc.poll()}"
        )
        health_resp = requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_S)
        assert health_resp.status_code == 200, (
            f"Serve CLI GET /health status {health_resp.status_code} != 200"
        )
        models_resp = requests.get(f"{base_url}/v1/models", timeout=REQUEST_TIMEOUT_S)
        assert models_resp.status_code == 200, (
            f"Serve CLI GET /v1/models status {models_resp.status_code} != 200"
        )
        served_models = [item["id"] for item in models_resp.json()["data"]]
        assert served_models == [model_dir], (
            f"Serve CLI /v1/models {served_models} != {[model_dir]}"
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

    img_a = _decode_b64_image_response(resp_a, "serve_seed42_a")
    img_b = _decode_b64_image_response(resp_b, "serve_seed42_b")
    img_ctrl = _decode_b64_image_response(resp_ctrl, "serve_seed99")

    assert np.array_equal(img_b, img_a), (
        "Serve CLI repeated seed=42 outputs diverged: "
        f"max_abs={int(np.abs(img_b.astype(int) - img_a.astype(int)).max())}"
    )
    assert not np.array_equal(img_ctrl, img_a), (
        "Serve CLI seed=99 output unexpectedly matched seed=42 bitwise"
    )


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "--api-server-count > 1 requires EngineCore AR stages, and TPUWorker "
        "lacks OmniConnectorModelRunnerMixin on main (vllm-torchtpu PR #2065)"
    ),
)
def test_multi_api_server_count(tmp_path: Path) -> None:
    """Multiple API server processes (--api-server-count).

    - TPU path: verifies ``OmniServeCommand.validate`` rejects
      ``--api-server-count 2`` on diffusion, then launches the multi-API-server
      EngineCore path (``_build_multi_api_stage_runtime`` +
      ``launch_stage_engines(2)`` + ``_start_api_server_process_manager``) on a
      tiny local ``Qwen3OmniMoeConfig`` directory.
    - Setup check: ``len(api_server_manager.processes) == 2``, both API worker
      subprocesses stay alive, and ``GET /health`` returns HTTP 200.
    - Outcome check: concurrent ``POST /v1/chat/completions`` requests across
      the shared socket match a single-request greedy reference.
    """
    parser, cmd = _build_serve_parser()

    diff_dir = tiny_flux_schnell()
    diff_args = parser.parse_args(
        ["serve", diff_dir, "--omni", "--api-server-count", str(NUM_API_SERVERS)]
    )
    with pytest.raises(ValueError, match="not supported for diffusion"):
        cmd.validate(diff_args)

    ar_dir = _write_tiny_qwen3_omni_config(
        tmp_path / "tiny_qwen3_omni",
        with_tokenizer=True,
    )
    stage_overrides = {
        str(stage_id): {
            "load_format": "dummy",
            "max_model_len": 512,
            "enforce_eager": True,
        }
        for stage_id in (0, 1, 2)
    }
    port = get_open_port()
    base_url = f"http://127.0.0.1:{port}"
    ar_args = parser.parse_args(
        [
            "serve",
            ar_dir,
            "--omni",
            "--api-server-count",
            str(NUM_API_SERVERS),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--stage-overrides",
            json.dumps(stage_overrides),
        ]
    )
    ar_args.model = ar_args.model_tag
    cmd.validate(ar_args)

    wait_for_tpu_release()
    setup_multiprocess_prometheus()
    listen_address, sock = setup_server(ar_args, reuse_port=True)
    shared_dir = create_api_server_shared_dir()
    ar_args._omni_api_server_shared_dir = shared_dir
    stage_runtime = None
    api_server_manager = None
    engine_launch = None
    try:
        stage_runtime = _build_multi_api_stage_runtime(ar_args, NUM_API_SERVERS)
        with stage_runtime.launch_stage_engines(NUM_API_SERVERS) as engine_launch:
            ar_args._omni_stage_client_configs = engine_launch.client_configs
            primary_addresses = engine_launch.resources[0].addresses
            assert primary_addresses is not None, (
                "Primary stage engine returned no ZMQ addresses"
            )
            api_server_manager = _start_api_server_process_manager(
                listen_address=listen_address,
                sock=sock,
                args=ar_args,
                num_servers=NUM_API_SERVERS,
                input_listeners=[
                    _EngineAddressListener(addr) for addr in primary_addresses.inputs
                ],
                output_listeners=[
                    _EngineAddressListener(addr) for addr in primary_addresses.outputs
                ],
                target_server_fn=run_omni_api_server_worker_proc,
            )
            engine_launch.watched_frontend_processes.extend(
                api_server_manager.processes
            )

        assert api_server_manager is not None, (
            "Expected non-None api_server_manager after launch"
        )
        assert len(api_server_manager.processes) == NUM_API_SERVERS, (
            f"Spawned {len(api_server_manager.processes)} API workers, "
            f"expected {NUM_API_SERVERS}"
        )
        assert all(proc.is_alive() for proc in api_server_manager.processes), (
            "One or more multi-API worker processes exited during startup"
        )

        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while True:
            try:
                if requests.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Multi-API server did not pass /health within "
                    f"{SERVER_READY_TIMEOUT_S}s"
                )
            time.sleep(POLL_INTERVAL_S)

        chat_payload = {
            "model": ar_dir,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": SEED_PRIMARY,
            "modalities": ["text"],
        }
        ref_resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=chat_payload,
            timeout=GENERATION_TIMEOUT_S,
        )
        assert ref_resp.status_code == 200, (
            f"Reference /v1/chat/completions status {ref_resp.status_code} != 200"
        )
        ref_text = ref_resp.json()["choices"][0]["message"]["content"]

        def _post_chat() -> str:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json=chat_payload,
                timeout=GENERATION_TIMEOUT_S,
            )
            assert resp.status_code == 200, (
                f"Concurrent /v1/chat/completions status {resp.status_code} != 200"
            )
            return str(resp.json()["choices"][0]["message"]["content"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_API_SERVERS) as pool:
            futures = [pool.submit(_post_chat) for _ in range(NUM_API_SERVERS * 2)]
            concurrent_texts = [fut.result() for fut in futures]

        assert concurrent_texts == [ref_text] * len(concurrent_texts), (
            f"Concurrent multi-API responses {concurrent_texts} != "
            f"expected {[ref_text] * len(concurrent_texts)}"
        )
    finally:
        if api_server_manager is not None:
            api_server_manager.shutdown(timeout=SHUTDOWN_TIMEOUT_S)
        if engine_launch is not None:
            engine_launch.shutdown()
        if stage_runtime is not None:
            stage_runtime.shutdown()
        sock.close()
        remove_api_server_shared_dir(shared_dir)
        wait_for_tpu_release()


@pytest.mark.timeout(PR_TIMEOUT_S)
def test_stage_overrides_cli_option(tmp_path: Path) -> None:
    """Stage Overrides CLI Option.

    - TPU path: parses ``serve <model> --omni --stage-overrides '<JSON>'``
      alongside global CLI flags through ``TrackingArgumentParser`` and
      ``OmniServeCommand.subparser_init``, then resolves both multi-stage
      registered pipelines and single-stage generic diffusion configs with
      ``OmniTpuPlatform`` active.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``args.get_explicit_kwargs_dict()`` captures ``stage_overrides``.
    - Outcome check: per-stage JSON overrides win over global CLI flags and
      deploy YAML defaults on targeted stages while leaving sibling stages on
      global/YAML values, and malformed JSON or non-integer stage IDs raise
      ``argparse.ArgumentTypeError``.
    """
    assert current_omni_platform.device_type == "tpu", (
        "Expected OmniTpuPlatform with device_type='tpu', got "
        f"{current_omni_platform.device_type!r}"
    )

    parser, _ = _build_serve_parser()
    qwen_dir = _write_tiny_qwen3_omni_config(
        tmp_path / "qwen3_omni_cfg",
        with_tokenizer=False,
    )
    overrides_payload = {
        "0": {"gpu_memory_utilization": 0.55},
        "2": {"enforce_eager": True, "max_num_seqs": 8},
    }
    multi_args = parser.parse_args(
        [
            "serve",
            qwen_dir,
            "--omni",
            "--gpu-memory-utilization",
            "0.75",
            "--max-num-seqs",
            "16",
            "--stage-overrides",
            json.dumps(overrides_payload),
        ]
    )
    explicit = multi_args.get_explicit_kwargs_dict()
    assert explicit["stage_overrides"] == overrides_payload, (
        f"Explicit stage_overrides {explicit.get('stage_overrides')} != "
        f"{overrides_payload}"
    )

    ref_multi_args = parser.parse_args(
        [
            "serve",
            qwen_dir,
            "--omni",
            "--gpu-memory-utilization",
            "0.75",
            "--max-num-seqs",
            "16",
        ]
    )
    resolved_multi = _resolve_from_cli_args(multi_args)
    ref_multi = _resolve_from_cli_args(ref_multi_args)

    stage0 = resolved_multi.stage_by_id(0)
    stage1 = resolved_multi.stage_by_id(1)
    stage2 = resolved_multi.stage_by_id(2)
    ref_stage1 = ref_multi.stage_by_id(1)
    ref_stage2 = ref_multi.stage_by_id(2)

    assert stage0.cache_config.gpu_memory_utilization == 0.55, (
        "Stage 0 gpu_memory_utilization "
        f"{stage0.cache_config.gpu_memory_utilization} != 0.55"
    )
    assert stage0.scheduler_config.max_num_seqs == 16, (
        f"Stage 0 max_num_seqs {stage0.scheduler_config.max_num_seqs} != 16"
    )
    assert (
        stage1.cache_config.gpu_memory_utilization
        == ref_stage1.cache_config.gpu_memory_utilization
    ), (
        "Untargeted stage 1 gpu_memory_utilization "
        f"{stage1.cache_config.gpu_memory_utilization} != "
        f"{ref_stage1.cache_config.gpu_memory_utilization}"
    )
    assert stage1.scheduler_config.max_num_seqs == 16, (
        f"Untargeted stage 1 max_num_seqs {stage1.scheduler_config.max_num_seqs} != 16"
    )
    assert ref_stage2.model_config.enforce_eager is False, (
        "Reference stage 2 enforce_eager "
        f"{ref_stage2.model_config.enforce_eager} != False"
    )
    assert stage2.model_config.enforce_eager is True, (
        f"Stage 2 enforce_eager {stage2.model_config.enforce_eager} != True"
    )
    assert stage2.scheduler_config.max_num_seqs == 8, (
        f"Stage 2 max_num_seqs {stage2.scheduler_config.max_num_seqs} != 8"
    )

    flux_dir = _write_minimal_flux_index(tmp_path / "flux_cfg")
    diff_args = parser.parse_args(
        [
            "serve",
            flux_dir,
            "--omni",
            "--max-num-seqs",
            "4",
            "--stage-overrides",
            json.dumps({"0": {"max_num_seqs": 2, "vae_use_slicing": True}}),
        ]
    )
    ref_diff_args = parser.parse_args(
        ["serve", flux_dir, "--omni", "--max-num-seqs", "4"]
    )
    resolved_diff = _resolve_from_cli_args(diff_args)
    ref_diff = _resolve_from_cli_args(ref_diff_args)

    diff_stage0 = resolved_diff.stage_by_id(0)
    ref_diff_stage0 = ref_diff.stage_by_id(0)
    assert ref_diff_stage0.scheduler_config.max_num_seqs == 4, (
        "Reference diffusion max_num_seqs "
        f"{ref_diff_stage0.scheduler_config.max_num_seqs} != 4"
    )
    assert diff_stage0.scheduler_config.max_num_seqs == 2, (
        "Overridden diffusion max_num_seqs "
        f"{diff_stage0.scheduler_config.max_num_seqs} != 2"
    )
    assert ref_diff_stage0.diffusion_config.vae_use_slicing is False, (
        "Reference diffusion vae_use_slicing "
        f"{ref_diff_stage0.diffusion_config.vae_use_slicing} != False"
    )
    assert diff_stage0.diffusion_config.vae_use_slicing is True, (
        "Overridden diffusion vae_use_slicing "
        f"{diff_stage0.diffusion_config.vae_use_slicing} != True"
    )

    with pytest.raises(argparse.ArgumentTypeError, match="not valid JSON"):
        _parse_stage_overrides("{malformed")
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="non-negative integer stage ids",
    ):
        _parse_stage_overrides('{"stage_zero": {"max_num_seqs": 2}}')
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["serve", flux_dir, "--omni", "--stage-overrides", "{malformed"]
        )
