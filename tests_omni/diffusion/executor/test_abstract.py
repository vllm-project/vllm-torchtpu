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
"""TPU tests for vLLM-Omni diffusion distributed executor backends."""

from __future__ import annotations

import multiprocessing
from multiprocessing.queues import Queue
from pathlib import Path

import numpy as np
import pytest
import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.ray_executor import RayDiffusionExecutor
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests.tpu_release import tpu_device_owners, wait_for_tpu_release
from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

IMAGE_SIZE = 256
PROMPT = "A geometric lantern glowing over calm water"
WARMUP_SEED = 7
TEST_SEED = 42
CHILD_JOIN_TIMEOUT_S = 240.0
# Post-warmup uni vs mp bf16 outputs on [0, 255] RGB pixels.
UNI_MP_MAX_REL_L2 = 1e-3
UNI_MP_MAX_ABS = 1.0


def _make_od_config(backend: str | None, num_gpus: int) -> OmniDiffusionConfig:
    cfg = object.__new__(OmniDiffusionConfig)
    cfg.distributed_executor_backend = backend
    cfg.num_gpus = num_gpus
    return cfg


def _generate_rgb_array(runner: OmniRunner, seed: int) -> np.ndarray:
    params = OmniDiffusionSamplingParams(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=seed,
    )
    outputs = list(runner.generate(PROMPT, params))
    img = outputs[0].images[0]
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _run_uniproc_in_child(
    model_dir: str,
    result_queue: Queue[tuple[str, bool, np.ndarray]],
) -> None:
    with OmniRunner(
        model_dir,
        distributed_executor_backend="uni",
        stage_init_timeout=120,
        init_timeout=240,
    ) as runner:
        assert runner.omni is not None, (
            "UniProc OmniRunner did not initialize runner.omni"
        )
        client = runner.omni.engine.stage_pools[0].stage_client
        executor = client._engine.executor
        # Flush any startup-warmup tail ops before the measured request.
        _generate_rgb_array(runner, WARMUP_SEED)
        arr = _generate_rgb_array(runner, TEST_SEED)
        result_queue.put((type(executor).__name__, executor.uses_multiproc, arr))


def test_distributed_executor_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed executor backends.

    - TPU path: resolve ``DiffusionExecutor.get_class`` across backends and run
      ``tiny_flux_schnell()`` on TPU with ``uni`` (child process) and ``mp``
      (``OmniRunner``).
    - Setup check: assert ``UniProcDiffusionExecutor`` (``uses_multiproc=False``)
      vs ``MultiprocDiffusionExecutor`` (``uses_multiproc=True``) and full TPU
      device release between runs.
    - Outcome check: compare ``(256, 256, 3)`` RGB outputs between ``uni`` and
      ``mp`` via ``assert_parity``.
    """
    assert DiffusionExecutor.get_class(_make_od_config(None, 1)) is (
        UniProcDiffusionExecutor
    ), "1-device default must resolve to UniProcDiffusionExecutor"
    assert DiffusionExecutor.get_class(_make_od_config(None, 2)) is (
        MultiprocDiffusionExecutor
    ), "Multi-device default must resolve to MultiprocDiffusionExecutor"
    assert DiffusionExecutor.get_class(_make_od_config("uni", 1)) is (
        UniProcDiffusionExecutor
    ), "Explicit 'uni' must resolve to UniProcDiffusionExecutor"
    assert DiffusionExecutor.get_class(_make_od_config("mp", 1)) is (
        MultiprocDiffusionExecutor
    ), "Explicit 'mp' must resolve to MultiprocDiffusionExecutor"
    assert DiffusionExecutor.get_class(_make_od_config("ray", 1)) is (
        RayDiffusionExecutor
    ), "Explicit 'ray' must resolve to RayDiffusionExecutor"
    with pytest.raises(NotImplementedError):
        DiffusionExecutor.get_class(_make_od_config("external_launcher", 1))

    monkeypatch.setenv(
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        str(tmp_path / "tpu_compile_cache"),
    )
    model_dir = tiny_flux_schnell()

    spawn_ctx = multiprocessing.get_context("spawn")
    result_queue: Queue[tuple[str, bool, np.ndarray]] = spawn_ctx.Queue()
    proc = spawn_ctx.Process(
        target=_run_uniproc_in_child,
        args=(model_dir, result_queue),
    )
    proc.start()
    try:
        uni_cls_name, uni_uses_mp, uni_arr = result_queue.get(
            timeout=CHILD_JOIN_TIMEOUT_S
        )
        proc.join(timeout=30.0)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10.0)
        result_queue.close()

    assert proc.exitcode == 0, (
        f"UniProc child process exited with code {proc.exitcode}, expected 0"
    )
    wait_for_tpu_release()
    assert tpu_device_owners() == {}, (
        f"Expected all TPU devices released after uni child, got {tpu_device_owners()}"
    )
    assert uni_cls_name == "UniProcDiffusionExecutor", (
        f"Expected 'UniProcDiffusionExecutor', got {uni_cls_name!r}"
    )
    assert uni_uses_mp is False, (
        f"Expected UniProcDiffusionExecutor.uses_multiproc=False, got {uni_uses_mp}"
    )

    with OmniRunner(
        model_dir,
        distributed_executor_backend="mp",
        stage_init_timeout=120,
        init_timeout=240,
    ) as mp_runner:
        assert mp_runner.omni is not None, (
            "Multiproc OmniRunner did not initialize mp_runner.omni"
        )
        mp_client = mp_runner.omni.engine.stage_pools[0].stage_client
        mp_executor = mp_client._engine.executor
        assert isinstance(mp_executor, MultiprocDiffusionExecutor), (
            f"Expected MultiprocDiffusionExecutor, got {type(mp_executor).__name__}"
        )
        assert mp_executor.uses_multiproc is True, (
            "Expected MultiprocDiffusionExecutor.uses_multiproc=True, "
            f"got {mp_executor.uses_multiproc}"
        )
        # Flush startup-warmup tail ops in the mp worker (omni/patches.py).
        _generate_rgb_array(mp_runner, WARMUP_SEED)
        mp_arr = _generate_rgb_array(mp_runner, TEST_SEED)

    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert uni_arr.shape == expected_shape, (
        f"uni image shape {uni_arr.shape} != expected {expected_shape}"
    )
    assert mp_arr.shape == expected_shape, (
        f"mp image shape {mp_arr.shape} != expected {expected_shape}"
    )
    assert_parity(
        torch.from_numpy(mp_arr),
        torch.from_numpy(uni_arr),
        name="mp vs uni tiny_flux_schnell",
        max_rel_l2=UNI_MP_MAX_REL_L2,
        max_abs=UNI_MP_MAX_ABS,
    )
