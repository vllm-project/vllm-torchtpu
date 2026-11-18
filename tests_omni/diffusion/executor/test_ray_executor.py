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
"""TPU engine test for RayDiffusionExecutor (distributed_executor_backend='ray')."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
import ray
import torch
from PIL import Image
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor import ray_executor
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.ray_executor import RayDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests.tpu_release import wait_for_tpu_release
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

PLACEMENT_GROUP_WAIT_TIMEOUT_S = 2.0
IMAGE_SIZE = 256
STEPS = 2
SEED = 42
MAX_REL_L2 = 1e-3
MAX_ABS = 1e-2
PROMPT = "a geometric cube on a plain table"


def _make_request(request_id: str) -> OmniDiffusionRequest:
    params = OmniDiffusionSamplingParams(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=STEPS,
        guidance_scale=0.0,
        seed=SEED,
        generator_device="cpu",
    )
    return OmniDiffusionRequest(
        prompt=PROMPT,
        sampling_params=params,
        request_id=request_id,
    )


def _extract_image_tensor(
    output: DiffusionOutput,
    post_process_func: Callable[..., Any] | None,
    label: str,
) -> torch.Tensor:
    assert output.error is None, f"{label}: DiffusionOutput error={output.error!r}"
    assert output.output is not None, f"{label}: DiffusionOutput.output is None"
    processed = (
        post_process_func(output.output)
        if post_process_func is not None
        else output.output
    )
    if isinstance(processed, list):
        assert len(processed) == 1, f"{label}: image count {len(processed)} != 1"
        processed = processed[0]
    if isinstance(processed, Image.Image):
        arr = np.asarray(processed.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)
    elif isinstance(processed, torch.Tensor):
        tensor = processed.detach().cpu().float()
    else:
        arr = np.asarray(cast(Any, processed), dtype=np.float32)
        tensor = torch.from_numpy(arr)
    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert tuple(tensor.shape) == expected_shape, (
        f"{label}: image tensor shape {tuple(tensor.shape)} != {expected_shape}"
    )
    return tensor


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "RayDiffusionExecutor requests {'GPU': 1} placement-group bundles, "
        "which cannot be reserved on GPU-less TPU Ray nodes"
    ),
)
def test_ray_diffusion_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-node serving with Ray.

    - TPU path: bounds ``ray_executor._PLACEMENT_GROUP_WAIT_TIMEOUT_S`` to 2.0 s,
      initializes ``DiffusionEngine`` with ``distributed_executor_backend="ray"``
      on ``tiny_flux_schnell()``, runs a 2-step 256x256 request on the Ray TPU
      worker, and compares against ``distributed_executor_backend="mp"``.
    - Setup check: ``engine.executor`` is a ``RayDiffusionExecutor`` with a live
      placement group and 1 healthy worker actor.
    - Outcome check: Ray worker image output matches the ``mp`` executor
      reference within relative L2 <= 1e-3 and max abs <= 1e-2.
    """
    monkeypatch.setattr(
        ray_executor,
        "_PLACEMENT_GROUP_WAIT_TIMEOUT_S",
        PLACEMENT_GROUP_WAIT_TIMEOUT_S,
    )
    model_dir = tiny_flux_schnell()
    wait_for_tpu_release()

    ray_engine: DiffusionEngine | None = None
    try:
        ray_config = OmniDiffusionConfig(
            model=model_dir,
            distributed_executor_backend="ray",
            num_gpus=1,
        )
        ray_engine = DiffusionEngine(ray_config)
        assert isinstance(ray_engine.executor, RayDiffusionExecutor), (
            f"Expected RayDiffusionExecutor, got {type(ray_engine.executor).__name__}"
        )
        assert ray_engine.executor._resources.placement_group is not None, (
            "RayDiffusionExecutor placement_group is None"
        )
        assert len(ray_engine.executor.workers) == 1, (
            f"RayDiffusionExecutor worker count {len(ray_engine.executor.workers)} != 1"
        )
        ray_engine.executor.check_health()

        ray_out = ray_engine.add_req_and_wait_for_response(_make_request("req-ray"))
        ray_post_fn = ray_engine.post_process_func
    finally:
        if ray_engine is not None:
            ray_engine.close()
        if ray.is_initialized():
            ray.shutdown()
        wait_for_tpu_release()

    mp_engine = DiffusionEngine(
        OmniDiffusionConfig(
            model=model_dir,
            distributed_executor_backend="mp",
            num_gpus=1,
        )
    )
    try:
        assert isinstance(mp_engine.executor, MultiprocDiffusionExecutor), (
            "Expected MultiprocDiffusionExecutor, got "
            f"{type(mp_engine.executor).__name__}"
        )
        mp_out = mp_engine.add_req_and_wait_for_response(_make_request("req-mp"))
        mp_post_fn = mp_engine.post_process_func
    finally:
        mp_engine.close()
        wait_for_tpu_release()

    ray_tensor = _extract_image_tensor(ray_out, ray_post_fn, "ray_executor")
    mp_tensor = _extract_image_tensor(mp_out, mp_post_fn, "mp_executor")
    assert_parity(
        ray_tensor,
        mp_tensor,
        name="ray_vs_mp_flux",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
