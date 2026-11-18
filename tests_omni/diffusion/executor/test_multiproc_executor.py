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
"""Multi-rank TPU tests for diffusion data parallelism on MultiprocDiffusionExecutor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch_tpu._internal.utils import hardware
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_qwen_image

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

DP_SIZE = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_STEPS = 2
REQUEST_SEED = 42
PROMPTS = [
    "a red wooden cube resting on a table",
    "a blue glass sphere in a green meadow",
]
DP_MAX_REL_L2 = 5.0e-3
DP_MAX_ABS = 1.0e-2


def _extract_prompt_images(outputs: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Order ``OmniRequestOutput`` items by prompt index and convert RGB pixels."""
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
    ordered = sorted(outputs, key=lambda item: int(item.request_id.split("_", 1)[0]))
    indices = [int(item.request_id.split("_", 1)[0]) for item in ordered]
    assert indices == [0, 1], f"Unexpected request_id prompt indices: {indices}"

    tensors: list[torch.Tensor] = []
    for idx, out in enumerate(ordered):
        img = out.images[0]
        assert img.size == (IMAGE_WIDTH, IMAGE_HEIGHT), (
            f"prompt {idx}: image size {img.size} != {(IMAGE_WIDTH, IMAGE_HEIGHT)}"
        )
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(rgb))
    return tensors[0], tensors[1]


def test_diffusion_data_parallelism() -> None:
    """Diffusion Data Parallelism: 2-worker MultiprocDiffusionExecutor on TPU.

    - TPU path: run ``OmniRunner`` on ``tiny_qwen_image()`` with
      ``data_parallel_size=2`` and ``enforce_eager=True`` across 2 spawned
      ``DiffusionWorker`` processes.
    - Setup check: assert ``hardware.get_tpu_device_count() >= 2`` and stage
      parallel ``data_parallel_size == 2`` and ``world_size == 2``.
    - Outcome check: verify distinct prompts produce distinct images and compare
      both outputs against a spawned ``data_parallel_size=1`` ``mp`` baseline via
      ``assert_parity``.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= DP_SIZE, (
        f"Test requires {DP_SIZE} TPU devices, found {device_count}"
    )

    model_dir = tiny_qwen_image()
    sampling_params = OmniDiffusionSamplingParams(
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=1.0,
        seed=REQUEST_SEED,
    )

    with OmniRunner(
        model_dir,
        data_parallel_size=DP_SIZE,
        enforce_eager=True,
    ) as dp_runner:
        dp_parallel = dp_runner.stage_configs[0].parallel_config
        assert dp_parallel.data_parallel_size == DP_SIZE, (
            f"Expected data_parallel_size={DP_SIZE}, got "
            f"{dp_parallel.data_parallel_size}"
        )
        assert dp_parallel.world_size == DP_SIZE, (
            f"Expected stage world_size={DP_SIZE}, got {dp_parallel.world_size}"
        )
        dp_outputs = list(dp_runner.generate(PROMPTS, sampling_params))

    dp_img0, dp_img1 = _extract_prompt_images(dp_outputs)

    with OmniRunner(
        model_dir,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        enforce_eager=True,
    ) as ref_runner:
        ref_parallel = ref_runner.stage_configs[0].parallel_config
        assert ref_parallel.data_parallel_size == 1, (
            f"Expected baseline data_parallel_size=1, got "
            f"{ref_parallel.data_parallel_size}"
        )
        assert ref_parallel.world_size == 1, (
            f"Expected baseline world_size=1, got {ref_parallel.world_size}"
        )
        ref_outputs = list(ref_runner.generate(PROMPTS, sampling_params))

    ref_img0, ref_img1 = _extract_prompt_images(ref_outputs)

    prompt_separation = float((dp_img0 - dp_img1).abs().max())
    assert prompt_separation > 0.0, (
        f"Distinct prompts produced identical images (max diff={prompt_separation})"
    )

    assert_parity(
        dp_img0,
        ref_img0,
        name="dp2_prompt_0",
        max_rel_l2=DP_MAX_REL_L2,
        max_abs=DP_MAX_ABS,
    )
    assert_parity(
        dp_img1,
        ref_img1,
        name="dp2_prompt_1",
        max_rel_l2=DP_MAX_REL_L2,
        max_abs=DP_MAX_ABS,
    )
