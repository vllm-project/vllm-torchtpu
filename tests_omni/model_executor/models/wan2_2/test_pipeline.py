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
"""TPU tests for Wan2.2 disaggregated diffusion stages (EG and EGD)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch_tpu._internal.utils import hardware
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.models.wan.utils import extract_wan_video
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

VIDEO_FRAMES = 5
VIDEO_HEIGHT = 256
VIDEO_WIDTH = 256
VIDEO_STEPS = 2
VIDEO_GUIDANCE = 5.0
VIDEO_SEED = 42
VIDEO_PROMPT = "A paper boat drifting down a sunlit forest stream"

DISAGG_CASES = [
    ("wan2_2_eg.yaml", ["0", "1"]),
    ("wan2_2_egd.yaml", ["0", "1", "2"]),
]


def _build_video_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        num_frames=VIDEO_FRAMES,
        num_inference_steps=VIDEO_STEPS,
        guidance_scale=VIDEO_GUIDANCE,
        seed=VIDEO_SEED,
        generator_device="cpu",
    )


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "Disaggregated diffusion stage 1 aborts on /tmp/libtpu_lockfile because "
        "OmniTpuPlatform.set_device does not set ALLOW_MULTIPLE_LIBTPU_LOAD=1"
    ),
)
@pytest.mark.parametrize(
    "deploy_config,expected_devices",
    DISAGG_CASES,
    ids=["eg", "egd"],
)
def test_disaggregated_diffusion_stages(
    deploy_config: str,
    expected_devices: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disaggregated diffusion stages (encoder / DiT / VAE).

    - TPU path: launch ``OmniRunner`` on ``tiny_wan2_1_t2v()`` with
      ``wan2_2_eg.yaml`` (2 stages) and ``wan2_2_egd.yaml`` (3 stages) over
      ``SharedMemoryConnector``.
    - Setup check: assert TPU device count, resolved stage count, and per-stage
      ``runtime_config.devices`` placement.
    - Outcome check: generate a ``(5, 256, 256, 3)`` 2-step video and assert
      bitwise parity against a single-stage ``distributed_executor_backend="mp"``
      reference.
    """
    num_stages = len(expected_devices)
    device_count = hardware.get_tpu_device_count()
    assert device_count >= num_stages, (
        f"{deploy_config}: requires {num_stages} TPU devices, found {device_count}"
    )

    monkeypatch.setenv(
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        str(tmp_path / "tpu_compile_cache"),
    )
    model_dir = tiny_wan2_1_t2v()
    video_shape = (VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 3)

    with OmniRunner(
        model_dir,
        deploy_config=deploy_config,
        stage_init_timeout=120,
        init_timeout=240,
    ) as disagg_runner:
        assert len(disagg_runner.stage_configs) == num_stages, (
            f"{deploy_config}: resolved {len(disagg_runner.stage_configs)} stages, "
            f"expected {num_stages}"
        )
        actual_devices = [s.runtime_config.devices for s in disagg_runner.stage_configs]
        assert actual_devices == expected_devices, (
            f"{deploy_config}: stage devices {actual_devices} != {expected_devices}"
        )
        disagg_outputs = list(
            disagg_runner.generate(VIDEO_PROMPT, _build_video_params())
        )
        disagg_video = extract_wan_video(disagg_outputs, video_shape)

    with OmniRunner(
        model_dir,
        distributed_executor_backend="mp",
        stage_init_timeout=120,
        init_timeout=240,
    ) as ref_runner:
        ref_outputs = list(ref_runner.generate(VIDEO_PROMPT, _build_video_params()))
        ref_video = extract_wan_video(ref_outputs, video_shape)

    np.testing.assert_array_equal(
        disagg_video,
        ref_video,
        err_msg=f"{deploy_config} video diverged from single-stage mp reference",
    )
