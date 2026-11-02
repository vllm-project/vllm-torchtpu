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
"""Nightly build integration and golden validation tests for Wan 2.1/2.2 on TPU v7x.

Full model family integration and numerical correctness tests for nightly builds:
1. Matrix validation across Wan model family:
   - Wan2.1 Text-to-Video (T2V 14B)
   - Wan2.1 VACE (1.3B)
   - Wan2.1 Image-to-Video (I2V 14B 480P & 720P)
   - Wan2.2 Text-Image-to-Video (TI2V 5B)
   - Wan2.2 Image-to-Video MoE (I2V-A14B)
2. Obtains and verifies golden answers and structural metric signatures:
   - MAE, SSIM, and mean shift relative to CPU fp32 golden
   - High-frequency energy and motion ratios
"""

import pytest

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Wan tests")

from tests_omni.conftest import OmniRunner
from tests_omni.models.wan.utils import (
    WAN2_1_I2V_14B_480P,
    WAN2_1_I2V_14B_720P,
    WAN2_1_T2V_14B,
    WAN2_1_VACE_1_3B,
    WAN2_2_I2V_A14B_MOE,
    WAN2_2_TI2V_5B,
    WanTestConfig,
    assert_wan_golden_parity,
    build_wan_golden_request,
    extract_wan_video,
)

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
]

TEST_CONFIGS = [
    WAN2_1_T2V_14B,
    WAN2_1_VACE_1_3B,
    WAN2_1_I2V_14B_480P,
    WAN2_1_I2V_14B_720P,
    WAN2_2_TI2V_5B,
    WAN2_2_I2V_A14B_MOE,
]


@pytest.mark.parametrize(
    "omni_runner,config",
    [pytest.param(c.path, c, id=c.config_id) for c in TEST_CONFIGS],
    indirect=["omni_runner"],
)
def test_wan_full_model_matrix_and_golden_answers_tpu(
    omni_runner: OmniRunner, config: WanTestConfig
):
    """Run functional and numerical sanity tests across Wan model variants on
    TPU v7x.

    Verifies:
    1. Model initialization and TPU engine execution.
    2. Matches the CPU fp32 golden video within the per-config limits.
    3. High-frequency energy and motion ratios.
    """
    inputs, sampling_params = build_wan_golden_request(config)
    outputs = list(omni_runner.generate(inputs, sampling_params))
    video = extract_wan_video(
        outputs, (config.num_frames, config.height, config.width, 3)
    )
    assert_wan_golden_parity(video, config)
