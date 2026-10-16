# SPDX-License-Identifier: Apache-2.0
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
   - Temporal consistency ($4k+1$ frames)
3. All executions run in-memory without disk leaks.
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
    run_wan_golden_test,
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
    2. Correct video temporal frame structure (4k + 1).
    3. Correct spatial resolution (H x W divisible by VAE scale factor 16).
    4. Matches the CPU fp32 golden video within the per-config limits.
    5. High-frequency energy and motion ratios.
    """
    run_wan_golden_test(omni_runner, config)
