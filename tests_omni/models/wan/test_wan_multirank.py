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
"""Nightly multi-rank golden validation tests for Wan2.1 T2V on TPU v7x."""

import pytest
from torch_tpu._internal.utils import hardware

pytest.importorskip("vllm_omni", reason="vllm_omni is required for Wan tests")

from vllm_omni.diffusion.data import DiffusionParallelConfig

from tests_omni.conftest import OmniRunner
from tests_omni.models.wan.utils import (
    WAN2_1_T2V_1_3B,
    WAN2_1_T2V_14B,
    WanTestConfig,
    assert_wan_golden_parity,
    build_wan_golden_request,
    extract_wan_video,
)

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.omni,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.multichip,
]

RING_UNSUPPORTED = (
    "Ring attention hangs on TPU: the ring K/V exchange never completes in "
    "torch_tpu's P2P send (ProcessGroupTpu.experimental_send)"
)

# Parallel degrees are passed as flat extra kwargs into OmniRunner
CASES = [
    (WAN2_1_T2V_1_3B, {"cfg_parallel_size": 2}, "cfg2", ()),
    (WAN2_1_T2V_1_3B, {"ulysses_degree": 2}, "u2", ()),
    (WAN2_1_T2V_1_3B, {"tensor_parallel_size": 2}, "tp2", ()),
    (WAN2_1_T2V_14B, {"cfg_parallel_size": 2, "ulysses_degree": 4}, "cfg2xu4", ()),
    (WAN2_1_T2V_14B, {"ulysses_degree": 8}, "u8", ()),
    (
        WAN2_1_T2V_1_3B,
        {"ring_degree": 2},
        "ring2",
        pytest.mark.skip(reason=RING_UNSUPPORTED),
    ),
]


@pytest.fixture(autouse=True)
def _check_tpu_devices(degrees: dict[str, int]) -> None:
    world_size = DiffusionParallelConfig(**degrees).world_size
    # PCI scan: counts TPU devices without starting the TPU runtime here.
    device_count = hardware.get_tpu_device_count()
    assert device_count >= world_size, (
        f"Test requires {world_size} TPU devices, found {device_count}"
    )


@pytest.mark.parametrize(
    "omni_runner,config,degrees",
    [
        pytest.param(
            (c.path, None, degrees),
            c,
            degrees,
            id=f"{c.config_id}_{case_id}",
            marks=marks,
        )
        for c, degrees, case_id, marks in CASES
    ],
    indirect=["omni_runner"],
)
def test_wan_multirank_golden_answers_tpu(
    omni_runner: OmniRunner,
    config: WanTestConfig,
    degrees: dict[str, int],
):
    """Verify multi-rank Wan2.1 T2V output against the CPU fp32 golden video."""
    stage_parallel = omni_runner.stage_configs[0].parallel_config
    expected = {
        "cfg_parallel_size": 1,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "tensor_parallel_size": 1,
        **degrees,
    }
    actual = {name: getattr(stage_parallel, name) for name in expected}
    assert actual == expected, (
        f"Engine parallel degrees {actual} != expected {expected}"
    )
    inputs, sampling_params = build_wan_golden_request(config)
    outputs = list(omni_runner.generate(inputs, sampling_params))
    video = extract_wan_video(
        outputs, (config.num_frames, config.height, config.width, 3)
    )
    assert_wan_golden_parity(video, config)
