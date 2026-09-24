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
"""Host checks for choosing asymmetric weight buffers within VMEM."""

import dataclasses

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import host

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "hidden,inter,weight_format,rhs_qb,expected,expected_mib",
    [
        (4096, 1024, host.WeightFormat.FP8, 512, (3, 2), 40.390625),
        (8192, 2048, host.WeightFormat.FP4, 64, (2, 1), 60.765625),
    ],
)
def test_model_geometry_selects_buffers_that_fit(
    hidden, inter, weight_format, rhs_qb, expected, expected_mib
):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    shape = {
        "g_local": 64,
        "capacity": 128,
        "hidden": hidden,
        "inter": inter,
        "weight_format": weight_format,
        "rhs_qb": rhs_qb,
    }
    plan = host.select_weight_buffers(**shape, info=info)
    assert (plan.w1_nbuf, plan.w2_nbuf) == expected
    assert plan.vmem_bytes == int(expected_mib * 2**20)
    arrays = {
        name: dims
        for name, dims, _ in host.vmem_scratch_arrays(
            **shape, nbuf=plan.w1_nbuf, w2_nbuf=plan.w2_nbuf
        )
    }
    assert arrays["w1_vm"][0] == expected[0]
    assert arrays["w2_vm"][0] == expected[1]
    scale_slots = expected[0] if weight_format == host.WeightFormat.FP4 else 64
    assert arrays["w1s_vm"][0] == arrays["w2s_vm"][0] == scale_slots


def test_budget_boundary_falls_back_and_rejects_when_minimum_does_not_fit(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    shape = {"g_local": 64, "capacity": 128, "hidden": 4096, "inter": 1024}
    monkeypatch.setattr(host, "VMEM_FRACTION", 1.0)
    full = host.vmem_estimate_bytes(**shape, nbuf=3, w2_nbuf=2, info=info)
    minimum = host.vmem_estimate_bytes(**shape, nbuf=2, w2_nbuf=1, info=info)
    exact = dataclasses.replace(info, vmem_capacity_bytes=full)
    tight = dataclasses.replace(info, vmem_capacity_bytes=full - 1)
    exhausted = dataclasses.replace(info, vmem_capacity_bytes=minimum - 1)
    assert host.select_weight_buffers(**shape, info=exact).w1_nbuf == 3
    assert host.select_weight_buffers(**shape, info=tight).w1_nbuf == 2
    with pytest.raises(ValueError, match="minimum W1/W2=2/1"):
        host.select_weight_buffers(**shape, info=exhausted)
