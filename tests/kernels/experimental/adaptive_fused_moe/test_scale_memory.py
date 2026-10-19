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
"""FP4 scale scratch is bounded by W1 pipeline depth, not expert count."""

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import host


@pytest.mark.parametrize("experts", [8, 64, 256])
@pytest.mark.parametrize("w1_nbuf,w2_nbuf,expected_mib", [(2, 1, 6), (3, 2, 9)])
def test_fp4_scale_memory_is_bounded_by_pipeline_depth(
    experts, w1_nbuf, w2_nbuf, expected_mib
):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    shape = dict(
        g_local=experts,
        capacity=128,
        hidden=8192,
        inter=2048,
        weight_format="fp4",
        rhs_qb=64,
        nbuf=w1_nbuf,
    )
    streamed = host.vmem_scratch_arrays(**shape, w2_nbuf=w2_nbuf)

    def scale_bytes(arrays):
        return sum(
            host.array_vmem_bytes(dims, dtype, info)
            for name, dims, dtype in arrays
            if name in ("w1s_vm", "w2s_vm")
        )

    # Both tables follow W1's ring, including W2 scales. Assert the actual
    # padded allocation against a fixed budget so restoring expert-resident
    # scales or incorrectly using W2's shorter ring fails this regression.
    assert scale_bytes(streamed) == expected_mib * 2**20
