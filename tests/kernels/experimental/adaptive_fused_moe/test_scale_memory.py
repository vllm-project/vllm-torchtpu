# SPDX-License-Identifier: Apache-2.0
"""FP4 scale scratch is bounded by W1 pipeline depth, not expert count."""

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import host

pytestmark = pytest.mark.cpu_test


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
