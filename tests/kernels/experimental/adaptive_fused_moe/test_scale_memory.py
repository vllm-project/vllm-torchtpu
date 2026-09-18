# SPDX-License-Identifier: Apache-2.0
"""Compare resident v2 FP4 scales with the experimental scale pipeline."""

import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import host
from vllm_torchtpu.kernels.fused_moe.v2 import host as original_host

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("experts", [8, 64, 256])
def test_fp4_scale_memory_is_bounded_by_pipeline_depth(experts):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    shape = dict(
        g_local=experts,
        capacity=128,
        hidden=8192,
        inter=2048,
        weight_format="fp4",
        rhs_qb=64,
        nbuf=2,
    )
    resident = original_host.vmem_scratch_arrays(**shape)
    streamed = host.vmem_scratch_arrays(**shape, w2_nbuf=1)

    def scale_bytes(arrays):
        return sum(
            host.array_vmem_bytes(dims, dtype, info)
            for name, dims, dtype in arrays if name in ("w1s_vm", "w2s_vm"))

    old_bytes, new_bytes = scale_bytes(resident), scale_bytes(streamed)
    assert old_bytes == experts * 3 * 2**20
    assert new_bytes == 6 * 2**20
    print(f"local_experts={experts} original_scale_MiB={old_bytes / 2**20} "
          f"adaptive_scale_MiB={new_bytes / 2**20}")
