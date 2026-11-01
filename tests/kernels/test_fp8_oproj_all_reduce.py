# SPDX-License-Identifier: Apache-2.0
"""Static geometry of the fused row-parallel FP8 matmul + all-reduce. The kernel itself
needs a multi-chip slice (remote DMA, barrier semaphore): it is checked on 2x2x2 against
the two production kernels, bitwise, by the t5-oproj-ar instruments."""

import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as k
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

pytestmark = pytest.mark.cpu_test


def test_tile_matches_fp8_skinny_o_proj_config():
    split, tile, nbuf = fp8_skinny.TUNED[(1024, 6144)]
    assert split == "n" and k.TUNED[(1024, 6144)] == (tile, nbuf)


@pytest.mark.parametrize(
    "m,kk,n,cores,ok",
    [
        (4, 1024, 6144, 16, True),
        (16, 1024, 6144, 16, True),
        (16, 1024, 6144, 8, True),
        (17, 1024, 6144, 16, False),  # above 16 rows
        (4, 2048, 16384, 16, False),  # unmeasured shape
        (4, 1024, 6144, 1, False),
        (4, 1024, 6144, 32, False),  # 192-lane chunks
    ],
)
def test_supports(m, kk, n, cores, ok):
    assert k.supports(m, kk, n, cores) is ok


@pytest.mark.parametrize(
    "m,max_m,ok",
    [
        (32, 32, True),
        (33, 32, False),
        (48, 64, True),
        (64, 64, True),
        (65, 64, False),
        (128, 128, False),  # above the kernel's VMEM limit
    ],
)
def test_supports_raised_row_cap(m, max_m, ok):
    assert k.supports(m, 1024, 6144, 16, max_m) is ok


def test_owner_waves_cover_every_owner_once_after_its_last_column():
    waves = k.owner_waves(16, 6144, 2048)
    assert waves == [list(range(0, 5)), list(range(5, 10)), list(range(10, 16))]
    own = 6144 // 16
    for tile, owners in enumerate(waves):
        for j in owners:
            assert ((j + 1) * own - 1) // 2048 == tile
