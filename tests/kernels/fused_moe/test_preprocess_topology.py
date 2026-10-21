# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_torchtpu.kernels.fused_moe.v2.preprocessing import (
    supports_activation_gather,
    topology_for_coordinates,
)


@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4, 5, 6, 7), (7, 5, 3, 1, 6, 4, 2, 0)])
def test_physical_neighbors_follow_mesh_order(order):
    coordinates = tuple((2 + (i >> 1 & 1), 4 + (i >> 2 & 1), 3, i & 1) for i in order)
    topology = topology_for_coordinates(coordinates)
    assert topology is not None
    for axis, mask in zip((0, 1, 3), topology.masks):
        for rank, coordinate in enumerate(coordinates):
            peer = coordinates[rank ^ mask]
            assert abs(peer[axis] - coordinate[axis]) == 1
            assert all(peer[a] == coordinate[a] for a in range(4) if a != axis)


pytestmark = pytest.mark.cpu_test


def test_non_affine_rank_order_falls_back():
    order = (1, 0, 2, 3, 4, 5, 6, 7)
    assert (
        topology_for_coordinates(
            tuple((i >> 1 & 1, i >> 2 & 1, 0, i & 1) for i in order)
        )
        is None
    )


@pytest.mark.parametrize(
    "size,supported",
    [
        (0, False),
        (288 * 5120, True),
        (8193, False),
        (8 * 1024 * 1024, True),
        (16 * 1024 * 1024, False),
    ],
)
def test_activation_scratch_and_dma_alignment(size, supported):
    assert supports_activation_gather(size) == supported
