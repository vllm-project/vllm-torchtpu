# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Unit tests for standalone distributed logic in llm_tpu_core."""

import numpy as np
import pytest

from vllm_torchtpu.llm_tpu_core.distributed.chip_topology import ChipTopology
from vllm_torchtpu.llm_tpu_core.distributed.group_provider import (
    DefaultGroupProvider,
    get_ep_group,
    get_group_provider,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    set_group_provider,
)
from vllm_torchtpu.llm_tpu_core.distributed.pallas_shapes import (
    get_global_shape,
    get_local_shape,
)
from vllm_torchtpu.llm_tpu_core.distributed.pcp_rank_order import (
    build_topology_mesh_groups,
    validate_mesh_permutation,
)


def test_chip_topology():
    chips = [[0, 1], [2, 3]]
    device_ids = [0, 1, 2, 3]
    topo = ChipTopology(chips, device_ids)

    assert topo.num_chips == 2
    assert topo.cores_per_chip == 2
    assert topo.chip_of(0) == (0, 0)
    assert topo.chip_of(1) == (0, 1)
    assert topo.chip_of(2) == (1, 0)
    assert topo.chip_of(3) == (1, 1)

    with pytest.raises(ValueError):
        topo.chip_of(4)


def test_group_provider():
    provider = get_group_provider()
    assert isinstance(provider, DefaultGroupProvider)
    assert get_tp_group() is None
    assert get_pp_group() is None
    assert get_pcp_group() is None
    assert get_ep_group() is None

    class MockProvider:
        def get_tp_group(self):
            return "mock_tp"

        def get_pp_group(self):
            return "mock_pp"

        def get_pcp_group(self):
            return "mock_pcp"

        def get_dcp_group(self):
            return "mock_dcp"

        def get_ep_group(self):
            return "mock_ep"

        def get_world_group(self):
            return "mock_world"

    set_group_provider(MockProvider())
    assert get_tp_group() == "mock_tp"
    assert get_pp_group() == "mock_pp"
    assert get_pcp_group() == "mock_pcp"
    # Reset back to default
    set_group_provider(DefaultGroupProvider())


class FakeMesh:
    def __init__(self, shape_dict):
        self.shape = shape_dict


def test_pallas_shapes():
    mesh = FakeMesh({"tp": 4, "pcp": 2})

    # 1D sharded axis
    local_shape = (16, 32)
    partition_spec = ("tp", None)
    global_shape = get_global_shape(local_shape, mesh, partition_spec)
    assert global_shape == (64, 32)
    assert get_local_shape(global_shape, mesh, partition_spec) == (16, 32)

    # Multi-axis tuple partition: ("tp", "pcp") on one dimension (factor = 8)
    local_shape = (8, 64)
    partition_spec = (("tp", "pcp"), None)
    global_shape = get_global_shape(local_shape, mesh, partition_spec)
    assert global_shape == (64, 64)
    assert get_local_shape(global_shape, mesh, partition_spec) == (8, 64)


def test_pcp_mesh_rank_ordering():
    # 3D mesh: (dp=1, pcp=4, tp=2)
    mesh = np.arange(8).reshape(1, 4, 2)
    groups = build_topology_mesh_groups(mesh, pcp_size=4, tp_size=2)

    assert "pcp" in groups
    assert "tp" in groups
    assert len(groups["pcp"]) == 2  # 2 rings of 4 ranks
    assert len(groups["tp"]) == 4  # 4 lanes of 2 ranks

    validate_mesh_permutation(groups, list(range(8)), (1, 4, 2))


def test_pp_positions_standalone():
    from vllm_torchtpu.llm_tpu_core.distributed.pp_shift import (
        get_or_create_pp_mesh,
        pp_positions,
    )

    mesh = get_or_create_pp_mesh()
    assert mesh is not None
    assert "pp" in mesh.axis_names
    positions = pp_positions()
    assert len(positions) >= 1
