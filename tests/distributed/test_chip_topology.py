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

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import pytest

from vllm_torchtpu.distributed import chip_topology as topology


@pytest.fixture(autouse=True)
def clear_topology_cache():
    topology.get_chip_topology.cache_clear()
    yield
    topology.get_chip_topology.cache_clear()


def test_chip_coordinates_and_runtime_rank_order(monkeypatch):
    devices = [
        SimpleNamespace(id=i, coords=coords)
        for i, coords in [(9, (1, 0)), (5, (0, 0)), (8, (1, 0)), (4, (0, 0))]
    ]
    enumerate_devices = Mock(return_value=devices)
    monkeypatch.setattr(jax, "devices", enumerate_devices)
    result = topology.get_chip_topology()
    assert result.chips == [[4, 5], [8, 9]]
    assert result.device_ids == [9, 5, 8, 4]
    assert [result.chip_of(r) for r in range(4)] == [(1, 1), (0, 1), (1, 0), (0, 0)]
    assert [topology.hierarchical_moe_split(4, r) for r in range(4)] == [
        (2, 1, 2, 1),
        (2, 0, 2, 1),
        (2, 1, 2, 0),
        (2, 0, 2, 0),
    ]
    assert topology.get_chip_topology() is result
    enumerate_devices.assert_called_once_with()


@pytest.mark.parametrize(
    "devices",
    [
        [],
        [SimpleNamespace(id=0)],
        [
            SimpleNamespace(id=0, coords=(0,)),
            SimpleNamespace(id=1, coords=(0,)),
            SimpleNamespace(id=2, coords=(1,)),
        ],
    ],
)
def test_unusable_topology_returns_none(monkeypatch, devices):
    monkeypatch.setattr(jax, "devices", lambda: devices)
    assert topology.get_chip_topology() is None
    assert topology.hierarchical_moe_split(4, 0) is None


def test_device_enumeration_failure_returns_none(monkeypatch):
    monkeypatch.setattr(jax, "devices", Mock(side_effect=RuntimeError("no runtime")))
    assert topology.get_chip_topology() is None


@pytest.mark.parametrize(
    "chips,ids,world_size",
    [([[0], [1]], [0, 1], 2), ([[0, 1], [2, 3]], [0, 1, 2, 3], 2), ([], [], 0)],
)
def test_hierarchical_split_requires_full_multicore_world(
    monkeypatch, chips, ids, world_size
):
    monkeypatch.setattr(
        topology, "get_chip_topology", lambda: topology.ChipTopology(chips, ids)
    )
    assert topology.hierarchical_moe_split(world_size, 0) is None


@pytest.mark.parametrize("rank", [-1, 2])
def test_chip_rank_out_of_bounds(rank):
    with pytest.raises(ValueError, match="outside"):
        topology.ChipTopology([[4, 5]], [5, 4]).chip_of(rank)


def test_device_missing_from_chips():
    with pytest.raises(RuntimeError, match="device 9 is on no chip"):
        topology.ChipTopology([[4, 5]], [9]).chip_of(0)
