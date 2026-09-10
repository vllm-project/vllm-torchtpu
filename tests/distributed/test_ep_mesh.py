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
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed import ep_mesh as ep


@pytest.fixture(autouse=True)
def mesh_cache(monkeypatch):
    monkeypatch.setattr(ep, "_MESH_CACHE", {})


@pytest.mark.parametrize(
    "group",
    [None, SimpleNamespace(world_size=1),
     SimpleNamespace(world_size=4)])
def test_ep_group_requires_multiple_ranks(monkeypatch, group):
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: group)
    assert ep.get_ep_group() is (group
                                 if group and group.world_size > 1 else None)


@pytest.mark.parametrize("error",
                         [AssertionError, AttributeError, RuntimeError])
def test_uninitialized_ep_group(monkeypatch, error):
    monkeypatch.setattr(parallel_state, "get_ep_group",
                        Mock(side_effect=error))
    assert ep.get_ep_group() is None


def test_disabled_ep_helpers(monkeypatch):
    monkeypatch.setattr(ep, "get_ep_group", lambda: None)
    assert ep.ep_device_ids() is None
    assert ep.ep_rank_order() is None
    assert ep.ep_mesh_index() is None
    assert ep.build_ep_mesh() is None


@pytest.fixture
def group(monkeypatch):
    group = SimpleNamespace(world_size=4, ranks=[2, 4, 6, 8])
    monkeypatch.setattr(ep, "get_ep_group", lambda: group)
    monkeypatch.setattr(ep, "_get_current_global_rank", lambda: 6)
    monkeypatch.setattr(ep, "_get_tpu_global_device_id", lambda: 10)
    return group


def test_ep_rank_mapping_and_mesh_index(monkeypatch, group):
    gather = Mock(return_value={8: 30, 2: 40, 6: 10, 4: 20})
    monkeypatch.setattr(ep, "_collect_rank_to_device_id", gather)
    assert ep.ep_device_ids() == (40, 20, 10, 30)
    gather.assert_called_once_with(group, 6, 10)
    assert ep.ep_rank_order() == (2, 1, 3, 0)
    assert ep.ep_mesh_index() == 0


def test_missing_rank_mapping(monkeypatch, group):
    monkeypatch.setattr(ep, "_collect_rank_to_device_id", lambda *args: {
        2: 40,
        4: 20,
        8: 30
    })
    with pytest.raises(RuntimeError, match=r"missing=\[6\]"):
        ep.ep_device_ids()


def test_duplicate_device_mapping(monkeypatch, group):
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 20, 30))
    with pytest.raises(RuntimeError, match="same TPU global device id"):
        ep.ep_rank_order()


def test_mesh_orders_devices_and_caches_by_axis_and_ids(monkeypatch):
    devices = [SimpleNamespace(id=i) for i in [40, 20, 10, 30]]
    monkeypatch.setattr(jax, "devices", Mock(return_value=devices))
    mesh = Mock(side_effect=lambda *args, **kwargs: object())
    monkeypatch.setattr(jax.sharding, "Mesh", mesh)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 10, 30))
    first = ep.build_ep_mesh()
    assert [d.id for d in mesh.call_args.args[0]] == [10, 20, 30, 40]
    assert mesh.call_args.kwargs == {"axis_names": ("d", )}
    assert ep.build_ep_mesh() is first
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (10, 20, 30, 40))
    assert ep.build_ep_mesh() is first
    assert ep.build_ep_mesh("experts") is not first
    assert mesh.call_args.kwargs == {"axis_names": ("experts", )}
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (10, 20))
    assert ep.build_ep_mesh() is not first
    assert mesh.call_count == 3


def test_mesh_rejects_invisible_device(monkeypatch):
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (10, 20))
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(id=10)])
    mesh = Mock()
    monkeypatch.setattr(jax.sharding, "Mesh", mesh)
    with pytest.raises(RuntimeError, match=r"missing=\[20\]"):
        ep.build_ep_mesh()
    mesh.assert_not_called()
    assert ep._MESH_CACHE == {}


def test_missing_native_ep_symbol(monkeypatch):
    monkeypatch.delattr(parallel_state, "get_ep_group")
    assert ep.get_ep_group() is None
