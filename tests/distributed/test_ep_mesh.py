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
    monkeypatch.setattr(ep, "_TOKEN_GROUP_CACHE", {})


@pytest.mark.parametrize(
    "group", [None, SimpleNamespace(world_size=1), SimpleNamespace(world_size=4)]
)
def test_ep_group_requires_multiple_ranks(monkeypatch, group):
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: group)
    assert ep.get_ep_group() is (group if group and group.world_size > 1 else None)


@pytest.mark.parametrize("error", [AssertionError, AttributeError, RuntimeError])
def test_uninitialized_ep_group(monkeypatch, error):
    monkeypatch.setattr(parallel_state, "get_ep_group", Mock(side_effect=error))
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
    """Device ids still describe the group; they no longer order the mesh.

    Before google-pytorch/torch_tpu#3522 the mesh had to be built in ascending
    device-id order, so mesh index i held some other EP rank's experts and
    this rank's index had to be reconstructed from where its device sorted.
    Since #3522 the mesh is rank-ordered: the permutation is the identity and
    the index is just this worker's position in the group.
    """
    gather = Mock(return_value={8: 30, 2: 40, 6: 10, 4: 20})
    monkeypatch.setattr(ep, "_collect_rank_to_device_id", gather)
    assert ep.ep_device_ids() == (40, 20, 10, 30)
    gather.assert_called_once_with(group, 6, 10)
    assert ep.ep_rank_order() == (0, 1, 2, 3)
    assert ep.ep_mesh_index() == 2


def test_missing_rank_mapping(monkeypatch, group):
    monkeypatch.setattr(
        ep, "_collect_rank_to_device_id", lambda *args: {2: 40, 4: 20, 8: 30}
    )
    with pytest.raises(RuntimeError, match=r"missing=\[6\]"):
        ep.ep_device_ids()


def test_rank_order_ignores_device_ids(monkeypatch, group):
    """It used to sort by device id, so a duplicate id was ambiguous and had
    to raise. The order is the identity now, so device ids -- duplicated or
    not -- cannot reach it."""
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 20, 30))
    assert ep.ep_rank_order() == (0, 1, 2, 3)


def test_mesh_comes_from_pallas_and_caches_by_axis_and_ids(monkeypatch):
    """The mesh is Torchtpu's, and jax.devices() is never consulted.

    get_pallas_mesh builds abstract rank-numbered devices out of the process
    group, so a worker no longer needs JAX to expose every peer -- which is
    also why the old "device not visible" rejection has no counterpart here.
    """
    import torch.distributed as dist
    from torch_tpu._internal import pallas as pallas_module

    monkeypatch.setattr(
        jax,
        "devices",
        Mock(side_effect=AssertionError("build_ep_mesh must not read jax.devices()")),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 4)
    made = Mock(side_effect=lambda **kwargs: object())
    monkeypatch.setattr(pallas_module, "get_pallas_mesh", made)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 10, 30))

    first = ep.build_ep_mesh()
    assert made.call_args.kwargs == {"axis_names": ("d",)}
    assert ep.build_ep_mesh() is first
    # Same devices in a different order is the same mesh: the key sorts them.
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (10, 20, 30, 40))
    assert ep.build_ep_mesh() is first
    assert ep.build_ep_mesh("experts") is not first
    assert made.call_args.kwargs == {"axis_names": ("experts",)}
    assert made.call_count == 2


def test_mesh_rejects_a_group_smaller_than_the_world(monkeypatch):
    """get_pallas_mesh spans the whole torch.distributed world and cannot
    describe a sub-group, so a smaller EP group has to fail rather than build
    a mesh of the wrong size."""
    import torch.distributed as dist
    from torch_tpu._internal import pallas as pallas_module

    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 8)
    made = Mock()
    monkeypatch.setattr(pallas_module, "get_pallas_mesh", made)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (10, 20))
    with pytest.raises(NotImplementedError, match="spans 2 devices"):
        ep.build_ep_mesh()
    made.assert_not_called()
    assert ep._MESH_CACHE == {}


def test_missing_native_ep_symbol(monkeypatch):
    monkeypatch.delattr(parallel_state, "get_ep_group")
    assert ep.get_ep_group() is None


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_token_groups_use_native_membership_and_mesh_order(
    monkeypatch, group, sequence_parallel
):
    group.cpu_group = object()
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 10, 30))
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(ranks=[6, 2])
    )
    reports = (
        [(r, (r,)) for r in group.ranks]
        if sequence_parallel
        else [(2, (6, 2)), (4, (8, 4)), (6, (6, 2)), (8, (8, 4))]
    )

    def gather(output, value, *, group):
        assert group is not None
        assert value == (6, (6,) if sequence_parallel else (6, 2))
        output[:] = reports

    collect = Mock(side_effect=gather)
    monkeypatch.setattr(ep.torch.distributed, "all_gather_object", collect)
    expected = ((0,), (1,), (2,), (3,)) if sequence_parallel else ((0, 3), (2, 1))
    assert (
        ep.ep_token_replica_groups(is_sequence_parallel=sequence_parallel) == expected
    )
    assert (
        ep.ep_token_replica_groups(is_sequence_parallel=sequence_parallel) == expected
    )
    assert collect.call_count == 1


@pytest.mark.parametrize(
    "reports",
    [
        [None] * 4,
        [(2, (2, 6)), (4, (4, 8)), (6, (6, 2)), (8, (4, 8))],
        [(2, (2, 6)), (4, (4, 8)), (6, (2, 6)), (8, (8, 10))],
        [(2, (2,)), (4, (4, 8)), (6, (6,)), (8, (4, 8))],
    ],
)
def test_token_groups_reject_inconsistent_membership(monkeypatch, group, reports):
    group.cpu_group = object()
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 10, 30))
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(ranks=[6, 2])
    )

    def gather(output, value, *, group):
        output[:] = reports

    monkeypatch.setattr(ep.torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="replica"):
        ep.ep_token_replica_groups()
