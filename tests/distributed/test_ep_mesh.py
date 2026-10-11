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
import numpy as np
import pytest
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed import ep_mesh as ep


@pytest.fixture(autouse=True)
def mesh_cache(monkeypatch):
    monkeypatch.setattr(ep, "_MESH_CACHE", {})
    monkeypatch.setattr(ep, "_TOKEN_GROUP_CACHE", {})
    monkeypatch.setattr(ep, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(ep, "get_dcp_world_size", lambda: 1)


# Each CP axis whose rank-ordered mesh pins the EP mesh to rank order, with a
# size that turns it on. DCP=4 is the case that conflicts on tpu7x: TP4/DCP4
# DCP groups are devices [0, 1, 4, 5] and [6, 7, 2, 3], not contiguous tiles
# of the sorted EP mesh [0..7].
_RANK_ORDERED_CP = pytest.mark.parametrize(
    ("cp_size_fn", "cp_size"),
    [("get_pcp_world_size", 2), ("get_dcp_world_size", 4)],
    ids=["pcp", "dcp"],
)


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
    """The mesh is in ascending device-id order, so mesh index i holds the
    EP rank whose device sorts i-th, and this rank's index is where its own
    device sorts."""
    gather = Mock(return_value={8: 30, 2: 40, 6: 10, 4: 20})
    monkeypatch.setattr(ep, "_collect_rank_to_device_id", gather)
    assert ep.ep_device_ids() == (40, 20, 10, 30)
    gather.assert_called_once_with(group, 6, 10)
    # Devices 10, 20, 30, 40 belong to EP ranks 2, 1, 3, 0.
    assert ep.ep_rank_order() == (2, 1, 3, 0)
    assert ep.ep_mesh_index() == 0


@_RANK_ORDERED_CP
def test_cp_keeps_rank_order(monkeypatch, group, cp_size_fn, cp_size):
    """With PCP or DCP > 1 the mesh stays in rank order: both CP meshes are
    rank-ordered (the PCP kernels index token shards and state by PCP rank),
    and torch_tpu rejects two Pallas kernels with different DeviceAssignments
    in one compiled module. So mesh index i is EP rank i and this rank's index
    is its EP rank, even though the same device ids reorder the mesh with both
    off."""
    monkeypatch.setattr(ep, cp_size_fn, lambda: cp_size)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 10, 30))
    assert ep.ep_rank_order() == (0, 1, 2, 3)
    # Global rank 6 is EP rank 2 of [2, 4, 6, 8].
    assert ep.ep_mesh_index() == 2


def test_missing_rank_mapping(monkeypatch, group):
    monkeypatch.setattr(
        ep, "_collect_rank_to_device_id", lambda *args: {2: 40, 4: 20, 8: 30}
    )
    with pytest.raises(RuntimeError, match=r"missing=\[6\]"):
        ep.ep_device_ids()


def test_rank_order_rejects_duplicate_device_ids(monkeypatch, group):
    """Sorting by device id is ambiguous when two ranks report one device."""
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 20, 30))
    with pytest.raises(RuntimeError, match="same TPU global device id"):
        ep.ep_rank_order()


def _pallas_devices(rank_device_ids):
    return [
        SimpleNamespace(id=device_id, process_index=rank)
        for rank, device_id in enumerate(rank_device_ids)
    ]


def _fake_mesh(devices, axis_names):
    return SimpleNamespace(
        devices=np.asarray(devices, dtype=object),
        axis_names=axis_names,
    )


@pytest.mark.tpu_test
def test_mesh_sorts_pallas_devices_by_id_and_caches_by_axis_and_ids(monkeypatch):
    """The devices are Torchtpu's (jax.devices() is never consulted), but the
    mesh orders them by device id rather than keeping get_pallas_mesh's rank
    order: that order is the kernel's DeviceAssignment, and the fused MoE's
    all-gathers are slower over the rank-ordered one."""
    import torch.distributed as dist
    from torch_tpu._internal import pallas as pallas_module

    monkeypatch.setattr(
        jax,
        "devices",
        Mock(side_effect=AssertionError("build_ep_mesh must not read jax.devices()")),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 4)
    rank_device_ids = (40, 20, 10, 30)
    made = Mock(
        side_effect=lambda axis_names: _fake_mesh(
            _pallas_devices(rank_device_ids), axis_names
        )
    )
    monkeypatch.setattr(pallas_module, "get_pallas_mesh", made)
    monkeypatch.setattr(jax.sharding, "Mesh", _fake_mesh)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: rank_device_ids)

    first = ep.build_ep_mesh()
    assert made.call_args.kwargs == {"axis_names": ("d",)}
    assert first.axis_names == ("d",)
    assert [d.id for d in first.devices.flat] == [10, 20, 30, 40]
    # Each device keeps the rank Torchtpu gave it; only the order changes.
    assert [d.process_index for d in first.devices.flat] == [2, 1, 3, 0]
    assert ep.build_ep_mesh() is first
    # Same devices in a different rank order is the same mesh: the key is the
    # mesh's device order, and both sort to it.
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (10, 20, 30, 40))
    assert ep.build_ep_mesh() is first
    assert ep.build_ep_mesh("experts") is not first
    assert made.call_args.kwargs == {"axis_names": ("experts",)}
    assert made.call_count == 2


@pytest.mark.tpu_test
@_RANK_ORDERED_CP
def test_mesh_keeps_pallas_rank_order_under_cp(monkeypatch, cp_size_fn, cp_size):
    """With PCP or DCP > 1 the EP mesh must share the CP mesh's
    DeviceAssignment, which is rank order -- on tpu7x, devices
    [0, 1, 4, 5, 6, 7, 2, 3]. Sorting them here is what failed to compile with
    "Disjoint or conflicting device assignments" on PCP + fused EP."""
    import torch.distributed as dist
    from torch_tpu._internal import pallas as pallas_module

    monkeypatch.setattr(ep, cp_size_fn, lambda: cp_size)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 8)
    rank_device_ids = (0, 1, 4, 5, 6, 7, 2, 3)
    monkeypatch.setattr(
        pallas_module,
        "get_pallas_mesh",
        lambda axis_names: _fake_mesh(_pallas_devices(rank_device_ids), axis_names),
    )
    monkeypatch.setattr(jax.sharding, "Mesh", _fake_mesh)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: rank_device_ids)

    mesh = ep.build_ep_mesh()
    assert [d.id for d in mesh.devices.flat] == list(rank_device_ids)
    assert [d.process_index for d in mesh.devices.flat] == list(range(8))


@pytest.mark.tpu_test
def test_mesh_rejects_pallas_ids_that_are_not_the_groups(monkeypatch):
    """The mesh's ids become the DeviceAssignment and ep_rank_order indexes
    them as the gathered device ids, so any other set must fail at startup."""
    import torch.distributed as dist
    from torch_tpu._internal import pallas as pallas_module

    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 4)
    monkeypatch.setattr(
        pallas_module,
        "get_pallas_mesh",
        lambda axis_names: _fake_mesh(_pallas_devices((0, 1, 2, 3)), axis_names),
    )
    monkeypatch.setattr(jax.sharding, "Mesh", _fake_mesh)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (40, 20, 10, 30))
    with pytest.raises(RuntimeError, match="do not match"):
        ep.build_ep_mesh()
    assert ep._MESH_CACHE == {}


@pytest.mark.tpu_test
def test_mesh_covers_the_stage_under_pipeline_parallelism(monkeypatch):
    """An EP group that is one pipeline stage gets a mesh of its own devices,
    built from its process group."""
    import torch.distributed as dist
    from torch_tpu._internal import pallas as pallas_module

    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 8)
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=3),
    )
    group = SimpleNamespace(world_size=2, ranks=[6, 7], device_group=object())
    monkeypatch.setattr(ep, "get_ep_group", lambda: group)
    monkeypatch.setattr(ep, "ep_device_ids", lambda: (2, 3))
    mesh = SimpleNamespace(
        devices=SimpleNamespace(flat=[SimpleNamespace(id=2), SimpleNamespace(id=3)])
    )
    made = Mock(return_value=mesh)
    monkeypatch.setattr(pallas_module, "get_pallas_mesh", made)
    monkeypatch.setattr(jax.sharding, "Mesh", _fake_mesh)

    assert [d.id for d in ep.build_ep_mesh().devices.flat] == [2, 3]
    made.assert_called_once_with(
        axis_names=("d",), mesh_shape=(2,), process_group=group.device_group
    )


@pytest.mark.tpu_test
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


@pytest.mark.parametrize(
    ("sequence_parallel", "rank_ordered_cp", "expected"),
    [
        (False, None, ((0, 3), (2, 1))),
        (True, None, ((0,), (1,), (2,), (3,))),
        # Rank order under PCP/DCP: mesh index i is EP rank i, i.e. global 2*i+2.
        (False, "get_pcp_world_size", ((2, 0), (3, 1))),
        (True, "get_pcp_world_size", ((0,), (1,), (2,), (3,))),
        (False, "get_dcp_world_size", ((2, 0), (3, 1))),
        (True, "get_dcp_world_size", ((0,), (1,), (2,), (3,))),
    ],
)
def test_token_groups_use_native_membership_and_mesh_order(
    monkeypatch, group, sequence_parallel, rank_ordered_cp, expected
):
    if rank_ordered_cp is not None:
        monkeypatch.setattr(ep, rank_ordered_cp, lambda: 2)
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


def test_ep_mesh_order_stays_rank_ordered_under_moe_sp(monkeypatch):
    device_ids = (0, 1, 4, 5, 6, 7, 2, 3)
    monkeypatch.setattr(ep.envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    assert ep._ep_mesh_order(device_ids) == (0, 1, 6, 7, 2, 3, 4, 5)
    monkeypatch.setattr(ep.envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    assert ep._ep_mesh_order(device_ids) == (0, 1, 2, 3, 4, 5, 6, 7)
