# SPDX-License-Identifier: Apache-2.0

import builtins
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed import mesh_utils, pcp


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch):
    monkeypatch.setattr(mesh_utils, "_LAYOUT_CACHE", {})
    monkeypatch.setattr(mesh_utils, "_MESH_CACHE", {})


@pytest.mark.parametrize(
    "group",
    [
        None,
        SimpleNamespace(world_size=1, rank_in_group=0),
        SimpleNamespace(world_size=4, rank_in_group=2),
    ],
)
def test_native_group_accessors(monkeypatch, group):
    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: group)
    assert pcp.get_pcp_group() is group
    assert pcp.get_pcp_rank() == (group.rank_in_group if group else 0)
    assert pcp.get_pcp_world_size() == (group.world_size if group else 1)


def test_get_pcp_group_returns_none_when_native_symbol_is_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.distributed.parallel_state":
            raise ImportError("native PCP group is unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert pcp.get_pcp_group() is None


def test_uninitialized_native_group(monkeypatch):
    monkeypatch.setattr(
        parallel_state, "get_pcp_group", Mock(side_effect=AssertionError)
    )
    assert pcp.get_pcp_group() is None


@pytest.mark.parametrize("group", [None, SimpleNamespace(world_size=1)])
def test_single_rank_collectives_preserve_tensor(monkeypatch, group):
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    tensor = torch.ones(2, 3).t()
    assert pcp.all_gather_equal_tokens(tensor) is tensor
    assert pcp.all_reduce_sum(tensor) is tensor


@pytest.mark.parametrize(
    "dim,as_list,contiguous",
    [
        (0, False, False),
        (1, False, False),
        (1, False, True),
        (1, True, False),
        (1, True, True),
    ],
)
def test_all_gather_equal_tokens_materializes_contiguous_input(
    monkeypatch, dim, as_list, contiguous
):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    if contiguous:
        tensor = tensor.contiguous()
    group = Mock(world_size=2)

    def gather(value, dim):
        assert value.is_contiguous()
        torch.testing.assert_close(value, tensor)
        parts = [value, value + 10]
        return parts if as_list else torch.cat(parts, dim=dim)

    group.all_gather.side_effect = gather
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    torch.testing.assert_close(
        pcp.all_gather_equal_tokens(tensor, dim=dim),
        torch.cat([tensor, tensor + 10], dim=dim),
    )
    group.all_gather.assert_called_once()
    assert group.all_gather.call_args.kwargs == {"dim": dim}


@pytest.mark.parametrize("contiguous", [True, False])
def test_reduce_materializes_input(monkeypatch, contiguous):
    tensor = torch.arange(6).reshape(2, 3)
    if not contiguous:
        tensor = tensor.t()
    group = Mock(world_size=2)
    group.all_reduce.side_effect = lambda value: value * 2
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    torch.testing.assert_close(pcp.all_reduce_sum(tensor), tensor * 2)
    value = group.all_reduce.call_args.args[0]
    assert value.is_contiguous()
    if contiguous:
        assert value is tensor


@pytest.mark.parametrize(
    "available,initialized,expected",
    [(False, False, 0), (True, False, 0), (True, True, 7)],
)
def test_global_rank_probe(monkeypatch, available, initialized, expected):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: available)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    get_rank = Mock(return_value=7)
    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    assert mesh_utils._get_current_global_rank() == expected
    assert get_rank.call_count == int(available and initialized)


@pytest.fixture
def runtime():
    from torch_tpu._internal.distributed import tpu_distributed

    return tpu_distributed


def test_native_device_id(monkeypatch, runtime):
    monkeypatch.setattr(runtime, "global_device_id", lambda: 9)
    assert mesh_utils._get_tpu_global_device_id() == 9


@pytest.mark.parametrize("devices", [[], [SimpleNamespace(id=17)], [SimpleNamespace()]])
def test_device_id_fallback(monkeypatch, runtime, devices):
    monkeypatch.setattr(runtime, "global_device_id", Mock(side_effect=RuntimeError))
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    assert mesh_utils._get_tpu_global_device_id() == (
        17 if devices and hasattr(devices[0], "id") else 0
    )


def test_device_id_without_runtime(monkeypatch, runtime):
    monkeypatch.setattr(runtime, "global_device_id", Mock(side_effect=RuntimeError))
    monkeypatch.setattr(jax, "local_devices", Mock(side_effect=RuntimeError))
    assert mesh_utils._get_tpu_global_device_id() == 0


@pytest.mark.parametrize("group", [None, SimpleNamespace(world_size=1)])
def test_single_rank_cache_rank(monkeypatch, group, runtime):
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    device_id = Mock()
    monkeypatch.setattr(runtime, "global_device_id", device_id)
    assert pcp.get_pcp_cache_rank() == 0
    device_id.assert_not_called()


@pytest.mark.parametrize(
    "device_id,ids,expected,error",
    [
        (6, [4, 7, 6, 5], 2, None),
        (6, [4, 6], None, "partition space"),
        (8, [4, 7, 6, 5], None, "not in torch_tpu"),
    ],
)
def test_cache_rank_uses_physical_partition_order(
    monkeypatch, runtime, device_id, ids, expected, error
):
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: SimpleNamespace(world_size=4))
    monkeypatch.setattr(runtime, "global_device_id", lambda: device_id)
    monkeypatch.setattr(runtime, "all_global_device_ids", lambda: ids)
    if error:
        with pytest.raises(RuntimeError, match=error):
            pcp.get_pcp_cache_rank()
    else:
        assert pcp.get_pcp_cache_rank() == expected


@pytest.mark.parametrize("missing", [False, True])
def test_rank_device_collective(monkeypatch, missing):
    cpu_group = object()
    group = SimpleNamespace(world_size=2, cpu_group=cpu_group)

    def gather(output, value, *, group):
        assert group is cpu_group
        assert value == (6, 20)
        assert output == [None, None]
        output[:] = [(6, 20), None if missing else (2, 40)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if missing:
        with pytest.raises(RuntimeError, match="empty entry"):
            mesh_utils._collect_rank_to_device_id(group, 6, 20)
    else:
        assert mesh_utils._collect_rank_to_device_id(group, 6, 20) == {6: 20, 2: 40}


def test_group_layout_orders_mapping_and_scopes_cache(monkeypatch):
    group = SimpleNamespace(world_size=4, ranks=[2, 4, 6, 8], rank_in_group=2)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    monkeypatch.setattr(mesh_utils, "_get_current_global_rank", lambda: 6)
    monkeypatch.setattr(mesh_utils, "_get_tpu_global_device_id", lambda: 10)
    gather = Mock(return_value={8: 30, 6: 10, 4: 20, 2: 40})
    monkeypatch.setattr(mesh_utils, "_collect_rank_to_device_id", gather)
    layout = pcp.get_pcp_group_layout()
    assert layout == pcp.PcpGroupLayout((2, 4, 6, 8), (40, 20, 10, 30), 2, 4)
    assert (layout.device_id, layout.prev_rank_in_group, layout.next_rank_in_group) == (
        10,
        1,
        3,
    )
    assert pcp.get_pcp_group_layout() is layout
    gather.assert_called_once_with(group, 6, 10)
    group.ranks = [8, 6, 4, 2]
    group.rank_in_group = 1
    assert pcp.get_pcp_group_layout().device_ids == (30, 10, 20, 40)
    assert gather.call_count == 2


@pytest.mark.parametrize("rank,prev,next_rank", [(0, 3, 1), (3, 2, 0)])
def test_ring_wraparound(rank, prev, next_rank):
    layout = pcp.PcpGroupLayout((0, 1, 2, 3), (10, 11, 12, 13), rank, 4)
    assert layout.prev_rank_in_group == prev
    assert layout.next_rank_in_group == next_rank


def test_single_rank_layout(monkeypatch):
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: None)
    monkeypatch.setattr(mesh_utils, "_get_current_global_rank", lambda: 6)
    monkeypatch.setattr(mesh_utils, "_get_tpu_global_device_id", lambda: 10)
    layout = pcp.get_pcp_group_layout()
    assert layout == pcp.PcpGroupLayout((6,), (10,), 0, 1)
    assert layout.prev_rank_in_group == layout.next_rank_in_group == 0


def test_layout_missing_rank_is_not_cached(monkeypatch):
    group = SimpleNamespace(world_size=2, ranks=[2, 6], rank_in_group=0)
    monkeypatch.setattr(pcp, "get_pcp_group", lambda: group)
    monkeypatch.setattr(mesh_utils, "_get_current_global_rank", lambda: 2)
    monkeypatch.setattr(mesh_utils, "_get_tpu_global_device_id", lambda: 10)
    monkeypatch.setattr(mesh_utils, "_collect_rank_to_device_id", lambda *args: {2: 10})
    with pytest.raises(RuntimeError, match=r"missing=\[6\]"):
        pcp.get_pcp_group_layout()
    assert mesh_utils._LAYOUT_CACHE == {}


def test_mesh_preserves_group_order_and_caches_by_axis_and_devices(monkeypatch):
    layout = pcp.PcpGroupLayout((2, 6), (30, 10), 0, 2)
    monkeypatch.setattr(mesh_utils, "get_cp_group_layout", lambda group: layout)
    monkeypatch.setattr(
        jax, "devices", lambda: [SimpleNamespace(id=i) for i in [10, 20, 30]]
    )
    mesh = Mock(side_effect=lambda *args, **kwargs: object())
    monkeypatch.setattr(jax.sharding, "Mesh", mesh)
    first = pcp.get_or_create_pcp_mesh()
    assert [device.id for device in mesh.call_args.args[0]] == [30, 10]
    assert mesh.call_args.kwargs == {"axis_names": ("pcp",)}
    assert pcp.get_or_create_pcp_mesh() is first
    assert pcp.get_or_create_pcp_mesh("ring") is not first
    layout = pcp.PcpGroupLayout((6, 2), (10, 30), 1, 2)
    assert pcp.get_or_create_pcp_mesh() is not first
    assert mesh.call_count == 3


def test_mesh_missing_device_is_not_cached(monkeypatch):
    monkeypatch.setattr(
        mesh_utils,
        "get_cp_group_layout",
        lambda group: pcp.PcpGroupLayout((2, 6), (30, 10), 0, 2),
    )
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(id=30)])
    with pytest.raises(RuntimeError, match=r"missing=\[10\]"):
        pcp.get_or_create_pcp_mesh()
    assert mesh_utils._MESH_CACHE == {}
