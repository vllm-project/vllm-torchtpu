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

from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.distributed import tpu_communicator as comm


@pytest.fixture
def communicator(monkeypatch):
    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    return comm.TpuDeviceCommunicator()


@pytest.mark.parametrize("use_all2all", [False, True])
def test_all2all_probe_manager(monkeypatch, use_all2all):
    cpu_group = object()

    def init(instance, *args, **kwargs):
        instance.use_all2all = use_all2all
        instance.cpu_group = cpu_group

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    manager = Mock()
    monkeypatch.setattr(comm, "TpuNullAll2AllManager", manager)
    result = comm.TpuDeviceCommunicator()
    if use_all2all:
        manager.assert_called_once_with(cpu_group)
        assert result.all2all_manager is manager.return_value
    else:
        manager.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
@pytest.mark.parametrize("dp_size", [1, 2])
def test_dispatch_and_combine_use_dp_group(
    monkeypatch, communicator, method, count, dp_size
):
    group = Mock(world_size=dp_size)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    group.reduce_scatter.return_value = torch.tensor([[11, 12]])
    result = getattr(communicator, method)(*tensors)
    if method == "combine":
        if dp_size == 1:
            assert result is tensors[0]
            group.reduce_scatter.assert_not_called()
        else:
            group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
            assert result is group.reduce_scatter.return_value
        group.all_gather.assert_not_called()
    elif dp_size == 1:
        assert len(result) == count
        assert all(actual is expected for actual, expected in zip(result, tensors))
        group.all_gather.assert_not_called()
    else:
        assert len(result) == count
        for actual, expected, invocation in zip(
            result, tensors, group.all_gather.call_args_list
        ):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
            assert invocation.args[0] is expected
            assert invocation.kwargs == {"dim": 0}
        assert group.all_gather.call_count == count
        group.reduce_scatter.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2)]
)
def test_extra_tensors_rejected_before_collectives(
    monkeypatch, communicator, method, count
):
    get_group = Mock()
    monkeypatch.setattr(comm, "get_dp_group", get_group)
    with pytest.raises(NotImplementedError, match="extra_tensors"):
        getattr(communicator, method)(*[torch.ones(1)] * count, extra_tensors=[])
    get_group.assert_not_called()


def test_moe_group_with_sequence_parallel(monkeypatch, communicator):
    ep_group = Mock(world_size=8)
    dp_group = Mock(world_size=1)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    assert communicator._moe_group(is_sequence_parallel=True) is ep_group


def test_moe_group_without_sequence_parallel(monkeypatch, communicator):
    dp_group_1 = Mock(world_size=1)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group_1)
    assert communicator._moe_group(is_sequence_parallel=False) is None

    dp_group_4 = Mock(world_size=4)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group_4)
    assert communicator._moe_group(is_sequence_parallel=False) is dp_group_4


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
def test_dispatch_and_combine_with_sequence_parallel(
    monkeypatch, communicator, method, count
):
    ep_group = Mock(world_size=8)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    # The runner marks a step at or above the row threshold as sliced.
    monkeypatch.setattr(comm.sequence_parallel, "_ACTIVE", True)
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    ep_group.reduce_scatter.return_value = torch.tensor([[11, 12]])

    result = getattr(communicator, method)(*tensors, is_sequence_parallel=True)
    if method == "combine":
        ep_group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
        assert result is ep_group.reduce_scatter.return_value
        ep_group.all_gather.assert_not_called()
    else:
        assert len(result) == count
        for actual, expected, invocation in zip(
            result, tensors, ep_group.all_gather.call_args_list
        ):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
            assert invocation.args[0] is expected
            assert invocation.kwargs == {"dim": 0}
        assert ep_group.all_gather.call_count == count
        ep_group.reduce_scatter.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
def test_dispatch_and_combine_with_replicated_sequence_parallel_rows(
    monkeypatch, communicator, method, count
):
    # A step below the row threshold: every rank holds every row, so dispatch
    # passes through and combine all-reduces over EP.
    ep_group = Mock(world_size=8)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm, "get_dp_group", lambda: Mock(world_size=1))
    monkeypatch.setattr(comm.sequence_parallel, "_ACTIVE", False)
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    ep_group.all_reduce.return_value = torch.tensor([[11, 12]])

    result = getattr(communicator, method)(*tensors, is_sequence_parallel=True)
    if method == "combine":
        ep_group.all_reduce.assert_called_once_with(tensors[0])
        assert result is ep_group.all_reduce.return_value
    else:
        assert all(actual is expected for actual, expected in zip(result, tensors))
        ep_group.all_reduce.assert_not_called()
    ep_group.all_gather.assert_not_called()
    ep_group.reduce_scatter.assert_not_called()


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2), ("combine", 1)]
)
def test_replicated_sequence_parallel_rows_under_data_parallelism(
    monkeypatch, communicator, method, count
):
    # DP4 x TP2: a step below the row threshold holds only this engine's rows
    # (replicated over its TP pair). Dispatch must gather the other engines'
    # rows over DP, and combine must return this engine's rows summed over
    # every EP rank: DP reduce-scatter, then TP all-reduce. An EP all-reduce
    # would add other engines' rows into this engine's.
    ep_group = Mock(world_size=8)
    dp_group = Mock(world_size=4)
    tp_group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(comm, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(comm.sequence_parallel, "_ACTIVE", False)
    monkeypatch.setattr(
        comm.small_all_reduce_op, "dp_combine_or", lambda x, fallback: fallback(x)
    )
    tensors = [torch.tensor([[i, i + 1]]) for i in range(count)]
    dp_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    dp_group.reduce_scatter.return_value = torch.tensor([[11, 12]])
    tp_group.all_reduce.return_value = torch.tensor([[22, 24]])

    result = getattr(communicator, method)(*tensors, is_sequence_parallel=True)
    if method == "combine":
        dp_group.reduce_scatter.assert_called_once_with(tensors[0], dim=0)
        tp_group.all_reduce.assert_called_once_with(
            dp_group.reduce_scatter.return_value
        )
        assert result is tp_group.all_reduce.return_value
        dp_group.all_gather.assert_not_called()
    else:
        assert len(result) == count
        for actual, expected in zip(result, tensors):
            torch.testing.assert_close(actual, torch.cat([expected, expected + 10]))
        assert dp_group.all_gather.call_count == count
        dp_group.reduce_scatter.assert_not_called()
        tp_group.all_reduce.assert_not_called()
    ep_group.all_reduce.assert_not_called()
    ep_group.all_gather.assert_not_called()
    ep_group.reduce_scatter.assert_not_called()


def test_all_reduce_goes_through_the_small_collective_bridge(monkeypatch, communicator):
    base = Mock(return_value=torch.tensor([7.0]))
    monkeypatch.setattr(comm.DeviceCommunicatorBase, "all_reduce", base)
    bridge = Mock(side_effect=lambda x, ranks, fallback: fallback(x))
    monkeypatch.setattr(comm.small_all_reduce_op, "all_reduce_or", bridge)
    communicator.ranks = [0, 1]
    x = torch.ones(2, 4)
    assert communicator.all_reduce(x) is base.return_value
    (block, ranks, _), _ = bridge.call_args
    assert block is x and ranks == [0, 1]
    base.assert_called_once_with(x)


def test_all_reduce_unarmed_is_the_base_all_reduce(monkeypatch, communicator):
    base = Mock(return_value=torch.tensor([7.0]))
    monkeypatch.setattr(comm.DeviceCommunicatorBase, "all_reduce", base)
    monkeypatch.setattr(comm.small_all_reduce_op, "_AR_OP", None)
    communicator.ranks = [0, 1]
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    assert communicator.all_reduce(x) is base.return_value
    base.assert_called_once_with(x)


def test_combine_goes_through_the_small_collective_bridge(monkeypatch, communicator):
    group = Mock(world_size=2)
    group.reduce_scatter.return_value = torch.tensor([[11.0, 12.0]])
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    bridge = Mock(side_effect=lambda x, fallback: fallback(x))
    monkeypatch.setattr(comm.small_all_reduce_op, "dp_combine_or", bridge)
    x = torch.ones(2, 2)
    assert communicator.combine(x) is group.reduce_scatter.return_value
    assert bridge.call_args.args[0] is x
    group.reduce_scatter.assert_called_once_with(x, dim=0)


def test_dispatch_router_logits_uses_fused_dp_dispatch_when_eligible(
    monkeypatch, communicator
):
    group = Mock(world_size=8)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    hidden = torch.zeros(1, 4096, dtype=torch.bfloat16)
    logits = torch.zeros(1, 512, dtype=torch.float32)
    expected = (
        torch.ones(8, 4096, dtype=torch.bfloat16),
        torch.ones(8, 512, dtype=torch.float32),
    )
    dp_dispatch = Mock(return_value=expected)
    monkeypatch.setattr(comm.small_all_reduce_op, "dp_dispatch_or", dp_dispatch)

    got = communicator.dispatch_router_logits(hidden, logits)
    assert got is expected
    assert dp_dispatch.call_count == 1
    assert dp_dispatch.call_args.args[:2] == (hidden, logits)
    group.all_gather.assert_not_called()


def test_dispatch_router_logits_sliced_sp_bypasses_dp_dispatch_bridge(
    monkeypatch, communicator
):
    ep_group = Mock(world_size=8)
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm.sequence_parallel, "_ACTIVE", True)
    dp_dispatch = Mock()
    monkeypatch.setattr(comm.small_all_reduce_op, "dp_dispatch_or", dp_dispatch)

    hidden = torch.zeros(8, 4096, dtype=torch.bfloat16)
    logits = torch.zeros(8, 512, dtype=torch.float32)
    got_h, got_lg = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    assert got_h.shape == (16, 4096) and got_lg.shape == (16, 512)
    assert ep_group.all_gather.call_count == 2
    dp_dispatch.assert_not_called()


def _replicated_dp4(monkeypatch):
    ep_group = Mock(world_size=8)
    dp_group = Mock(world_size=4)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(comm, "get_tp_group", lambda: Mock(world_size=2))
    monkeypatch.setattr(comm.sequence_parallel, "_ACTIVE", False)
    # Four engines: engine k holds its rows scaled by k + 1 (elementwise, so a
    # packed and an unpacked gather see the same per-engine values).
    dp_group.all_gather.side_effect = lambda x, dim: torch.cat(
        [x * (k + 1) for k in range(4)], dim=dim
    )
    return dp_group


@pytest.mark.parametrize("rows", [16, 28])
def test_packed_router_gather_is_one_all_gather_with_identical_values(
    monkeypatch, communicator, rows
):
    # DP4 x TP2 replicated step (B 64: 16 rows per engine x 4 MTP rows):
    # TPU_DP_PACKED_ROUTER_GATHER gathers hidden rows and router logits in one
    # DP all-gather, and the split returns bit-identical tensors.
    gen = torch.Generator().manual_seed(rows)
    hidden = torch.randn(rows * 4, 4096, generator=gen).to(torch.bfloat16)
    logits = torch.randn(rows * 4, 512, generator=gen).to(torch.bfloat16)

    monkeypatch.setattr(comm.envs, "TPU_DP_PACKED_ROUTER_GATHER", False)
    dp_group = _replicated_dp4(monkeypatch)
    ref_h, ref_l = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    assert dp_group.all_gather.call_count == 2

    monkeypatch.setattr(comm.envs, "TPU_DP_PACKED_ROUTER_GATHER", True)
    dp_group = _replicated_dp4(monkeypatch)
    got_h, got_l = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    assert dp_group.all_gather.call_count == 1
    (packed_in,), kwargs = dp_group.all_gather.call_args
    assert kwargs == {"dim": 0}
    assert packed_in.shape == (rows * 4, 4096 + 512)
    assert got_h.shape == ref_h.shape == (rows * 16, 4096)
    assert got_l.shape == ref_l.shape == (rows * 16, 512)
    assert got_h.dtype == got_l.dtype == torch.bfloat16
    assert torch.equal(got_h, ref_h)
    assert torch.equal(got_l, ref_l)


@pytest.mark.parametrize(
    "logits_dtype,dp_size,active",
    [(torch.float32, 4, False), (torch.bfloat16, 1, False), (torch.bfloat16, 4, True)],
)
def test_packed_router_gather_only_on_replicated_dp_same_dtype(
    monkeypatch, communicator, logits_dtype, dp_size, active
):
    # Mixed dtypes keep two gathers; a single engine and a sliced (SP-active)
    # step never take the packed path.
    monkeypatch.setattr(comm.envs, "TPU_DP_PACKED_ROUTER_GATHER", True)
    dp_group = _replicated_dp4(monkeypatch)
    dp_group.world_size = dp_size
    ep_group = Mock(world_size=8)
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(comm.sequence_parallel, "_ACTIVE", active)
    hidden = torch.ones(8, 16, dtype=torch.bfloat16)
    logits = torch.ones(8, 4, dtype=logits_dtype)
    h, lg = communicator.dispatch_router_logits(
        hidden, logits, is_sequence_parallel=True
    )
    assert lg.dtype == logits_dtype and lg.shape[-1] == 4 and h.shape[-1] == 16
    if active:
        assert ep_group.all_gather.call_count == 2
    elif dp_size == 1:
        assert h is hidden and lg is logits
        dp_group.all_gather.assert_not_called()
    else:
        assert dp_group.all_gather.call_count == 2
