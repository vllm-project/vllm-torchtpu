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
from vllm_torchtpu.distributed.ep_combine_layout import (
    COMBINE_BLOCKS,
    COMBINE_SPLIT_MIN_ROWS,
    combine_piece_rows,
    split_pieces,
)


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
    "method,count", [("dispatch_router_logits", 2), ("combine", 1)]
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
    "method,count", [("dispatch_router_logits", 2), ("combine", 1)]
)
def test_dispatch_and_combine_with_sequence_parallel(
    monkeypatch, communicator, method, count
):
    ep_group = Mock(world_size=8)
    monkeypatch.setattr(comm, "get_ep_group", lambda: ep_group)
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


def _rank_offset_all_gather(world):
    """all_gather where rank r contributes the caller's tensor plus r."""
    return lambda x, dim: torch.cat([x + r for r in range(world)], dim=dim)


@pytest.mark.parametrize("is_sequence_parallel", [False, True])
@pytest.mark.parametrize(
    "weights_dtype,ids_dtype",
    [
        (torch.bfloat16, torch.int32),
        (torch.float32, torch.int32),
        (torch.float32, torch.int64),
    ],
)
def test_dispatch_gathers_routes_in_one_packed_gather(
    monkeypatch, communicator, is_sequence_parallel, weights_dtype, ids_dtype
):
    world, tokens, topk = 4, 5, 6
    group = Mock(world_size=world)
    group.all_gather.side_effect = _rank_offset_all_gather(world)
    getter = "get_ep_group" if is_sequence_parallel else "get_dp_group"
    monkeypatch.setattr(comm, getter, lambda: group)
    hidden = torch.zeros(tokens, 8)
    weights = torch.randn(tokens, topk).to(weights_dtype)
    weights[0, 0] = float("nan")
    ids = torch.randint(0, 256, (tokens, topk), dtype=ids_dtype)

    out_hidden, out_weights, out_ids = communicator.dispatch(
        hidden, weights, ids, is_sequence_parallel=is_sequence_parallel
    )

    assert group.all_gather.call_count == 2
    assert group.all_gather.call_args_list[0].args[0] is hidden
    packed = group.all_gather.call_args_list[1].args[0]
    assert packed.dtype == torch.int32 and packed.dim() == 1
    torch.testing.assert_close(
        out_hidden, torch.cat([hidden + r for r in range(world)])
    )
    assert out_weights.shape == (world * tokens, topk)
    assert out_ids.shape == (world * tokens, topk)
    assert out_weights.dtype == weights_dtype and out_ids.dtype == ids_dtype
    # Rank 0 sent its routes unchanged: they must arrive bit for bit.
    words = torch.int16 if weights_dtype == torch.bfloat16 else torch.int32
    assert torch.equal(out_weights[:tokens].view(words), weights.view(words))
    assert torch.equal(out_ids[:tokens], ids)
    # Every other rank's rows are its own, in rank order.
    for r in range(1, world):
        assert not torch.equal(out_ids[r * tokens : (r + 1) * tokens], ids)


def test_dispatch_single_rank_returns_inputs(monkeypatch, communicator):
    group = Mock(world_size=1)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    tensors = (torch.ones(2, 4), torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int32))
    result = communicator.dispatch(*tensors)
    assert all(a is b for a, b in zip(result, tensors))
    group.all_gather.assert_not_called()


def test_dispatch_keeps_a_dynamic_token_count_dynamic(monkeypatch, communicator):
    """The packed gather reshapes by the traced token count."""

    class Group:  # Dynamo does not trace into a Mock
        world_size = 8
        all_gather = staticmethod(_rank_offset_all_gather(8))

    monkeypatch.setattr(comm, "get_dp_group", lambda: Group)
    torch._dynamo.reset()
    inputs = (
        torch.ones(1024, 4),
        torch.ones(1024, 6, dtype=torch.bfloat16),
        torch.zeros(1024, 6, dtype=torch.int32),
    )
    for tensor in inputs:
        torch._dynamo.mark_dynamic(tensor, 0)
    result = torch.compile(communicator.dispatch, backend="eager", fullgraph=True)(
        *inputs
    )
    assert [t.shape for t in result] == [(8192, 4), (8192, 6), (8192, 6)]


def _replicated_reduce_scatter(world, rank):
    """reduce_scatter seen by `rank` when every rank holds the same input."""
    return lambda x, dim: world * x.unflatten(0, (world, -1))[rank]


@pytest.mark.parametrize("rank", [0, 5])
def test_combine_in_pieces_matches_one_reduce_scatter(monkeypatch, communicator, rank):
    world = 8
    group = Mock(world_size=world)
    group.reduce_scatter.side_effect = _replicated_reduce_scatter(world, rank)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    monkeypatch.setattr(comm, "get_ep_group", lambda: group)
    hidden = torch.arange(COMBINE_SPLIT_MIN_ROWS * 2.0).view(-1, 2)

    piece_rows = combine_piece_rows(hidden.shape[0])
    result = communicator.combine(torch.cat(split_pieces(hidden, piece_rows)))

    assert group.reduce_scatter.call_count == 3
    torch.testing.assert_close(result, world * hidden.view(world, -1, 2)[rank])


def test_combine_drafter_keeps_one_reduce_scatter(monkeypatch, communicator):
    monkeypatch.setattr(comm.vllm_compilation_backends, "model_tag", "eagle_head")
    group = Mock(world_size=8)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    hidden = torch.ones(COMBINE_SPLIT_MIN_ROWS, 2)

    result = communicator.combine(hidden)

    group.reduce_scatter.assert_called_once_with(hidden, dim=0)
    assert result is group.reduce_scatter.return_value


@pytest.mark.parametrize(
    "dp,ep,expert_parallel",
    [(8, 8, True), (16, 16, True), (8, 16, True), (8, 8, False)],
)
def test_combine_splits_only_under_ep_with_two_blocks_per_rank_and_equal_groups(
    monkeypatch, dp, ep, expert_parallel
):
    dp_group = Mock(world_size=dp)
    dp_group.device_communicator.expert_parallel = expert_parallel
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(comm, "get_ep_group", lambda: Mock(world_size=ep))
    assert comm.combine_splits() == (dp == ep == 8 and expert_parallel)


@pytest.mark.parametrize(
    "rows,expected",
    [
        (8192, (128, 128, 256)),
        (12288, (192, 192, 384)),
        (8200, ()),
        (4096, ()),
        (6144, ()),
        (16384, ()),
        (2048, ()),
    ],
)
def test_combine_piece_rows(rows, expected):
    assert combine_piece_rows(rows) == expected


def test_combine_piece_rows_keeps_a_dynamic_row_count_dynamic():
    """combine reads the row count while vLLM traces the token count dynamic."""

    def pieces(hidden):
        rows = combine_piece_rows(hidden.shape[0])
        return hidden.split([COMBINE_BLOCKS * r for r in rows])

    torch._dynamo.reset()
    hidden = torch.ones(COMBINE_SPLIT_MIN_ROWS)
    torch._dynamo.mark_dynamic(hidden, 0)
    result = torch.compile(pieces, backend="eager", fullgraph=True)(hidden)
    assert [p.shape[0] for p in result] == [2048, 2048, 4096]
