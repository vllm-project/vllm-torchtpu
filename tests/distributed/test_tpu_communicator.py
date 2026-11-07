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
    monkeypatch.setattr(comm, "PACKED_DISPATCH_MAX_TOKENS", 0)
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
    monkeypatch.setattr(comm, "PACKED_DISPATCH_MAX_TOKENS", 0)
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


def _dispatch_tensors(method, num_tokens):
    """Hidden states, then top-k or router logits, with every bit pattern
    (NaN payloads and denormals included) in the 32-bit words."""
    gen = torch.Generator().manual_seed(0)

    def words(*shape):
        return torch.randint(-(2**31), 2**31, shape, generator=gen, dtype=torch.int64)

    hidden = words(num_tokens, 64).to(torch.int16).view(torch.bfloat16)
    if method == "dispatch_router_logits":
        return [hidden, words(num_tokens, 8).to(torch.int32).view(torch.float32)]
    return [
        hidden,
        words(num_tokens, 4).to(torch.int32).view(torch.float32),
        words(num_tokens, 4).to(torch.int32),
    ]


def _two_rank_gather(x, dim):
    # The peer's rows: any row permutation keeps the gather row-aligned.
    return torch.cat([x, x.flip(0)], dim=dim)


def _bits(x):
    return x.view({2: torch.int16, 4: torch.int32}[x.element_size()])


@pytest.mark.parametrize("method", ["dispatch", "dispatch_router_logits"])
def test_decode_sized_dispatch_gathers_once_bit_exact(
    monkeypatch, communicator, method
):
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = _two_rank_gather
    tensors = _dispatch_tensors(method, comm.PACKED_DISPATCH_MAX_TOKENS)
    result = getattr(communicator, method)(*tensors)
    group.all_gather.assert_called_once()
    assert group.all_gather.call_args.args[0].dtype == torch.int16
    assert len(result) == len(tensors)
    for actual, expected in zip(result, tensors):
        assert actual.dtype == expected.dtype
        assert torch.equal(_bits(actual), _bits(_two_rank_gather(expected, 0)))


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2)]
)
def test_prefill_sized_dispatch_gathers_each_tensor(
    monkeypatch, communicator, method, count
):
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = _two_rank_gather
    tensors = _dispatch_tensors(method, comm.PACKED_DISPATCH_MAX_TOKENS + 1)
    getattr(communicator, method)(*tensors)
    calls = group.all_gather.call_args_list
    assert len(calls) == count
    assert all(c.args[0] is t for c, t in zip(calls, tensors))


@pytest.mark.parametrize(
    "method,count", [("dispatch", 3), ("dispatch_router_logits", 2)]
)
def test_drafter_dispatch_gathers_each_tensor(monkeypatch, communicator, method, count):
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    monkeypatch.setattr(comm.vllm_compilation_backends, "model_tag", "eagle_head")
    group.all_gather.side_effect = _two_rank_gather
    tensors = _dispatch_tensors(method, 8)
    getattr(communicator, method)(*tensors)
    assert group.all_gather.call_count == count
