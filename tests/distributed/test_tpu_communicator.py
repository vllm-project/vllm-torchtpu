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


def _other_rank(x):
    # What the second DP rank contributes: distinct values, same shape.
    return x * 3 + 1


@pytest.mark.parametrize("dp_size", [1, 2])
def test_packed_dispatch_matches_two_gathers(monkeypatch, communicator, dp_size):
    group = Mock(world_size=dp_size)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat(
        [x, _other_rank(x)], dim=dim
    )
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(3, 16, generator=generator).to(torch.bfloat16)
    logits = torch.randn(3, 5, generator=generator).to(torch.bfloat16)

    monkeypatch.setattr(comm.envs, "TPU_MOE_PACKED_DISPATCH", False, raising=False)
    expected = communicator.dispatch_router_logits(hidden, logits)
    unpacked_calls = group.all_gather.call_count
    group.all_gather.reset_mock()
    monkeypatch.setattr(comm.envs, "TPU_MOE_PACKED_DISPATCH", True, raising=False)
    result = communicator.dispatch_router_logits(hidden, logits)

    assert group.all_gather.call_count == (0 if dp_size == 1 else 1)
    assert unpacked_calls == (0 if dp_size == 1 else 2)
    for actual, reference in zip(result, expected):
        assert actual.dtype == reference.dtype
        assert torch.equal(actual, reference)
        assert actual.is_contiguous()
    if dp_size == 2:
        (packed,), kwargs = group.all_gather.call_args
        assert kwargs == {"dim": 0}
        assert packed.shape == (3, 21)


def test_packed_dispatch_keeps_two_gathers_for_mixed_dtypes(monkeypatch, communicator):
    group = Mock(world_size=2)
    monkeypatch.setattr(comm, "get_dp_group", lambda: group)
    group.all_gather.side_effect = lambda x, dim: torch.cat([x, x], dim=dim)
    monkeypatch.setattr(comm.envs, "TPU_MOE_PACKED_DISPATCH", True, raising=False)
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    logits = torch.ones(2, 3, dtype=torch.float32)

    result = communicator.dispatch_router_logits(hidden, logits)

    assert group.all_gather.call_count == 2
    assert [t.dtype for t in result] == [torch.bfloat16, torch.float32]
