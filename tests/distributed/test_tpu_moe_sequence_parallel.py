# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

import vllm_torchtpu.distributed.tpu_communicator as communicator


class _Group:

    def __init__(self, name: str, world_size: int = 2) -> None:
        self.name = name
        self.world_size = world_size
        self.calls = []
        self.device_communicator = self

    def all_gather(self, values: torch.Tensor, dim: int):
        self.calls.append(("all_gather", values.shape, dim))
        return values.repeat(self.world_size, *([1] * (values.ndim - 1)))

    def reduce_scatter(self, values: torch.Tensor, dim: int):
        self.calls.append(("reduce_scatter", values.shape, dim))
        return values[:values.shape[0] // self.world_size]

    def custom_all_gather(self, values: torch.Tensor):
        self.calls.append(("all_gather", values.shape, 0))
        return values.repeat(self.world_size, *([1] * (values.ndim - 1)))

    def custom_reduce_scatter(self, values: torch.Tensor):
        self.calls.append(("reduce_scatter", values.shape, 0))
        return values[:values.shape[0] // self.world_size]


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_tpu_custom_all_gather_uses_dim_zero_functional_collective(
        monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    calls = []

    def all_gather(values, gather_dim, group):
        calls.append((values.shape, gather_dim, group))
        return torch.cat((values, values), dim=0)

    monkeypatch.setattr(communicator.funcol, "all_gather_tensor", all_gather)
    monkeypatch.setattr(communicator.funcol, "wait_tensor",
                        lambda value: value)
    device_communicator = communicator.TpuDeviceCommunicator.__new__(
        communicator.TpuDeviceCommunicator)
    device_communicator.world_size = 2
    device_communicator.device_group = "tp"
    values = torch.arange(6).reshape(2, 3)

    gathered = device_communicator.custom_all_gather(values, dim=dim)

    expected_dim = dim % values.ndim
    torch.testing.assert_close(gathered,
                               torch.cat((values, values), dim=expected_dim))
    assert calls == [(values.shape, 0, "tp")]


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_tpu_custom_reduce_scatter_uses_dim_zero_functional_collective(
        monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    calls = []

    def reduce_scatter(values, op, scatter_dim, group):
        calls.append((values.shape, op, scatter_dim, group))
        return values[:values.shape[0] // 2]

    monkeypatch.setattr(communicator.funcol, "reduce_scatter_tensor",
                        reduce_scatter)
    monkeypatch.setattr(communicator.funcol, "wait_tensor",
                        lambda value: value)
    device_communicator = communicator.TpuDeviceCommunicator.__new__(
        communicator.TpuDeviceCommunicator)
    device_communicator.world_size = 2
    device_communicator.device_group = "tp"
    values = torch.arange(24).reshape(4, 6)

    scattered = device_communicator.custom_reduce_scatter(values, dim=dim)

    expected_dim = dim % values.ndim
    expected_shape = list(values.shape)
    expected_shape[expected_dim] //= 2
    assert list(scattered.shape) == expected_shape
    assert calls[0][1:] == ("sum", 0, "tp")


@pytest.mark.parametrize(("is_sequence_parallel", "expected_group"),
                         [(False, "dp"), (True, "ep")])
def test_tpu_moe_dispatch_and_combine_select_the_right_group(
        monkeypatch: pytest.MonkeyPatch, is_sequence_parallel: bool,
        expected_group: str) -> None:
    dp = _Group("dp")
    ep = _Group("ep")
    monkeypatch.setattr(communicator, "get_dp_group", lambda: dp)
    monkeypatch.setattr(communicator, "get_ep_group", lambda: ep)
    device_communicator = communicator.TpuDeviceCommunicator.__new__(
        communicator.TpuDeviceCommunicator)
    hidden_states = torch.ones(2, 4)
    router_logits = torch.ones(2, 8)

    dispatched, logits = device_communicator.dispatch_router_logits(
        hidden_states,
        router_logits,
        is_sequence_parallel=is_sequence_parallel,
    )
    combined = device_communicator.combine(
        dispatched,
        is_sequence_parallel=is_sequence_parallel,
    )

    selected = ep if expected_group == "ep" else dp
    unselected = dp if expected_group == "ep" else ep
    assert dispatched.shape == (4, 4)
    assert logits.shape == (4, 8)
    assert combined.shape == hidden_states.shape
    assert [call[0] for call in selected.calls
            ] == ["all_gather", "all_gather", "reduce_scatter"]
    assert unselected.calls == []
