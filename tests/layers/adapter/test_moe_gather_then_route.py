# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner

forward_impl = VllmMoERunner._forward_impl


class _FakeDpGroup:
    def __init__(self):
        self.gather_args = []

    def all_gather(self, tensor, dim=0):
        self.gather_args.append((tensor, dim))
        return tensor.repeat(4, 1)


class _FakeEpGroup:
    def __init__(self):
        self.combine_args = None
        self.dispatch_args = []

    def dispatch_router_logits(
        self, hidden_states, router_logits, is_sequence_parallel
    ):
        self.dispatch_args.append((hidden_states, router_logits, is_sequence_parallel))
        return hidden_states.repeat(4, 1), router_logits.repeat(4, 1)

    def combine(self, hidden_states, is_sequence_parallel):
        self.combine_args = (hidden_states, is_sequence_parallel)
        return hidden_states[: hidden_states.shape[0] // 4]


class _FakeGate:
    def __init__(self):
        self.input = None

    def __call__(self, hidden_states):
        self.input = hidden_states
        return hidden_states[:, :2], None


class _FakeQuantMethod:
    def __init__(self):
        self.external_args = None

    def apply_monolithic_external_dispatch(self, **kwargs):
        self.external_args = kwargs
        return kwargs["x"] + 1


def test_small_dp_bucket_gathers_hidden_before_routing(monkeypatch):
    runner_cls = moe_runner_mod.MoERunner
    original_calls = []

    def original_forward_impl(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "original"

    monkeypatch.setattr(runner_cls, "_forward_impl", original_forward_impl)
    monkeypatch.setattr(envs, "TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS", 32)

    dp_group = _FakeDpGroup()
    ep_group = _FakeEpGroup()
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(moe_runner_mod, "get_ep_group", lambda: ep_group)

    gate = _FakeGate()
    quant_method = _FakeQuantMethod()
    routed_experts = SimpleNamespace(
        quant_method=quant_method,
        _ensure_moe_quant_config_init=lambda: None,
    )
    shared_orders = []
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(dp_size=4, pcp_size=1, is_sequence_parallel=False),
        routed_experts=routed_experts,
        gate=gate,
        _fse_fuse_gate=False,
        _shared_experts=None,
        _maybe_apply_shared_experts=lambda value, order: shared_orders.append(order),
        _sequence_parallel_context=lambda: nullcontext(),
    )

    hidden_states = torch.arange(96, dtype=torch.float32).reshape(32, 3)
    actual = forward_impl(runner, hidden_states, torch.empty(32, 2), None)

    assert not original_calls
    assert len(dp_group.gather_args) == 1
    assert dp_group.gather_args[0][0] is hidden_states
    assert dp_group.gather_args[0][1] == 0
    assert gate.input.shape == (128, 3)
    assert quant_method.external_args["x"] is gate.input
    assert quant_method.external_args["router_logits"].shape == (128, 2)
    assert ep_group.combine_args[0].shape == (128, 3)
    assert ep_group.combine_args[1] is False
    assert actual.shape == hidden_states.shape
    assert shared_orders == [moe_runner_mod.SharedExpertsOrder.NO_OVERLAP]

    assert (
        forward_impl(runner, torch.empty(33, 3), torch.empty(33, 2), None) == "original"
    )
    assert len(original_calls) == 1
    assert len(dp_group.gather_args) == 1


def test_larger_bucket_keeps_compute_then_gather(monkeypatch):
    runner_cls = moe_runner_mod.MoERunner
    original_calls = []

    def original_forward_impl(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "internal"

    monkeypatch.setattr(runner_cls, "_forward_impl", original_forward_impl)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 256)
    monkeypatch.setattr(envs, "TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS", 32)

    dp_group = _FakeDpGroup()
    ep_group = _FakeEpGroup()
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(moe_runner_mod, "get_ep_group", lambda: ep_group)

    gate = _FakeGate()
    quant_method = _FakeQuantMethod()
    routed_experts = SimpleNamespace(
        quant_method=quant_method,
        _ensure_moe_quant_config_init=lambda: None,
    )
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(dp_size=4, pcp_size=1, is_sequence_parallel=False),
        routed_experts=routed_experts,
        gate=gate,
        _fse_fuse_gate=False,
        _shared_experts=None,
        _maybe_apply_shared_experts=lambda value, order: None,
        _sequence_parallel_context=lambda: nullcontext(),
    )

    hidden_states = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3)
    actual = forward_impl(runner, hidden_states, torch.empty(64, 2), None)

    assert not original_calls
    assert not dp_group.gather_args
    assert gate.input is hidden_states
    assert len(ep_group.dispatch_args) == 1
    assert ep_group.dispatch_args[0][0] is hidden_states
    assert quant_method.external_args["x"].shape == (256, 3)
    assert quant_method.external_args["router_logits"].shape == (256, 2)
    assert actual.shape == hidden_states.shape


def test_zero_thresholds_keep_upstream_forward(monkeypatch):
    runner_cls = moe_runner_mod.MoERunner

    def original_forward_impl(*args, **kwargs):
        return "original"

    monkeypatch.setattr(runner_cls, "_forward_impl", original_forward_impl)
    monkeypatch.setattr(envs, "TPU_MOE_GATHER_THEN_ROUTE_MAX_TOKENS", 0)
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 0)

    runner = SimpleNamespace(moe_config=SimpleNamespace(dp_size=4))
    actual = forward_impl(runner, torch.empty(8, 3), torch.empty(8, 2), None)
    assert actual == "original"
