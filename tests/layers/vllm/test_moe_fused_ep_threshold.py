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
from vllm.model_executor.layers.fused_moe.runner import \
    moe_runner as moe_runner_mod

import vllm_torchtpu.envs as envs
from vllm_torchtpu import _patch_moe_fused_ep_threshold_contract


class _FakeEpGroup:

    def __init__(self):
        self.dispatch_args = None
        self.combine_args = None

    def dispatch_router_logits(self, hidden_states, router_logits,
                               is_sequence_parallel):
        self.dispatch_args = (hidden_states, router_logits,
                              is_sequence_parallel)
        return hidden_states.repeat(4, 1), router_logits.repeat(4, 1)

    def combine(self, hidden_states, is_sequence_parallel):
        self.combine_args = (hidden_states, is_sequence_parallel)
        return hidden_states[:hidden_states.shape[0] // 4]


class _FakeQuantMethod:

    def __init__(self):
        self.external_args = None

    def apply_monolithic_external_dispatch(self, **kwargs):
        self.external_args = kwargs
        return kwargs["x"] + 1


def test_small_fused_ep_bucket_restores_external_dispatch(monkeypatch):
    runner_cls = moe_runner_mod.MoERunner
    original_calls = []

    def original_forward_impl(*args, **kwargs):
        original_calls.append((args, kwargs))
        return "internal"

    monkeypatch.setattr(runner_cls, "_forward_impl", original_forward_impl)
    monkeypatch.setattr(runner_cls, "_tpu_fused_ep_threshold_patch", False,
                        raising=False)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 256)

    ep_group = _FakeEpGroup()
    monkeypatch.setattr(moe_runner_mod, "get_ep_group", lambda: ep_group)
    _patch_moe_fused_ep_threshold_contract()

    quant_method = _FakeQuantMethod()
    routed_experts = SimpleNamespace(
        quant_method=quant_method,
        _ensure_moe_quant_config_init=lambda: None,
    )
    shared_orders = []
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(dp_size=4,
                                   is_sequence_parallel=False),
        routed_experts=routed_experts,
        gate=None,
        _shared_experts=None,
        _maybe_sync_shared_experts_stream=lambda value: None,
        _maybe_apply_shared_experts=lambda value, order: shared_orders.append(
            order),
        _sequence_parallel_context=lambda: nullcontext(),
    )

    hidden_states = torch.arange(96, dtype=torch.float32).reshape(32, 3)
    router_logits = torch.arange(32 * 8,
                                 dtype=torch.float32).reshape(32, 8)
    actual = runner_cls._forward_impl(runner, hidden_states, router_logits,
                                      None)

    assert not original_calls
    assert ep_group.dispatch_args[0] is hidden_states
    assert ep_group.dispatch_args[1] is router_logits
    assert ep_group.dispatch_args[2] is False
    assert quant_method.external_args["x"].shape == (128, 3)
    assert quant_method.external_args["router_logits"].shape == (128, 8)
    assert ep_group.combine_args[0].shape == (128, 3)
    assert ep_group.combine_args[1] is False
    assert actual.shape == hidden_states.shape
    assert shared_orders == [
        moe_runner_mod.SharedExpertsOrder.NO_OVERLAP,
        moe_runner_mod.SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
    ]

    large_hidden = torch.empty(256, 3)
    large_logits = torch.empty(256, 8)
    assert runner_cls._forward_impl(runner, large_hidden, large_logits,
                                    None) == "internal"
    assert len(original_calls) == 1
