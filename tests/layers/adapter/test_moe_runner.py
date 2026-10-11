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
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.fused_moe_ep import FUSED_MOE_EP_OP_ATTR


class _QuantMethod:
    """Stands in for a TPU MoE method whose inherited hook reads unset state.

    VllmCompressedTensorsW4A16MoEMethod inherits an upstream
    get_fused_moe_quant_config that reads self.wna16_backend, which only the
    upstream __init__ it skips sets.
    """

    moe_quant_config = None

    def __init__(self, *, is_monolithic):
        self.is_monolithic = is_monolithic
        self.hook_calls = 0

    def get_fused_moe_quant_config(self, layer):
        self.hook_calls += 1
        if self.is_monolithic:
            raise AttributeError("'_QuantMethod' has no attribute 'wna16_backend'")
        return "built-config"


class _RoutedExperts:
    # The real upstream method, so the test fails if upstream stops routing
    # quant-config init through the quant method's hook.
    _ensure_moe_quant_config_init = RoutedExperts._ensure_moe_quant_config_init

    def __init__(self, quant_method):
        self.quant_method = quant_method


def _runner(quant_method):
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = _RoutedExperts(quant_method)
    runner._take_over_moe_quant_config_init()
    return runner


def test_upstream_constructs_the_tpu_runner():
    """Every layer built as MoERunner(...) gets VllmMoERunner."""
    assert isinstance(MoERunner.__new__(MoERunner), VllmMoERunner)


def test_upstream_hook_crashes_without_the_runner():
    """Pins the failure the takeover exists to prevent."""
    experts = _RoutedExperts(_QuantMethod(is_monolithic=True))
    with pytest.raises(AttributeError, match="wna16_backend"):
        experts._ensure_moe_quant_config_init()


def test_monolithic_method_hook_is_never_called():
    quant_method = _QuantMethod(is_monolithic=True)
    runner = _runner(quant_method)

    runner.routed_experts._ensure_moe_quant_config_init()

    assert quant_method.hook_calls == 0
    assert quant_method.moe_quant_config is None


def test_non_monolithic_method_still_builds_its_config():
    quant_method = _QuantMethod(is_monolithic=False)
    runner = _runner(quant_method)

    runner.routed_experts._ensure_moe_quant_config_init()

    assert quant_method.hook_calls == 1
    assert quant_method.moe_quant_config == "built-config"


def test_follows_a_replaced_quant_method():
    """Upstream can swap the quant method after construction."""
    runner = _runner(_QuantMethod(is_monolithic=False))
    replacement = _QuantMethod(is_monolithic=True)
    runner.routed_experts.quant_method = replacement

    runner.routed_experts._ensure_moe_quant_config_init()

    assert replacement.hook_calls == 0


def test_direct_forward_entry():
    """Upstream picks the direct entry only when is_tpu(), False under OOT."""
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)

    runner._shared_experts = None
    assert runner._select_forward() is moe_runner_mod._moe_forward

    runner._shared_experts = SimpleNamespace()
    assert runner._select_forward() is moe_runner_mod._moe_forward_shared


@pytest.mark.parametrize(
    ("fused_ep_kernel", "armed", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_fused_output_is_reduced_under_torch_compile(fused_ep_kernel, armed, expected):
    """The MoE forward is traced by dynamo with fullgraph=True, as vLLM
    compiles it. Read through super(), the upstream answer compiles to a
    truthy value and skips the routed output's TP all-reduce; eager tests,
    and a non-fullgraph compile that falls back to eager, cannot see that.
    """
    quant_method = SimpleNamespace(moe_kernel=None)
    if armed:
        setattr(quant_method, FUSED_MOE_EP_OP_ATTR, "op")
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(quant_method=quant_method)
    runner.moe_config = SimpleNamespace(skip_final_all_reduce=False)

    def reduce_unless_reduced(x):
        return x if runner._fused_output_is_reduced else x * 2

    x = torch.ones(2)
    with patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", fused_ep_kernel):
        torch._dynamo.reset()
        compiled = torch.compile(reduce_unless_reduced, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), x if expected else x * 2)


def _make_sp_runner(*, gate, is_sp=True, dp_size=1, pcp_size=1, naive=True):
    """Bare VllmMoERunner instance with the attributes _forward_impl touches."""
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    runner.gate = gate
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=is_sp, dp_size=dp_size, pcp_size=pcp_size
    )
    runner.routed_experts = Mock()
    runner.routed_experts.quant_method = SimpleNamespace(supports_internal_mk=not naive)
    runner._fse_fuse_gate = False
    runner._apply_quant_method = Mock(
        side_effect=lambda **kw: (None, kw["hidden_states"])
    )
    runner._maybe_combine = Mock(side_effect=lambda shared, hidden: hidden)
    runner._maybe_sync_shared_experts_stream = Mock()
    runner._shared_experts = None
    return runner


def _mock_ep_group(monkeypatch, world_size):
    from vllm.distributed import parallel_state

    ep_group = Mock(world_size=world_size)
    ep_group.all_gather.side_effect = lambda x, dim: torch.cat(
        [x + 100 * r for r in range(world_size)], dim=dim
    )
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: ep_group)
    return ep_group


def test_gate_runs_on_gathered_hidden_states(monkeypatch):
    """SP + runner-owned gate: gather hidden only, gate sees the full [T, H]."""
    from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import (
        _HAS_SYNC_SHARED_EXPERTS_STREAM,
    )

    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SP_GATE_AFTER_GATHER", True)
    ep_group = _mock_ep_group(monkeypatch, world_size=4)
    seen = {}

    def gate(x):
        seen["gate_input"] = x
        return x.sum(dim=-1, keepdim=True), None

    runner = _make_sp_runner(gate=gate)
    chunk = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    stale_logits = torch.full((2, 1), -1.0)

    out = runner._forward_impl(chunk, stale_logits, None)

    ep_group.all_gather.assert_called_once()
    assert ep_group.all_gather.call_args.args[0] is chunk
    gathered = torch.cat([chunk + 100 * r for r in range(4)])
    torch.testing.assert_close(seen["gate_input"], gathered)
    kwargs = runner._apply_quant_method.call_args.kwargs
    torch.testing.assert_close(kwargs["hidden_states"], gathered)
    torch.testing.assert_close(kwargs["router_logits"], gathered.sum(-1, keepdim=True))
    torch.testing.assert_close(out, gathered)
    if _HAS_SYNC_SHARED_EXPERTS_STREAM:
        runner._maybe_sync_shared_experts_stream.assert_called_once_with(None)
    runner.routed_experts._ensure_moe_quant_config_init.assert_called_once()


def test_fused_shared_expert_gate_uses_combined_weight(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SP_GATE_AFTER_GATHER", True)
    _mock_ep_group(monkeypatch, world_size=2)
    runner = _make_sp_runner(gate=Mock())
    runner._fse_fuse_gate = True
    runner._maybe_fuse_gate_weights = Mock()
    runner._combined_gate_weight = torch.eye(3)
    chunk = torch.ones(2, 3)

    runner._forward_impl(chunk, chunk, None)

    runner._maybe_fuse_gate_weights.assert_called_once()
    runner.gate.assert_not_called()
    logits = runner._apply_quant_method.call_args.kwargs["router_logits"]
    assert logits.shape == (4, 3)


def _spy_upstream(monkeypatch):
    upstream = Mock(name="upstream_forward_impl", return_value="upstream")
    monkeypatch.setattr(MoERunner, "_forward_impl", upstream)
    return upstream


@pytest.mark.parametrize(
    "case",
    [
        dict(gate=None),
        dict(is_sp=False),
        dict(pcp_size=2),
        dict(naive=False),
    ],
)
def test_falls_back_to_upstream_forward_impl(monkeypatch, case):
    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SP_GATE_AFTER_GATHER", True)
    ep_group = _mock_ep_group(monkeypatch, world_size=4)
    upstream = _spy_upstream(monkeypatch)
    kwargs = dict(gate=Mock(return_value=(torch.zeros(2, 1), None)))
    kwargs.update(case)
    runner = _make_sp_runner(**kwargs)
    hidden, logits = torch.ones(2, 3), torch.zeros(2, 1)

    out = runner._forward_impl(hidden, logits, None)

    assert out == "upstream"
    upstream.assert_called_once_with(hidden, logits, None, None)
    ep_group.all_gather.assert_not_called()
    runner._apply_quant_method.assert_not_called()


def test_env_flag_disables_gate_after_gather(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SP_GATE_AFTER_GATHER", False)
    ep_group = _mock_ep_group(monkeypatch, world_size=4)
    upstream = _spy_upstream(monkeypatch)
    runner = _make_sp_runner(gate=Mock(return_value=(torch.zeros(2, 1), None)))

    runner._forward_impl(torch.ones(2, 3), torch.zeros(2, 1), None)

    upstream.assert_called_once()
    ep_group.all_gather.assert_not_called()


def test_gate_runs_on_tp_gathered_hidden_states_when_ep_disabled(monkeypatch):
    """Under TP+SP (ep_size=1, tp_size>1), _forward_impl gathers over tp_group."""
    from vllm.distributed import parallel_state

    monkeypatch.setattr(envs, "VLLM_TPU_MOE_SP_GATE_AFTER_GATHER", True)
    _mock_ep_group(monkeypatch, world_size=1)
    tp_group = Mock(world_size=4)
    tp_group.all_gather.side_effect = lambda x, dim: torch.cat(
        [x + 10 * r for r in range(4)], dim=dim
    )
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)

    runner = _make_sp_runner(gate=lambda x: (x.sum(dim=-1, keepdim=True), None))
    chunk = torch.ones(2, 3)
    out = runner._forward_impl(chunk, torch.zeros(2, 1), None)

    tp_group.all_gather.assert_called_once()
    assert out.shape == (8, 3)
