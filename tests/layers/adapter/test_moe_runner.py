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
from unittest.mock import patch

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


_DP, _TP, _ROWS, _HIDDEN = 2, 4, 3, 8


class _Group(SimpleNamespace):
    """A process group seen from one rank; all_reduce returns a fixed sum."""

    def all_reduce(self, x):
        self.inputs.append(x)
        return self.total if self.total is not None else x


def _combine_runner(
    *,
    shared=True,
    internal_mk=False,
    sequence_parallel=False,
    pcp_size=1,
    scale=1.0,
):
    runner = VllmMoERunner.__new__(VllmMoERunner)
    torch.nn.Module.__init__(runner)
    quant_method = SimpleNamespace(
        moe_kernel=None,
        supports_internal_mk=internal_mk,
        skip_forward_padding=False,
        has_unpadded_output=False,
        is_monolithic=True,
    )
    runner.routed_experts = SimpleNamespace(quant_method=quant_method)
    runner.moe_config = SimpleNamespace(
        dp_size=_DP,
        is_sequence_parallel=sequence_parallel,
        pcp_size=pcp_size,
        skip_final_all_reduce=False,
        tp_size=1,
        ep_size=_DP * _TP,
        hidden_dim=_HIDDEN,
        hidden_dim_unpadded=_HIDDEN,
    )
    runner._shared_experts = SimpleNamespace() if shared else None
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    runner.routed_scaling_factor = scale
    runner.router = SimpleNamespace()
    runner.layer_name = "model.layers.0.mlp.experts"
    return runner


def _patch_groups(dp_rank, ep_total=None, ep_size=_DP * _TP):
    dp = SimpleNamespace(world_size=_DP, rank_in_group=dp_rank)
    tp = SimpleNamespace(world_size=_TP)
    ep = _Group(world_size=ep_size, inputs=[], total=ep_total)
    mod = "vllm_torchtpu.layers.adapter.custom_ops.moe_runner"
    return ep, (
        patch(f"{mod}.get_dp_group", lambda: dp),
        patch(f"{mod}.get_tp_group", lambda: tp),
        patch(f"{mod}.get_ep_group", lambda: ep),
    )


def _rank_inputs():
    """Per rank (d, t): routed partials for the whole DP group's tokens and
    the TP-partial shared expert output for its own tokens, in float64 so
    the two reduction orders agree to rounding."""
    gen = torch.Generator().manual_seed(0)
    routed = torch.randn(_DP, _TP, _DP * _ROWS, _HIDDEN, generator=gen).double()
    shared = torch.randn(_DP, _TP, _ROWS, _HIDDEN, generator=gen).double()
    return routed, shared


def _two_step_reference(routed, shared, d):
    """DP reduce-scatter, + shared, TP all-reduce: what rank (d, *) holds."""
    block = slice(d * _ROWS, (d + 1) * _ROWS)
    return sum(routed[:, t, block].sum(0) + shared[d, t] for t in range(_TP))


@pytest.mark.parametrize("shared", [True, False])
def test_single_ep_combine_matches_two_step_combine(monkeypatch, shared):
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_COMBINE", True, raising=False)
    routed, shared_out = _rank_inputs()
    if not shared:
        shared_out = torch.zeros_like(shared_out)
    runner = _combine_runner(shared=shared)

    # Pass 1: what each rank hands to the EP all-reduce.
    sent = {}
    for d in range(_DP):
        for t in range(_TP):
            ep, patches = _patch_groups(d)
            with patches[0], patches[1], patches[2]:
                runner._maybe_combine(
                    shared_out[d, t] if shared else None, routed[d, t]
                )
            (sent[d, t],) = ep.inputs
    total = sum(sent.values())

    # Pass 2: every rank receives the sum and keeps its own tokens' rows.
    for d in range(_DP):
        for t in range(_TP):
            ep, patches = _patch_groups(d, ep_total=total)
            with patches[0], patches[1], patches[2]:
                result = runner._maybe_combine(
                    shared_out[d, t] if shared else None, routed[d, t]
                )
            if shared:
                shared_result, result = result
                assert shared_result is None
            torch.testing.assert_close(
                result, _two_step_reference(routed, shared_out, d)
            )


def test_single_ep_combine_forward_adds_nothing_more(monkeypatch):
    """forward() must neither add the shared output again nor all-reduce."""
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_COMBINE", True, raising=False)
    routed, shared_out = _rank_inputs()
    runner = _combine_runner()
    d, t = 1, 2
    total = sum(
        routed[dd, tt]
        + torch.nn.functional.pad(
            shared_out[dd, tt],
            (0, 0, dd * _ROWS, (_DP - dd - 1) * _ROWS),
        )
        for dd in range(_DP)
        for tt in range(_TP)
    )
    runner._forward_entry = lambda *args: runner._maybe_combine(
        shared_out[d, t], routed[d, t]
    )

    def no_all_reduce(x):
        raise AssertionError("the final TP all-reduce must be skipped")

    ep, patches = _patch_groups(d, ep_total=total)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch.object(moe_runner_mod, "tensor_model_parallel_all_reduce", no_all_reduce),
    ):
        out = runner.forward(
            torch.zeros(_ROWS, _HIDDEN, dtype=torch.float64),
            torch.zeros(_ROWS, 4, dtype=torch.float64),
        )
    torch.testing.assert_close(out, _two_step_reference(routed, shared_out, d))
    assert len(ep.inputs) == 1


@pytest.mark.parametrize(
    ("knob", "kwargs", "ep_size", "armed"),
    [
        (True, {}, _DP * _TP, True),
        (False, {}, _DP * _TP, False),
        (True, {"internal_mk": True}, _DP * _TP, False),
        (True, {"sequence_parallel": True}, _DP * _TP, False),
        (True, {"pcp_size": 2}, _DP * _TP, False),
        (True, {"scale": 2.5}, _DP * _TP, False),
        (True, {}, _DP * _TP * 2, False),
    ],
)
def test_single_ep_combine_arming(monkeypatch, knob, kwargs, ep_size, armed):
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_COMBINE", knob, raising=False)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", False, raising=False)
    runner = _combine_runner(**kwargs)
    _, patches = _patch_groups(0, ep_size=ep_size)
    with patches[0], patches[1], patches[2]:
        assert runner._single_ep_combine_armed() is armed
        assert runner._fused_output_is_reduced is armed


def test_single_ep_combine_off_keeps_upstream_combine(monkeypatch):
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_COMBINE", False, raising=False)
    runner = _combine_runner()
    sentinel = object()
    with patch.object(MoERunner, "_maybe_combine", return_value=sentinel) as up:
        assert runner._maybe_combine("shared", "routed") is sentinel
    up.assert_called_once_with("shared", "routed")


def test_single_ep_combine_armed_under_torch_compile(monkeypatch):
    """forward() reads _fused_output_is_reduced inside the dynamo trace."""
    monkeypatch.setattr(envs, "TPU_MOE_FUSED_COMBINE", True, raising=False)
    runner = _combine_runner()
    _, patches = _patch_groups(0)

    def reduce_unless_reduced(x):
        return x if runner._fused_output_is_reduced else x * 2

    x = torch.ones(2)
    with patches[0], patches[1], patches[2]:
        torch._dynamo.reset()
        compiled = torch.compile(reduce_unless_reduced, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), x)
