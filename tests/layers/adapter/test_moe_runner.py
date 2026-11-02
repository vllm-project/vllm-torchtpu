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
