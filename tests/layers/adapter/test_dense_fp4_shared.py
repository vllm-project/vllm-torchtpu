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
"""Which shared experts dense_fp4_shared hands to the routed op, and what
arming changes on the runner. Stub runners and linears; no TPU needed."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu.layers.adapter import dense_fp4_shared as bridge
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP
from vllm_torchtpu.models.minimax_m3.quant import MiniMaxM3Mxfp8LinearMethod

H, I_S = 256, 128


def _linear(k, n):
    linear = nn.Module()
    linear.quant_method = MiniMaxM3Mxfp8LinearMethod(keep_fp8=True)
    linear.weight = nn.Parameter(
        torch.randn(k, n).to(torch.float8_e4m3fn), requires_grad=False
    )
    linear.weight_scale = nn.Parameter(
        torch.full((n,), 0.25, dtype=torch.bfloat16), requires_grad=False
    )
    linear.bias = None
    return linear


def _mlp(limit=7.0, alpha=1.702, beta=1.0):
    mlp = MiniMaxM3MLP.__new__(MiniMaxM3MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = _linear(H, 2 * I_S)
    mlp.down_proj = _linear(I_S, H)
    # The attributes CustomOp.__init__ would set need a vLLM config; the
    # admission reads only these.
    mlp.act_fn = SiluAndMulWithClamp.__new__(SiluAndMulWithClamp)
    nn.Module.__init__(mlp.act_fn)
    mlp.act_fn.swiglu_limit, mlp.act_fn.alpha, mlp.act_fn.beta = limit, alpha, beta
    return mlp


class _Runner(MoERunner):
    # Shadows MoERunner's read-only property.
    _quant_method = SimpleNamespace(_tpu_activation_str="swigluoai")

    def __init__(self, mlp):
        nn.Module.__init__(self)
        self._shared_experts = None if mlp is None else _Shared(mlp)
        self.routed_experts = nn.Module()
        self.moe_config = SimpleNamespace(experts_per_token=4)
        self._forward_entry = self._select_forward()

    def _select_forward(self):
        return "shared" if self._shared_experts is not None else "routed"


class _Shared:
    def __init__(self, mlp):
        self._layer = mlp


def test_shared_tensors_in_kernel_layout():
    mlp = _mlp()
    w1, s1, w2, s2 = bridge._shared_tensors(_Runner(mlp))
    assert w1.data_ptr() == mlp.gate_up_proj.weight.data_ptr()
    assert w2.data_ptr() == mlp.down_proj.weight.data_ptr()
    assert s1.dtype == s2.dtype == torch.float32
    assert tuple(s1.shape) == (1, 2 * I_S) and tuple(s2.shape) == (1, H)
    assert torch.equal(s1, torch.full((1, 2 * I_S), 0.25))


@pytest.mark.parametrize(
    "spoil",
    [
        lambda mlp: setattr(mlp, "act_fn", _mlp(limit=10.0).act_fn),
        lambda mlp: setattr(mlp, "act_fn", _mlp(beta=0.0).act_fn),
        lambda mlp: setattr(mlp.down_proj, "bias", torch.zeros(H)),
        lambda mlp: setattr(
            mlp.gate_up_proj,
            "weight_scale",
            nn.Parameter(torch.ones(2 * I_S, H // 32), requires_grad=False),
        ),
        lambda mlp: setattr(
            mlp.gate_up_proj,
            "weight",
            nn.Parameter(torch.zeros(H, 2 * I_S, dtype=torch.bfloat16)),
        ),
    ],
)
def test_other_shared_experts_are_refused(spoil):
    mlp = _mlp()
    spoil(mlp)
    assert bridge._shared_tensors(_Runner(mlp)) is None


def test_no_shared_expert_is_refused():
    assert bridge._shared_tensors(_Runner(None)) is None


def test_arming_moves_the_shared_expert_into_the_routed_op(monkeypatch):
    from vllm_torchtpu.layers.adapter import fused_moe

    built = []
    monkeypatch.setattr(
        fused_moe, "prebuild_fused_moe_kernel", lambda **kw: built.append(kw)
    )
    armed, refused = _Runner(_mlp()), _Runner(_mlp())
    model = nn.ModuleList([armed, refused])
    monkeypatch.setattr(
        bridge, "_admit", lambda r: bridge._shared_tensors(r) if r is armed else None
    )
    assert bridge.shared_operands(armed.routed_experts) is None
    assert bridge.arm_dense_fp4_shared(model) == 1
    operands = bridge.shared_operands(armed.routed_experts)
    assert operands is not None and len(operands) == 4
    assert armed._shared_experts is None and armed._forward_entry == "routed"
    assert built == [
        dict(topk=4, activation="swigluoai", use_ep=True, with_shared=True)
    ]
    assert refused._forward_entry == "shared"
    assert bridge.shared_operands(refused.routed_experts) is None
