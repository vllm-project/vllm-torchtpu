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
"""Which DSpark draft MLPs arm the fused op, and what an armed forward calls.
Stub linears; no TPU needed (the op itself is tests/kernels/test_dspark_mlp)."""

from types import SimpleNamespace

import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMulWithClamp

from vllm_torchtpu.models.minimax_m3 import dspark_mlp as bridge
from vllm_torchtpu.models.minimax_m3.moe import e4m3_mlp_operands
from vllm_torchtpu.models.minimax_m3.quant import MiniMaxM3Mxfp8LinearMethod

H, I_FF = 256, 128


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )


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
    linear.reduce_results = True
    return linear


def _mlp(limit=7.0, alpha=1.702, beta=1.0):
    mlp = bridge.MiniMaxM3DSparkMLP.__new__(bridge.MiniMaxM3DSparkMLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = _linear(H, 2 * I_FF)
    mlp.down_proj = _linear(I_FF, H)
    # CustomOp.__init__ needs a vLLM config; the admission reads only these.
    mlp.act_fn = SiluAndMulWithClamp.__new__(SiluAndMulWithClamp)
    nn.Module.__init__(mlp.act_fn)
    mlp.act_fn.swiglu_limit, mlp.act_fn.alpha, mlp.act_fn.beta = limit, alpha, beta
    return mlp


def test_operands_in_kernel_layout():
    mlp = _mlp()
    w1, s1, w2, s2 = e4m3_mlp_operands(mlp)
    assert w1.data_ptr() == mlp.gate_up_proj.weight.data_ptr()
    assert w2.data_ptr() == mlp.down_proj.weight.data_ptr()
    assert s1.dtype == s2.dtype == torch.float32
    assert tuple(s1.shape) == (1, 2 * I_FF) and tuple(s2.shape) == (1, H)
    assert torch.equal(s1, torch.full((1, 2 * I_FF), 0.25))


@pytest.mark.parametrize(
    "spoil",
    [
        lambda mlp: setattr(mlp, "act_fn", _mlp(limit=10.0).act_fn),
        lambda mlp: setattr(mlp, "act_fn", _mlp(alpha=1.0).act_fn),
        lambda mlp: setattr(mlp.down_proj, "bias", torch.zeros(H)),
        lambda mlp: setattr(
            mlp.gate_up_proj,
            "weight",
            nn.Parameter(torch.zeros(H, 2 * I_FF, dtype=torch.bfloat16)),
        ),
        lambda mlp: setattr(mlp, "down_proj", _linear(I_FF, H + 128)),
    ],
)
def test_other_mlps_are_refused(spoil):
    mlp = _mlp()
    spoil(mlp)
    assert e4m3_mlp_operands(mlp) is None
    assert bridge.arm_fused_mlp(nn.ModuleList([mlp])) == 0


@pytest.mark.parametrize("tp", [1, 8])
def test_armed_forward_calls_the_op_then_reduces(monkeypatch, tp):
    mlp, other = _mlp(), _mlp(beta=0.0)
    assert bridge.arm_fused_mlp(nn.ModuleList([mlp, other])) == 1
    calls = []

    def op(x, *operands):
        calls.append(operands)
        return x + 1

    monkeypatch.setattr(bridge, "_op", op)
    monkeypatch.setattr(bridge, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(bridge, "tensor_model_parallel_all_reduce", lambda t: t * 8)
    x = torch.zeros(8, H, dtype=torch.bfloat16)
    out = mlp(x)
    assert len(calls) == 1
    assert calls[0][0].data_ptr() == mlp.gate_up_proj.weight.data_ptr()
    assert torch.equal(out, torch.full_like(x, 8.0 if tp > 1 else 1.0))


def test_unarmed_all_reduce_keeps_the_separate_reduction(monkeypatch):
    """fold_all_reduce without the Pallas all-reduce armed: the MLP op is
    armed, its all-reduce stays outside."""
    monkeypatch.setattr(
        bridge,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=list(range(8)), world_size=8),
    )
    monkeypatch.setattr(bridge.pallas_all_reduce, "_ARMED", None)
    mlp = _mlp()
    assert bridge.arm_fused_mlp(nn.ModuleList([mlp]), fold_all_reduce=True) == 1
    assert mlp._reducing_op is None


def test_folded_all_reduce_replaces_the_reduction(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=list(range(8)), world_size=8, cpu_group=None),
    )
    armed = SimpleNamespace(rank=torch.tensor([[3]], dtype=torch.int32), mesh=object())
    monkeypatch.setattr(bridge.pallas_all_reduce, "armed_for", lambda ranks: armed)
    monkeypatch.setattr(
        bridge, "_make_reducing_op", lambda a, eps=None: lambda x, *o: x + 2
    )
    monkeypatch.setattr(bridge, "_selftest", lambda mlp, ops, rank: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda seen, obj, group: seen.__setitem__(slice(None), [obj] * len(seen)),
    )
    reduced = []
    monkeypatch.setattr(bridge, "tensor_model_parallel_all_reduce", reduced.append)
    mlp = _mlp()
    assert bridge.arm_fused_mlp(nn.ModuleList([mlp]), fold_all_reduce=True) == 1
    x = torch.zeros(8, H, dtype=torch.bfloat16)
    assert torch.equal(mlp(x), x + 2) and not reduced
    assert int(mlp._mesh_rank) == 3


def test_failed_selftest_keeps_the_separate_reduction(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=list(range(8)), world_size=8, cpu_group=None),
    )
    armed = SimpleNamespace(rank=torch.tensor([[3]], dtype=torch.int32), mesh=object())
    monkeypatch.setattr(bridge.pallas_all_reduce, "armed_for", lambda ranks: armed)
    monkeypatch.setattr(bridge, "_make_reducing_op", lambda a, eps=None: None)
    monkeypatch.setattr(bridge, "_selftest", lambda mlp, ops, rank: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda seen, obj, group: seen.__setitem__(slice(None), [True, False] * 4),
    )
    mlp = _mlp()
    assert bridge.arm_fused_mlp(nn.ModuleList([mlp]), fold_all_reduce=True) == 1
    assert mlp._reducing_op is None


def _gemma_norm(eps=1e-6):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    norm = GemmaRMSNorm.__new__(GemmaRMSNorm)
    nn.Module.__init__(norm)
    norm.weight = nn.Parameter(torch.full((H,), 0.5, dtype=torch.bfloat16))
    norm.variance_epsilon = eps
    return norm


def _layer(mlp, norm):
    layer = nn.Module()
    layer.mlp = mlp
    layer.post_attention_layernorm = norm
    return layer


@pytest.mark.parametrize("tp", [1, 8])
def test_norm_is_handed_to_the_op(monkeypatch, tp):
    built = []

    def make_norm_op(eps):
        built.append(eps)
        return lambda x, residual, norm_w, *o: (x + norm_w, x + residual)

    monkeypatch.setattr(bridge, "_make_norm_op", make_norm_op)
    monkeypatch.setattr(bridge, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(bridge, "tensor_model_parallel_all_reduce", lambda t: t * 8)
    mlp = _mlp()
    model = nn.ModuleList([_layer(mlp, _gemma_norm(1e-5))])
    assert bridge.arm_fused_mlp(model) == 1
    assert built == [1e-5]
    assert torch.equal(mlp._fused_norm_w, torch.full((1, H), 1.5))
    x = torch.ones(8, H, dtype=torch.bfloat16)
    out, residual = mlp.forward_with_norm(x, x * 2)
    assert torch.equal(residual, x * 3)
    assert torch.equal(out.float(), torch.full((8, H), 2.5 * (8 if tp > 1 else 1)))
    # A residual of another shape takes the unfused path.
    assert mlp.forward_with_norm(x, x[:4]) is None


def test_other_norms_keep_the_unfused_path(monkeypatch):
    monkeypatch.setattr(bridge, "_make_norm_op", lambda eps: pytest.fail("built"))
    mlps = [_mlp(), _mlp()]
    model = nn.ModuleList(
        [_layer(mlps[0], _gemma_norm(1e-6)), _layer(mlps[1], _gemma_norm(1e-5))]
    )
    assert bridge.arm_fused_mlp(model) == 2
    x = torch.ones(8, H, dtype=torch.bfloat16)
    assert all(m.forward_with_norm(x, x) is None for m in mlps)
