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
"""Correctness and single-FX-graph tests for TpuRMSNorm + arm_prequant."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu.layers.adapter import rmsnorm as tpu_rmsnorm  # noqa: E402
from vllm_torchtpu.layers.adapter.linear_common import PREQUANT_ATTR  # noqa: E402
from vllm_torchtpu.layers.adapter.quantization.fp8 import (  # noqa: E402
    VllmFp8LinearMethodTPU,
)

pytestmark = pytest.mark.tpu_test

H = 6144
N = 2624


def _norm():
    from vllm.model_executor.layers.layernorm import RMSNorm

    n = RMSNorm(H, eps=1e-5)
    assert isinstance(n, tpu_rmsnorm.TpuRMSNorm)
    with torch.no_grad():
        n.weight.copy_(1 + 0.2 * torch.randn(H))
    return n.to(torch.bfloat16).to("tpu")


class _Linear(torch.nn.Module):
    def __init__(self, w8a16_max_rows: int = 0):
        super().__init__()
        w = (64 * torch.randn(H, N)).to(torch.float8_e4m3fn)
        self.weight = torch.nn.Parameter(w.to("tpu"), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            (1e-4 + 9e-4 * torch.rand(N)).to("tpu"), requires_grad=False
        )
        self.bias = None
        self.quant_method = object.__new__(VllmFp8LinearMethodTPU)
        self.quant_method.w8a16_max_rows = w8a16_max_rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x)


def _layer(n: torch.nn.Module, lin: torch.nn.Module, x: torch.Tensor, r: torch.Tensor):
    y, res = n(x, r)
    return lin(y), res


@pytest.mark.parametrize("w8a16_max_rows", [0, 32])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [16, 64, 1024])
def test_paired_matches_unpaired(
    vllm_config_context, rows: int, compiled: bool, w8a16_max_rows: int
):
    torch.manual_seed(rows + w8a16_max_rows)
    torch._dynamo.reset()
    n = _norm()
    lin = _Linear(w8a16_max_rows=w8a16_max_rows)
    x = (0.7 * torch.randn(rows, H)).to(torch.bfloat16).to("tpu")
    r = (4.0 * torch.randn(rows, H)).to(torch.bfloat16).to("tpu")
    fn = torch.compile(_layer, fullgraph=True, dynamic=False) if compiled else _layer
    ref_out, ref_res = fn(n, lin, x.clone(), r.clone())
    assert tpu_rmsnorm._pair(n, lin, False)
    torch._dynamo.reset()
    fn = torch.compile(_layer, fullgraph=True, dynamic=False) if compiled else _layer
    got_out, got_res = fn(n, lin, x.clone(), r.clone())
    assert getattr(lin, PREQUANT_ATTR).value is None
    assert torch.equal(got_res.cpu(), ref_res.cpu())
    mismatch = int(
        (got_out.cpu().view(torch.int16) != ref_out.cpu().view(torch.int16)).sum()
    )
    assert mismatch == 0


def test_dynamic_single_fx_trace_across_all_buckets(vllm_config_context):
    """Verify zero token-count Python branches: one dynamic FX graph serves
    both decode (16, 64, 256) and prefill (512, 1024) buckets without recompile.
    """
    torch.manual_seed(7)
    torch._dynamo.reset()
    n, lin = _norm(), _Linear(w8a16_max_rows=0)
    assert tpu_rmsnorm._pair(n, lin, True)

    traces = []

    def counting_backend(gm, example_inputs):
        traces.append(len(gm.graph.nodes))
        return gm.forward

    compiled_fn = torch.compile(
        _layer, backend=counting_backend, fullgraph=True, dynamic=True
    )
    for rows in (1024, 512, 256, 64, 16):
        x = (0.7 * torch.randn(rows, H)).to(torch.bfloat16).to("tpu")
        r = (4.0 * torch.randn(rows, H)).to(torch.bfloat16).to("tpu")
        compiled_fn(n, lin, x, r)
        assert getattr(lin, PREQUANT_ATTR).value is None

    assert len(traces) == 1


def test_arm_prequant_pairs_moe_shared_experts(vllm_config_context):
    """Verify arm_prequant pairs MoE post_attention_layernorm with shared_experts."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    class _FakeMoERunner(MoERunner):
        def __init__(self, shared_mlp):
            torch.nn.Module.__init__(self)
            self._shared_experts = SimpleNamespace(_layer=shared_mlp)
            self._fse_fuse_gate = False

        @property
        def shared_experts(self):
            return self._shared_experts

    class _FakeMoELayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = _norm()
            self.post_attention_layernorm = _norm()
            self.self_attn = torch.nn.Module()
            self.mlp = torch.nn.Module()
            self.mlp.is_sequence_parallel = False
            self.mlp.shared_experts = torch.nn.Module()
            self.mlp.shared_experts.gate_up_proj = _Linear()
            self.mlp.experts = _FakeMoERunner(self.mlp.shared_experts)

    model = _FakeMoELayer()
    assert tpu_rmsnorm.arm_prequant(model) == 1
    assert model.post_attention_layernorm._prequant_slot is not None
    assert (
        getattr(model.mlp.shared_experts.gate_up_proj, PREQUANT_ATTR)
        is model.post_attention_layernorm._prequant_slot
    )
