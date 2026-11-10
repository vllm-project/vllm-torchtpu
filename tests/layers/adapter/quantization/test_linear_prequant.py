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
"""apply_prequant on the TPU FP8 and unquantized linear methods: the layer
applied to rows already quantized per row to FP8 with their scales, as the
Qwen3.5 model patch's sequence-parallel FP8 all-gather hands them over."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)

pytestmark = pytest.mark.cpu_test

_M, _K, _N = 8, 32, 16


def _rows(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(_M, _K, generator=g)
    scale = x.abs().amax(dim=1, keepdim=True) / 448.0
    x_q = (x / scale).to(torch.float8_e4m3fn)
    return x_q, scale.to(torch.float32)


@pytest.mark.parametrize("keep_vllm_layout", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp8_method_runs_the_prequant_matmul_on_the_kn_weight(
    monkeypatch, keep_vllm_layout, with_bias
):
    seen = {}

    def prequant(x_q, x_s, w_q, w_s):
        seen.update(x_q=x_q, x_s=x_s, w_q=w_q, w_s=w_s)
        return torch.ones(x_q.shape[0], w_q.shape[-1])

    monkeypatch.setattr(linear_common, "quantized_matmul_prequant", prequant)
    weight = torch.randn(_K, _N).to(torch.float8_e4m3fn)
    layer = SimpleNamespace(
        weight=weight.t() if keep_vllm_layout else weight,
        weight_scale=torch.rand(_N),
    )
    if keep_vllm_layout:
        setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    bias = torch.full((_N,), 2.0) if with_bias else None
    x_q, x_s = _rows()
    method = object.__new__(VllmFp8LinearMethodTPU)
    out = method.apply_prequant(layer, x_q, x_s, bias)
    assert seen["x_q"] is x_q and seen["x_s"] is x_s
    assert seen["w_s"] is layer.weight_scale
    # The canonical (k, n) weight, whichever layout the layer keeps.
    assert tuple(seen["w_q"].shape) == (_K, _N)
    torch.testing.assert_close(seen["w_q"].float(), weight.float())
    torch.testing.assert_close(out, torch.full((_M, _N), 3.0 if with_bias else 1.0))


@pytest.mark.parametrize("keep_vllm_layout", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_unquantized_method_matches_the_dequantized_rows(keep_vllm_layout, with_bias):
    g = torch.Generator().manual_seed(1)
    weight = torch.randn(_K, _N, generator=g).to(torch.bfloat16)
    layer = SimpleNamespace(
        weight=weight.t().contiguous() if keep_vllm_layout else weight
    )
    if keep_vllm_layout:
        setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    bias = torch.randn(_N, generator=g).to(torch.bfloat16) if with_bias else None
    x_q, x_s = _rows(seed=2)
    method = object.__new__(VllmUnquantizedLinearMethod)
    out = method.apply_prequant(layer, x_q, x_s, bias)
    want = (x_q.float() * x_s) @ weight.float()
    if bias is not None:
        want = want + bias.float()
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), want, rtol=2e-2, atol=2e-2)
