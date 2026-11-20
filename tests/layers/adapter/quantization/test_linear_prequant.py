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

from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    PrequantLinearMethod,
)
from vllm_torchtpu.layers.adapter.quantization import fp8
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

    monkeypatch.setattr(fp8, "quantized_matmul_prequant", prequant)
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
    method.w8a16_max_rows = 0
    out = method.apply_prequant(layer, x_q, x_s, bias)
    assert seen["x_q"] is x_q and seen["x_s"] is x_s
    assert seen["w_s"] is layer.weight_scale
    # The canonical (k, n) weight, whichever layout the layer keeps.
    assert tuple(seen["w_q"].shape) == (_K, _N)
    torch.testing.assert_close(seen["w_q"].float(), weight.float())
    torch.testing.assert_close(out, torch.full((_M, _N), 3.0 if with_bias else 1.0))


@pytest.mark.parametrize(
    "weight_dtype, scale_shape, w8a16_max_rows",
    [
        (torch.float8_e4m3fn, (1, _K // 16, 1, _N), 0),
        (torch.int8, (_N,), 0),
        (torch.float8_e5m2, (_N,), 0),
        (torch.float8_e4m3fn, (_N,), 64),
    ],
    ids=["block_scales", "int8_weight", "e5m2_weight", "w8a16_small_steps"],
)
@pytest.mark.parametrize("keep_vllm_layout", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp8_method_runs_apply_on_the_dequantized_rows_for_other_weights(
    monkeypatch, weight_dtype, scale_shape, w8a16_max_rows, keep_vllm_layout, with_bias
):
    """Only a float8_e4m3fn weight with per-channel scales, on a method without
    W8A16 small steps, takes the pre-quantized matmul. A block-scaled weight
    (REQUANTIZE_BLOCK_SIZE gives a rank-4 scale), an int8 or float8_e5m2
    weight (REQUANTIZE_WEIGHT_DTYPE), or a method that runs small steps W8A16
    (TPU_FP8_LINEAR_W8A16_MAX_ROWS) runs the method's own apply on the
    dequantized rows instead."""
    seen = {}

    def apply(self, layer, x, bias=None):
        seen.update(layer=layer, x=x, bias=bias)
        return torch.full((x.shape[0], _N), 7.0)

    def prequant(*args):
        raise AssertionError("the pre-quantized matmul must not run")

    monkeypatch.setattr(VllmFp8LinearMethodTPU, "apply", apply)
    monkeypatch.setattr(fp8, "quantized_matmul_prequant", prequant)
    weight = (torch.randn(_K, _N) * 4).to(weight_dtype)
    layer = SimpleNamespace(
        weight=weight.t() if keep_vllm_layout else weight,
        weight_scale=torch.rand(*scale_shape),
    )
    if keep_vllm_layout:
        setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    bias = torch.full((_N,), 2.0) if with_bias else None
    x_q, x_s = _rows()
    method = object.__new__(VllmFp8LinearMethodTPU)
    method.w8a16_max_rows = w8a16_max_rows
    out = method.apply_prequant(layer, x_q, x_s, bias)
    assert seen["layer"] is layer and seen["bias"] is bias
    assert seen["x"].dtype == torch.bfloat16
    torch.testing.assert_close(
        seen["x"], (x_q.to(torch.float32) * x_s).to(torch.bfloat16), rtol=0, atol=0
    )
    torch.testing.assert_close(out, torch.full((_M, _N), 7.0))


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


def test_only_a_method_that_says_so_serves_pre_quantized_rows():
    """A bf16 `apply` does not see FP8-rounded rows, so the unquantized method
    (and any method that does not say otherwise) does not serve them; the TPU
    FP8 method does, for an e4m3 weight with per-channel scales."""
    layer = SimpleNamespace(
        weight=torch.randn(_K, _N).to(torch.float8_e4m3fn), weight_scale=torch.rand(_N)
    )
    assert not PrequantLinearMethod().serves_prequant_rows(layer)
    unquantized = object.__new__(VllmUnquantizedLinearMethod)
    assert not unquantized.serves_prequant_rows(
        SimpleNamespace(weight=torch.randn(_K, _N))
    )
    fp8_method = object.__new__(VllmFp8LinearMethodTPU)
    fp8_method.w8a16_max_rows = 0
    assert fp8_method.serves_prequant_rows(layer)
