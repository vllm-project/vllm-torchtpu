# Copyright 2025 Google LLC
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
"""
Tests for the TPU unquantized dense-linear method.

`VllmUnquantizedLinearMethod` stores the weight in the canonical (k, n)
layout `[n_in, n_out]` so the runtime matmul is `(m, k) @ (k, n)` instead of
vLLM's `(m, k) @ (n, k).T`. This is the default path for every unquantized
(bf16/fp16) model, so these tests pin both the layout and the requirement
that results match `F.linear`: bit-identical without a bias, and with one up
to where the bias is rounded in (see `_assert_matches_f_linear_with_bias`).
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)

pytestmark = pytest.mark.tpu_test

SHAPES = [(7, 64, 32), (128, 256, 512), (1, 128, 64)]


def _make_layer(w_nmajor: torch.Tensor) -> torch.nn.Module:
    """A layer holding vLLM's as-loaded `[n_out, n_in]` weight."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_nmajor.clone(), requires_grad=False)
    return layer


def _assert_matches_f_linear_with_bias(
    out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor
) -> None:
    """apply() matches F.linear up to where the bias is rounded in.

    On torch_tpu, F.linear rounds x @ w.T to the activation dtype and then adds
    the bias. torch.addmm, which apply() uses, did the same up to torch_tpu
    0.2.x; since 0.3.0 it adds the bias first and rounds once, which is more
    accurate. The two differ by at most one bf16 rounding step of the product,
    which this allows; a wrong layout or a dropped bias is off by far more.
    """
    ref = F.linear(x, w, bias)
    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    product = (x.float() @ w.float().transpose(0, 1)).abs()
    one_step = (product + ref.float().abs()) * 2.0**-7
    assert ((out.float() - ref.float()).abs() <= one_step).all()


class TestUnquantizedLinearLayout:
    """process_weights_after_loading must produce the (k, n) layout."""

    @pytest.mark.parametrize("m,k,n", SHAPES)
    def test_weight_is_transposed_to_kn(self, m, k, n, device):
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        layer = _make_layer(w)
        VllmUnquantizedLinearMethod().process_weights_after_loading(layer)
        # [n_out, n_in] -> [n_in, n_out]
        assert tuple(layer.weight.shape) == (k, n)

    def test_transpose_is_idempotent(self, device):
        """A second call must not transpose the weight back."""
        w = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
        layer = _make_layer(w)
        method = VllmUnquantizedLinearMethod()
        method.process_weights_after_loading(layer)
        first = tuple(layer.weight.shape)
        method.process_weights_after_loading(layer)
        assert tuple(layer.weight.shape) == first == (64, 32)

    def test_values_are_preserved(self, device):
        """The transpose must move data, not reinterpret it."""
        w = torch.randn(8, 16, dtype=torch.float32, device=device)
        layer = _make_layer(w)
        VllmUnquantizedLinearMethod().process_weights_after_loading(layer)
        assert torch.equal(layer.weight.data, w.transpose(0, 1))

    def test_non_2d_weight_is_left_alone(self, device):
        """Only plain dense weights are converted."""
        w = torch.randn(2, 8, 16, dtype=torch.bfloat16, device=device)
        layer = _make_layer(w)
        VllmUnquantizedLinearMethod().process_weights_after_loading(layer)
        assert tuple(layer.weight.shape) == (2, 8, 16)


class TestUnquantizedLinearNumerics:
    """apply() must match vLLM's stock F.linear path."""

    @pytest.mark.parametrize("m,k,n", SHAPES)
    def test_matches_f_linear_without_bias(self, m, k, n, device):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        layer = _make_layer(w)
        method = VllmUnquantizedLinearMethod()
        method.process_weights_after_loading(layer)
        assert torch.equal(F.linear(x, w), method.apply(layer, x, None))

    @pytest.mark.parametrize("m,k,n", SHAPES)
    def test_matches_f_linear_with_bias(self, m, k, n, device):
        """The bias is added to every output row."""
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        bias = torch.randn(n, dtype=torch.bfloat16, device=device)
        layer = _make_layer(w)
        method = VllmUnquantizedLinearMethod()
        method.process_weights_after_loading(layer)
        _assert_matches_f_linear_with_bias(method.apply(layer, x, bias), x, w, bias)

    def test_matches_f_linear_for_3d_input(self, device):
        """Leading dims are flattened for the matmul and restored after."""
        b, m, k, n = 2, 5, 64, 32
        x = torch.randn(b, m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        bias = torch.randn(n, dtype=torch.bfloat16, device=device)
        layer = _make_layer(w)
        method = VllmUnquantizedLinearMethod()
        method.process_weights_after_loading(layer)
        out = method.apply(layer, x, bias)
        assert tuple(out.shape) == (b, m, n)
        _assert_matches_f_linear_with_bias(out, x, w, bias)
