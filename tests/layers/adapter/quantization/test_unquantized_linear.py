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
(bf16/fp16) model, so these tests pin both the layout and the numerics: no
bias matches `F.linear` exactly, and a bias is added to the float32
accumulator before a single rounding.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)

SHAPES = [(7, 64, 32), (128, 256, 512), (1, 128, 64)]


def _single_rounding_ref(x, w_nmajor, bias):
    """`x @ w.T + bias` in float32, rounded once to the activation dtype."""
    return F.linear(x.float(), w_nmajor.float(), bias.float()).to(x.dtype)


def _assert_rounded_once(out, ref):
    """`out` matches the single-rounding reference up to summation order.

    Rounding the product to bf16 before adding the bias changes about a
    quarter of the outputs; a different float32 summation order changes
    only the rare ones that sit on a rounding boundary, by at most one ulp.
    """
    assert out.dtype == ref.dtype and out.shape == ref.shape
    mismatch = out != ref
    assert mismatch.float().mean().item() < 0.01
    ulp = torch.finfo(ref.dtype).eps * ref.float().abs()
    diff = (out.float() - ref.float()).abs()
    assert bool((diff[mismatch] <= ulp[mismatch]).all())


def _make_layer(w_nmajor: torch.Tensor) -> torch.nn.Module:
    """A layer holding vLLM's as-loaded `[n_out, n_in]` weight."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_nmajor.clone(), requires_grad=False)
    return layer


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
    """apply() numerics for the bias-free and bias paths."""

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
        """Bias must be added to the float32 accumulator.

        On torch_tpu, `F.linear`, `addmm` and `matmul(x, w) + bias` all round
        the product to the activation dtype before adding the bias, so the
        reference is the float32 result rounded once.
        """
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
        bias = torch.randn(n, dtype=torch.bfloat16, device=device)
        layer = _make_layer(w)
        method = VllmUnquantizedLinearMethod()
        method.process_weights_after_loading(layer)
        _assert_rounded_once(
            method.apply(layer, x, bias), _single_rounding_ref(x, w, bias)
        )

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
        _assert_rounded_once(out, _single_rounding_ref(x, w, bias))
