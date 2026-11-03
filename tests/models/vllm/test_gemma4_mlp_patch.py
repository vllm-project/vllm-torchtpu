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
"""Gemma4 GeGLU with its down projection's FP8 input in one pass: the patch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.models.gemma4 import Gemma4MLP

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.vllm import gemma4_mlp_patch as patch


class _Method:
    """Stands in for VllmFp8LinearMethodTPU's quantized-input contract."""

    def __init__(self, supported: bool = True, w8a16_max_rows: int = 0) -> None:
        self.supported = supported
        self.w8a16_max_rows = w8a16_max_rows
        self.calls: list[tuple] = []

    def supports_quantized_input(self, layer, rows):
        return self.supported

    def apply_quantized_input(self, layer, x_q, x_scale):
        self.calls.append((layer, x_q, x_scale))
        return x_q.to(torch.float32)[:, :3] * x_scale


class _Down:
    def __init__(self, method, reduce_results=False):
        self.quant_method = method
        self.weight = torch.zeros(4, 3, dtype=torch.float8_e4m3fn)
        self.reduce_results = reduce_results
        self.tp_size = 2
        self.calls: list[str] = []
        self.all_reduce = lambda t: t

    def __call__(self, x):
        # RowParallelLinear's forward: the matmul, then its own reduction.
        self.calls.append("down")
        out = x * 2
        if self.reduce_results and self.tp_size > 1:
            out = self.all_reduce(out)
        return out, None


def _mlp(reduce_results=False, supported=True, w8a16_max_rows=0):
    method = _Method(supported, w8a16_max_rows)
    calls = []
    mlp = SimpleNamespace(
        down_proj=_Down(method, reduce_results),
        gate_up_proj=None,
        act_fn=lambda x: calls.append("act") or x + 1,
    )
    return mlp, method


@pytest.mark.parametrize("flag,w8a16_max_rows", [(False, 0), (True, 16)])
def test_unfused_ops_without_the_flag_or_under_the_w8a16_cutoff(
    monkeypatch, flag, w8a16_max_rows
):
    mlp, method = _mlp(w8a16_max_rows=w8a16_max_rows)
    monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_GELU_QUANT", flag)
    out = patch.down_proj_from_gate_up(mlp, torch.ones(16, 8))
    assert mlp.down_proj.calls == ["down"] and not method.calls
    torch.testing.assert_close(out, torch.full((16, 8), 4.0))


def test_fused_path_quantizes_once_and_runs_the_prequantized_matmul(monkeypatch):
    mlp, method = _mlp()
    gate_up = torch.randn(16, 8).to(torch.bfloat16)
    x_q = torch.ones(16, 4, dtype=torch.float8_e4m3fn)
    x_scale = torch.full((16, 1), 0.5)
    seen = []

    def fused(x, w_q):
        seen.append((x, w_q))
        return x_q, x_scale

    monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_GELU_QUANT", True)
    monkeypatch.setattr(patch, "gelu_tanh_and_mul_quantize", fused)
    out = patch.down_proj_from_gate_up(mlp, gate_up)
    ((x, w_q),) = seen
    assert x is gate_up and w_q is mlp.down_proj.weight
    ((layer, got_q, got_scale),) = method.calls
    assert layer is mlp.down_proj and got_q is x_q and got_scale is x_scale
    torch.testing.assert_close(out, torch.full((16, 3), 0.5))


def test_fused_path_refuses_an_unsupported_down_projection(monkeypatch):
    mlp, _ = _mlp(supported=False)
    monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_GELU_QUANT", True)
    with pytest.raises(NotImplementedError, match="per-channel"):
        patch.down_proj_from_gate_up(mlp, torch.randn(16, 8).to(torch.bfloat16))


@pytest.mark.parametrize("w8a16_max_rows", [0, 16])
@pytest.mark.parametrize("reduce_results", [False, True])
def test_forward_reduces_once_where_the_down_projection_would(
    monkeypatch, reduce_results, w8a16_max_rows
):
    """Fused, or unfused under the W8A16 cutoff (where the down projection's
    forward reduces by itself), the output is reduced once if at all."""
    mlp, _ = _mlp(reduce_results=reduce_results, w8a16_max_rows=w8a16_max_rows)
    mlp.gate_up_proj = lambda x: (x, None)
    reduced = []

    def all_reduce(t):
        reduced.append(t)
        return t * 10

    mlp.down_proj.all_reduce = all_reduce
    monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_GELU_QUANT", True)
    monkeypatch.setattr(
        patch,
        "gelu_tanh_and_mul_quantize",
        lambda x, w: (
            torch.ones(x.shape[0], 4, dtype=w.dtype),
            torch.ones(x.shape[0], 1),
        ),
    )
    monkeypatch.setattr(patch, "tensor_model_parallel_all_reduce", all_reduce)
    x = torch.randn(16, 8).to(torch.bfloat16)
    out = patch._patched_mlp_forward(mlp, x)
    want = (x + 1) * 2 if w8a16_max_rows else torch.ones(16, 3)
    assert len(reduced) == int(reduce_results)
    torch.testing.assert_close(out, want * 10 if reduce_results else want)


def test_patch_needs_the_flag_and_a_gemma4_config(monkeypatch):
    original, original_init = Gemma4MLP.forward, Gemma4MLP.__init__
    gemma4 = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="gemma4_text"))
    other = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="llama"))
    try:
        monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_GELU_QUANT", False)
        assert patch.maybe_patch_gemma4_fused_gelu_quant(gemma4) is None
        assert Gemma4MLP.forward is original
        monkeypatch.setattr(envs, "TPU_GEMMA4_FUSED_GELU_QUANT", True)
        assert patch.maybe_patch_gemma4_fused_gelu_quant(None) is False
        assert patch.maybe_patch_gemma4_fused_gelu_quant(other) is False
        assert Gemma4MLP.forward is original
        assert patch.maybe_patch_gemma4_fused_gelu_quant(gemma4) is None
        assert Gemma4MLP.forward is patch._patched_mlp_forward
        assert Gemma4MLP.__init__ is patch._patched_mlp_init
    finally:
        Gemma4MLP.forward = original
        Gemma4MLP.__init__ = original_init


def _gelu_and_mul(approximate: str) -> GeluAndMul:
    # Built without CustomOp.__init__, which needs a resolved vLLM config.
    act = object.__new__(GeluAndMul)
    act.approximate = approximate
    return act


def _fake_init(act, method, intermediate=10752):
    def init(self, *args, **kwargs):
        self.act_fn = act
        self.down_proj = SimpleNamespace(
            quant_method=method, input_size_per_partition=intermediate
        )

    return init


def test_init_refuses_another_activation(monkeypatch):
    method = object.__new__(VllmFp8LinearMethodTPU)
    monkeypatch.setattr(
        patch, "_orig_mlp_init", _fake_init(_gelu_and_mul("none"), method)
    )
    with pytest.raises(NotImplementedError, match="tanh"):
        patch._patched_mlp_init(SimpleNamespace())


def test_init_refuses_a_non_fp8_down_projection(monkeypatch):
    monkeypatch.setattr(
        patch, "_orig_mlp_init", _fake_init(_gelu_and_mul("tanh"), object())
    )
    with pytest.raises(NotImplementedError, match="FP8"):
        patch._patched_mlp_init(SimpleNamespace())


def test_init_accepts_the_served_configuration(monkeypatch):
    method = object.__new__(VllmFp8LinearMethodTPU)
    monkeypatch.setattr(
        patch, "_orig_mlp_init", _fake_init(_gelu_and_mul("tanh"), method)
    )
    patch._patched_mlp_init(SimpleNamespace())


def test_init_refuses_a_partial_lane_intermediate_size(monkeypatch):
    method = object.__new__(VllmFp8LinearMethodTPU)
    monkeypatch.setattr(
        patch,
        "_orig_mlp_init",
        _fake_init(_gelu_and_mul("tanh"), method, intermediate=1056),
    )
    with pytest.raises(NotImplementedError, match="multiple of 128"):
        patch._patched_mlp_init(SimpleNamespace())
