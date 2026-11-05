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
"""Tests for pre-quantized FP8 linear inputs and fused GeGLU quantization op."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
    xla_quantized_matmul,
)
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    _prequantized_matmul_bf16_jax,
    _quantize_activation_jax,
    _quantized_matmul_jax,
    gelu_and_mul_quantize_op,
    prequantized_matmul,
    quantize_activation,
)
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

_K, _N = 256, 128


def _torch_to_jax(t: torch.Tensor) -> jnp.ndarray:
    name = str(t.dtype).removeprefix("torch.")
    return jnp.asarray(t.detach().cpu().float().numpy()).astype(jnp.dtype(name))


def _jax_to_torch(a: jnp.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dtype)


def _install_cpu_jax_bridges(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route custom-op wrappers through their real JAX bodies on CPU."""
    cpu_quant_ops = {}
    for torch_dtype in linear_common._ACTIVATION_QUANT_DTYPES:
        jax_dtype = jnp.dtype(str(torch_dtype).removeprefix("torch."))

        def _run_quant(
            x: torch.Tensor, *, _td=torch_dtype, _jd=jax_dtype
        ) -> tuple[torch.Tensor, torch.Tensor]:
            q_j, s_j = _quantize_activation_jax(_torch_to_jax(x), quant_dtype=_jd)
            return _jax_to_torch(q_j, _td), _jax_to_torch(s_j, torch.float32)

        cpu_quant_ops[torch_dtype] = _run_quant

    def _run_prequant(
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
    ) -> torch.Tensor:
        out_j = _prequantized_matmul_bf16_jax(
            _torch_to_jax(x_q),
            _torch_to_jax(x_scale),
            _torch_to_jax(w_q),
            _torch_to_jax(w_s),
        )
        return _jax_to_torch(out_j, torch.bfloat16)

    def _run_full_matmul(
        x: torch.Tensor,
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        w8a16_max_rows: int = 0,
    ) -> torch.Tensor:
        out_j = _quantized_matmul_jax(
            _torch_to_jax(x),
            _torch_to_jax(w_q),
            _torch_to_jax(w_s),
            w8a16_max_rows=w8a16_max_rows,
        )
        return _jax_to_torch(out_j, x.dtype)

    monkeypatch.setattr(linear_common, "_quantize_activation_ops", cpu_quant_ops)
    monkeypatch.setattr(linear_common, "_prequantized_matmul_bf16_op", _run_prequant)
    monkeypatch.setattr(
        linear_common, "_get_quantized_matmul_op", lambda: _run_full_matmul
    )


def _make_layer(
    weight_scale: torch.Tensor,
    params_dtype: torch.dtype = torch.bfloat16,
    *,
    with_layer_bias: bool = False,
    seed: int = 0,
) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    w_f32 = torch.randn((_K, _N), generator=g, dtype=torch.float32) * 20.0
    w_q = w_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_q, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.params_dtype = params_dtype
    if with_layer_bias:
        bias_val = torch.randn((_N,), generator=g, dtype=params_dtype)
        layer.bias = torch.nn.Parameter(bias_val, requires_grad=False)
    return layer


@pytest.mark.cpu_test
@pytest.mark.parametrize("w_dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.int8])
def test_split_jax_ops_match_quantized_matmul(w_dtype):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, _K)) * np.linspace(0.01, 30.0, 64)[:, None]
    x = jnp.asarray(x, jnp.bfloat16)
    w_q = jnp.asarray(np.clip(rng.standard_normal((_K, _N)) * 40, -127, 127), w_dtype)
    w_s = jnp.asarray(rng.uniform(0.005, 0.02, size=_N), jnp.float32)

    quantize = functools.partial(_quantize_activation_jax, quant_dtype=w_q.dtype)
    shards = [quantize(part) for part in jnp.split(x, 8)]
    x_q = jnp.concatenate([q for q, _ in shards])
    x_scale = jnp.concatenate([s for _, s in shards])

    assert x_q.dtype == w_q.dtype and x_scale.shape == (64, 1)
    np.testing.assert_array_equal(
        np.asarray(_prequantized_matmul_bf16_jax(x_q, x_scale, w_q, w_s)),
        np.asarray(_quantized_matmul_jax(x, w_q, w_s)),
    )
    np.testing.assert_array_equal(
        np.asarray(
            xla_prequantized_matmul(*quantize_tensor(x, w_q.dtype), w_q, w_s, x.dtype)
        ),
        np.asarray(xla_quantized_matmul(x, w_q, w_s)),
    )


@pytest.mark.cpu_test
def test_every_requantized_weight_dtype_has_activation_quant():
    assert set(fp8._MOE_REQUANT_WEIGHT_DTYPES.values()) <= set(
        linear_common._quantize_activation_ops
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    (
        "scale_shape",
        "params_dtype",
        "keep_layout",
        "is_vocab",
        "rows",
        "max_rows",
        "expected",
    ),
    [
        ((_N,), torch.bfloat16, False, False, 16384, 0, True),
        ((_N,), torch.bfloat16, False, False, 0, 0, False),
        ((_N,), torch.bfloat16, False, False, 64, 64, False),
        ((_N,), torch.bfloat16, False, False, 65, 64, True),
        ((1, 2, 1, _N), torch.bfloat16, False, False, 16384, 0, False),
        ((_N,), torch.float16, False, False, 16384, 0, False),
        ((_N,), torch.bfloat16, True, False, 16384, 0, False),
        ((_N,), torch.bfloat16, False, True, 16384, 0, False),
    ],
)
def test_supports_quantized_input(
    monkeypatch,
    scale_shape,
    params_dtype,
    keep_layout,
    is_vocab,
    rows,
    max_rows,
    expected,
):
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", str(max_rows))
    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))
    if is_vocab:
        layer = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(layer)
        layer.weight = torch.nn.Parameter(
            torch.zeros((_K, _N), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(scale_shape), requires_grad=False
        )
        layer.params_dtype = params_dtype
    else:
        layer = _make_layer(torch.ones(scale_shape), params_dtype)
    if keep_layout:
        setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    assert method.supports_quantized_input(layer, rows) is expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("x_shape", [(16, _K), (2, 8, _K)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_apply_quantized_input_matches_apply_2d_and_3d_with_and_without_bias(
    monkeypatch, x_shape, use_bias
):
    _install_cpu_jax_bridges(monkeypatch)
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "0")
    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))

    g = torch.Generator().manual_seed(42)
    w_scale = torch.rand((_N,), generator=g, dtype=torch.float32) * 0.02 + 0.005
    # Attach a layer.bias attribute too so bias=None verifies that neither
    # apply nor apply_quantized_input implicitly adds layer.bias when omitted.
    layer = _make_layer(w_scale, with_layer_bias=True, seed=11)
    x = torch.randn(x_shape, generator=g, dtype=torch.bfloat16) * 2.0
    bias = (
        torch.randn((_N,), generator=g, dtype=torch.bfloat16) * 0.5
        if use_bias
        else None
    )

    rows = x.numel() // x.shape[-1]
    assert method.supports_quantized_input(layer, rows)

    x_q, x_scale = method.quantize_input(layer, x)
    assert x_q.shape == x_shape and x_q.dtype == torch.float8_e4m3fn
    assert x_scale.shape == (*x_shape[:-1], 1) and x_scale.dtype == torch.float32

    out_split = method.apply_quantized_input(layer, x_q, x_scale, bias)
    out_apply = method.apply(layer, x, bias)

    assert out_split.shape == (*x_shape[:-1], _N)
    assert out_split.dtype == torch.bfloat16
    assert torch.equal(out_split, out_apply)

    if use_bias:
        out_no_bias = method.apply_quantized_input(layer, x_q, x_scale, None)
        assert not torch.equal(out_split, out_no_bias)
        torch.testing.assert_close(out_split, out_no_bias + bias, rtol=0, atol=0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("gate_up_shape", [(16, 512), (2, 8, 512)])
def test_custom_ops_meta_fake_registrations(gate_up_shape):
    with torch.device("meta"):
        gate_up = torch.empty(gate_up_shape, dtype=torch.bfloat16)
        q, s = gelu_and_mul_quantize_op(gate_up)
        assert q.shape == (*gate_up_shape[:-1], gate_up_shape[-1] // 2)
        assert q.dtype == torch.float8_e4m3fn
        assert s.shape == (*gate_up_shape[:-1], 1)
        assert s.dtype == torch.float32

        x = torch.empty((*gate_up_shape[:-1], _K), dtype=torch.bfloat16)
        w_q = torch.empty((_K, _N), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((_N,), dtype=torch.float32)
        xq, xs = quantize_activation(x, w_q)
        out = prequantized_matmul(xq, xs, w_q, w_s)
        assert out.shape == (*gate_up_shape[:-1], _N)
        assert out.dtype == torch.bfloat16


@pytest.mark.tpu_test
@pytest.mark.parametrize("x_shape", [(16, _K), (2, 8, _K)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_fp8_prequantized_and_gelu_quant_on_tpu(monkeypatch, x_shape, use_bias):
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "0")
    method = VllmFp8LinearMethodTPU(Fp8Config(is_checkpoint_fp8_serialized=True))

    g = torch.Generator().manual_seed(99)
    w_scale = (torch.rand((_N,), generator=g, dtype=torch.float32) * 0.02 + 0.005).to(
        "tpu"
    )
    layer = _make_layer(w_scale.cpu(), seed=19).to("tpu")
    x = (torch.randn(x_shape, generator=g, dtype=torch.bfloat16) * 2.0).to("tpu")
    bias = (
        (torch.randn((_N,), generator=g, dtype=torch.bfloat16) * 0.5).to("tpu")
        if use_bias
        else None
    )

    x_q, x_scale = method.quantize_input(layer, x)
    out_split = method.apply_quantized_input(layer, x_q, x_scale, bias)
    out_apply = method.apply(layer, x, bias)
    assert out_split.shape == (*x_shape[:-1], _N)
    assert torch.equal(out_split.cpu(), out_apply.cpu())

    gate_up = (
        torch.randn((*x_shape[:-1], 2 * _K), generator=g, dtype=torch.bfloat16) * 2.0
    ).to("tpu")
    gq, gs = gelu_and_mul_quantize_op(gate_up)
    out_gelu = method.apply_quantized_input(layer, gq, gs, bias)
    assert out_gelu.shape == (*x_shape[:-1], _N)
    assert not torch.isnan(out_gelu.cpu().float()).any()
