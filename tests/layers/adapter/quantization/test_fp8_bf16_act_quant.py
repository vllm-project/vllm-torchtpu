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
"""Tests for bfloat16 FP8 activation quantization (VLLM_TPU_FP8_FAST_BF16_ACT_QUANT)."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantLinearConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

_K, _N = 256, 128


def _weights(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((_K, _N)).astype(np.float32)
    scale = np.abs(w).max(axis=0) / 448.0
    w_q = jnp.asarray(w / scale, dtype=jnp.float8_e4m3fn)
    return w_q, jnp.asarray(scale, dtype=jnp.float32)


def test_quantize_block_bf16_hlo_and_roundtrip() -> None:
    x = jax.random.normal(jax.random.key(0), (64, 512), dtype=jnp.bfloat16)

    lowered_default = jax.jit(
        lambda a: util.quantize_block(
            a, axis=1, target_dtype=jnp.float8_e4m3fn, bf16_act_quant=False
        )
    ).lower(x)
    assert "f32[64,512]{1,0} divide(" in lowered_default.as_text("hlo")

    lowered_bf16 = jax.jit(
        lambda a: util.quantize_block(
            a, axis=1, target_dtype=jnp.float8_e4m3fn, bf16_act_quant=True
        )
    ).lower(x)
    hlo_text = lowered_bf16.as_text("hlo")
    assert "bf16[64,512]{1,0} divide(" in hlo_text
    assert "f32[64,512]{1,0} divide(" not in hlo_text

    x_q_fp32, x_s_fp32 = util.quantize_block(
        x, axis=1, target_dtype=jnp.float8_e4m3fn, bf16_act_quant=False
    )
    x_q, x_s = util.quantize_block(
        x, axis=1, target_dtype=jnp.float8_e4m3fn, bf16_act_quant=True
    )
    assert x_q.dtype == jnp.float8_e4m3fn
    assert x_s.dtype == jnp.float32
    assert jnp.array_equal(x_s, x_s.astype(jnp.bfloat16).astype(jnp.float32))
    assert jnp.allclose(x_s, x_s_fp32, rtol=2**-7, atol=0.0)

    x_deq_fp32 = x_q_fp32.astype(jnp.float32) * x_s_fp32
    x_deq = x_q.astype(jnp.float32) * x_s
    rel_err = jnp.mean(jnp.abs(x.astype(jnp.float32) - x_deq)) / jnp.mean(
        jnp.abs(x.astype(jnp.float32))
    )
    assert float(rel_err) < 0.05

    # Compare bf16-division output directly against the fp32-division path:
    # double rounding can shift at most 1 fp8_e4m3fn ULP plus 1 bf16 scale ULP,
    # and mean drift across the tensor stays << 1%.
    fp8_eps = float(jnp.finfo(jnp.float8_e4m3fn).eps)
    bf16_eps = float(jnp.finfo(jnp.bfloat16).eps)
    assert jnp.allclose(
        x_deq,
        x_deq_fp32,
        rtol=fp8_eps + bf16_eps,
        atol=(2**-9) * float(jnp.max(x_s_fp32)),
    )
    bf16_vs_fp32_rel_err = jnp.mean(jnp.abs(x_deq - x_deq_fp32)) / jnp.mean(
        jnp.abs(x_deq_fp32)
    )
    assert float(bf16_vs_fp32_rel_err) < 0.01


def test_quantize_block_edge_cases() -> None:
    # Zero and near-subnormal bf16 rows must not produce NaN or Inf.
    x_edge = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1e-37, -1e-37, 0.0, 0.0],
            [1.0, -2.0, 4.0, -8.0],
        ],
        dtype=jnp.bfloat16,
    )
    q, s = util.quantize_block(
        x_edge, axis=1, target_dtype=jnp.float8_e4m3fn, bf16_act_quant=True
    )
    assert not jnp.isnan(q.astype(jnp.float32)).any()
    assert not jnp.isnan(s).any()
    assert jnp.all(q[0] == 0)
    assert jnp.all(q[1] == 0)

    # float16 small-magnitude rows (1e-4 / 448 underflows fp16) still divide in fp32.
    x_fp16 = (
        jax.random.uniform(jax.random.key(1), (8, 128), minval=0.5, maxval=1.0) * 1e-4
    ).astype(jnp.float16)
    q_fp16, s_fp16 = util.quantize_block(
        x_fp16, axis=1, target_dtype=jnp.float8_e4m3fn, bf16_act_quant=True
    )
    deq_fp16 = q_fp16.astype(jnp.float32) * s_fp16
    assert not jnp.isnan(deq_fp16).any()
    assert jnp.allclose(deq_fp16, x_fp16.astype(jnp.float32), rtol=0.1, atol=1e-5)

    # Integer quantization ignores bf16_act_quant.
    q_i8_off, s_i8_off = util.quantize_block(
        x_edge[2:], axis=1, target_dtype=jnp.int8, bf16_act_quant=False
    )
    q_i8_on, s_i8_on = util.quantize_block(
        x_edge[2:], axis=1, target_dtype=jnp.int8, bf16_act_quant=True
    )
    assert jnp.array_equal(q_i8_off, q_i8_on)
    assert jnp.array_equal(s_i8_off, s_i8_on)


@pytest.mark.parametrize("bf16_act_quant", [False, True])
def test_quantized_matmul_jax_forwards_bf16_act_quant(
    monkeypatch: pytest.MonkeyPatch, bf16_act_quant: bool
) -> None:
    seen: dict[str, bool] = {}
    real = linear_common.xla_quantized_matmul

    def spy(x, w_q, w_s, quantize_activation=True, *, bf16_act_quant=False):
        seen["bf16_act_quant"] = bf16_act_quant
        return real(
            x,
            w_q,
            w_s,
            quantize_activation=quantize_activation,
            bf16_act_quant=bf16_act_quant,
        )

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", spy)
    w_q, w_s = _weights()
    out = _quantized_matmul_jax(
        jnp.ones((16, _K), jnp.bfloat16),
        w_q,
        w_s,
        bf16_act_quant=bf16_act_quant,
    )
    assert out.shape == (16, _N)
    assert out.dtype == jnp.bfloat16
    assert seen["bf16_act_quant"] is bf16_act_quant


@pytest.mark.parametrize("env_val,expected", [(None, False), ("0", False), ("1", True)])
def test_fp8_linear_method_forwards_bf16_act_quant(
    monkeypatch: pytest.MonkeyPatch, env_val: str | None, expected: bool
) -> None:
    if env_val is None:
        monkeypatch.delenv("VLLM_TPU_FP8_FAST_BF16_ACT_QUANT", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_FP8_FAST_BF16_ACT_QUANT", env_val)

    mock_linear = MagicMock(spec=LinearBase)
    mock_linear.output_size = _N
    linear_config = VllmQuantLinearConfig(None, mock_linear)
    assert linear_config.bf16_act_quant is expected

    # Flip env afterwards to verify linear_config snapshot is used.
    monkeypatch.setenv("VLLM_TPU_FP8_FAST_BF16_ACT_QUANT", "0" if expected else "1")
    method = VllmFp8LinearMethodTPU(
        Fp8Config(is_checkpoint_fp8_serialized=True), linear_config=linear_config
    )
    assert method.bf16_act_quant is expected

    seen: dict[str, bool] = {}

    def fake_quantized_matmul(x, w_q, w_s, w8a16_max_rows=0, bf16_act_quant=False):
        del w_s, w8a16_max_rows
        seen["bf16_act_quant"] = bf16_act_quant
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(fp8, "quantized_matmul", fake_quantized_matmul)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((_K, _N), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((_N,), dtype=torch.float32), requires_grad=False
    )
    method.apply(layer, torch.ones((16, _K), dtype=torch.bfloat16))
    assert seen["bf16_act_quant"] is expected


def test_quantized_matmul_op_accepts_bf16_act_quant() -> None:
    op = linear_common._get_quantized_matmul_op()
    with torch.device("meta"):
        x = torch.empty((16, _K), dtype=torch.bfloat16)
        w_q = torch.empty((_K, _N), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((_N,), dtype=torch.float32)
        out = op(x, w_q, w_s, 0, True)
    assert out.shape == (16, _N)
    assert out.dtype == torch.bfloat16
