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
"""The decode weight-streaming FP8 matmul (TPU_FP8_DECODE_GEMV_MAX_ROWS)."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.kernels.quantized_matmul import decode_gemv
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
    xla_quantized_matmul,
)
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantLinearConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU


def _operands(rows, k, n, seed=0):
    x = jax.random.normal(jax.random.key(seed), (rows, k), jnp.float32)
    x = (x * jnp.linspace(0.01, 30.0, rows)[:, None]).astype(jnp.bfloat16)
    x = x.at[rows // 2].set(0)  # a padding row: scale 1, zeros out
    w = jax.random.normal(jax.random.key(seed + 1), (k, n), jnp.float32) * 40.0
    w_q = jnp.clip(w, -448, 448).astype(jnp.float8_e4m3fn)
    w_s = jax.random.uniform(jax.random.key(seed + 2), (n,), jnp.float32) * 0.01
    return x, w_q, w_s


# (rows, k, n, tile_bytes): tile budgets small enough for several tiles per
# call, so the ring's slot reuse and the tail without refills both run.
_CASES = [
    (16, 512, 1536, 512 * 256),
    (32, 384, 1280, 384 * 128),
    (16, 256, 896, 256 * 384),
    (48, 128, 384, 128 * 128),
]


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows,k,n,tile_bytes", _CASES)
def test_matches_xla_quantized_matmul(rows, k, n, tile_bytes):
    x, w_q, w_s = _operands(rows, k, n)
    tn, depth, _ = decode_gemv.tile_config(k, n, tile_bytes=tile_bytes)
    assert n // tn > 1
    ref = jax.jit(xla_quantized_matmul)(x, w_q, w_s)
    out = decode_gemv.fp8_decode_matmul(
        x, w_q, w_s, tile_bytes=tile_bytes, interpret=True
    )
    assert out.shape == ref.shape and out.dtype == ref.dtype
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows,k,n,tile_bytes", _CASES)
def test_prequantized_matches_xla(rows, k, n, tile_bytes):
    x, w_q, w_s = _operands(rows, k, n, seed=3)
    x_q, x_scale = jax.jit(lambda x: quantize_tensor(x, w_q.dtype))(x)
    ref = jax.jit(lambda *a: xla_prequantized_matmul(*a, out_dtype=jnp.bfloat16))(
        x_q, x_scale, w_q, w_s
    )
    out = decode_gemv.fp8_decode_prequantized_matmul(
        x_q, x_scale, w_q, w_s, jnp.bfloat16, tile_bytes=tile_bytes, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "k,n", [(5376, 8192), (4096, 5376), (5376, 21504), (10752, 5376), (8192, 5376)]
)
def test_tile_config_fits_budget(k, n):
    tn, depth, splits = decode_gemv.tile_config(k, n)
    assert n % tn == 0 and tn % 128 == 0
    assert k * tn <= decode_gemv._TILE_BYTES
    assert 1 <= depth <= n // tn
    assert (k // splits) % 32 == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rows,n,w_dtype,scale_ndim,expected",
    [
        (16, 256, jnp.float8_e4m3fn, 1, True),
        (24, 256, jnp.float8_e4m3fn, 1, False),  # rows not a whole bf16 tile
        (16, 200, jnp.float8_e4m3fn, 1, False),  # columns not whole lanes
        (16, 256, jnp.int8, 1, False),
        (16, 256, jnp.float8_e4m3fn, 4, False),  # blockwise scales
    ],
)
def test_supports(rows, n, w_dtype, scale_ndim, expected):
    assert decode_gemv.supports(rows, 256, n, w_dtype, scale_ndim) is expected


@pytest.mark.cpu_test
def test_supports_needs_vmem():
    """Buckets whose rows and output outgrow the kernel's VMEM keep XLA."""
    assert decode_gemv.supports(32, 5376, 21504, jnp.float8_e4m3fn, 1)
    assert not decode_gemv.supports(1024, 5376, 21504, jnp.float8_e4m3fn, 1)
    # Gemma4's bf16 LM head: the TP2 vocab shard at 16 rows, the whole vocab
    # at 32 (a 16 MiB output next to the 21 MiB ring) does not fit.
    assert decode_gemv.supports_nt(16, 5376, 131072)
    assert not decode_gemv.supports_nt(32, 5376, 262144)


def _route(monkeypatch, rows, n=256, w8a16_max_rows=0, gemv_max_rows=16):
    seen = {"kernel": False}
    real = decode_gemv.fp8_decode_matmul

    def spy(x, w_q, w_s):
        seen["kernel"] = True
        return real(x, w_q, w_s, interpret=True)

    monkeypatch.setattr(decode_gemv, "fp8_decode_matmul", spy)
    x, w_q, w_s = _operands(rows, 256, n)
    out = linear_common._quantized_matmul_jax(
        x, w_q, w_s, w8a16_max_rows=w8a16_max_rows, decode_gemv_max_rows=gemv_max_rows
    )
    assert out.shape == (rows, n)
    return seen["kernel"]


@pytest.mark.cpu_test
def test_routing_follows_threshold(monkeypatch):
    assert _route(monkeypatch, 16)
    assert not _route(monkeypatch, 32)  # above the threshold
    assert not _route(monkeypatch, 16, gemv_max_rows=0)  # default off
    assert not _route(monkeypatch, 16, w8a16_max_rows=16)  # W8A16 wins
    assert not _route(monkeypatch, 16, n=200)  # does not fit: XLA


@pytest.mark.cpu_test
def test_prequantized_routing(monkeypatch):
    seen = {"kernel": False}
    real = decode_gemv.fp8_decode_prequantized_matmul

    def spy(x_q, x_scale, w_q, w_s, out_dtype):
        seen["kernel"] = True
        return real(x_q, x_scale, w_q, w_s, out_dtype, interpret=True)

    monkeypatch.setattr(decode_gemv, "fp8_decode_prequantized_matmul", spy)
    x, w_q, w_s = _operands(16, 256, 256)
    x_q, x_scale = quantize_tensor(x, w_q.dtype)
    linear_common._prequantized_matmul_bf16_jax(x_q, x_scale, w_q, w_s, 16)
    assert seen["kernel"]


@pytest.mark.cpu_test
def test_fp8_linear_method_forwards_decode_gemv_max_rows(monkeypatch):
    # VllmQuantLinearConfig reads the env when the layer is built.
    layer = MagicMock(spec=LinearBase)
    layer.output_size = 128
    monkeypatch.setenv("TPU_FP8_DECODE_GEMV_MAX_ROWS", "16")
    linear_config = VllmQuantLinearConfig(None, layer)
    monkeypatch.setenv("TPU_FP8_DECODE_GEMV_MAX_ROWS", "0")
    method = VllmFp8LinearMethodTPU(
        Fp8Config(is_checkpoint_fp8_serialized=True), linear_config=linear_config
    )
    assert method.decode_gemv_max_rows == 16

    seen = {}

    def fake_quantized_matmul(x, w_q, w_s, w8a16_max_rows=0, decode_gemv_max_rows=0):
        seen["decode_gemv_max_rows"] = decode_gemv_max_rows
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(fp8, "quantized_matmul", fake_quantized_matmul)
    lin = torch.nn.Module()
    lin.weight = torch.nn.Parameter(
        torch.zeros((256, 128), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    lin.weight_scale = torch.nn.Parameter(
        torch.ones((128,), dtype=torch.float32), requires_grad=False
    )
    method.apply(lin, torch.ones((16, 256), dtype=torch.bfloat16))
    assert seen["decode_gemv_max_rows"] == 16


@pytest.mark.cpu_test
def test_ops_accept_decode_gemv_max_rows():
    op = linear_common._get_quantized_matmul_op()
    with torch.device("meta"):
        x = torch.empty((16, 256), dtype=torch.bfloat16)
        w_q = torch.empty((256, 128), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((128,), dtype=torch.float32)
        assert op(x, w_q, w_s, 0, 16).shape == (16, 128)
        x_q = torch.empty((16, 256), dtype=torch.float8_e4m3fn)
        x_scale = torch.empty((16, 1), dtype=torch.float32)
        out = linear_common._prequantized_matmul_bf16_op(x_q, x_scale, w_q, w_s, 16)
        assert out.shape == (16, 128)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rows,k,n,tile_bytes",
    [
        (8, 512, 2048, 512 * 256 * 2),
        (16, 384, 1280, 384 * 128 * 2),
        (8, 256, 384, 1 << 20),
    ],
)
def test_nt_matches_xla(rows, k, n, tile_bytes):
    x = jax.random.normal(jax.random.key(0), (rows, k), jnp.float32).astype(
        jnp.bfloat16
    )
    w = jax.random.normal(jax.random.key(1), (n, k), jnp.float32).astype(jnp.bfloat16)
    ref = jax.jit(
        lambda a, b: jnp.einsum(
            "mk,nk->mn", a, b, preferred_element_type=jnp.float32
        ).astype(a.dtype)
    )(x, w)
    out = decode_gemv.decode_matmul_nt(x, w, tile_bytes=tile_bytes, interpret=True)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))


def _head(n=256, k=128, dtype=torch.bfloat16):
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.randn(n, k).to(dtype), requires_grad=False)
    return head


@pytest.mark.cpu_test
def test_lm_head_method_is_opt_in(monkeypatch):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmDecodeLMHeadMethod,
        VllmUnquantizedConfig,
    )

    head = MagicMock(spec=ParallelLMHead)
    embedding = MagicMock(spec=VocabParallelEmbedding)
    get = VllmUnquantizedConfig.get_quant_method
    monkeypatch.setenv("TPU_LM_HEAD_DECODE_GEMV_MAX_ROWS", "0")
    assert get(None, head, "lm_head") is None
    monkeypatch.setenv("TPU_LM_HEAD_DECODE_GEMV_MAX_ROWS", "16")
    method = get(None, head, "lm_head")
    assert isinstance(method, VllmDecodeLMHeadMethod) and method.max_rows == 16
    assert get(None, embedding, "model.embed_tokens") is None


@pytest.mark.cpu_test
def test_lm_head_method_routes_small_buckets(monkeypatch):
    from vllm_torchtpu.layers.adapter.quantization import unquantized

    calls = []

    def fake_nt(x, w):
        calls.append(tuple(x.shape))
        return torch.nn.functional.linear(x, w)

    monkeypatch.setattr(unquantized, "decode_matmul_nt", fake_nt)
    method = unquantized.VllmDecodeLMHeadMethod(16)
    head = _head()
    x = torch.randn(8, 128).to(torch.bfloat16)
    torch.testing.assert_close(
        method.apply(head, x), torch.nn.functional.linear(x, head.weight)
    )
    assert calls == [(8, 128)]
    method.apply(head, torch.randn(32, 128).to(torch.bfloat16))  # above max_rows
    method.apply(head, x, bias=torch.zeros(256, dtype=torch.bfloat16))
    method.apply(head, torch.randn(12, 128).to(torch.bfloat16))  # not a row tile
    method.apply(_head(n=200), x)  # columns not whole lanes
    assert calls == [(8, 128)]


@pytest.mark.cpu_test
def test_fp8_config_lm_head_gets_the_decode_method(monkeypatch):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmDecodeLMHeadMethod,
    )

    config = VllmFp8Config(is_checkpoint_fp8_serialized=True)
    head = MagicMock(spec=ParallelLMHead)
    monkeypatch.setenv("TPU_LM_HEAD_DECODE_GEMV_MAX_ROWS", "0")
    assert config.get_quant_method(head, "lm_head") is None
    monkeypatch.setenv("TPU_LM_HEAD_DECODE_GEMV_MAX_ROWS", "8")
    assert isinstance(config.get_quant_method(head, "lm_head"), VllmDecodeLMHeadMethod)


@pytest.mark.cpu_test
def test_decode_matmul_nt_op_shape():
    with torch.device("meta"):
        x = torch.empty((8, 128), dtype=torch.bfloat16)
        w = torch.empty((256, 128), dtype=torch.bfloat16)
        out = linear_common._decode_matmul_nt_op(x, w)
    assert out.shape == (8, 256) and out.dtype == torch.bfloat16
