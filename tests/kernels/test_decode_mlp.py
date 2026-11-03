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
"""The Gemma4 decode post-attention block (TPU_GEMMA4_DECODE_FUSED_MLP_MAX_ROWS)."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.activation import GeluAndMul

from vllm_torchtpu.kernels.quantized_matmul import decode_gemv, decode_mlp
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

_EPS = 1e-6


def _rms_norm(x, w):
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(var + _EPS) * w.astype(jnp.float32)).astype(x.dtype)


def _unfused(attn, res, w_post, w_pre, w1, s1, w2, s2):
    """The layer's ops in order, as the TPU rms_norm, add and FP8 linears run them."""
    r = (_rms_norm(attn, w_post).astype(jnp.float32) + res.astype(jnp.float32)).astype(
        attn.dtype
    )
    gu = xla_quantized_matmul(_rms_norm(r, w_pre), w1, s1).astype(jnp.float32)
    inter = w2.shape[0]
    g, u = gu[:, :inter], gu[:, inter:]
    gelu = 0.5 * g * (1 + jnp.tanh(math.sqrt(2 / math.pi) * (g + 0.044715 * g**3)))
    return xla_quantized_matmul((gelu * u).astype(attn.dtype), w2, s2), r


def _operands(rows, hidden, inter, norm_dtype=jnp.float32):
    k = jax.random.split(jax.random.key(hidden + inter), 8)
    attn = (jax.random.normal(k[0], (rows, hidden)) * 3).astype(jnp.bfloat16)
    res = (jax.random.normal(k[1], (rows, hidden)) * 10).astype(jnp.bfloat16)
    w_post = jax.random.uniform(k[2], (hidden,), minval=0.5, maxval=1.5).astype(
        norm_dtype
    )
    w_pre = jax.random.uniform(k[3], (hidden,), minval=0.5, maxval=1.5).astype(
        norm_dtype
    )

    def fp8(key, shape):
        w = jax.random.normal(key, shape) * 40
        return jnp.clip(w, -448, 448).astype(jnp.float8_e4m3fn)

    w1 = fp8(k[4], (hidden, 2 * inter))
    s1 = jax.random.uniform(k[5], (2 * inter,)) * 0.002
    w2 = fp8(k[6], (inter, hidden))
    s2 = jax.random.uniform(k[7], (hidden,)) * 0.002
    return attn, res, w_post, w_pre, w1, s1, w2, s2


# (rows, hidden, inter, tile_bytes): several tiles per projection, with the
# down ring as deep as, shallower than and one tile of the gate/up ring.
_CASES = [
    (16, 256, 384, 256 * 128),
    (32, 384, 256, 384 * 128 * 2),
    (16, 512, 1024, 512 * 256),
    (16, 256, 1280, 256 * 256),
]


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows,hidden,inter,tile_bytes", _CASES)
@pytest.mark.parametrize("norm_dtype", [jnp.float32, jnp.bfloat16])
def test_matches_unfused_layer(rows, hidden, inter, tile_bytes, norm_dtype):
    args = _operands(rows, hidden, inter, norm_dtype)
    tn1, _, _ = decode_gemv.tile_config(hidden, 2 * inter, tile_bytes=tile_bytes)
    assert 2 * inter // tn1 > 1
    out_ref, res_ref = jax.jit(_unfused)(*args)
    out, res = decode_mlp.decode_geglu_mlp(
        *args, eps=_EPS, tile_bytes=tile_bytes, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(res), np.asarray(res_ref))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(out_ref))


@pytest.mark.cpu_test
def test_supports_gemma4_buckets_and_refuses_large_ones():
    # Gemma4-31B per TP2 rank: hidden 5376, intermediate 10752.
    assert decode_mlp.supports(16, 5376, 10752)
    assert decode_mlp.supports(32, 5376, 10752)
    assert not decode_mlp.supports(512, 5376, 10752)
    assert not decode_mlp.supports(24, 5376, 10752)  # not a whole bf16 tile


@pytest.mark.cpu_test
def test_op_shapes():
    with torch.device("meta"):
        attn = torch.empty((16, 256), dtype=torch.bfloat16)
        w1 = torch.empty((256, 768), dtype=torch.float8_e4m3fn)
        w2 = torch.empty((384, 256), dtype=torch.float8_e4m3fn)
        s = torch.empty((768,), dtype=torch.float32)
        n = torch.empty((256,), dtype=torch.float32)
        out, res = linear_common._decode_geglu_mlp_op(
            attn, attn, n, n, w1, s, w2, n, _EPS
        )
    assert out.shape == res.shape == (16, 256)


def _layer(**overrides):
    proj = SimpleNamespace(
        quant_method=MagicMock(spec=VllmFp8LinearMethodTPU), bias=None
    )
    act = MagicMock(spec=GeluAndMul)
    act.approximate = "tanh"
    norm = SimpleNamespace(has_weight=True, variance_epsilon=_EPS)
    layer = SimpleNamespace(
        enable_moe_block=False,
        per_layer_input_gate=None,
        mlp=SimpleNamespace(act_fn=act, gate_up_proj=proj, down_proj=proj),
        post_attention_layernorm=norm,
        pre_feedforward_layernorm=norm,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


@pytest.mark.cpu_test
def test_refusals():
    from vllm_torchtpu.models.vllm import gemma4_decode_patch

    refusal = gemma4_decode_patch._refusal
    assert refusal(_layer()) is None
    assert refusal(_layer(enable_moe_block=True)) == "MoE layers"
    assert refusal(_layer(per_layer_input_gate=object())) == "per-layer embeddings"
    other_eps = SimpleNamespace(has_weight=True, variance_epsilon=1e-5)
    assert refusal(_layer(pre_feedforward_layernorm=other_eps)) is not None
    unquantized = SimpleNamespace(quant_method=object(), bias=None)
    mlp = _layer().mlp
    mlp.down_proj = unquantized
    assert refusal(_layer(mlp=mlp)) is not None


def _proj(quantizes_rows=True, dtype=torch.float8_e4m3fn, shape=(256, 768)):
    method = MagicMock(spec=VllmFp8LinearMethodTPU)
    method.supports_quantized_input.return_value = quantizes_rows
    return SimpleNamespace(quant_method=method, weight=torch.empty(shape, dtype=dtype))


@pytest.mark.cpu_test
def test_fits_needs_fp8_w8a8_projections(monkeypatch):
    from vllm_torchtpu.models.vllm import gemma4_decode_patch

    monkeypatch.setattr(gemma4_decode_patch, "_max_rows", 16)
    fits = gemma4_decode_patch._fits

    def layer(gate_up, down):
        return SimpleNamespace(
            mlp=SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
        )

    down = _proj(shape=(384, 256))
    assert fits(layer(_proj(), down), 16)
    # W8A16 buckets (no per-token quantization) and int8 weights keep the forward.
    assert not fits(layer(_proj(quantizes_rows=False), down), 16)
    assert not fits(layer(_proj(dtype=torch.int8), down), 16)
