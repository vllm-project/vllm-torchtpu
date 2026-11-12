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
"""The fused GDN gated norm and FP8 quantization (TPU_GDN_FUSED_NORM_QUANT)
matches vLLM's RMSNormGated followed by the per-channel FP8 out-projection."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.layernorm import RMSNormGated

from vllm_torchtpu.kernels.gated_norm_quant import (
    FP8_ROW_TILE_ALIGN,
    fp8_matmul_on_quantized_rows,
    gated_norm_quant,
    legal_row_tile,
)
from vllm_torchtpu.kernels.quantized_matmul import util as quantized_matmul_util

# Jitted like the compiled step, so XLA simplifies the constant division the
# same way it does in serving.
quantize_tensor = jax.jit(quantized_matmul_util.quantize_tensor, static_argnums=1)
xla_quantized_matmul = jax.jit(quantized_matmul_util.xla_quantized_matmul)

HEADS = 4
HEAD_DIM = 128
HIDDEN = 256
EPS = 1e-6


def _to_torch(x: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.array(x.astype(jnp.float32))).to(torch.bfloat16)


def _to_jax(x: torch.Tensor) -> jax.Array:
    return jnp.asarray(x.float().numpy()).astype(jnp.bfloat16)


def _inputs(tokens: int, seed: int = 0):
    keys = jax.random.split(jax.random.key(seed), 5)
    attn_out = jax.random.normal(keys[0], (tokens, HEADS, HEAD_DIM)).astype(
        jnp.bfloat16
    )
    gate = (2.0 * jax.random.normal(keys[1], (tokens, HEADS, HEAD_DIM))).astype(
        jnp.bfloat16
    )
    norm_weight = (1.0 + 0.1 * jax.random.normal(keys[2], (HEAD_DIM,))).astype(
        jnp.bfloat16
    )
    weight = jax.random.normal(keys[3], (HIDDEN, HEADS * HEAD_DIM)) * 0.05
    # The out-projection's runtime FP8 layout: [n_in, n_out], scale [n_out].
    w_q, w_scale = quantize_tensor(weight, jnp.float8_e4m3fn)
    return attn_out, gate, norm_weight, w_q.T, w_scale.reshape(-1)


def _unfused_norm(attn_out, gate, norm_weight) -> jax.Array:
    """vLLM's RMSNormGated as the GDN layer configures it, in torch."""
    normed = RMSNormGated.forward_static(
        _to_torch(attn_out),
        _to_torch(gate),
        _to_torch(norm_weight),
        EPS,
        torch.bfloat16,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
    )
    return _to_jax(normed).reshape(attn_out.shape[0], -1)


@pytest.mark.parametrize("tokens", [1, 4, 12, 32, 48, 64, 80, 96, 300])
def test_fused_output_projection_matches_unfused(tokens):
    attn_out, gate, norm_weight, w_q, w_scale = _inputs(tokens)
    normed = _unfused_norm(attn_out, gate, norm_weight)
    expected_q, expected_scale = quantize_tensor(normed, jnp.float8_e4m3fn)
    expected_out = xla_quantized_matmul(normed, w_q, w_scale)

    q, scale = gated_norm_quant(
        attn_out, gate, norm_weight, EPS, max_tile=32, interpret=True
    )
    assert q.shape == (tokens, HEADS * HEAD_DIM)
    assert scale.shape == (tokens, 1)
    out = fp8_matmul_on_quantized_rows(q, scale, w_q, w_scale)

    np.testing.assert_array_equal(
        np.asarray(q.astype(jnp.float32)),
        np.asarray(expected_q.astype(jnp.float32)),
    )
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(expected_scale))
    assert out.dtype == expected_out.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(out.astype(jnp.float32)),
        np.asarray(expected_out.astype(jnp.float32)),
    )


def test_all_zero_row_gets_unit_scale():
    attn_out, gate, norm_weight, _, _ = _inputs(8)
    attn_out = attn_out.at[3].set(0)
    q, scale = gated_norm_quant(attn_out, gate, norm_weight, EPS, interpret=True)
    assert q.shape == (8, HEADS * HEAD_DIM)
    assert float(scale[3, 0]) == 1.0
    assert not np.any(np.asarray(q[3].astype(jnp.float32)))
    expected_q, expected_scale = quantize_tensor(
        _unfused_norm(attn_out, gate, norm_weight), jnp.float8_e4m3fn
    )
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(expected_scale))
    np.testing.assert_array_equal(
        np.asarray(q.astype(jnp.float32)),
        np.asarray(expected_q.astype(jnp.float32)),
    )


@pytest.mark.parametrize(
    ("rows", "max_tile", "expected"),
    [
        (1, 32, 1),
        (4, 32, 4),
        (12, 32, 12),
        (16, 32, 16),
        (32, 32, 32),
        (40, 64, 40),
        (48, 32, 32),
        (64, 32, 32),
        (80, 32, 32),
        (96, 64, 32),
        (128, 64, 64),
        (256, 32, 32),
        (300, 64, 64),
        (64, 4, 32),
    ],
)
def test_legal_row_tile(rows, max_tile, expected):
    tile = legal_row_tile(rows, max_tile)
    assert tile == expected
    assert tile <= max(max_tile, FP8_ROW_TILE_ALIGN)
    assert tile == rows or tile % FP8_ROW_TILE_ALIGN == 0


@pytest.mark.parametrize(("rows", "max_tile"), [(0, 32), (-4, 32), (32, 0), (32, -8)])
def test_legal_row_tile_rejects_nonpositive(rows, max_tile):
    with pytest.raises(ValueError, match="must be positive"):
        legal_row_tile(rows, max_tile)


def test_rejects_mismatched_shapes():
    attn_out, gate, norm_weight, _, _ = _inputs(4)
    with pytest.raises(ValueError, match="shapes do not match"):
        gated_norm_quant(attn_out, gate[:, :2], norm_weight, EPS, interpret=True)


@pytest.mark.parametrize("bad_head_dim", [8, 64, 192])
def test_rejects_unaligned_head_dim_on_tpu(bad_head_dim):
    attn_out = jnp.ones((4, 2, bad_head_dim), dtype=jnp.bfloat16)
    gate = jnp.ones_like(attn_out)
    weight = jnp.ones((bad_head_dim,), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="head_dim must be a multiple of 128"):
        gated_norm_quant(attn_out, gate, weight, EPS, interpret=False)


@pytest.mark.parametrize("bad_eps", [0.0, -1e-6, float("nan"), float("inf")])
def test_rejects_invalid_eps(bad_eps):
    attn_out, gate, norm_weight, _, _ = _inputs(4)
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        gated_norm_quant(attn_out, gate, norm_weight, bad_eps, interpret=True)


def _fp8_ulp_index(arr: jax.Array) -> np.ndarray:
    """Map sign-magnitude FP8 bytes to signed indices (+0 == -0 == 0)."""
    raw = np.asarray(jax.lax.bitcast_convert_type(arr, jnp.uint8), dtype=np.int16)
    mag = raw & 0x7F
    return np.where((raw & 0x80) != 0, -mag, mag)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("tokens", "heads", "head_dim", "max_tile"),
    [
        (1, 4, 128, 64),
        (12, 1, 128, 32),
        (64, 1, 128, 32),
        (40, 64, 128, 64),
        (80, 4, 128, 32),
        (300, 64, 128, 64),
        (384, 64, 128, 64),
        (64, 2, 128, 32),
    ],
)
def test_tpu_2d_layout_and_hlo_avoids_4d_bitcast(tokens, heads, head_dim, max_tile):
    keys = jax.random.split(jax.random.key(tokens + heads), 5)
    attn_out = jax.random.normal(keys[0], (tokens, heads, head_dim)).astype(
        jnp.bfloat16
    )
    gate = (2.0 * jax.random.normal(keys[1], (tokens, heads, head_dim))).astype(
        jnp.bfloat16
    )
    norm_weight = (1.0 + 0.1 * jax.random.normal(keys[2], (head_dim,))).astype(
        jnp.bfloat16
    )
    w_raw = jax.random.normal(keys[3], (HIDDEN, heads * head_dim)) * 0.05
    w_q_row, w_scale = quantize_tensor(w_raw, jnp.float8_e4m3fn)
    w_q = w_q_row.T
    w_scale = w_scale.reshape(-1)

    @jax.jit
    def fused_fn(a, g, nw, wq, ws):
        q, s = gated_norm_quant(a, g, nw, EPS, max_tile=max_tile)
        return q, s, fp8_matmul_on_quantized_rows(q, s, wq, ws)

    @jax.jit
    def unfused_tpu_fn(a, g, nw, wq, ws):
        af = a.astype(jnp.float32)
        gf = g.astype(jnp.float32)
        var = jnp.mean(af * af, axis=-1, keepdims=True)
        normed = (
            af * jax.lax.rsqrt(var + EPS) * nw.astype(jnp.float32)[None, None, :]
        ) * (gf * jax.nn.sigmoid(gf))
        normed_2d = normed.astype(jnp.bfloat16).reshape(tokens, heads * head_dim)
        eq, es = quantized_matmul_util.quantize_tensor(normed_2d, jnp.float8_e4m3fn)
        eout = quantized_matmul_util.xla_quantized_matmul(normed_2d, wq, ws)
        return eq, es, eout

    q, s, out = fused_fn(attn_out, gate, norm_weight, w_q, w_scale)
    expected_q, expected_s, expected_out = unfused_tpu_fn(
        attn_out, gate, norm_weight, w_q, w_scale
    )

    assert q.shape == (tokens, heads * head_dim)
    assert np.all(np.isfinite(np.asarray(q.astype(jnp.float32))))
    np.testing.assert_array_equal(np.asarray(s), np.asarray(expected_s))
    # Pallas lowers `normed / scale` directly while XLA's HLO simplifier rewrites
    # `normed / (amax / 448)` into multiplication by the reciprocal, which can
    # differ by at most 1 FP8 ULP on ties (~0.0002% of elements).
    assert int(np.max(np.abs(_fp8_ulp_index(q) - _fp8_ulp_index(expected_q)))) <= 1
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)),
        np.asarray(expected_out.astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )

    cpu_ref_out = xla_quantized_matmul(
        _unfused_norm(attn_out, gate, norm_weight), w_q, w_scale
    )
    np.testing.assert_allclose(
        np.asarray(out.astype(jnp.float32)),
        np.asarray(cpu_ref_out.astype(jnp.float32)),
        rtol=2e-2,
        atol=2e-2,
    )

    compiled_hlo = (
        fused_fn.lower(attn_out, gate, norm_weight, w_q, w_scale).compile().as_text()
    )
    assert not re.search(r"f8e4m3fn\[\d+,\d+,\d+,\d+\]", compiled_hlo), (
        f"Unexpected 4D FP8 bitcast in compiled HLO:\n{compiled_hlo}"
    )
    assert not re.search(r"=\s*bf16\[[^\]]+\][^\n]*\bpad\(", compiled_hlo), (
        f"Unexpected HBM bf16 pad copy in compiled HLO:\n{compiled_hlo}"
    )
