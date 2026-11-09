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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.layernorm import RMSNormGated

from vllm_torchtpu.kernels.gated_norm_quant import (
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


@pytest.mark.parametrize("tokens", [1, 4, 12, 16, 48, 80])
def test_fused_output_projection_matches_unfused(tokens):
    attn_out, gate, norm_weight, w_q, w_scale = _inputs(tokens)
    normed = _unfused_norm(attn_out, gate, norm_weight)
    expected_q, expected_scale = quantize_tensor(normed, jnp.float8_e4m3fn)
    expected_out = xla_quantized_matmul(normed, w_q, w_scale)

    q, scale = gated_norm_quant(
        attn_out, gate, norm_weight, EPS, max_tile=16, interpret=True
    )
    out = fp8_matmul_on_quantized_rows(q.reshape(tokens, -1), scale, w_q, w_scale)

    np.testing.assert_array_equal(
        np.asarray(q.reshape(tokens, -1).astype(jnp.float32)),
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
    assert float(scale[3, 0]) == 1.0
    assert not np.any(np.asarray(q[3].astype(jnp.float32)))
    expected_q, expected_scale = quantize_tensor(
        _unfused_norm(attn_out, gate, norm_weight), jnp.float8_e4m3fn
    )
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(expected_scale))
    np.testing.assert_array_equal(
        np.asarray(q.reshape(8, -1).astype(jnp.float32)),
        np.asarray(expected_q.astype(jnp.float32)),
    )


@pytest.mark.parametrize(
    ("rows", "max_tile", "expected"),
    [
        (1, 32, 1),
        (4, 32, 4),
        (12, 32, 12),
        (16, 32, 16),
        (48, 32, 24),
        (80, 32, 16),
        (256, 32, 32),
        (36, 32, 36),
        (64, 4, 64),
    ],
)
def test_legal_row_tile(rows, max_tile, expected):
    tile = legal_row_tile(rows, max_tile)
    assert tile == expected
    assert rows % tile == 0
    assert tile == rows or tile % 8 == 0


def test_rejects_mismatched_shapes():
    attn_out, gate, norm_weight, _, _ = _inputs(4)
    with pytest.raises(ValueError, match="shapes do not match"):
        gated_norm_quant(attn_out, gate[:, :2], norm_weight, EPS, interpret=True)
