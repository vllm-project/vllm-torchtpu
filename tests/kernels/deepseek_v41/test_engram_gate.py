# SPDX-License-Identifier: Apache-2.0
"""The fused Engram gate is as close to an exact reference as the XLA path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.engram_gate import engram_gate
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

EPS, CLAMP = 1e-20, 1e-6


def _xla_path(x, w, w_scale, hidden, q_weight, k_weight, mask):
    """The module's forward (common/engram.py) as XLA computes it."""
    heads, dim = hidden.shape[1:]
    kv = xla_quantized_matmul(x, w, w_scale)
    keys = kv[:, : heads * dim].reshape(-1, heads, dim).astype(jnp.float32)
    values = kv[:, heads * dim :].astype(jnp.float32)
    h = hidden.astype(jnp.float32)
    dot = (h * q_weight * k_weight * keys).sum(-1)
    dot *= jax.lax.rsqrt(jnp.square(h).mean(-1) + EPS)
    dot *= jax.lax.rsqrt(jnp.square(keys).mean(-1) + EPS) * dim**-0.5
    magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), CLAMP))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -magnitude, magnitude)) * mask[:, None]
    return (h + gate[:, :, None] * values[:, None, :]).astype(hidden.dtype)


def _exact(x, w, w_scale, hidden, q_weight, k_weight, mask):
    """FP64 math on the module's rounded tensors: FP8 activations, BF16 kv."""
    heads, dim = hidden.shape[1:]
    a = np.asarray(x, np.float64)
    scale = np.abs(a).max(1, keepdims=True) / 448.0
    scale[scale == 0] = 1
    xq = np.asarray(
        jnp.asarray(np.clip(a / scale, -448, 448), jnp.float32).astype(
            jnp.float8_e4m3fn
        ),
        np.float64,
    )
    kv = (xq @ np.asarray(w, np.float64)) * scale * np.asarray(w_scale, np.float64)
    kv = np.asarray(jnp.asarray(kv, jnp.float32).astype(jnp.bfloat16), np.float64)
    keys = kv[:, : heads * dim].reshape(-1, heads, dim)
    values = kv[:, heads * dim :]
    h = np.asarray(hidden, np.float64)
    qk = np.asarray(q_weight, np.float64) * np.asarray(k_weight, np.float64)
    dot = (h * qk * keys).sum(-1)
    dot /= np.sqrt((h * h).mean(-1) + EPS)
    dot /= np.sqrt((keys * keys).mean(-1) + EPS) * dim**0.5
    magnitude = np.sqrt(np.maximum(np.abs(dot), CLAMP))
    gate = 1 / (1 + np.exp(-np.where(dot < 0, -magnitude, magnitude)))
    return h + (gate * np.asarray(mask)[:, None])[:, :, None] * values[:, None, :]


@pytest.mark.parametrize(
    "tokens,heads,dim,k_dim,window,tile,max_tokens",
    [
        pytest.param(128, 2, 256, 256, 128, 64, 64, id="small-interpret"),
        pytest.param(100, 2, 256, 256, 128, 64, 64, id="ragged-interpret"),
        # Two token tiles: the first head runs tile-major and quantizes the
        # second tile under the first, its last block clamped; ragged tail.
        pytest.param(136, 2, 256, 256, 128, 128, 2048, id="fused-interpret"),
        pytest.param(2048, 4, 5120, 6144, 1280, 512, 1024, id="prefill"),
    ],
)
def test_engram_gate_error_no_worse_than_xla(
    tokens, heads, dim, k_dim, window, tile, max_tokens
):
    interpret = jax.default_backend() != "tpu"
    if not interpret and tokens < 2048 or interpret and tokens >= 2048:
        pytest.skip("small shape interprets on CPU; the prefill shape runs on TPU")
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((tokens, k_dim)), jnp.bfloat16)
    w = jnp.asarray(
        rng.standard_normal((k_dim, (heads + 1) * dim)) * 30, jnp.float8_e4m3fn
    )
    w_scale = jnp.asarray(rng.uniform(1e-3, 2e-3, (heads + 1) * dim), jnp.float32)
    hidden = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    q_weight = jnp.asarray(1 + 0.1 * rng.standard_normal((heads, dim)), jnp.bfloat16)
    k_weight = jnp.asarray(1 + 0.1 * rng.standard_normal((heads, dim)), jnp.bfloat16)
    mask = jnp.asarray(rng.random(tokens) > 0.1, jnp.float32)
    args = (x, w, w_scale, hidden, q_weight, k_weight, mask)

    fused = engram_gate(
        *args,
        eps=EPS,
        clamp=CLAMP,
        window=window,
        token_tile=tile,
        quant_rows=min(64, tokens),
        max_tokens=max_tokens,
        interpret=interpret,
    )
    xla = jax.jit(_xla_path)(*args)
    exact = _exact(*args)
    fused_err = np.abs(np.asarray(fused, np.float64) - exact)
    xla_err = np.abs(np.asarray(xla, np.float64) - exact)
    assert fused_err.mean() <= xla_err.mean() * 1.05
    assert fused_err.max() <= xla_err.max() * 1.05 + 1e-6
    masked = ~np.asarray(mask, bool)
    np.testing.assert_array_equal(np.asarray(fused)[masked], np.asarray(hidden)[masked])


@pytest.mark.parametrize("tokens", [288, 2048])
def test_engram_gate_packed_rows_match_bf16_rows(tokens):
    """Packed staging words give the bits of the rows dequant_fp8_rows builds,
    including -0 bytes, byte-255 (invalid) and byte-0 scales and zero rows."""
    if jax.default_backend() != "tpu":
        pytest.skip("the packed prologue is validated on TPU")
    from vllm_torchtpu.models.vllm.deepseek_v41.common.engram_hbm import dequant

    heads, dim, k_dim, block = 4, 5120, 6144, 32
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (tokens, k_dim), dtype=np.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x80
    raw[5] = 0
    scale = (127 + rng.integers(-12, 12, (tokens, k_dim // block))).astype(np.uint8)
    scale[rng.random(scale.shape) < 0.02] = 255
    scale[7] = 255
    scale[9, :3] = 0
    w = jnp.asarray(
        rng.standard_normal((k_dim, (heads + 1) * dim)) * 30, jnp.float8_e4m3fn
    )
    w_scale = jnp.asarray(rng.uniform(1e-3, 2e-3, (heads + 1) * dim), jnp.float32)
    hidden = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    q_weight = jnp.asarray(1 + 0.1 * rng.standard_normal((heads, dim)), jnp.bfloat16)
    k_weight = jnp.asarray(1 + 0.1 * rng.standard_normal((heads, dim)), jnp.bfloat16)
    mask = jnp.asarray(rng.random(tokens) > 0.1, jnp.float32)
    rest = (w, w_scale, hidden, q_weight, k_weight, mask)
    rows = dequant(jnp.asarray(raw), jnp.asarray(scale), block)
    expected = engram_gate(rows, *rest, eps=EPS, clamp=CLAMP)
    words = (jnp.asarray(raw.view(np.int32)), jnp.asarray(scale.view(np.int32)))
    actual = engram_gate(words, *rest, eps=EPS, clamp=CLAMP, block_size=block)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16)
    )
