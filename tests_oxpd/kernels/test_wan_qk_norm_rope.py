# SPDX-License-Identifier: Apache-2.0
"""Numerics tests for the fused Wan q/k RMSNorm + RoPE Pallas kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.diffusion_attention.qk_norm_rope import qk_norm_rope

pytestmark = pytest.mark.tpu_test


def _reference(qkv, wq, wk, cos_half, sin_half, eps, head_dim):
    """float64 norm (rounded to bf16 like the model) + interleaved RoPE."""
    qkv = np.asarray(qkv, np.float64)
    dim = qkv.shape[1] // 3
    c = np.asarray(cos_half, np.float64)[:, None, :]
    s = np.asarray(sin_half, np.float64)[:, None, :]
    outs = []
    for j, w in ((0, wq), (1, wk)):
        x = qkv[:, j * dim : (j + 1) * dim]
        y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
        y = y * np.asarray(w, np.float64)
        y = np.asarray(jnp.asarray(y, jnp.bfloat16), np.float64)
        y = y.reshape(x.shape[0], -1, head_dim // 2, 2)
        y0, y1 = y[..., 0], y[..., 1]
        out = np.stack((y0 * c - y1 * s, y0 * s + y1 * c), -1)
        outs.append(out.reshape(x.shape[0], dim))
    return outs


@pytest.mark.parametrize("rows,heads", [(8190, 40), (300, 4), (16, 2)])
def test_qk_norm_rope_matches_reference(rows, heads):
    head_dim, eps = 128, 1e-6
    dim = heads * head_dim
    kx, kw, kp = jax.random.split(jax.random.PRNGKey(0), 3)
    qkv = (3 * jax.random.normal(kx, (rows, 3 * dim))).astype(jnp.bfloat16)
    wq, wk = (1 + 0.1 * jax.random.normal(kw, (2, dim))).astype(jnp.bfloat16)
    pos = jax.random.randint(kp, (rows, 1), 0, 64).astype(jnp.float32)
    freqs = 1.0 / 10000 ** (jnp.arange(0, head_dim, 2) / head_dim)
    cos_half = jnp.cos(pos * freqs).astype(jnp.bfloat16)
    sin_half = jnp.sin(pos * freqs).astype(jnp.bfloat16)

    q, k = qk_norm_rope(
        qkv,
        wq,
        wk,
        jnp.repeat(cos_half, 2, axis=-1),
        jnp.repeat(sin_half, 2, axis=-1),
        eps=eps,
        head_dim=head_dim,
    )
    ref = _reference(qkv, wq, wk, cos_half, sin_half, eps, head_dim)
    for out, want in zip((q, k), ref):
        out = np.asarray(out, np.float64)
        assert out.shape == want.shape and np.all(np.isfinite(out))
        err = np.abs(out - want)
        # One bf16 rounding of values up to ~8 (half an ulp is 1/64).
        assert err.max() <= 2.0 / 64, err.max()
        assert err.mean() < 2e-3, err.mean()
