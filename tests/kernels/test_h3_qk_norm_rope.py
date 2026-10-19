# SPDX-License-Identifier: Apache-2.0
"""Numerics tests for the MiniMax-H3 variant of the fused q/k norm + RoPE kernel.

``per_head_norm`` + ``half_split_rotary_dim=96``: RMSNorm per 128-lane head
with one shared weight, NeoX half-split rotation of the first 96 lanes, 32
lanes passed through (vLLM-Omni fused_qk_norm_rope.py, half-split mode).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.diffusion_attention.qk_norm_rope import qk_norm_rope

pytestmark = pytest.mark.tpu_test

HEAD_DIM, ROT_DIM = 128, 96


def _tables(rope_table):
    """[T, 96] [cos | sin] -> (T, 128) [c, c, 1...] / [s, s, 0...]."""
    half = ROT_DIM // 2
    cos, sin = rope_table[:, :half], rope_table[:, half:]
    pad = HEAD_DIM - ROT_DIM
    ones = jnp.ones((rope_table.shape[0], pad), rope_table.dtype)
    return (
        jnp.concatenate((cos, cos, ones), -1),
        jnp.concatenate((sin, sin, jnp.zeros_like(ones)), -1),
    )


def _reference(qkv, wq, wk, rope_table, eps):
    """float64 per-head norm (rounded to bf16 like the kernel) + half-split RoPE."""
    qkv = np.asarray(qkv, np.float64)
    dim = qkv.shape[1] // 3
    half = ROT_DIM // 2
    table = np.asarray(rope_table, np.float64)
    c, s = table[:, None, :half], table[:, None, half:]
    outs = []
    for j, w in ((0, wq), (1, wk)):
        x = qkv[:, j * dim : (j + 1) * dim].reshape(qkv.shape[0], -1, HEAD_DIM)
        y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
        y = y * np.asarray(w, np.float64)
        y = np.asarray(jnp.asarray(y, jnp.bfloat16), np.float64)
        first, second = y[..., :half], y[..., half:ROT_DIM]
        out = np.concatenate(
            (first * c - second * s, second * c + first * s, y[..., ROT_DIM:]), -1
        )
        outs.append(out.reshape(x.shape[0], dim))
    return outs


@pytest.mark.parametrize("rows,heads", [(2304, 7), (300, 7), (16, 2)])
def test_h3_qk_norm_rope_matches_reference(rows, heads):
    eps = 1e-5
    dim = heads * HEAD_DIM
    kx, kw, kp = jax.random.split(jax.random.PRNGKey(0), 3)
    qkv = (3 * jax.random.normal(kx, (rows, 3 * dim))).astype(jnp.bfloat16)
    wq, wk = (1 + 0.1 * jax.random.normal(kw, (2, HEAD_DIM))).astype(jnp.bfloat16)
    pos = jax.random.randint(kp, (rows, 1), 0, 64).astype(jnp.float32)
    freqs = 1.0 / 10000 ** (jnp.arange(ROT_DIM // 2) / (ROT_DIM // 2))
    rope_table = jnp.concatenate(
        (jnp.cos(pos * freqs), jnp.sin(pos * freqs)), -1
    ).astype(jnp.bfloat16)
    cos, sin = _tables(rope_table)

    q, k = qk_norm_rope(
        qkv,
        wq,
        wk,
        cos,
        sin,
        eps=eps,
        head_dim=HEAD_DIM,
        per_head_norm=True,
        half_split_rotary_dim=ROT_DIM,
    )
    ref = _reference(qkv, wq, wk, rope_table, eps)
    for out, want in zip((q, k), ref):
        out = np.asarray(out, np.float64)
        assert out.shape == want.shape and np.all(np.isfinite(out))
        err = np.abs(out - want)
        # One bf16 rounding of values up to ~8 (<= 1 bf16 ulp at |y| < 8).
        assert err.max() <= 2.0 / 64, err.max()
        assert err.mean() < 2e-3, err.mean()
