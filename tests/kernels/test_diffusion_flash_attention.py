# SPDX-License-Identifier: Apache-2.0
"""Numerics tests for the diffusion flash-attention Pallas kernel."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.diffusion_attention import block_sizes as bs
from vllm_torchtpu.kernels.diffusion_attention.kernel import (
    FIXED_M_SAFE_BOUND,
    LOG2_E,
    fixed_m_operands,
    flash_attention_bshd,
)

pytestmark = pytest.mark.tpu_test


def _reference(q, k, v, scale):
    """f32 softmax attention over (B, S, H, D) inputs."""
    heads_per_kv = q.shape[2] // k.shape[2]
    k = jnp.repeat(k, heads_per_kv, axis=2).astype(jnp.float32)
    v = jnp.repeat(v, heads_per_kv, axis=2).astype(jnp.float32)
    s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k) * scale
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v)


def _inputs(batch, q_len, kv_len, heads, kv_heads, head_dim, seed=0):
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(kq, (batch, q_len, heads, head_dim), jnp.float32)
    k = jax.random.normal(kk, (batch, kv_len, kv_heads, head_dim), jnp.float32)
    v = jax.random.normal(kv, (batch, kv_len, kv_heads, head_dim), jnp.float32)
    return q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16)


def _check(out, ref, max_rel=3e-2, mean_rel=1e-2):
    out = np.asarray(out, np.float32)
    ref = np.asarray(ref, np.float32)
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out))
    err = np.abs(out - ref)
    assert err.max() / np.abs(ref).max() < max_rel, err.max()
    assert err.mean() / np.abs(ref).mean() < mean_rel, err.mean()


@pytest.mark.parametrize(
    "batch,q_len,kv_len,heads,kv_heads,head_dim,fixed_m",
    [
        # Self-attention spanning several K/V blocks with ragged tails.
        (1, 4133, 4133, 4, 4, 128, True),
        (1, 4133, 4133, 4, 4, 128, False),
        # Batch folding and GQA.
        (2, 2600, 2600, 4, 2, 128, True),
        # Cross-attention: K/V fits one block (online path only).
        (1, 3000, 512, 4, 4, 128, True),
        # Narrow heads use the transposing layout.
        (1, 2304, 2304, 2, 2, 64, True),
    ],
)
def test_matches_reference(batch, q_len, kv_len, heads, kv_heads, head_dim, fixed_m):
    q, k, v = _inputs(batch, q_len, kv_len, heads, kv_heads, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    out = jax.jit(
        lambda q, k, v: flash_attention_bshd(q, k, v, scale=scale, use_fixed_m=fixed_m)
    )(q, k, v)
    assert out.dtype == jnp.bfloat16
    _check(out, _reference(q, k, v, scale))


def test_outlier_heads_fall_back_to_online():
    """Heads above the fixed-m gate must stay exact via online softmax."""
    q, k, v = _inputs(1, 4133, 4133, 4, 4, 128, seed=1)
    # Head 0 gets logits far above the gate; keys get a large common offset
    # (removed by the mean-centring) on head 1.
    q = q.at[:, :, 0].multiply(12.0)
    k = k.at[:, :, 1].add(3.0)
    scale = 1.0 / math.sqrt(128)
    q_scaled = (q.astype(jnp.float32) * (scale * LOG2_E)).astype(jnp.bfloat16)
    k32 = k.astype(jnp.float32)
    k_smoothed = (k32 - k32.mean(axis=1, keepdims=True)).astype(jnp.bfloat16)
    flags = np.asarray(
        fixed_m_operands(q_scaled, k_smoothed, gate=FIXED_M_SAFE_BOUND, seq_axis=1)
    )[1]
    assert flags[0] == 0.0 and flags[1:].min() == 1.0
    out = jax.jit(lambda q, k, v: flash_attention_bshd(q, k, v, scale=scale))(q, k, v)
    # Reference on the bf16 pre-scaled query the kernel consumes: with logits
    # this large, rounding q * scale * log2(e) to bf16 is itself a few percent.
    _check(out, _reference(q_scaled, k, v, 1.0 / LOG2_E))


def test_gate_constant_is_inside_flush_window():
    # Row max >= 0 after k-smoothing, so the gap m - rowmax <= ceil(bound)
    # must stay below 88 + 126 for the row-max term to remain normal f32.
    assert math.ceil(FIXED_M_SAFE_BOUND) < 88 + 126


def test_block_sizes():
    spec = bs.V7_SPEC
    online = bs.autotune_block_sizes(75600, 128, 2, spec, 75600, "online")
    fixed = bs.autotune_block_sizes(75600, 128, 2, spec, 75600, "fixedm")
    assert online["block_kv"] == fixed["block_kv"] == spec.safe_kv_block
    assert fixed["block_q"] <= 4096
    assert fixed["block_q"] % spec.mxu_dim == 0
    assert online["block_q"] > fixed["block_q"]
    # K/V shorter than the safe tile uses one full-extent block.
    cross = bs.autotune_block_sizes(32760, 128, 2, spec, 512)
    assert cross["block_kv"] == cross["block_kv_compute"] == 512
    assert bs.autotune_block_sizes(32760, 128, 2, spec, 769)["block_kv"] == 769
    # Short queries use one full-extent block.
    tiny = bs.autotune_block_sizes(100, 128, 2, spec, 100)
    assert tiny["block_q"] == 100 and tiny["block_kv"] == 100
