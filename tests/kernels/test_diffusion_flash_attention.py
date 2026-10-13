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
        # Wan 720p single-device cross-attention: the largest query block,
        # sized by the VMEM cap.
        (1, 75600, 512, 1, 1, 128, True),
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
    flags = np.asarray(
        fixed_m_operands(
            q.astype(jnp.float32) * (scale * LOG2_E),
            k,
            gate=FIXED_M_SAFE_BOUND,
            seq_axis=1,
        )
    )[1]
    assert flags[0] == 0.0 and flags[1:].min() == 1.0
    out = jax.jit(lambda q, k, v: flash_attention_bshd(q, k, v, scale=scale))(q, k, v)
    _check(out, _reference(q, k, v, scale))


def test_fixed_m_gate_boundary_accuracy(monkeypatch):
    # Verify per-row accuracy on inputs with a 1e3x dynamic range of query row
    # norms (geomspace 1e-3..1.0): Head 0 sits just below FIXED_M_SAFE_BOUND
    # (fixed-m fast path) and Head 1 sits in (210, 213) where the old 213 gate
    # flushed low-norm rows to zero.
    import vllm_torchtpu.kernels.diffusion_attention.kernel as kernel_mod

    rng = np.random.default_rng(0)
    seq_len, num_heads, head_dim = 4133, 2, 128
    scale = 1.0 / math.sqrt(head_dim)

    k_raw = rng.normal(size=(1, seq_len, num_heads, head_dim)).astype(np.float32)
    k_centered = k_raw - k_raw.mean(axis=1, keepdims=True)
    k = jnp.asarray(k_centered, dtype=jnp.bfloat16)
    # The kernel's fixed-m bound uses f32 scaled queries and f32 centred keys.
    k32 = k.astype(jnp.float32)
    k_smoothed = k32 - k32.mean(axis=1, keepdims=True)

    q_dir = rng.normal(size=(1, seq_len, num_heads, head_dim)).astype(np.float32)
    q_dir /= np.linalg.norm(q_dir, axis=-1, keepdims=True)
    row_profile = np.geomspace(1e-3, 1.0, seq_len, dtype=np.float32).reshape(
        1, seq_len, 1, 1
    )
    k_max_norm = np.asarray(
        kernel_mod.max_row_norm_per_head(k_smoothed, seq_axis=1), dtype=np.float32
    )
    target_bounds = np.array([195.0, 211.5], dtype=np.float32)
    q_peak_norm = target_bounds / (k_max_norm * scale * LOG2_E)
    q = jnp.asarray(
        q_dir * row_profile * q_peak_norm.reshape(1, 1, num_heads, 1),
        dtype=jnp.bfloat16,
    )
    v = jnp.asarray(
        rng.normal(size=(1, seq_len, num_heads, head_dim)), dtype=jnp.bfloat16
    )

    q_scaled = q.astype(jnp.float32) * (scale * LOG2_E)
    bounds = np.asarray(
        kernel_mod.max_row_norm_per_head(q_scaled, seq_axis=1)
        * kernel_mod.max_row_norm_per_head(k_smoothed, seq_axis=1)
    )
    assert 190.0 < bounds[0] <= FIXED_M_SAFE_BOUND < 210.0 < bounds[1] < 213.0

    mk = fixed_m_operands(q_scaled, k_smoothed, gate=FIXED_M_SAFE_BOUND, seq_axis=1)
    assert list(np.asarray(mk[1])) == [1.0, 0.0]

    ref = np.asarray(_reference(q, k, v, scale), dtype=np.float64)
    ref_row_max = np.abs(ref).max(axis=-1)

    out = np.asarray(
        flash_attention_bshd(q, k, v, scale=scale).astype(jnp.float32),
        dtype=np.float64,
    )
    row_rel_err = np.abs(out - ref).max(axis=-1) / ref_row_max
    assert row_rel_err.max() < 3e-2

    # Must-fail calibration check: raising the gate back to 213.0 routes Head 1
    # through fixed-m, flushing its low-norm query rows on TPU.
    monkeypatch.setattr(kernel_mod, "FIXED_M_SAFE_BOUND", 213.0)
    out_213 = np.asarray(
        flash_attention_bshd(q, k, v, scale=scale).astype(jnp.float32),
        dtype=np.float64,
    )
    head1_row_rel_err = (
        np.abs(out_213[0, :, 1, :] - ref[0, :, 1, :]).max(axis=-1)
        / ref_row_max[0, :, 1]
    )
    assert head1_row_rel_err.max() > 0.05


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16), np.float64)


@pytest.mark.parametrize("fixed_m", [True, False])
def test_error_at_bf16_floor(fixed_m):
    """Per-row error vs float64 stays at the floor of a bf16-output kernel.

    Large logits and keys with a common offset, as in Wan's attention, make
    any extra bf16 rounding of the kernel's inputs (pre-scaled q, centred k)
    show up several times above the floor: the error of rounding the
    normalized probabilities and the output to bf16.
    """
    q, k, v = _inputs(1, 4133, 4133, 2, 2, 128, seed=2)
    q = (q.astype(jnp.float32) * 8.0).astype(jnp.bfloat16)
    k = (k.astype(jnp.float32) + 1.0).astype(jnp.bfloat16)
    scale = 1.0 / math.sqrt(128)
    rows = np.random.default_rng(0).choice(4133, size=128, replace=False)
    qn, kn, vn = (np.asarray(x, np.float64) for x in (q, k, v))
    ref, floor = [], []
    for h in range(2):
        s = qn[0, rows, h] @ kn[0, :, h].T * scale
        p = np.exp(s - s.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        ref.append(p @ vn[0, :, h])
        floor.append(_bf16(_bf16(p) @ vn[0, :, h]))
    ref, floor = np.stack(ref, axis=1), np.stack(floor, axis=1)

    def max_row_rel(out):
        err = np.abs(out - ref).max(axis=-1)
        return (err / np.abs(ref).max(axis=-1)).max()

    out = jax.jit(
        lambda q, k, v: flash_attention_bshd(q, k, v, scale=scale, use_fixed_m=fixed_m)
    )(q, k, v)
    out = np.asarray(out.astype(jnp.float32), np.float64)[0, rows]
    assert max_row_rel(out) < 1.5 * max_row_rel(floor)


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
