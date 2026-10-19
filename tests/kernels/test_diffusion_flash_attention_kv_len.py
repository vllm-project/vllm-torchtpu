# SPDX-License-Identifier: Apache-2.0
"""Key-padded (``kv_len``) path of the diffusion flash-attention Pallas kernel.

The fixed-m operand checks run on any JAX backend (CPU included). The kernel
checks need a TPU backend and are skipped elsewhere.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.diffusion_attention.kernel import (
    FIXED_M_SAFE_BOUND,
    LOG2_E,
    fixed_m_operands,
    flash_attention_bshd,
)


@pytest.fixture
def requires_tpu():
    # Resolved at run time, not collection: initializing a JAX backend during
    # collection claims the TPU for the pytest process (see conftest).
    if jax.default_backend() != "tpu":
        pytest.skip("the Pallas kernel needs a TPU backend")


def _masked_reference(q, k, v, scale, kv_len):
    """f32 softmax attention with a (B, S_kv) prefix key mask (SDPA semantics)."""
    heads_per_kv = q.shape[2] // k.shape[2]
    k = jnp.repeat(k, heads_per_kv, axis=2).astype(jnp.float32)
    v = jnp.repeat(v, heads_per_kv, axis=2).astype(jnp.float32)
    hi = jax.lax.Precision.HIGHEST
    s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k, precision=hi)
    valid = jnp.arange(k.shape[1])[None, :] < jnp.asarray(kv_len)[:, None]
    s = jnp.where(valid[:, None, None, :], s * scale, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v, precision=hi)


def _inputs(batch, seq, heads, kv_heads, head_dim, seed=0):
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(seed), 3)
    q = jax.random.normal(kq, (batch, seq, heads, head_dim), jnp.float32)
    k = jax.random.normal(kk, (batch, seq, kv_heads, head_dim), jnp.float32)
    v = jax.random.normal(kv, (batch, seq, kv_heads, head_dim), jnp.float32)
    return q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16)


def _errors(out, ref):
    out = np.asarray(out, np.float32)
    ref = np.asarray(ref, np.float32)
    assert out.shape == ref.shape
    assert np.all(np.isfinite(out))
    err = np.abs(out - ref)
    return err.max() / np.abs(ref).max(), err.mean() / np.abs(ref).mean()


# -----------------------------------------------------------------------------
# Fixed-m operands over the real keys (any backend).
# -----------------------------------------------------------------------------


def _bound_inputs(seq, used, heads, offset):
    rng = np.random.default_rng(0)
    scale = 1.0 / math.sqrt(128)
    q = rng.normal(size=(1, seq, heads, 128)).astype(np.float32)
    k = rng.normal(size=(1, seq, heads, 128)).astype(np.float32)
    k[:, used:] += offset  # padding rows far from the real keys
    q_scaled = jnp.asarray(q, jnp.bfloat16).astype(jnp.float32) * (scale * LOG2_E)
    return q_scaled, jnp.asarray(k, jnp.bfloat16)


def test_fixed_m_operands_key_valid_uses_real_keys_only():
    seq, used = 640, 601
    q_scaled, k = _bound_inputs(seq, used, heads=4, offset=40.0)
    valid = (jnp.arange(seq) < used)[None, :, None, None]
    masked = np.asarray(
        fixed_m_operands(
            q_scaled, k, gate=FIXED_M_SAFE_BOUND, seq_axis=1, key_valid=valid
        )
    )
    real = np.asarray(
        fixed_m_operands(q_scaled, k[:, :used], gate=FIXED_M_SAFE_BOUND, seq_axis=1)
    )
    unmasked = np.asarray(
        fixed_m_operands(q_scaled, k, gate=FIXED_M_SAFE_BOUND, seq_axis=1)
    )
    # Row 0 is ceil() of the bound: equal bounds up to f32 summation order.
    np.testing.assert_allclose(masked[0], real[0], atol=1.0)
    np.testing.assert_array_equal(masked[1], real[1])
    # Padding keys would loosen the bound (and may fail the gate) otherwise.
    assert np.all(unmasked[0] > real[0] + 1.0)


def test_fixed_m_operands_all_valid_matches_unmasked():
    q_scaled, k = _bound_inputs(512, 512, heads=3, offset=0.0)
    valid = jnp.ones((1, 512, 1, 1), bool)
    masked = np.asarray(
        fixed_m_operands(
            q_scaled, k, gate=FIXED_M_SAFE_BOUND, seq_axis=1, key_valid=valid
        )
    )
    unmasked = np.asarray(
        fixed_m_operands(q_scaled, k, gate=FIXED_M_SAFE_BOUND, seq_axis=1)
    )
    np.testing.assert_allclose(masked[0], unmasked[0], atol=1.0)
    np.testing.assert_array_equal(masked[1], unmasked[1])


# -----------------------------------------------------------------------------
# Kernel numerics (TPU).
# -----------------------------------------------------------------------------


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch,seq,kv_len,heads,kv_heads,fixed_m",
    [
        # MiniMax-H3 256x256 T2VA scale at TP8 (7 heads per TensorCore):
        # used % 64 != 0, so the packed row ends in padding.
        (1, 2240, (2201,), 7, 7, True),
        (1, 2240, (2201,), 7, 7, False),
        # 832x480-class and long sequences.
        (1, 12288, (12230,), 7, 7, True),
        # Ragged tail, padding inside the last K/V block.
        (1, 4133, (4100,), 4, 4, True),
        (1, 4133, (4100,), 4, 4, False),
        # Real keys end inside the first K/V block: later blocks are all
        # padding and must contribute nothing.
        (1, 2240, (100,), 7, 7, True),
        (1, 2240, (100,), 7, 7, False),
        # Batch rows with different lengths, GQA.
        (2, 2600, (2500, 1999), 4, 2, True),
        (2, 2600, (2600, 2600), 4, 2, True),
        (2, 4133, (4133, 4100), 4, 4, True),
    ],
)
def test_kv_len_matches_masked_reference(
    requires_tpu, batch, seq, kv_len, heads, kv_heads, fixed_m
):
    q, k, v = _inputs(batch, seq, heads, kv_heads, 128)
    scale = 1.0 / math.sqrt(128)
    lengths = jnp.asarray(kv_len, jnp.int32)
    out = jax.jit(
        lambda q, k, v, n: flash_attention_bshd(
            q, k, v, scale=scale, use_fixed_m=fixed_m, kv_len=n
        )
    )(q, k, v, lengths)
    assert out.dtype == jnp.bfloat16
    ref = _masked_reference(q, k, v, scale, lengths)
    max_rel, mean_rel = _errors(out, ref)
    assert max_rel < 3e-2, max_rel
    assert mean_rel < 1e-2, mean_rel
    # Valid-row error stays at the unmasked kernel's level: the same kernel on
    # K/V truncated to the real keys (one batch row at a time).
    for b, n in enumerate(kv_len):
        trunc = flash_attention_bshd(
            q[b : b + 1],
            k[b : b + 1, :n],
            v[b : b + 1, :n],
            scale=scale,
            use_fixed_m=fixed_m,
        )
        trunc_max, _ = _errors(trunc, ref[b : b + 1])
        row_max, _ = _errors(out[b : b + 1], ref[b : b + 1])
        assert row_max <= 2.0 * trunc_max + 2e-3, (b, row_max, trunc_max)


@pytest.mark.tpu_test
def test_kv_len_full_length_matches_unmasked_kernel(requires_tpu):
    q, k, v = _inputs(1, 2240, 7, 7, 128, seed=3)
    scale = 1.0 / math.sqrt(128)
    full = flash_attention_bshd(
        q, k, v, scale=scale, kv_len=jnp.asarray([2240], jnp.int32)
    )
    plain = flash_attention_bshd(q, k, v, scale=scale)
    np.testing.assert_allclose(
        np.asarray(full, np.float32), np.asarray(plain, np.float32), atol=2e-2
    )
    max_rel, _ = _errors(full, _masked_reference(q, k, v, scale, [2240]))
    assert max_rel < 3e-2
