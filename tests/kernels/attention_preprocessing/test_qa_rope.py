# SPDX-License-Identifier: Apache-2.0
"""Q-A/KV + native RMSNorm + RoPE from per-step gathered rows.

Config tests run anywhere; kernel tests need TPU (the manual weight DMA is not
discharged by interpret mode).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.attention_preprocessing import (
    qa_rope,
    reference,
    rope_gather,
)

RANK, HEAD_DIM, HIDDEN = 1280, 512, 5120


def _inputs(tokens, positions, cache_rows=4096, hidden=HIDDEN):
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(size=(tokens, hidden)), jnp.bfloat16)
    # Zero row (scale 1 branch) and a tiny row near the FP8 underflow edge.
    x = x.at[0].set(0).at[1].multiply(1e-6)
    weight = jnp.asarray(rng.normal(size=(hidden, RANK + HEAD_DIM)), jnp.float8_e4m3fn)
    scale = jnp.asarray(rng.uniform(0.01, 0.1, RANK + HEAD_DIM), jnp.float32)
    gamma_qr = jnp.asarray(rng.normal(size=(RANK,)), jnp.bfloat16)
    gamma_kv = jnp.asarray(rng.normal(size=(HEAD_DIM,)), jnp.bfloat16)
    angles = rng.uniform(-3, 3, size=(cache_rows, 32))
    cache = jnp.asarray(
        np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32
    )
    rows = rope_gather.gather_rows(cache, jnp.asarray(positions, jnp.int32))
    return x, weight, scale, gamma_qr, gamma_kv, rows


def _split(rows):
    half = rows.shape[1] // 2
    return rows[:, :half], rows[:, half:]


def _expected(x, weight, scale, gamma_qr, gamma_kv, rows, eps=1e-6):
    """Dense chain in epic order: linear (BF16 out) -> native norm -> RoPE."""
    projected = reference.projection(x, weight, scale).astype(jnp.float32)
    qr = qa_rope.native_rms_norm(projected[:, :RANK], gamma_qr[None, :], eps)
    kv = qa_rope.native_rms_norm(projected[:, RANK:], gamma_kv[None, :], eps)
    return qr, kv, reference.rotate(kv, *_split(rows))


def _positions(tokens, kind):
    if kind == "contiguous":
        return np.arange(tokens) + 17
    if kind == "jumps":  # request boundaries
        half = tokens // 2
        return np.concatenate([np.arange(half) + 3000, np.arange(tokens - half)])
    positions = np.arange(tokens) + 4096 - tokens // 2  # past the table end
    positions[:4] = -1
    return positions


def _requires_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")


# ----------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("tokens", "block"),
    [
        (2048, 256),
        (6144, 256),
        (384, 192),
        (288, 144),
        (256, 256),
        (240, 240),
        (96, 96),
        (112, 112),
    ],
)
def test_default_config_tiles_every_token(tokens, block):
    config = qa_rope.default_config(tokens)
    assert config.token_block == block
    qa_rope.check_config(config, tokens, 1792)
    qa_rope.check_config(config, tokens, 512)  # kv-only width


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (qa_rope.QARopeConfig(token_block=200), "token_block"),
        (qa_rope.QARopeConfig(token_block=256, weight_chunk=384), "weight_chunk"),
        (qa_rope.QARopeConfig(token_block=256, weight_chunk=100), "weight_chunk"),
    ],
)
def test_invalid_config_is_rejected(config, match):
    with pytest.raises(ValueError, match=f"Invalid QARopeConfig.*{match}"):
        qa_rope.check_config(config, 2048, 1792)


def test_native_norm_rounds_before_the_weight_product():
    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.normal(size=(64, 512)) * 7, jnp.float32)
    gamma = jnp.asarray(rng.normal(size=(512,)), jnp.bfloat16)
    actual = qa_rope.native_rms_norm(values, gamma[None, :], 1e-6)
    inv = 1 / np.sqrt(np.mean(np.square(np.asarray(values)), -1, keepdims=True) + 1e-6)
    normalized = jnp.asarray(np.asarray(values) * inv, jnp.float32).astype(jnp.bfloat16)
    # torch: x.to(bf16) * weight is one BF16-rounded multiply.
    expected = (normalized.astype(jnp.float32) * gamma.astype(jnp.float32)).astype(
        jnp.bfloat16
    )
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    )


# ----------------------------------------------------------------- kernel


@pytest.mark.parametrize(
    ("tokens", "kind"),
    [
        (2048, "contiguous"),
        (288, "jumps"),
        (240, "contiguous"),
        (100, "clamped"),
        (96, "jumps"),
        (6144, "contiguous"),
    ],
)
def test_matches_dense_chain(tokens, kind):
    _requires_tpu()
    args = _inputs(tokens, _positions(tokens, kind))
    qr_ref, _, kv_ref = _expected(*args)
    qr, kv = jax.jit(qa_rope.fused_qa_kv_norm_rope)(*args)
    assert qr.shape == (tokens, RANK) and kv.shape == (tokens, HEAD_DIM)
    for actual, expected in ((qr, qr_ref), (kv, kv_ref)):
        _assert_close_ulps(actual, expected)


def _assert_close_ulps(actual, expected, max_ulps=2, max_fraction=1e-4):
    """The RMSNorm reduction and rsqrt differ between Mosaic and XLA, so a few
    elements land 1-2 BF16 ulps apart before RoPE (TPU census: at most 2 ulp,
    under 5e-5 of elements; W1 saw the same against the epic chain). The
    matmul and the in-kernel RoPE are bitwise (test_modes_agree_with_full_output).
    Inputs whose RoPE output cancels to near zero can amplify this further, so
    keep these cases free of that by construction rather than widening the bound."""
    a = np.asarray(actual).view(np.uint16).astype(np.int32)
    e = np.asarray(expected).view(np.uint16).astype(np.int32)
    # Sign-magnitude BF16 bits -> monotonic integers, so |a - e| counts ulps.
    a = np.where(a & 0x8000, 0x8000 - a, a)
    e = np.where(e & 0x8000, 0x8000 - e, e)
    ulps = np.abs(a - e)
    assert ulps.max() <= max_ulps, f"max {ulps.max()} ulp"
    assert np.count_nonzero(ulps) <= max_fraction * ulps.size


@pytest.mark.parametrize("tokens", [288, 2048])
def test_modes_agree_with_full_output(tokens):
    _requires_tpu()
    args = _inputs(tokens, _positions(tokens, "jumps"))
    # Modes are compared with the kernel's own full output: same arithmetic,
    # so bitwise. (Against the dense chain see test_matches_dense_chain.)
    qr_full, rotated = jax.jit(qa_rope.fused_qa_kv_norm_rope)(*args)
    qr_kept, pre_rope = jax.jit(
        functools.partial(qa_rope.fused_qa_kv_norm_rope, rotate_kv=False)
    )(*args)
    _, kv_only = jax.jit(
        functools.partial(qa_rope.fused_qa_kv_norm_rope, emit_qr=False)
    )(*args)
    np.testing.assert_array_equal(
        np.asarray(kv_only, np.float32), np.asarray(rotated, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(qr_kept, np.float32), np.asarray(qr_full, np.float32)
    )
    # The in-kernel RoPE alone is bitwise against the dense rotation.
    cos, sin = _split(args[-1])
    np.testing.assert_array_equal(
        np.asarray(rotated, np.float32),
        np.asarray(reference.rotate(pre_rope, cos, sin), np.float32),
    )


@pytest.mark.parametrize(("tokens", "block"), [(2048, 128), (288, 96), (288, 48)])
def test_block_sizes_do_not_change_values(tokens, block):
    _requires_tpu()
    args = _inputs(tokens, _positions(tokens, "contiguous"))
    default = jax.jit(qa_rope.fused_qa_kv_norm_rope)(*args)
    other = jax.jit(
        functools.partial(
            qa_rope.fused_qa_kv_norm_rope,
            config=qa_rope.QARopeConfig(token_block=block),
        )
    )(*args)
    for a, b in zip(default, other):
        np.testing.assert_array_equal(
            np.asarray(a, np.float32), np.asarray(b, np.float32)
        )
