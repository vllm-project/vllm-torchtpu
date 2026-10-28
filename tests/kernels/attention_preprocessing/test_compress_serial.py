# SPDX-License-Identifier: Apache-2.0
"""Serial compressor kernel matches the dense epic/dsv41 compressor reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.attention_preprocessing import (
    compress_reference,
    compress_serial,
)


def make_inputs(tokens, hidden, ratio, seed=7):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(tokens, hidden)) * 0.5, jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(hidden, ratio * 512)) * 0.05, jnp.bfloat16)
    gamma = jnp.asarray(rng.normal(size=(512,)), jnp.bfloat16)
    # Two requests: the first starts mid-group (needs its ring row), the second
    # starts on a group boundary inside the second token tile.
    split = tokens // 2 + 3
    has_prev = np.ones((tokens,), bool)
    has_prev[0] = has_prev[split] = False
    positions = np.concatenate((np.arange(11, 11 + split), np.arange(tokens - split)))
    ring = jnp.asarray(rng.normal(size=(tokens, ratio * 512)), jnp.float32)
    valid = np.ones((tokens,), bool) if ratio == 1 else positions % 2 == 1
    valid[-5:] = False  # padding rows
    starts = positions // ratio * ratio
    angles = rng.normal(size=(4096, 32))
    cos = jnp.asarray(np.cos(angles)[starts], jnp.float32)
    sin = jnp.asarray(np.sin(angles)[starts], jnp.float32)
    return x, weight, gamma, jnp.asarray(has_prev), ring, jnp.asarray(valid), cos, sin


@pytest.mark.cpu_test
@pytest.mark.parametrize("ratio", [1, 2])
def test_matches_reference(ratio):
    x, w, g, has_prev, ring, valid, cos, sin = make_inputs(512, 256, ratio)
    expected = jax.jit(compress_reference.compress, static_argnames=("ratio",))(
        x, w, g, has_prev, ring, valid, cos, sin, ratio=ratio
    )
    actual = compress_serial.fused_compress_serial(
        x,
        w,
        g,
        has_prev.astype(jnp.int32)[:, None],
        ring,
        valid.astype(jnp.int32)[:, None],
        cos,
        sin,
        ratio=ratio,
        interpret=True,
    )
    score, latent, words, scales = actual
    np.testing.assert_allclose(score, expected[0], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(latent, np.float32),
        np.asarray(expected[1], np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_allclose(scales[:, 0], expected[3], rtol=1e-2)
    # Words are packed FP8 bytes; compare where both encodings agree in latent.
    same_latent = np.all(np.asarray(latent) == np.asarray(expected[1]), axis=1)
    assert same_latent.mean() > 0.95
    np.testing.assert_array_equal(
        np.asarray(words)[same_latent], np.asarray(expected[2])[same_latent]
    )
    assert np.all(np.asarray(latent)[~np.asarray(valid)] == 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("ratio", [1, 2])
def test_pipelined_matches_serial_bitwise(ratio):
    x, w, g, has_prev, ring, valid, cos, sin = make_inputs(512, 256, ratio)
    args = (
        x,
        w,
        g,
        has_prev.astype(jnp.int32)[:, None],
        ring,
        valid.astype(jnp.int32)[:, None],
        cos,
        sin,
    )
    serial_out = compress_serial.fused_compress_serial(
        *args, ratio=ratio, interpret=True
    )
    piped_out = compress_serial.fused_compress_pipelined(
        *args, ratio=ratio, interpret=True
    )
    frag_out = compress_serial.fused_compress_fragmented(
        *args, ratio=ratio, interpret=True
    )
    for a, b, c in zip(serial_out, piped_out, frag_out):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))
