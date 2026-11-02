# SPDX-License-Identifier: Apache-2.0
"""Numerical and staging-layout coverage for packed Engram fusion."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.engram.kernel import (
    dequantize_rows,
    fused_engram,
    gated_residual,
)
from vllm_torchtpu.kernels.engram.tuned_block_sizes import get_block_sizes
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

pytestmark = pytest.mark.cpu_test


def inputs(tokens=64, width=256, streams=4, dim=128, seed=53):
    rng = np.random.default_rng(seed)
    raw = (
        np.asarray(
            jnp.asarray(rng.normal(size=(tokens, width)), jnp.float32).astype(
                jnp.float8_e4m3fn
            )
        )
        .view(np.uint8)
        .copy()
    )
    scales = rng.integers(123, 130, (tokens, width // 32), dtype=np.uint8)
    raw[::7] = 127  # FP8 NaN must be isolated by the invalid scale sentinel.
    scales[::7] = 255
    packed = np.concatenate((raw.reshape(-1), scales.reshape(-1))).view(np.int32)
    return (
        jnp.asarray(packed),
        jnp.asarray(rng.normal(size=(tokens, streams, dim)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(width, (streams + 1) * dim)), jnp.float8_e4m3fn),
        jnp.asarray(rng.uniform(0.005, 0.03, ((streams + 1) * dim,)), jnp.float32),
        jnp.asarray(rng.normal(size=(streams, dim)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(streams, dim)), jnp.bfloat16),
        jnp.asarray(np.arange(tokens) % 3 != 0),
    )


def reference(packed, hidden, weight, scale, q, k, mask):
    tokens, streams, dim = hidden.shape
    width = weight.shape[0]
    raw_end = tokens * width // 4
    raw = packed[:raw_end].reshape(tokens, width // 4)
    scales = packed[raw_end:].reshape(tokens, width // 32 // 4)
    embedded = dequantize_rows(raw, scales)
    projection = xla_quantized_matmul(embedded, weight, scale)
    return gated_residual(hidden, projection, q, k, mask, 1e-20, 1e-6)


@pytest.mark.parametrize("tokens", [32, 64, 288])
def test_fused_engram_matches_unfused_reference(tokens):
    data = inputs(tokens)
    expected = jax.jit(reference)(*data)
    actual = jax.jit(
        functools.partial(fused_engram, eps=1e-20, clamp_value=1e-6, interpret=True)
    )(*data)
    actual, expected = (np.asarray(x.astype(jnp.float32)) for x in (actual, expected))
    # The source PR's numerical contract; keep it fixed across candidates.
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.01)
    np.testing.assert_array_equal(
        actual[~np.asarray(data[-1])],
        np.asarray(data[1].astype(jnp.float32))[~np.asarray(data[-1])],
    )
    assert np.isfinite(actual).all()


def test_decode_tile_reads_the_weight_once():
    assert get_block_sizes(288, 6144, 4, 5120) == (288, 256)


def test_packed_rows_keep_raw_and_scale_regions_separate():
    data = inputs(32)
    packed, hidden, weight, *_ = data
    width = weight.shape[0]
    raw_end = hidden.shape[0] * width // 4
    embedded = dequantize_rows(
        packed[:raw_end].reshape(32, width // 4),
        packed[raw_end:].reshape(32, width // 128),
    )
    assert np.all(np.asarray(embedded)[::7] == 0)
    assert np.isfinite(np.asarray(embedded.astype(jnp.float32))).all()


def test_rejects_a_packed_buffer_with_the_wrong_capacity():
    packed, *rest = inputs(32)
    with pytest.raises(AssertionError):
        fused_engram(packed[:-1], *rest, 1e-20, 1e-6, interpret=True)
