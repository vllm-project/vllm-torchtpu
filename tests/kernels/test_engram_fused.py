# SPDX-License-Identifier: Apache-2.0
"""Numerical and staging-layout coverage for packed Engram fusion."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.engram.acc_kernel import fused_engram_acc
from vllm_torchtpu.kernels.engram.kernel import dequantize_rows, gated_residual
from vllm_torchtpu.kernels.engram.tuned_block_sizes import get_block_sizes
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul

pytestmark = pytest.mark.cpu_test
# Explicit MXU accumulators have no CPU interpreter; the gate and row decoding
# below remain CPU-checked, the full kernel only on TPU.
on_tpu = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu", reason="explicit MXU accumulators need a TPU"
)


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
    return reference_gate(hidden, projection, q, k, mask)


def reference_gate(hidden, projection, q, k, mask):
    streams, dim = hidden.shape[-2:]
    keys = projection[:, : streams * dim].reshape(-1, streams, dim).astype(jnp.float32)
    values = projection[:, streams * dim :].astype(jnp.float32)
    h = hidden.astype(jnp.float32)
    dot = jnp.sum(h * q * k * keys, axis=-1)
    dot *= jax.lax.rsqrt(jnp.mean(h * h, axis=-1) + 1e-20)
    dot *= jax.lax.rsqrt(jnp.mean(keys * keys, axis=-1) + 1e-20) * dim**-0.5
    magnitude = jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6))
    gate = jax.nn.sigmoid(jnp.where(dot < 0, -magnitude, magnitude)) * mask[:, None]
    return (h + gate[..., None] * values[:, None, :]).astype(hidden.dtype)


def test_gate_matches_the_original_gate():
    packed, hidden, weight, scale, q, k, mask = inputs(32)
    projection = xla_quantized_matmul(
        dequantize_rows(
            packed[: 32 * 64].reshape(32, 64), packed[32 * 64 :].reshape(32, 2)
        ),
        weight,
        scale,
    )
    actual = gated_residual(hidden, projection, q, k, mask, 1e-20, 1e-6)
    expected = reference_gate(hidden, projection, q, k, mask)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@on_tpu
@pytest.mark.parametrize("tokens", [32, 64, 288])
def test_fused_engram_matches_unfused_reference(tokens):
    data = inputs(tokens, dim=1024)
    expected = jax.jit(reference)(*data)
    actual = jax.jit(functools.partial(fused_engram_acc, eps=1e-20, clamp_value=1e-6))(
        *data
    )
    actual, expected = (np.asarray(x.astype(jnp.float32)) for x in (actual, expected))
    # The source PR's numerical contract; keep it fixed across candidates.
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.01)
    np.testing.assert_array_equal(
        actual[~np.asarray(data[-1])],
        np.asarray(data[1].astype(jnp.float32))[~np.asarray(data[-1])],
    )
    assert np.isfinite(actual).all()


@on_tpu
def test_projection_preserves_stream_column_order():
    data = inputs(32, width=256, dim=2048)
    actual = jax.jit(functools.partial(fused_engram_acc, eps=1e-20, clamp_value=1e-6))(
        *data
    )
    expected = jax.jit(reference)(*data)
    np.testing.assert_allclose(
        np.asarray(actual.astype(jnp.float32)),
        np.asarray(expected.astype(jnp.float32)),
        atol=0.02,
        rtol=0.01,
    )


@pytest.mark.parametrize("gate_rows", [16, 32])
def test_gate_token_tiles_preserve_each_rows_reduction(gate_rows):
    _, hidden, _, _, q, k, mask = inputs(64, dim=5120)
    projection = jnp.asarray(
        np.random.default_rng(54).normal(size=(64, 5 * 5120)), jnp.bfloat16
    )
    expected = gated_residual(hidden, projection, q, k, mask, 1e-20, 1e-6)
    actual = jnp.concatenate(
        [
            gated_residual(
                hidden[start : start + gate_rows],
                projection[start : start + gate_rows],
                q,
                k,
                mask[start : start + gate_rows],
                1e-20,
                1e-6,
            )
            for start in range(0, 64, gate_rows)
        ]
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


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
        fused_engram_acc(packed[:-1], *rest, 1e-20, 1e-6)
