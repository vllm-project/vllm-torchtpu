# SPDX-License-Identifier: Apache-2.0
"""Serial projection fusion preserves rounding, token tiles, and head order."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.attention_preprocessing import reference, serial


def make_inputs(tokens=512, hidden=128, rank=128, head_dim=128, heads=16):
    rng = np.random.default_rng(31)

    def normal(shape, dtype):
        return jnp.asarray(rng.normal(size=shape), dtype)

    x = normal((tokens, hidden), jnp.bfloat16)
    # Cover the zero-scale branch and nonzero values below the pseudocode clamp.
    x = x.at[0].set(0).at[1].multiply(1e-6)
    weight = normal((hidden, rank + head_dim), jnp.float8_e4m3fn)
    scale = jnp.asarray(rng.uniform(0.01, 0.1, rank + head_dim), jnp.float32)
    gamma_qr = normal((rank,), jnp.bfloat16)
    gamma_kv = normal((head_dim,), jnp.bfloat16)
    angles = normal((tokens, 32), jnp.float32)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    qb = normal((rank, heads * head_dim), jnp.float8_e4m3fn)
    qb_scale = jnp.asarray(rng.uniform(0.01, 0.1, heads * head_dim), jnp.float32)
    return (x, weight, scale, gamma_qr, gamma_kv, cos, sin), (qb, qb_scale)


def assert_close(actual, expected):
    # Existing BF16 RMSNorm/RoPE tests use the same 1e-2 bounds.
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=1e-2,
        rtol=1e-2,
    )
    assert actual.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(actual)))


@pytest.mark.cpu_test
@pytest.mark.parametrize(("tokens", "head_dim"), [(256, 128), (512, 256)])
def test_two_kernels_preserve_rounding_and_grid_order(tokens, head_dim):
    args, qb_args = make_inputs(tokens=tokens, head_dim=head_dim)
    qr, kv = serial.fused_qa_kv_norm_swa_serial(*args, interpret=True)
    qr_ref, kv_ref = jax.jit(reference.qa_kv_norm_swa)(*args)
    assert_close(qr, qr_ref)
    assert_close(kv, kv_ref)
    np.testing.assert_array_equal(np.asarray(qr[0]), 0)
    np.testing.assert_array_equal(np.asarray(kv[0]), 0)
    cos, sin = args[-2:]
    q = serial.fused_qb_rope_serial(
        qr, *qb_args, cos, sin, head_dim=head_dim, interpret=True
    )
    q_ref = jax.jit(reference.qb_rope, static_argnames=("head_dim",))(
        qr, *qb_args, cos, sin, head_dim=head_dim
    )
    assert q.shape == (tokens, 16, head_dim)
    assert_close(q, q_ref)
    # Identity coefficients preserve every projected channel, including the tail.
    identity = serial.fused_qb_rope_serial(
        qr,
        *qb_args,
        jnp.ones_like(cos),
        jnp.zeros_like(sin),
        head_dim=head_dim,
        interpret=True,
    )
    projected = reference.projection(qr, *qb_args).reshape(identity.shape)
    assert_close(identity, projected)


@pytest.mark.cpu_test
def test_full_workload_shapes_and_invalid_rows():
    array = jax.ShapeDtypeStruct
    bf16, fp8, fp32 = jnp.bfloat16, jnp.float8_e4m3fn, jnp.float32
    args = (
        array((2048, 5120), bf16),
        array((5120, 1792), fp8),
        array((1792,), fp32),
        array((1280,), bf16),
        array((512,), bf16),
        array((2048, 32), fp32),
        array((2048, 32), fp32),
    )
    qr, kv = jax.eval_shape(serial.fused_qa_kv_norm_swa_serial, *args)
    assert qr.shape == (2048, 1280) and kv.shape == (2048, 512)
    q = jax.eval_shape(
        serial.fused_qb_rope_serial,
        qr,
        array((1280, 32768), fp8),
        array((32768,), fp32),
        *args[-2:],
    )
    assert q.shape == (2048, 64, 512)
    with pytest.raises(ValueError, match="Pad token rows"):
        jax.eval_shape(
            serial.fused_qa_kv_norm_swa_serial,
            array((288, 5120), bf16),
            *args[1:],
        )
    with pytest.raises(ValueError, match="Weight scales must be FP32"):
        jax.eval_shape(
            serial.fused_qa_kv_norm_swa_serial,
            *args[:2],
            array((1792,), bf16),
            *args[3:],
        )
    with pytest.raises(ValueError, match="multiple of 128"):
        jax.eval_shape(
            serial.fused_qa_kv_norm_swa_serial,
            *args[:1],
            array((5120, 1472), fp8),
            array((1472,), fp32),
            args[3],
            array((192,), bf16),
            *args[5:],
        )


def test_full_prefill_on_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")
    args, qb_args = make_inputs(2048, 5120, 1280, 512, 64)
    qr, kv = jax.jit(serial.fused_qa_kv_norm_swa_serial)(*args)
    qr_ref, kv_ref = jax.jit(reference.qa_kv_norm_swa)(*args)
    assert_close(qr, qr_ref)
    assert_close(kv, kv_ref)
    q = jax.jit(serial.fused_qb_rope_serial)(qr, *qb_args, *args[-2:])
    q_ref = jax.jit(reference.qb_rope)(qr, *qb_args, *args[-2:])
    assert_close(q, q_ref)
