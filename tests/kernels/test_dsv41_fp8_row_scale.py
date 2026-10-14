# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token FP32 scale serialization, quantization and Pallas dequantization."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import words_to_slabs
from vllm_torchtpu.kernels.deepseek_v41.fp8_cache import (
    NOPE_DIM,
    dequant_fp8_nope,
    encode_fp8_kv,
)


def test_fp8_record_has_one_fp32_scale_and_zero_padding():
    rng = np.random.default_rng(31)
    latent = jnp.asarray(
        rng.normal(size=(37, 512)) * rng.uniform(0.01, 30, (37, 1)), jnp.bfloat16
    )
    latent = latent.at[0, :NOPE_DIM].set(0)
    words, rope = jax.jit(encode_fp8_kv)(latent)
    assert words.shape == rope.shape == (37, 128)
    payload = np.asarray(words_to_slabs(words)).reshape(37, 512)
    scale = payload[:, NOPE_DIM : NOPE_DIM + 4].copy().view("<f4")
    nope = np.asarray(latent, np.float32)[:, :NOPE_DIM]
    amax = np.abs(nope).max(axis=-1, keepdims=True)
    np.testing.assert_allclose(scale, np.maximum(amax, 1e-4) / 448, rtol=2e-7)
    assert np.any(np.log2(scale[1:]) % 1 != 0)  # Full FP32, not E8M0 exponents.
    assert not payload[:, NOPE_DIM + 4 :].any()
    values = payload[:, :NOPE_DIM].view(jnp.float8_e4m3fn).astype(np.float32)
    # Half of the largest E4M3 spacing is 16, at a maximum magnitude of 448.
    error = np.abs(values * scale - nope)
    assert np.all(error <= np.maximum(amax, 1e-4) / 28 + 1e-5)
    np.testing.assert_array_equal(
        np.asarray(rope[:, :64]).astype(np.uint16),
        np.asarray(latent[:, NOPE_DIM:]).view(np.uint16),
    )
    assert not np.asarray(rope[:, 64:]).any()


def test_pallas_dequant_uses_fp32_scale_and_ignores_padding():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(32, NOPE_DIM)).astype(jnp.float8_e4m3fn)
    scales = rng.uniform(0.001, 20, (32, 1)).astype("<f4")
    payload = rng.integers(0, 256, (32, 512), dtype=np.uint8)
    payload[:, :NOPE_DIM] = values.view(np.uint8)
    payload[:, NOPE_DIM : NOPE_DIM + 4] = scales.view(np.uint8)

    def kernel(x_ref, out_ref):
        out_ref[...] = dequant_fp8_nope(x_ref[...])

    decoded = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((32, NOPE_DIM), jnp.bfloat16),
        grid=(1,),
        in_specs=[pl.BlockSpec((32, 512), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((32, NOPE_DIM), lambda i: (i, 0)),
        interpret=jax.default_backend() != "tpu",
    )(jnp.asarray(payload))
    expected = (values.astype(np.float32) * scales).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(decoded).view(np.uint16), expected.view(np.uint16)
    )
