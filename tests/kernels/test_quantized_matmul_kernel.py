# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for quantized matmul kernels.

This is a pytest port of
../tpu-inference/tests/kernels/quantized_matmul_kernel_test.py.
"""

import itertools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import rope
from vllm_torchtpu.kernels.quantized_matmul import (
    blockwise_kernel,
    fused_quantize_kernel,
    kernel,
    tuned_block_sizes,
    util,
)

pytestmark = pytest.mark.tpu_test

jax.config.update("jax_numpy_dtype_promotion", "standard")

xla_quantized_matmul = util.xla_quantized_matmul
per_channel_kernel = kernel.quantized_matmul_kernel
blockwise_kernel = blockwise_kernel.quantized_matmul_kernel
quantize_tensor = util.quantize_tensor
TunedValue = tuned_block_sizes.TunedValue


def _require_tpu_v7() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")

    device_kind = jax.devices()[0].device_kind
    match = re.match(r"^TPU[^\d]*(\d+)", device_kind)
    if match is None or int(match.group(1)) < 7:
        pytest.fail(f"Expected TPU v7+, got {device_kind}.")


def _assert_allclose(actual, expected, atol=0.5, rtol=0.5) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol
    )


def reference_block_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    block_size: int,
    x_q_dtype: jnp.dtype,
) -> jax.Array:
    """Pure JAX reference for blockwise quantized matmul."""
    n_batch, n_in = x.shape
    n_out, _ = w_q.shape

    if n_in % block_size != 0:
        raise ValueError(
            f"Input dimension {n_in} not divisible by block_size {block_size}"
        )

    x_reshaped = x.reshape(n_batch, -1, block_size)
    x_q, x_s = util.quantize_block(x_reshaped, axis=2, target_dtype=x_q_dtype)

    w_q_reshaped = w_q.reshape(n_out, -1, block_size)
    w_s_aligned = w_scale.transpose(1, 0, 2)

    dot_blocks = jnp.einsum(
        "bnk, onk -> bno", x_q.astype(jnp.float32), w_q_reshaped.astype(jnp.float32)
    )
    scaled_blocks = dot_blocks * x_s * w_s_aligned
    out = jnp.sum(scaled_blocks, axis=1)

    return out.astype(x.dtype)


def _test_quantized_matmul(
    dtype: jnp.dtype,
    q_dtype: jnp.dtype,
    bs: int,
    n_input_features: int,
    n_output_features: int,
    quantize_activation: bool,
    tuned_value=None,
    atol=0.5,
    rtol=0.5,
    block_size: int | None = None,
    x_q_dtype: jnp.dtype | None = None,
) -> None:
    _require_tpu_v7()

    prng_key = jax.random.key(1234)
    k0, k1 = jax.random.split(prng_key, 2)
    x = jax.random.uniform(k0, (bs, n_input_features), dtype=dtype, minval=0, maxval=1)
    w = jax.random.uniform(
        k1, (n_output_features, n_input_features), dtype=dtype, minval=-1, maxval=1
    )

    w_q, w_scale = quantize_tensor(w, q_dtype, block_size=block_size)
    if block_size is None:
        w_scale = jnp.squeeze(w_scale)
        assert w_scale.shape == (n_output_features,)
    else:
        assert w_scale.shape == (n_input_features // block_size, 1, n_output_features)

    if x_q_dtype is None:
        x_q_dtype = w_q.dtype if quantize_activation else dtype

    kernel_fn = per_channel_kernel if block_size is None else blockwise_kernel
    output = kernel_fn(
        x,
        w_q,
        w_scale,
        block_size=block_size,
        x_q_dtype=x_q_dtype,
        tuned_value=tuned_value,
    )

    if block_size is None:
        # `w_q` is N-major `[n_out, n_in]` for the kernel under test;
        # `xla_quantized_matmul` takes the canonical (k, n) layout.
        expected = xla_quantized_matmul(
            x, w_q.T, w_scale, quantize_activation=quantize_activation
        )
    else:
        expected = reference_block_quantized_matmul(
            x, w_q, w_scale, block_size, x_q_dtype
        )

    _assert_allclose(output, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    (
        "dtype",
        "q_dtype",
        "bs",
        "n_input_features",
        "n_output_features",
        "quantize_activation",
    ),
    itertools.product(
        [jnp.bfloat16, jnp.float32],
        [jnp.int8, jnp.float8_e4m3fn],
        [128, 256, 512],
        [128, 256, 512],
        [128, 256, 512],
        [True],
    ),
)
def test_quantized_matmul_various_input_shapes(
    dtype: jnp.dtype,
    q_dtype: jnp.dtype,
    bs: int,
    n_input_features: int,
    n_output_features: int,
    quantize_activation: bool,
) -> None:
    # bf16 / fp8_e4m3fn at (256, 512, 128) has one borderline element
    # (~0.55 abs diff); loosen atol just for this combo. Mirrors the same
    # carve-out in tpu-inference's quantized_matmul_kernel_test.
    noisy = (
        dtype == jnp.bfloat16
        and q_dtype == jnp.float8_e4m3fn
        and bs == 256
        and n_input_features == 512
        and n_output_features == 128
    )
    _test_quantized_matmul(
        dtype,
        q_dtype,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=quantize_activation,
        tuned_value=None,
        atol=0.7 if noisy else 0.5,
    )


@pytest.mark.parametrize(
    (
        "dtype",
        "q_dtype",
        "bs",
        "n_input_features",
        "n_output_features",
        "quantize_activation",
    ),
    itertools.product(
        [jnp.bfloat16, jnp.float32],
        [jnp.int8, jnp.float8_e4m3fn],
        [64, 192],
        [64, 192],
        [64, 192],
        [True],
    ),
)
def test_quantized_matmul_unaligned_input_shapes(
    dtype: jnp.dtype,
    q_dtype: jnp.dtype,
    bs: int,
    n_input_features: int,
    n_output_features: int,
    quantize_activation: bool,
) -> None:
    _test_quantized_matmul(
        dtype,
        q_dtype,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=quantize_activation,
        tuned_value=None,
    )


@pytest.mark.parametrize(
    (
        "dtype",
        "q_dtype",
        "bs",
        "n_input_features",
        "n_output_features",
        "quantize_activation",
    ),
    [
        (jnp.bfloat16, jnp.int8, 128, 1280, 8192, True),
        (jnp.bfloat16, jnp.int8, 128, 28672, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 4096, 14336, True),
        (jnp.bfloat16, jnp.int8, 128, 4096, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 6144, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 7168, 8192, True),
        (jnp.bfloat16, jnp.int8, 128, 8192, 1024, True),
        (jnp.bfloat16, jnp.int8, 128, 8192, 3584, True),
    ],
)
def test_quantized_matmul_use_tuned_block_sizes(
    dtype: jnp.dtype,
    q_dtype: jnp.dtype,
    bs: int,
    n_input_features: int,
    n_output_features: int,
    quantize_activation: bool,
) -> None:
    _test_quantized_matmul(
        dtype,
        q_dtype,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=quantize_activation,
        tuned_value=None,
    )


@pytest.mark.parametrize(
    ("dtype", "bs", "n_input_features", "n_output_features"),
    itertools.product(
        [jnp.bfloat16, jnp.float32],
        [512, 1024],
        [512, 1024],
        [512, 1024, 2048],
    ),
)
def test_quantized_matmul_blockwise_w8a8(
    dtype: jnp.dtype,
    bs: int,
    n_input_features: int,
    n_output_features: int,
) -> None:
    _test_quantized_matmul(
        dtype,
        jnp.float8_e4m3fn,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=True,
        tuned_value=TunedValue(512, 512, 512, 2),
        block_size=512,
        x_q_dtype=jnp.float8_e4m3fn,
    )


def test_quantized_matmul_blockwise_rejects_unaligned_input_features() -> None:
    _require_tpu_v7()

    x = jnp.ones((512, 768), dtype=jnp.bfloat16)
    w_q = jnp.ones((512, 768), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.ones((2, 1, 512), dtype=jnp.float32)

    with pytest.raises(ValueError, match="must be divisible"):
        blockwise_kernel(
            x,
            w_q,
            w_scale,
            block_size=512,
            x_q_dtype=jnp.float8_e4m3fn,
            tuned_value=TunedValue(512, 512, 512, 2),
        )


def test_quantized_matmul_blockwise_rejects_mismatched_scale_blocks() -> None:
    _require_tpu_v7()

    x = jnp.ones((512, 1024), dtype=jnp.bfloat16)
    w_q = jnp.ones((512, 1024), dtype=jnp.float8_e4m3fn)
    w_scale = jnp.ones((1, 1, 512), dtype=jnp.float32)

    with pytest.raises(ValueError, match="w_scale block dim"):
        blockwise_kernel(
            x,
            w_q,
            w_scale,
            block_size=512,
            x_q_dtype=jnp.float8_e4m3fn,
            tuned_value=TunedValue(512, 512, 512, 2),
        )


@pytest.mark.parametrize(
    ("dtype", "bs", "n_input_features", "n_output_features"),
    [
        (jnp.bfloat16, 512, 512, 1024),
        (jnp.bfloat16, 512, 512, 512),
        (jnp.bfloat16, 512, 1024, 512),
        (jnp.bfloat16, 512, 1024, 1024),
        (jnp.bfloat16, 512, 1024, 2048),
        (jnp.float32, 512, 512, 1024),
        (jnp.float32, 512, 512, 512),
        (jnp.float32, 512, 1024, 512),
        (jnp.float32, 512, 1024, 1024),
        (jnp.float32, 512, 1024, 2048),
    ],
)
def test_quantized_matmul_blockwise_w4a8(
    dtype: jnp.dtype,
    bs: int,
    n_input_features: int,
    n_output_features: int,
) -> None:
    _test_quantized_matmul(
        dtype,
        jnp.float4_e2m1fn,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=True,
        tuned_value=TunedValue(512, 512, 512, 2),
        block_size=512,
        x_q_dtype=jnp.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    ("dtype", "bs", "n_input_features", "n_output_features"),
    [
        (jnp.bfloat16, 512, 512, 1024),
        (jnp.bfloat16, 512, 512, 512),
        (jnp.bfloat16, 512, 1024, 512),
        (jnp.bfloat16, 512, 1024, 1024),
        (jnp.bfloat16, 512, 1024, 2048),
        (jnp.float32, 512, 512, 1024),
        (jnp.float32, 512, 512, 512),
        (jnp.float32, 512, 1024, 512),
        (jnp.float32, 512, 1024, 1024),
        (jnp.float32, 512, 1024, 2048),
    ],
)
def test_quantized_matmul_blockwise_int4_fp8(
    dtype: jnp.dtype,
    bs: int,
    n_input_features: int,
    n_output_features: int,
) -> None:
    _test_quantized_matmul(
        dtype,
        jnp.int4,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation=True,
        tuned_value=TunedValue(512, 512, 512, 2),
        block_size=512,
        x_q_dtype=jnp.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    ("bs", "n_input_features", "n_output_features"),
    [
        (1024, 8192, 4096),
        (1024, 4096, 1536),
        (512, 2048, 768),
        (1024, 1024, 32768),
        (512, 512, 8192),
    ],
)
def test_fused_quantized_matmul_matches_xla(
    bs: int, n_input_features: int, n_output_features: int
) -> None:
    """The fused kernel must reproduce `xla_quantized_matmul` bit for bit."""
    _require_tpu_v7()
    kx, kw, ks = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(kx, (bs, n_input_features), jnp.float32) * 3
    x = x.at[5].set(0).at[7, 3].set(1e4).astype(jnp.bfloat16)
    w_q = (jax.random.normal(kw, (n_input_features, n_output_features)) * 100).astype(
        jnp.float8_e4m3fn
    )
    w_scale = jax.random.uniform(ks, (n_output_features,), jnp.float32, 1e-3, 1e-2)
    assert fused_quantize_kernel.supports(x, w_q, w_scale)

    actual = fused_quantize_kernel.fused_quantized_matmul(x, w_q, w_scale)
    expected = jax.jit(xla_quantized_matmul)(x, w_q, w_scale)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16)
    )


@pytest.mark.parametrize("emit_normed", [True, False])
@pytest.mark.parametrize(
    ("bs", "n_input_features", "n_output_features"),
    [(1024, 4096, 1536), (512, 2048, 768)],
)
def test_fused_rms_norm_quantized_matmul_matches_xla(
    bs: int, n_input_features: int, n_output_features: int, emit_normed: bool
) -> None:
    """The norm-fused kernel must reproduce XLA's RMSNorm, then the matmul."""
    _require_tpu_v7()
    kx, kg, kw, ks = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(kx, (bs, n_input_features), jnp.float32) * 3
    x = x.at[5].set(0).at[7, 3].set(1e4).astype(jnp.bfloat16)
    gamma = (1 + 0.3 * jax.random.normal(kg, (n_input_features,))).astype(jnp.bfloat16)
    w_q = (jax.random.normal(kw, (n_input_features, n_output_features)) * 100).astype(
        jnp.float8_e4m3fn
    )
    w_scale = jax.random.uniform(ks, (n_output_features,), jnp.float32, 1e-3, 1e-2)
    assert fused_quantize_kernel.supports_rms_norm(x, gamma, w_q, w_scale)

    @jax.jit
    def rms_norm(x):
        # vLLM's `rms_norm`, rounding before the weight multiply as the served
        # program does; reduce_precision keeps XLA from eliding that rounding.
        x_f32 = x.astype(jnp.float32)
        rstd = jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
        x_scaled = jax.lax.reduce_precision(x_f32 * rstd, 8, 7)
        return rstd, (x_scaled * gamma.astype(jnp.float32)).astype(jnp.bfloat16)

    rstd, expected_normed = rms_norm(x)
    expected = jax.jit(xla_quantized_matmul)(expected_normed, w_q, w_scale)
    actual = fused_quantize_kernel.fused_rms_norm_quantized_matmul(
        x, rstd, gamma, w_q, w_scale, emit_normed=emit_normed
    )
    if emit_normed:
        actual, actual_normed = actual
        np.testing.assert_array_equal(
            np.asarray(actual_normed).view(np.uint16),
            np.asarray(expected_normed).view(np.uint16),
        )
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16)
    )


@pytest.mark.parametrize(
    ("bs", "num_heads"),
    [(1024, 64), (128, 64), (96, 16)],
)
def test_fused_quantized_matmul_rope_quant_matches_rope_quant(
    bs: int, num_heads: int
) -> None:
    """The kernel must reproduce `rope_quant` of `xla_quantized_matmul`."""
    _require_tpu_v7()
    n_input_features, head_dim, max_position = 1024, 128, 4096
    kx, kw, ks, kc, kp = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(kx, (bs, n_input_features), jnp.float32) * 3
    x = x.at[5].set(0).at[7, 3].set(1e4).astype(jnp.bfloat16)
    w_q = (
        jax.random.normal(kw, (n_input_features, num_heads * head_dim)) * 100
    ).astype(jnp.float8_e4m3fn)
    w_q = w_q.at[:, head_dim : 2 * head_dim].set(0)  # a head of zero rows
    w_scale = jax.random.uniform(ks, (num_heads * head_dim,), jnp.float32, 1e-3, 1e-2)
    cos_sin_cache = jax.random.uniform(kc, (max_position, 64), jnp.float32, -1, 1)
    positions = jax.random.randint(kp, (bs,), 0, max_position, jnp.int32)
    assert fused_quantize_kernel.supports_rope_quant(x, w_q, w_scale, num_heads)

    q = jax.jit(xla_quantized_matmul)(x, w_q, w_scale)
    expected_q, expected_scale = rope.rope_quant(
        q.reshape(bs, num_heads, head_dim),
        positions,
        cos_sin_cache,
        quant_dtype=jnp.float8_e4m3fn,
    )
    actual_q, actual_scale = fused_quantize_kernel.fused_quantized_matmul_rope_quant(
        x, w_q, w_scale, positions, cos_sin_cache
    )
    np.testing.assert_array_equal(
        np.asarray(actual_q).view(np.uint8), np.asarray(expected_q).view(np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(actual_scale).view(np.uint32),
        np.asarray(expected_scale).view(np.uint32),
    )


@pytest.mark.parametrize(
    ("bs", "num_heads"),
    [(1024, 64), (256, 16), (32, 8)],
)
def test_fused_quantized_matmul_qnorm_rope_matches_qnorm_rope(
    bs: int, num_heads: int
) -> None:
    """The kernel must reproduce `qnorm_rope` of `xla_quantized_matmul`."""
    _require_tpu_v7()
    n_input_features, head_dim, max_position = 1024, 512, 4096
    kx, kw, ks, kc, kp = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(kx, (bs, n_input_features), jnp.float32) * 3
    x = x.at[5].set(0).at[7, 3].set(1e4).astype(jnp.bfloat16)
    # Within e4m3's range, so every head is finite.
    w_q = (jax.random.normal(kw, (n_input_features, num_heads * head_dim)) * 30).astype(
        jnp.float8_e4m3fn
    )
    w_q = w_q.at[:, head_dim : 2 * head_dim].set(0)  # a head of zero rows
    w_scale = jax.random.uniform(ks, (num_heads * head_dim,), jnp.float32, 1e-3, 1e-2)
    cos_sin_cache = jax.random.uniform(kc, (max_position, 64), jnp.float32, -1, 1)
    positions = jax.random.randint(kp, (bs,), 0, max_position, jnp.int32)
    assert fused_quantize_kernel.supports_qnorm_rope(x, w_q, w_scale, head_dim)

    q = jax.jit(xla_quantized_matmul)(x, w_q, w_scale)
    expected = rope.qnorm_rope(
        q.reshape(bs, num_heads, head_dim), positions, cos_sin_cache, eps=1e-6
    )
    actual = fused_quantize_kernel.fused_quantized_matmul_qnorm_rope(
        x, w_q, w_scale, positions, cos_sin_cache, head_dim=head_dim, eps=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16),
        np.asarray(expected).reshape(bs, -1).view(np.uint16),
    )
