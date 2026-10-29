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
"""`quantize_rows` against the XLA quantization it replaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_rows,
    quantize_tensor,
    xla_quantized_matmul,
)

FP8 = jnp.float8_e4m3fn


def _rows(m: int, k: int) -> jax.Array:
    """bf16 rows over many magnitudes, plus an all-zero row and a -0."""
    key_x, key_mag = jax.random.split(jax.random.key(m * 131 + k))
    x = jax.random.normal(key_x, (m, k), jnp.float32)
    x = x * jnp.exp(jax.random.uniform(key_mag, (m, 1), minval=-30, maxval=30))
    return x.at[0].set(0).at[1, 5].set(-0.0).astype(jnp.bfloat16)


def _assert_same_bits(got: tuple, want: tuple) -> None:
    (got_q, got_s), (want_q, want_s) = got, want
    assert got_q.shape == want_q.shape and got_s.shape == want_s.shape
    np.testing.assert_array_equal(
        np.asarray(got_q).view(np.uint8), np.asarray(want_q).view(np.uint8)
    )
    np.testing.assert_array_equal(
        np.asarray(got_s).view(np.uint32), np.asarray(want_s).view(np.uint32)
    )


# One row group, several groups, a partial last block, rows that are not a
# group multiple, and k that is not a column-chunk multiple.
@pytest.mark.parametrize(
    "m,k", [(16, 1024), (64, 1024), (300, 1024), (48, 512), (8, 640)]
)
def test_quantize_rows_matches_quantize_tensor(m, k):
    x = _rows(m, k)
    _assert_same_bits(quantize_rows(x, FP8, interpret=True), quantize_tensor(x, FP8))


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "m,k,n", [(128, 8192, 512), (1024, 4096, 1536), (16, 1024, 256)]
)
def test_one_pass_quantized_matmul_is_bit_identical_on_tpu(m, k, n):
    if jax.default_backend() != "tpu":
        pytest.fail(f"needs the TPU backend, got {jax.default_backend()}")
    x = _rows(m, k)
    key_w, key_s = jax.random.split(jax.random.key(1))
    w_q = jax.random.normal(key_w, (k, n), jnp.float32).astype(FP8)
    w_s = jax.random.uniform(key_s, (n,), jnp.float32)

    _assert_same_bits(
        jax.jit(quantize_rows, static_argnums=1)(x, FP8),
        jax.jit(quantize_tensor, static_argnums=1)(x, FP8),
    )
    want = jax.jit(xla_quantized_matmul)(x, w_q, w_s)
    got = jax.jit(lambda *a: xla_quantized_matmul(*a, quantize_in_one_pass=True))(
        x, w_q, w_s
    )
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )
