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
"""Compact GDN inputs preserve the existing cast and reshape contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput
from vllm_torchtpu.kernels.qwen4_exp.gdn_input import compact_qkv_input


@pytest.mark.parametrize(
    "tokens,width,dtype",
    [
        (16, 10240, jnp.bfloat16),
        (127, 10240, jnp.bfloat16),
        pytest.param(128, 10240, jnp.bfloat16, marks=pytest.mark.tpu_test),
        (257, 10240, jnp.bfloat16),
        pytest.param(8192, 10240, jnp.bfloat16, marks=pytest.mark.tpu_test),
        pytest.param(128, 12288, jnp.bfloat16, marks=pytest.mark.tpu_test),
        pytest.param(128, 16384, jnp.bfloat16, marks=pytest.mark.tpu_test),
        (128, 24576, jnp.bfloat16),
        pytest.param(128, 256, jnp.bfloat16, marks=pytest.mark.tpu_test),
        (128, 133, jnp.bfloat16),
        (128, 10240, jnp.float32),
    ],
)
def test_compact_qkv_preserves_value_bits(tokens, width, dtype):
    # A wrong token/feature mapping or narrowing before conversion changes
    # these bits, including on the fallback paths and across tile boundaries.
    rng = np.random.default_rng(54)
    values = rng.standard_normal((tokens, width), dtype=np.float32)
    values *= np.exp2(rng.integers(-10, 11, size=(tokens, 1)))
    values[0, :4] = [0.0, -0.0, 1.0, -1.0]
    qkv = jnp.asarray(values, dtype=dtype)
    expected = qkv.astype(jnp.float32).reshape(tokens, 1, width)

    actual = jax.jit(compact_qkv_input)(qkv)

    assert actual.shape == expected.shape
    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32),
        np.asarray(expected).view(np.uint32),
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "tokens,state_dtype",
    [(128, jnp.bfloat16), (8192, jnp.bfloat16), (128, jnp.float32)],
)
def test_prepared_input_preserves_recurrence_and_decode(tokens, state_dtype):
    rng = np.random.default_rng(54)
    n_kq, n_v, dim, width = 16, 48, 128, 10240

    def normal(shape, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(0, 0.02, shape), dtype)

    weight = normal((width, 1, 4))
    bias = normal((width,))
    a_log = normal((n_v,), jnp.float32)
    dt_bias = normal((n_v,), jnp.float32) - 2
    states = (
        normal((2, 3, width), jnp.float32),
        normal((2, n_v, dim, dim), state_dtype),
    )
    prepared_states = states

    def run(states, qkv, b, a, context, *, prepared):
        count = qkv.shape[0]
        inputs = Conv1DInput(compact_qkv_input(qkv), qkv.dtype) if prepared else qkv
        return wrapper.fused_conv1d_gdn(
            inputs,
            b,
            a,
            *states,
            weight,
            bias,
            a_log,
            dt_bias,
            jnp.array([0, count], jnp.int32),
            jnp.array([1], jnp.int32),
            jnp.array([0, 1, 1] if count > 1 else [1, 1, 1], jnp.int32),
            jnp.array([context + count], jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=dim,
            d_v=dim,
            kernel_size=4,
        )

    compiled = jax.jit(run, static_argnames=("prepared",))
    context = 256
    for count in (tokens, 1, 1, 1):
        qkv, b, a = normal((count, width)), normal((count, n_v)), normal((count, n_v))
        states, expected = compiled(states, qkv, b, a, context, prepared=False)
        prepared_states, actual = compiled(
            prepared_states, qkv, b, a, context, prepared=True
        )
        assert actual.dtype == expected.dtype == qkv.dtype
        for got, want in zip((actual, *prepared_states), (expected, *states)):
            got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
            assert np.isfinite(got).all() and np.isfinite(want).all()
            np.testing.assert_array_equal(got, want)
        context += count
