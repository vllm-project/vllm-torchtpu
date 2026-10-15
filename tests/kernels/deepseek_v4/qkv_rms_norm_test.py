# SPDX-License-Identifier: Apache-2.0
"""Independent Q/KV statistics and weight-rounding semantics."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.qkv_rms_norm import qkv_rms_norm

pytestmark = pytest.mark.tpu_test


@pytest.mark.parametrize("weight_rank", [1, 2])
@pytest.mark.parametrize("q_width,kv_width", [(128, 256), (1024, 512)])
@pytest.mark.parametrize("input_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("weight_dtype", [jnp.bfloat16, jnp.float32])
def test_qkv_norm_independent_statistics(
    q_width, kv_width, input_dtype, weight_dtype, weight_rank
):
    q = np.tile(np.array([0.5, -0.5], dtype=np.float32), q_width // 2)[None]
    kv = np.tile(np.array([2.0, -2.0], dtype=np.float32), kv_width // 2)[None]
    qw = np.tile(np.array([0.75, 1.25], dtype=np.float32), q_width // 2)
    kw = np.tile(np.array([1.5, 0.5], dtype=np.float32), kv_width // 2)
    x = jnp.asarray(np.concatenate((q, kv), axis=1), input_dtype)
    qw, kw = jnp.asarray(qw, weight_dtype), jnp.asarray(kw, weight_dtype)

    if weight_rank == 2:
        qw, kw = qw.reshape(1, -1), kw.reshape(1, -1)

    def run(value):
        return qkv_rms_norm(value, qw, kw, q_eps=0.75, kv_eps=5.0)

    actual = run(x)
    # Q's reciprocal RMS is 1; KV's is 1/3. Preserve the weight-dtype
    # rounding before and after multiplication, as vLLM RMSNorm does.
    for got, value, weight, factor in zip(actual, (q, kv), (qw, kw), (1.0, 1 / 3)):
        expected = (value * factor).astype(weight_dtype).astype(np.float32)
        expected = (expected * np.asarray(weight).astype(np.float32)).astype(
            weight_dtype
        )
        expected = expected.astype(input_dtype).astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(got).astype(np.float32), expected, rtol=1e-6, atol=1e-6
        )

    changed_q = run(x.at[:, :q_width].multiply(8))
    changed_kv = run(x.at[:, q_width:].multiply(8))
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(changed_q[1]))
    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(changed_kv[0]))
    for output in run(jnp.zeros_like(x)):
        np.testing.assert_array_equal(np.asarray(output), 0)
