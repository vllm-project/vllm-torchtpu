# SPDX-License-Identifier: Apache-2.0
"""Single-token projection preserves quantization, within BF16 rounding error."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul.single_token import single_token_mxu
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)

pytestmark = pytest.mark.tpu_test


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(12)
    weight = jnp.asarray(rng.normal(size=(1024, 32768)), jnp.float8_e4m3fn)
    scale = jnp.asarray(rng.uniform(0.001, 0.2, size=32768), jnp.float32)
    return weight, scale


@pytest.mark.parametrize("case", range(8))
def test_single_token_projection_against_float64_reference(weights, case):
    weight, scale = weights
    rng = np.random.default_rng(case)
    values = rng.normal(size=(1, 1024)).astype(np.float32)
    if case == 0:
        values[:] = 0
    elif case == 1:
        values[:] = 1
    elif case == 2:
        values *= 1e-8
    elif case == 3:
        values[0, 0] = 1000
    x = jnp.asarray(values, jnp.bfloat16)
    actual = np.asarray(jax.jit(single_token_mxu)(x, weight, scale)).astype(np.float32)
    baseline = np.asarray(jax.jit(xla_quantized_matmul)(x, weight, scale)).astype(
        np.float32
    )
    spacing = np.exp2(np.floor(np.log2(np.maximum(np.abs(baseline), 1e-30))) - 7)
    assert np.all(np.abs(actual - baseline) <= np.maximum(spacing, 1e-7))
    xq, xs = jax.jit(quantize_tensor, static_argnums=1)(x, weight.dtype)
    columns = np.arange(0, weight.shape[1], 257)
    a = np.asarray(xq).astype(np.float64)
    b = np.asarray(weight[:, columns]).astype(np.float64)
    scales = np.asarray(xs).astype(np.float64) * np.asarray(scale)[columns]
    reference = (a @ b) * scales
    # Float32 summation error plus a final BF16 round-to-nearest. The sum
    # bound uses K*eps/(1-K*eps), independent of the candidate's implementation.
    epsilon = np.finfo(np.float32).eps
    gamma = a.shape[1] * epsilon / (1 - a.shape[1] * epsilon)
    sum_bound = gamma * (np.abs(a) @ np.abs(b)) * np.abs(scales)
    rounded_spacing = np.exp2(
        np.floor(np.log2(np.maximum(np.abs(reference), 1e-30))) - 7
    )
    assert np.all(
        np.abs(actual[:, columns] - reference) <= sum_bound + rounded_spacing / 2
    )


@pytest.mark.parametrize("shape", [(2, 1024), (1, 512)])
def test_single_token_projection_rejects_unmeasured_shapes(weights, shape):
    weight, scale = weights
    with pytest.raises(AssertionError):
        jax.jit(single_token_mxu)(jnp.zeros(shape, jnp.bfloat16), weight, scale)


def test_single_token_projection_rejects_block_scales(weights):
    weight, scale = weights
    with pytest.raises(AssertionError):
        jax.jit(single_token_mxu)(
            jnp.zeros((1, 1024), jnp.bfloat16), weight, scale[None, :]
        )
