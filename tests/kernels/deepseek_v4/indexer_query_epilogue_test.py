# SPDX-License-Identifier: Apache-2.0
"""Indexer epilogue follows the actual head-weight dtype and model scale."""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.query_epilogue import (
    rope_quant_fold_weights,
)

pytestmark = pytest.mark.tpu_test


@pytest.mark.parametrize("batch", [1, 4, 64])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("scale", [128**-0.5, 1 / math.sqrt(128), 0.125])
def test_weight_dtype_and_scale(batch, dtype, scale):
    rng = np.random.default_rng(batch)
    q = jnp.asarray(rng.normal(size=(batch, 64, 128)), jnp.bfloat16)
    pos = jnp.arange(batch, dtype=jnp.int32)
    angles = rng.normal(size=(128, 32)).astype(np.float32)
    cs = jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), 1))
    weights = jnp.asarray(rng.normal(size=(batch, 64)), dtype)

    def reference(q, weights):
        quant, scales = rope_quant(q, pos, cs)
        return quant, weights * scale * (64**-0.5) * scales

    expected = jax.jit(reference)(q, weights)
    fn = jax.jit(functools.partial(rope_quant_fold_weights, softmax_scale=scale))
    actual = fn(q, pos, cs, weights)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
