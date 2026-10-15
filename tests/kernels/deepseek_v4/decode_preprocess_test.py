# SPDX-License-Identifier: Apache-2.0
"""Exact regression against separate query normalization and key rotation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import rope
from vllm_torchtpu.kernels.deepseek_v4.qkv_preprocess import qkv_preprocess
from vllm_torchtpu.utils import get_device_name

pytestmark = pytest.mark.tpu_test


@pytest.fixture(scope="module")
def rotary_cache():
    angles = np.arange(65540)[:, None] * np.linspace(0.001, 0.01, 32)
    return jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32)


@pytest.mark.parametrize(
    "batch,heads", [(1, 64), (4, 64), (8, 64), (16, 64), (32, 64), (64, 64), (2, 8)]
)
def test_joint_preprocessing(batch, heads, rotary_cache):
    rng = np.random.default_rng(726 + batch)
    positions = jnp.asarray(np.resize([0, 1, 65535, 65539], batch), jnp.int32)
    old = jax.jit(
        lambda q, kv: (
            rope.qnorm_rope(q, positions, rotary_cache, eps=1e-6),
            rope.rope(kv, positions, rotary_cache),
        )
    )
    new = jax.jit(
        lambda q, kv: qkv_preprocess(q, kv, positions, rotary_cache, eps=1e-6)
    )
    for magnitude in (0, 0.01, 32):
        q = jnp.asarray(rng.normal(size=(batch, heads, 512)) * magnitude, jnp.bfloat16)
        kv = jnp.asarray(rng.normal(size=(batch, 512)) * magnitude, jnp.bfloat16)
        expected, actual = old(q, kv), new(q, kv)
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    hlo = new.lower(q, kv).compile().as_text()
    assert ("qkv_preprocess_joint" in hlo) == (
        heads == 64 and get_device_name() == "TPU v7"
    )
