# SPDX-License-Identifier: Apache-2.0
"""Exact regression for inline rotary-table gathering in output projection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import o_projection, o_projection_gather
from vllm_torchtpu.utils import get_device_name

pytestmark = pytest.mark.tpu_test


@pytest.fixture(scope="module")
def rotary_cache():
    angles = np.arange(65540)[:, None] * np.linspace(0.001, 0.01, 32)
    return jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32)


@pytest.fixture(scope="module")
def projection_weights():
    rng = np.random.default_rng(139)
    return (
        jnp.asarray(rng.normal(0, 0.03, (4096, 8192)), jnp.float8_e4m3fn),
        jnp.asarray(rng.uniform(0.3, 2.0, 8192), jnp.float32),
    )


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32, 64, 2])
def test_inline_projection_gather(batch, rotary_cache, projection_weights):
    w, scale = projection_weights
    positions = jnp.asarray(np.resize([0, 1, 65535, 65539], batch), jnp.int32)
    old = jax.jit(
        lambda x: o_projection.fused_reverse_rope_wo_a_projection(
            x, positions, rotary_cache, w, scale, head_dim=512
        )
    )
    new = jax.jit(
        lambda x: o_projection_gather.fused(
            x, positions, rotary_cache, w, scale, head_dim=512
        )
    )
    rng = np.random.default_rng(803 + batch)
    for magnitude in (0, 0.01, 32):
        x = jnp.asarray(rng.normal(size=(batch, 64, 512)) * magnitude, jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(old(x)), np.asarray(new(x)))
    hlo = new.lower(x).compile().as_text()
    assert ("wo_a_inline_cos_sin_once" in hlo) == (
        batch != 2 and get_device_name() == "TPU v7"
    )
