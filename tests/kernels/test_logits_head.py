# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import logits_head as kernel


def test_supported_shapes():
    assert kernel.is_supported_shape(240, (129280, 5120))
    assert not kernel.is_supported_shape(288, (129280, 5120))
    assert not kernel.is_supported_shape(240, (129280, 4096))


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="Pallas TPU kernel")
@pytest.mark.parametrize("seed", [0, 1])
def test_logits_head_matches_xla_dot_within_one_bf16_ulp(seed):
    keys = jax.random.split(jax.random.PRNGKey(seed))
    hidden = jax.random.normal(keys[0], (240, 5120)).astype(jnp.bfloat16)
    weight = (jax.random.normal(keys[1], (129280, 5120)) * 0.02).astype(jnp.bfloat16)
    want = jax.jit(
        lambda h, w: jnp.dot(h, w.T, preferred_element_type=jnp.float32).astype(
            jnp.bfloat16
        )
    )(hidden, weight)
    got = jax.jit(kernel.logits_head)(hidden, weight)
    want, got = np.asarray(want, np.float32), np.asarray(got, np.float32)
    # Only the FP32 accumulation order differs: one BF16 rounding step, or an
    # FP32-noise-sized difference next to zero.
    bound = np.maximum(np.abs(want), np.abs(got)) * 2.0**-7 + 1e-4
    assert np.all(np.abs(got - want) <= bound)
    assert np.mean(got != want) < 1e-3
    np.testing.assert_array_equal(got.argmax(-1), want.argmax(-1))
