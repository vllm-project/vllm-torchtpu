# SPDX-License-Identifier: Apache-2.0
"""Check fused selection across slot and tile boundaries, including IEEE ties."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.markov_sampling import sample_slot


@pytest.mark.parametrize("slot", [0, 4])
def test_slot_selection_preserves_ties_nans_and_last_vocabulary_entry(slot):
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled MXU kernel requires TPU")
    x = jnp.zeros((48, 256), jnp.bfloat16)
    weight = jnp.zeros((129280, 256), jnp.bfloat16)
    base = jnp.full((240, 129280), -10, jnp.bfloat16)
    values = jnp.full((48, 129280), -10, jnp.bfloat16)
    expected = np.arange(48, dtype=np.int64) * 2600
    expected[-1] = 129279
    values = values.at[jnp.arange(48), jnp.asarray(expected)].set(10)
    values = values.at[0, :].set(-jnp.inf)
    expected[0] = 0
    values = values.at[1, 16383].set(jnp.nan).at[1, 16384].set(jnp.nan)
    expected[1] = 16383
    values = values.at[2, :].set(-1).at[2, 16383].set(-0.0).at[2, 16384].set(0.0)
    expected[2] = 16383
    base = base.at[slot * 48 : (slot + 1) * 48].set(values)
    actual = jax.jit(functools.partial(sample_slot, slot=slot))(x, weight, base)
    np.testing.assert_array_equal(actual, expected)


def test_projection_changes_the_selected_token():
    if jax.default_backend() != "tpu":
        pytest.skip("Compiled MXU kernel requires TPU")
    x = jnp.ones((48, 256), jnp.bfloat16)
    weight = jnp.zeros((129280, 256), jnp.bfloat16).at[129279].set(1)
    base = jnp.zeros((240, 129280), jnp.bfloat16).at[:, 1].set(100)
    actual = jax.jit(functools.partial(sample_slot, slot=4))(x, weight, base)
    np.testing.assert_array_equal(actual, np.full(48, 129279))


def test_unsupported_shape_is_rejected():
    with pytest.raises(AssertionError):
        sample_slot(
            jnp.zeros((1, 256)), jnp.zeros((128, 256)), jnp.zeros((5, 128)), slot=0
        )
