# SPDX-License-Identifier: Apache-2.0
"""CPU checks for scale layouts consumed by the fused PCP projection."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3.projection_scale import (
    normalize_projection_scale, projection_scale_layout)


@pytest.mark.parametrize("shape", [(), (1, ), (1, 1)])
def test_tensor_scale_broadcasts_without_changing_values(shape):
    scale = jnp.full(shape, 0.125, dtype=jnp.float32)
    np.testing.assert_array_equal(
        normalize_projection_scale((256, 512), scale), np.full(256, 0.125))


def test_channel_scale_keeps_existing_layout():
    scale = jnp.arange(256, dtype=jnp.float32)
    assert normalize_projection_scale((256, 512), scale) is scale


def test_block_grid_and_linear_runtime_layout_agree():
    grid = jnp.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected = np.repeat(np.asarray(grid), 128, axis=0).T
    linear_runtime = jnp.asarray(expected)[None, :, None, :]
    np.testing.assert_array_equal(normalize_projection_scale((256, 512), grid),
                                  expected)
    np.testing.assert_array_equal(
        normalize_projection_scale((256, 512), linear_runtime), expected)


def test_n_only_blocks_use_full_k_fast_path():
    grid = jnp.array([[2.], [3.]])
    np.testing.assert_array_equal(normalize_projection_scale((256, 512), grid),
                                  np.repeat([2., 3.], 128))


@pytest.mark.parametrize("shape", [(3, ), (3, 4), (2, 3), (2, 8), (0, 4),
                                   (2, 0), (2, 4, 1, 256), (1, 4, 2, 256),
                                   (1, 4, 1, 128)])
def test_invalid_scale_geometry_is_rejected(shape):
    with pytest.raises(ValueError):
        projection_scale_layout((256, 512), shape)
