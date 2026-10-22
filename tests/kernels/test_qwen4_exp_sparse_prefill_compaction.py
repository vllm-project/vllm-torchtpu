# SPDX-License-Identifier: Apache-2.0
"""Sort-free active KV tile compaction for QSA block-sparse prefill."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    _compact_active,
)


def _reference(mask):
    size = mask.shape[-1]
    return np.stack(
        [
            np.asarray(jnp.nonzero(row, size=size, fill_value=0)[0], np.int32)
            for row in mask.reshape(-1, size)
        ]
    ).reshape(mask.shape)


def _cases():
    rng = np.random.default_rng(0)
    yield np.zeros((4, 37), bool)
    yield np.ones((4, 37), bool)
    yield np.eye(16, dtype=bool)
    tail = np.zeros((3, 64), bool)
    tail[:, -1] = True
    yield tail
    head = np.zeros((3, 64), bool)
    head[:, 0] = True
    yield head
    for density in (0.01, 0.2, 0.5, 0.9):
        yield rng.random((9, 129)) < density
    mixed = rng.random((6, 257)) < 0.3
    mixed[1] = False
    mixed[4] = True
    yield mixed
    yield np.zeros((1, 1), bool)
    yield np.ones((1, 1), bool)


@pytest.mark.parametrize("mask", list(_cases()))
def test_compaction_matches_nonzero(mask):
    expected = _reference(mask)
    actual = jax.jit(_compact_active)(jnp.asarray(mask))
    assert actual.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(actual), expected)
    # One-dimensional form, also under vmap and lax.map (the bounded path).
    for row, want in zip(mask, expected):
        np.testing.assert_array_equal(np.asarray(_compact_active(jnp.asarray(row))), want)
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(_compact_active)(jnp.asarray(mask))), expected
    )
    mapped = jax.lax.map(_compact_active, jnp.asarray(mask), batch_size=2)
    np.testing.assert_array_equal(np.asarray(mapped), expected)


def test_compaction_scatter_is_unique_and_sort_free():
    hlo = jax.jit(_compact_active).lower(jnp.zeros((8, 300), bool)).as_text()
    # jnp.nonzero emits a scatter-add with repeated indices, which XLA:TPU
    # lowers through a sort; every scatter here must carry unique indices.
    scatters = hlo.count("#stablehlo.scatter<")
    assert scatters == hlo.count("unique_indices = true") > 0
    assert "stablehlo.sort" not in hlo
