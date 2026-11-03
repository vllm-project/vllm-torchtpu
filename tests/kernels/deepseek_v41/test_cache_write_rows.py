# SPDX-License-Identifier: Apache-2.0
"""SparseCore whole-row cache writes preserve every untouched row."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.cache_write import write_cache_rows


@pytest.fixture
def sparsecore_device():
    if jax.default_backend() != "tpu":
        pytest.skip("Row writes require a TPU with SparseCore")
    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("Row writes require SparseCore")


def _random_words(rng, shape):
    return np.asarray(rng.integers(-(2**31), 2**31, shape), np.int32)


def _expected(cache, rows, words):
    expected = np.array(cache)
    for row, word in zip(rows, words):
        if 0 <= row < expected.shape[0]:
            expected[row] = word
    return expected


@pytest.mark.parametrize("count", [1, 37, 512, 4096])
def test_row_writes_match_reference(sparsecore_device, count):
    # Unique destinations with dropped negative and out-of-range rows; the
    # count need not fill a whole SparseCore round.
    rng = np.random.default_rng(count)
    cache = _random_words(rng, (8192, 128))
    rows = rng.permutation(8192)[:count].astype(np.int32)
    # Offset before dropping: 8192 + (-1) would alias row 8191 several times.
    rows[3::11] = 8192 + rows[3::11]
    rows[::7] = -1
    words = _random_words(rng, (count, 128))
    got = write_cache_rows(jnp.asarray(cache), jnp.asarray(rows), jnp.asarray(words))
    np.testing.assert_array_equal(got, _expected(cache, rows, words))


def test_row_writes_accept_identical_duplicates(sparsecore_device):
    # Merged sub-row writes give every record of one row the same full row.
    rng = np.random.default_rng(5)
    cache = _random_words(rng, (64, 128))
    rows = np.array([3, 9, 3, -1, 9, 63, 3], np.int32)
    by_row = {row: _random_words(rng, (128,)) for row in set(rows.tolist())}
    words = np.stack([by_row[row] for row in rows.tolist()])
    got = write_cache_rows(jnp.asarray(cache), jnp.asarray(rows), jnp.asarray(words))
    np.testing.assert_array_equal(got, _expected(cache, rows, words))


def test_row_writes_update_the_owner_in_place(sparsecore_device):
    def update(cache, rows, words):
        for layer in range(3):
            cache = write_cache_rows(cache, rows, words + layer)
        return cache

    cache = jnp.zeros((1 << 16, 128), jnp.int32)
    pool_bytes = cache.nbytes
    rows = jnp.arange(0, 4096, 2, dtype=jnp.int32)
    words = jnp.ones((rows.shape[0], 128), jnp.int32)
    compiled = jax.jit(update, donate_argnums=(0,)).lower(cache, rows, words).compile()
    # Consecutive writes must reuse the donated owner, not copy the pool.
    assert compiled.memory_analysis().temp_size_in_bytes < pool_bytes
    got = np.asarray(compiled(cache, rows, words))
    expected = np.zeros((1 << 16, 128), np.int32)
    expected[np.asarray(rows)] = 3
    np.testing.assert_array_equal(got, expected)


def test_row_writes_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="INT32"):
        write_cache_rows(
            jnp.zeros((8, 64), jnp.int32),
            jnp.zeros(1, jnp.int32),
            jnp.zeros((1, 128), jnp.int32),
        )
    with pytest.raises(ValueError, match="INT32"):
        write_cache_rows(
            jnp.zeros((8, 128), jnp.int32),
            jnp.zeros(2, jnp.int32),
            jnp.zeros((1, 128), jnp.int32),
        )
    cache = jnp.zeros((8, 128), jnp.int32)
    empty = write_cache_rows(
        cache, jnp.zeros(0, jnp.int32), jnp.zeros((0, 128), jnp.int32)
    )
    np.testing.assert_array_equal(empty, 0)
