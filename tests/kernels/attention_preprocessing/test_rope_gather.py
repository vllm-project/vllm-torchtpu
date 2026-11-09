# SPDX-License-Identifier: Apache-2.0
"""Paired-row, SparseCore and blocked RoPE row gathers equal `gather_rows`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.attention_preprocessing import rope_gather
from vllm_torchtpu.runner import utils


@pytest.mark.parametrize("rows", [5, 240, 256, 257, 288, 520, 1024, 2048])
def test_blocked_rope_gather_matches_one_gather(rows):
    table = jax.random.normal(jax.random.PRNGKey(0), (4096, 64), jnp.float32)
    positions = jax.random.randint(jax.random.PRNGKey(1), (rows,), -8, 4104)
    ref = rope_gather.gather_rows(table, positions)
    got = rope_gather.gather_rows_blocked(table, positions)
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(got))
    inside = jnp.clip(positions, 0, table.shape[0] - 1)
    ref = table[inside]
    got = rope_gather.by_row_blocks(lambda r: table[r], inside, True)
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(got))


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows", [4096, 4097, 99328])
@pytest.mark.parametrize("seed", range(3))
def test_paired_rows_match_gather_rows(rows, seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-3, 3, size=(rows, 32))
    table = np.concatenate((np.cos(angles), np.sin(angles)), 1).astype(np.float32)
    table[0, 0] = -0.0  # sign of zero must survive the select
    # Out-of-range positions on both sides exercise the clamp.
    positions = rng.integers(-5, rows + 5, 288).astype(np.int32)
    positions[:3] = (0, rows - 1, rows)
    pairs = utils._pair_rows(torch.from_numpy(table)).numpy()
    assert pairs.shape == ((rows + 1) // 2, 128)
    ref = rope_gather.gather_rows(jnp.asarray(table), jnp.asarray(positions))
    got = rope_gather.gather_paired_rows(jnp.asarray(pairs), jnp.asarray(positions))
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32)
    )


@pytest.mark.parametrize("rows", [4096, 4097, 99328])
@pytest.mark.parametrize("tokens", [1, 288, 300])
def test_sc_paired_rows_match_gather_rows(rows, tokens):
    """The SparseCore row fetch returns the same bits as the XLA gather."""
    import jax

    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")
    rng = np.random.default_rng(rows + tokens)
    angles = rng.uniform(-3, 3, size=(rows, 32))
    table = np.concatenate((np.cos(angles), np.sin(angles)), 1).astype(np.float32)
    table[0, 0] = -0.0
    positions = rng.integers(-5, rows + 5, tokens).astype(np.int32)
    positions[: min(3, tokens)] = (0, rows - 1, rows)[: min(3, tokens)]
    pairs = utils._pair_rows(torch.from_numpy(table)).numpy()
    ref = rope_gather.gather_rows(jnp.asarray(table), jnp.asarray(positions))
    got = jax.jit(rope_gather.gather_paired_rows_sc)(
        jnp.asarray(pairs), jnp.asarray(positions)
    )
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32)
    )


@pytest.mark.parametrize("seed", range(3))
def test_paired_rows_sc_match_gather_rows(seed):
    import jax

    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")
    rows = 98816
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-3, 3, size=(rows, 32))
    table = np.concatenate((np.cos(angles), np.sin(angles)), 1).astype(np.float32)
    table[0, 0] = -0.0
    positions = rng.integers(-5, rows + 5, 288).astype(np.int32)
    positions[:3] = (0, rows - 1, rows)
    pairs = utils._pair_rows(torch.from_numpy(table)).numpy()
    ref = rope_gather.gather_rows(jnp.asarray(table), jnp.asarray(positions))
    got = jax.jit(rope_gather.gather_paired_rows_sc)(
        jnp.asarray(pairs), jnp.asarray(positions)
    )
    assert got.dtype == ref.dtype and got.shape == ref.shape
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint32), np.asarray(ref).view(np.uint32)
    )
