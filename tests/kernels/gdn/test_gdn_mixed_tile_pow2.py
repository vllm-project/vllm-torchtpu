# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_GDN_MIXED_TILE_POW2 (PR #2821 ece3d83b5): the triangular solve
refuses a chunk its block does not divide instead of dropping rows."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3.compute_gdn import invert_triangular_matrix


def _strict_lower(n):
    rng = np.random.default_rng(0)
    return jnp.asarray(np.tril(rng.standard_normal((2, n, n)) * 0.1, -1), jnp.float32)


def test_divisible_chunk_same_result_on_and_off(monkeypatch):
    t = _strict_lower(32)
    monkeypatch.delenv("VLLM_TPU_GDN_MIXED_TILE_POW2", raising=False)
    off = invert_triangular_matrix(t, block_size=16)
    monkeypatch.setenv("VLLM_TPU_GDN_MIXED_TILE_POW2", "1")
    on = invert_triangular_matrix(t, block_size=16)
    np.testing.assert_array_equal(np.asarray(off), np.asarray(on))


def test_indivisible_chunk_asserts_when_on(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_GDN_MIXED_TILE_POW2", "1")
    with pytest.raises(AssertionError, match="divisible"):
        invert_triangular_matrix(_strict_lower(24), block_size=16)
