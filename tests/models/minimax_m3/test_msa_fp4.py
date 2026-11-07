# SPDX-License-Identifier: Apache-2.0
"""4-bit MSA cache format: quantizer, packing and the Hadamard rotation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4


def test_hadamard_is_exactly_orthogonal_up_to_n():
    h = msa_fp4.hadamard(128)
    assert set(np.unique(h)) == {-1.0, 1.0}
    np.testing.assert_array_equal(h @ h.T, 128 * np.eye(128))


def test_rotation_leaves_scores_and_outputs_unchanged():
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((5, 128)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((7, 128)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((7, 128)), jnp.float32)
    s = q @ k.T
    s_rot = msa_fp4.rotate_queries(q) @ msa_fp4.rotate_keys(k).T
    np.testing.assert_allclose(s_rot, s, rtol=1e-5, atol=1e-4)
    p = jax.nn.softmax(s, axis=-1)
    out = msa_fp4.unrotate_output(p @ msa_fp4.rotate_keys(v))
    np.testing.assert_allclose(out, p @ v, rtol=1e-5, atol=1e-5)


def test_quantize_error_is_at_most_half_a_step():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((64, 3, 128)) * rng.uniform(0.01, 30, (64, 3, 1)), jnp.float32)
    nib, scale = msa_fp4.quantize(x)
    assert nib.dtype == jnp.uint8 and int(nib.max()) <= 15
    err = jnp.abs(msa_fp4.dequantize(nib, scale) - x)
    assert bool(jnp.all(err <= scale[..., None] / 2 + 1e-6))
    # The vector's max magnitude is representable exactly.
    amax = jnp.max(jnp.abs(x), axis=-1)
    np.testing.assert_allclose(jnp.max(jnp.abs(msa_fp4.dequantize(nib, scale)), axis=-1), amax, rtol=1e-6)


def test_zero_vector_has_zero_scale_and_values():
    nib, scale = msa_fp4.quantize(jnp.zeros((2, 128)))
    assert not bool(jnp.any(nib)) and not bool(jnp.any(scale))
    assert bool(jnp.all(jnp.isfinite(msa_fp4.dequantize(nib, scale))))


def test_pack_unpack_round_trip_and_bit_positions():
    rng = np.random.default_rng(2)
    nib = jnp.asarray(rng.integers(0, 16, (3, 128, 256)), jnp.uint8)
    cache = msa_fp4.pack_rows(nib)
    assert cache.shape == (3, 16, 4, 256) and cache.dtype == jnp.uint8
    np.testing.assert_array_equal(msa_fp4.unpack_rows(cache), nib)
    # Slot r * 8 + j sits in bits 4j..4j+3 of word row r (byte j // 2).
    words = np.asarray(cache).transpose(0, 1, 3, 2).copy().view(np.uint32)[..., 0]
    r, j, w = 5, 3, 17
    assert (int(words[1, r, w]) >> (4 * j)) & 0xF == int(nib[1, r * 8 + j, w])


def test_index_scale_row_bounds():
    assert msa_fp4.index_scale_row(1) == 2
    assert msa_fp4.index_scale_row(3) == 6
    with pytest.raises(NotImplementedError):
        msa_fp4.index_scale_row(4)
