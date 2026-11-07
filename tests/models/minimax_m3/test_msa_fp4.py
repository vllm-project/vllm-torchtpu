# SPDX-License-Identifier: Apache-2.0
"""4-bit MSA cache format: quantizer, packing and the Hadamard rotation."""

import jax
import jax.numpy as jnp
import numpy as np

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


def test_scale_page_rows():
    assert msa_fp4.index_scale_row(1) == 2
    assert msa_fp4.index_scale_row(4) == 8
    assert [msa_fp4.scale_rows(g) for g in (1, 2, 3, 4, 7)] == [8, 8, 8, 16, 16]


def test_write_fp4_stores_rotated_quantized_rows_and_scales():
    import functools

    from vllm_torchtpu.models.minimax_m3.kernels import msa

    rng = np.random.default_rng(4)
    num_pages, page, d = 5, 128, 128
    kv_cache = msa_fp4.pack_rows(jnp.zeros((num_pages, page, 2 * d), jnp.uint8))
    index_cache = msa_fp4.pack_rows(jnp.zeros((num_pages, page, d), jnp.uint8))
    scale_cache = jnp.full((num_pages, 8, 128), jnp.nan, jnp.float32)
    t = 6
    k = jnp.asarray(rng.standard_normal((t, 1, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((t, 1, d)), jnp.float32)
    ik = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    pages = jnp.asarray([3, 3, 3, 1, 1, 2**30], jnp.int32)  # last token dropped
    pos = jnp.asarray([126, 127, 128 + 0, 5, 6, 0], jnp.int32) % 256
    pages = pages.at[2].set(4)
    write = functools.partial(msa.write_rows, pages=pages, pos=pos, max_num_seqs=3, interpret=True)
    kv_cache, index_cache, scale_cache = msa.write_fp4(
        kv_cache, index_cache, scale_cache, k, v, ik, write, pages, pos
    )
    kv = np.asarray(msa_fp4.unpack_rows(kv_cache))
    ix = np.asarray(msa_fp4.unpack_rows(index_cache))
    sc = np.asarray(scale_cache)
    k_nib, k_s = msa_fp4.quantize(msa_fp4.rotate_keys(k[:, 0]))
    v_nib, v_s = msa_fp4.quantize(msa_fp4.rotate_keys(v[:, 0]))
    i_nib, i_s = msa_fp4.quantize(msa_fp4.rotate_keys(ik))
    for i in range(t - 1):
        p, s = int(pages[i]), int(pos[i]) % page
        np.testing.assert_array_equal(kv[p, s, :d], k_nib[i])
        np.testing.assert_array_equal(kv[p, s, d:], v_nib[i])
        np.testing.assert_array_equal(ix[p, s], i_nib[i])
        np.testing.assert_allclose(sc[p, :3, s], [k_s[i], v_s[i], i_s[i]], rtol=1e-6)
    assert np.isnan(sc[0]).all()  # untouched page


def test_cache_specs_match_allocated_shapes():
    import torch

    from vllm_torchtpu.models.minimax_m3 import attention as attn

    for cls_args, expected in [
        ((256, torch.uint8, "l.attn"), (7, 16, 4, 256)),
        ((128, torch.uint8, "l.attn.index_cache"), (7, 16, 4, 128)),
    ]:
        cache = object.__new__(attn.MiniMaxM3Cache)
        cache.row_width, cache.dtype, cache.prefix = cls_args
        cache.token_shards, cache.block_size = 1, 128
        assert cache.kv_cache_shape(7) == expected
    scale = object.__new__(attn.MiniMaxM3ScaleCache)
    scale.block_size, scale.row_width = 128, msa_fp4.scale_rows(4)
    assert scale.kv_cache_shape(7) == (7, 16, 128)


def test_write_fp4_multi_head_rows_and_scale_rows():
    """G = 4 KV heads per rank (DCP wider than a head's replicas): rows
    [K_0 | V_0 | ... | K_3 | V_3], scale rows 2h / 2h + 1, index row 8."""
    import functools

    from vllm_torchtpu.models.minimax_m3.kernels import msa

    rng = np.random.default_rng(5)
    g, d, t = 4, 128, 3
    kv_cache = msa_fp4.pack_rows(jnp.zeros((2, 128, 2 * d * g), jnp.uint8))
    index_cache = msa_fp4.pack_rows(jnp.zeros((2, 128, d), jnp.uint8))
    scale_cache = jnp.zeros((2, msa_fp4.scale_rows(g), 128), jnp.float32)
    k = jnp.asarray(rng.standard_normal((t, g, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((t, g, d)), jnp.float32)
    ik = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    pages = jnp.asarray([1, 1, 1], jnp.int32)
    pos = jnp.asarray([10, 11, 12], jnp.int32)
    write = functools.partial(msa.write_rows, pages=pages, pos=pos, max_num_seqs=1, interpret=True)
    kv_cache, _, scale_cache = msa.write_fp4(
        kv_cache, index_cache, scale_cache, k, v, ik, write, pages, pos
    )
    kv = np.asarray(msa_fp4.unpack_rows(kv_cache))
    sc = np.asarray(scale_cache)
    for h in range(g):
        k_nib, k_s = msa_fp4.quantize(msa_fp4.rotate_keys(k[:, h]))
        v_nib, v_s = msa_fp4.quantize(msa_fp4.rotate_keys(v[:, h]))
        np.testing.assert_array_equal(kv[1, 10:13, 2 * h * d : (2 * h + 1) * d], k_nib)
        np.testing.assert_array_equal(kv[1, 10:13, (2 * h + 1) * d : (2 * h + 2) * d], v_nib)
        np.testing.assert_allclose(sc[1, 2 * h, 10:13], k_s, rtol=1e-6)
        np.testing.assert_allclose(sc[1, 2 * h + 1, 10:13], v_s, rtol=1e-6)
    _, i_s = msa_fp4.quantize(msa_fp4.rotate_keys(ik))
    np.testing.assert_allclose(sc[1, msa_fp4.index_scale_row(g), 10:13], i_s, rtol=1e-6)
