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
    x = jnp.asarray(
        rng.standard_normal((64, 3, 128)) * rng.uniform(0.01, 30, (64, 3, 1)),
        jnp.float32,
    )
    nib, scale = msa_fp4.quantize(x)
    assert nib.dtype == jnp.uint8 and int(nib.max()) <= 15
    err = jnp.abs(msa_fp4.dequantize(nib, scale) - x)
    assert bool(jnp.all(err <= scale[..., None] / 2 + 1e-6))
    # The vector's max magnitude is representable exactly.
    amax = jnp.max(jnp.abs(x), axis=-1)
    np.testing.assert_allclose(
        jnp.max(jnp.abs(msa_fp4.dequantize(nib, scale)), axis=-1), amax, rtol=1e-6
    )


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


def test_kv_scales_read_each_heads_column_block():
    """Folded K/V page of G heads: scale j of slot s at lane j * S + s, j = 2h
    (K of head h) and 2h + 1 (V), i.e. head h's 2D-wide block holds its K
    scales in its first S lanes and its V scales in the next S."""
    g, page = 3, 128
    lanes = jnp.arange(2, 2 + 2 * page * g, dtype=jnp.float32)[None]  # [1, W]
    cache = msa_fp4.pack_rows(jnp.zeros((1, page, 2 * page * g), jnp.uint8), lanes)
    assert cache.shape == (1, page // 8 + 1, 4, 2 * page * g)
    k_s, v_s = (np.asarray(x) for x in msa_fp4.kv_scales(cache))
    assert k_s.shape == v_s.shape == (1, g, page)
    for h in range(g):
        np.testing.assert_array_equal(k_s[0, h], np.asarray(lanes[0, 2 * h * page :][:page]))
        np.testing.assert_array_equal(
            v_s[0, h], np.asarray(lanes[0, (2 * h + 1) * page :][:page])
        )


def test_write_fp4_stores_rotated_quantized_rows_and_scales():
    import functools

    from vllm_torchtpu.models.minimax_m3.kernels import msa

    rng = np.random.default_rng(4)
    num_pages, page, d = 5, 128, 128
    kv_cache = msa_fp4.pack_rows(
        jnp.zeros((num_pages, page, 2 * d), jnp.uint8),
        scales=jnp.full((num_pages, 2 * d), jnp.nan, jnp.float32),
    )
    index_cache = msa_fp4.pack_rows(
        jnp.zeros((num_pages, page, d), jnp.uint8),
        scales=jnp.full((num_pages, page), jnp.nan, jnp.float32),
    )
    t = 6
    k = jnp.asarray(rng.standard_normal((t, 1, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((t, 1, d)), jnp.float32)
    ik = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    pages = jnp.asarray([3, 3, 3, 1, 1, 2**30], jnp.int32)  # last token dropped
    pos = jnp.asarray([126, 127, 128 + 0, 5, 6, 0], jnp.int32) % 256
    pages = pages.at[2].set(4)
    write = functools.partial(
        msa.write_rows, pages=pages, pos=pos, max_num_seqs=3, interpret=True
    )
    kv_cache, index_cache = msa.write_fp4(kv_cache, index_cache, k, v, ik, write)
    kv = np.asarray(msa_fp4.unpack_rows(kv_cache))
    ix = np.asarray(msa_fp4.unpack_rows(index_cache))
    ix_s = np.asarray(msa_fp4.folded_scales(index_cache))
    ks, vs = (np.asarray(x[:, 0]) for x in msa_fp4.kv_scales(kv_cache))
    k_nib, k_s = msa_fp4.quantize(msa_fp4.rotate_keys(k[:, 0]))
    v_nib, v_s = msa_fp4.quantize(msa_fp4.rotate_keys(v[:, 0]))
    i_nib, i_s = msa_fp4.quantize(msa_fp4.rotate_keys(ik))
    for i in range(t - 1):
        p, s = int(pages[i]), int(pos[i]) % page
        np.testing.assert_array_equal(kv[p, s, :d], k_nib[i])
        np.testing.assert_array_equal(kv[p, s, d:], v_nib[i])
        np.testing.assert_array_equal(ix[p, s], i_nib[i])
        np.testing.assert_allclose([ks[p, s], vs[p, s]], [k_s[i], v_s[i]], rtol=1e-6)
        np.testing.assert_allclose(ix_s[p, s], i_s[i], rtol=1e-6)
    assert np.isnan(ks[0]).all() and np.isnan(vs[0]).all()  # untouched page
    assert np.isnan(ix_s[0]).all()
    # Untouched slots of a written page.
    assert np.isnan(ix_s[3, :126]).all()
    assert np.isnan(ks[3, :126]).all() and np.isnan(vs[3, :126]).all()


def test_cache_specs_match_allocated_shapes():
    """Folded 4-bit pages: the spec's bytes per block equal the allocated
    page (K/V of G heads: 128 G data + 8 G scale bytes per token)."""
    import types

    import torch

    from vllm_torchtpu.models.minimax_m3 import attention as attn

    config = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=128))
    for cls_args, expected, per_token in [
        ((256, torch.uint8, "l.attn"), (7, 17, 4, 256), 128 + 8),
        ((1024, torch.uint8, "l.attn"), (7, 17, 4, 1024), 4 * (128 + 8)),
        ((128, torch.uint8, "l.attn.index_cache"), (7, 17, 4, 128), 64 + 4),
    ]:
        cache = object.__new__(attn.MiniMaxM3Cache)
        cache.row_width, cache.dtype, cache.prefix = cls_args
        cache.scales_in_page = True
        cache.token_shards, cache.block_size = 1, 128
        shape = cache.kv_cache_shape(7)
        assert shape == expected
        spec = cache.get_kv_cache_spec(config)
        assert spec.head_size == per_token
        assert spec.page_size_bytes == int(np.prod(shape[1:]))
    assert not hasattr(attn, "MiniMaxM3ScaleCache")


def test_write_fp4_multi_head_rows_and_scale_rows():
    """G = 4 KV heads per rank (DCP wider than a head's replicas): rows
    [K_0 | V_0 | ... | K_3 | V_3], head h's K / V scales in its own column
    block of the last word row, index scales in the index page."""
    import functools

    from vllm_torchtpu.models.minimax_m3.kernels import msa

    rng = np.random.default_rng(5)
    g, d, t = 4, 128, 3
    kv_cache = msa_fp4.pack_rows(
        jnp.zeros((2, 128, 2 * d * g), jnp.uint8),
        scales=jnp.zeros((2, 2 * d * g), jnp.float32),
    )
    index_cache = msa_fp4.pack_rows(
        jnp.zeros((2, 128, d), jnp.uint8), scales=jnp.zeros((2, 128), jnp.float32)
    )
    k = jnp.asarray(rng.standard_normal((t, g, d)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((t, g, d)), jnp.float32)
    ik = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    pages = jnp.asarray([1, 1, 1], jnp.int32)
    pos = jnp.asarray([10, 11, 12], jnp.int32)
    write = functools.partial(
        msa.write_rows, pages=pages, pos=pos, max_num_seqs=1, interpret=True
    )
    kv_cache, index_cache = msa.write_fp4(kv_cache, index_cache, k, v, ik, write)
    kv = np.asarray(msa_fp4.unpack_rows(kv_cache))
    ks, vs = (np.asarray(x) for x in msa_fp4.kv_scales(kv_cache))
    lanes = np.asarray(msa_fp4.folded_scales(kv_cache))
    for h in range(g):
        k_nib, k_s = msa_fp4.quantize(msa_fp4.rotate_keys(k[:, h]))
        v_nib, v_s = msa_fp4.quantize(msa_fp4.rotate_keys(v[:, h]))
        np.testing.assert_array_equal(kv[1, 10:13, 2 * h * d : (2 * h + 1) * d], k_nib)
        np.testing.assert_array_equal(
            kv[1, 10:13, (2 * h + 1) * d : (2 * h + 2) * d], v_nib
        )
        np.testing.assert_allclose(ks[1, h, 10:13], k_s, rtol=1e-6)
        np.testing.assert_allclose(vs[1, h, 10:13], v_s, rtol=1e-6)
        # Lane j * S + s, j = 2h / 2h + 1: inside head h's 2D-wide block.
        np.testing.assert_allclose(lanes[1, 2 * h * d + 10 : 2 * h * d + 13], k_s, rtol=1e-6)
        np.testing.assert_allclose(
            lanes[1, (2 * h + 1) * d + 10 : (2 * h + 1) * d + 13], v_s, rtol=1e-6
        )
    assert not lanes[1, 13:d].any() and not lanes[0].any()  # other slots untouched
    _, i_s = msa_fp4.quantize(msa_fp4.rotate_keys(ik))
    ix_s = np.asarray(msa_fp4.folded_scales(index_cache))
    np.testing.assert_allclose(ix_s[1, 10:13], i_s, rtol=1e-6)
