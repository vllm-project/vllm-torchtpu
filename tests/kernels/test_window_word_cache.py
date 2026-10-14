# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and untouched-byte contracts for window attention and writes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.cache_layout import pack_cache_values
from vllm_torchtpu.kernels.mla.window_attention import attend_windows
from vllm_torchtpu.kernels.mla.word_cache import scatter_word_records


@pytest.mark.parametrize("width", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("count", [7, 65])
def test_scatter_preserves_unwritten_segments(width, count):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore record writes require a TPU")
    cache = jnp.arange(256 * 128, dtype=jnp.int32).reshape(256, 128)
    # Consecutive but permuted records share word rows. Invalid records must
    # neither overwrite row zero nor wrap to the final physical row.
    records = np.random.default_rng(0).permutation(count).astype(np.int32)
    records[-2:] = [-1, cache.size // width]
    words = -jnp.arange(count * width, dtype=jnp.int32).reshape(count, width)
    expected = np.asarray(cache).copy().reshape(-1, width)
    expected[records[:-2]] = np.asarray(words[:-2])
    result = scatter_word_records(cache, jnp.asarray(records), words)
    np.testing.assert_array_equal(result, expected.reshape(cache.shape))


@pytest.mark.parametrize("width", [128, 256, 1024])
@pytest.mark.parametrize("count", [7, 65])
def test_scatter_keeps_old_rows_before_ring_write(width, count):
    if jax.default_backend() != "tpu":
        pytest.skip("Cache DMA requires a TPU")
    cache = jnp.arange(1024 * 128, dtype=jnp.int32).reshape(1024, 128)
    records = np.random.default_rng(0).permutation(count).astype(np.int32)
    records[-2:] = [-1, cache.size // width]
    words = -jnp.arange(count * width, dtype=jnp.int32).reshape(count, width)
    expected = np.asarray(cache).copy().reshape(-1, width)
    expected[records[:-2]] = np.asarray(words[:-2])

    @jax.jit
    def read_and_write(cache, records, words):
        rows = jnp.arange(count)[:, None] * (width // 128) + jnp.arange(width // 128)
        old = cache[rows]
        return scatter_word_records(cache, records, words, after=old), old

    result, old = read_and_write(cache, jnp.asarray(records), words)
    np.testing.assert_array_equal(result, expected.reshape(cache.shape))
    np.testing.assert_array_equal(
        old.reshape(count, width), np.asarray(cache).reshape(-1, width)[:count]
    )


@pytest.mark.parametrize(
    "queries,heads,dim,keys", [(8, 8, 256, 128), (64, 16, 512, 640)]
)
def test_window_attention_matches_dense_masked_attention(queries, heads, dim, keys):
    groups = 2
    q = jax.random.normal(jax.random.key(0), (groups * queries, heads, dim)).astype(
        jnp.bfloat16
    )
    kv = jax.random.normal(jax.random.key(1), (groups, keys, dim)).astype(jnp.bfloat16)
    kv = kv.at[:, queries + 64 :].set(jnp.nan)
    words = pack_cache_values(kv)
    words = words.reshape(groups, keys, dim // 256, 128).transpose(2, 0, 1, 3)
    column = jnp.arange(keys)[None, None]
    row = jnp.arange(queries)[None, :, None]
    mask = jnp.broadcast_to(
        (column >= row) & (column < row + 64), (groups, queries, keys)
    )
    mask = mask.at[0, 0].set(False)
    scale = dim**-0.5
    maximum, denominator, numerator = attend_windows(
        q, words, mask, scale=scale, interpret=jax.default_backend() != "tpu"
    )
    scores = (
        jnp.einsum(
            "gthd,gkd->gthk",
            q.reshape(groups, queries, heads, dim).astype(jnp.float32),
            kv.astype(jnp.float32),
        )
        * scale
    )
    scores = jnp.where(mask[:, :, None], scores, -jnp.inf)
    expected_max = jnp.maximum(scores.max(-1), -1e30)
    probs = jnp.exp(scores - expected_max[..., None])
    expected_num = jnp.einsum(
        "gthk,gkd->gthd",
        probs.astype(jnp.bfloat16),
        jnp.nan_to_num(kv),
        preferred_element_type=jnp.float32,
    )
    np.testing.assert_allclose(
        maximum, expected_max.reshape(maximum.shape), atol=2e-5, rtol=2e-5
    )
    np.testing.assert_allclose(
        denominator, probs.sum(-1).reshape(denominator.shape), atol=2e-4, rtol=2e-5
    )
    np.testing.assert_allclose(
        numerator, expected_num.reshape(numerator.shape), atol=2e-3, rtol=2e-3
    )
    np.testing.assert_array_equal(numerator[0], 0)
    np.testing.assert_array_equal(denominator[0], 0)


def test_scatter_rejects_unaligned_records():
    with pytest.raises(ValueError, match="16-byte"):
        scatter_word_records(
            jnp.zeros((8, 128), jnp.int32),
            jnp.zeros(1, jnp.int32),
            jnp.zeros((1, 2), jnp.int32),
        )


@pytest.mark.parametrize("width", [16, 32, 64, 128, 256])
def test_scatter_does_not_materialize_a_narrow_cache(width):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore memory analysis requires a TPU")
    cache = jax.ShapeDtypeStruct((1 << 20, 128), jnp.int32)
    records = jax.ShapeDtypeStruct((65,), jnp.int32)
    words = jax.ShapeDtypeStruct((65, width), jnp.int32)
    from jax.experimental.layout import Format, Layout

    owner_layout = Format(
        Layout(major_to_minor=(0, 1), tiling=((8, 128),)),
        jax.sharding.SingleDeviceSharding(jax.devices()[0]),
    )
    compiled = (
        jax.jit(
            scatter_word_records,
            donate_argnums=(0,),
            in_shardings=(owner_layout, None, None),
            out_shardings=owner_layout,
        )
        .lower(cache, records, words)
        .compile()
    )
    memory = compiled.memory_analysis()
    cache_bytes = cache.size * cache.dtype.itemsize
    assert memory.alias_size_in_bytes == cache_bytes
    assert memory.temp_size_in_bytes < cache_bytes // 4
