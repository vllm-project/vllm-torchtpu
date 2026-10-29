"""Tests for decode index scores straight from gathered index-cache word rows."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_scores
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import indexer_impl as indexer_ops
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    ROW_WORDS,
    _split_words,
    gather_cache_page_values,
    gather_cache_page_words,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

PAGE = 16  # records per page: 8 word rows of two 128-element BF16 keys


def _words(rng, shape):
    groups = shape[:-2]
    keys = jnp.asarray(rng.standard_normal((*groups, shape[-2] * 2, 128)), jnp.bfloat16)
    encoded = index_key_layout.encode_records(keys.reshape(-1, 128)).reshape(
        *groups, shape[-2] * 2, 128
    )
    return (encoded[..., ::2, :] | (encoded[..., 1::2, :] << 16)).astype(jnp.int32)


def _reference(q, weights, words):
    """Independent low/high-half decoding followed by dense FP32 scoring."""
    groups, rows, _ = words.shape
    tokens = q.shape[0]
    bits = np.asarray(words).view(np.uint32)
    record_bits = np.stack((bits & 65535, bits >> 16), axis=-2).astype(np.uint16)
    keys = jax.lax.bitcast_convert_type(jnp.asarray(record_bits), jnp.bfloat16)
    keys = jnp.repeat(keys.reshape(groups, 2 * rows, -1), tokens // groups, axis=0)
    logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
    scores = (jnp.maximum(logits, 0) * weights).sum(1)
    return np.asarray(scores)  # record order: key 2r + s is slot s of row r


@pytest.mark.parametrize(
    "tokens,groups,rows",
    [
        (2, 2, 16),
        (6, 1, 16),
        (4, 2, 16),
        (12, 2, 16),
        (1, 1, 2 * packed_scores.MAX_BLOCK_ROWS),
    ],
)
def test_score_word_rows_matches_xla(tokens, groups, rows):
    rng = np.random.default_rng(rows + tokens * 10 + groups)
    words = _words(rng, (groups, rows, ROW_WORDS))
    q = jnp.asarray(rng.standard_normal((tokens, 3, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, 3, 1)), jnp.float32)
    out = packed_scores.score_word_rows(q, weights, words, interpret=True)
    assert out.shape == (tokens, index_key_layout.SLOTS * rows)
    # Interpret mode sums on CPU; TPU is bitwise identical.
    np.testing.assert_allclose(
        np.asarray(out), _reference(q, weights, words), rtol=1e-6, atol=1e-5
    )


@pytest.mark.parametrize(
    "q_shape,words_shape",
    [
        ((1, 2, 128), (1, 16, 64)),  # not 128-word rows
        ((1, 2, 64), (1, 16, 128)),  # not 128-element keys
        ((1, 2, 128), (1, 12, 128)),  # rows not a power of two
        ((1, 2, 128), (1, 4, 128)),  # fewer than 8 rows
        ((3, 2, 128), (2, 16, 128)),  # tokens % groups != 0
        ((2, 2, 128), (4, 16, 128)),  # tokens < groups
    ],
)
def test_score_word_rows_rejects_shapes(q_shape, words_shape):
    q = jnp.zeros(q_shape, jnp.bfloat16)
    with pytest.raises(ValueError):
        packed_scores.score_word_rows(
            q, jnp.zeros((*q_shape[:2], 1)), jnp.zeros(words_shape, jnp.int32)
        )


def test_gather_cache_page_words_is_raw_page_rows():
    rng = np.random.default_rng(0)
    bits = rng.integers(-(2**31), 2**31, size=(64, ROW_WORDS), dtype=np.int32)
    pool = jnp.asarray((bits & np.int32(-0x40004001)) | np.int32(0x00800080))
    first = jnp.asarray([32, -1, 0], jnp.int32)
    words = gather_cache_page_words(pool, first, PAGE, 128)
    assert words.shape == (3 * 8, ROW_WORDS)
    np.testing.assert_array_equal(np.asarray(words[:8]), np.asarray(pool[16:24]))
    np.testing.assert_array_equal(np.asarray(words[16:]), np.asarray(pool[:8]))
    values = gather_cache_page_values(pool, first, PAGE, 128)
    split = jnp.concatenate(_split_words(words.reshape(-1, 64)), axis=1)
    np.testing.assert_array_equal(np.asarray(values), np.asarray(split))
    with pytest.raises(ValueError):
        gather_cache_page_words(pool, first, 3, 128)


def _paged_inputs(rng, tokens, width, lengths, q_dtype=jnp.bfloat16):
    rows = 4096
    pool = _words(rng, (rows, ROW_WORDS))
    pages = -(-width // PAGE)
    table = np.stack([rng.permutation(rows * 2 // PAGE)[:pages] for _ in range(tokens)])
    records = (table[:, :, None] * PAGE + np.arange(PAGE)).reshape(tokens, -1)
    records = records[:, :width].astype(np.int32)
    records[0, -5:] = -1
    q = jnp.asarray(rng.standard_normal((tokens, 4, 128)), q_dtype)
    weights = jnp.asarray(rng.standard_normal((tokens, 4)), jnp.float32)
    return q, pool, records, weights, jnp.asarray(lengths, jnp.int32)


@pytest.mark.parametrize(
    "tokens,width,lengths", [(2, 3000, (3000, 1700)), (3, 2048, (0, 2048, 900))]
)
def test_paged_index_scores_matches_dense(tokens, width, lengths):
    rng = np.random.default_rng(width)
    q, pool, records, weights, n = _paged_inputs(rng, tokens, width, lengths)
    actual = indexer_ops.paged_index_scores_jax(
        q, pool, jnp.asarray(records), weights, n, None, PAGE
    )
    expected = _paged_reference(q, weights, pool, records, lengths)
    _assert_paged_scores(actual, expected, records, lengths)


def _paged_reference(q, weights, pool, records, lengths):
    bits = np.asarray(pool).view(np.uint32)
    ids = np.maximum(records, 0)
    selected = bits[ids // 2]
    halves = (selected >> ((ids % 2) * 16)[..., None]).astype(np.uint16)
    keys = jax.lax.bitcast_convert_type(jnp.asarray(halves), jnp.bfloat16)
    q = jnp.concatenate((q[..., ::2], q[..., 1::2]), axis=-1)
    logits = jnp.einsum("thd,tkd->thk", q, keys, preferred_element_type=jnp.float32)
    scores = (jnp.maximum(logits, 0) * weights[..., None]).sum(1)
    valid = (records >= 0) & (
        np.arange(records.shape[1])[None] < np.asarray(lengths)[:, None]
    )
    return jnp.where(valid, scores, -jnp.inf)


def _assert_paged_scores(actual, expected, records, lengths):
    # The bridge scores whole tiles; its caller masks invalid/causally hidden IDs.
    valid = (records >= 0) & (
        np.arange(records.shape[1])[None] < np.asarray(lengths)[:, None]
    )
    masked = jnp.where(valid, actual, -jnp.inf)
    np.testing.assert_allclose(masked, expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    "tokens,requests",
    [
        (6, [1, 1, 1, 1, 1, 1]),
        (12, [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]),
        (6, [0, 0, 0, 1, 1, 1]),
        (4, [0, 0, 1, 1]),
        (4, [0, 1, 2, 3]),
        (3, [0, 0, 1]),
    ],
)
def test_paged_index_scores_grouped_decode_requests(tokens, requests):
    rng = np.random.default_rng(tokens * 100)
    width = 1024
    _, pool, records, _, _ = _paged_inputs(
        rng, max(tokens, 4), width, [width] * max(tokens, 4)
    )
    q = jnp.asarray(rng.standard_normal((tokens, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, 4)), jnp.float32)
    lengths = jnp.full((tokens,), width, jnp.int32)
    req_arr = jnp.asarray(requests, jnp.int32)
    grouped = indexer_ops.paged_index_scores_jax(
        q, pool, jnp.asarray(records), weights, lengths, req_arr, PAGE
    )
    expanded_records = np.asarray(records)[np.asarray(requests)]
    expected = _paged_reference(q, weights, pool, expanded_records, lengths)
    _assert_paged_scores(grouped, expected, expanded_records, lengths)


def test_paged_index_scores_fp32_query_matches_dense():
    rng = np.random.default_rng(1)
    q, pool, records, weights, n = _paged_inputs(
        rng, 2, 1024, (1024, 512), q_dtype=jnp.float32
    )
    actual = indexer_ops.paged_index_scores_jax(
        q, pool, jnp.asarray(records), weights, n, None, PAGE
    )
    expected = _paged_reference(q, weights, pool, records, n)
    _assert_paged_scores(actual, expected, records, n)


def test_score_word_rows_wrapper_mesh_and_standalone():
    rng = np.random.default_rng(2)
    words = _words(rng, (1, 8, ROW_WORDS))
    q = jnp.asarray(rng.standard_normal((1, 2, 128)), jnp.bfloat16)
    weights = jnp.ones((1, 2, 1), jnp.float32)
    reference = _reference(q, weights, words)
    with mock.patch(
        "vllm_torchtpu.models.vllm.vllm_model_wrapper_context."
        "get_vllm_model_wrapper_context",
        side_effect=RuntimeError("no context"),
    ):
        standalone = indexer_ops._score_word_rows(q, weights, words)
    np.testing.assert_allclose(np.asarray(standalone), reference, rtol=1e-6, atol=1e-5)
    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    with (
        set_vllm_model_wrapper_context(mesh=mesh),
        mock.patch(
            "jax.shard_map", side_effect=lambda fn, **kw: lambda *args: fn(*args)
        ) as shard_map,
    ):
        sharded = indexer_ops._score_word_rows(q, weights, words)
    shard_map.assert_called_once()
    np.testing.assert_allclose(np.asarray(sharded), reference, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("rows", [2, 8, 64, 128])
def test_lane_dense_page_rows_match_iota(rows):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import _lane_dense_page_rows

    # Starts span the full 2-D pool row range, including 12-bit split edges.
    rng = np.random.default_rng(rows)
    starts = rng.integers(0, 33423360, size=8 * 128 // rows, dtype=np.int32)
    starts[:3] = [0, 4095, 4096]
    starts = jnp.asarray(starts)
    expected = (starts[:, None] + jnp.arange(rows, dtype=jnp.int32)[None]).reshape(-1)
    np.testing.assert_array_equal(_lane_dense_page_rows(starts, rows), expected)


def test_gather_cache_page_words_lane_dense_rows():
    rng = np.random.default_rng(5)
    cache = _words(rng, (1024, ROW_WORDS))
    # 128 pages of 8 rows: 1024 gathered rows, eligible for the lane-dense path.
    first = jnp.asarray(rng.permutation(128).astype(np.int32) * PAGE)
    first = first.at[3].set(-1)
    got = gather_cache_page_words(cache, first, PAGE, 128)
    rows = PAGE * 64 // ROW_WORDS
    starts = np.asarray(first) // 2
    for page, start in enumerate(starts):
        if start >= 0:
            np.testing.assert_array_equal(
                np.asarray(got).reshape(128, rows, -1)[page],
                np.asarray(cache)[start : start + rows],
            )
