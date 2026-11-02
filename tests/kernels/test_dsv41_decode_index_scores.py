# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request isolation and ragged masking for grouped decode scoring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode_scores import (
    grouped_decode_scores,
)


def _reference_scores(q, weights, pool, pages, lengths):
    requests, queries = q.shape[:2]
    records = (pages[..., None] + jnp.arange(128)).reshape(requests, -1)
    keys = index_key_layout.gather_records(pool, records)
    logits = jnp.einsum("rqhd,rkd->rqhk", q, keys, preferred_element_type=jnp.float32)
    expected = (jnp.maximum(logits, 0) * weights[..., None]).sum(2)
    valid = (jnp.arange(records.shape[1]) < lengths[..., None]) & jnp.repeat(
        pages >= 0, 128, axis=1
    )[:, None]
    return jnp.where(valid, expected, -jnp.inf).reshape(requests * queries, -1)


@pytest.mark.parametrize("requests", [3, 4, 7])
@pytest.mark.parametrize("block_size", [0, 8])
@pytest.mark.parametrize("banked", [False, True])
def test_grouped_decode_scores_preserve_requests_and_causal_lengths(
    requests, block_size, banked
):
    if jax.default_backend() != "tpu":
        pytest.skip("Persistent DMA scoring requires TPU")
    rng = np.random.default_rng(47)
    queries, heads, width = 6, 32, 512
    keys = jnp.asarray(rng.normal(size=(requests * width, 128)), jnp.bfloat16)
    encoded = index_key_layout.encode_records(keys)
    words = (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    pool = (words[:128], words[128:]) if banked else words
    order = rng.permutation(requests * width // 128).reshape(requests, -1)
    pages = jnp.asarray(order * 128, jnp.int32).at[0, 3].set(-1)
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(requests, queries, heads)), jnp.float32)
    lengths = jnp.asarray(
        np.resize(np.array([[512, 511, 128, 73, 0, 5], [0] * 6]), (requests, queries)),
        jnp.int32,
    )
    result = grouped_decode_scores(
        q, weights, pool, pages, lengths, page_records=128, block_size=block_size
    )
    expected = _reference_scores(q, weights, pool, pages, lengths)
    if block_size:
        result, block_scores = result
        np.testing.assert_allclose(
            block_scores,
            expected.reshape(requests * queries, width // 8, 8).max(-1),
            rtol=1e-4,
            atol=1e-4,
        )
    assert result.shape == (requests * queries, width)
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.cpu_test
@pytest.mark.parametrize("weight_shape", [(1, 5, 32), (1, 6, 32, 1)])
def test_grouped_decode_scores_reject_incompatible_query_weights(weight_shape):
    with pytest.raises(ValueError, match="request-major"):
        grouped_decode_scores(
            jnp.zeros((1, 6, 32, 128), jnp.bfloat16),
            jnp.zeros(weight_shape, jnp.float32),
            (jnp.zeros((64, 128), jnp.int32),) * 2,
            jnp.zeros((1, 2), jnp.int32),
            jnp.zeros((1, 6), jnp.int32),
            page_records=128,
        )


def test_grouped_decode_page_table_beyond_128_entries():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU SMEM lowering")
    requests, queries, width, page = 3, 6, 32768, 128
    values = (jnp.arange(requests * width) % 37 - 18).astype(jnp.bfloat16)
    keys = jnp.broadcast_to(values[:, None], (requests * width, 128))
    encoded = index_key_layout.encode_records(keys)
    words = (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    pool = (words[:2048], words[2048:])
    pages = (
        jnp.arange(requests * width // page, dtype=jnp.int32).reshape(requests, -1)
        * page
    )
    result = grouped_decode_scores(
        jnp.ones((requests, queries, 32, 128), jnp.bfloat16),
        jnp.ones((requests, queries, 32), jnp.float32),
        pool,
        pages,
        jnp.full((requests, queries), width, jnp.int32),
        page_records=128,
        block_size=0,
    ).reshape(requests, queries, width)
    expected = jnp.broadcast_to(
        jnp.maximum(values.reshape(requests, 1, width).astype(jnp.float32), 0)
        * (128 * 32),
        result.shape,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.cpu_test
def test_candidate_union_membership_overflow_and_invalid_pages():
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        union_metadata,
        union_record_starts,
    )

    candidate_block_ids = jnp.array([[[2, 0, 2, -1], [1, 2, 9, -1]]], jnp.int32)
    meta = union_metadata(candidate_block_ids, logical_blocks=4, capacity_blocks=4)
    np.testing.assert_array_equal(meta.union_block_ids, [[0, 1, 2, -1]])
    np.testing.assert_array_equal(meta.membership, [[1, 2, 3, 0]])
    np.testing.assert_array_equal(meta.union_block_count, [3])
    np.testing.assert_array_equal(meta.overflow, [False])
    truncated = union_metadata(candidate_block_ids, logical_blocks=4, capacity_blocks=2)
    np.testing.assert_array_equal(truncated.union_block_count, [3])
    np.testing.assert_array_equal(truncated.overflow, [True])
    records = jnp.arange(32, dtype=jnp.int32)[None].at[:, 8:16].set(-1)
    records = records.at[:, 16:24].set(64 + jnp.arange(8))
    starts, bits = union_record_starts(meta, records, block_size=8, pool_records=32)
    np.testing.assert_array_equal(starts, [[0, -1, -1, -1]])
    np.testing.assert_array_equal(bits, [[1, 0, 0, 0]])
    shifted = jnp.arange(32, dtype=jnp.int32)[None] + 2
    starts, bits = union_record_starts(meta, shifted, block_size=8, pool_records=34)
    np.testing.assert_array_equal(starts, [[2, 10, 18, -1]])
    np.testing.assert_array_equal(bits, [[1, 2, 3, 0]])


@pytest.mark.cpu_test
@pytest.mark.parametrize("logical_blocks", [16384, 16385, 65536, 131072])
@pytest.mark.parametrize("capacity", [7, 32])
@pytest.mark.parametrize("queries", [1, 6, 8])
def test_long_candidate_union_preserves_membership_and_overflow(
    logical_blocks, capacity, queries
):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        union_metadata,
    )

    rng = np.random.default_rng(927)
    domain = np.array([-1, logical_blocks, 0, 1, logical_blocks - 1, 17001, 32768])
    candidate_block_ids = rng.choice(domain, size=(3, queries, 17)).astype(np.int32)
    candidate_block_ids[0, :, :3] = logical_blocks - 1  # Duplicate within each query.
    candidate_block_ids[1] = -1
    candidate_block_ids[2] = rng.integers(
        0, logical_blocks, (queries, 17), dtype=np.int32
    )
    result = jax.jit(
        lambda x: union_metadata(
            x, logical_blocks=logical_blocks, capacity_blocks=capacity
        )
    )(jnp.asarray(candidate_block_ids))
    for request, rows in enumerate(candidate_block_ids):
        expected = {}
        for query, ids in enumerate(rows):
            for block in ids:
                if 0 <= block < logical_blocks:
                    expected[int(block)] = expected.get(int(block), 0) | (1 << query)
        ids = sorted(expected)[:capacity]
        np.testing.assert_array_equal(
            result.union_block_ids[request], ids + [-1] * (capacity - len(ids))
        )
        np.testing.assert_array_equal(
            result.membership[request],
            [expected[i] for i in ids] + [0] * (capacity - len(ids)),
        )
        assert int(result.union_block_count[request]) == len(expected)
        assert bool(result.overflow[request]) == (len(expected) > capacity)


@pytest.mark.cpu_test
@pytest.mark.parametrize("seed", [71, 72])
@pytest.mark.parametrize(
    "requests,queries", [(1, 1), (2, 6), (3, 3), (7, 5), (7, 6), (8, 4), (8, 8)]
)
@pytest.mark.parametrize(
    "interpret,start,head_stripe",
    [(True, 0, None), (True, 1, 8), (False, 0, None), (False, 1, 8)],
)
def test_candidate_scores_match_independent_per_head_reference(
    seed, requests, queries, interpret, start, head_stripe
):
    if not interpret and jax.default_backend() != "tpu":
        pytest.skip("Native candidate DMA scoring requires TPU")
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        _candidate_scores,
        union_metadata,
    )

    rng = np.random.default_rng(seed)
    heads, width = max(4, head_stripe or 1), 128
    start = min(start, requests - 1)
    native = jnp.asarray(rng.normal(size=(requests, width, 128)), jnp.bfloat16)
    expected_keys = jnp.concatenate((native[..., ::2], native[..., 1::2]), axis=-1)
    candidate_block_ids = rng.integers(
        -2, 18, size=(requests, queries, 12), dtype=np.int32
    )
    meta = union_metadata(
        jnp.asarray(candidate_block_ids), logical_blocks=16, capacity_blocks=16
    )
    # A dense key cache whose slots correspond to the compact logical IDs.
    slots = np.maximum(np.asarray(meta.union_block_ids), 0)[..., None] * 8 + np.arange(
        8
    )
    gathered = expected_keys[jnp.arange(requests)[:, None], slots.reshape(requests, -1)]
    ordinary_gathered = native[
        jnp.arange(requests)[:, None], slots.reshape(requests, -1)
    ]
    encoded = layout.encode_records(ordinary_gathered.reshape(-1, 128)).reshape(
        requests, width, 128
    )
    score_keys = (encoded[:, ::2] | (encoded[:, 1::2] << 16)).astype(jnp.int32)
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(requests, queries, heads)), jnp.float32)
    lengths = jnp.asarray(rng.integers(0, width + 1, (requests, queries)), jnp.int32)
    lengths = lengths.at[-1, -1].set(0)
    scores = _candidate_scores(
        q,
        weights,
        score_keys,
        meta.union_block_ids,
        meta.membership,
        lengths,
        start_request=jnp.int32(start),
        request_tile=requests - start,
        block_size=8,
        key_tile=64 if interpret else 128,
        compute_tile=32 if interpret else 128,
        head_stripe=head_stripe,
        interpret=interpret,
    )
    logits = jnp.einsum(
        "rqhd,rkd->rqhk", q, gathered, preferred_element_type=jnp.float32
    )
    expected = np.asarray((jnp.maximum(logits, 0) * weights[..., None]).sum(2)).copy()
    for r in range(requests):
        for query in range(queries):
            for slot in range(width):
                block = int(meta.union_block_ids[r, slot // 8])
                logical = block * 8 + slot % 8
                if (
                    block < 0
                    or block not in candidate_block_ids[r, query]
                    or logical >= lengths[r, query]
                ):
                    expected[r, query, slot] = -np.inf
    assert scores.shape == ((requests - start) * queries, width)
    np.testing.assert_allclose(
        scores, expected[start:].reshape(-1, width), rtol=1e-5, atol=4e-5
    )


@pytest.mark.parametrize(
    "queries,logical_blocks,capacity",
    [
        (1, 129, 7),
        (6, 12352, 6144),
        (8, 16384, 6144),
        (6, 12352, 1),
        (6, 12352, 12352),
    ],
)
def test_sparse_candidate_union_exact(queries, logical_blocks, capacity):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        union_metadata,
    )

    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore lowering requires a TPU")
    rng = np.random.default_rng(20261017)
    values = rng.integers(-2, logical_blocks + 2, (5, queries, 2048), dtype=np.int32)
    values[0] = -1
    values[1] = logical_blocks - 1
    values[2, :, ::2] = 0
    result = jax.jit(
        lambda x: union_metadata(
            x, logical_blocks=logical_blocks, capacity_blocks=capacity
        )
    )(jnp.asarray(values))
    for request, rows in enumerate(values):
        expected = {}
        for query, ids in enumerate(rows):
            for block in ids:
                if 0 <= block < logical_blocks:
                    expected[int(block)] = expected.get(int(block), 0) | (1 << query)
        ids = sorted(expected)[:capacity]
        np.testing.assert_array_equal(
            result.union_block_ids[request], ids + [-1] * (capacity - len(ids))
        )
        np.testing.assert_array_equal(
            result.membership[request],
            [expected[i] for i in ids] + [0] * (capacity - len(ids)),
        )
        assert int(result.union_block_count[request]) == len(expected)
        assert bool(result.overflow[request]) == (len(expected) > capacity)


@pytest.mark.parametrize("block_size", [0, 8])
@pytest.mark.parametrize(
    "requests,queries", [(3, 6), (7, 6), (9, 1), (3, 4), (9, 5), (3, 8), (3, 12)]
)
@pytest.mark.parametrize("key_tile,empty", [(4096, False), (6144, False), (6144, True)])
def test_decode_scores_write_flat_rows_with_ragged_lengths(
    block_size, requests, queries, key_tile, empty
):
    if jax.default_backend() != "tpu":
        pytest.skip("Persistent DMA scoring requires TPU")
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import decode_scores

    rng = np.random.default_rng(949)
    heads, width = 32, 4096
    keys = jnp.asarray(rng.integers(-2, 3, (requests * width, 128)), jnp.bfloat16)
    encoded = index_key_layout.encode_records(keys)
    words = (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    pool = (words[:2048], words[2048:])
    pages = (
        jnp.arange(requests * width // 128, dtype=jnp.int32).reshape(requests, -1) * 128
    )
    pages = pages.at[0, 2].set(-1)
    lengths = jnp.asarray(
        np.resize(
            np.array([0, 1, 255, 1023, 4095, 4096], np.int32), (requests, queries)
        )
    )
    lengths = lengths.at[1].set(0)
    if empty:
        lengths = jnp.zeros_like(lengths)
    q = jnp.asarray(rng.integers(-2, 3, (requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.integers(-2, 3, (requests, queries, heads)), jnp.float32)
    actual = decode_scores._grouped_decode_scores(
        q,
        weights,
        pool,
        pages,
        lengths,
        page_records=128,
        key_tile=key_tile,
        # Exercise the production compute fragment and head reduction across
        # ragged batches, output boundaries and several query counts.
        compute_tile=1024,
        head_group=32,
        block_size=block_size,
    )
    expected = _reference_scores(q, weights, pool, pages, lengths)
    reference = (
        (expected, expected.reshape(requests * queries, width // 8, 8).max(-1))
        if block_size
        else expected
    )
    for output, expected in zip(jax.tree.leaves(actual), jax.tree.leaves(reference)):
        assert output.ndim == 2
        np.testing.assert_array_equal(output, expected)
