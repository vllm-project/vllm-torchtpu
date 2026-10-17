# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request isolation and ragged masking for grouped decode scoring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode import (
    grouped_decode_scores,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("fuse_mask", [False, True])
@pytest.mark.parametrize("score_window", [128, 32768])
def test_grouped_decode_scores_preserve_requests_and_causal_lengths(
    banks, prefetch, fuse_mask, score_window
):
    rng = np.random.default_rng(47)
    requests, queries, heads, width, page = 3, 6, 4, 256, 16
    keys = jnp.asarray(rng.normal(size=(requests * width, 128)), jnp.bfloat16)
    words = jax.lax.bitcast_convert_type(keys.reshape(-1, 64, 2), jnp.int32)
    words = words.reshape(-1, 128)
    pool = (words[:176], words[176:]) if banks == 2 else words
    order = rng.permutation(requests * width // page).reshape(requests, -1)
    pages = order.astype(np.int32) * page
    pages[0, 3] = -1
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(requests, queries, heads, 1)), jnp.float32)
    lengths = jnp.array(
        [[256, 255, 128, 73, 0, 5], [0] * 6, [3, 64, 129, 230, 254, 256]], jnp.int32
    )
    result = grouped_decode_scores(
        q,
        weights,
        pool,
        jnp.asarray(pages),
        lengths,
        page_records=page,
        key_tile=64,
        prefetch=prefetch,
        interpret=True,
        fuse_mask=fuse_mask,
        score_window=score_window,
    ).reshape(requests, queries, width)
    logical = keys.reshape(-1, page, 128)[order].reshape(requests, width, 128)
    logical = jnp.concatenate((logical[..., ::2], logical[..., 1::2]), axis=-1)
    logits = jnp.einsum(
        "rqhd,rkd->rqhk", q, logical, preferred_element_type=jnp.float32
    )
    expected = (jnp.maximum(logits, 0) * weights).sum(2)
    valid = (jnp.arange(width)[None, None] < lengths[:, :, None]) & jnp.repeat(
        jnp.asarray(pages >= 0), page, axis=1
    )[:, None]
    expected = jnp.where(valid, expected, -jnp.inf)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=3e-5)


@pytest.mark.cpu_test
def test_grouped_decode_scores_reject_incompatible_query_weights():
    with pytest.raises(ValueError, match="request-major"):
        grouped_decode_scores(
            jnp.zeros((1, 6, 4, 128), jnp.bfloat16),
            jnp.zeros((1, 5, 4, 1), jnp.float32),
            jnp.zeros((128, 128), jnp.int32),
            jnp.zeros((1, 16), jnp.int32),
            jnp.zeros((1, 6), jnp.int32),
            page_records=16,
            key_tile=64,
            interpret=True,
        )


@pytest.mark.parametrize("fuse_mask", [False, True])
def test_grouped_decode_page_table_beyond_128_entries(fuse_mask):
    if jax.default_backend() != "tpu":
        pytest.skip("TPU SMEM lowering")
    requests, queries, width, page = 3, 6, 4096, 16
    values = jnp.arange(requests * width, dtype=jnp.float32).astype(jnp.bfloat16)
    keys = jnp.broadcast_to(values[:, None], (requests * width, 128))
    words = jax.lax.bitcast_convert_type(keys.reshape(-1, 64, 2), jnp.int32)
    pool = words.reshape(-1, 128)
    pages = (
        jnp.arange(requests * width // page, dtype=jnp.int32).reshape(requests, -1)
        * page
    )
    result = grouped_decode_scores(
        jnp.ones((requests, queries, 1, 128), jnp.bfloat16),
        jnp.ones((requests, queries, 1, 1), jnp.float32),
        pool,
        pages,
        jnp.full((requests, queries), width, jnp.int32),
        page_records=page,
        key_tile=256,
        fuse_mask=fuse_mask,
    ).reshape(requests, queries, width)
    expected = jnp.broadcast_to(
        values.reshape(requests, 1, width).astype(jnp.float32) * 128, result.shape
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.cpu_test
def test_candidate_union_membership_overflow_and_invalid_pages():
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        union_metadata,
        union_record_starts,
    )

    candidates = jnp.array([[[2, 0, 2, -1], [1, 2, 9, -1]]], jnp.int32)
    meta = union_metadata(candidates, logical_blocks=4, capacity_blocks=4)
    np.testing.assert_array_equal(meta.ids, [[0, 1, 2, -1]])
    np.testing.assert_array_equal(meta.membership, [[1, 2, 3, 0]])
    np.testing.assert_array_equal(meta.actual_count, [3])
    np.testing.assert_array_equal(meta.overflow, [False])
    truncated = union_metadata(candidates, logical_blocks=4, capacity_blocks=2)
    np.testing.assert_array_equal(truncated.actual_count, [3])
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
    candidates = rng.choice(domain, size=(3, queries, 17)).astype(np.int32)
    candidates[0, :, :3] = logical_blocks - 1  # Duplicate within each query.
    candidates[1] = -1
    candidates[2] = rng.integers(0, logical_blocks, (queries, 17), dtype=np.int32)
    result = jax.jit(
        lambda x: union_metadata(
            x, logical_blocks=logical_blocks, capacity_blocks=capacity
        )
    )(jnp.asarray(candidates))
    for request, rows in enumerate(candidates):
        expected = {}
        for query, ids in enumerate(rows):
            for block in ids:
                if 0 <= block < logical_blocks:
                    expected[int(block)] = expected.get(int(block), 0) | (1 << query)
        ids = sorted(expected)[:capacity]
        np.testing.assert_array_equal(
            result.ids[request], ids + [-1] * (capacity - len(ids))
        )
        np.testing.assert_array_equal(
            result.membership[request],
            [expected[i] for i in ids] + [0] * (capacity - len(ids)),
        )
        assert int(result.actual_count[request]) == len(expected)
        assert bool(result.overflow[request]) == (len(expected) > capacity)


@pytest.mark.cpu_test
@pytest.mark.parametrize("seed", [71, 72])
@pytest.mark.parametrize("packed", [False, True])
def test_candidate_scores_match_independent_per_head_reference(seed, packed):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        candidate_scores,
        union_metadata,
        unpack_candidate_words,
    )

    rng = np.random.default_rng(seed)
    requests, queries, heads, width = 2, 6, 4, 128
    native = jnp.asarray(rng.normal(size=(requests, width, 128)), jnp.bfloat16)
    words = jax.lax.bitcast_convert_type(native.reshape(requests, -1, 64, 2), jnp.int32)
    keys = unpack_candidate_words(words.reshape(requests, -1, 128), interpret=True)
    expected_keys = jnp.concatenate((native[..., ::2], native[..., 1::2]), axis=-1)
    np.testing.assert_array_equal(keys, expected_keys)
    candidates = rng.integers(-2, 18, size=(requests, queries, 12), dtype=np.int32)
    meta = union_metadata(
        jnp.asarray(candidates), logical_blocks=16, capacity_blocks=16
    )
    # A dense key cache whose slots correspond to the compact logical IDs.
    slots = np.maximum(np.asarray(meta.ids), 0)[..., None] * 8 + np.arange(8)
    gathered = expected_keys[jnp.arange(requests)[:, None], slots.reshape(requests, -1)]
    if packed:
        native_gathered = native[
            jnp.arange(requests)[:, None], slots.reshape(requests, -1)
        ]
        score_keys = jax.lax.bitcast_convert_type(
            native_gathered.reshape(requests, -1, 64, 2), jnp.int32
        ).reshape(requests, -1, 128)
    else:
        score_keys = gathered
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(requests, queries, heads)), jnp.float32)
    lengths = jnp.array([[128, 127, 43, 9, 0, 2], [0, 1, 8, 77, 93, 128]], jnp.int32)
    scores = candidate_scores(
        q,
        weights,
        score_keys,
        meta.ids,
        meta.membership,
        lengths,
        block_size=8,
        key_tile=64,
        interpret=True,
    )
    logits = jnp.einsum(
        "rqhd,rkd->rqhk", q, gathered, preferred_element_type=jnp.float32
    )
    expected = np.asarray((jnp.maximum(logits, 0) * weights[..., None]).sum(2)).copy()
    for r in range(requests):
        for query in range(queries):
            for slot in range(width):
                block = int(meta.ids[r, slot // 8])
                logical = block * 8 + slot % 8
                if (
                    block < 0
                    or block not in candidates[r, query]
                    or logical >= lengths[r, query]
                ):
                    expected[r, query, slot] = -np.inf
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=4e-5)
