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


def _paged_decode_inputs(seed, requests, queries, heads, width, page, banks):
    """Random paged decode inputs with ragged/zero lengths and invalid pages."""
    rng = np.random.default_rng(seed)
    records = requests * width
    keys = rng.normal(size=(records, 128)).astype(np.float32)
    words = jax.lax.bitcast_convert_type(
        jnp.asarray(keys, jnp.bfloat16).reshape(-1, 64, 2), jnp.int32
    ).reshape(-1, 128)
    split = (records // 2 // 3) // (page // 2) * (page // 2)
    pool = (words[:split], words[split:]) if banks == 2 else words
    order = rng.permutation(records // page).reshape(requests, -1)
    pages = order.astype(np.int32) * page
    pages[rng.random(pages.shape) < 0.05] = -1
    lengths = rng.integers(0, width + 1, size=(requests, queries)).astype(np.int32)
    lengths[0] = width
    lengths[-1, 0] = 0
    if requests > 2:
        lengths[1] = 0
        lengths[2] = rng.integers(0, min(width, 300), size=queries)
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(requests, queries, heads, 1)), jnp.float32)
    return q, weights, pool, jnp.asarray(pages), jnp.asarray(lengths)


def _assert_2d_matches_legacy(
    seed, requests, queries, heads, width, page, key_tile, banks, **options
):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode import (
        flat_request_group,
    )

    q, weights, pool, pages, lengths = _paged_decode_inputs(
        seed, requests, queries, heads, width, page, banks
    )
    group = options.pop("request_group", None) or flat_request_group(requests, queries)
    assert group
    out_window = options.pop("out_window", 0)
    kwargs = dict(page_records=page, key_tile=key_tile, **options)
    legacy = grouped_decode_scores(q, weights, pool, pages, lengths, **kwargs)
    flat = grouped_decode_scores(
        q,
        weights,
        pool,
        pages,
        lengths,
        request_group=group,
        out_window=out_window,
        **kwargs,
    )
    assert flat.shape == legacy.shape == (requests * queries, width)
    assert flat.dtype == legacy.dtype
    # Bitwise, including -inf masking and signed zeros.
    np.testing.assert_array_equal(
        np.asarray(flat).view(np.uint32), np.asarray(legacy).view(np.uint32)
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "requests,queries,group",
    [(4, 6, None), (8, 6, None), (8, 6, 8), (8, 4, None), (8, 3, None), (2, 8, 1)],
)
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("fuse_mask", [False, True])
@pytest.mark.parametrize("score_window,out_window", [(32768, 0), (32768, 128)])
def test_grouped_decode_scores_2d_out_is_bitwise_legacy(
    requests, queries, group, banks, fuse_mask, score_window, out_window
):
    _assert_2d_matches_legacy(
        11,
        requests,
        queries,
        4,
        512,
        16,
        64,
        banks,
        request_group=group,
        out_window=out_window,
        prefetch=True,
        interpret=True,
        fuse_mask=fuse_mask,
        score_window=score_window,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("halfplane", [False, True])
@pytest.mark.parametrize("banks", [1, 2])
def test_grouped_decode_scores_2d_out_halfplane(halfplane, banks):
    # The 2D request-major path keeps the halfplane Index-K.
    _assert_2d_matches_legacy(
        13,
        8,
        6,
        4,
        512,
        16,
        64,
        banks,
        prefetch=True,
        interpret=True,
        fuse_mask=True,
        halfplane=halfplane,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("prefetch", [False, True])
def test_grouped_decode_scores_2d_out_serving_heads_and_tiles(prefetch):
    # Serving head/query/tile geometry with a reduced width (CPU interpret).
    _assert_2d_matches_legacy(
        5,
        16,
        6,
        32,
        8192,
        256,
        4096,
        2,
        prefetch=prefetch,
        interpret=True,
        fuse_mask=True,
        score_window=131072,
    )


@pytest.mark.cpu_test
def test_grouped_decode_scores_2d_out_rejects_unaligned_groups():
    q, weights, pool, pages, lengths = _paged_decode_inputs(0, 6, 6, 4, 256, 16, 1)
    with pytest.raises(ValueError, match="8-row"):
        grouped_decode_scores(
            q,
            weights,
            pool,
            pages,
            lengths,
            page_records=16,
            key_tile=64,
            interpret=True,
            request_group=2,
        )


@pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="Production widths are TPU-sized"
)
@pytest.mark.parametrize("requests,width", [(16, 131072), (12, 65536)])
@pytest.mark.parametrize("out_window", [0, 32768])
def test_grouped_decode_scores_2d_out_production_shapes(requests, width, out_window):
    _assert_2d_matches_legacy(
        3,
        requests,
        6,
        32,
        width,
        256,
        4096,
        2,
        out_window=out_window,
        fuse_mask=True,
        score_window=131072,
    )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="SparseCore top-k")
@pytest.mark.parametrize("requests,width", [(16, 131072), (12, 65536)])
@pytest.mark.parametrize("candidate_count", [0, 128])
def test_decode_indexer_2d_out_selection_is_identical(
    monkeypatch, requests, width, candidate_count
):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import decode

    q, weights, pool, pages, lengths = _paged_decode_inputs(
        9, requests, 6, 32, width, 256, 2
    )
    kwargs = dict(
        page_records=256,
        key_tile=4096,
        request_tile=requests,
        candidate_count=candidate_count,
        candidate_block_size=128,
    )

    def run(enabled):
        # Read at trace time: drop the other setting's trace.
        monkeypatch.setenv("TPU_DEEPSEEK_V41_INDEX_SCORES_2D", enabled)
        decode.decode_indexer.clear_cache()
        return jax.tree.map(
            np.asarray,
            decode.decode_indexer(q, weights, pool, pages, lengths, **kwargs),
        )

    legacy, flat = run("0"), run("1")
    jax.tree.map(np.testing.assert_array_equal, flat, legacy)


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
            result.ids[request], ids + [-1] * (capacity - len(ids))
        )
        np.testing.assert_array_equal(
            result.membership[request],
            [expected[i] for i in ids] + [0] * (capacity - len(ids)),
        )
        assert int(result.actual_count[request]) == len(expected)
        assert bool(result.overflow[request]) == (len(expected) > capacity)


def _assert_scores_match(actual, reference, *, exact):
    """Identical -inf masks, no NaN/+inf; finite values exact or within ~ULPs.

    Infinities only come from the explicit -inf masks, so their positions must
    match exactly. Finite values come from the same FP32 dot + ReLU-weighted
    head sum, but on the TPU backend (even with interpret=True) XLA lowers
    each kernel body separately: in-kernel key unpack/bitcast or the flat
    kernel's per-request loop changes the dot/head-reduction order, giving a
    few-ULP differences (and +0 vs -0 for all-zero ReLU slots). ``exact``
    keeps bitwise equality (up to the sign of zero) for identical bodies.
    """
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert not np.isnan(actual).any() and not np.isnan(reference).any()
    assert not np.isposinf(actual).any() and not np.isposinf(reference).any()
    masked = np.isneginf(reference)
    np.testing.assert_array_equal(np.isneginf(actual), masked)
    if exact:
        np.testing.assert_array_equal(
            np.where(actual == 0, np.float32(0), actual).view(np.uint32),
            np.where(reference == 0, np.float32(0), reference).view(np.uint32),
        )
    else:
        # Head sums cancel, so bound the error by the score scale, not each value.
        finite = reference[~masked]
        scale = float(np.abs(finite).max()) if finite.size else 0.0
        np.testing.assert_allclose(
            actual[~masked], finite, rtol=1e-5, atol=1e-6 * scale
        )


def _candidate_skip_inputs(seed, requests=4, width=256):
    """Unions with an empty capacity tail: ordinary/halfplane words and BF16 keys."""
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        union_metadata,
        unpack_candidate_words,
    )

    rng = np.random.default_rng(seed)
    queries, heads = 6, 4
    native = jnp.asarray(rng.integers(-8, 9, (requests, width, 128)), jnp.bfloat16)
    ordinary = jax.lax.bitcast_convert_type(
        native.reshape(requests, -1, 64, 2), jnp.int32
    ).reshape(requests, -1, 128)
    bits = layout.encode_records(native.reshape(-1, 128))
    halfplane = (bits[::2] | (bits[1::2] << 16)).astype(jnp.int32)
    halfplane = halfplane.reshape(requests, -1, 128)
    candidates = rng.integers(-2, 10, size=(requests, queries, 5), dtype=np.int32)
    candidates[1] = -1  # Empty union: every tile is skipped.
    candidates[-1, :, 0] = 31  # Union reaches the last tile.
    meta = union_metadata(
        jnp.asarray(candidates), logical_blocks=40, capacity_blocks=width // 8
    )
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(requests, queries, heads)), jnp.float32)
    lengths = jnp.asarray(rng.integers(0, 320, (requests, queries)), jnp.int32)
    keys = {
        (False, False): unpack_candidate_words(ordinary, interpret=True),
        (True, False): ordinary,
        (False, True): unpack_candidate_words(
            halfplane, interpret=True, halfplane=True
        ),
        (True, True): halfplane,
    }
    return q, weights, keys, meta.ids, meta.membership, lengths


@pytest.mark.cpu_test
@pytest.mark.parametrize("seed", [81, 82])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("halfplane", [False, True])
def test_candidate_scores_skip_matches_default(seed, packed, halfplane):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        candidate_scores,
    )

    q, weights, keys, ids, membership, lengths = _candidate_skip_inputs(seed)
    reference = candidate_scores(
        q,
        weights,
        keys[False, False],
        ids,
        membership,
        lengths,
        block_size=8,
        key_tile=32,
        interpret=True,
    )
    actual = candidate_scores(
        q,
        weights,
        keys[packed, halfplane],
        ids,
        membership,
        lengths,
        block_size=8,
        key_tile=32,
        interpret=True,
        halfplane=halfplane,
        skip=True,
    )
    # BF16 keys run the same kernel body as the reference; packed keys add an
    # in-kernel unpack/bitcast that changes the TPU reduction order.
    _assert_scores_match(actual, reference, exact=not packed)
    assert np.isneginf(np.asarray(actual)[1]).all()


def _reference_select(scores, lengths, k, *args):
    width = scores.shape[1]
    masked = jnp.where(jnp.arange(width)[None] < lengths[:, None], scores, -jnp.inf)
    values, selected = jax.lax.top_k(masked, k)
    selected = jnp.sort(jnp.where(values > -jnp.inf, selected, width), axis=1)
    return jnp.where(selected < width, selected, -1)


def _run_candidate_indexer(
    monkeypatch, inputs, halfplane, overlap, request_tile=2, flat=0, skip=False
):
    import functools

    from vllm_torchtpu.kernels.deepseek_v41.index_selection import candidate

    with monkeypatch.context() as patch:
        for name in ("candidate_scores", "candidate_scores_flat"):
            if hasattr(candidate, name):
                patch.setattr(
                    candidate,
                    name,
                    functools.partial(getattr(candidate, name), interpret=True),
                )
        patch.setattr(candidate.decode, "_select", _reference_select)
        patch.setenv("TPU_DEEPSEEK_V41_CANDIDATE_FLAT", str(flat))
        patch.setenv("TPU_DEEPSEEK_V41_CANDIDATE_SKIP", str(int(skip)))
        # Switches are read at trace time; drop traces from other settings.
        jax.clear_caches()
        result = candidate.candidate_indexer(
            *inputs,
            block_size=8,
            key_tile=32,
            request_tile=request_tile,
            k=16,
            overlap=overlap,
            halfplane=halfplane,
        )
    jax.clear_caches()
    return np.asarray(result)


@pytest.mark.cpu_test
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("halfplane", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_candidate_indexer_skip_is_identical(monkeypatch, packed, halfplane, overlap):
    q, weights, keys, ids, membership, lengths = _candidate_skip_inputs(83)
    reference = _run_candidate_indexer(
        monkeypatch,
        (q, weights, keys[False, False], ids, membership, lengths),
        False,
        overlap,
    )
    actual = _run_candidate_indexer(
        monkeypatch,
        (q, weights, keys[packed, halfplane], ids, membership, lengths),
        halfplane,
        overlap,
        skip=True,
    )
    np.testing.assert_array_equal(actual, reference)
    assert (reference >= 0).any()


@pytest.mark.cpu_test
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("halfplane", [False, True])
@pytest.mark.parametrize("skip", [False, True])
# Eight-request groups take 2-D [G, W] records; four-request groups [G, 1, W].
@pytest.mark.parametrize("group_tile", [4, 8])
def test_candidate_scores_flat_matches_grouped(packed, halfplane, skip, group_tile):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        candidate_scores,
        candidate_scores_flat,
    )

    q, weights, keys, ids, membership, lengths = _candidate_skip_inputs(84, requests=8)
    reference = candidate_scores(
        q,
        weights,
        keys[False, False],
        ids,
        membership,
        lengths,
        block_size=8,
        key_tile=32,
        interpret=True,
    ).reshape(-1, ids.shape[1] * 8)
    actual = candidate_scores_flat(
        q,
        weights,
        keys[packed, halfplane],
        ids,
        membership,
        lengths,
        block_size=8,
        key_tile=32,
        group_tile=group_tile,
        skip=skip,
        interpret=True,
        halfplane=halfplane,
    )
    _assert_scores_match(actual, reference, exact=False)


@pytest.mark.cpu_test
def test_candidate_scores_flat_rejects_unaligned_request_tile():
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        candidate_scores_flat,
    )

    q, weights, keys, ids, membership, lengths = _candidate_skip_inputs(85)
    with pytest.raises(ValueError, match="8-row-aligned"):
        candidate_scores_flat(
            q,
            weights,
            keys[False, False],
            ids,
            membership,
            lengths,
            block_size=8,
            key_tile=32,
            group_tile=2,
            interpret=True,
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("halfplane", [False, True])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("flat", [4, 8])
def test_candidate_indexer_flat_is_identical(
    monkeypatch, packed, halfplane, skip, flat
):
    q, weights, keys, ids, membership, lengths = _candidate_skip_inputs(86, requests=8)
    reference = _run_candidate_indexer(
        monkeypatch,
        (q, weights, keys[False, False], ids, membership, lengths),
        False,
        False,
        request_tile=8,
    )
    actual = _run_candidate_indexer(
        monkeypatch,
        (q, weights, keys[packed, halfplane], ids, membership, lengths),
        halfplane,
        True,
        request_tile=8,
        flat=flat,
        skip=skip,
    )
    np.testing.assert_array_equal(actual, reference)
    assert (reference >= 0).any()


@pytest.mark.cpu_test
def test_candidate_indexer_flat_falls_back_for_unaligned_tiles(monkeypatch):
    # Two six-query requests per tile cannot fill 8-row flat blocks of 8.
    q, weights, keys, ids, membership, lengths = _candidate_skip_inputs(87)
    inputs = (q, weights, keys[False, False], ids, membership, lengths)
    reference = _run_candidate_indexer(monkeypatch, inputs, False, False)
    actual = _run_candidate_indexer(monkeypatch, inputs, False, True, flat=8)
    np.testing.assert_array_equal(actual, reference)
