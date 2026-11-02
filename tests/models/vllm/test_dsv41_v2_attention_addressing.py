"""CPU checks for DeepSeek V4.1 cache addressing and gather paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
    _index_page_records,
    _query_split,
    _request_row_slots,
    _row_slots,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _attend_window,
    _attention_state,
    paged_attention_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    IndexCacheRows,
    PagedCacheRows,
    gather_cache_page_values,
    gather_cache_pieces,
    gather_cache_values,
    merge_halves,
    merge_sub_rows,
    scatter_cache_values,
    split_halves,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import (
    _visible_top_k,
    paged_index_scores_jax,
    paged_indexer_jax,
)


@pytest.mark.parametrize("width", [7, 128, 300, 2560])
def test_row_slots_matches_elementwise_gather(width):
    rows_per_page = 32
    generator = torch.Generator().manual_seed(width)
    table = torch.randint(0, 1000, (4, 90), generator=generator, dtype=torch.int32)
    requests = torch.randint(0, 4, (17,), generator=generator)
    logical = torch.randint(
        -5, 90 * rows_per_page + 5, (17, width), generator=generator
    )
    cache = torch.empty((1, rows_per_page))
    safe = logical.clamp(0, table.shape[1] * rows_per_page - 1)
    expected = (
        table[requests[:, None], safe // rows_per_page].long() * rows_per_page
        + safe % rows_per_page
    )
    torch.testing.assert_close(_row_slots(cache, table, requests, logical), expected)


@pytest.mark.parametrize("width", [7, 300, 2560, 4000])
def test_request_row_slots_matches_token_row_slots(width):
    rows_per_page = 32
    generator = torch.Generator().manual_seed(width)
    table = torch.randint(0, 1000, (4, 90), generator=generator, dtype=torch.int32)
    requests = torch.randint(0, 4, (17,), generator=generator)
    cache = torch.empty((1, rows_per_page))
    logical = torch.arange(width)[None, :].expand(requests.shape[0], -1)
    expected = _row_slots(cache, table, requests, logical)
    got = _request_row_slots(cache, table, width).index_select(0, requests)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("elements,fp32", [(512, False), (128, False), (64, True)])
def test_gather_cache_values_negative_records(elements, fp32):
    pool = jnp.asarray(
        np.random.default_rng(0).integers(0, 2**31, (64, 128)), dtype=jnp.int32
    ) & jnp.int32(0x3FFF3FFF)  # finite BF16/FP32 values
    record_words = elements if fp32 else elements // 2
    count = 64 * 128 // record_words
    records = jnp.asarray([3, -1, count - 1, -1, -7, 0], dtype=jnp.int32)
    got = gather_cache_values(pool, records, elements, fp32)
    valid = records >= 0
    expected = gather_cache_values(pool, jnp.where(valid, records, 0), elements, fp32)
    np.testing.assert_array_equal(np.asarray(got)[valid], np.asarray(expected)[valid])
    assert got.shape == (records.size, elements)


@pytest.mark.parametrize("elements", [512, 128])
def test_gather_cache_values_split_order(elements):
    # Finite, normal BF16 values (permutations are matmuls; subnormals
    # flush); signs are kept.
    pool = jnp.asarray(
        np.random.default_rng(1).integers(-(2**31), 2**31, (64, 128)), jnp.int32
    ) & jnp.int32(-0x40004001) | jnp.int32(0x00800080)  # 0xBFFFBFFF
    records = jnp.asarray([5, 0, 17, 2], dtype=jnp.int32)
    natural = gather_cache_values(pool, records, elements)
    split = gather_cache_values(pool, records, elements, split=True)
    np.testing.assert_array_equal(
        np.asarray(split, np.float32), np.asarray(split_halves(natural), np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(merge_halves(split), np.float32), np.asarray(natural, np.float32)
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [
                np.asarray(h, np.float32)
                for h in gather_cache_pieces(pool, records, elements)
            ],
            axis=1,
        ),
        np.asarray(split, np.float32),
    )
    numerator = jnp.asarray(
        np.random.default_rng(2).standard_normal((3, elements)), jnp.float32
    )
    np.testing.assert_array_equal(
        np.asarray(merge_halves(split_halves(numerator))), np.asarray(numerator)
    )


@pytest.mark.parametrize("elements", [512, 128])
def test_pack_bf16_words_and_split_pieces_roundtrip(elements):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
        _pack_bf16_words,
        merge_pieces,
        split_pieces,
    )

    rng = np.random.default_rng(elements)
    raw = rng.standard_normal((16, elements)).astype(np.float32)
    values = jnp.asarray(raw, dtype=jnp.bfloat16)
    got_words = _pack_bf16_words(values)
    ref_words = jax.lax.bitcast_convert_type(
        values.reshape(values.shape[0], -1, 2), jnp.int32
    )
    np.testing.assert_array_equal(np.asarray(got_words), np.asarray(ref_words))

    pieces = elements // 128
    q = jnp.asarray(rng.standard_normal((8, 64, elements)), dtype=jnp.bfloat16)
    q_pieces = split_pieces(q, pieces)
    assert q_pieces.shape == (pieces, 8, 64, 128)
    num = jnp.asarray(rng.standard_normal((pieces, 8, 64, 128)), dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(merge_pieces(split_pieces(merge_pieces(num), pieces)), np.float32),
        np.asarray(merge_pieces(num), np.float32),
    )


@pytest.mark.parametrize("width", [64, 200])
def test_paged_attention_matches_dense(width):
    rng = np.random.default_rng(width)
    tokens, heads, dim = 4, 2, 512
    pool = jnp.asarray(rng.integers(0, 2**31, (512, 128)), jnp.int32) & jnp.int32(
        0x3EFF3EFF
    )  # finite, moderate BF16 values
    records = rng.integers(0, 256, (tokens, width)).astype(np.int32)
    records[:, :3] = -1
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    sink = jnp.asarray(rng.standard_normal(heads), jnp.float32)
    empty = jnp.zeros((tokens, 0), jnp.int32)
    scale = dim**-0.5
    got = paged_attention_jax(q, pool, pool, jnp.asarray(records), empty, sink, scale)
    kv = gather_cache_values(pool, jnp.asarray(np.maximum(records, 0)).reshape(-1), dim)
    kv = np.asarray(kv, np.float32).reshape(tokens, width, dim)
    s = np.einsum("thd,tkd->thk", np.asarray(q, np.float32), kv) * scale
    s = np.where((records >= 0)[:, None], s, -np.inf)
    m = np.maximum(s.max(-1), np.asarray(sink)[None])
    p = np.exp(s - m[..., None])
    den = p.sum(-1) + np.exp(np.asarray(sink)[None] - m)
    ref = np.einsum("thk,tkd->thd", p, kv) / den[..., None]
    np.testing.assert_allclose(np.asarray(got, np.float32), ref, atol=2e-3)


@pytest.mark.parametrize("tokens,width", [(1, 300), (8, 2560), (64, 640)])
def test_paged_index_scores_matches_dense(tokens, width):
    rng = np.random.default_rng(tokens)
    dim, heads = 128, 4
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import scatter_index_keys

    all_keys = jnp.asarray(rng.standard_normal((512, dim)), jnp.bfloat16)
    pool = scatter_index_keys(
        jnp.zeros((256, 128), jnp.int32), jnp.arange(512, dtype=jnp.int32), all_keys
    )
    records = rng.integers(0, 256 * 2, (tokens, width)).astype(np.int32)
    records[:, width - 5 :] = -1
    lengths = rng.integers(1, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    got = np.asarray(
        paged_index_scores_jax(q, pool, jnp.asarray(records), weights, lengths)
    )
    keys = all_keys[jnp.asarray(np.maximum(records, 0))]
    logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
    dense = np.asarray((jnp.maximum(logits, 0) * weights[:, :, None]).sum(1))
    valid = (records >= 0) & (np.arange(width)[None] < lengths[:, None])
    np.testing.assert_allclose(got[valid], dense[valid], rtol=1e-5, atol=1e-3)


def _paged_records(rng, rows, width, page, pool_records):
    """`_request_row_slots`-style records: random pages, contiguous inside."""
    pages = -(-width // page)
    table = rng.integers(0, pool_records // page, (rows, pages))
    logical = np.arange(width)
    return (table[:, logical // page] * page + logical % page).astype(np.int32)


@pytest.mark.parametrize("page", [8, 16])
def test_gather_cache_page_values_matches_records(page):
    rng = np.random.default_rng(page)
    pool = jnp.asarray(
        rng.integers(-(2**31), 2**31, (256, 128)), jnp.int32
    ) & jnp.int32(0x3FFF3FFF)
    records = _paged_records(rng, 1, 6 * page, page, 512)[0]
    records[-page:] = -1  # an invalid page reads arbitrary rows
    expected = gather_cache_values(pool, jnp.asarray(records), 128, split=True)
    got = gather_cache_page_values(pool, jnp.asarray(records[::page]), page, 128)
    valid = records >= 0
    assert got.shape == expected.shape
    np.testing.assert_array_equal(
        np.asarray(got, np.float32)[valid], np.asarray(expected, np.float32)[valid]
    )
    with pytest.raises(ValueError, match="whole pool word rows"):
        gather_cache_page_values(pool, jnp.asarray(records[::page]), 3, 128)


@pytest.mark.parametrize(
    "tokens,width,page", [(3, 300, 16), (8, 2560, 16), (2, 640, 8)]
)
def test_paged_index_scores_page_gather_bitwise(tokens, width, page):
    rng = np.random.default_rng(width)
    dim, heads = 128, 4
    pool = jnp.asarray(
        rng.integers(-(2**31), 2**31, (256, 128)), jnp.int32
    ) & jnp.int32(0x3FFF3FFF)
    records = _paged_records(rng, tokens, width, page, 512)
    records[0, -(width % page or page) :] = -1  # a trailing invalid page
    lengths = rng.integers(0, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    args = (q, pool, jnp.asarray(records), weights, lengths)
    expected = np.asarray(paged_index_scores_jax(*args))
    got = np.asarray(paged_index_scores_jax(*args, page_records=page))
    valid = (records >= 0) & (np.arange(width)[None] < lengths[:, None])
    np.testing.assert_array_equal(got[valid], expected[valid])


@pytest.mark.parametrize(
    "candidate_write,candidate_filter",
    [(False, False), (True, False), (False, True)],
)
@pytest.mark.parametrize("unordered", [False, True])
@pytest.mark.parametrize("block_size,page", [(16, 16), (8, 16), (16, 8)])
def test_paged_indexer_page_gather_matches_records(
    candidate_write, candidate_filter, unordered, block_size, page
):
    """Decode (one token per request row): page gathers select identically."""
    rng = np.random.default_rng(5)
    width, topk, tokens = 640, 64, 6
    pool = jnp.asarray(
        rng.integers(-(2**31), 2**31, (256, 128)), jnp.int32
    ) & jnp.int32(0x3FFF3FFF)
    records = _paged_records(rng, tokens, width, page, 512)
    records[1, -page:] = -1
    lengths = rng.integers(topk + 1, width + 1, tokens).astype(np.int32)
    lengths[-1] = 0  # a padding row
    q = jnp.asarray(rng.standard_normal((tokens, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, 4)), jnp.float32)
    candidates = jnp.asarray(
        rng.integers(-1, width // block_size, (tokens, 8)), dtype=jnp.int32
    )
    args = (
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        candidates,
        jnp.arange(tokens, dtype=jnp.int32),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered,
    )
    expected = paged_indexer_jax(*args)
    got = paged_indexer_jax(*args, page_records=page)
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("width,page", [(1300, 16), (700, 8)])
def test_paged_index_scores_shared_rows_page_gather_bitwise(width, page):
    """Prefill (tokens share request rows): page gathers score identically."""
    rng = np.random.default_rng(width)
    pool = jnp.asarray(
        rng.integers(-(2**31), 2**31, (256, 128)), jnp.int32
    ) & jnp.int32(0x3FFF3FFF)
    records = _paged_records(rng, 3, width, page, 512)
    records[1, -(width % page or page) :] = -1
    requests = np.sort(rng.integers(0, 2, 40)).astype(np.int32)  # row 2 unused
    lengths = rng.integers(0, width + 1, 40).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((40, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((40, 4)), jnp.float32)
    args = (q, pool, jnp.asarray(records), weights, lengths, jnp.asarray(requests))
    expected = np.asarray(paged_index_scores_jax(*args))
    got = np.asarray(paged_index_scores_jax(*args, page_records=page))
    token_records = records[requests]
    valid = (token_records >= 0) & (np.arange(width)[None] < lengths[:, None])
    np.testing.assert_array_equal(got[valid], expected[valid])


@pytest.mark.parametrize("layout", ["balanced", "shuffled", "uneven", "empty"])
def test_shared_index_scores_grouped_queries_match_dense(layout):
    # Decode verification has 16 * 6 real queries in a padded 128-row batch.
    # Padding belongs to request zero but must not consume grouping capacity.
    rng = np.random.default_rng(219)
    tokens, rows, width, page = 128, 16, 257, 16
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import scatter_index_keys

    all_keys = jnp.asarray(rng.standard_normal((512, 128)), jnp.bfloat16)
    pool = scatter_index_keys(
        jnp.zeros((256, 128), jnp.int32), jnp.arange(512, dtype=jnp.int32), all_keys
    )
    records = _paged_records(rng, rows, width, page, 512)
    records[3, -page:] = -1
    requests = np.pad(np.repeat(np.arange(rows), 6), (0, 32)).astype(np.int32)
    lengths = np.pad(rng.integers(129, width + 1, 96), (0, 32)).astype(np.int32)
    if layout == "shuffled":
        order = rng.permutation(tokens)
        requests, lengths = requests[order], lengths[order]
    elif layout == "uneven":
        requests[:24] = 0  # exceeds static capacity and requires the fallback
    elif layout == "empty":
        lengths[:] = 0
    q = jnp.asarray(rng.standard_normal((tokens, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, 4)), jnp.float32)
    keys = split_halves(all_keys[jnp.asarray(np.maximum(records[requests], 0))])
    logits = jnp.einsum(
        "thd,tcd->thc",
        split_halves(q),
        keys,
        preferred_element_type=jnp.float32,
    )
    expected = (jnp.maximum(logits, 0) * weights[:, :, None]).sum(1)
    got = paged_index_scores_jax(
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        jnp.asarray(requests),
        page_records=page,
    )
    valid = (records[requests] >= 0) & (np.arange(width) < lengths[:, None])
    np.testing.assert_allclose(
        np.asarray(got)[valid], np.asarray(expected)[valid], rtol=1e-5, atol=1e-5
    )
    # Verify the actual selected indices, not only approximate score equality.
    expected_top = jax.lax.top_k(jnp.where(valid, expected, -jnp.inf), 32)[1]
    got_top = jax.lax.top_k(jnp.where(valid, got, -jnp.inf), 32)[1]
    np.testing.assert_array_equal(np.asarray(got_top), np.asarray(expected_top))


@pytest.mark.parametrize("rows,tokens", [(1, 16), (2, 32)])
def test_shared_index_scores_large_query_groups_keep_matmul(monkeypatch, rows, tokens):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        indexer_impl as indexer_ops,
    )

    def unexpected_word_score(*args, **kwargs):
        raise AssertionError("large prefill groups must not unroll the packed scorer")

    monkeypatch.setattr(indexer_ops, "_score_word_rows", unexpected_word_score)
    rng = np.random.default_rng(29)
    width, page = 257, 16
    pool, _, _, lengths, q, weights = _request_inputs(rng, rows, tokens, width)
    records = _paged_records(rng, rows, width, page, 512)
    requests = jnp.asarray(np.repeat(np.arange(rows), tokens // rows), jnp.int32)
    expected = paged_index_scores_jax(q, pool, records, weights, lengths, requests)
    got = paged_index_scores_jax(
        q, pool, records, weights, lengths, requests, page_records=page
    )
    valid = np.arange(width) < lengths[:, None]
    np.testing.assert_array_equal(np.asarray(got)[valid], np.asarray(expected)[valid])


@pytest.mark.parametrize("per_token", [False, True])
def test_index_score_tiles_by_request_rows_are_bitwise(monkeypatch, per_token):
    # Wide decode contexts span several key tiles. Request-row (grouped) and
    # per-token word-row tiles only change loop blocking: every key score must
    # be bitwise identical to the historical 512-key tiling.
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        indexer_impl as indexer_ops,
    )

    rng = np.random.default_rng(1913)
    rows, width, page = 16, 5000, 16
    tokens = 128
    pool = jnp.asarray(
        rng.integers(-(2**31), 2**31, (256, 128)), jnp.int32
    ) & jnp.int32(0x3FFF3FFF)
    records = _paged_records(rng, rows, width, page, 512)
    records[5, -3 * page :] = -1
    requests = np.pad(np.repeat(np.arange(rows), 6), (0, 32)).astype(np.int32)
    lengths = np.pad(rng.integers(1, width + 1, 96), (0, 32)).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, 4, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, 4)), jnp.float32)
    if per_token:
        args = (q, pool, jnp.asarray(records[requests]), weights, jnp.asarray(lengths))
        kwargs = {"page_records": page}
    else:
        args = (q, pool, jnp.asarray(records), weights, jnp.asarray(lengths))
        kwargs = {"requests": jnp.asarray(requests), "page_records": page}
    got = paged_index_scores_jax(*args, **kwargs)
    monkeypatch.setattr(indexer_ops, "_INDEX_KEYS_PER_ITER", 512 * rows)
    monkeypatch.setattr(indexer_ops, "_INDEX_WORD_KEYS_PER_ITER", 512)
    expected = paged_index_scores_jax(*args, **kwargs)
    valid = (records[requests] >= 0) & (np.arange(width) < lengths[:, None])
    np.testing.assert_array_equal(np.asarray(got)[valid], np.asarray(expected)[valid])


def test_candidate_filter_sparsecore_selection_matches_top_k(monkeypatch):
    """Unordered candidate scoring on SparseCore selects the top_k set."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        indexer_impl as indexer_ops,
    )

    def fake_sparsecore(scores, k, lengths):
        columns = jnp.arange(scores.shape[1])[None]
        masked = jnp.where(columns < lengths[:, None], scores, -jnp.inf)
        values, indices = jax.lax.top_k(masked, k)
        # Reversed, like an unordered selection.
        return jnp.where(values > -jnp.inf, indices, -1)[:, ::-1]

    monkeypatch.setattr(indexer_ops, "_sparsecore_top_k", fake_sparsecore)
    rng = np.random.default_rng(9)
    width, topk, block_size, tokens = 4096, 64, 16, 4
    pool, records, _, lengths, q, weights = _request_inputs(rng, tokens, tokens, width)
    # Long rows: total key reads exceed the candidate pool -> candidate_context.
    lengths = rng.integers(3500, width + 1, tokens).astype(np.int32)
    lengths[-1] = 0  # a padding row selects nothing
    candidates = rng.integers(-1, width // block_size, (tokens, 128)).astype(np.int32)
    candidates[0, :8] = candidates[0, 8]  # duplicate blocks
    args = (
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        jnp.asarray(candidates),
        jnp.arange(tokens, dtype=jnp.int32),
        topk,
        block_size,
        False,
        True,
    )
    ordered, _ = paged_indexer_jax(*args, unordered=False)
    got, _ = paged_indexer_jax(*args, unordered=True)
    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(indexer_ops._compact_selected(ordered))
    )
    assert (np.asarray(got)[-1] == -1).all()


def test_index_page_records_requires_index_owner():
    raw = torch.zeros((64, 128), dtype=torch.int32)
    layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 16 * 256)
    assert _index_page_records(IndexCacheRows(raw, layout)) == 16
    assert _index_page_records(PagedCacheRows(raw, layout)) == 0
    assert _index_page_records(raw) == 0


def _request_inputs(rng, requests_count, tokens, width, heads=4, dim=128):
    pool = jnp.asarray(rng.integers(-(2**31), 2**31, (256, 128)), dtype=jnp.int32)
    pool = pool & jnp.int32(0x3FFF3FFF)  # finite BF16 keys
    records = rng.integers(0, 256 * 2, (requests_count, width)).astype(np.int32)
    records[:, width - 5 :] = -1
    requests = np.sort(rng.integers(0, requests_count, tokens)).astype(np.int32)
    lengths = rng.integers(1, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    return pool, records, requests, lengths, q, weights


@pytest.mark.parametrize(
    "requests_count,tokens,width",
    [(1, 1, 300), (8, 8, 2560), (8, 2, 2560), (2, 64, 640)],
)
def test_paged_index_scores_shared_tiles_match_per_token(requests_count, tokens, width):
    rng = np.random.default_rng(tokens)
    pool, records, requests, lengths, q, weights = _request_inputs(
        rng, requests_count, tokens, width
    )
    expected = paged_index_scores_jax(
        q, pool, jnp.asarray(records[requests]), weights, lengths
    )
    got = paged_index_scores_jax(
        q, pool, jnp.asarray(records), weights, lengths, jnp.asarray(requests)
    )
    # Scores at -1 records are undefined (callers mask them). The shared path
    # contracts each request row's key tile in one matmul for all tokens:
    # equal up to FP32 rounding.
    valid = records[requests] >= 0
    np.testing.assert_allclose(
        np.asarray(got)[valid], np.asarray(expected)[valid], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "candidate_write,candidate_filter",
    [(False, False), (True, False), (False, True)],
)
def test_paged_indexer_request_records_match_per_token(
    candidate_write, candidate_filter
):
    rng = np.random.default_rng(7)
    width, topk, block_size = 640, 64, 16
    pool, records, requests, lengths, q, weights = _request_inputs(rng, 3, 24, width)
    lengths = np.maximum(lengths, topk + 1)  # long enough for candidate_context
    candidates = jnp.asarray(
        rng.integers(-1, width // block_size, (24, 8)), dtype=jnp.int32
    )
    args = (topk, block_size, candidate_write, candidate_filter)
    tokens = jnp.arange(24, dtype=jnp.int32)
    expected = paged_indexer_jax(
        q,
        pool,
        jnp.asarray(records[requests]),
        weights,
        jnp.asarray(lengths),
        candidates,
        tokens,
        *args,
    )
    got = paged_indexer_jax(
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        candidates,
        jnp.asarray(requests),
        *args,
    )
    for a, b in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize(
    "candidate_write,candidate_filter",
    [(False, False), (True, False), (False, True)],
)
def test_paged_indexer_query_split_matches_whole_batch(
    candidate_write, candidate_filter
):
    rng = np.random.default_rng(11)
    width, topk, block_size, tokens = 640, 64, 16, 32
    pool, records, requests, lengths, q, weights = _request_inputs(
        rng, 3, tokens, width
    )
    # Short first half, long second half: per-part maxima straddle topk.
    lengths[: tokens // 2] = np.minimum(lengths[: tokens // 2], topk)
    lengths[tokens // 2 :] = np.maximum(lengths[tokens // 2 :], topk + 1)
    candidates = jnp.asarray(
        rng.integers(-1, width // block_size, (tokens, 8)), dtype=jnp.int32
    )
    args = (topk, block_size, candidate_write, candidate_filter)
    lengths = jnp.asarray(lengths)
    requests = jnp.asarray(requests)
    records = jnp.asarray(records)
    whole = paged_indexer_jax(
        q, pool, records, weights, lengths, candidates, requests, *args
    )
    half = tokens // 2
    parts = [
        paged_indexer_jax(
            q[part],
            pool,
            records,
            weights[part],
            lengths[part],
            candidates[part],
            requests[part],
            *args,
            batch_max=lengths.max()[None],
        )
        for part in (slice(0, half), slice(half, tokens))
    ]
    for index, expected in enumerate(whole):
        got = np.concatenate([np.asarray(part[index]) for part in parts])
        np.testing.assert_array_equal(got, np.asarray(expected))


@pytest.mark.parametrize("width,count", [(64, 40), (32, 40), (64, 1), (32, 7)])
def test_merge_sub_rows_matches_lane_scatter(width, count):
    rng = np.random.default_rng(width + count)
    per_row = 128 // width
    cache = jnp.asarray(rng.integers(-(2**31), 2**31, (32, 128)), dtype=jnp.int32)
    # Unique records, dense enough that rows hold several batch records.
    records = rng.permutation(32 * per_row)[:count].astype(np.int32)
    records[::5] = -1
    words = jnp.asarray(rng.integers(-(2**31), 2**31, (count, width)), jnp.int32)
    expected = scatter_cache_values(cache, jnp.asarray(records), words)
    rows, full = merge_sub_rows(cache, jnp.asarray(records), words)
    got = cache.at[jnp.where(rows >= 0, rows, cache.shape[0])].set(full, mode="drop")
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("rows", [3, 16])
@pytest.mark.parametrize("bound", [0, 7, 512, 1024, 1025, 3000, 5000])
def test_visible_top_k_matches_full(bound, rows):
    rng = np.random.default_rng(bound)
    width, k = 5000, 512
    # Coarse values create ties (incl. -0.0 vs +0.0); columns >= bound and a
    # filtered gap are invisible.
    scores = np.round(rng.standard_normal((rows, width)), 1).astype(np.float32)
    zeros = np.where(rng.random(width) < 0.5, -0.0, 0.0).astype(np.float32)
    scores[::2] = np.where(np.abs(scores[::2]) < 0.3, zeros, scores[::2])
    scores[1, 100:900] = -np.inf
    scores[:, bound:] = -np.inf
    scores = jnp.asarray(scores)
    got = _visible_top_k(scores, k, jnp.int32(bound))
    expected = jnp.sort(jax.lax.top_k(scores, k)[1], axis=-1)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("lengths", [(256,), (100, 3, 1, 152), (37, 90, 129)])
def test_attend_window_branch_free_matches_conditional(lengths):
    # Contiguous requests (chunk tails, decode tokens) packed into one batch,
    # padded to 256 tokens; W = 128 window records per token.
    rng = np.random.default_rng(sum(lengths))
    tokens, heads, dim, window = 256, 2, 512, 128
    pool = jnp.asarray(rng.integers(0, 2**31, (4096, 128)), jnp.int32) & jnp.int32(
        0x3EFF3EFF
    )
    records = np.full((tokens, window), -1, np.int32)
    start = 0
    for request, length in enumerate(lengths):
        base = 1000 * request + int(rng.integers(0, 300))
        for t in range(length):
            position = base + t
            logical = position - window + 1 + np.arange(window)
            records[start + t] = np.where(logical >= 0, logical, -1)
        start += length
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    state = _attention_state(q)
    # fallback=tokens: the conditional version patches instead of redoing
    # the whole batch with `_attend`.
    expected = _attend_window(
        q, pool, jnp.asarray(records), state, dim**-0.5, fallback=tokens
    )
    got = _attend_window(
        q, pool, jnp.asarray(records), state, dim**-0.5, max_requests=len(lengths)
    )
    for e, g in zip(expected, got):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


@pytest.mark.parametrize(
    "candidate_write,candidate_filter,short_ctx",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_paged_indexer_unordered_is_precompacted(
    candidate_write, candidate_filter, short_ctx
):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        _pad_compacted_records,
        compact_records,
    )

    rng = np.random.default_rng(19)
    width, topk, block_size = 640, 64, 16
    pool, records, requests, lengths, q, weights = _request_inputs(rng, 3, 24, width)
    if short_ctx:
        lengths = np.minimum(lengths, topk // 2)
    else:
        lengths = np.maximum(lengths, topk + 1)
    # Include underfull/zero-length rows to test -1 tail compaction.
    lengths[0] = 0
    lengths[1] = 5
    candidates = jnp.asarray(
        rng.integers(-1, width // block_size, (24, 3)), dtype=jnp.int32
    )
    ordered_sel, ordered_pub = paged_indexer_jax(
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        candidates,
        jnp.asarray(requests),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered=False,
    )
    unordered_sel, unordered_pub = paged_indexer_jax(
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        candidates,
        jnp.asarray(requests),
        topk,
        block_size,
        candidate_write,
        candidate_filter,
        unordered=True,
    )
    np.testing.assert_array_equal(np.asarray(unordered_pub), np.asarray(ordered_pub))
    np.testing.assert_array_equal(
        np.asarray(_pad_compacted_records(unordered_sel)),
        np.asarray(compact_records(ordered_sel)),
    )


@pytest.mark.parametrize("unordered", [False, True])
@pytest.mark.parametrize("decode_mode", [False, True])
def test_paged_indexer_candidate_filter_shared_tiles_and_decode_branches(
    unordered, decode_mode
):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import (
        select_indices_jax,
    )

    rng = np.random.default_rng(31)
    width, topk, block_size = 640, 64, 16
    pool, records, requests, lengths, q, weights = _request_inputs(rng, 2, 16, width)
    lengths = np.maximum(lengths, topk * 4)
    if decode_mode:
        # Only 1 active token (rest padded with length 0) so
        # row_lengths.sum() > cand_keys exercises candidate_context.
        lengths[1:] = 0
    candidates = jnp.asarray(
        rng.integers(-1, width // block_size, (16, 4)), dtype=jnp.int32
    )
    got_sel, got_pub = paged_indexer_jax(
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        candidates,
        jnp.asarray(requests),
        topk,
        block_size,
        candidate_write=False,
        candidate_filter=True,
        unordered=unordered,
    )
    scores = paged_index_scores_jax(
        q,
        pool,
        jnp.asarray(records),
        weights,
        jnp.asarray(lengths),
        jnp.asarray(requests),
    )
    scores = jnp.where(
        jnp.take(jnp.asarray(records) >= 0, jnp.asarray(requests), axis=0),
        scores,
        -jnp.inf,
    )
    ref_sel, ref_pub = select_indices_jax(
        scores,
        jnp.asarray(lengths),
        candidates,
        topk,
        block_size,
        candidate_write=False,
        candidate_filter=True,
        unordered=unordered,
    )
    np.testing.assert_array_equal(np.asarray(got_pub), np.asarray(ref_pub))
    np.testing.assert_array_equal(np.asarray(got_sel), np.asarray(ref_sel))


@pytest.mark.parametrize("ranks", [2, 4, 8])
def test_query_split_parts_are_even_and_covered(ranks):
    for count in range(1, 2 * 2048 + 1):
        part, padding = _query_split(count, ranks)
        assert part >= 2 and part % 2 == 0
        assert 0 <= ranks * part - count <= padding
        assert ranks * (part - 2) < count  # no whole spare part per rank


def test_query_split_traces_one_graph_for_all_buckets():
    # A 1-row part (ceil split at T=4/8 on 8 ranks) 0/1-specialized and traced
    # a second graph; the base 16-row split traced one.
    from torch._dynamo.testing import CompileCounter

    def split(x):
        part, padding = _query_split(x.shape[0], 8)
        mine = 3 * part + torch.arange(part, dtype=torch.int32)
        rows = torch.cat((x, x.new_zeros((padding, *x.shape[1:])))).index_select(
            0, mine
        )
        return rows.repeat(8, 1)[: x.shape[0]]

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(split, backend=counter)
    for tokens in (2048, 4, 8, 16):
        x = torch.randn(tokens, 4)
        torch._dynamo.mark_dynamic(x, 0)
        compiled(x)
    torch._dynamo.reset()
    assert counter.frame_count == 1


@pytest.mark.parametrize("layout", ["shared", "boundary", "decode"])
@pytest.mark.parametrize("banked", [False, True])
def test_pure_swa_pallas_preserves_cache_addressing(monkeypatch, layout, banked):
    """Check union reuse, request-tail patching and per-token windows end to end."""
    import functools

    from vllm_torchtpu.kernels.mla import swa_attention
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import _pack_bf16_words

    if jax.default_backend() != "tpu":
        monkeypatch.setattr(swa_attention, "supports_swa", lambda q, records: True)
        monkeypatch.setattr(
            swa_attention,
            "attend_swa",
            functools.partial(swa_attention.attend_swa, interpret=True),
        )
    rng = np.random.default_rng(17)
    tokens, heads, dim = 128, 8, 512
    values = jnp.asarray(rng.normal(size=(512, dim)), jnp.bfloat16)
    pool = _pack_bf16_words(values).reshape(-1, 128)
    if banked:
        pool = (pool[:512], pool[512:])
    positions = np.arange(tokens) + 127
    if layout == "boundary":
        positions[37:] = np.arange(tokens - 37) + 300
    elif layout == "decode":
        positions = rng.integers(127, 512, tokens)
    records = positions[:, None] - 127 + np.arange(128)[None]
    records[-1] = -1
    records = jnp.asarray(records, jnp.int32)
    q = jnp.asarray(rng.normal(size=(tokens, heads, dim)), jnp.bfloat16)
    sink = jnp.asarray(rng.normal(size=heads), jnp.float32)
    got = paged_attention_jax(
        q, pool, pool, records, jnp.empty((tokens, 0), jnp.int32), sink, dim**-0.5
    )
    kv = values[jnp.maximum(records, 0)].astype(jnp.float32)
    scores = jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), kv) * dim**-0.5
    scores = jnp.where((records >= 0)[:, None], scores, -jnp.inf)
    maximum = jnp.maximum(scores.max(-1), sink)
    prob = jnp.exp(scores - maximum[..., None])
    expected = jnp.einsum("thk,tkd->thd", prob, kv)
    expected /= (prob.sum(-1) + jnp.exp(sink - maximum))[..., None]
    np.testing.assert_allclose(
        np.asarray(got, np.float32), expected, atol=3e-3, rtol=1e-2
    )
    np.testing.assert_array_equal(got[-1], 0)
