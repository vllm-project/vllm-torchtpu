"""CPU checks for DeepSeek V4.1 cache addressing and gather paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41_csa2.attention import (
    _request_row_slots,
    _row_slots,
)
from vllm_torchtpu.models.vllm.deepseek_v41_csa2.attention_kernels import (
    _attend_window,
    _attention_state,
    _visible_top_k,
    paged_attention_jax,
    paged_index_scores_jax,
    paged_indexer_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41_csa2.cache import (
    gather_cache_pieces,
    gather_cache_values,
    merge_halves,
    merge_sub_rows,
    scatter_cache_values,
    split_halves,
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
    pool = jnp.asarray(rng.integers(-(2**31), 2**31, (256, 128)), dtype=jnp.int32)
    # Keep BF16 keys finite: clear exponent-overflow patterns.
    pool = pool & jnp.int32(0x3FFF3FFF)
    records = rng.integers(0, 256 * 2, (tokens, width)).astype(np.int32)
    records[:, width - 5 :] = -1
    lengths = rng.integers(1, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    got = np.asarray(
        paged_index_scores_jax(q, pool, jnp.asarray(records), weights, lengths)
    )
    keys = gather_cache_values(
        pool, jnp.asarray(np.maximum(records, 0)).reshape(-1), dim
    ).reshape(tokens, width, dim)
    logits = jnp.einsum("thd,tcd->thc", q, keys, preferred_element_type=jnp.float32)
    dense = np.asarray((jnp.maximum(logits, 0) * weights[:, :, None]).sum(1))
    valid = (records >= 0) & (np.arange(width)[None] < lengths[:, None])
    np.testing.assert_allclose(got[valid], dense[valid], rtol=1e-5, atol=1e-3)


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
