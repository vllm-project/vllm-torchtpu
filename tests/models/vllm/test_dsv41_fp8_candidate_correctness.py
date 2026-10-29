# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-query semantics from vLLM v0.31.1rc0 attention/dsa/{candidate_blocks,
sparse_mqa_logits}.py, with CPU stand-ins for TPU DMA and MXU scoring.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import (
    candidates as fp8_candidate_cache,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
    read_fp8_records,
    write_fp8_records,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import indexer_impl

pytestmark = pytest.mark.cpu_test


def _cpu_gather(banks, physical_block_ids, offsets, *, block, layout):
    del offsets
    per_page = PAGE_RECORDS // block
    source_per_page = layout.page_size // block
    safe = jnp.maximum(physical_block_ids, 0)
    addresses = (
        (layout.page_offset + safe // source_per_page * layout.page_stride)[..., None]
        * per_page
        + jnp.arange(PAGE_ROWS) * source_per_page
        + (safe % source_per_page)[..., None]
    )
    source = jnp.concatenate(banks).reshape(-1, block)
    words = jnp.where((physical_block_ids >= 0)[..., None, None], source[addresses], 0)
    return (
        words.reshape(physical_block_ids.shape[0], -1, per_page, PAGE_ROWS, block)
        .transpose(0, 1, 3, 2, 4)
        .reshape(-1, PAGE_RECORDS)
    )


def _cpu_logits(q, keys, weights, lengths, pages, **kwargs):
    start = kwargs.get("query_start", 0)
    count = kwargs.get("query_count", q.shape[0])
    rows = start + jnp.arange(count)
    groups = jnp.searchsorted(kwargs["query_start_loc"][1:], rows, side="right")
    layout = kwargs["cache_layout"]
    cols = jnp.arange(kwargs["max_model_len"])
    records = pages[groups[:, None], cols // layout.page_size] * layout.page_size
    records += cols % layout.page_size
    banks = keys if isinstance(keys, tuple) else (keys,)
    k = read_fp8_records(banks, records, layout).astype(jnp.float32)
    products = jnp.einsum("rhd,rkd->rhk", q[rows], k)
    scores = (jnp.maximum(products, 0) * weights[rows, :, None]).sum(axis=1)
    allowed = cols < lengths[rows, None]
    if kwargs.get("block_membership") is not None:
        allowed &= kwargs["block_membership"][
            rows[:, None], cols // kwargs["membership_block_size"]
        ]
    return jnp.where(allowed, scores, -jnp.inf)


def _cpu_gathered_logits(q, keys, weights, windows, **kwargs):
    offsets = kwargs["query_start_loc"]
    groups = offsets.shape[0] - 1
    pages_per_group = keys.shape[0] // groups // PAGE_ROWS
    pages = jnp.arange(groups * pages_per_group).reshape(groups, pages_per_group)
    capacity = pages_per_group * PAGE_RECORDS
    dense = _cpu_logits(
        q,
        keys,
        weights,
        jnp.full((q.shape[0],), capacity),
        pages,
        cache_layout=PackedFP8Layout(groups * pages_per_group),
        max_model_len=capacity,
        **kwargs,
    )
    rows = kwargs.get("query_start", 0) + jnp.arange(
        kwargs.get("query_count", q.shape[0])
    )
    start, split, end1, end2 = windows[rows].T
    columns = start[:, None] + jnp.arange(kwargs["output_width"])
    allowed = (columns < end1[:, None]) | (
        (columns >= split[:, None]) & (columns < end2[:, None])
    )
    return jnp.where(
        allowed,
        jnp.take_along_axis(dense, jnp.minimum(columns, capacity - 1), axis=1),
        -jnp.inf,
    )


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(fp8_candidate_cache, "gather_candidate_keys", _cpu_gather)
    monkeypatch.setattr(indexer_impl, "fp8_paged_mqa_logits", _cpu_logits)
    monkeypatch.setattr(indexer_impl, "fp8_gathered_mqa_logits", _cpu_gathered_logits)
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _inputs(requests, page_size, two_banks, width=64):
    rng = np.random.default_rng(826)
    groups = max(1, max(requests) + 1)
    pages_per_request = (width + page_size - 1) // page_size
    num_pages = groups * pages_per_request
    layout = PackedFP8Layout(num_pages, page_size=page_size)
    pool = jnp.zeros((num_pages * layout.page_rows, 128), jnp.int32)
    banks = tuple(jnp.split(pool, 2)) if two_banks else (pool,)
    keys = jnp.asarray(rng.normal(size=(num_pages * page_size, 128)))
    banks = write_fp8_records(banks, jnp.arange(keys.shape[0]), keys, layout)
    pages = jnp.asarray(rng.permutation(num_pages).reshape(groups, pages_per_request))
    tokens = len(requests)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.float32)
    weights = jnp.asarray(rng.uniform(0.1, 1, (tokens, 32)), jnp.float32)
    lengths = jnp.asarray([width - (i % 3) * 7 for i in range(tokens)], jnp.int32)
    requests = jnp.asarray(requests, jnp.int32)
    return banks, pages, layout, q, weights, lengths, requests


@pytest.mark.parametrize(
    "requests,sizes",
    [
        ([0, 0], [2]),
        ([0, 0, 1, 1, 1, 2, -1], [2, 1, 1, 1, 1, 1]),
        ([1, 1, 0, 0, -1, -1], [2, 2, 1, 1]),
    ],
)
@pytest.mark.parametrize("page_size,two_banks", [(8, True), (128, False)])
@pytest.mark.parametrize("unordered", [False, True])
@pytest.mark.parametrize("equal_scores", [False, True])
def test_paired_gather_preserves_each_query(
    cpu_kernels,
    requests,
    sizes,
    page_size,
    two_banks,
    unordered,
    equal_scores,
    monkeypatch,
):
    banks, pages, layout, q, weights, lengths, requests = _inputs(
        requests, page_size, two_banks
    )
    block, width, topk = 16, 64, 5
    # Disjoint candidates, overlap, duplicates, invalid IDs and empty rows.
    patterns = [[3, 1, 1], [1, 0, -1], [2, 0, 99], [-1, -1, -1]]
    # The A-only block is invisible while the later shared segment is visible.
    lengths = lengths.at[0].set(24)
    if equal_scores:
        q = jnp.zeros_like(q)
    candidates = jnp.asarray([patterns[i % 4] for i in range(q.shape[0])], jnp.int32)
    prepare = jax.jit(
        functools.partial(
            fp8_candidate_cache.prepare_candidates,
            layout=layout,
            block=block,
            width=width,
        )
    )
    gathered_candidates = prepare(banks, pages, candidates, requests)
    query_group_offsets = np.asarray(gathered_candidates.offsets)
    np.testing.assert_array_equal(np.diff(query_group_offsets)[: len(sizes)], sizes)
    assert np.all(np.diff(query_group_offsets)[len(sizes) :] == 0)
    for row, owner in enumerate(np.asarray(requests)):
        expected = set(np.asarray(candidates[row]).tolist()) & set(
            range(width // block)
        )
        if owner < 0:
            expected = set()
        candidate_block_ids = np.asarray(gathered_candidates.candidate_block_ids[row])
        actual = candidate_block_ids[candidate_block_ids >= 0]
        assert set(actual.tolist()) == expected

    if unordered:
        monkeypatch.setitem(
            indexer_impl.PACKED_FP8_PREPARED_QUERY_ROWS,
            (
                q.shape[0],
                gathered_candidates.candidate_block_ids.shape[1] * block,
                topk,
            ),
            1,
        )
        monkeypatch.setattr(
            _cpu_gathered_logits, "__wrapped__", _cpu_gathered_logits, raising=False
        )
    actual = indexer_impl._score_and_select_candidates(
        q, weights, lengths, gathered_candidates, topk, block, unordered, width
    )
    # Independent dense reference: score the original paged K and filter by
    # each original query's candidates, as NVIDIA's dense fallback does.
    expected = []
    for row, owner in enumerate(np.asarray(requests)):
        if owner < 0:
            expected.append([-1] * topk)
            continue
        logical = np.arange(width)
        records = pages[owner, logical // page_size] * page_size + logical % page_size
        keys = np.asarray(read_fp8_records(banks, records, layout), dtype=np.float32)
        scores = (
            np.maximum(np.asarray(q[row]) @ keys.T, 0)
            * np.asarray(weights[row])[:, None]
        ).sum(0)
        allowed = (logical < lengths[row]) & np.isin(logical // block, candidates[row])
        indices = logical[allowed]
        winners = indices[np.argsort(-scores[allowed], kind="stable")[:topk]]
        expected.append(sorted(winners.tolist()) + [-1] * (topk - len(winners)))
        if unordered:
            chosen = np.asarray(actual[row])
            chosen = chosen[chosen >= 0]
            assert len(chosen) == len(winners)
            assert len(np.unique(chosen)) == len(chosen)
            assert np.all(allowed[chosen])
            np.testing.assert_allclose(
                np.sort(scores[chosen]), np.sort(scores[winners]), rtol=1e-5, atol=1e-5
            )
    if unordered:
        assert np.all(np.asarray(actual)[np.asarray(requests) < 0] == -1)
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("streaming", [False, True])
def test_source_publishes_per_query_across_score_chunks(
    cpu_kernels, monkeypatch, streaming
):
    width, topk = (1024, 512) if streaming else (64, 5)
    if streaming:
        monkeypatch.setattr(_cpu_logits, "__wrapped__", _cpu_logits, raising=False)
        monkeypatch.setattr(
            _cpu_gathered_logits, "__wrapped__", _cpu_gathered_logits, raising=False
        )

        def select(scores, *, k, valid_lengths, **kwargs):
            scores = jnp.where(
                jnp.arange(scores.shape[1]) < valid_lengths[:, None], scores, -jnp.inf
            )
            values, indices = jax.lax.top_k(scores, k)
            return jnp.where(values > -jnp.inf, indices, -1)

        monkeypatch.setattr(indexer_impl.streaming_topk, "streaming_topk", select)
    banks, pages, layout, q, weights, lengths, requests = _inputs(
        [0, 0, 0, 0], 8, False, width
    )
    # Force different best blocks for queries sharing the same request.
    keys = read_fp8_records(banks, pages[0] * 8, layout).astype(jnp.float32)
    q = jnp.broadcast_to(keys[jnp.array([0, 2, 4, 6]), None, :], q.shape)
    candidates = jnp.full((4, 2), -1, jnp.int32)
    if streaming:
        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    selected, published = indexer_impl._select_fp8_indexer_jax(
        q,
        banks,
        pages,
        weights,
        lengths,
        candidates,
        requests,
        lengths.max(),
        (layout.pages, layout.page_stride, layout.page_offset, layout.page_size),
        topk,
        8,
        True,
        False,
        streaming,
        width,
        prefill_chunk_rows=2,
    )
    scores = np.asarray(
        _cpu_logits(
            q,
            banks,
            weights,
            lengths,
            pages,
            query_start_loc=jnp.array([0, 4]),
            max_model_len=width,
            cache_layout=layout,
        )
    )
    expected = []
    for row in range(4):
        block_scores = scores[row].reshape(-1, 8).max(1)
        block_scores[(int(lengths[row]) - 1) // 8] = np.inf
        expected.append(np.argsort(-block_scores, kind="stable")[:2])
    assert len({tuple(row) for row in expected}) > 1
    np.testing.assert_array_equal(published, expected)
    assert selected.shape == (4, topk)
