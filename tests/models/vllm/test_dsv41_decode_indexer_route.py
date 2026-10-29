# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode routing preserves request ownership and candidate semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as ak


@pytest.mark.cpu_test
@pytest.mark.parametrize("width", [6144, 49408, 98816])
@pytest.mark.parametrize("mode", ["regular", "padded", "draft", "ordered", "publish"])
@pytest.mark.parametrize("banked", [False, True])
def test_decode_route(monkeypatch, width, mode, banked):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    tpu_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: tpu_info)
    count, queries, heads = 4, 6, 32
    rows = count * queries
    q = jnp.arange(rows * heads * 128, dtype=jnp.float32).reshape(rows, heads, 128)
    q = q.astype(jnp.bfloat16)
    weights = jnp.arange(rows * heads, dtype=jnp.float32).reshape(rows, heads)
    records = jnp.arange(count * width, dtype=jnp.int32).reshape(count, width)
    lengths = jnp.full(rows, width - 1, jnp.int32)
    requests = jnp.repeat(jnp.arange(count, dtype=jnp.int32), queries)
    if mode == "padded":
        lengths = lengths.at[-3:].set(0)
        requests = requests.at[-3:].set(0)
    elif mode == "draft":
        requests = jnp.arange(rows, dtype=jnp.int32) % count
    candidates = jnp.zeros((rows, 128 if mode == "publish" else 0), jnp.int32)
    words = jnp.zeros((256, 128), jnp.int32)
    pool = (words[:128], words[128:]) if banked else words
    calls = []

    def new_select(
        grouped_q,
        grouped_weights,
        pool,
        pages,
        limits,
        *,
        page_records,
        candidate_count,
        candidate_block_size,
    ):
        calls.append(True)
        assert page_records == 128
        assert pages.shape == (count, (1 << (width - 1).bit_length()) // 128)
        np.testing.assert_array_equal(pages[:, : width // 128], records[:, ::128])
        np.testing.assert_array_equal(pages[:, width // 128 :], -1)
        np.testing.assert_array_equal(grouped_q.reshape(q.shape), ak.split_halves(q))
        assert grouped_weights.shape == (count, queries, heads)
        np.testing.assert_array_equal(grouped_weights.reshape(weights.shape), weights)
        np.testing.assert_array_equal(limits.reshape(-1), lengths)
        result = jnp.where(lengths[:, None] > 0, jnp.full((rows, 512), 2), -1)
        assert candidate_block_size == 8
        if mode == "publish":
            assert candidate_count == 128
            return result, jnp.full(candidates.shape, 3, jnp.int32)
        assert candidate_count == 0
        return result

    def old_score(q, pool, records, weights, lengths, requests, page_records):
        return jnp.ones((q.shape[0], records.shape[1]), jnp.float32)

    def old_select(scores, lengths, candidates, *args):
        result = jnp.where(lengths[:, None] > 0, scores[:, :512].astype(jnp.int32), -1)
        return result, candidates

    monkeypatch.setattr(ak, "_decode_select", new_select)
    monkeypatch.setattr(ak, "paged_index_scores_jax", old_score)
    monkeypatch.setattr(ak, "select_indices_jax", old_select)
    with jax.disable_jit():
        selected, published = ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            512,
            8,
            mode == "publish",
            False,
            mode != "ordered",
            page_records=128,
        )
    optimized = mode in ("regular", "padded", "publish")
    assert bool(calls) == optimized
    expected = np.where(np.asarray(lengths)[:, None] > 0, 2 if optimized else 1, -1)
    np.testing.assert_array_equal(selected, np.broadcast_to(expected, (rows, 512)))
    np.testing.assert_array_equal(published, 3 if mode == "publish" else candidates)


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU decode pipeline")
@pytest.mark.parametrize("width", [49408, 98816])
@pytest.mark.parametrize("candidate_count", [0, 2048])
@pytest.mark.parametrize("banked", [False, True])
def test_decode_route_tpu_matches_existing(monkeypatch, width, candidate_count, banked):
    rng = np.random.default_rng(721)
    count, queries, page = 4, 6, 128
    keys = jnp.asarray(rng.normal(size=(count * width, 128)), jnp.bfloat16)
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout

    encoded = layout.encode_records(keys)
    words = (encoded[::2] | (encoded[1::2] << layout.KEY_BITS)).astype(jnp.int32)
    bank_rows = words.shape[0] // 2
    pool = (words[:bank_rows], words[bank_rows:]) if banked else words
    page_order = rng.permutation(count * width // page).reshape(count, -1)
    records = jnp.asarray(
        (page_order[:, :, None] * page + np.arange(page)).reshape(count, width),
        jnp.int32,
    )
    records = records.at[0, page : 2 * page].set(-1)
    q = jnp.asarray(rng.normal(size=(count * queries, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.uniform(0.1, 1, size=(count * queries, 32)), jnp.float32)
    lengths = jnp.full(count * queries, width - 200, jnp.int32)
    lengths = lengths.at[-queries:].set(jnp.array([0, 1, 127, 511, 512, 513]))
    requests = jnp.repeat(jnp.arange(count, dtype=jnp.int32), queries)
    candidates = jnp.zeros((count * queries, candidate_count), jnp.int32)

    def run(q, pool, records, weights, lengths, candidates, requests):
        return ak.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            512,
            8,
            candidate_count > 0,
            False,
            True,
            page_records=page,
        )

    args = (q, pool, records, weights, lengths, candidates, requests)
    candidate = jax.tree.map(np.asarray, jax.jit(run)(*args))
    monkeypatch.setattr(ak.decode_index_selection, "supports_decode", lambda *a: False)
    baseline = jax.tree.map(np.asarray, jax.jit(lambda *a: run(*a))(*args))
    np.testing.assert_array_equal(candidate[0], baseline[0])
    np.testing.assert_array_equal(np.sort(candidate[1]), np.sort(baseline[1]))


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU candidate pipeline")
@pytest.mark.parametrize(
    "mode",
    [
        "regular",
        "overflow",
        "reordered",
        "short",
        "ties",
        "invalid",
        "ragged_rows",
        "shifted_pages",
    ],
)
def test_candidate_cache_matches_existing_with_fallback(mode):
    _check_candidate_cache(mode, groups=2, width=8192, capacity=4096)


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU candidate pipeline")
@pytest.mark.parametrize(
    "groups,mode",
    [
        (1, "regular"),
        (1, "overflow"),
        (1, "invalid"),
        (1, "short"),
        (4, "regular"),
        (4, "reordered"),
    ],
)
def test_long_candidate_cache_matches_existing(groups, mode):
    _check_candidate_cache(mode, groups=groups, width=1048576, capacity=49152)


def _check_candidate_cache(mode, *, groups, width, capacity, queries=6):
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout as layout
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.candidate_cache import (
        prepare_candidate_cache_jax,
        select_candidate_cache_jax,
    )

    rng = np.random.default_rng(104)
    rows = groups * queries
    if mode == "ragged_rows":
        rows += 1
    keys = jnp.asarray(rng.normal(size=(groups * width, 128)), jnp.bfloat16)
    encoded = layout.encode_records(keys)
    words = (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    pool = (words[:4096], words[4096:]) if groups > 1 else words
    pages = rng.permutation(groups * width // 128).reshape(groups, -1) * 128
    records = jnp.asarray(
        (pages[..., None] + np.arange(128)).reshape(groups, width), jnp.int32
    )
    if mode == "shifted_pages":
        # Native pages may start on any pool word row, not only block8.
        records += 2
        pool = (jnp.pad(words[:4096], ((1, 0), (0, 0))), words[4096:])
    records = records.at[-1, 128:256].set(-1)
    q = jnp.asarray(rng.normal(size=(rows, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(rows, 32)), jnp.float32)
    candidate_count = 2048 if width > 131072 else 128
    offset = width // 8 - 16384 if width > 131072 else 0
    candidates = jnp.broadcast_to(
        jnp.arange(candidate_count, dtype=jnp.int32) + offset,
        (rows, candidate_count),
    )
    lengths = jnp.full((rows,), width - 3, jnp.int32)
    if rows > 1:
        lengths = lengths.at[-1].set(0)
    requests = jnp.repeat(jnp.arange(groups, dtype=jnp.int32), queries)
    if mode == "ragged_rows":
        requests = jnp.concatenate((requests, jnp.array([1], jnp.int32)))
    if mode == "overflow":
        candidates += jnp.tile(jnp.arange(queries) * candidate_count, groups)[:, None]
    elif mode == "reordered":
        requests = requests[::-1]
    elif mode == "short":
        lengths = jnp.arange(rows, dtype=jnp.int32) * 7
    elif mode == "ties":
        q = jnp.zeros_like(q)
    elif mode == "invalid":
        candidates = candidates.at[:, :4].set(jnp.array([-1, width // 8, 8, 8]))
        candidates = candidates.at[0].set(-1)
        lengths = lengths.at[min(1, rows - 1)].set(519)
    batch_max = lengths.max().reshape(1)
    prepare = jax.jit(prepare_candidate_cache_jax, static_argnums=(5, 6))
    dense, ids, bits, actual, usable = prepare(
        pool, records, candidates, requests, lengths, 8, capacity
    )
    assert bool(usable[0]) == (mode not in ("overflow", "reordered", "ragged_rows"))
    if mode == "overflow":
        assert np.all(np.asarray(actual) * 8 > capacity)
    select = jax.jit(select_candidate_cache_jax, static_argnums=(12, 13, 14))
    result = select(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        dense,
        ids,
        bits,
        usable,
        8,
        128,
        None,
    )
    expected, _ = ak.paged_indexer_jax(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        512,
        8,
        False,
        True,
        True,
        batch_max,
        128,
    )
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="TPU candidate pipeline")
@pytest.mark.parametrize("groups,queries", [(1, 1), (3, 2), (4, 3), (7, 5), (8, 8)])
@pytest.mark.parametrize("mode", ["regular", "invalid"])
def test_candidate_cache_basic_config_covers_non_k5_batches(groups, queries, mode):
    _check_candidate_cache(
        mode, groups=groups, queries=queries, width=8192, capacity=4096
    )
