# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-query membership, including duplicates and invalid union slots."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
    read_fp8_records,
    write_fp8_records,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    canonical_candidate_blocks,
    gather_candidate_keys,
    prepare_candidate_metadata,
    prepare_candidates,
)


@pytest.mark.parametrize("case", ["even", "odd", "padding", "invalid", "ragged"])
@pytest.mark.parametrize("width,per_query", [(1024, 48), (98816, 2048)])
def test_decode_metadata_preserves_request_segments(case, width, per_query):
    """Fused grouping must preserve unions across odd and invalid requests."""
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises real SparseCore metadata")
    tokens, requests, block = 288, 48, 8
    rng = np.random.default_rng(826)
    owners = np.repeat(np.arange(requests, dtype=np.int32), 6)
    if case == "odd":
        owners[-3:] = -1
    elif case == "padding":
        owners[-24:] = -1
    elif case == "invalid":
        owners[:] = -1
    elif case == "ragged":
        owners = np.repeat(np.arange(requests, dtype=np.int32), [3, 9] * 24)
    layout = PackedFP8Layout(requests * width // 128, page_size=128)
    pages = rng.permutation(layout.pages).reshape(requests, -1).astype(np.int32)
    pages.reshape(-1)[::13] = -1
    candidates = rng.integers(
        -2, width // block + 2, (tokens, per_query), dtype=np.int32
    )
    metadata = jax.jit(
        functools.partial(
            prepare_candidate_metadata, layout=layout, block=block, width=width
        )
    )(jnp.asarray(pages), jnp.asarray(candidates), jnp.asarray(owners))
    physical_block_ids, candidate_block_ids, offsets, valid, spans = map(
        np.asarray, metadata
    )

    expected_offsets = [0]
    begin = 0
    while begin < tokens:
        end = begin + 1
        while end < tokens and owners[end] == owners[begin]:
            end += 1
        paired = 0 <= owners[begin] < requests and (end - begin) % 2 == 0
        expected_offsets.extend(
            range(begin + (2 if paired else 1), end + 1, 2 if paired else 1)
        )
        begin = end
    expected_offsets.extend([tokens] * (tokens + 1 - len(expected_offsets)))
    np.testing.assert_array_equal(offsets, expected_offsets)
    np.testing.assert_array_equal(valid, (owners >= 0) & (owners < requests))
    for group, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if begin == end:
            continue
        union = np.unique(candidates[begin:end])
        union = union[(union >= 0) & (union < width // block)]
        if valid[begin]:
            union = union[pages[owners[begin], union // 16] >= 0]
        else:
            union = union[:0]
        first = np.intersect1d(union, candidates[begin])
        second = (
            np.intersect1d(union, candidates[begin + 1])
            if end - begin == 2
            else np.array([], np.int32)
        )
        exclusive = np.setdiff1d(first, second)
        shared = np.intersect1d(first, second)
        following = np.setdiff1d(second, first)
        union = np.concatenate((exclusive, shared, following))
        for query in range(begin, end):
            selected = (
                np.concatenate((exclusive, shared))
                if query == begin
                else np.concatenate((shared, following))
            )
            expected = np.full(candidate_block_ids.shape[1], -1, np.int32)
            expected[: len(selected)] = selected
            np.testing.assert_array_equal(candidate_block_ids[query], expected)
            np.testing.assert_array_equal(
                spans[query],
                [0, len(exclusive)]
                if query == begin
                else [len(exclusive), len(shared)],
            )
        expected_physical = np.full(physical_block_ids.shape[1], -1, np.int32)
        if valid[begin]:
            expected_physical[: len(union)] = (
                pages[owners[begin], union // 16] * 16 + union % 16
            )
        np.testing.assert_array_equal(physical_block_ids[group], expected_physical)


@pytest.mark.parametrize("page_size", [8, 128])
def test_canonical_blocks_tag_per_query_ownership(page_size):
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises real SparseCore metadata")
    width, block = 1024, 16
    blocks = width // block
    rng = np.random.default_rng(891)
    # Two groups, each pairing two queries that share one sorted union.
    first = rng.integers(-3, blocks + 3, (2, 48), dtype=np.int32)
    second = rng.integers(-3, blocks + 3, (2, 48), dtype=np.int32)
    first[1] = -1
    candidates = jnp.asarray(np.stack((first, second), axis=1))
    table = jnp.asarray(rng.integers(0, 7, (2, width // page_size), dtype=np.int32))
    candidate_block_ids, _, counts = jax.jit(
        functools.partial(
            canonical_candidate_blocks,
            block=block,
            width=width,
            cache_pages=1 << 30,
            page_size=page_size,
        )
    )(candidates, table, jnp.array([0, 1, 2], jnp.int32))
    candidate_block_ids = np.asarray(candidate_block_ids)
    for group in range(2):
        a, shared, b = np.asarray(counts[group])
        first_ids = candidate_block_ids[group, : a + shared]
        second_ids = candidate_block_ids[group, a : a + shared + b]
        for actual, source in ((first_ids, first[group]), (second_ids, second[group])):
            expected = np.unique(source[(source >= 0) & (source < blocks)])
            np.testing.assert_array_equal(np.sort(actual), expected)


@pytest.mark.parametrize("owners", [[0, 0], [0, 0, 1, 1, 1, 2, -1]])
@pytest.mark.parametrize("page_size", [8, 128])
def test_prepared_keys_and_membership_keep_original_token_mapping(owners, page_size):
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises real SparseCore metadata and DMA")
    width, block = 64, 16
    requests = max(owners) + 1
    pages_per_request = (width + page_size - 1) // page_size
    layout = PackedFP8Layout(requests * pages_per_request, page_size=page_size)
    rng = np.random.default_rng(192)
    pages = jnp.asarray(rng.permutation(layout.pages).reshape(requests, -1))
    pool = jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32)
    banks = tuple(jnp.split(pool, 2)) if page_size == 8 else (pool,)
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    banks = write_fp8_records(banks, jnp.arange(keys.shape[0]), keys, layout)
    patterns = np.array([[0, 2, 2], [1, 3, -1], [2, 0, 99], [-1, -1, -1]], np.int32)
    candidates = jnp.asarray(patterns[np.arange(len(owners)) % len(patterns)])
    prepared = jax.jit(
        functools.partial(prepare_candidates, layout=layout, block=block, width=width)
    )(banks, pages, candidates, jnp.asarray(owners))
    offsets = np.asarray(prepared.offsets)
    compact_layout = PackedFP8Layout(prepared.keys.shape[0] // 33)
    for row, owner in enumerate(owners):
        candidate_block_ids = np.asarray(prepared.candidate_block_ids[row])
        expected_ids = set(np.asarray(candidates[row])) & set(range(width // block))
        if owner < 0:
            expected_ids = set()
        group = np.searchsorted(offsets[1:], row, side="right")
        start = int(prepared.block_spans[row, 0])
        assert set(candidate_block_ids[candidate_block_ids >= 0]) == expected_ids
        live = np.flatnonzero(candidate_block_ids >= 0)
        if not len(live):
            continue
        compact = ((start + live[:, None]) * block + np.arange(block)).reshape(-1)
        group_capacity = prepared.keys.shape[0] // (len(offsets) - 1) // 33 * 128
        compact_records = group * group_capacity + compact
        logical = (candidate_block_ids[live, None] * block + np.arange(block)).reshape(
            -1
        )
        records = pages[owner, logical // page_size] * page_size + logical % page_size
        np.testing.assert_array_equal(
            np.asarray(
                read_fp8_records((prepared.keys,), compact_records, compact_layout)
            ),
            np.asarray(read_fp8_records(banks, records, layout)),
        )


@pytest.mark.parametrize("page_size", [8, 128])
@pytest.mark.parametrize("bank_count", [1, 2])
def test_gather_skips_empty_tiles_without_changing_live_slots(page_size, bank_count):
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises SparseCore DMA")
    layout = PackedFP8Layout(128, page_size=page_size)
    rng = np.random.default_rng(729)
    pool = jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32)
    banks = tuple(jnp.split(pool, bank_count))
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    banks = write_fp8_records(banks, jnp.arange(keys.shape[0]), keys, layout)
    physical_block_ids = rng.integers(0, keys.shape[0] // 8, (3, 4096), dtype=np.int32)
    physical_block_ids[0, 129:] = -1
    physical_block_ids[1] = -1
    physical_block_ids[2, 2049:] = -1
    offsets = jnp.array([0, 1, 1, 2], jnp.int32)
    run = jax.jit(functools.partial(gather_candidate_keys, block=8, layout=layout))
    # Changing occupancy also exercises buffer reuse and final partial tiles.
    for ids in (physical_block_ids, physical_block_ids[:, ::-1].copy()):
        gathered = run(banks, jnp.asarray(ids), offsets)
        slots = np.flatnonzero(ids.reshape(-1) >= 0)
        target = (slots[:, None] * 8 + np.arange(8)).reshape(-1)
        source = (ids.reshape(-1)[slots, None] * 8 + np.arange(8)).reshape(-1)
        np.testing.assert_array_equal(
            np.asarray(
                read_fp8_records(
                    (gathered,),
                    jnp.asarray(target),
                    PackedFP8Layout(gathered.shape[0] // 33),
                )
            ),
            np.asarray(read_fp8_records(banks, jnp.asarray(source), layout)),
        )
    run(banks, jnp.full_like(physical_block_ids, -1), offsets).block_until_ready()


@pytest.mark.parametrize(
    "bank_count,groups,columns,page_size,block",
    [
        (banks, groups, columns, page, 8)
        for banks, groups, columns in [(1, 1, 16), (2, 3, 96), (2, 24, 32768)]
        for page in (8, 128)
    ]
    + [(banks, 3, 96, 128, block) for banks in (1, 2) for block in (16, 32, 64, 128)],
)
def test_gather_preserves_words_with_strides_and_metadata_refill(
    bank_count, groups, columns, page_size, block
):
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises SparseCore DMA and metadata-buffer refill")
    stride = 6 if page_size == 8 else 40
    layout = PackedFP8Layout(
        128, page_stride=stride, page_offset=3, page_size=page_size
    )
    rng = np.random.default_rng(751)
    pool = rng.integers(0, 2**31, (layout.pages * layout.page_stride, 128), np.int32)
    banks = tuple(jnp.asarray(b) for b in np.split(pool, bank_count))
    physical_block_ids = np.full((groups, columns), -1, np.int32)
    fragments = layout.pages * page_size // block
    physical_block_ids[:, :17] = rng.integers(
        0, fragments, physical_block_ids[:, :17].shape
    )
    physical_block_ids[:, -33:] = rng.integers(
        0, fragments, physical_block_ids[:, -33:].shape
    )
    offsets = jnp.arange(groups + 1, dtype=jnp.int32)
    run = jax.jit(functools.partial(gather_candidate_keys, block=block, layout=layout))
    gathered = run(banks, jnp.asarray(physical_block_ids), offsets)
    slots = np.flatnonzero(physical_block_ids.reshape(-1) >= 0)
    fields = np.arange(33)[None, :, None]
    tokens = np.arange(block)[None, None, :]
    destination = (slots[:, None, None] // (128 // block) * 33 + fields) * 128
    destination = destination + slots[:, None, None] % (128 // block) * block + tokens
    selected = physical_block_ids.reshape(-1)[slots]
    source = layout.page_offset + selected // (page_size // block) * layout.page_stride
    source = source[:, None, None] * 128 + fields * page_size + tokens
    source = source + (selected % (page_size // block))[:, None, None] * block
    np.testing.assert_array_equal(
        np.asarray(gathered.reshape(-1)[jnp.asarray(destination)]),
        pool.reshape(-1)[source],
    )
    run(
        banks, jnp.asarray(physical_block_ids), jnp.zeros_like(offsets)
    ).block_until_ready()


@pytest.mark.cpu_test
@pytest.mark.parametrize("page_size", [8, 128])
def test_candidate_addresses_deduplicate_and_drop_invalid_pages(monkeypatch, page_size):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with jax.default_device(jax.devices("cpu")[0]):
        candidates = jnp.array([[[2, 0, 2, 3, -1, 999]]], jnp.int32)
        table = [[2, 3, -1, 4, 5, 6, -1, -1]] if page_size == 8 else [[2]]
        candidate_block_ids, physical_block_ids, _ = canonical_candidate_blocks(
            candidates,
            jnp.array(table, jnp.int32),
            jnp.array([0, 1], jnp.int32),
            block=16,
            width=64,
            cache_pages=7,
            page_size=page_size,
        )
        ids = [0, 2] if page_size == 8 else [0, 2, 3]
        addresses = [2, 3, 5, 6] if page_size == 8 else [16, 18, 19]
        np.testing.assert_array_equal(
            candidate_block_ids[0], ids + [-1] * (6 - len(ids))
        )
        np.testing.assert_array_equal(
            physical_block_ids[0],
            addresses + [-1] * (physical_block_ids.shape[1] - len(addresses)),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("page_size", [8, 128])
def test_candidate_long_context_partial_block(page_size):
    width, block = 2**20 + 5, 16
    last = width // block
    table = np.zeros((1, (width + page_size - 1) // page_size), np.int32)
    table[0, 0] = -1
    ids = jnp.array([[[last, 0, last, last + 1, -1]]], jnp.int32)
    candidate_block_ids, physical_block_ids, _ = canonical_candidate_blocks(
        ids,
        jnp.asarray(table),
        jnp.array([0, 1]),
        block=block,
        width=width,
        cache_pages=1,
        page_size=page_size,
    )
    np.testing.assert_array_equal(candidate_block_ids, [[last, -1, -1, -1, -1]])
    expected = np.full(physical_block_ids.shape, -1, np.int32)
    expected[0, 0] = 0
    np.testing.assert_array_equal(physical_block_ids, expected)
