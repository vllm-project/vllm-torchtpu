# SPDX-License-Identifier: Apache-2.0
"""A stage-shared SWA plan shifted by a layer's row offset equals the direct path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (
    offset_swa_metadata,
    prepare_swa_metadata,
    prepare_swa_metadata_base,
)
from vllm_torchtpu.kernels.mla import paged_swa


def _requests(rng, tokens, requests, page_size, pages_per_request):
    counts = rng.integers(0, 7, requests)
    counts[rng.integers(0, requests)] = 0
    starts = np.concatenate(([0], np.cumsum(counts)))
    assert starts[-1] <= tokens
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        first = rng.integers(0, page_size * (pages_per_request - 2))
        positions[starts[r] : starts[r + 1]] = first + np.arange(counts[r])
    return jnp.asarray(starts, jnp.int32), jnp.asarray(positions)


@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("non_causal", [False, True])
@pytest.mark.parametrize("onehot", [False, True])
def test_offset_metadata_matches_direct(page_size, non_causal, onehot):
    rng = np.random.default_rng(page_size + 2 * non_causal + 4 * onehot)
    tokens, requests, pages_per_request, layers = 72, 12, 8, 5
    starts, positions = _requests(rng, tokens, requests, page_size, pages_per_request)
    table = jnp.asarray(
        rng.permutation(requests * pages_per_request).reshape(requests, -1) + 1,
        jnp.int32,
    )
    minimum = jnp.asarray(rng.integers(0, 64, requests), jnp.int32)
    page_stride = layers * page_size
    kwargs = dict(
        page_size=page_size,
        page_stride=page_stride,
        non_causal=non_causal,
        onehot_requests=onehot,
    )
    metadata, bounds, records, visible, valid = prepare_swa_metadata_base(
        table, starts, positions, minimum, **kwargs
    )
    for layer in range(layers):
        row_offset = layer * page_size
        direct = prepare_swa_metadata(
            table, starts, positions, minimum, row_offset=row_offset, **kwargs
        )
        shifted = offset_swa_metadata(
            metadata,
            visible,
            records,
            valid,
            page_size=page_size,
            row_offset=row_offset,
        )
        np.testing.assert_array_equal(shifted[0], direct[0])
        np.testing.assert_array_equal(bounds, direct[1])
        np.testing.assert_array_equal(shifted[1], direct[2])


def _apply(pool_words, destinations, masks, words):
    """Host model of ``write_swa_owned_pages`` on the INT32 view of the owner."""
    pool = pool_words.copy()
    for destination, mask, word in zip(destinations, masks, words):
        if destination < 0:
            continue
        old = pool[destination] if mask[0] != -1 else 0
        pool[destination] = (old & ~mask) | word
    return pool


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("page_size", [8, 64])
def test_planned_writes_match_direct(banked, page_size):
    # Records hit page zero, both bank edges, odd/even neighbours and padding;
    # some offsets push rows out of a bank, onto page zero or past the pool.
    rng = np.random.default_rng(page_size + banked)
    pages = 12
    bank_pages = (5, pages - 5) if banked else (pages,)
    capacity = pages * page_size
    count = 90
    records = rng.choice(np.arange(-page_size, capacity), count, replace=False)
    records[rng.random(count) < 0.15] = -1
    records = records.astype(np.int32)
    valid = records >= 0
    values = jnp.asarray(rng.standard_normal((count, 512)), jnp.bfloat16)
    plan = paged_swa.plan_swa_writes(
        jnp.asarray(records), jnp.asarray(valid), block_rows=128
    )
    padding = -count % 128
    initial = rng.integers(-(2**31), 2**31 - 1, (capacity // 2, 512), np.int64)
    initial = initial.astype(np.int32)
    for row_offset in (0, page_size, 3 * page_size, capacity - 2 * page_size):
        shifted = np.where(valid, records + row_offset, -1).astype(np.int32)
        expected, actual, base = initial.copy(), initial.copy(), 0
        for size in bank_pages:
            cap = size * page_size
            rows = slice(base // 2, (base + cap) // 2)
            direct = paged_swa._bank_writes(
                jnp.asarray(shifted), values, base, cap, page_size, padding
            )
            planned = paged_swa._planned_bank_writes(
                plan, values, row_offset, base, cap, page_size
            )
            expected[rows] = _apply(expected[rows], *map(np.asarray, direct))
            actual[rows] = _apply(actual[rows], *map(np.asarray, planned))
            base += cap
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif("jax.default_backend() != 'tpu'", reason="SparseCore publication")
@pytest.mark.parametrize("banked", [False, True])
def test_planned_write_swa_pages_matches_device(banked):
    rng = np.random.default_rng(5 + banked)
    page_size, pages, count = 64, 16, 288
    records = rng.choice(
        np.arange(page_size, pages * page_size // 2), count, replace=False
    )
    records[-20:] = -1
    records = jnp.asarray(records, jnp.int32)
    valid = records >= 0
    values = jnp.asarray(rng.standard_normal((count, 512)), jnp.bfloat16)
    initial = jnp.asarray(rng.standard_normal((pages, page_size, 512)), jnp.bfloat16)
    plan = paged_swa.plan_swa_writes(
        records, valid, block_rows=paged_swa.swa_write_block_rows()
    )
    for row_offset in (0, 2 * page_size, 5 * page_size):

        def owner():
            copy = initial + 0
            return (copy[:7], copy[7:]) if banked else copy

        direct = paged_swa.write_swa_pages(
            owner(), jnp.where(valid, records + row_offset, -1), values
        )
        planned = paged_swa.write_swa_pages_planned(
            owner(), plan, values, row_offset=row_offset
        )
        for a, b in zip(jax.tree.leaves(direct), jax.tree.leaves(planned)):
            np.testing.assert_array_equal(
                jax.lax.bitcast_convert_type(a, jnp.uint16),
                jax.lax.bitcast_convert_type(b, jnp.uint16),
            )
