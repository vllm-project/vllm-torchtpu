# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 page writer: runs of consecutive slots, permuted runs, scattered slots."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import cache


@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize(
    "pattern", ["prefill_runs", "shuffled_runs", "random", "decode_rows"]
)
def test_packed_page_writer_matches_scatter(page_size, banks, pattern):
    """The page writer (interpreted) is bitwise a per-word scatter, for runs of
    consecutive slots (prefill chunks), permuted runs and scattered slots."""
    rng = np.random.default_rng(page_size * 7 + banks)
    fields = cache.PAGE_ROWS
    rows = (fields * page_size + 127) // 128
    pages, stride, offset = 24, rows + 1, 3
    layout = cache.PackedFP8Layout(pages, stride, offset, page_size)
    split = offset + 12 * stride if banks == 2 else offset + pages * stride + 5
    sizes = (split, offset + pages * stride + 5 - split)[:banks]
    bank_arrays = tuple(
        jnp.asarray(rng.integers(-(2**31), 2**31, (n, 128)).astype(np.int32))
        for n in sizes
    )
    slots = pages * page_size
    if pattern == "decode_rows":
        records = np.full(16, -1)
        records[:6] = rng.integers(0, slots - 6) + np.arange(6)
    elif pattern == "random":
        records = rng.choice(slots, 300, replace=False)
    else:
        # Two chunks of consecutive slots, starting mid-page, plus padding.
        first = rng.integers(0, slots // 2 - 700)
        records = np.concatenate(
            (first + np.arange(512), first + 1100 + np.arange(277), np.full(235, -1))
        )
        if pattern == "shuffled_runs":
            records[:300] = rng.permutation(records[:300])
    records = jnp.asarray(records, jnp.int32)
    values = jnp.asarray(
        rng.integers(-(2**31), 2**31, (records.size, fields)).astype(np.int32)
    )
    base, columns, valid = cache._record_addresses(records, layout)
    got = cache.write_packed_pages(
        bank_arrays,
        jnp.where(valid, base, -1),
        columns,
        values,
        page_size=page_size,
        interpret=True,
    )
    words = (
        base[:, None] * 128 + jnp.arange(fields)[None, :] * page_size + columns[:, None]
    )
    start = 0
    for bank, out in zip(bank_arrays, got):
        local = words // 128 - start
        active = valid[:, None] & (local >= 0) & (local < bank.shape[0])
        local = jnp.where(active, local, bank.shape[0])
        want = bank.at[local, words % 128].set(values, mode="drop")
        np.testing.assert_array_equal(np.asarray(out), np.asarray(want))
        start += bank.shape[0]
