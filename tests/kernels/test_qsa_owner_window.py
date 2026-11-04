# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Request-prefix address certificates and bounded staging allocation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.sparse_gqa_prefill import owner_window as window
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_GATHER_MAX_CACHE_BYTES,
)


def inputs(starts, lengths, columns=5, page_size=128, budget=8):
    starts = np.asarray(starts, np.int32)
    lengths = np.asarray(lengths, np.int32)
    table = np.full((len(lengths), columns), np.iinfo(np.int32).max, np.int32)
    blocks = np.full((len(lengths), budget), -1, np.int32)
    for request, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        prefix = (max(lengths[a:b], default=0) + page_size - 1) // page_size
        # Repeated physical pages deliberately preserve aliasing.
        table[a:b, :prefix] = (np.arange(prefix) + request) % 3
    for row, length in enumerate(lengths):
        count = min(length // 4, budget)
        blocks[row, :count] = np.arange(count)
    return blocks, lengths, table, starts


def plan(xs, data_pages):
    return window.plan_pages(
        *map(jnp.asarray, xs),
        page_size=128,
        owner_pages=9,
        ratio=4,
        data_pages=data_pages,
    )


def address_oracle(xs):
    blocks, lengths, table, starts = xs
    ids = []
    remapped = np.zeros_like(table)
    for a, b in zip(starts[:-1], starts[1:]):
        count = (max(lengths[a:b], default=0) + 127) // 128
        offset = len(ids)
        if count:
            ids.extend(table[a, :count])
        for row in range(a, b):
            if lengths[row] > 0:
                remapped[row, :count] = np.arange(count) + offset + 1
    return np.asarray(ids, np.int32), remapped


@pytest.mark.parametrize(
    "starts,lengths",
    [
        ([0, 4], [1, 127, 128, 129]),
        ([0, 0, 2, 2, 4, 4], [129, 257, 1, 128]),
        ([0, 1, 4], [257, 127, 128, 129]),
        ([0, 2, 2, 4], [0, 0, 127, 129]),
        ([0, 4, 4, 4], [0, 0, 0, 0]),
    ],
)
@pytest.mark.parametrize("extra", [0, 3])
def test_prefix_addresses_and_aliases(starts, lengths, extra):
    xs = inputs(starts, lengths)
    before = [x.copy() for x in xs]
    ids, remapped = address_oracle(xs)
    p = plan(xs, max(1, len(ids) + extra))
    assert bool(p.valid)
    assert int(p.count) == len(ids)
    np.testing.assert_array_equal(p.physical[: len(ids)], ids)
    np.testing.assert_array_equal(p.table, remapped)
    # Every selected token maps to the same owner page and within-page offset.
    for row, length in enumerate(xs[1]):
        tokens = list(range(min(length // 4, xs[0].shape[1]) * 4))
        tokens += list(range(length // 4 * 4, length))
        for token in tokens:
            logical, offset = divmod(token, 128)
            compact = int(p.table[row, logical])
            assert compact > 0
            assert ids[compact - 1] == xs[2][row, logical]
            assert offset == token % 128
    for old, new in zip(before, xs):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize(
    "mutation",
    [
        "capacity",
        "start_first",
        "start_last",
        "start_order",
        "negative_length",
        "large_length",
        "physical_negative",
        "physical_oob",
        "table_mismatch",
        "selected_negative",
        "selected_oob",
    ],
)
def test_failed_certificate(mutation):
    xs = list(inputs([0, 2, 4], [129, 257, 127, 129]))
    capacity = 5
    if mutation == "capacity":
        capacity = 4
    elif mutation == "start_first":
        xs[3][0] = 1
    elif mutation == "start_last":
        xs[3][-1] = 3
    elif mutation == "start_order":
        xs[3] = np.array([0, 3, 2, 4], np.int32)
    elif mutation == "negative_length":
        xs[1][0] = -1
    elif mutation == "large_length":
        xs[1][0] = np.iinfo(np.int32).max
    elif mutation == "physical_negative":
        xs[2][:2, 0] = -1
    elif mutation == "physical_oob":
        xs[2][:2, 0] = 9
    elif mutation == "table_mismatch":
        xs[2][1, 0] = 8
    elif mutation == "selected_negative":
        xs[0][0, 0] = -1
    elif mutation == "selected_oob":
        xs[0][0, 0] = xs[1][0] // 4
    assert not bool(plan(xs, capacity).valid)


@pytest.mark.parametrize(
    "dtype,packing,heads",
    [
        (jnp.float8_e4m3fn, 4, 1),
        (jnp.bfloat16, 2, 2),
    ],
)
def test_resource_rounding(dtype, packing, heads):
    cache = jax.ShapeDtypeStruct((15561, 1664, heads, packing, 256), dtype)
    starts = jax.ShapeDtypeStruct((9,), jnp.int32)
    table = jax.ShapeDtypeStruct((8192, 158), jnp.int32)
    page_bytes = 1664 * heads * packing * 256 * jnp.dtype(dtype).itemsize
    for byte_limit in [
        0,
        page_bytes - 1,
        page_bytes,
        2 * page_bytes - 1,
        2 * page_bytes,
        17 * page_bytes + 1,
        SC_GATHER_MAX_CACHE_BYTES * 2,
    ]:
        n = window.window_data_pages(cache, starts, table, byte_limit)
        assert n == max(
            0,
            min(8 * 158, min(byte_limit, SC_GATHER_MAX_CACHE_BYTES) // page_bytes - 1),
        )
        if n:
            assert (n + 1) * page_bytes <= min(byte_limit, SC_GATHER_MAX_CACHE_BYTES)
            assert (n + 1) * 1664 * heads <= np.iinfo(np.int32).max
    with pytest.raises(ValueError, match="nonnegative"):
        window.window_data_pages(cache, starts, table, -1)


@pytest.mark.parametrize(
    "starts,lengths",
    [
        ([2, 2, 4, 4], [0, 0, 1, 129, 0, 0]),
        ([1, 3, 5, 5], [0, 1, 129, 4, 257, 0]),
        ([2, 2, 2], [0, 0, 0, 0]),
        ([0, 0, 4, 4], [1, 127, 128, 129, 0, 0]),
    ],
)
def test_clipped_prefill_interval_requires_inactive_exterior(starts, lengths):
    xs = inputs(starts, lengths)
    ids, expected = address_oracle(xs)
    p = plan(xs, max(1, len(ids)))
    assert bool(p.valid)
    assert int(p.count) == len(ids)
    np.testing.assert_array_equal(p.physical[: len(ids)], ids)
    np.testing.assert_array_equal(p.table, expected)


@pytest.mark.parametrize("row", [0, 5])
def test_clipped_prefill_rejects_active_exterior(row):
    xs = inputs([2, 2, 4, 4], [0, 0, 1, 129, 0, 0])
    xs[1][row] = 1
    assert not bool(plan(xs, 4).valid)
