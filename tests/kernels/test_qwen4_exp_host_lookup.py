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
"""Public HOST lookup: byte accuracy, boundary rows, and input contracts."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.host_lookup import (
    _compact_owned_queries,
    finish_host_lookup,
    host_table_num_words,
    start_host_lookup,
)


@pytest.mark.parametrize(
    "table,ids,rows,message",
    [
        ((128, 40), (1, 16), 128, "flat int32"),
        ((5120,), (16,), 128, "tokens, heads"),
        ((5120,), (1, 16), 0, "positive int32"),
        ((5120,), (1, 16), 2**31, "positive int32"),
        ((5080,), (1, 16), 127, "page-aligned"),
        ((5120,), (1, 16), 256, "page-aligned"),
    ],
)
def test_host_lookup_rejects_invalid_layout(table, ids, rows, message):
    with pytest.raises(ValueError, match=message):
        start_host_lookup(
            jax.ShapeDtypeStruct(table, jnp.int32),
            jax.ShapeDtypeStruct(ids, jnp.int32),
            rows,
        )


def test_finish_masks_invalid_rows_and_preserves_hidden():
    ids = np.array([[-(2**31), -1, 0, 127, 128, 2**31 - 1]], np.int32)
    words = np.arange(ids.size * 40, dtype=np.int32).reshape(ids.size, 5, 8)
    hidden = np.arange(24, dtype=np.float32).reshape(2, 12)
    raw, actual_hidden = jax.jit(functools.partial(finish_host_lookup, num_rows=128))(
        jnp.asarray(words), jnp.asarray(ids), jnp.asarray(hidden)
    )
    expected = words.reshape(*ids.shape, 40).copy()
    expected[(ids < 0) | (ids >= 128)] = 0
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(actual_hidden, hidden)


@pytest.mark.tpu_test
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("rows", [64, 192, 128 * 64])
def test_host_lookup_page_crossings_and_last_row(invalid, rows):
    data = np.random.default_rng(20261101).integers(
        -(2**31), 2**31 - 1, size=rows * 40, dtype=np.int32
    )
    choices = [-(2**31), -1, 0, 1, 25, 51, 76, 102, 127, 128, rows - 1, rows, 2**31 - 1]
    ids = np.resize(np.asarray(choices, np.int32), (16, 16))
    if invalid:
        ids.fill(-1)
    device = jax.devices("tpu")[0]
    host = jax.device_put(
        np.pad(data, (0, host_table_num_words(rows) - data.size)),
        jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host"),
    )

    @jax.jit
    def lookup(table, indices, hidden):
        pending = start_host_lookup(table, indices, rows)
        return finish_host_lookup(pending, indices, hidden + 1, rows)

    raw, hidden = lookup(host, jnp.asarray(ids), jnp.zeros((16, 128), jnp.float32))
    jax.block_until_ready((raw, hidden))
    valid = (ids >= 0) & (ids < rows)
    expected = np.zeros((*ids.shape, 40), np.int32)
    expected[valid] = data.reshape(rows, 40)[ids[valid]]
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(hidden, np.ones((16, 128), np.float32))
    assert host.sharding.memory_kind == "pinned_host"


def test_compacted_queries_preserve_encoded_rows():
    rows = 257
    rng = np.random.default_rng(17)
    table = rng.integers(-(2**31), 2**31, (rows, 40), dtype=np.int32)
    empty = np.full(512, -1, dtype=np.int32)
    one = empty.copy()
    one[-1] = rows - 1
    partial = empty.copy()
    partial[:257] = np.arange(257, dtype=np.int32)
    sparse = empty.copy()
    sparse[::8] = np.arange(64, dtype=np.int32)
    cases = [
        empty,
        one,
        partial,
        sparse,
        np.arange(512, dtype=np.int32) % rows,
        np.resize(np.array([25, 26, 51, 127, 128, 256], np.int32), 512),
        np.resize(np.array([-(2**31), -1, rows, 2**31 - 1], np.int32), 512),
    ]
    run = jax.jit(functools.partial(_compact_owned_queries, num_rows=rows))
    jax.block_until_ready(run(jnp.asarray(cases[0].reshape(-1, 16))))
    with jax.no_tracing():
        for ids in cases:
            metadata, count = run(jnp.asarray(ids.reshape(-1, 16)))
            pairs = np.asarray(metadata).reshape(-1, 2)
            count = int(count)
            valid = (ids >= 0) & (ids < rows)
            np.testing.assert_array_equal(pairs[:count, 1], np.flatnonzero(valid))
            np.testing.assert_array_equal(pairs[:count, 0], ids[valid])
            np.testing.assert_array_equal(pairs[count:], -1)
            actual = np.zeros((ids.size, 40), dtype=np.int32)
            actual[pairs[:count, 1]] = table[pairs[:count, 0]]
            expected = np.zeros_like(actual)
            expected[valid] = table[ids[valid]]
            np.testing.assert_array_equal(actual, expected)


def test_compacted_dp_shards_preserve_uneven_ownership():
    total_rows, ranks, shard_rows = 2053, 8, 257
    rng = np.random.default_rng(23)
    table = rng.integers(-(2**31), 2**31, (total_rows, 40), dtype=np.int32)
    ids = rng.integers(-1, total_rows + 1, (256, 16), dtype=np.int32)
    flat = ids.reshape(-1)
    combined = np.zeros((ids.size, 40), dtype=np.int32)
    owners = np.zeros(ids.size, dtype=np.int32)
    for rank in range(ranks):
        start = rank * shard_rows
        count_rows = min(shard_rows, total_rows - start)
        metadata, count = jax.jit(_compact_owned_queries, static_argnums=1)(
            jnp.asarray(ids - start), count_rows
        )
        pairs = np.asarray(metadata).reshape(-1, 2)[: int(count)]
        combined[pairs[:, 1]] += table[start + pairs[:, 0]]
        owners[pairs[:, 1]] += 1
    valid = (flat >= 0) & (flat < total_rows)
    np.testing.assert_array_equal(owners, valid.astype(np.int32))
    expected = np.zeros_like(combined)
    expected[valid] = table[flat[valid]]
    np.testing.assert_array_equal(combined, expected)


def test_compaction_at_dp8_prefill_query_count():
    rows = 4099
    rng = np.random.default_rng(31)
    ids = rng.integers(0, rows * 8, (8192 * 8, 16), dtype=np.int32)
    metadata, count = jax.jit(_compact_owned_queries, static_argnums=1)(
        jnp.asarray(ids), rows
    )
    valid = ids.reshape(-1) < rows
    pairs = np.asarray(metadata).reshape(-1, 2)
    count = int(count)
    np.testing.assert_array_equal(pairs[:count, 1], np.flatnonzero(valid))
    np.testing.assert_array_equal(pairs[:count, 0], ids.reshape(-1)[valid])
    np.testing.assert_array_equal(pairs[count:], -1)
    assert count == np.count_nonzero(valid)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens,num_shards", [(64, 8), (512, 1), (512, 8)])
def test_host_query_compaction_dma(tokens, num_shards):
    if jax.default_backend() != "tpu":
        pytest.skip("Native HOST DMA requires TPU")
    rows = 257
    rng = np.random.default_rng(29)
    bits = rng.integers(-(2**31), 2**31, host_table_num_words(rows), dtype=np.int32)
    device = jax.devices()[0]
    host_sharding = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
    device_sharding = jax.sharding.SingleDeviceSharding(device)
    table = jax.device_put(bits, host_sharding)

    def lookup(table, ids, hidden):
        pending = start_host_lookup(table, ids, rows, num_shards=num_shards)
        return finish_host_lookup(pending, ids, hidden, rows)

    run = jax.jit(
        lookup,
        in_shardings=(host_sharding, device_sharding, device_sharding),
        out_shardings=(device_sharding, device_sharding),
    )
    none = np.full(tokens * 16, -1, dtype=np.int32)
    last = none.copy()
    last[-1] = rows - 1
    partial = none.copy()
    partial[:257] = np.arange(257, dtype=np.int32)
    sparse = none.copy()
    sparse[::8] = np.arange(tokens * 2, dtype=np.int32) % rows
    cases = [
        none,
        last,
        partial,
        np.arange(tokens * 16, dtype=np.int32) % rows,
        np.resize(np.array([25, 26, 51, 127, 128, 256], np.int32), tokens * 16),
        sparse,
        none,
    ]
    hidden = jax.device_put(np.arange(32, dtype=np.float32), device_sharding)
    ids = jax.device_put(cases[0].reshape(tokens, 16), device_sharding)
    jax.block_until_ready(run(table, ids, hidden))
    with jax.no_tracing():
        for flat in cases:
            ids = jax.device_put(flat.reshape(tokens, 16), device_sharding)
            actual, actual_hidden = run(table, ids, hidden)
            actual = np.asarray(actual).reshape(-1, 40)
            valid = (flat >= 0) & (flat < rows)
            expected = np.zeros_like(actual)
            expected[valid] = bits[: rows * 40].reshape(rows, 40)[flat[valid]]
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                actual_hidden, np.arange(32, dtype=np.float32)
            )
    assert table.sharding.memory_kind == "pinned_host"
