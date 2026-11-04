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
"""CPU regressions for live sparse-prefill addressing and bounded dispatch."""

import functools
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles
from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive, grouped
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_GATHER_MAX_CACHE_BYTES,
)


@pytest.fixture
def tpu_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


def metadata_inputs():
    blocks = np.full((128, 4), -1, np.int32)
    lengths = np.zeros(128, np.int32)
    lengths[:6] = [0, 1, 4, 5, 8, 129]
    blocks[0] = [0, 1, 2, 3]
    blocks[1] = [0, 1, 2, 3]
    blocks[2] = [0, 1, 2, 3]
    blocks[3] = [0, 1, -1, -1]
    blocks[4] = [0, 2, 1, 3]
    blocks[5] = [0, 1, 2, 3]
    table = np.broadcast_to(np.arange(4, dtype=np.int32), (128, 4)).copy()
    return jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table)


def test_bounded_metadata_filters_inactive_prefix_and_implicit_tail(tpu_info):
    blocks, lengths, table = metadata_inputs()
    _, tiles, counts, bits = tc._prepare_metadata_bounded(
        blocks, lengths, table, query_block=128, physical_page_size=128
    )
    actual = np.asarray(bits)[0, :, :128].T
    expected = np.zeros((128, 128), np.int32)
    expected[:6, 0] = [0, 1, 1, 3, 1, 15]
    expected[5, 1] = 1
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.asarray(counts), [1])
    assert int(tiles[0, 0]) == 0


def test_bounded_metadata_duplicate_bits_cannot_carry_to_another_group(tpu_info):
    blocks = jnp.full((128, 4), -1, jnp.int32).at[0, :2].set(0)
    lengths = jnp.zeros(128, jnp.int32).at[0].set(16)
    table = jnp.zeros((128, 1), jnp.int32)
    _, _, _, bits = tc._prepare_metadata_bounded(
        blocks, lengths, table, query_block=128, physical_page_size=128
    )
    assert int(bits[0, 0, 0]) == 1
    np.testing.assert_array_equal(np.asarray(bits)[0, 1:], 0)


def capture_tc(
    monkeypatch,
    query,
    blocks,
    lengths,
    table,
    cache,
    *,
    ratio=4,
    kv_tile=256,
    bf16_io=True,
):
    captured = {}

    def call(kernel, *, out_shape, **kwargs):
        captured.update(kwargs)

        def run(*args):
            captured["args"] = args
            return jnp.zeros(out_shape.shape, out_shape.dtype)

        return run

    monkeypatch.setattr(pl, "pallas_call", call)
    with jax.disable_jit():
        result = tc.block_sparse_gqa_prefill(
            query,
            blocks,
            lengths,
            table,
            cache,
            ratio=ratio,
            kv_heads=2,
            query_block=128,
            kv_tile=kv_tile,
            bf16_io=bf16_io,
        )
    return captured, result


def test_full_metadata_matches_bounded_and_prefetches_last_live_page(
    monkeypatch, tpu_info
):
    blocks, lengths, table = metadata_inputs()
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    cache = jnp.zeros((4, 128, 2, 2, 128), jnp.bfloat16)
    captured, _ = capture_tc(monkeypatch, query, blocks, lengths, table, cache)
    args = captured["args"]
    np.testing.assert_array_equal(args[3], [1])
    _, _, _, expected = tc._prepare_metadata_bounded(
        blocks, lengths, table, query_block=128, physical_page_size=128
    )
    np.testing.assert_array_equal(args[5], expected)
    cost = captured["cost_estimate"]
    assert cost.flops > 0 and cost.transcendentals > 0
    assert cost.bytes_accessed >= query.nbytes + cache.nbytes
    assert captured["compiler_params"].vmem_limit_bytes < tpu_info.vmem_capacity_bytes


@pytest.mark.parametrize("length,last", [(0, 0), (1664, 0), (1665, 1)])
def test_prefetch_bound_ignores_padded_table_capacity(
    monkeypatch, tpu_info, length, last
):
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    blocks = jnp.full((128, 4), -1, jnp.int32)
    lengths = jnp.full(128, length, jnp.int32)
    table = jnp.zeros((128, 158), jnp.int32).at[:, 2:].set(999999)
    cache = jnp.zeros((2, 1664, 2, 2, 128), jnp.bfloat16)
    captured, _ = capture_tc(monkeypatch, query, blocks, lengths, table, cache)
    assert int(captured["args"][3][0]) == last
    if length == 0:
        np.testing.assert_array_equal(captured["args"][2], 0)


def test_ratio8_page1664_general_tile_is_legal(monkeypatch, tpu_info):
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    blocks = jnp.full((128, 4), -1, jnp.int32)
    lengths = jnp.full(128, 1664, jnp.int32)
    table = jnp.zeros((128, 3), jnp.int32)
    cache = jnp.zeros((1, 1664, 2, 2, 128), jnp.bfloat16)
    captured, result = capture_tc(
        monkeypatch, query, blocks, lengths, table, cache, ratio=8
    )
    assert result.shape == query.shape
    assert int(captured["args"][3][0]) == 0


def test_ragged_general_tc_uses_bounded_pack_and_each_requests_table(
    monkeypatch, tpu_info
):
    sizes = []

    def fake(query, blocks, lengths, table, cache, bitset, **kwargs):
        sizes.append(len(query))
        return jnp.broadcast_to(table[:, :1, None], query.shape).astype(query.dtype)

    monkeypatch.setattr(tc, "_prefill", fake)
    query = jnp.zeros((17, 4, 128), jnp.bfloat16)
    lengths = jnp.ones(17, jnp.int32)
    table = jnp.concatenate((jnp.full((3, 1), 2), jnp.full((14, 1), 7)))
    args = (
        query,
        query,
        jnp.zeros((17, 4), jnp.int32),
        lengths,
        table,
        jnp.zeros((8, 128, 2, 2, 128), jnp.bfloat16),
    )
    short = jnp.array([0, 3, 17], jnp.int32)
    padded = jnp.pad(short, (0, 253), constant_values=17)
    run = functools.partial(
        tc.block_sparse_gqa_attention, kv_heads=2, ratio=4, query_block=128, kv_tile=256
    )
    actual, _ = run(*args, padded)
    expected, _ = run(*args, short)
    # Each nonempty request may add one partial tile; empty capacity is capped
    # by the number of query rows, not the padded starts array.
    assert sizes == [len(query) * 128, (len(short) - 1) * 128]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual[:, 0, 0]), [1] * 3 + [3.5] * 14)


@pytest.mark.parametrize(
    "page_size,columns,heads,chip",
    [
        (896, 293, 12, pltpu.ChipVersion.TPU_7X),
        (1664, 159, 24, pltpu.ChipVersion.TPU_7),
    ],
)
def test_adaptive_admission_uses_resources_not_calibration_geometry(
    monkeypatch, page_size, columns, heads, chip
):
    info = pltpu.get_tpu_info_for_chip(chip, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    shape = jax.ShapeDtypeStruct
    q = shape((256, heads, 256), jnp.bfloat16)
    assert adaptive.is_supported_by_sc_window(
        q,
        q,
        shape((256, 512), jnp.int32),
        shape((256,), jnp.int32),
        shape((256, columns), jnp.int32),
        shape((columns, page_size, 2, 2, 256), jnp.bfloat16),
        shape((257,), jnp.int32),
        kv_heads=2,
        ratio=4,
        k_scale=1.0,
        v_scale=1.0,
    )


def test_resource_cap_and_working_set_select_smaller_tiles(monkeypatch, tpu_info):
    shape = jax.ShapeDtypeStruct
    query = shape((256, 48, 256), jnp.bfloat16)
    cache = shape((2, 1664, 4, 2, 256), jnp.bfloat16)
    assert prefill_tiles.get_sparse_prefill_tiles(
        query, cache, kv_heads=4, ratio=4
    ) == (128, 128)
    small = replace(tpu_info, vmem_capacity_bytes=16 * 1024**2)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: small)
    qb, kt = prefill_tiles.get_sparse_prefill_tiles(query, cache, kv_heads=4, ratio=4)
    assert qb < 128
    footprint = prefill_tiles._paired_working_set(48, 4, 256, 2, 2, qb, kt)
    assert footprint <= prefill_tiles.prefill_vmem_limit() < small.vmem_capacity_bytes


@pytest.mark.parametrize("bf16_io", [False, True])
def test_tc_hint_counts_cast_boundaries_and_main_matrix_products(
    monkeypatch, tpu_info, bf16_io
):
    query = jnp.zeros((128, 4, 128), jnp.bfloat16)
    blocks, lengths, table = metadata_inputs()
    cache = jnp.zeros((4, 128, 2, 2, 128), jnp.bfloat16)
    captured, output = capture_tc(
        monkeypatch,
        query,
        blocks,
        lengths,
        table,
        cache,
        kv_tile=128,
        bf16_io=bf16_io,
    )
    args = captured["args"]
    boundary_dtype = jnp.bfloat16 if bf16_io else jnp.float32
    assert args[4].dtype == output.dtype == boundary_dtype
    repeated_kv_bytes = 4 * 128 * 2 * 2 * 128 * 2
    cost = captured["cost_estimate"]
    assert cost.flops == 4 * 128 * 4 * 128 * 128 * 4
    assert cost.transcendentals == 262144 + 2048
    assert (
        cost.bytes_accessed
        == sum(x.nbytes for x in args[:-1]) + output.nbytes + repeated_kv_bytes
    )


def test_sc_attention_hint_separates_score_alpha_and_gate_transcendentals(
    monkeypatch, tpu_info
):
    captured = []

    def compact(*args, **kwargs):
        return jnp.zeros((8 * 256, 2, 2, 128), jnp.bfloat16), 256

    def call(kernel, *, out_shape, **kwargs):
        def run(*args):
            captured.append((kwargs["cost_estimate"], args))
            return jnp.zeros(out_shape.shape, out_shape.dtype)

        return run

    monkeypatch.setattr(pl, "pallas_call", call)
    query = jnp.zeros((8, 12, 128), jnp.bfloat16)
    from types import SimpleNamespace

    compact_array = compact()[0]
    destination = SimpleNamespace(
        shape=(
            compact_array.shape[0] * compact_array.shape[1],
            *compact_array.shape[2:],
        ),
        dtype=compact_array.dtype,
    )
    output = grouped._attend_compact_ref(
        query,
        query,
        jnp.ones(8, jnp.int32),
        destination,
        kv_heads=2,
        ratio=4,
        query_group=8,
        source_group=8,
        tile_k=256,
        capacity=256,
        budget=4,
    )
    cost, args = captured[0]
    assert args[2].nbytes > query.nbytes
    assert cost.flops == 16777216
    assert cost.transcendentals == 32768 + 128 + 16384
    assert cost.bytes_accessed == (
        sum(x.nbytes for x in args) + compact_array.nbytes + args[2].nbytes
    )
    assert args[2].dtype == args[3].dtype == output.dtype == jnp.bfloat16
    assert output.shape == query.shape


@pytest.mark.parametrize("case", ["at_capacity", "over_capacity", "address_overflow"])
def test_sparse_gather_admission_uses_packed_row_address_range(tpu_info, case):
    shape = jax.ShapeDtypeStruct
    q = shape((8, 4, 256), jnp.bfloat16)
    page_elements = 128 * 4 * 256
    page_bytes = page_elements * jnp.dtype(jnp.float8_e4m3fn).itemsize
    pages = SC_GATHER_MAX_CACHE_BYTES // page_bytes
    if case == "over_capacity":
        pages += 1
    elif case == "address_overflow":
        pages = jnp.iinfo(jnp.int32).max // 128 + 1
    expected = case == "at_capacity"
    cache = shape((pages, 128, 1, 4, 256), jnp.float8_e4m3fn)
    # Admission also respects SparseCore's per-tensor gather-source cap,
    # so the packed-row range is exercised with the largest in-cap pool.
    if expected:
        assert cache.size * cache.dtype.itemsize <= SC_GATHER_MAX_CACHE_BYTES
    assert (
        adaptive.is_supported_by_sc_window(
            q,
            q,
            shape((8, 512), jnp.int32),
            shape((8,), jnp.int32),
            shape((8, 1), jnp.int32),
            cache,
            shape((9,), jnp.int32),
            kv_heads=2,
            ratio=4,
            k_scale=1.0,
            v_scale=1.0,
        )
        is expected
    )
    if case == "address_overflow":
        assert not grouped._supports_whole_slab(
            shape((8, 512), jnp.int32),
            shape((8, 1), jnp.int32),
            cache,
            query_group=8,
            tile_k=256,
            padding_rows=128,
        )
