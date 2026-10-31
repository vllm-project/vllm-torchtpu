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
"""CPU checks for bounded TC packs, ownership and exactly-once gating."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.kernels.prefill_work_model_helpers import work_model
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.sparse_gqa_prefill import adaptive
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import sc_cost, tc_cost


@pytest.fixture
def core_calls(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    calls = []

    def pallas(kernel, *, grid_spec, **kwargs):
        def run(table, tiles, counts, last, query, bits, lengths, cache):
            nblocks = grid_spec.grid[0]
            owners = table.reshape(nblocks, -1)[:, 0]
            owners = jnp.repeat(owners, len(query) // nblocks)
            calls.append((query.shape, counts, bits, lengths, owners))
            return (query.astype(jnp.float32) + owners[:, None, None]).astype(
                query.dtype
            )

        return run

    monkeypatch.setattr(pl, "pallas_call", pallas)
    return calls


def inputs(tokens, starts, columns=1):
    row = jnp.arange(tokens, dtype=jnp.float32)
    query = jnp.broadcast_to((row / 128 + 1)[:, None, None], (tokens, 4, 128))
    gate = jnp.broadcast_to(
        (row[:, None, None] % 7 - 3) + jnp.arange(128)[None, None, :] / 64,
        query.shape,
    )
    lengths = 1 + jnp.arange(tokens, dtype=jnp.int32) % 8
    lengths = lengths.at[::11].set(0)
    blocks = jnp.tile(jnp.arange(4, dtype=jnp.int32), (tokens, 1))
    request = jnp.searchsorted(
        jnp.asarray(starts[1:]), jnp.arange(tokens), side="right"
    )
    owners = request % 7
    table = jnp.full((tokens, columns), 999999, jnp.int32).at[:, 0].set(owners)
    return (
        query.astype(jnp.bfloat16),
        gate.astype(jnp.bfloat16),
        blocks,
        lengths,
        table,
        jnp.zeros((8, 896, 2, 2, 128), jnp.bfloat16),
        jnp.asarray(starts, jnp.int32),
    )


def expected(args):
    query, gate, _, lengths, table, _, starts = args
    result = jnp.zeros_like(query)
    # Original request/query order; this does not use the packing descriptors.
    for lo, hi in zip(np.asarray(starts[:-1]), np.asarray(starts[1:])):
        if lo == hi:
            continue
        raw = (query[lo:hi].astype(jnp.float32) + table[lo, 0]).astype(jnp.bfloat16)
        value = bf16(
            bf16(raw).astype(jnp.float32)
            * bf16(jax.nn.sigmoid(gate[lo:hi].astype(jnp.float32))).astype(jnp.float32)
        )
        result = result.at[lo:hi].set(
            jnp.where((lengths[lo:hi] > 0)[:, None, None], value, 0)
        )
    return result


@pytest.mark.parametrize(
    "tokens,starts,columns",
    [
        (1, [0, 0, 1, 1], 1),
        (17, [0, 0, 3, 3, 17, 17], 293),
        (256, [0, 128, 256], 1),
        (385, [0, 3, 3, 258, 385, 385], 1),
    ],
)
def test_general_tc_packs_preserve_owner_tail_and_nonconstant_gate(
    core_calls, tokens, starts, columns
):
    args = inputs(tokens, starts, columns)
    with jax.disable_jit():
        actual, cache = tc.block_sparse_gqa_attention(
            *args, kv_heads=2, ratio=4, query_block=128, kv_tile=128
        )
    np.testing.assert_array_equal(actual, expected(args))
    assert cache is args[5]
    words = ((columns * 896 // 128 + 127) // 128) * 128
    plan = tc._ragged_plan(
        tokens,
        4,
        128,
        words,
        columns * 896 // 4,
        columns,
        4,
        128,
        request_capacity=tokens,
    )
    assert all(shape[0] <= max(tokens, plan[0] * 128) for shape, *_ in core_calls)
    for _, counts, bits, lengths, _ in core_calls:
        inactive = np.asarray(lengths).reshape(len(counts), -1).max(axis=1) == 0
        np.testing.assert_array_equal(np.asarray(counts)[inactive], 0)
        np.testing.assert_array_equal(np.asarray(bits)[inactive], 0)


def test_adaptive_forced_tc_and_offset_preserve_prefix_and_gate(
    core_calls, monkeypatch
):
    args = inputs(273, [0, 3, 131, 273, 273])
    monkeypatch.setattr(
        adaptive, "_select_sc_for_chunk", lambda *args: jnp.array(False)
    )
    with jax.disable_jit():
        actual, cache = adaptive._one_chunk(args[:5], args[5], args[6], work_model())
        prefix = jnp.full_like(args[0], 13)
        updated = adaptive._write_tc_chunk(
            args[:5],
            args[5],
            args[6],
            prefix,
            17,
            256,
            kv_heads=2,
            ratio=4,
            kv_tile=256,
        )
        middle = adaptive._write_tc_chunk(
            args[:5],
            args[5],
            args[6],
            prefix,
            17,
            128,
            tail=True,
            kv_heads=2,
            ratio=4,
            kv_tile=256,
        )
    np.testing.assert_array_equal(actual, expected(args))
    np.testing.assert_array_equal(updated[:17], prefix[:17])
    np.testing.assert_array_equal(updated[17:], expected(args)[17:])
    np.testing.assert_array_equal(middle[:17], prefix[:17])
    np.testing.assert_array_equal(middle[17:145], expected(args)[17:145])
    np.testing.assert_array_equal(middle[145:], prefix[145:])
    assert cache is args[5]


def test_padded_request_capacity_preserves_output_and_static_bounds(core_calls):
    args = inputs(17, [0, 3, 17])
    run = functools.partial(
        tc.block_sparse_gqa_attention, kv_heads=2, ratio=4, query_block=128, kv_tile=128
    )
    with jax.disable_jit():
        short, _ = run(*args)
        short_shapes = [shape for shape, *_ in core_calls]
        core_calls.clear()
        padded, _ = run(*args[:-1], jnp.pad(args[-1], (0, 1024), constant_values=17))
    # Padded capacity loosens the static bound, within the true-row/byte ceiling.
    assert all(shape[0] <= 2 * 128 for shape in short_shapes)
    assert all(shape[0] <= 17 * 128 for shape, *_ in core_calls)
    np.testing.assert_array_equal(short, padded)


@pytest.mark.parametrize("tokens,offset,size", [(273, 0, 273), (385, 17, 273)])
def test_multiple_source_windows_preserve_both_output_sentinels(
    core_calls, monkeypatch, tokens, offset, size
):
    starts = [0, 3, 131, 273] + ([385] if tokens == 385 else [])
    args = inputs(tokens, starts)
    original_plan = tc._ragged_plan
    tile_rows = []

    def one_tile_plan(
        size,
        query_heads,
        head_dim,
        words,
        groups,
        columns,
        budget,
        query_block,
        *,
        request_capacity: int,
        query_dtype=jnp.bfloat16,
    ):
        _, _, selection_rows = original_plan(
            size,
            query_heads,
            head_dim,
            words,
            groups,
            columns,
            budget,
            query_block,
            query_dtype=query_dtype,
            request_capacity=request_capacity,
        )
        source_rows = min(size, query_block)
        tile_rows.append(query_block)
        return 1, source_rows, min(source_rows, selection_rows)

    monkeypatch.setattr(tc, "_ragged_plan", one_tile_plan)
    windows = []
    original = tc._gate_output

    def observed(output, gate, lengths):
        windows.append(len(output))
        return original(output, gate, lengths)

    monkeypatch.setattr(tc, "_gate_output", observed)
    sentinel = jnp.full_like(args[0], 13)
    with jax.disable_jit():
        actual = adaptive._write_tc_chunk(
            args[:5],
            args[5],
            args[6],
            sentinel,
            offset,
            size,
            tail=True,
            kv_heads=2,
            ratio=4,
            kv_tile=256,
        )
    wanted = sentinel.at[offset : offset + size].set(
        expected(args)[offset : offset + size]
    )
    assert len(windows) > 1
    assert len(core_calls) > 1
    assert all(rows == tile_rows[0] for rows in windows)
    assert all(shape[0] == tile_rows[0] for shape, *_ in core_calls)
    np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(actual[:offset], sentinel[:offset])
    np.testing.assert_array_equal(actual[offset + size :], sentinel[offset + size :])


@pytest.mark.parametrize("size,split", [(17, 3), (256, 1), (385, 129)])
def test_tile_descriptors_and_call_count_match_actual_packs(size, split):
    starts = jnp.array([0, 0, split, split, size, size], jnp.int32)
    first, ends, total = tc._request_tiles(starts, 0, size, 128)
    plan = tc._ragged_plan(size, 24, 256, 128, 416, 1, 512, 128, request_capacity=size)
    pack_tiles, source_rows, _ = plan
    reference_tiles = [
        (lo + step, min(lo + step + 128, hi))
        for lo, hi in zip(np.asarray(starts[:-1]), np.asarray(starts[1:]))
        for step in range(0, int(hi - lo), 128)
    ]
    assert (
        list(zip(np.asarray(first)[: int(total)], np.asarray(ends)[: int(total)]))
        == reference_tiles
    )
    calls, cursor, visited = 0, 0, []
    while cursor < int(total):
        stop = min(
            int(
                np.searchsorted(
                    ends, min(int(first[cursor]) + source_rows, size), side="right"
                )
            ),
            int(total),
        )
        calls += (stop - cursor + pack_tiles - 1) // pack_tiles
        for a, b in zip(np.asarray(first)[cursor:stop], np.asarray(ends)[cursor:stop]):
            visited.extend(range(int(a), int(b)))
        cursor = stop
    assert visited == list(range(size))
    assert (
        int(tc._ragged_call_count(first, ends, total, source_rows, pack_tiles, size))
        == calls
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_batched_cost_uses_external_calls_and_bounds_fine_cost(dtype):
    starts = jnp.arange(257, dtype=jnp.int32)
    first, ends, tiles = tc._request_tiles(starts, 0, 256, 256)
    pack_tiles, source_rows, _ = tc._ragged_plan(
        256, 24, 256, 128, 416, 1, 512, 256, request_capacity=256
    )
    calls = tc._ragged_call_count(first, ends, tiles, source_rows, pack_tiles, 256)
    assert int(calls) < int(tiles)
    model = work_model(dtype)
    lower = tc_cost(
        model,
        queries=calls * pack_tiles * 256,
        num_tiles=7,
        tile_visits=jnp.int32(0),
        launches=calls,
    )
    fine = tc_cost(
        model,
        queries=calls * pack_tiles * 256,
        num_tiles=7,
        tile_visits=tiles,
        launches=calls,
    )
    assert bool(lower <= fine)


def test_selection_rows_have_separate_scratch_bound_at_long_context(monkeypatch):
    words, budget = 2048, 512
    _, source_rows, selection_rows = tc._ragged_plan(
        8192, 24, 256, words, words * 32, 1024, budget, 256, request_capacity=8192
    )
    index_bytes = jnp.dtype(jnp.int32).itemsize
    mask_bytes = jnp.dtype(jnp.bool_).itemsize
    # Masked and sorted K IDs; selected/word/masks and two-column indices K+1.
    row_bytes = (2 * budget + 5 * (budget + 1) + words + 3) * index_bytes
    row_bytes += (4 * budget + 1) * mask_bytes
    assert selection_rows == min(source_rows, tc.TC_SELECTION_TARGET_BYTES // row_bytes)
    assert selection_rows * row_bytes <= tc.TC_SELECTION_TARGET_BYTES
    monkeypatch.setattr(tc, "TC_SELECTION_TARGET_BYTES", row_bytes * 2)
    _, same_source_rows, limited_rows = tc._ragged_plan(
        8192, 24, 256, words, words * 32, 1024, budget, 256, request_capacity=8192
    )
    assert same_source_rows == source_rows
    assert limited_rows == 2


def test_long_mixed_requests_use_packed_source_windows():
    size, query_block = 8192, 256
    starts = [0, 1, size]
    pack_tiles, source_rows, selection_rows = tc._ragged_plan(
        size, 24, 256, 2176, 65728, 158, 512, query_block, request_capacity=size
    )
    assert (pack_tiles, source_rows, selection_rows) == (15, 3840, 3840)
    reference_tiles = [
        (lo + step, min(lo + step + query_block, hi))
        for lo, hi in zip(starts[:-1], starts[1:])
        for step in range(0, hi - lo, query_block)
    ]

    def reference_calls(window_rows):
        cursor, calls = 0, 0
        while cursor < len(reference_tiles):
            limit = min(reference_tiles[cursor][0] + window_rows, size)
            stop = cursor
            while stop < len(reference_tiles) and reference_tiles[stop][1] <= limit:
                stop += 1
            assert stop > cursor
            calls += (stop - cursor + pack_tiles - 1) // pack_tiles
            cursor = stop
        return calls

    assert reference_calls(query_block) == 33
    assert reference_calls(source_rows) == 3
    first, ends, total = tc._request_tiles(
        jnp.asarray(starts, jnp.int32), 0, size, query_block
    )
    assert (
        list(zip(np.asarray(first)[: int(total)], np.asarray(ends)[: int(total)]))
        == reference_tiles
    )
    assert (
        int(tc._ragged_call_count(first, ends, total, source_rows, pack_tiles, size))
        == 3
    )


def test_source_block_gather_is_in_the_packing_byte_bound():
    size, budget, query_block = 8192, 65536, 256
    pack_tiles, source_rows, selection_rows = tc._ragged_plan(
        size, 24, 256, 2176, 65728, 158, budget, query_block, request_capacity=size
    )
    row_bytes = 4 * (24 * 256 * 2) + 2 * 2176 * 4 + 158 * 4 + budget * 4 + 16
    assert pack_tiles == 3
    assert source_rows <= pack_tiles * query_block
    assert source_rows * row_bytes <= 256 * 1024**2
    assert source_rows * budget * 4 <= 256 * 1024**2
    assert selection_rows <= source_rows
    assert tc._ragged_plan(
        256, 24, 256, 128, 416, 1, 512, 256, request_capacity=256
    ) == (20, 256, 256)


def test_true_query_selection_is_shared_between_pack_calls(core_calls, monkeypatch):
    args = list(inputs(23, list(range(24))))
    args[0] = jnp.broadcast_to(args[0][:, :1, :1], (23, 48, 256))
    args[1] = jnp.broadcast_to(args[1][:, :1, :1], (23, 48, 256))
    args[5] = jnp.zeros((8, 896, 2, 2, 256), jnp.bfloat16)
    rows, gated_rows = [], []
    original = tc._selection_bitset
    original_gate = tc._gate_output

    def observed(blocks, lengths, **kwargs):
        rows.append(len(blocks))
        return original(blocks, lengths, **kwargs)

    monkeypatch.setattr(tc, "_selection_bitset", observed)

    def observed_gate(output, gate, lengths):
        gated_rows.append(len(output))
        return original_gate(output, gate, lengths)

    monkeypatch.setattr(tc, "_gate_output", observed_gate)
    with jax.disable_jit():
        output, _ = tc.block_sparse_gqa_attention(
            *args, kv_heads=2, ratio=4, query_block=128, kv_tile=128
        )
    assert len(core_calls) > 1
    assert sum(rows) == 23
    assert sum(gated_rows) == 23
    np.testing.assert_array_equal(output, expected(args))


def test_aligned_selection_keeps_one_call_without_ragged_descriptors(monkeypatch):
    calls = []

    def selected(*args, **kwargs):
        calls.append(1)
        return jnp.array(False)

    def descriptor(*args):
        raise AssertionError("Aligned selection entered the ragged fine path")

    monkeypatch.setattr(adaptive, "select_sc", selected)
    monkeypatch.setattr(tc, "_request_tiles", descriptor)
    with jax.disable_jit():
        choice = adaptive._select_sc_for_chunk(
            jnp.full((256, 512), -1, jnp.int32),
            jnp.zeros(256, jnp.int32),
            jnp.array([0, 256], jnp.int32),
            0,
            256,
            7,
            jnp.bfloat16,
            tc._ragged_plan(256, 24, 256, 128, 416, 1, 512, 256, request_capacity=256),
            work_model(jnp.bfloat16),
        )
    assert not bool(choice)
    assert calls == []


@pytest.mark.parametrize("ratio,groups,words", [(4, 416, 128), (8, 65536, 2176)])
def test_integrated_sort_matches_scalar_or_for_prefix_tail_and_invalid_duplicates(
    ratio, groups, words
):
    blocks = np.full((4, 512), -1, np.int32)
    blocks[:, :8] = [0, 31, 31, 32, 63, 64, groups - 1, groups]
    lengths = np.array([0, ratio - 1, 65 * ratio + 1, groups * ratio - 1], np.int32)
    wanted = np.zeros((4, words), np.uint32)
    for row, length in enumerate(lengths):
        complete = min(int(length) // ratio, groups)
        for group in blocks[row, : min(complete, blocks.shape[1])]:
            if length > 0 and 0 <= group < complete:
                wanted[row, group // 32] |= np.uint32(1) << np.uint32(group % 32)
        tail = int(length) // ratio
        if length > 0 and length % ratio and tail < groups:
            wanted[row, tail // 32] |= np.uint32(1) << np.uint32(tail % 32)
    actual = jax.jit(
        functools.partial(tc._selection_bitset, ratio=ratio, groups=groups, words=words)
    )(jnp.asarray(blocks), jnp.asarray(lengths))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), wanted)


def test_global_selection_wrapper_chunks_long_context_rows(monkeypatch):
    monkeypatch.setattr(tc, "TC_SELECTION_TARGET_BYTES", 64 * 1024)
    row_counts = []
    original = tc._selection_bitset_chunk

    def observed(blocks, lengths, **kwargs):
        row_counts.append(len(blocks))
        return original(blocks, lengths, **kwargs)

    monkeypatch.setattr(tc, "_selection_bitset_chunk", observed)
    rows, groups, words = 129, 65536, 2176
    limit = tc._selection_batch_rows(4, groups, words)
    with jax.disable_jit():
        bits = tc._selection_bitset(
            jnp.tile(jnp.arange(4, dtype=jnp.int32), (rows, 1)),
            jnp.full(rows, 17, jnp.int32),
            ratio=4,
            groups=groups,
            words=words,
        )
    wanted = np.zeros((rows, words), np.int32)
    wanted[:, 0] = 31
    np.testing.assert_array_equal(bits, wanted)
    assert len(row_counts) > 1
    assert max(row_counts) <= limit


def test_bounded_metadata_parallel_rows_share_the_same_scratch_budget(monkeypatch):
    monkeypatch.setattr(tc, "TC_SELECTION_TARGET_BYTES", 2 * 1024**2)
    calls = []
    original = tc.lax.map

    def observed(*args, **kwargs):
        calls.append(kwargs.get("batch_size"))
        return original(*args, **kwargs)

    monkeypatch.setattr(tc.lax, "map", observed)
    with jax.disable_jit():
        _, _, counts, bits = tc._prepare_metadata_bounded(
            jnp.zeros((256, 4), jnp.int32),
            jnp.ones(256, jnp.int32),
            jnp.zeros((256, 293), jnp.int32),
            ratio=4,
            query_block=128,
            kv_tile=128,
            physical_page_size=896,
            batch_size=8,
        )
    assert calls[0] == 1
    np.testing.assert_array_equal(counts, [1, 1])
    np.testing.assert_array_equal(np.asarray(bits)[:, 0], 1)
    np.testing.assert_array_equal(np.asarray(bits)[:, 1:], 0)


@pytest.mark.parametrize("requests", [2, 8, 256])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_cheap_batched_lower_bounds_fragmented_fine_work(requests, dtype):
    starts = jnp.asarray(np.linspace(0, 256, requests + 1, dtype=np.int32))
    first, ends, total = tc._request_tiles(starts, 0, 256, 256)
    pack_tiles = 5
    cheap_calls = (total + pack_tiles - 1) // pack_tiles
    for source_rows in (256, 129, 64):
        # Each request tile here is at most 128 rows, or one row at R256.
        if source_rows < int(np.max(np.diff(starts))):
            continue
        actual_calls = tc._ragged_call_count(
            first, ends, total, source_rows, pack_tiles, 256
        )
        assert int(cheap_calls) <= int(actual_calls)
        model = work_model(dtype)
        lower = tc_cost(
            model,
            queries=cheap_calls * pack_tiles * 256,
            num_tiles=7,
            tile_visits=jnp.int32(0),
            launches=cheap_calls,
        )
        fine = tc_cost(
            model,
            queries=actual_calls * pack_tiles * 256,
            num_tiles=7,
            tile_visits=total,
            launches=actual_calls,
        )
        assert bool(lower <= fine)


@pytest.mark.parametrize("requests,expected_fine", [(8, 0), (32, 1), (256, 0)])
def test_fast_sc_skips_descriptors_and_undecided_can_choose_batched_tc(
    monkeypatch, requests, expected_fine
):
    descriptors = []
    original = tc._request_tiles

    def observed(*args):
        descriptors.append(1)
        return original(*args)

    monkeypatch.setattr(tc, "_request_tiles", observed)
    starts = jnp.asarray(np.linspace(0, 256, requests + 1, dtype=np.int32))
    ids = jnp.full((256, 512), -1, jnp.int32).at[:, :16].set(jnp.arange(16))
    lengths = jnp.full(256, 64, jnp.int32)
    plan = tc._ragged_plan(256, 24, 256, 128, 416, 1, 512, 256, request_capacity=256)
    model = work_model(jnp.bfloat16)
    with jax.disable_jit():
        choose = adaptive._select_sc_for_chunk(
            ids,
            lengths,
            starts,
            0,
            256,
            7,
            jnp.bfloat16,
            plan,
            model,
        )
    assert bool(choose) == (requests == 256)
    assert len(descriptors) == expected_fine


def test_request_capacity_bounds_partial_tiles_and_preserves_many_request_plan():
    import itertools

    for lengths in itertools.product([0, 1, 127, 128, 255, 256, 257], repeat=3):
        size = sum(lengths)
        if not size:
            continue
        actual = sum((n + 255) // 256 for n in lengths)
        upper = min(size, (size + 255) // 256 + min(len(lengths), size) - 1)
        assert actual <= upper
        pack, _, _ = tc._ragged_plan(
            size, 8, 128, 128, 32, 2, 16, 256, request_capacity=len(lengths)
        )
        assert 1 <= pack <= upper
    assert tc._ragged_plan(256, 8, 128, 128, 32, 2, 16, 256, request_capacity=2) == (
        2,
        256,
        256,
    )
    assert tc._ragged_plan(
        256, 24, 256, 128, 416, 1, 512, 256, request_capacity=256
    ) == (20, 256, 256)


def test_two_short_requests_choose_tc_after_capacity_is_accounted(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    query = jax.ShapeDtypeStruct((256, 4, 128), jnp.bfloat16)
    blocks = jnp.tile(jnp.arange(16, dtype=jnp.int32), (256, 1))
    lengths = 64 + jnp.arange(256, dtype=jnp.int32) % 4
    table = jax.ShapeDtypeStruct((256, 1), jnp.int32)
    cache = jax.ShapeDtypeStruct((4, 256, 1, 4, 128), jnp.float8_e4m3fn)
    model = work_model(
        jnp.float8_e4m3fn,
        heads=4,
        dim=128,
        budget=16,
        page_size=256,
        kv_heads=1,
        ratio=4,
    )
    plan = adaptive._tc_plan(
        query,
        blocks,
        table,
        cache,
        256,
        kv_heads=1,
        ratio=4,
        kv_tile=256,
        request_capacity=2,
    )
    assert plan == (2, 256, 256)
    choose = adaptive._select_sc_for_chunk(
        blocks,
        lengths,
        jnp.array([0, 128, 256], jnp.int32),
        0,
        256,
        1,
        cache.dtype,
        plan,
        model,
    )
    assert not bool(choose)


@pytest.mark.parametrize(
    "size,starts,offset",
    [
        (256, [0, 128, 256], 0),
        (128, [0, 17, 145, 273], 17),
        (17, [0, 3, 17], 0),
        (256, [0, 0, 0], 0),
    ],
)
def test_single_pack_tc_bound_matches_independent_fine_work(
    monkeypatch, size, starts, offset
):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    model = work_model(
        jnp.float8_e4m3fn,
        heads=4,
        dim=128,
        budget=16,
        page_size=256,
        kv_heads=1,
        ratio=4,
    )
    lengths = 64 + np.arange(size, dtype=np.int32) % 4
    if starts[-1] == 0:
        lengths[:] = 0
    ids = np.tile(np.arange(16, dtype=np.int32), (size, 1))
    ids[lengths == 0] = -1
    plan = tc._ragged_plan(
        size, 4, 128, 128, 64, 1, 16, 256, request_capacity=len(starts) - 1
    )
    pack, source_rows, _ = plan
    local = np.clip(np.asarray(starts) - offset, 0, size)
    visits, tiles = 0, 0
    for lo, hi in zip(local[:-1], local[1:]):
        for first in range(int(lo), int(hi), 256):
            touched = set()
            for row in range(first, min(first + 256, int(hi))):
                if lengths[row] > 0:
                    for group in ids[row]:
                        if group >= 0:
                            touched.update(
                                token // 256
                                for token in range(int(group) * 4, (int(group) + 1) * 4)
                            )
                    touched.update(
                        token // 256
                        for token in range(
                            int(lengths[row]) // 4 * 4, int(lengths[row])
                        )
                    )
            visits += len(touched)
            tiles += 1
    assert source_rows == size and tiles <= pack
    calls = int(tiles > 0)
    fine = tc_cost(
        model,
        queries=calls * pack * 256,
        num_tiles=1,
        tile_visits=jnp.int32(visits),
        launches=calls,
    )
    upper = tc_cost(
        model, queries=pack * 256, num_tiles=1, tile_visits=jnp.int32(pack), launches=1
    )
    assert float(fine) <= float(upper)
    sc_rows = (size + model.sc_chunk - 1) // model.sc_chunk * model.sc_chunk
    ls = jnp.pad(jnp.asarray(lengths), (0, sc_rows - size))
    target = float(sc_cost(model, ls)) * 1.02
    choice = adaptive._select_sc_for_chunk(
        jnp.asarray(ids),
        jnp.asarray(lengths),
        jnp.asarray(starts, jnp.int32),
        offset,
        size,
        1,
        jnp.float8_e4m3fn,
        plan,
        model,
    )
    assert bool(choice) == (float(fine) > target)


def test_single_pack_static_tc_skips_runtime_descriptor_work(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    descriptors = []
    original = tc._request_tiles

    def observed(*args):
        descriptors.append(1)
        return original(*args)

    monkeypatch.setattr(tc, "_request_tiles", observed)
    model = work_model(
        jnp.float8_e4m3fn,
        heads=4,
        dim=128,
        budget=16,
        page_size=256,
        kv_heads=1,
        ratio=4,
    )
    with jax.disable_jit():
        choice = adaptive._select_sc_for_chunk(
            jnp.tile(jnp.arange(16, dtype=jnp.int32), (256, 1)),
            jnp.full(256, 64, jnp.int32),
            jnp.array([0, 128, 256], jnp.int32),
            0,
            256,
            1,
            jnp.float8_e4m3fn,
            (2, 256, 256),
            model,
        )
    assert not bool(choice)
    assert descriptors == []


def test_empty_request_partition_keeps_output_and_skips_core(core_calls):
    args = inputs(17, [0, 0, 0])
    sentinel = jnp.full_like(args[0], 13)
    with jax.disable_jit():
        result = tc._ragged_prefill(
            *args[:6],
            args[6],
            sentinel,
            0,
            17,
            ratio=4,
            kv_heads=2,
            query_block=128,
            kv_tile=128,
            k_scale=1.0,
            v_scale=1.0,
        )
    np.testing.assert_array_equal(result, sentinel)
    assert core_calls == []
