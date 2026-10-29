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
"""CPU checks for sparse-prefill logical workspace planning."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc


@pytest.mark.parametrize("budget,expected_limit", [(512, 5), (65536, 1)])
def test_selection_chunks_match_unique_prefix_and_tail(
    monkeypatch, budget, expected_limit
):
    groups, words = 65728, 2176
    monkeypatch.setattr(tc, "TC_SELECTION_TARGET_BYTES", 128 * 1024)
    limit = tc._selection_batch_rows(budget, groups, words)
    assert limit == expected_limit
    rows = limit + 1
    blocks = np.full((rows, budget), -1, np.int32)
    blocks[1:, :3] = [0, 1, 2]
    lengths = np.full(rows, 13, np.int32)
    lengths[0] = 0
    chunk_rows = []
    original = tc._selection_bitset_chunk

    def observed(blocks, lengths, **kwargs):
        chunk_rows.append(len(blocks))
        return original(blocks, lengths, **kwargs)

    monkeypatch.setattr(tc, "_selection_bitset_chunk", observed)
    with jax.disable_jit():
        actual = tc._selection_bitset(
            jnp.asarray(blocks),
            jnp.asarray(lengths),
            ratio=4,
            groups=groups,
            words=words,
        )
    expected = np.zeros((rows, words), np.int32)
    for row, length in enumerate(lengths):
        for group in blocks[row, : min(int(length) // 4, budget)]:
            if 0 <= group < int(length) // 4:
                expected[row, group // 32] |= np.int32(1 << (group % 32))
        if length > 0 and length % 4:
            group = int(length) // 4
            expected[row, group // 32] |= np.int32(1 << (group % 32))
    np.testing.assert_array_equal(actual, expected)
    assert len(chunk_rows) == 2
    assert chunk_rows == [limit, limit]


@pytest.mark.parametrize(
    "args,expected",
    [
        ((256, 24, 256, 128, 416, 1, 512, 256), (20, 256, 256)),
        ((8192, 24, 256, 2176, 65728, 158, 512, 256), (15, 3840, 3840)),
        ((8192, 24, 256, 2176, 65728, 158, 65536, 256), (3, 768, 127)),
        ((17, 12, 128, 2176, 65632, 293, 16, 128), (17, 17, 17)),
    ],
)
def test_default_plan_and_separate_selection_chunks(args, expected):
    plan = tc._ragged_plan(*args, request_capacity=args[0])
    assert plan == expected
    pack_tiles, source_rows, selection_rows = plan
    assert source_rows <= pack_tiles * args[-1]
    assert selection_rows <= source_rows


def test_compaction_inventory_uses_actual_query_dtype():
    args = (256, 24, 256, 128, 416, 1, 512, 256)
    bf16 = tc._ragged_plan(*args, query_dtype=jnp.bfloat16, request_capacity=args[0])
    fp32 = tc._ragged_plan(*args, query_dtype=jnp.float32, request_capacity=args[0])
    assert fp32[0] < bf16[0]
    assert fp32[1:] == bf16[1:]


def test_soft_selection_target_keeps_one_oversized_row(monkeypatch):
    monkeypatch.setattr(tc, "TC_SELECTION_TARGET_BYTES", 1)
    assert tc._selection_batch_rows(4, 32, 128) == 1


@pytest.mark.parametrize(
    "budget,words,expected",
    [(512, 128, 15856), (512, 2176, 10685), (65536, 2176, 127)],
)
def test_selection_plan_uses_sort_arrays_and_output_width(budget, words, expected):
    assert tc._selection_batch_rows(budget, 65728, words) == expected


def test_real8k_wrapper_has_one_encoder_call(monkeypatch):
    observed = []
    original = tc._selection_bitset_chunk

    def record(blocks, lengths, **kwargs):
        observed.append(blocks.shape[0])
        return original(blocks, lengths, **kwargs)

    monkeypatch.setattr(tc, "_selection_bitset_chunk", record)
    shape = jax.eval_shape(
        lambda blocks, lengths: tc._selection_bitset(
            blocks, lengths, ratio=4, groups=2496, words=128
        ),
        jax.ShapeDtypeStruct((8192, 512), jnp.int32),
        jax.ShapeDtypeStruct((8192,), jnp.int32),
    )
    assert shape.shape == (8192, 128)
    assert shape.dtype == jnp.int32
    assert observed == [8192]


@pytest.mark.parametrize("slots", [0, 1, 32, 64])
def test_integrated_sort_matches_scalar_or_with_implicit_tail(slots):
    blocks = np.resize(np.array([0, 31, 31, 32, 63, -1, 96], np.int32), (6, slots))
    lengths = np.array([0, 1, 4, 129, 255, 385], np.int32)
    expected = np.zeros((6, 4), np.uint32)
    for row, length in enumerate(lengths):
        complete = min(int(length) // 4, 128)
        for group in blocks[row, : min(complete, slots)]:
            if length > 0 and 0 <= group < complete:
                expected[row, group // 32] |= np.uint32(1) << np.uint32(group % 32)
        tail = int(length) // 4
        if length > 0 and length % 4 and tail < 128:
            expected[row, tail // 32] |= np.uint32(1) << np.uint32(tail % 32)
    actual = jax.jit(
        lambda ids, live: tc._selection_bitset(ids, live, ratio=4, groups=128, words=4)
    )(jnp.asarray(blocks), jnp.asarray(lengths))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected)


def test_integrated_sort_preserves_all_word_bits_and_deduplicates():
    blocks = np.tile(np.arange(64, dtype=np.int32) % 32, (3, 1))
    lengths = np.array([128, 129, 256], np.int32)
    actual = tc._selection_bitset(
        jnp.asarray(blocks), jnp.asarray(lengths), ratio=4, groups=64, words=2
    )
    expected = np.array([[0xFFFFFFFF, 0], [0xFFFFFFFF, 1], [0xFFFFFFFF, 0]], np.uint32)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), expected)
