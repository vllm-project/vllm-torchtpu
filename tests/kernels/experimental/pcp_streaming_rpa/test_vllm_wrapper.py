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

import importlib.util
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import kernel as pcp_kernel
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    stage_plan as pcp_stage_plan,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as pcp_wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)

pytestmark = pytest.mark.multichip


def _build_expected_pcp_writeback_mapping(
    q_lens: np.ndarray,
    seq_lens: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    local_kv_cache_num_blocks: int,
):
    """Independent NumPy reference for dense ring-capture destinations."""
    q_lens = np.asarray(q_lens, dtype=np.int64)
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int64) % int(
        local_kv_cache_num_blocks
    )
    owner_starts = np.zeros(q_lens.shape, dtype=np.int64)
    if q_lens.size > 1:
        np.cumsum(q_lens[:-1], out=owner_starts[1:])
    token_order, _ = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )

    cache_owners = np.full(padded_num_tokens, -1, dtype=np.int32)
    cache_slots = np.full(padded_num_tokens, -1, dtype=np.int32)
    capture_offsets = np.full(padded_num_tokens, -1, dtype=np.int32)
    writeback_counts = np.zeros(pcp_size, dtype=np.int32)
    compact_slots = np.full((pcp_size, padded_num_tokens), -1, dtype=np.int32)

    req_ends = np.cumsum(q_lens)
    absolute_starts = seq_lens - q_lens
    virtual_block_size = block_size * pcp_size
    for source_row, request_major_row in enumerate(token_order):
        if request_major_row < 0:
            continue
        req_id = int(np.searchsorted(req_ends, request_major_row, side="right"))
        req_start = 0 if req_id == 0 else int(req_ends[req_id - 1])
        absolute_position = (
            int(absolute_starts[req_id]) + int(request_major_row) - req_start
        )
        block_idx = absolute_position // virtual_block_size
        virtual_offset = absolute_position % virtual_block_size
        owner = (virtual_offset // interleave_size) % pcp_size
        local_offset = (
            virtual_offset // (pcp_size * interleave_size)
        ) * interleave_size + virtual_offset % interleave_size
        slot = int(block_tables[req_id, block_idx] * block_size + local_offset)
        cache_owners[source_row] = owner
        cache_slots[source_row] = slot

    for req_id in range(q_lens.size):
        req_rows = np.flatnonzero(
            (token_order >= (0 if req_id == 0 else req_ends[req_id - 1]))
            & (token_order < req_ends[req_id])
        )
        req_rows = req_rows[
            np.argsort(
                np.asarray(
                    [
                        absolute_starts[req_id]
                        + token_order[row]
                        - (0 if req_id == 0 else req_ends[req_id - 1])
                        for row in req_rows
                    ]
                )
            )
        ]
        for source_row in req_rows:
            owner = int(cache_owners[source_row])
            capture_offset = int(writeback_counts[owner])
            writeback_counts[owner] += 1
            capture_offsets[source_row] = capture_offset
            compact_slots[owner, capture_offset] = cache_slots[source_row]

    return SimpleNamespace(
        cache_owner_ranks=cache_owners,
        cache_local_slot_ids=cache_slots,
        capture_offsets=capture_offsets,
        writeback_counts=writeback_counts,
        compact_slot_ids=compact_slots,
    )


def test_sharded_wrapper_uses_real_metadata_streaming_entry():
    source = inspect.getsource(pcp_wrapper)

    assert "pcp_streaming_attention_page_groups_packed_local_from_metadata" in source


@pytest.mark.parametrize(
    ("kv_dtype", "kv_heads", "q_per_kv", "packed_kv_groups", "k_scale", "v_scale"),
    [
        (jnp.bfloat16, 1, 1, 1, None, None),
        (jnp.bfloat16, 2, 1, 2, None, None),
        (jnp.bfloat16, 2, 2, 2, None, None),
        (jnp.float8_e4m3fn, 1, 2, 1, 0.5, 0.5),
        (jnp.float8_e4m3fn, 2, 1, 1, 0.5, 0.5),
        (jnp.float8_e4m3fn, 3, 2, 2, 0.5, 0.5),
    ],
)
def test_packed_kv_layout_validation_accepts_unified_head_matrix(
    kv_dtype, kv_heads, q_per_kv, packed_kv_groups, k_scale, v_scale
):
    local_tokens = 32
    page_size = 16
    head_dim = 128
    kv_packing = pcp_wrapper.get_dtype_packing(kv_dtype)
    q_local = jnp.zeros(
        (local_tokens, kv_heads, q_per_kv, head_dim), dtype=jnp.bfloat16
    )
    kv_cache_local = jnp.zeros(
        (2, page_size, packed_kv_groups, kv_packing, head_dim),
        dtype=kv_dtype,
    )

    assert pcp_kernel._validate_packed_kv_layout(
        q_local, kv_cache_local, k_scale=k_scale, v_scale=v_scale
    ) == (
        kv_heads,
        packed_kv_groups,
        kv_packing,
    )


def test_packed_kv_layout_validation_rejects_fp8_without_scales():
    q_local = jnp.zeros((32, 1, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((2, 16, 1, 4, 128), dtype=jnp.float8_e4m3fn)

    with pytest.raises(ValueError, match="FP8 KV cache requires"):
        pcp_kernel._validate_packed_kv_layout(q_local, kv_cache_local)


def test_packed_kv_layout_validation_rejects_wrong_physical_group_count():
    q_local = jnp.zeros((32, 3, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((2, 16, 1, 4, 128), dtype=jnp.float8_e4m3fn)

    with pytest.raises(ValueError, match="packed layout does not match"):
        pcp_kernel._validate_packed_kv_layout(
            q_local, kv_cache_local, k_scale=0.5, v_scale=0.5
        )


def test_seq_along_lane_layout_validation_accepts_fp8_multi_head():
    q_local = jnp.zeros((32, 2, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((2, 4, 32, 4, 128), dtype=jnp.float8_e4m3fn)

    assert pcp_kernel._validate_packed_kv_layout(
        q_local,
        kv_cache_local,
        kv_layout=pcp_wrapper.KVLayout.SEQ_ALONG_LANE,
        k_scale=0.5,
        v_scale=0.5,
    ) == (2, 4, 4)


def test_prepare_packed_kv_accepts_equivalent_tail_layout(monkeypatch):
    packed_kv = jnp.arange(3 * 8 * 2 * 8, dtype=jnp.float32).reshape(3, 8, 2, 8)

    monkeypatch.setattr(
        pcp_wrapper, "_pack_kv_for_cache", lambda *_args, **_kwargs: packed_kv
    )

    kv_cache = jnp.zeros((2, 4, 16, 1, 8), dtype=jnp.float32)
    k = jnp.zeros((3, 8, 8), dtype=jnp.float32)
    v = jnp.zeros_like(k)
    prepared = pcp_wrapper._prepare_packed_kv_for_cache(k, v, kv_cache)
    expected = packed_kv.reshape(3, 16, 1, 8)
    np.testing.assert_array_equal(np.asarray(prepared), np.asarray(expected))


def test_seq_along_lane_kv_packing_pads_inside_pcp_wrapper():
    k = jnp.ones((384, 1, 128), dtype=jnp.bfloat16)
    padded = pcp_wrapper._pack_kv_for_cache(
        k,
        jnp.ones_like(k),
        jnp.float8_e4m3fn,
        kv_layout=pcp_wrapper.KVLayout.SEQ_ALONG_LANE,
        page_size=256,
    )

    assert padded.shape == (2, 32, 4, 512)
    np.testing.assert_array_equal(np.asarray(padded[..., :384]), 1)
    np.testing.assert_array_equal(np.asarray(padded[..., 384:]), 0)


def test_captured_kv_writeback_coalesces_contiguous_cache_slots():
    compact_slots = jnp.asarray([0, 1, 2, 8, 9, 20, -1, -1], dtype=jnp.int32)
    descriptors, num_segments = pcp_wrapper._build_writeback_segment_descriptors(
        compact_slots,
        jnp.asarray(6),
    )

    assert int(num_segments) == 3
    np.testing.assert_array_equal(
        np.asarray(descriptors),
        np.asarray(
            [
                [0, 3, 5, 0, 0, 0, 0, 0],
                [0, 8, 20, 0, 0, 0, 0, 0],
                [3, 2, 1, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_captured_kv_writeback_preserves_fully_fragmented_capacity():
    compact_slots = jnp.arange(16, dtype=jnp.int32) * 2
    descriptors, num_segments = pcp_wrapper._build_writeback_segment_descriptors(
        compact_slots,
        jnp.asarray(compact_slots.shape[0]),
    )

    assert int(num_segments) == compact_slots.shape[0]
    np.testing.assert_array_equal(
        np.asarray(descriptors[0]), np.arange(16, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(descriptors[1]), np.arange(16, dtype=np.int32) * 2
    )
    np.testing.assert_array_equal(
        np.asarray(descriptors[2]), np.ones(16, dtype=np.int32)
    )


def test_seq_along_lane_writeback_segments_stop_at_page_boundaries():
    compact_slots = jnp.arange(256, dtype=jnp.int32)
    descriptors, num_segments = pcp_wrapper._build_writeback_segment_descriptors(
        compact_slots,
        jnp.asarray(256),
        kv_layout=pcp_wrapper.KVLayout.SEQ_ALONG_LANE,
        page_size=128,
    )

    assert int(num_segments) == 2
    np.testing.assert_array_equal(
        np.asarray(descriptors[:, :2]),
        np.asarray([[0, 128], [0, 128], [128, 128]], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(descriptors[:, 2:]), 0)


def test_metadata_reconstruction_preserves_query_row_coordinates():
    kv_lens = jnp.asarray([8], dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, 5], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    mapping = pcp_wrapper._compute_pcp_local_query_row_mapping_from_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        local_padded_tokens=4,
        pcp_size=2,
        pcp_rank=0,
        interleave_size=2,
    )

    np.testing.assert_array_equal(
        np.asarray(mapping.absolute_positions),
        np.array([3, 4, 7, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(mapping.is_valid),
        np.array([True, True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(mapping.request_indices),
        np.array([0, 0, 0, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(mapping.source_ranks),
        np.zeros(4, dtype=np.int32),
    )


def test_query_row_mapping_separates_owner_and_request_absolute_starts():
    common_kwargs = dict(
        q_lens=jnp.asarray([3, 4], dtype=jnp.int32),
        token_owner_starts=jnp.asarray([0, 3], dtype=jnp.int32),
        request_absolute_starts=jnp.asarray([5, 100], dtype=jnp.int32),
        local_padded_tokens=4,
        pcp_size=2,
        interleave_size=2,
    )

    rank_0 = pcp_wrapper._compute_pcp_local_query_row_mapping(
        pcp_rank=0, **common_kwargs
    )
    rank_1 = pcp_wrapper._compute_pcp_local_query_row_mapping(
        pcp_rank=1, **common_kwargs
    )

    np.testing.assert_array_equal(
        np.asarray(rank_0.absolute_positions), np.array([5, 6, 101, 102])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_0.request_indices), np.array([0, 0, 1, 1])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_1.absolute_positions), np.array([7, 100, 103, -1])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_1.request_indices), np.array([0, 1, 1, -1])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_1.is_valid), np.array([True, True, True, False])
    )
    # Source-rank identity remains available even when the row itself is pad.
    np.testing.assert_array_equal(
        np.asarray(rank_1.source_ranks), np.ones(4, dtype=np.int32)
    )


def test_kv_writeback_mapping_matches_independent_dense_capture_reference():
    pcp_size = 4
    page_size = 4
    interleave_size = 2
    local_padded_tokens = 16
    local_kv_cache_num_blocks = 16
    q_lens = np.array([5, 17, 6], dtype=np.int32)
    absolute_starts = np.array([3, 9, 0], dtype=np.int32)
    seq_lens = absolute_starts + q_lens
    cu_q_lens = np.concatenate(([0], np.cumsum(q_lens))).astype(np.int32)
    distribution = np.array([0, 0, q_lens.size], dtype=np.int32)
    block_tables = (
        np.arange(q_lens.size * 16, dtype=np.int32).reshape(q_lens.size, 16) + 7
    )

    query_rows = pcp_wrapper._compute_pcp_rank_major_query_row_mapping_from_metadata(
        jnp.asarray(seq_lens),
        jnp.asarray(cu_q_lens),
        jnp.asarray(distribution),
        local_padded_tokens=local_padded_tokens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    actual = pcp_wrapper._compute_pcp_kv_writeback_mapping(
        query_rows,
        jnp.asarray(block_tables),
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    jitted_actual = pcp_wrapper.jax.jit(
        lambda rows, tables: pcp_wrapper._compute_pcp_kv_writeback_mapping(
            rows,
            tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
    )(query_rows, jnp.asarray(block_tables))
    expected = _build_expected_pcp_writeback_mapping(
        q_lens,
        seq_lens,
        block_tables,
        page_size,
        pcp_size,
        interleave_size,
        local_padded_tokens * pcp_size,
        local_kv_cache_num_blocks,
    )

    for field in pcp_wrapper._PCPKVWritebackMapping._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, field)), getattr(expected, field)
        )
        np.testing.assert_array_equal(
            np.asarray(getattr(jitted_actual, field)), getattr(expected, field)
        )
    for cache_rank in range(pcp_size):
        compact_slots = pcp_wrapper._compact_writeback_slot_ids_for_cache_rank(
            actual, cache_rank
        )
        np.testing.assert_array_equal(
            np.asarray(compact_slots), expected.compact_slot_ids[cache_rank]
        )


def test_one_token_writeback_reaches_cache_owner_without_local_q():
    pcp_size = 8
    page_size = interleave_size = 256
    local_padded_tokens = 4
    local_kv_cache_num_blocks = 32
    block_tables = jnp.asarray([[7]], dtype=jnp.int32)
    query_rows = pcp_wrapper._compute_pcp_rank_major_query_row_mapping_from_metadata(
        kv_lens=jnp.asarray([301], dtype=jnp.int32),
        cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        local_padded_tokens=local_padded_tokens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    mapping = pcp_wrapper._compute_pcp_kv_writeback_mapping(
        query_rows,
        block_tables,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )

    valid_row = int(np.flatnonzero(np.asarray(query_rows.is_valid))[0])
    assert int(query_rows.source_ranks[valid_row]) == 0
    assert int(mapping.cache_owner_ranks[valid_row]) == 1
    assert int(mapping.capture_offsets[valid_row]) == 0
    np.testing.assert_array_equal(
        np.asarray(mapping.writeback_counts),
        np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    )
    rank_1_slots = pcp_wrapper._compact_writeback_slot_ids_for_cache_rank(mapping, 1)
    assert int(rank_1_slots[0]) == 7 * page_size + 44
    assert np.all(np.asarray(rank_1_slots[1:]) == -1)


def test_batch_flat_metadata_32x256_balances_rows_with_cache_owner_skew():
    pcp_size = 8
    interleave_size = page_size = 256
    num_reqs = 32
    q_len = 256
    local_padded_tokens = 1280
    local_kv_cache_num_blocks = 32
    q_lens = jnp.full((num_reqs,), q_len, dtype=jnp.int32)
    token_owner_starts = jnp.arange(num_reqs, dtype=jnp.int32) * q_len
    request_absolute_starts = jnp.zeros((num_reqs,), dtype=jnp.int32)
    page_numbers = np.asarray(
        [
            19,
            3,
            27,
            11,
            0,
            24,
            8,
            30,
            14,
            6,
            22,
            1,
            17,
            29,
            10,
            25,
            5,
            21,
            13,
            31,
            2,
            18,
            28,
            9,
            23,
            7,
            16,
            4,
            26,
            12,
            20,
            15,
        ],
        dtype=np.int32,
    )
    block_tables = jnp.asarray(page_numbers[:, None])
    cu_q_lens = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_lens, dtype=jnp.int32))
    )
    distribution = jnp.asarray([0, 0, num_reqs], dtype=jnp.int32)

    mappings = [
        pcp_wrapper._compute_pcp_local_query_row_mapping(
            q_lens,
            token_owner_starts,
            request_absolute_starts,
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
        for pcp_rank in range(pcp_size)
    ]
    metadata_mappings = [
        pcp_wrapper._compute_pcp_local_query_row_mapping_from_metadata(
            q_lens,
            cu_q_lens,
            distribution,
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
        for pcp_rank in range(pcp_size)
    ]
    for expected, actual in zip(mappings, metadata_mappings):
        for field in pcp_wrapper._PCPQueryRowMapping._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(actual, field)), np.asarray(getattr(expected, field))
            )
    rank_major_query_rows = pcp_wrapper._PCPQueryRowMapping(
        *(
            jnp.concatenate([getattr(mapping, field) for mapping in mappings])
            for field in pcp_wrapper._PCPQueryRowMapping._fields
        )
    )
    writeback_mapping = pcp_wrapper._compute_pcp_kv_writeback_mapping(
        rank_major_query_rows,
        block_tables,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )

    for source_rank, mapping in enumerate(mappings):
        assert int(np.asarray(mapping.is_valid).sum()) == 4 * q_len
        np.testing.assert_array_equal(
            np.asarray(mapping.source_ranks),
            np.full(local_padded_tokens, source_rank, dtype=np.int32),
        )
        assert np.all(np.asarray(mapping.request_indices)[4 * q_len :] == -1)

    cache_owner_ranks = np.asarray(writeback_mapping.cache_owner_ranks)
    cache_local_slot_ids = np.asarray(writeback_mapping.cache_local_slot_ids)
    cache_rank_0_slots = np.where(cache_owner_ranks == 0, cache_local_slot_ids, -1)
    cache_rank_1_slots = np.where(cache_owner_ranks == 1, cache_local_slot_ids, -1)
    assert np.count_nonzero(cache_rank_0_slots >= 0) == num_reqs * q_len
    assert np.all(cache_rank_1_slots < 0)
    np.testing.assert_array_equal(
        np.asarray(writeback_mapping.writeback_counts),
        np.array([num_reqs * q_len, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    )
    capture_offsets = np.asarray(writeback_mapping.capture_offsets)
    assert np.all(capture_offsets[~np.asarray(rank_major_query_rows.is_valid)] == -1)
    np.testing.assert_array_equal(
        np.sort(capture_offsets[capture_offsets >= 0]),
        np.arange(num_reqs * q_len, dtype=np.int32),
    )
    cache_rank_0_compact_slots = pcp_wrapper._compact_writeback_slot_ids_for_cache_rank(
        writeback_mapping, 0
    )
    np.testing.assert_array_equal(
        np.asarray(cache_rank_0_compact_slots)[: num_reqs * q_len].reshape(
            num_reqs, q_len
        )[:, 0],
        page_numbers * page_size,
    )
    valid = np.asarray(rank_major_query_rows.is_valid)
    request_indices = np.asarray(rank_major_query_rows.request_indices)[valid]
    absolute_positions = np.asarray(rank_major_query_rows.absolute_positions)[valid]
    np.testing.assert_array_equal(
        cache_rank_0_slots[valid],
        page_numbers[request_indices] * page_size + absolute_positions,
    )


def test_sharded_wrapper_generates_slot_ids_when_metadata_omits_them(monkeypatch):
    pcp_size = 4
    pcp_rank = 2
    interleave_size = page_size = 4
    local_padded_tokens = 16
    target_num_reqs = 4
    local_kv_cache_num_blocks = 8
    q_lens = np.array([5, 17, 6], dtype=np.int32)
    q_starts = np.array([3, 9, 0], dtype=np.int32)
    seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
    seq_lens[: q_lens.size] = q_starts + q_lens
    cu_q_lens = np.zeros(target_num_reqs + 1, dtype=np.int32)
    cu_q_lens[1 : q_lens.size + 1] = np.cumsum(q_lens)
    cu_q_lens[q_lens.size + 1 :] = cu_q_lens[q_lens.size]
    distribution = np.array([0, 0, q_lens.size], dtype=np.int32)
    block_tables = (
        np.arange(target_num_reqs * 8, dtype=np.int32).reshape(target_num_reqs, 8) + 11
    )
    expected_writeback = _build_expected_pcp_writeback_mapping(
        q_lens,
        q_starts + q_lens,
        block_tables,
        page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=local_padded_tokens * pcp_size,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
    )
    expected = expected_writeback.compact_slot_ids[pcp_rank]

    captured = {}

    def fake_streaming(
        q_streaming,
        _current_kv,
        kv_cache,
        _kv_lens,
        _cu_q_lens,
        _distribution,
        segment_descriptors,
        num_segments,
        *,
        stage_plan,
        **_kwargs,
    ):
        captured["segment_descriptors"] = segment_descriptors
        captured["num_segments"] = num_segments
        captured["is_valid"] = stage_plan.query_row_valid
        return jnp.full_like(q_streaming, 5), kv_cache

    def fake_prepare_current(k_local, _v_local, cache, **_kwargs):
        return jnp.zeros((k_local.shape[0], *cache.shape[2:]), dtype=cache.dtype)

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        del mesh, in_specs, out_specs, check_vma

        def wrapped(*args):
            args = list(args)
            token_start = pcp_rank * local_padded_tokens
            token_end = token_start + local_padded_tokens
            block_start = pcp_rank * local_kv_cache_num_blocks
            block_end = block_start + local_kv_cache_num_blocks
            args[0] = args[0][token_start:token_end]
            args[1] = args[1][token_start:token_end]
            args[2] = args[2][token_start:token_end]
            args[3] = args[3][block_start:block_end]
            return fn(*args)

        return wrapped

    monkeypatch.setattr(
        pcp_wrapper, "_prepare_packed_kv_for_cache", fake_prepare_current
    )
    monkeypatch.setattr(
        pcp_wrapper,
        "pcp_streaming_attention_page_groups_packed_local_from_metadata",
        fake_streaming,
    )
    monkeypatch.setattr(
        pcp_wrapper._stage_plan_mod,
        "get_pcp_rank_without_axis_index",
        lambda _axis, _size: jnp.asarray(pcp_rank, dtype=jnp.int32),
    )
    monkeypatch.setattr(pcp_wrapper.jax, "shard_map", fake_shard_map)

    mesh = SimpleNamespace(axis_names=("pcp",), shape={"pcp": pcp_size})
    global_padded_tokens = local_padded_tokens * pcp_size
    q = jnp.ones((global_padded_tokens, 2, 128), dtype=jnp.float32)
    k = jnp.ones((global_padded_tokens, 1, 128), dtype=jnp.float32)
    v = jnp.ones((global_padded_tokens, 1, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros(
        (local_kv_cache_num_blocks * pcp_size, page_size, 2, 1, 128), dtype=jnp.float32
    )

    output, new_cache = pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(seq_lens),
        page_indices=jnp.asarray(block_tables.reshape(-1)),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution),
        attention_sink=None,
        sm_scale=0.25,
        cp_kv_cache_interleave_size=interleave_size,
    )

    expected_descriptors, expected_num_segments = (
        pcp_wrapper._build_writeback_segment_descriptors(
            jnp.asarray(expected),
            jnp.asarray(expected_writeback.writeback_counts[pcp_rank]),
            max_segments=local_padded_tokens,
        )
    )
    np.testing.assert_array_equal(
        np.asarray(captured["segment_descriptors"]),
        np.asarray(expected_descriptors),
    )
    assert int(captured["num_segments"]) == int(expected_num_segments)
    assert np.any(expected < 0)
    expected_output = np.zeros((local_padded_tokens, 2, 128), dtype=np.float32)
    expected_output[np.asarray(captured["is_valid"])] = 5
    np.testing.assert_array_equal(
        np.asarray(output),
        expected_output,
    )
    assert new_cache.shape == (local_kv_cache_num_blocks, page_size, 2, 1, 128)


def test_sharded_wrapper_masks_output_with_query_validity(monkeypatch):
    query_rows = pcp_wrapper._PCPQueryRowMapping(
        absolute_positions=jnp.array([0, -1, 2, -1], dtype=jnp.int32),
        request_indices=jnp.array([0, -1, 0, -1], dtype=jnp.int32),
        source_ranks=jnp.zeros(4, dtype=jnp.int32),
        is_valid=jnp.array([True, False, True, False]),
    )
    captured = {}

    def fake_compute_query_rows(*_args, **_kwargs):
        return query_rows

    def fake_streaming(
        q_streaming,
        _current_kv,
        kv_cache,
        _kv_lens,
        _cu_q_lens,
        _distribution,
        _segment_descriptors,
        _num_segments,
        **_kwargs,
    ):
        captured["writeback_called"] = True
        return jnp.full_like(q_streaming, 5), kv_cache

    def fake_prepare_current(k_local, _v_local, cache, **_kwargs):
        return jnp.zeros((k_local.shape[0], *cache.shape[2:]), dtype=cache.dtype)

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        del mesh, in_specs, out_specs, check_vma
        return fn

    monkeypatch.setattr(
        pcp_wrapper._stage_plan_mod,
        "get_pcp_rank_without_axis_index",
        lambda _axis, _size: jnp.asarray(0, dtype=jnp.int32),
    )
    monkeypatch.setattr(
        pcp_wrapper._stage_plan_mod,
        "_compute_pcp_local_query_row_mapping",
        fake_compute_query_rows,
    )
    monkeypatch.setattr(
        pcp_wrapper, "_prepare_packed_kv_for_cache", fake_prepare_current
    )
    monkeypatch.setattr(
        pcp_wrapper,
        "pcp_streaming_attention_page_groups_packed_local_from_metadata",
        fake_streaming,
    )
    monkeypatch.setattr(pcp_wrapper.jax, "shard_map", fake_shard_map)

    num_tokens = 4
    page_size = 4
    mesh = SimpleNamespace(axis_names=("pcp",), shape={"pcp": 1})
    q = jnp.ones((num_tokens, 2, 128), dtype=jnp.float32)
    k = jnp.ones((num_tokens, 1, 128), dtype=jnp.float32)
    v = jnp.ones((num_tokens, 1, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros((1, page_size, 2, 1, 128), dtype=jnp.float32)

    output, _ = pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray([num_tokens]),
        page_indices=jnp.asarray([0]),
        cu_q_lens=jnp.asarray([0, num_tokens]),
        distribution=jnp.asarray([0, 0, 1]),
        attention_sink=None,
        sm_scale=0.25,
        cp_kv_cache_interleave_size=page_size,
    )

    assert captured["writeback_called"]
    expected_output = np.zeros((num_tokens, 2, 128), dtype=np.float32)
    expected_output[[0, 2]] = 5
    np.testing.assert_array_equal(np.asarray(output), expected_output)


def test_reshape_packed_kv_cache_for_attention_rejects_ambiguous_layout():
    kv_cache = jnp.arange(2 * 4 * 16 * 1 * 8, dtype=jnp.bfloat16).reshape(
        2, 4, 16, 1, 8
    )

    with pytest.raises(ValueError, match="packing does not match"):
        pcp_wrapper._reshape_packed_kv_cache_for_attention(kv_cache)


@pytest.mark.parametrize(
    ("q_per_kv", "expected_stacked"),
    [(2, 2), (6, 2), (8, 8), (16, 16)],
)
def test_metadata_entry_caps_stacked_q_heads_at_q_per_kv(
    monkeypatch, q_per_kv, expected_stacked
):
    captured = {"num_stacked_q_heads": 0}

    def fake_attend(
        _kv_cache_local, q_scaled_hm, _stage_plan, *, splash_cfg, **_kwargs
    ):
        captured["num_stacked_q_heads"] = splash_cfg.num_stacked_q_heads
        return jnp.zeros_like(q_scaled_hm, dtype=jnp.float32)

    monkeypatch.setattr(
        pcp_kernel,
        "_route_current_kv_to_captured_kv",
        lambda kv_in, *_args, **_kwargs: kv_in,
    )
    monkeypatch.setattr(
        pcp_kernel,
        "write_captured_kv_to_local_cache",
        lambda cache, *_args, **_kwargs: cache,
    )
    monkeypatch.setattr(pcp_kernel, "_attend_over_vpage_groups", fake_attend)

    q_local = jnp.zeros((128, 2, q_per_kv, 128), dtype=jnp.bfloat16)
    current_kv_local = jnp.zeros((128, 2, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((4, 128, 2, 2, 128), dtype=jnp.bfloat16)
    stage_plan = SimpleNamespace(
        needs_reroute=jnp.asarray(False),
        q_seg_ids=jnp.zeros((128,), dtype=jnp.int32),
        block_tables=None,
    )

    pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
        q_local,
        current_kv_local,
        kv_cache_local,
        jnp.asarray([128], dtype=jnp.int32),
        jnp.asarray([0, 128], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        jnp.zeros((3, 8), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        stage_plan=stage_plan,
        vpage_group_size=128,
        pcp_size=8,
        interleave_size=128,
    )

    assert captured["num_stacked_q_heads"] == expected_stacked
    assert q_per_kv % captured["num_stacked_q_heads"] == 0


@pytest.mark.parametrize(
    ("q_lens", "hist_lens"),
    [
        ([192], [301]),
        ([95, 95], [0, 0]),
        ([96, 96], [0, 0]),
        ([97, 97], [0, 0]),
        ([127, 127], [0, 0]),
    ],
)
def test_capture_source_rows_follow_token_cache_owner(q_lens, hist_lens):
    pcp_size, interleave_size, local_tokens, max_num_seqs = 2, 32, 128, 2
    pad = max_num_seqs - len(q_lens)
    cu_q_lens = jnp.asarray(
        np.concatenate([[0], np.cumsum(q_lens + [0] * pad)]), dtype=jnp.int32
    )
    kv_lens = jnp.asarray(
        [h + q for h, q in zip(hist_lens, q_lens)] + [0] * pad, dtype=jnp.int32
    )
    distribution = jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32)
    rows = pcp_stage_plan._compute_pcp_rank_major_query_row_mapping_from_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        local_padded_tokens=local_tokens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    positions = np.asarray(rows.absolute_positions)
    requests = np.asarray(rows.request_indices)
    seen = []
    for rank in range(pcp_size):
        src, valid = pcp_kernel._compute_pcp_capture_source_rows_for_cache_rank(
            kv_lens,
            cu_q_lens,
            distribution,
            rank,
            local_tokens=local_tokens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
        src = np.asarray(src)[np.asarray(valid)]
        expected = [
            (r, p)
            for r, (h, q) in enumerate(zip(hist_lens, q_lens))
            for p in range(h, h + q)
            if (p // interleave_size) % pcp_size == rank
        ]
        actual = [(int(r), int(p)) for r, p in zip(requests[src], positions[src])]
        assert actual == expected
        seen.extend(src.tolist())
    assert len(seen) == len(set(seen)) == sum(q_lens)


def test_gather_vpage_group_zeroes_unwritten_fp8_slots(monkeypatch):
    nan = jnp.asarray(0x7F, dtype=jnp.uint8).view(jnp.float8_e4m3fn)
    tokens = jnp.arange(16)[None, :, None]
    raw = jnp.broadcast_to(
        jnp.where(tokens < 5, jnp.asarray(1.5, dtype=jnp.float8_e4m3fn), nan),
        (1, 16, 128),
    )
    monkeypatch.setattr(
        pcp_kernel, "_gather_vpages_to_head_major", lambda *_a, **_k: (raw, raw)
    )
    plan = SimpleNamespace(
        sorted_block_ids=jnp.asarray([0, 1], dtype=jnp.int32),
        sorted_seg_ids=jnp.asarray([0, -1], dtype=jnp.int32),
        sorted_vpage_valid_tokens=jnp.asarray([5, 0], dtype=jnp.int32),
    )
    k_hm, v_hm, kv_seg_ids = pcp_kernel._gather_vpage_group_kv_and_seg(
        jnp.zeros((2, 4, 1, 4, 128), dtype=jnp.float8_e4m3fn),
        plan,
        0,
        2,
        kv_heads=1,
        head_dim=128,
        page_size=4,
        pcp_size=2,
        interleave_size=2,
        kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
        pcp_axis_name="pcp",
    )
    assert not bool(jnp.isnan(k_hm.astype(jnp.float32)).any())
    assert not bool(jnp.isnan(v_hm.astype(jnp.float32)).any())
    assert bool((k_hm[:, :5].astype(jnp.float32) == 1.5).all())
    assert int((kv_seg_ids == 0).sum()) == 5


def test_merge_attention_partials_matches_full_softmax():
    rng = np.random.default_rng(2026)
    heads, q_tokens, s1, s2, head_dim = 4, 32, 96, 64, 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = rng.normal(size=(heads, q_tokens, head_dim)).astype(np.float64)
    k = rng.normal(size=(heads, s1 + s2, head_dim)).astype(np.float64)
    v = rng.normal(size=(heads, s1 + s2, head_dim)).astype(np.float64)

    def exact_part(k_slice, v_slice):
        scores = np.einsum("htd,hsd->hts", q, k_slice) * sm_scale
        m = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - m)
        denom = exp_s.sum(axis=-1, keepdims=True)
        out = np.einsum("hts,hsd->htd", exp_s / denom, v_slice)
        lse = m[..., 0] + np.log(denom[..., 0])
        return out.astype(np.float32), lse.astype(np.float32)

    o1, lse1 = exact_part(k[:, :s1], v[:, :s1])
    o2, lse2 = exact_part(k[:, s1:], v[:, s1:])
    o_full, lse_full = exact_part(k, v)

    o_merged, lse_merged = pcp_kernel._merge_attention_partials(
        jnp.asarray(o1),
        jnp.asarray(lse1),
        jnp.asarray(o2),
        jnp.asarray(lse2),
    )
    o_merged_np = np.asarray(o_merged, dtype=np.float64)
    lse_merged_np = np.asarray(lse_merged, dtype=np.float64)

    u_f32 = 2.0**-24
    max_abs_v = float(np.max(np.abs(v)))
    max_abs_lse = float(np.max(np.abs(lse_full)))
    # Omitted f32 terms: input cast (1 * u_f32), two exp weights + sum/inv
    # (4 * u_f32), and two weighted FMA products (3 * u_f32).
    tol_o = (1.0 + 4.0 + 3.0) * u_f32 * max_abs_v
    tol_lse = (1.0 + 4.0 + 3.0) * u_f32 * max(1.0, max_abs_lse)

    assert float(np.max(np.abs(o_merged_np - o_full))) <= tol_o
    assert float(np.max(np.abs(lse_merged_np - lse_full))) <= tol_lse
    assert float(np.max(np.abs(o1.astype(np.float64) - o_full))) > 100.0 * tol_o


def test_merge_attention_partials_ignores_empty_splash_group():
    rng = np.random.default_rng(7)
    o_valid = jnp.asarray(rng.normal(size=(2, 16, 64)).astype(np.float32))
    lse_valid = jnp.asarray(rng.normal(size=(2, 16)).astype(np.float32))
    o_zero = jnp.zeros_like(o_valid)
    sentinel = float(
        pcp_kernel._splash_kernel.DEFAULT_MASK_VALUE / pcp_kernel._splash_kernel.LOG2E
    )
    lse_sentinel = jnp.full_like(lse_valid, sentinel)

    o_right, lse_right = pcp_kernel._merge_attention_partials(
        o_valid, lse_valid, o_zero, lse_sentinel
    )
    np.testing.assert_array_equal(np.asarray(o_right), np.asarray(o_valid))
    np.testing.assert_array_equal(np.asarray(lse_right), np.asarray(lse_valid))

    o_left, lse_left = pcp_kernel._merge_attention_partials(
        o_zero, lse_sentinel, o_valid, lse_valid
    )
    np.testing.assert_array_equal(np.asarray(o_left), np.asarray(o_valid))
    np.testing.assert_array_equal(np.asarray(lse_left), np.asarray(lse_valid))

    lse_neginf = jnp.full_like(lse_valid, -jnp.inf)
    o_empty, lse_empty = pcp_kernel._merge_attention_partials(
        o_zero, lse_neginf, o_zero, lse_neginf
    )
    assert not bool(jnp.isnan(o_empty).any())
    assert not bool(jnp.isnan(lse_empty).any())
    np.testing.assert_array_equal(
        np.asarray(o_empty), np.zeros_like(np.asarray(o_valid))
    )
    assert bool(jnp.isneginf(lse_empty).all())


def test_attend_over_vpage_groups_routes_rungs_and_multi_group(monkeypatch):
    calls = []

    def fake_single(_kv, q_scaled_hm, _plan, group_size, **_kwargs):
        calls.append(("single", int(group_size)))
        return jnp.full_like(q_scaled_hm, float(group_size), dtype=jnp.float32)

    def fake_multi(_kv, q_scaled_hm, _plan, group_size, **_kwargs):
        calls.append(("multi", int(group_size)))
        return jnp.full_like(q_scaled_hm, -float(group_size), dtype=jnp.float32)

    monkeypatch.setattr(pcp_kernel, "_attend_single_vpage_group", fake_single)
    monkeypatch.setattr(pcp_kernel, "_attend_multi_vpage_groups", fake_multi)

    q_scaled_hm = jnp.zeros((2, 128, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((8, 128, 1, 2, 128), dtype=jnp.bfloat16)
    splash_cfg = pcp_kernel._splash_kernel.SplashConfig(
        block_q=128,
        block_kv=512,
        block_kv_compute=256,
        num_stacked_q_heads=2,
        use_base2_exp=True,
        q_prescaled_log2e=True,
    )
    common_kwargs = dict(
        splash_cfg=splash_cfg,
        kv_heads=1,
        head_dim=128,
        page_size=128,
        pcp_size=2,
        interleave_size=32,
        kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
        pcp_axis_name="pcp",
    )

    def make_plan(total_vpages: int, num_active_vpages: int):
        return SimpleNamespace(
            sorted_block_ids=jnp.zeros((total_vpages,), dtype=jnp.int32),
            num_active_vpages=jnp.asarray(num_active_vpages, dtype=jnp.int32),
        )

    for active_vpages, expected_rung in [
        (1, 32.0),
        (32, 32.0),
        (33, 64.0),
        (128, 128.0),
        (257, 512.0),
        (512, 512.0),
    ]:
        out = pcp_kernel._attend_over_vpage_groups(
            kv_cache_local,
            q_scaled_hm,
            make_plan(512, active_vpages),
            vpage_group_size=128,
            **common_kwargs,
        )
        assert float(out[0, 0, 0]) == expected_rung

    out_513 = pcp_kernel._attend_over_vpage_groups(
        kv_cache_local,
        q_scaled_hm,
        make_plan(640, 513),
        vpage_group_size=128,
        **common_kwargs,
    )
    assert float(out_513[0, 0, 0]) == -128.0

    calls.clear()
    out_g2 = pcp_kernel._attend_over_vpage_groups(
        kv_cache_local,
        q_scaled_hm,
        make_plan(6, 6),
        vpage_group_size=2,
        **common_kwargs,
    )
    assert float(out_g2[0, 0, 0]) == -2.0
    assert calls == [("multi", 2)]


def test_attend_multi_vpage_groups_streams_causal_groups_and_sentinels(monkeypatch):
    rng = np.random.default_rng(2027)
    kv_heads, q_per_kv, head_dim = 1, 2, 128
    q_heads = kv_heads * q_per_kv
    q_tokens = 128
    page_size, pcp_size = 128, 2
    virtual_page_size = page_size * pcp_size
    group_size = 2
    total_vpages = 8

    sm_scale = 1.0 / math.sqrt(head_dim)
    q_raw = rng.normal(size=(q_heads, q_tokens, head_dim)).astype(np.float32)
    q_scaled_hm = jnp.asarray(
        q_raw * (sm_scale * pcp_kernel._splash_kernel.LOG2E), dtype=jnp.float32
    )

    req0_kv_len, req1_kv_len = 652, 692
    k_req0 = rng.normal(size=(kv_heads, req0_kv_len, head_dim)).astype(np.float32)
    v_req0 = rng.normal(size=(kv_heads, req0_kv_len, head_dim)).astype(np.float32)
    k_req1 = rng.normal(size=(kv_heads, req1_kv_len, head_dim)).astype(np.float32)
    v_req1 = rng.normal(size=(kv_heads, req1_kv_len, head_dim)).astype(np.float32)

    k_vpages = np.full(
        (kv_heads, total_vpages, virtual_page_size, head_dim), 1e6, dtype=np.float32
    )
    v_vpages = np.full(
        (kv_heads, total_vpages, virtual_page_size, head_dim), 1e6, dtype=np.float32
    )
    for vp in range(3):
        s, e = vp * virtual_page_size, min((vp + 1) * virtual_page_size, req0_kv_len)
        k_vpages[:, vp, : e - s, :] = k_req0[:, s:e, :]
        v_vpages[:, vp, : e - s, :] = v_req0[:, s:e, :]
    for vp in range(3):
        s, e = vp * virtual_page_size, min((vp + 1) * virtual_page_size, req1_kv_len)
        k_vpages[:, 3 + vp, : e - s, :] = k_req1[:, s:e, :]
        v_vpages[:, 3 + vp, : e - s, :] = v_req1[:, s:e, :]
    k_vpages_jnp = jnp.asarray(k_vpages)
    v_vpages_jnp = jnp.asarray(v_vpages)

    req0_abs = np.arange(592, 652, dtype=np.int32)
    req1_abs = np.arange(632, 692, dtype=np.int32)
    q_seg_ids = jnp.asarray(
        np.concatenate(
            [
                np.zeros(60, dtype=np.int32),
                np.ones(60, dtype=np.int32),
                np.full(8, -1, dtype=np.int32),
            ]
        )
    )
    q_sequence = jnp.asarray(
        np.concatenate(
            [
                req0_abs,
                3 * virtual_page_size + req1_abs,
                np.full(8, -1, dtype=np.int32),
            ]
        )
    )
    stage_plan = SimpleNamespace(
        sorted_block_ids=jnp.arange(total_vpages, dtype=jnp.int32),
        sorted_seg_ids=jnp.asarray([0, 0, 0, 1, 1, 1, 2, -2], dtype=jnp.int32),
        sorted_vpage_valid_tokens=jnp.asarray(
            [256, 256, 140, 256, 256, 180, 64, 0], dtype=jnp.int32
        ),
        num_active_vpages=jnp.asarray(7, dtype=jnp.int32),
        q_seg_ids=q_seg_ids,
        q_sequence=q_sequence,
    )
    splash_cfg = pcp_kernel._splash_kernel.SplashConfig(
        block_q=128,
        block_kv=512,
        block_kv_compute=256,
        num_stacked_q_heads=2,
        use_base2_exp=True,
        q_prescaled_log2e=True,
    )
    kv_cache_local = jnp.zeros(
        (total_vpages, page_size, kv_heads, 2, head_dim), dtype=jnp.float32
    )

    def fake_gather(_kv_cache_local, block_ids, **_kwargs):
        k_sel = k_vpages_jnp[:, block_ids, :, :].reshape(kv_heads, -1, head_dim)
        v_sel = v_vpages_jnp[:, block_ids, :, :].reshape(kv_heads, -1, head_dim)
        return k_sel, v_sel

    class FakeSplashKernel:
        def __init__(self, fwd_mask_info, dkv_mask_info, **kwargs):
            self.fwd_mask_info = fwd_mask_info
            self.dkv_mask_info = dkv_mask_info
            self.kwargs = kwargs
            self.mask_function = kwargs.get("mask_function")

        def __call__(self, q_scaled, k_hm, v_hm, segment_ids, save_residuals=False):
            assert save_residuals
            q_rep_factor = q_scaled.shape[0] // k_hm.shape[0]
            k_rep = jnp.repeat(k_hm.astype(jnp.float32), q_rep_factor, axis=0)
            v_rep = jnp.repeat(v_hm.astype(jnp.float32), q_rep_factor, axis=0)
            scores_b2 = jnp.einsum(
                "hqd,hkd->hqk",
                q_scaled.astype(jnp.float32),
                k_rep,
                precision="highest",
            )
            q_seq = jnp.asarray(self.fwd_mask_info.q_sequence, dtype=jnp.int32)[:, None]
            kv_idx = jnp.arange(k_hm.shape[1], dtype=jnp.int32)[None, :]
            causal_ok = self.mask_function(q_seq, kv_idx)
            seg_ok = (segment_ids.q[:, None] == segment_ids.kv[None, :]) & (
                segment_ids.q[:, None] >= 0
            )
            valid = (causal_ok & seg_ok)[None, :, :]
            any_block_valid = jnp.any(valid)
            sentinel = jnp.asarray(
                pcp_kernel._splash_kernel.DEFAULT_MASK_VALUE
                / pcp_kernel._splash_kernel.LOG2E,
                dtype=jnp.float32,
            )
            masked_b2 = jnp.where(valid, scores_b2, -jnp.inf)
            m_b2 = jnp.max(masked_b2, axis=-1)
            m_safe = jnp.where(jnp.isneginf(m_b2), 0.0, m_b2)
            weights = jnp.where(valid, jnp.exp2(masked_b2 - m_safe[..., None]), 0.0)
            l_sum = jnp.sum(weights, axis=-1)
            inv_l = 1.0 / jnp.where(l_sum > 0.0, l_sum, 1.0)
            out_nonempty = jnp.einsum(
                "hqk,hkd->hqd",
                weights * inv_l[..., None],
                v_rep,
                precision="highest",
            )
            lse_nonempty = (m_b2 + jnp.log2(l_sum)) / jnp.asarray(
                pcp_kernel._splash_kernel.LOG2E, dtype=jnp.float32
            )
            out = jnp.where(any_block_valid, out_nonempty, jnp.zeros_like(out_nonempty))
            lse = jnp.where(
                any_block_valid,
                lse_nonempty,
                jnp.full_like(lse_nonempty, sentinel),
            )
            return out, {"logsumexp": lse}

    monkeypatch.setattr(pcp_kernel, "_gather_vpages_to_head_major", fake_gather)
    monkeypatch.setattr(
        pcp_kernel._splash_kernel, "SplashAttentionKernel", FakeSplashKernel
    )

    actual = np.asarray(
        pcp_kernel._attend_multi_vpage_groups(
            kv_cache_local,
            q_scaled_hm,
            stage_plan,
            group_size,
            splash_cfg=splash_cfg,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=32,
            kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
            pcp_axis_name="pcp",
        ),
        dtype=np.float64,
    )
    assert not np.isnan(actual).any()
    np.testing.assert_array_equal(actual[:, 120:, :], 0.0)

    def ref_causal(q_slice, k_req, v_req, abs_pos, kv_cutoff=None):
        if kv_cutoff is not None:
            k_req = k_req[:, :kv_cutoff, :]
            v_req = v_req[:, :kv_cutoff, :]
        k_rep = np.repeat(k_req.astype(np.float64), q_per_kv, axis=0)
        v_rep = np.repeat(v_req.astype(np.float64), q_per_kv, axis=0)
        scores = np.einsum("hqd,hkd->hqk", q_slice.astype(np.float64), k_rep) * sm_scale
        mask = abs_pos[:, None] >= np.arange(k_rep.shape[1])[None, :]
        scores = np.where(mask[None, :, :], scores, -np.inf)
        m = scores.max(axis=-1, keepdims=True)
        w = np.exp(scores - m)
        return np.einsum("hqk,hkd->hqd", w / w.sum(axis=-1, keepdims=True), v_rep)

    ref_req0 = ref_causal(q_raw[:, :60, :], k_req0, v_req0, req0_abs)
    ref_req1 = ref_causal(q_raw[:, 60:120, :], k_req1, v_req1, req1_abs)
    ref_req0_g0_only = ref_causal(
        q_raw[:, :60, :], k_req0, v_req0, req0_abs, kv_cutoff=512
    )

    u_f32 = 2.0**-24
    max_abs_v = max(float(np.max(np.abs(v_req0))), float(np.max(np.abs(v_req1))))
    # Omitted f32 terms: LOG2E prescale + QK dot + exp2/log2 over <=692 keys
    # (32 * u_f32) plus 3 online LSE merge steps (24 * u_f32).
    tol_stream = (32.0 + 24.0) * u_f32 * max_abs_v
    assert float(np.max(np.abs(actual[:, :60, :] - ref_req0))) <= tol_stream
    assert float(np.max(np.abs(actual[:, 60:120, :] - ref_req1))) <= tol_stream
    assert float(np.max(np.abs(ref_req0_g0_only - ref_req0))) > 100.0 * tol_stream


def _load_pcp8_bench_module():
    bench_path = Path(__file__).with_name("test_pcp8_160k_chunk_benchmark.py")
    spec = importlib.util.spec_from_file_location("_pcp8_bench_test_mod", bench_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gather_vpages_to_head_major_with_odd_fp8_kv_heads():
    """Verifies FP8 gather drops padded sublane planes after all_gather."""
    pcp_size = 8
    page_size = 16
    interleave_size = 16
    num_vpages = 4
    num_blocks = 6
    head_dim = 128
    kv_packing = 4
    block_ids = jnp.asarray([1, 3, 0, 5], dtype=jnp.int32)
    virtual_page_size = page_size * pcp_size
    num_tokens = num_vpages * virtual_page_size

    def _run_gather(local_caches: jax.Array, num_kvh: int):
        return jax.vmap(
            lambda cache_shard: pcp_kernel._gather_vpages_to_head_major(
                cache_shard,
                block_ids,
                kv_heads=num_kvh,
                head_dim=head_dim,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                pcp_axis_name="pcp",
            ),
            in_axes=0,
            out_axes=0,
            axis_name="pcp",
        )(local_caches)

    for kv_heads in (1, 3):
        packed_groups = math.ceil(2 * kv_heads / kv_packing)
        rng = np.random.default_rng(100 + kv_heads)
        k_true = rng.choice(
            np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float32),
            size=(kv_heads, num_blocks * virtual_page_size, head_dim),
        ).astype(jnp.float8_e4m3fn)
        v_true = rng.choice(
            np.array([-1.5, -0.5, 0.25, 0.75, 1.5, 3.0], dtype=np.float32),
            size=(kv_heads, num_blocks * virtual_page_size, head_dim),
        ).astype(jnp.float8_e4m3fn)

        kv_interleaved = np.zeros(
            (num_blocks, pcp_size, page_size, packed_groups * kv_packing, head_dim),
            dtype=np.float32,
        )
        # Seed padded sublane planes so unsliced planes fail parity loudly.
        kv_interleaved[:, :, :, 2 * kv_heads :, :] = -7.0
        k_np = np.asarray(k_true, dtype=np.float32).reshape(
            kv_heads, num_blocks, pcp_size, page_size, head_dim
        )
        v_np = np.asarray(v_true, dtype=np.float32).reshape(
            kv_heads, num_blocks, pcp_size, page_size, head_dim
        )
        for h in range(kv_heads):
            kv_interleaved[:, :, :, 2 * h, :] = np.transpose(k_np[h], (0, 1, 2, 3))
            kv_interleaved[:, :, :, 2 * h + 1, :] = np.transpose(v_np[h], (0, 1, 2, 3))

        # Per-rank cache shards [pcp, blocks, page_size, groups, packing, head_dim]
        local_caches = jnp.asarray(
            np.transpose(kv_interleaved, (1, 0, 2, 3, 4)).reshape(
                pcp_size, num_blocks, page_size, packed_groups, kv_packing, head_dim
            ),
            dtype=jnp.float8_e4m3fn,
        )

        gathered_k, gathered_v = _run_gather(local_caches, kv_heads)

        expected_k = (
            np.asarray(k_true, dtype=np.float32)
            .reshape(kv_heads, num_blocks, virtual_page_size, head_dim)[
                :, np.asarray(block_ids), :, :
            ]
            .reshape(kv_heads, num_tokens, head_dim)
        )
        expected_v = (
            np.asarray(v_true, dtype=np.float32)
            .reshape(kv_heads, num_blocks, virtual_page_size, head_dim)[
                :, np.asarray(block_ids), :, :
            ]
            .reshape(kv_heads, num_tokens, head_dim)
        )
        for rank in range(pcp_size):
            np.testing.assert_allclose(
                np.asarray(gathered_k[rank], dtype=np.float32),
                expected_k,
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                np.asarray(gathered_v[rank], dtype=np.float32),
                expected_v,
                rtol=0.0,
                atol=0.0,
            )
        # Negative control: padded plane sentinel differs from valid planes by >= 4.0.
        pad_plane = np.asarray(local_caches[0, 1, :, -1, -1, :], dtype=np.float32)
        assert float(np.max(np.abs(pad_plane - expected_v[-1, :page_size, :]))) >= 4.0


def test_packed_local_from_metadata_with_overflow_writeback():
    """Verifies multi-window writeback when a cache rank owns > local_tokens rows."""
    pcp_size = 8
    page_size = 128
    interleave_size = 128
    local_tokens = 128
    kv_heads = 2
    q_per_kv = 2
    head_dim = 128
    kv_packing = 4
    num_blocks = 8
    max_seqs = 4
    steps_added = [[53, 1, 129], [1, 3, 2], [2, 1, 1]]
    block_tables_np = np.zeros((max_seqs, 2), dtype=np.int32)
    block_tables_np[0] = [0, 1]
    block_tables_np[1] = [2, 3]
    block_tables_np[2] = [4, 5]

    orig_write = pcp_kernel.write_captured_kv_to_local_cache
    orig_attend = pcp_kernel._attend_over_vpage_groups
    orig_stage_rank = pcp_stage_plan.get_pcp_rank_without_axis_index
    orig_kernel_rank = pcp_kernel._get_pcp_rank_without_axis_index

    seen_desc_shapes: list[tuple[int, ...]] = []

    def _pure_write(
        kv_cache_local,
        captured_kv_local,
        segment_descriptors,
        num_segments,
        *,
        kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
    ):
        seen_desc_shapes.append(tuple(int(d) for d in segment_descriptors.shape))
        max_segs = int(segment_descriptors.shape[1])
        cap = int(
            captured_kv_local.shape[-1]
            if kv_layout == pcp_kernel.KVLayout.SEQ_ALONG_LANE
            else captured_kv_local.shape[0]
        )
        seg_ids = jnp.arange(max_segs, dtype=jnp.int32)[:, None]
        offs = jnp.arange(cap, dtype=jnp.int32)[None, :]
        s0 = segment_descriptors[0, :][:, None]
        d0 = segment_descriptors[1, :][:, None]
        ln = segment_descriptors[2, :][:, None]
        src = s0 + offs
        valid = (seg_ids < jnp.reshape(num_segments, ())) & (offs < ln)
        src_safe = jnp.clip(src, 0, cap - 1)
        flat_slots = d0 + offs
        if kv_layout == pcp_kernel.KVLayout.SEQ_ALONG_LANE:
            page_sz = kv_cache_local.shape[-1]
            total_slots = kv_cache_local.shape[0] * page_sz
            flat_cache = kv_cache_local.transpose(0, 4, 1, 2, 3).reshape(
                total_slots, *kv_cache_local.shape[1:4]
            )
            cap_rows = captured_kv_local.transpose(3, 0, 1, 2)
            vals = cap_rows[src_safe.reshape(-1)]
            idx = jnp.where(valid.reshape(-1), flat_slots.reshape(-1), total_slots)
            updated = flat_cache.at[idx].set(vals, mode="drop")
            return updated.reshape(
                kv_cache_local.shape[0], page_sz, *kv_cache_local.shape[1:4]
            ).transpose(0, 2, 3, 4, 1)
        page_sz = kv_cache_local.shape[1]
        total_slots = kv_cache_local.shape[0] * page_sz
        flat_cache = kv_cache_local.reshape(total_slots, *kv_cache_local.shape[2:])
        vals = captured_kv_local[src_safe.reshape(-1)]
        idx = jnp.where(valid.reshape(-1), flat_slots.reshape(-1), total_slots)
        return flat_cache.at[idx].set(vals, mode="drop").reshape(kv_cache_local.shape)

    def _stub_attend(kv_cache_local, q_scaled_hm, stage_plan, **kwargs):
        return jnp.zeros(q_scaled_hm.shape, dtype=jnp.float32)

    def _make_step_runner(kv_layout_val):
        @jax.jit
        def _run_step(
            q_sharded, kv_sharded, cache_sharded, kv_lens_j, pages_j, cu_q_j, dist_j
        ):
            def _per_rank(q_loc, kv_loc, cache_loc):
                plan = pcp_stage_plan.build_pcp_local_stage_plan(
                    kv_lens_j,
                    pages_j,
                    cu_q_j,
                    dist_j,
                    local_tokens=local_tokens,
                    local_kv_cache_num_blocks=num_blocks,
                    page_size=page_size,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                    vpage_group_size=4,
                    kv_layout=kv_layout_val,
                    pcp_axis_name="pcp",
                )
                _, next_cache = (
                    pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                        q_loc,
                        kv_loc,
                        cache_loc,
                        kv_lens_j,
                        cu_q_j,
                        dist_j,
                        plan.segment_descriptors,
                        plan.num_segments,
                        pcp_size=pcp_size,
                        interleave_size=interleave_size,
                        sm_scale=1.0,
                        k_scale=1.0,
                        v_scale=1.0,
                        kv_layout=kv_layout_val,
                        mesh_axis_names=("pcp",),
                        pcp_axis_name="pcp",
                        stage_plan=plan,
                        vpage_group_size=4,
                    )
                )
                return next_cache

            return jax.vmap(_per_rank, in_axes=(0, 0, 0), out_axes=0, axis_name="pcp")(
                q_sharded, kv_sharded, cache_sharded
            )

        return _run_step

    try:
        pcp_kernel.write_captured_kv_to_local_cache = _pure_write
        pcp_kernel._attend_over_vpage_groups = _stub_attend
        pcp_stage_plan.get_pcp_rank_without_axis_index = (
            lambda axis_name, _pcp_size: jax.lax.axis_index(axis_name)
        )
        pcp_kernel._get_pcp_rank_without_axis_index = (
            lambda axis_name, _pcp_size: jax.lax.axis_index(axis_name)
        )

        for layout in (
            pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
            pcp_kernel.KVLayout.SEQ_ALONG_LANE,
        ):
            if layout == pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE:
                cache_shape = (
                    pcp_size,
                    num_blocks,
                    page_size,
                    1,
                    kv_packing,
                    head_dim,
                )
            else:
                cache_shape = (
                    pcp_size,
                    num_blocks,
                    2 * kv_heads,
                    head_dim // kv_packing,
                    kv_packing,
                    page_size,
                )

            run_step_fn = _make_step_runner(layout)
            cache_actual = jnp.full(cache_shape, -2.0, dtype=jnp.float8_e4m3fn)
            cache_ref = np.full(cache_shape, -2.0, dtype=np.float32)
            seq_so_far = [0, 0, 0]
            step0_inputs = None
            window0_only_ref = None

            for step_idx, added in enumerate(steps_added):
                q_starts = [0]
                for a in added:
                    q_starts.append(q_starts[-1] + a)
                total_q = q_starts[-1]
                cu_q_np = np.full((max_seqs + 1,), total_q, dtype=np.int32)
                for r_i, s in enumerate(q_starts):
                    cu_q_np[r_i] = s
                kv_lens_np = np.zeros((max_seqs,), dtype=np.int32)
                for r_i in range(3):
                    seq_so_far[r_i] += added[r_i]
                    kv_lens_np[r_i] = seq_so_far[r_i]
                dist_np = np.array([0, 0, 3], dtype=np.int32)

                if layout == pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE:
                    cur_kv_np = np.zeros(
                        (pcp_size, local_tokens, 1, kv_packing, head_dim),
                        dtype=np.float32,
                    )
                else:
                    cur_kv_np = np.zeros(
                        (
                            pcp_size,
                            2 * kv_heads,
                            head_dim // kv_packing,
                            kv_packing,
                            local_tokens,
                        ),
                        dtype=np.float32,
                    )

                cycle = pcp_size * interleave_size
                step0_rank0_writes = 0
                for rank in range(pcp_size):
                    row_cursor = 0
                    for r_i in range(3):
                        h_start = seq_so_far[r_i] - added[r_i]
                        o_start = q_starts[r_i]
                        o_end = q_starts[r_i + 1]
                        c_pos = (o_start // cycle) * cycle + rank * interleave_size
                        if c_pos + interleave_size <= o_start:
                            c_pos += cycle
                        while c_pos < o_end:
                            s_lo = max(c_pos, o_start)
                            s_hi = min(c_pos + interleave_size, o_end)
                            for op in range(s_lo, s_hi):
                                abs_pos = h_start + (op - o_start)
                                val = float(
                                    ((step_idx + 1) * 3 + r_i + abs_pos) % 7 + 1
                                )
                                vpage = abs_pos // (pcp_size * page_size)
                                rem = abs_pos % (pcp_size * page_size)
                                c_rank = (rem // interleave_size) % pcp_size
                                p_off = (
                                    rem // (pcp_size * interleave_size)
                                ) * interleave_size + (rem % interleave_size)
                                blk = int(block_tables_np[r_i, vpage])
                                if layout == pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE:
                                    cur_kv_np[rank, row_cursor] = val
                                    cache_ref[c_rank, blk, p_off] = val
                                else:
                                    cur_kv_np[rank, :, :, :, row_cursor] = val
                                    cache_ref[c_rank, blk, :, :, :, p_off] = val
                                if step_idx == 0 and c_rank == 0:
                                    step0_rank0_writes += 1
                                row_cursor += 1
                            c_pos += cycle

                if step_idx == 0:
                    # Window-0-only control drops writes 128..181 on cache rank 0.
                    assert step0_rank0_writes == 182
                    window0_only_ref = cache_ref.copy()
                    # Token abs_pos=128 of req 2 is capture_offset 181 on rank 0.
                    if layout == pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE:
                        window0_only_ref[0, int(block_tables_np[2, 0]), 127] = -2.0
                    else:
                        window0_only_ref[
                            0, int(block_tables_np[2, 0]), :, :, :, 127
                        ] = -2.0

                q_dummy = jnp.zeros(
                    (pcp_size, local_tokens, kv_heads, q_per_kv, head_dim),
                    dtype=jnp.bfloat16,
                )
                cache_actual = run_step_fn(
                    q_dummy,
                    jnp.asarray(cur_kv_np, dtype=jnp.float8_e4m3fn),
                    cache_actual,
                    jnp.asarray(kv_lens_np),
                    jnp.asarray(block_tables_np.reshape(-1)),
                    jnp.asarray(cu_q_np),
                    jnp.asarray(dist_np),
                )
                if step_idx == 0:
                    step0_inputs = (
                        jnp.asarray(kv_lens_np),
                        jnp.asarray(block_tables_np.reshape(-1)),
                        jnp.asarray(cu_q_np),
                        jnp.asarray(dist_np),
                    )

            expected_fp8 = np.asarray(
                jnp.asarray(cache_ref, dtype=jnp.float8_e4m3fn), dtype=np.float32
            )
            actual_f32 = np.asarray(cache_actual, dtype=np.float32)
            np.testing.assert_allclose(actual_f32, expected_fp8, rtol=0.0, atol=0.0)
            assert step0_inputs is not None
            step0_max_wb = jax.vmap(
                lambda _, inps=step0_inputs, kv_layout_val=layout: (
                    pcp_stage_plan.build_pcp_local_stage_plan(
                        *inps,
                        local_tokens=local_tokens,
                        local_kv_cache_num_blocks=num_blocks,
                        page_size=page_size,
                        pcp_size=pcp_size,
                        interleave_size=interleave_size,
                        vpage_group_size=4,
                        kv_layout=kv_layout_val,
                        pcp_axis_name="pcp",
                    ).max_writeback_count
                ),
                in_axes=0,
                out_axes=0,
                axis_name="pcp",
            )(jnp.arange(pcp_size, dtype=jnp.int32))
            assert int(np.asarray(step0_max_wb)[0]) == 182
            assert seen_desc_shapes and all(
                sh == (3, local_tokens) for sh in seen_desc_shapes
            )
            assert window0_only_ref is not None
            assert float(np.max(np.abs(window0_only_ref - expected_fp8))) >= 3.0
    finally:
        pcp_kernel.write_captured_kv_to_local_cache = orig_write
        pcp_kernel._attend_over_vpage_groups = orig_attend
        pcp_stage_plan.get_pcp_rank_without_axis_index = orig_stage_rank
        pcp_kernel._get_pcp_rank_without_axis_index = orig_kernel_rank


def test_tpu7x_compile_only_zero_kv_cache_copy_and_zero_entry_hoist():
    """Compiles on tpu7x:2x2x1 and asserts zero KV-cache copies or ENTRY hoists."""
    if jax.default_backend() == "tpu" and len(jax.devices()) >= 8:
        tpu_devices = np.asarray(jax.devices()[:8])
    else:
        topo = topologies.get_topology_desc(topology_name="tpu7x:2x2x1", platform="tpu")
        tpu_devices = np.asarray(topo.devices).reshape(-1)[:8]
    mesh = Mesh(tpu_devices, ("pcp",))

    def _compile_hlo(
        kv_heads: int, num_blocks: int, page_size: int, local_tokens: int
    ) -> str:
        packed_groups = math.ceil(2 * kv_heads / 4)
        q_per_kv = 8 if kv_heads == 2 else 4
        head_dim = 256 if kv_heads == 2 else 128
        max_seqs = 8

        @jax.shard_map(
            mesh=mesh,
            in_specs=(
                P("pcp", None, None, None),
                P("pcp", None, None, None),
                P("pcp", None, None, None, None, None),
                P(None),
                P(None),
                P(None),
                P(None),
            ),
            out_specs=(
                P("pcp", None, None, None),
                P("pcp", None, None, None, None, None),
            ),
            check_vma=False,
        )
        def _shard_fn(q_sh, kv_sh, cache_sh, kv_lens, pages, cu_q, dist):
            cache_loc = jnp.squeeze(cache_sh, axis=0)
            plan = pcp_stage_plan.build_pcp_local_stage_plan(
                kv_lens,
                pages,
                cu_q,
                dist,
                local_tokens=local_tokens,
                local_kv_cache_num_blocks=num_blocks,
                page_size=page_size,
                pcp_size=8,
                interleave_size=page_size,
                vpage_group_size=32,
                kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                pcp_axis_name="pcp",
            )
            out_loc, next_cache = (
                pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                    q_sh,
                    kv_sh,
                    cache_loc,
                    kv_lens,
                    cu_q,
                    dist,
                    plan.segment_descriptors,
                    plan.num_segments,
                    pcp_size=8,
                    interleave_size=page_size,
                    sm_scale=1.0,
                    k_scale=1.0,
                    v_scale=1.0,
                    kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                    mesh_axis_names=("pcp",),
                    pcp_axis_name="pcp",
                    stage_plan=plan,
                    vpage_group_size=32,
                )
            )
            return out_loc, jnp.expand_dims(next_cache, axis=0)

        fn = jax.jit(_shard_fn, donate_argnums=(2,))
        lowered = fn.lower(
            jax.ShapeDtypeStruct(
                (8 * local_tokens, kv_heads, q_per_kv, head_dim), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct(
                (8 * local_tokens, packed_groups, 4, head_dim), jnp.float8_e4m3fn
            ),
            jax.ShapeDtypeStruct(
                (8, num_blocks, page_size, packed_groups, 4, head_dim),
                jnp.float8_e4m3fn,
            ),
            jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs * num_blocks,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )
        return lowered.compile().as_text()

    def _scan_slow_paths(hlo_text: str, cache_prefix: str, full_rows: int):
        lines = hlo_text.splitlines()
        entry_idx = next(
            (i for i, line in enumerate(lines) if line.startswith("ENTRY ")), 0
        )
        entry_lines = lines[entry_idx:]
        donated_copies = [
            line.strip()
            for line in lines
            if " copy(" in line
            and cache_prefix in line
            and 'metadata={op_name="jit(_shard_fn)/shard_map"' not in line
        ]
        entry_scatters = [
            line.strip()
            for line in entry_lines
            if 'op_name="jit(_shard_fn)/shard_map/while/body/' in line
            or ("scatter" in line and f"s32[{full_rows}]" in line)
        ]
        odd_f8_copies = [
            line.strip() for line in lines if " copy(" in line and "f8e4m3fn[6," in line
        ]
        entry_whiles = [line.strip() for line in entry_lines if " while(" in line]
        return donated_copies, entry_scatters, odd_f8_copies, entry_whiles

    hlo_qwen35 = _compile_hlo(
        kv_heads=2, num_blocks=128, page_size=256, local_tokens=4096
    )
    copies_q35, scatters_q35, _, whiles_q35 = _scan_slow_paths(
        hlo_qwen35, "f8e4m3fn[128,256,", 8 * 4096
    )
    assert len(whiles_q35) >= 1, "Expected overflow writeback while loop in ENTRY"
    assert copies_q35 == [], f"Unexpected donated KV-cache kCopy ops: {copies_q35}"
    assert scatters_q35 == [], f"Unexpected hoisted ENTRY scatter ops: {scatters_q35}"

    hlo_odd3 = _compile_hlo(kv_heads=3, num_blocks=64, page_size=128, local_tokens=128)
    copies_odd, scatters_odd, odd_f8_copies, whiles_odd = _scan_slow_paths(
        hlo_odd3, "f8e4m3fn[64,128,", 8 * 128
    )
    assert len(whiles_odd) >= 1, "Expected overflow writeback while loop in ENTRY"
    assert copies_odd == [], f"Unexpected donated KV-cache kCopy ops: {copies_odd}"
    assert scatters_odd == [], f"Unexpected hoisted ENTRY scatter ops: {scatters_odd}"
    assert odd_f8_copies == [], (
        f"Unexpected odd-FP8 transpose kCopy ops: {odd_f8_copies}"
    )

    # Negative control: detector catches synthetic %copy.198, scatter, and f8e4m3fn[6,
    synthetic_bad = (
        "ENTRY %main {\n"
        "  %copy.198 = f8e4m3fn[128,256,1,4,256] copy(%bitcast.858)\n"
        "  %copy.236 = f8e4m3fn[6,32,1,128,128] copy(%slice.9)\n"
        "  %scatter_offload = ((s32[512]{0}, s32[32768]{0})) async-update(%p)\n"
        "}\n"
    )
    bad_c, bad_s, bad_o, _ = _scan_slow_paths(
        synthetic_bad, "f8e4m3fn[128,256,", 8 * 4096
    )
    assert len(bad_c) == 1 and len(bad_s) == 1 and len(bad_o) == 1


def test_extract_profiled_kernel_summary_with_unified_splash_ops(tmp_path, monkeypatch):
    """Verifies unified Splash XProf summary and misaligned_overflow reference."""
    bench_mod = _load_pcp8_bench_module()

    def _write_xspace(out_dir: Path, inject_copy_ns: int = 0) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        planes_txt = []
        wb_name = "%pcp_write_captured_kv_to_local_cache.6"
        sp_name = "%splash_mha_fwd_segmented_part.1"
        copy_hbm_name = (
            "%copy.198 = f8e4m3fn[128,256,1,4,256]"
            "{3,2,1,0:T(8,128)(4,1)} copy(%param.23)"
        )
        copy_start_name = (
            "%copy-start.70 = (s32[2]{0:T(128)S(1)}, s32[2]{0:T(128)}, "
            "u32[]{:S(2)}) copy-start(%p)"
        )
        copy_s32_name = (
            "%copy.257 = s32[16,16,8,128]{3,1,2,0:T(8,128)} "
            "copy(%select_bitcast_fusion.8)"
        )
        copy_vmem_name = (
            "%copy.160 = f8e4m3fn[16,256,1,4,256]"
            "{4,1,3,0,2:T(8,128)(4,1)S(1)} copy(%gte)"
        )
        for dev_idx in range(bench_mod.WORLD_SIZE):
            events_txt = []
            # Metadata 1 = writeback, 2 = splash, 3 = 5D HBM copy, 4..6 = excluded
            for step in range(bench_mod.PROFILE_STEPS):
                base_ps = (step + 1) * 100_000_000_000  # 100 ms in picoseconds
                # Two writeback windows (0.4 ms and 0.6 ms = 1.0 ms total)
                events_txt.append(
                    f"events {{ metadata_id: 1 offset_ps: {base_ps} "
                    "duration_ps: 400000000 }"
                )
                events_txt.append(
                    f"events {{ metadata_id: 1 offset_ps: {base_ps + 500_000_000} "
                    "duration_ps: 600000000 }"
                )
                for mid in (4, 5, 6):
                    noise_ps = base_ps + 1_100_000_000 + mid * 10_000_000
                    events_txt.append(
                        f"events {{ metadata_id: {mid} offset_ps: {noise_ps}"
                        " duration_ps: 50000000 }"
                    )
                if inject_copy_ns > 0:
                    events_txt.append(
                        f"events {{ metadata_id: 3 offset_ps: {base_ps + 1_200_000_000}"
                        f" duration_ps: {inject_copy_ns * 1000} "
                        "stats { metadata_id: 1 str_value: "
                        '"f8e4m3fn[128,256,1,4,256]{4,3,1,0,2:T(4,128)(4,1)}" } }'
                    )
                # Three splash groups (2.0 ms + 3.0 ms + 4.0 ms + 0.1*dev_idx ms)
                extra_ps = dev_idx * 100_000_000
                events_txt.append(
                    f"events {{ metadata_id: 2 offset_ps: {base_ps + 2_000_000_000} "
                    "duration_ps: 2000000000 }"
                )
                events_txt.append(
                    f"events {{ metadata_id: 2 offset_ps: {base_ps + 5_000_000_000} "
                    "duration_ps: 3000000000 }"
                )
                events_txt.append(
                    f"events {{ metadata_id: 2 offset_ps: {base_ps + 9_000_000_000} "
                    f"duration_ps: {4000000000 + extra_ps} }}"
                )
            line_joined = "\n      ".join(events_txt)
            planes_txt.append(
                f"""planes {{
  id: {dev_idx + 1}
  name: "/device:TPU:{dev_idx}"
  event_metadata {{ key: 1 value {{ id: 1 name: "{wb_name}" }} }}
  event_metadata {{ key: 2 value {{ id: 2 name: "{sp_name}" }} }}
  event_metadata {{ key: 3 value {{ id: 3 name: "{copy_hbm_name}" }} }}
  event_metadata {{ key: 4 value {{ id: 4 name: "{copy_start_name}" }} }}
  event_metadata {{ key: 5 value {{ id: 5 name: "{copy_s32_name}" }} }}
  event_metadata {{ key: 6 value {{ id: 6 name: "{copy_vmem_name}" }} }}
  stat_metadata {{ key: 1 value {{ id: 1 name: "shape_with_layout" }} }}
  lines {{
    id: 1
    name: "XLA Ops"
    timestamp_ns: 1000
    {line_joined}
  }}
}}"""
            )
        xspace_bytes = jax.profiler.ProfileData.text_proto_to_serialized_xspace(
            "\n".join(planes_txt)
        )
        (out_dir / "host0.xplane.pb").write_bytes(xspace_bytes)

    clean_dir = tmp_path / "xprof_clean"
    _write_xspace(clean_dir, inject_copy_ns=0)
    monkeypatch.setenv("TPU_PCP8_160K_PROFILE_DIR", str(clean_dir))
    summary_clean = bench_mod._extract_profiled_kernel_summary()
    # Slowest core is dev_idx=7: writeback=1.0 ms, splash=9.7 ms -> critical=10.7 ms
    assert math.isclose(
        float(summary_clean["profiled_kernel_current_state_core_step_median_ms"]),
        1.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(summary_clean["profiled_kernel_copy_core_step_median_ms"]),
        0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(summary_clean["profiled_kernel_critical_step_median_ms"]),
        10.7,
        abs_tol=1e-6,
    )

    # Negative control: injected 2.5 ms %copy.198 raises copy_ms to 2.5 and sum to 13.2.
    bad_dir = tmp_path / "xprof_with_copy"
    _write_xspace(bad_dir, inject_copy_ns=2_500_000)
    monkeypatch.setenv("TPU_PCP8_160K_PROFILE_DIR", str(bad_dir))
    summary_bad = bench_mod._extract_profiled_kernel_summary()
    assert math.isclose(
        float(summary_bad["profiled_kernel_copy_core_step_median_ms"]),
        2.5,
        abs_tol=1e-6,
    )
    assert not bench_mod._is_donated_kv_cache_hbm_copy_event("%copy.198")
    assert not bench_mod._is_donated_kv_cache_hbm_copy_event(
        "%copy.198", [("shape_with_layout", "s32[64]{0}")]
    )
    assert not bench_mod._is_donated_kv_cache_hbm_copy_event(
        "%copy.168",
        [
            (
                "shape_with_layout",
                "f8e4m3fn[128,256,1,4,256]{4,1,3,0,2:T(8,128)(4,1)}S(1)",
            )
        ],
    )
    assert bench_mod._is_donated_kv_cache_hbm_copy_event(
        "%copy.168",
        [("shape_with_layout", "f8e4m3fn[128,256,1,4,256]{4,1,3,0,2:T(8,128)(4,1)}")],
    )
    assert bench_mod._is_donated_kv_cache_hbm_copy_event(
        "%copy.198",
        [("shape_with_layout", "f8e4m3fn[128,256,1,4,256]{4,3,2,1,0:T(4,128)(4,1)}")],
    )
    summary_override = bench_mod._extract_profiled_kernel_summary(bad_dir)
    assert math.isclose(
        float(summary_override["profiled_kernel_copy_core_step_median_ms"]),
        2.5,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(summary_bad["profiled_kernel_critical_step_median_ms"]),
        13.2,
        abs_tol=1e-6,
    )

    # Legacy reference arm (writeback + current + history, zero splash) must not
    # enter the unified branch despite matching the writeback custom-call prefix.
    legacy_dir = tmp_path / "xprof_legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_planes = []
    legacy_cur = "%pcp_streaming_attention_current_state_page_groups_multi_head.1"
    legacy_wb = "%pcp_write_captured_kv_to_local_cache.1"
    legacy_hist = (
        "%pcp_streaming_attention_history_output_page_groups_multi_head.1.o_in_ref"
    )
    for dev_idx in range(bench_mod.WORLD_SIZE):
        legacy_events = []
        for step in range(bench_mod.PROFILE_STEPS):
            base_ps = (step + 1) * 100_000_000_000
            legacy_events.append(
                f"events {{ metadata_id: 1 offset_ps: {base_ps} "
                "duration_ps: 2000000000 }"
            )
            legacy_events.append(
                f"events {{ metadata_id: 2 offset_ps: {base_ps + 3_000_000_000} "
                "duration_ps: 500000000 }"
            )
            legacy_events.append(
                f"events {{ metadata_id: 3 offset_ps: {base_ps + 4_000_000_000} "
                "duration_ps: 20000000000 }"
            )
        joined_leg = "\n      ".join(legacy_events)
        legacy_planes.append(
            f"""planes {{
  id: {dev_idx + 1}
  name: "/device:TPU:{dev_idx}"
  event_metadata {{ key: 1 value {{ id: 1 name: "{legacy_cur}" }} }}
  event_metadata {{ key: 2 value {{ id: 2 name: "{legacy_wb}" }} }}
  event_metadata {{ key: 3 value {{ id: 3 name: "{legacy_hist}" }} }}
  lines {{
    id: 1
    name: "XLA Ops"
    timestamp_ns: 1000
    {joined_leg}
  }}
}}"""
        )
    (legacy_dir / "host0.xplane.pb").write_bytes(
        jax.profiler.ProfileData.text_proto_to_serialized_xspace(
            "\n".join(legacy_planes)
        )
    )
    monkeypatch.setenv("TPU_PCP8_160K_PROFILE_DIR", str(legacy_dir))
    summary_legacy = bench_mod._extract_profiled_kernel_summary()
    assert math.isclose(
        float(summary_legacy["profiled_kernel_critical_step_median_ms"]),
        22.0,
        abs_tol=1e-6,
    )

    # Verify misaligned_overflow output matches _compute_pcp_local_query_row_mapping.
    monkeypatch.setenv("TPU_PCP8_160K_WORKLOAD", "misaligned_overflow")
    active_reqs = bench_mod.OVERFLOW_ACTIVE_REQS
    q_toks = bench_mod.OVERFLOW_QUERY_TOKENS
    hist_toks = bench_mod.OVERFLOW_HISTORY_TOKENS
    q_lens = jnp.asarray(
        [q_toks] * active_reqs + [0] * (bench_mod.MAX_NUM_SEQS - active_reqs),
        dtype=jnp.int32,
    )
    q_starts = jnp.asarray(
        [i * q_toks for i in range(active_reqs)]
        + [0] * (bench_mod.MAX_NUM_SEQS - active_reqs),
        dtype=jnp.int32,
    )
    hist_starts = jnp.asarray(
        [hist_toks] * active_reqs + [0] * (bench_mod.MAX_NUM_SEQS - active_reqs),
        dtype=jnp.int32,
    )
    for rank in range(bench_mod.WORLD_SIZE):
        mapping = pcp_stage_plan._compute_pcp_local_query_row_mapping(
            q_lens,
            token_owner_starts=q_starts,
            request_absolute_starts=hist_starts,
            local_padded_tokens=bench_mod.LOCAL_QUERY_TOKENS,
            pcp_size=bench_mod.WORLD_SIZE,
            pcp_rank=rank,
            interleave_size=bench_mod.PCP_INTERLEAVE_SIZE,
        )
        abs_p = np.asarray(mapping.absolute_positions, dtype=np.float64)
        valid_p = np.asarray(mapping.is_valid, dtype=bool)
        safe_abs = np.where(valid_p, abs_p, 0.0)
        ref_out = np.where(valid_p, safe_abs / (safe_abs + 1.0), 0.0).astype(np.float32)
        got_out = bench_mod._expected_local_output(torch, rank).numpy()
        np.testing.assert_allclose(got_out, ref_out, rtol=0.0, atol=1e-7)


def test_attend_over_vpage_groups_routes_rung_96_on_nondivisible_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS == (
        32,
        64,
        96,
        128,
        256,
        512,
    )

    def _fake_single_group(
        _kv_cache_local,
        q_scaled_hm,
        _stage_plan,
        rung_vpages: int,
        **_kwargs,
    ):
        return jnp.full_like(q_scaled_hm, float(rung_vpages), dtype=jnp.float32)

    monkeypatch.setattr(pcp_kernel, "_attend_single_vpage_group", _fake_single_group)
    monkeypatch.setattr(
        pcp_kernel,
        "_attend_multi_vpage_groups",
        lambda _kv, q_scaled_hm, _plan, _vpg, **_kw: jnp.full_like(
            q_scaled_hm, -1.0, dtype=jnp.float32
        ),
    )
    cfg = pcp_kernel._splash_kernel.SplashConfig(
        block_q=128,
        block_kv=1024,
        block_kv_compute=256,
        num_stacked_q_heads=16,
        use_base2_exp=True,
        q_prescaled_log2e=True,
    )
    for total_vpages in (512, 8192):
        assert total_vpages % 96 != 0
        for active_vpages, expected_rung in (
            (8, 32),
            (40, 64),
            (80, 96),
            (96, 96),
            (124, 128),
            (508, 512),
        ):
            plan = pcp_stage_plan.PCPStagePlan(
                segment_descriptors=jnp.zeros((3, 8), dtype=jnp.int32),
                num_segments=jnp.asarray(0, dtype=jnp.int32),
                sorted_block_ids=jnp.zeros((total_vpages,), dtype=jnp.int32),
                sorted_seg_ids=jnp.zeros((total_vpages,), dtype=jnp.int32),
                sorted_vpage_valid_tokens=jnp.full(
                    (total_vpages,), 2048, dtype=jnp.int32
                ),
                num_active_vpages=jnp.asarray(active_vpages, dtype=jnp.int32),
                q_seg_ids=jnp.zeros((128,), dtype=jnp.int32),
                q_sequence=jnp.arange(128, dtype=jnp.int32),
                query_row_valid=jnp.ones((128,), dtype=bool),
                needs_reroute=jnp.asarray(False),
            )
            out = pcp_kernel._attend_over_vpage_groups(
                jnp.zeros((128, 256, 1, 4, 256), dtype=jnp.float8_e4m3fn),
                jnp.zeros((32, 128, 256), dtype=jnp.bfloat16),
                plan,
                splash_cfg=cfg,
                vpage_group_size=128,
                kv_heads=2,
                head_dim=256,
                page_size=256,
                pcp_size=8,
                interleave_size=256,
                kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                pcp_axis_name="pcp",
            )
            assert float(out[0, 0, 0]) == float(expected_rung)


def test_pcp_streaming_attention_selects_block_kv_1024_for_aligned_vpages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_block_kv: list[int] = []
    monkeypatch.setattr(
        pcp_kernel,
        "_route_current_kv_to_captured_kv",
        lambda kv_in, *_args, **_kwargs: kv_in,
    )
    monkeypatch.setattr(
        pcp_kernel,
        "write_captured_kv_to_local_cache",
        lambda cache, *_args, **_kwargs: cache,
    )

    def _fake_attend(
        _kv_cache_local, q_scaled_hm, _stage_plan, *, splash_cfg, **_kwargs
    ):
        seen_block_kv.append(int(splash_cfg.block_kv))
        return jnp.zeros_like(q_scaled_hm, dtype=jnp.float32)

    monkeypatch.setattr(pcp_kernel, "_attend_over_vpage_groups", _fake_attend)
    stage_plan = SimpleNamespace(
        needs_reroute=jnp.asarray(False),
        q_seg_ids=jnp.zeros((128,), dtype=jnp.int32),
        block_tables=None,
    )
    for page_size, expected_bkv in ((256, 1024), (128, 1024), (64, 512)):
        seen_block_kv.clear()
        pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
            jnp.zeros((128, 2, 16, 256), dtype=jnp.bfloat16),
            jnp.zeros((128, 1, 4, 256), dtype=jnp.float8_e4m3fn),
            jnp.zeros((4, page_size, 1, 4, 256), dtype=jnp.float8_e4m3fn),
            jnp.asarray([128], dtype=jnp.int32),
            jnp.asarray([0, 128], dtype=jnp.int32),
            jnp.asarray([0, 0, 1], dtype=jnp.int32),
            jnp.zeros((3, 8), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            k_scale=1.0,
            v_scale=1.0,
            stage_plan=stage_plan,
            vpage_group_size=128,
            pcp_size=8,
            interleave_size=page_size,
        )
        assert seen_block_kv == [expected_bkv]


def test_ab_bench_helpers_and_tpu7x_compile_only_rung_96(tmp_path: Path) -> None:
    bench_mod = _load_pcp8_bench_module()

    expected_vpages = {
        "qwen35_160k_80vp": 80,
        "qwen38_p2_8vp": 8,
        "qwen38_p1_40vp": 40,
        "qwen38_p3b_124vp": 124,
        "qwen38_p4_508vp": 508,
    }
    for wl_name, spec in bench_mod.AB_WORKLOAD_SPECS.items():
        requests = spec["requests"]
        local_q = int(spec["local_query_tokens"])
        interleave = int(spec["interleave_size"])
        meta = bench_mod._build_pcp8_workload_host_metadata(
            requests,
            max_num_seqs=bench_mod.MAX_NUM_SEQS,
            max_local_pages_per_seq=int(spec["max_local_pages_per_seq"]),
            page_size=int(spec["page_size"]),
            interleave_size=interleave,
            local_query_tokens=local_q,
            pcp_size=bench_mod.WORLD_SIZE,
        )
        assert meta["active_vpages"] == expected_vpages[wl_name]
        q_lens = jnp.asarray(
            [q for q, _ in requests] + [0] * (bench_mod.MAX_NUM_SEQS - len(requests)),
            dtype=jnp.int32,
        )
        owner_starts = jnp.asarray(meta["query_start_loc"][:-1], dtype=jnp.int32)
        hist_starts = jnp.asarray(
            [h for _, h in requests] + [0] * (bench_mod.MAX_NUM_SEQS - len(requests)),
            dtype=jnp.int32,
        )
        for rank in (0, 3, 7):
            mapping = pcp_stage_plan._compute_pcp_local_query_row_mapping(
                q_lens,
                token_owner_starts=owner_starts,
                request_absolute_starts=hist_starts,
                local_padded_tokens=local_q,
                pcp_size=bench_mod.WORLD_SIZE,
                pcp_rank=rank,
                interleave_size=interleave,
            )
            valid = np.asarray(mapping.is_valid, dtype=bool)
            np.testing.assert_array_equal(valid, meta["valid_mask_by_rank"][rank])
            req_ids = np.clip(np.asarray(mapping.request_indices), 0, len(requests) - 1)
            abs_pos = np.asarray(mapping.absolute_positions, dtype=np.float64)
            hist_np = np.asarray([h for _, h in requests], dtype=np.float64)[req_ids]
            v_scales = np.asarray(
                [bench_mod._request_value_scale(i) for i in range(len(requests))],
                dtype=np.float64,
            )[req_ids]
            q_off = np.where(valid, abs_pos - hist_np, 0.0)
            denom = np.where(valid, abs_pos + 1.0, 1.0)
            ref = np.where(valid, v_scales * (q_off + 1.0) / denom, 0.0).astype(
                np.float32
            )
            np.testing.assert_allclose(
                meta["expected_by_rank"][rank], ref, rtol=0.0, atol=1e-7
            )

    # Incremental per-workload flush aggregation check (complete + interrupted runs).
    flush_dir = tmp_path / "ab_flush"
    flush_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(2):
        (flush_dir / f"ab_rank_{rank}_wl_a.json").write_text(
            json.dumps({"rank": rank, "workload": "wl_a", "active_vpages": 80}),
            encoding="utf-8",
        )
    by_wl_partial, errs_partial = bench_mod._collect_ab_workload_payloads(
        flush_dir, ("wl_a", "wl_b"), world_size=2
    )
    assert len(by_wl_partial["wl_a"]) == 2
    assert len(errs_partial) == 2 and all(e["workload"] == "wl_b" for e in errs_partial)
    for rank in range(2):
        (flush_dir / f"ab_rank_{rank}_wl_b.json").write_text(
            json.dumps({"rank": rank, "workload": "wl_b", "active_vpages": 8}),
            encoding="utf-8",
        )
    by_wl_full, errs_full = bench_mod._collect_ab_workload_payloads(
        flush_dir, ("wl_a", "wl_b"), world_size=2
    )
    assert not errs_full and len(by_wl_full["wl_b"]) == 2

    # Negative control: dropping history tokens on Qwen3.5 160K exceeds AB_PARITY_ATOL.
    q35_meta = bench_mod._build_pcp8_workload_host_metadata(
        ((32768, 131072),),
        max_num_seqs=64,
        max_local_pages_per_seq=128,
        page_size=256,
        interleave_size=256,
        local_query_tokens=4096,
    )
    corrupted_meta = bench_mod._build_pcp8_workload_host_metadata(
        ((32768, 131072 - 1024),),
        max_num_seqs=64,
        max_local_pages_per_seq=128,
        page_size=256,
        interleave_size=256,
        local_query_tokens=4096,
    )
    valid = q35_meta["valid_mask_by_rank"]
    exp_v = q35_meta["expected_by_rank"][valid]
    cor_v = corrupted_meta["expected_by_rank"][valid]
    assert np.all(np.abs(exp_v - cor_v) > bench_mod.AB_PARITY_RTOL * np.abs(exp_v))

    ok_fast, t_fast, _ = bench_mod._one_sided_welch_non_regression(
        [42.70, 42.68, 42.75, 42.71, 42.85],
        [38.10, 38.05, 38.12, 38.08, 38.15],
    )
    assert ok_fast and t_fast < 0.0
    ok_reg, t_reg, t_crit_reg = bench_mod._one_sided_welch_non_regression(
        [5.10, 5.12, 5.11, 5.09, 5.13],
        [5.80, 5.82, 5.79, 5.85, 5.81],
    )
    assert (not ok_reg) and t_reg > t_crit_reg

    # Compile-only tpu7x:2x2x1 check of base override and candidate rung-96.
    if jax.default_backend() == "tpu" and len(jax.devices()) >= 8:
        tpu_devices = np.asarray(jax.devices()[:8])
    else:
        topo = topologies.get_topology_desc(topology_name="tpu7x:2x2x1", platform="tpu")
        tpu_devices = np.asarray(topo.devices).reshape(-1)[:8]
    mesh = Mesh(tpu_devices, ("pcp",))

    def _compile_q35_hlo() -> str:
        @jax.shard_map(
            mesh=mesh,
            in_specs=(
                P("pcp", None, None, None),
                P("pcp", None, None, None),
                P("pcp", None, None, None, None, None),
                P(None),
                P(None),
                P(None),
                P(None),
            ),
            out_specs=(
                P("pcp", None, None, None),
                P("pcp", None, None, None, None, None),
            ),
            check_vma=False,
        )
        def _q35_rung96_fn(q_sh, kv_sh, cache_sh, kv_lens, pages, cu_q, dist):
            cache_loc = jnp.squeeze(cache_sh, axis=0)
            plan = pcp_stage_plan.build_pcp_local_stage_plan(
                kv_lens,
                pages,
                cu_q,
                dist,
                local_tokens=128,
                local_kv_cache_num_blocks=128,
                page_size=256,
                pcp_size=8,
                interleave_size=256,
                vpage_group_size=128,
                kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                pcp_axis_name="pcp",
            )
            out_sh, cache_out = (
                pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                    q_sh,
                    kv_sh,
                    cache_loc,
                    kv_lens,
                    cu_q,
                    dist,
                    plan.segment_descriptors,
                    plan.num_segments,
                    pcp_size=8,
                    interleave_size=256,
                    sm_scale=1.0,
                    k_scale=1.0,
                    v_scale=1.0,
                    kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                    mesh_axis_names=("pcp",),
                    pcp_axis_name="pcp",
                    stage_plan=plan,
                    vpage_group_size=128,
                )
            )
            return out_sh, jnp.expand_dims(cache_out, axis=0)

        return (
            jax.jit(_q35_rung96_fn, donate_argnums=(2,))
            .lower(
                jax.ShapeDtypeStruct((8 * 128, 2, 16, 256), jnp.bfloat16),
                jax.ShapeDtypeStruct((8 * 128, 1, 4, 256), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8, 128, 256, 1, 4, 256), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((4,), jnp.int32),
                jax.ShapeDtypeStruct((4, 128), jnp.int32),
                jax.ShapeDtypeStruct((5,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            )
            .compile()
            .as_text()
        )

    orig_sp_rungs = pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS
    orig_k_rungs = pcp_kernel.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS
    orig_cfg_cls = pcp_kernel._splash_kernel.SplashConfig
    try:
        with bench_mod._override_pcp_splash_arm(
            pcp_stage_plan, pcp_kernel, (32, 64), force_bkv_512=True
        ):
            raise RuntimeError("verify finally restore")
    except RuntimeError:
        pass
    assert tuple(pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS) == orig_sp_rungs
    assert tuple(pcp_kernel.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS) == orig_k_rungs
    assert pcp_kernel._splash_kernel.SplashConfig is orig_cfg_cls

    with bench_mod._override_pcp_splash_arm(
        pcp_stage_plan,
        pcp_kernel,
        (32, 64, 128, 256, 512),
        force_bkv_512=True,
    ):
        base_hlo_txt = _compile_q35_hlo()
    assert tuple(pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS) == orig_sp_rungs
    assert tuple(pcp_kernel.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS) == orig_k_rungs
    assert pcp_kernel._splash_kernel.SplashConfig is orig_cfg_cls
    assert "f8e4m3fn[4,128,1,2048,256]" in base_hlo_txt
    assert ".active_cols_ref = s16[2048]" in base_hlo_txt
    assert "f8e4m3fn[4,96,1,2048,256]" not in base_hlo_txt
    assert '\\"block_kv\\": 512' in base_hlo_txt
    assert '\\"block_kv\\": 1024' not in base_hlo_txt

    hlo_txt = _compile_q35_hlo()
    assert "{1}: (2, {}, may-alias)" in hlo_txt
    assert "f8e4m3fn[4,96,1,2048,256]" in hlo_txt
    assert ".active_cols_ref = s16[64]" in hlo_txt
    assert ".active_cols_ref = s16[192]" in hlo_txt
    assert ".active_cols_ref = s16[2048]" not in hlo_txt
    assert '\\"block_kv\\": 1024' in hlo_txt
    assert '\\"block_kv\\": 512' not in hlo_txt
    hlo_lines = hlo_txt.splitlines()
    entry_idx = next(i for i, ln in enumerate(hlo_lines) if ln.startswith("ENTRY "))
    entry_copies = [
        ln.strip()
        for ln in hlo_lines[entry_idx:]
        if " copy(" in ln and "f8e4m3fn[128,256," in ln
    ]
    assert entry_copies == [], f"Unexpected ENTRY 5D KV-cache copies: {entry_copies}"
    base_5d_copies = [
        ln.strip()
        for ln in base_hlo_txt.splitlines()
        if " copy(" in ln and "f8e4m3fn[128,256," in ln
    ]
    cand_5d_copies = [
        ln.strip() for ln in hlo_lines if " copy(" in ln and "f8e4m3fn[128,256," in ln
    ]
    assert len(cand_5d_copies) <= len(base_5d_copies) and len(cand_5d_copies) <= 2, (
        f"Unexpected 5D KV-cache copies: cand={cand_5d_copies} vs base={base_5d_copies}"
    )

    # Verify _build_arm_attention registers annotated custom ops and exports
    # TPU StableHLO across Qwen3.5 (2KV, p=256) and Qwen3.8 (4KV, p=128).
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
        vllm_adapter as pcp_vllm_adapter,
    )

    ab_mesh = Mesh(
        tpu_devices.reshape(8, 1),
        (pcp_wrapper.PCP_AXIS_NAME, pcp_wrapper.TP_AXIS_NAME),
    )
    captured_callables: list[object] = []

    def _capturing_jax_op(name, fn, **kwargs):
        captured_callables.append(
            pcp_vllm_adapter.build_pcp_streaming_callable(name, fn, **kwargs)
        )
        return pcp_vllm_adapter.pcp_streaming_jax_op(name, fn, **kwargs)

    mlir_by_tag: dict[str, str] = {}
    for tag, wl_name, rungs_ov, force_512 in (
        ("q35_b512", "qwen35_160k_80vp", (32, 64, 128, 256, 512), True),
        ("q35_c1024", "qwen35_160k_80vp", (32, 64, 96, 128, 256, 512), False),
        ("q38_c1024", "qwen38_p4_508vp", (32, 64, 96, 128, 256, 512), False),
    ):
        spec = bench_mod.AB_WORKLOAD_SPECS[wl_name]
        nq = int(spec["num_q_heads"])
        nkv = int(spec["num_kv_heads"])
        ps = int(spec["page_size"])
        ilv = int(spec["interleave_size"])
        lq = int(spec["local_query_tokens"])
        mp = int(spec["max_local_pages_per_seq"])
        pg = math.ceil(2 * nkv / bench_mod.KV_PACKING)
        arm_op = bench_mod._build_arm_attention(
            jax,
            pcp_stage_plan,
            pcp_kernel,
            pcp_vllm_adapter.make_pcp_streaming_rpa_kernel,
            _capturing_jax_op,
            pcp_vllm_adapter.PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            mesh=ab_mesh,
            kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
            op_tag=f"unit_{tag}",
            interleave_size=ilv,
            rungs_override=rungs_ov,
            force_bkv_512=force_512,
        )
        assert callable(arm_op)
        jc = captured_callables[-1]
        t_cache = torch.empty(
            (mp, ps, pg, bench_mod.KV_PACKING, bench_mod.HEAD_DIM),
            dtype=torch.float8_e4m3fn,
        )
        t_q = torch.empty((lq, nq, bench_mod.HEAD_DIM), dtype=torch.bfloat16)
        t_k = torch.empty((lq, nkv, bench_mod.HEAD_DIM), dtype=torch.bfloat16)
        t_v = torch.empty((lq, nkv, bench_mod.HEAD_DIM), dtype=torch.bfloat16)
        t_sl = torch.empty((bench_mod.MAX_NUM_SEQS,), dtype=torch.int32)
        t_bt = torch.empty((bench_mod.MAX_NUM_SEQS, mp), dtype=torch.int32)
        t_qsl = torch.empty((bench_mod.MAX_NUM_SEQS + 1,), dtype=torch.int32)
        t_dist = torch.empty((3,), dtype=torch.int32)
        jax_args = pallas_impl.jax_placeholders(
            (t_cache, t_q, t_k, t_v, t_sl, t_bt, t_qsl, t_dist),
            mesh=ab_mesh,
            partition_specs=pcp_vllm_adapter.PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
        )
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = jc.exported(*jax_args)
        local_shapes = [
            pallas_impl.get_local_shape(aval.shape, ab_mesh, out_spec)
            for aval, out_spec in zip(
                lowered.out_avals,
                pcp_vllm_adapter.PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS,
            )
        ]
        assert local_shapes == [tuple(t_cache.shape), tuple(t_q.shape)]
        mlir_by_tag[tag] = str(lowered.mlir_module())
        assert (
            tuple(pcp_stage_plan.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS) == orig_sp_rungs
        )
        assert tuple(pcp_kernel.PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS) == orig_k_rungs
        assert pcp_kernel._splash_kernel.SplashConfig is orig_cfg_cls

    assert mlir_by_tag["q35_b512"] != mlir_by_tag["q35_c1024"]
    assert "4x96x1x2048x256xf8E4M3FN" not in mlir_by_tag["q35_b512"]
    assert "4x96x1x2048x256xf8E4M3FN" in mlir_by_tag["q35_c1024"]
