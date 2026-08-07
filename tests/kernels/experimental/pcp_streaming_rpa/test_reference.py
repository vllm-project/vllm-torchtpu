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

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import pytest
from schedule_reference import generate_pcp_streaming_schedule_reference

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.reference import (
    execute_pcp_streaming_reference, )
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    _generate_pcp_streaming_schedule_vectorized_aligned, )

pytestmark = pytest.mark.multichip


def _pack_q_rank_major(q_full, token_order, pcp_size):
    padded = np.zeros((token_order.size, ) + q_full.shape[1:],
                      dtype=np.float32)
    valid = token_order >= 0
    padded[valid] = q_full[token_order[valid]]
    local_padded = token_order.size // pcp_size
    return padded.reshape((pcp_size, local_padded) + q_full.shape[1:])


def _build_pcp_kv_cache(k_full, v_full, block_tables, page_size, pcp_size):
    num_pages = int(block_tables.max()) + 1
    _, kv_heads, head_dim = k_full.shape
    kv_cache = np.zeros(
        (pcp_size, num_pages, page_size, kv_heads, 2, head_dim),
        dtype=np.float32,
    )
    for pos in range(k_full.shape[0]):
        global_page = pos // page_size
        src_rank = global_page % pcp_size
        local_page_index = global_page // pcp_size
        page = int(block_tables[0, local_page_index])
        offset = pos % page_size
        kv_cache[src_rank, page, offset, :, 0, :] = k_full[pos]
        kv_cache[src_rank, page, offset, :, 1, :] = v_full[pos]
    return kv_cache


def _build_unique_block_tables(kv_lens, page_size, pcp_size):
    local_pages_per_req = [
        (kv_len + page_size * pcp_size - 1) // (page_size * pcp_size)
        for kv_len in kv_lens
    ]
    max_local_pages = max(local_pages_per_req)
    block_tables = np.zeros((len(kv_lens), max_local_pages), dtype=np.int32)
    next_page = 0
    for req_idx, local_pages in enumerate(local_pages_per_req):
        block_tables[req_idx, :local_pages] = np.arange(next_page,
                                                        next_page +
                                                        local_pages,
                                                        dtype=np.int32)
        next_page += local_pages
    return block_tables, next_page


def _build_pcp_kv_cache_for_requests(k_by_req, v_by_req, block_tables,
                                     page_size, pcp_size, interleave_size,
                                     num_cache_pages):
    kv_heads = k_by_req[0].shape[1]
    head_dim = k_by_req[0].shape[2]
    kv_cache = np.zeros(
        (pcp_size, num_cache_pages, page_size, kv_heads, 2, head_dim),
        dtype=np.float32,
    )
    virtual_page_size = pcp_size * page_size
    for req_idx, (k_full, v_full) in enumerate(zip(k_by_req, v_by_req)):
        for absolute_pos in range(k_full.shape[0]):
            local_page_index = absolute_pos // virtual_page_size
            position_in_virtual_page = absolute_pos % virtual_page_size
            chunk_index = position_in_virtual_page // interleave_size
            src_rank = chunk_index % pcp_size
            local_chunk_index = chunk_index // pcp_size
            local_offset = (local_chunk_index * interleave_size +
                            absolute_pos % interleave_size)
            page = int(block_tables[req_idx, local_page_index])
            kv_cache[src_rank, page, local_offset, :,
                     0, :] = k_full[absolute_pos]
            kv_cache[src_rank, page, local_offset, :,
                     1, :] = v_full[absolute_pos]
    return kv_cache


def _naive_full_attention(q_full, k_full, v_full, q_global_base, sm_scale):
    q_len = q_full.shape[0]
    out = np.zeros_like(q_full, dtype=np.float32)
    for q_idx in range(q_len):
        q_global = q_global_base + q_idx
        k_ctx = k_full[:q_global + 1]
        v_ctx = v_full[:q_global + 1]
        scores = np.einsum("hqd,shd->hqs", q_full[q_idx], k_ctx) * sm_scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        out[q_idx] = np.einsum("hqs,shd->hqd", probs, v_ctx)
    return out


def _assert_schedule_q_rows_match_packed_absolute_positions(
        schedule, token_order, q_lens, request_absolute_starts,
        interleave_size):
    pcp_size = schedule.pcp_size
    local_padded_tokens = token_order.size // pcp_size
    token_order_by_rank = token_order.reshape((pcp_size, local_padded_tokens))
    cu_q_lens = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(q_lens, dtype=np.int64)))

    for rank in range(pcp_size):
        expected_absolute_positions = np.full(local_padded_tokens,
                                              -1,
                                              dtype=np.int64)
        valid_rows = token_order_by_rank[rank] >= 0
        for local_row, request_major_row in enumerate(
                token_order_by_rank[rank]):
            if request_major_row < 0:
                continue
            req_idx = int(
                np.searchsorted(cu_q_lens[1:], request_major_row,
                                side="right"))
            request_relative_row = request_major_row - cu_q_lens[req_idx]
            expected_absolute_positions[local_row] = (
                request_absolute_starts[req_idx] + request_relative_row)

        visited = np.zeros(local_padded_tokens, dtype=bool)
        for step in range(int(schedule.actual_steps[rank])):
            if not schedule.load_q[rank, step, 0]:
                continue
            q_hbm_offset = int(schedule.q_hbm_offset[rank, step, 0])
            q_tile_size = int(schedule.q_tile_size[rank, step, 0])
            q_global_start = int(schedule.q_global_start[rank, step, 0])
            q_rows = np.arange(q_tile_size, dtype=np.int64)
            kernel_absolute_positions = (
                q_global_start +
                (q_rows // interleave_size) * pcp_size * interleave_size +
                q_rows % interleave_size)
            row_slice = slice(q_hbm_offset, q_hbm_offset + q_tile_size)
            np.testing.assert_array_equal(
                kernel_absolute_positions,
                expected_absolute_positions[row_slice],
            )
            assert not np.any(visited[row_slice])
            visited[row_slice] = True

        np.testing.assert_array_equal(visited, valid_rows)


def _run_dual_coordinate_reference_case(q_lens, request_absolute_starts, *,
                                        page_size, interleave_size,
                                        q_block_size):
    pcp_size = 4
    kv_heads = 2
    q_per_kv = 2
    head_dim = 4
    q_lens = np.asarray(q_lens, dtype=np.int64)
    request_absolute_starts = np.asarray(request_absolute_starts,
                                         dtype=np.int64)
    token_owner_starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(q_lens[:-1], dtype=np.int64)))
    kv_lens = request_absolute_starts + q_lens
    cu_q_lens = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(q_lens, dtype=np.int64)))
    assert np.all(token_owner_starts != request_absolute_starts)

    rng = np.random.default_rng(123)
    q_by_req = [
        rng.normal(size=(q_len, kv_heads, q_per_kv,
                         head_dim)).astype(np.float32) for q_len in q_lens
    ]
    k_by_req = [
        rng.normal(size=(kv_len, kv_heads, head_dim)).astype(np.float32)
        for kv_len in kv_lens
    ]
    v_by_req = [
        rng.normal(size=(kv_len, kv_heads, head_dim)).astype(np.float32)
        for kv_len in kv_lens
    ]
    q_full = np.concatenate(q_by_req, axis=0)

    local_token_counts = pcp_local_token_counts(
        q_lens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        token_owner_start_offsets_per_req=token_owner_starts,
    )
    local_padded_tokens = int(local_token_counts.max())
    padded_num_tokens = local_padded_tokens * pcp_size
    token_order, inverse_order = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=padded_num_tokens,
        token_owner_start_offsets_per_req=token_owner_starts,
    )
    q_by_rank = _pack_q_rank_major(q_full, token_order, pcp_size)

    block_tables, num_cache_pages = _build_unique_block_tables(
        kv_lens, page_size, pcp_size)
    kv_cache = _build_pcp_kv_cache_for_requests(
        k_by_req,
        v_by_req,
        block_tables,
        page_size,
        pcp_size,
        interleave_size,
        num_cache_pages,
    )
    schedule = _generate_pcp_streaming_schedule_vectorized_aligned(
        kv_lens=kv_lens,
        cu_q_lens=cu_q_lens,
        token_owner_start_offsets=token_owner_starts,
        request_absolute_query_start_offsets=request_absolute_starts,
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
    )
    _assert_schedule_q_rows_match_packed_absolute_positions(
        schedule,
        token_order,
        q_lens,
        request_absolute_starts,
        interleave_size,
    )

    sm_scale = 1.0 / np.sqrt(head_dim)
    packed_output = execute_pcp_streaming_reference(
        q_by_rank,
        kv_cache,
        schedule,
        sm_scale=sm_scale,
        interleave_size=interleave_size,
    )
    unpacked_output = packed_output.reshape((padded_num_tokens, ) +
                                            q_full.shape[1:])[inverse_order]
    expected = np.concatenate([
        _naive_full_attention(q, k, v, int(absolute_start), sm_scale)
        for q, k, v, absolute_start in zip(q_by_req, k_by_req, v_by_req,
                                           request_absolute_starts)
    ])
    np.testing.assert_allclose(unpacked_output, expected, rtol=1e-5, atol=1e-5)


def _run_reference_case(q_len, q_global_base, kv_len, num_lanes):
    pcp_size = 4
    page_size = interleave_size = 2
    kv_heads = 2
    q_per_kv = 2
    head_dim = 4
    rng = np.random.default_rng(123)
    q_full = rng.normal(size=(q_len, kv_heads, q_per_kv,
                              head_dim)).astype(np.float32)
    k_full = rng.normal(size=(kv_len, kv_heads, head_dim)).astype(np.float32)
    v_full = rng.normal(size=(kv_len, kv_heads, head_dim)).astype(np.float32)
    block_tables = np.arange(
        (kv_len + page_size * pcp_size - 1) // (page_size * pcp_size),
        dtype=np.int32).reshape(1, -1)

    padded_num_tokens = ((q_len + pcp_size - 1) // pcp_size) * pcp_size
    token_order, inverse_order = build_pcp_rank_major_token_order(
        [q_len],
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=padded_num_tokens,
        token_start_offsets_per_req=[q_global_base],
    )
    q_by_rank = _pack_q_rank_major(q_full, token_order, pcp_size)
    kv_cache = _build_pcp_kv_cache(k_full, v_full, block_tables, page_size,
                                   pcp_size)
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[q_global_base],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=2,
    )
    sm_scale = 1.0 / np.sqrt(head_dim)

    packed_output = execute_pcp_streaming_reference(q_by_rank,
                                                    kv_cache,
                                                    schedule,
                                                    sm_scale=sm_scale)
    unpacked_output = packed_output.reshape((padded_num_tokens, ) +
                                            q_full.shape[1:])[inverse_order]
    expected = _naive_full_attention(q_full, k_full, v_full, q_global_base,
                                     sm_scale)
    np.testing.assert_allclose(unpacked_output, expected, rtol=1e-5, atol=1e-5)


def test_reference_matches_naive_attention_for_single_lane_partial_chunk():
    _run_reference_case(q_len=7, q_global_base=5, kv_len=12, num_lanes=1)


def test_reference_matches_naive_attention_for_two_lanes():
    _run_reference_case(q_len=16, q_global_base=0, kv_len=16, num_lanes=2)


def test_batch_flat_owner_matches_naive_attention_for_short_multi_request():
    _run_dual_coordinate_reference_case(
        q_lens=[13, 11],
        request_absolute_starts=[20, 0],
        page_size=8,
        interleave_size=2,
        q_block_size=4,
    )


@pytest.mark.parametrize("history_tokens", [25_600, 128 * 1024])
def test_batch_flat_owner_matches_naive_attention_with_long_history(
        history_tokens):
    _run_dual_coordinate_reference_case(
        q_lens=[8],
        request_absolute_starts=[history_tokens],
        page_size=128,
        interleave_size=2,
        q_block_size=8,
    )
