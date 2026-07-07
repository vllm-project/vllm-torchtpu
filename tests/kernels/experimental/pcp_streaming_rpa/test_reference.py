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

from schedule_reference import \
    generate_pcp_streaming_schedule_reference  # noqa: E402

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_rank_major_token_order
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.reference import \
    execute_pcp_streaming_reference


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


def _naive_full_attention(q_full, k_full, v_full, q_global_base, sm_scale):
    q_len, kv_heads, q_per_kv, head_dim = q_full.shape
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
