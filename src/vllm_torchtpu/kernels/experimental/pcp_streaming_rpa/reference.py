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
"""Reference implementation for PCP streaming prefill RPA schedules."""

import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    PcpStreamingSchedule, validate_pcp_streaming_schedule)


def execute_pcp_streaming_reference(
    q_by_rank: np.ndarray,
    kv_cache_by_rank: np.ndarray,
    schedule: PcpStreamingSchedule,
    *,
    sm_scale: float,
    interleave_size: int | None = None,
) -> np.ndarray:
    """Run the PCP streaming schedule with NumPy online softmax.

    Args:
        q_by_rank: [pcp_size, local_padded_tokens, kv_heads, q_per_kv, head_dim].
        kv_cache_by_rank: [pcp_size, pages, page_size, kv_heads, 2, head_dim].
        schedule: Host-side PCP streaming schedule.
        sm_scale: Attention softmax scale.
        interleave_size: Number of consecutive global tokens assigned to one
            PCP rank before rotating to the next rank. Defaults to page_size.

    Returns:
        Rank-local packed output with the same shape as q_by_rank.
    """
    validate_pcp_streaming_schedule(schedule)

    pcp_size, _, kv_heads, q_per_kv, head_dim = q_by_rank.shape
    page_size = kv_cache_by_rank.shape[2]
    if interleave_size is None:
        interleave_size = page_size
    if pcp_size != schedule.pcp_size:
        raise ValueError("q_by_rank first dimension must match schedule.")
    output = np.zeros_like(q_by_rank, dtype=np.float32)

    for consumer_rank in range(schedule.pcp_size):
        q_tiles = [None] * schedule.num_lanes
        m_states = [None] * schedule.num_lanes
        l_states = [None] * schedule.num_lanes
        acc_states = [None] * schedule.num_lanes
        q_tile_sizes = [0] * schedule.num_lanes
        for step in range(int(schedule.actual_steps[consumer_rank])):
            for lane in range(schedule.num_lanes):
                req_id = int(schedule.req_id[consumer_rank, step, lane])
                if req_id == -1:
                    continue

                if schedule.load_q[consumer_rank, step, lane]:
                    q_offset = int(schedule.q_hbm_offset[consumer_rank, step,
                                                         lane])
                    q_tile_size = int(schedule.q_tile_size[consumer_rank, step,
                                                           lane])
                    q_tile_sizes[lane] = q_tile_size
                    q_tiles[lane] = q_by_rank[
                        consumer_rank,
                        q_offset:q_offset + q_tile_size,
                    ].astype(np.float32)

                if schedule.is_first_kv[consumer_rank, step, lane]:
                    q_tile_size = q_tile_sizes[lane]
                    m_states[lane] = np.full((q_tile_size, kv_heads, q_per_kv),
                                             -np.inf,
                                             dtype=np.float32)
                    l_states[lane] = np.zeros(
                        (q_tile_size, kv_heads, q_per_kv), dtype=np.float32)
                    acc_states[lane] = np.zeros(
                        (q_tile_size, kv_heads, q_per_kv, head_dim),
                        dtype=np.float32)

                q_tile = q_tiles[lane]
                m = m_states[lane]
                l_state = l_states[lane]
                acc = acc_states[lane]
                if (q_tile is None or m is None or l_state is None
                        or acc is None):
                    raise ValueError(
                        "schedule entry used before Q load/reset.")

                src_rank = int(schedule.kv_page_rank[consumer_rank, step,
                                                     lane])
                page_idx = int(schedule.kv_page_idx[consumer_rank, step, lane])
                kv_valid_len = int(schedule.kv_valid_len[consumer_rank, step,
                                                         lane])
                kv_global_start = int(schedule.kv_global_start[consumer_rank,
                                                               step, lane])
                q_global_start = int(schedule.q_global_start[consumer_rank,
                                                             step, lane])
                if schedule.kv_page_indices is None:
                    k = kv_cache_by_rank[src_rank, page_idx, :kv_valid_len, :,
                                         0, :].astype(np.float32)
                    v = kv_cache_by_rank[src_rank, page_idx, :kv_valid_len, :,
                                         1, :].astype(np.float32)
                    local_kv_pos = np.arange(kv_valid_len)
                    kv_pos = (kv_global_start +
                              (local_kv_pos // interleave_size) *
                              schedule.pcp_size * interleave_size +
                              local_kv_pos % interleave_size)
                else:
                    page_ids = schedule.kv_page_indices[consumer_rank, step,
                                                        lane]
                    k_pages = []
                    v_pages = []
                    remaining = kv_valid_len
                    for page_id in page_ids:
                        if remaining <= 0:
                            break
                        take = min(page_size, remaining)
                        k_pages.append(kv_cache_by_rank[src_rank,
                                                        page_id, :take, :,
                                                        0, :])
                        v_pages.append(kv_cache_by_rank[src_rank,
                                                        page_id, :take, :,
                                                        1, :])
                        remaining -= take
                    k = np.concatenate(k_pages, axis=0).astype(np.float32)
                    v = np.concatenate(v_pages, axis=0).astype(np.float32)
                    local_kv_pos = np.arange(kv_valid_len)
                    kv_page_offset = local_kv_pos // page_size
                    kv_token_offset = local_kv_pos % page_size
                    kv_pos = (kv_global_start +
                              kv_page_offset * schedule.pcp_size * page_size +
                              (kv_token_offset // interleave_size) *
                              schedule.pcp_size * interleave_size +
                              kv_token_offset % interleave_size)

                scores = np.einsum("thqd,shd->thqs", q_tile, k) * sm_scale
                q_rows = np.arange(q_tile_size)
                q_pos = (q_global_start + (q_rows // interleave_size) *
                         schedule.pcp_size * interleave_size +
                         q_rows % interleave_size)
                mask = q_pos[:, None] >= kv_pos[None, :]
                scores = np.where(mask[:, None, None, :], scores, -np.inf)

                m_curr = np.max(scores, axis=-1)
                m_next = np.maximum(m, m_curr)
                p = np.exp(scores - m_next[..., None])
                alpha = np.exp(m - m_next)
                l_state = alpha * l_state + np.sum(p, axis=-1)
                acc = alpha[..., None] * acc + np.einsum(
                    "thqs,shd->thqd", p, v)
                m_states[lane] = m_next
                l_states[lane] = l_state
                acc_states[lane] = acc

                if schedule.is_last_kv[consumer_rank, step, lane]:
                    out_offset = int(schedule.o_hbm_offset[consumer_rank, step,
                                                           lane])
                    output[consumer_rank, out_offset:out_offset +
                           q_tile_size] = acc / l_state[..., None]
                    q_tiles[lane] = None
                    m_states[lane] = None
                    l_states[lane] = None
                    acc_states[lane] = None

    return output
