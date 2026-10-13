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
"""Host-side PCP batch partitioning (`runner/pcp_partition.py`)."""

import itertools

import pytest

from vllm_torchtpu.runner.pcp_partition import (
    build_pcp_batch_plan,
    build_workspace_plan,
    pcp_rank_token_counts,
)


def _spans(segments):
    return [(s.req_idx, s.global_start, s.global_stop) for s in segments]


def _check_invariants(plan, num_scheduled_tokens):
    # Every gathered row that holds a token round-trips through the restore map.
    for g, row in enumerate(plan.hidden_restore_idx):
        assert plan.gathered_valid[row]
        assert plan.padded_gather_idx[row] == g
    # Split segments lead, replicated ones trail.
    for rank, segments in enumerate(plan.segments_by_rank):
        n = plan.num_split_segments(rank)
        assert all(not plan.replicated[s.req_idx] for s in segments[:n])
        assert all(plan.replicated[s.req_idx] for s in segments[n:])
    # Local slices tile each rank's prefix without gaps or overlap.
    for rank, segments in enumerate(plan.segments_by_rank):
        cursor = 0
        for seg in segments:
            assert seg.local_start == cursor
            assert seg.local_stop - seg.local_start == seg.num_tokens
            cursor = seg.local_stop
        assert cursor == plan.per_rank_num_tokens[rank]
        assert cursor <= plan.num_local_tokens_padded
    # A split request's tokens appear exactly once across ranks; a replicated
    # one appears whole on every rank.
    starts = list(itertools.accumulate([0] + list(num_scheduled_tokens)))
    for req_idx, q_len in enumerate(num_scheduled_tokens):
        owned = [
            t
            for segments in plan.segments_by_rank
            for s in segments
            if s.req_idx == req_idx
            for t in range(s.global_start, s.global_stop)
        ]
        full = list(range(starts[req_idx], starts[req_idx] + q_len))
        if plan.replicated[req_idx]:
            assert sorted(owned) == sorted(full * plan.pcp_size)
        else:
            assert sorted(owned) == full


def test_single_prefill_dual_chunk_swap():
    plan = build_pcp_batch_plan([16], [True], pcp_size=4)
    assert plan.replicated == [False]
    # 8 chunks of 2: rank r takes chunk r and chunk 7 - r.
    assert _spans(plan.local_segments(0)) == [(0, 0, 2), (0, 14, 16)]
    assert _spans(plan.local_segments(1)) == [(0, 2, 4), (0, 12, 14)]
    assert _spans(plan.local_segments(3)) == [(0, 6, 8), (0, 8, 10)]
    assert plan.per_rank_num_tokens == [4, 4, 4, 4]
    _check_invariants(plan, [16])


def test_decode_is_replicated_and_ordered_last():
    # A decode listed before a prefill still lands after the split segments.
    plan = build_pcp_batch_plan([1, 8], [False, True], pcp_size=2)
    assert plan.replicated == [True, False]
    assert _spans(plan.local_segments(0)) == [(1, 1, 3), (1, 7, 9), (0, 0, 1)]
    assert _spans(plan.local_segments(1)) == [(1, 3, 5), (1, 5, 7), (0, 0, 1)]
    assert plan.num_split_segments(0) == 2
    # The decode token restores from rank 0's copy (local row 4).
    assert plan.hidden_restore_idx[0] == 4
    _check_invariants(plan, [1, 8])


def test_short_prefill_is_split():
    plan = build_pcp_batch_plan([3], [True], pcp_size=2)
    # 4 chunks of 1: rank 0 takes chunk 0 (chunk 3 is empty), rank 1 chunks 1, 2.
    assert _spans(plan.local_segments(0)) == [(0, 0, 1)]
    assert _spans(plan.local_segments(1)) == [(0, 1, 2), (0, 2, 3)]
    _check_invariants(plan, [3])


def test_only_decodes_replicate():
    plan = build_pcp_batch_plan([3, 1, 63], [True, False, True], pcp_size=4)
    assert plan.replicated == [False, True, False]
    for rank in range(4):
        segments = plan.local_segments(rank)
        tail = segments[plan.num_split_segments(rank):]
        assert [s.req_idx for s in tail] == [1]
    _check_invariants(plan, [3, 1, 63])


def test_uneven_prefill_and_padding():
    plan = build_pcp_batch_plan(
        [10, 1, 7], [True, False, True], pcp_size=2, num_local_tokens_padded=16,
    )
    assert plan.num_local_tokens_padded == 16
    assert len(plan.padded_gather_idx) == 32
    # Padding rows are flagged invalid and never referenced by the restore map.
    referenced = set(plan.hidden_restore_idx)
    for row, valid in enumerate(plan.gathered_valid):
        if not valid:
            assert row not in referenced
    _check_invariants(plan, [10, 1, 7])


def test_padding_smaller_than_largest_rank_rejected():
    with pytest.raises(ValueError):
        build_pcp_batch_plan([16], [True], pcp_size=4, num_local_tokens_padded=3)


@pytest.mark.parametrize("pcp_size", [1, 2, 4])
def test_random_batches_keep_invariants(pcp_size):
    cases = [
        ([1, 1, 1, 1], [False] * 4),
        ([4096], [True]),
        ([2048, 2048], [True, True]),
        ([5, 1, 300, 1, 17], [True, False, True, False, True]),
        ([4, 4], [False, False]),  # speculative verifies stay replicated
    ]
    for q_lens, prefilling in cases:
        plan = build_pcp_batch_plan(q_lens, prefilling, pcp_size)
        _check_invariants(plan, q_lens)
        assert (
            pcp_rank_token_counts(q_lens, prefilling, pcp_size)
            == plan.per_rank_num_tokens
        )


def test_workspace_plan_lays_requests_back_to_back():
    # virtual page = 64 tokens; req 1 is a decode and is left out.
    plan = build_workspace_plan([0, 2], [100, 7, 64], virtual_page_size=64,
                                num_pages=4)
    assert plan.page_offset == {0: 0, 2: 2}
    assert plan.ws_req == [0, 0, 2, 0]
    assert plan.ws_vpage == [0, 1, 0, 0]
    assert plan.num_used_pages == 3


def test_workspace_plan_overflow_rejected():
    with pytest.raises(ValueError):
        build_workspace_plan([0], [129], virtual_page_size=64, num_pages=2)
