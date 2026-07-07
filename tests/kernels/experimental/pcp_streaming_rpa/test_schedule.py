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

import dataclasses
import functools
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from schedule_reference import \
    generate_pcp_streaming_schedule_reference  # noqa: E402

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    PcpStreamingSchedule, ScheduleField,
    build_pcp_streaming_active_page_groups,
    build_pcp_streaming_schedule_inputs_from_metadata_host,
    build_pcp_streaming_schedule_inputs_from_metadata_jax,
    estimate_pcp_streaming_metadata_schedule_steps_ub,
    estimate_pcp_streaming_schedule_steps_ub, generate_pcp_streaming_schedule,
    generate_pcp_streaming_schedule_from_metadata_host,
    unpack_pcp_streaming_schedule_field, validate_pcp_streaming_schedule)


def _assert_schedule_equal(actual: PcpStreamingSchedule,
                           expected: PcpStreamingSchedule):
    for field in dataclasses.fields(PcpStreamingSchedule):
        name = field.name
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
        else:
            np.testing.assert_array_equal(actual_value,
                                          expected_value,
                                          err_msg=name)


def test_generate_schedule_uses_interleave_q_ownership_and_page_mapping():
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    assert schedule.req_id.shape == (4, 6, 1)
    np.testing.assert_array_equal(schedule.actual_steps,
                                  np.array([5, 6, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(schedule.global_actual_steps,
                                  np.array([6], dtype=np.int32))

    # PCP=4/interleave=2/q_global_base=5 assigns Q chunks:
    # rank0 [8,10), rank1 [10,12), rank2 [5,6), rank3 [6,8).
    np.testing.assert_array_equal(schedule.q_global_start[:, 0, 0],
                                  np.array([8, 10, 5, 6], dtype=np.int32))
    np.testing.assert_array_equal(schedule.q_tile_size[:, 0, 0],
                                  np.array([2, 2, 1, 2], dtype=np.int32))

    np.testing.assert_array_equal(schedule.kv_page_rank[0, :5, 0],
                                  np.array([0, 1, 2, 3, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        schedule.kv_page_idx[0, :5, 0],
        np.array([100, 100, 100, 100, 101], dtype=np.int32),
    )
    np.testing.assert_array_equal(schedule.kv_global_start[0, :5, 0],
                                  np.array([0, 2, 4, 6, 8], dtype=np.int32))
    np.testing.assert_array_equal(schedule.is_first_kv[0, :5, 0],
                                  np.array([1, 0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(schedule.is_last_kv[0, :5, 0],
                                  np.array([0, 0, 0, 0, 1], dtype=np.int32))


def test_schedule_q_offsets_follow_rank_major_packed_order_across_requests():
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[12, 3],
        cu_q_lens=[0, 7, 10],
        q_start_offsets=[5, 0],
        block_tables=np.array([[100, 101], [200, 0]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    # Consumer rank 0 first handles request 0 chunk [8,10) at local offset 0.
    assert schedule.req_id[0, 0, 0] == 0
    assert schedule.q_hbm_offset[0, 0, 0] == 0
    assert schedule.q_tile_size[0, 0, 0] == 2

    # Then request 1 chunk [0,2) is appended after rank 0's first two Q rows.
    assert schedule.req_id[0, 5, 0] == 1
    assert schedule.q_global_start[0, 5, 0] == 0
    assert schedule.q_hbm_offset[0, 5, 0] == 2
    assert schedule.o_hbm_offset[0, 5, 0] == 2


def test_validate_schedule_lane_invariant_accepts_generated_schedule():
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[16],
        cu_q_lens=[0, 16],
        q_start_offsets=[0],
        block_tables=np.array([[10, 11]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=2,
        bq_sz=2,
    )

    validate_pcp_streaming_schedule(schedule)


def test_generate_schedule_can_pad_kv_pages_to_pcp_groups():
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[10],
        cu_q_lens=[0, 2],
        q_start_offsets=[8],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
    )

    assert schedule.actual_steps[0] == 8
    assert schedule.global_actual_steps[0] == 8
    np.testing.assert_array_equal(schedule.req_id[0, :5, 0],
                                  np.zeros(5, dtype=np.int32))
    np.testing.assert_array_equal(schedule.req_id[0, 5:8, 0],
                                  np.full(3, -1, dtype=np.int32))
    np.testing.assert_array_equal(
        schedule.kv_page_rank[0, :8, 0],
        np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(
        schedule.kv_page_idx[0, :5, 0],
        np.array([100, 100, 100, 100, 101], dtype=np.int32))
    np.testing.assert_array_equal(
        schedule.is_first_kv[0, :8, 0],
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        schedule.is_last_kv[0, :8, 0],
        np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int32))
    validate_pcp_streaming_schedule(schedule)


def test_build_active_page_groups_from_padded_schedule():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[8],
        cu_q_lens=[0, 8],
        q_start_offsets=[0],
        block_tables=np.array([[100]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
    )

    np.testing.assert_array_equal(
        build_pcp_streaming_active_page_groups(schedule),
        np.array([1], dtype=np.int32),
    )


def test_build_active_page_groups_rejects_unpadded_schedule_steps():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[8],
        cu_q_lens=[0, 8],
        q_start_offsets=[0],
        block_tables=np.array([[100]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
    )
    schedule = dataclasses.replace(schedule,
                                   global_actual_steps=np.array(
                                       [5], dtype=np.int32))

    with pytest.raises(ValueError, match="PCP page group"):
        build_pcp_streaming_active_page_groups(schedule)


def test_generate_schedule_can_pad_steps_without_changing_actual_steps():
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[10],
        cu_q_lens=[0, 2],
        q_start_offsets=[8],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=12,
    )

    assert schedule.req_id.shape == (4, 12, 1)
    assert schedule.packed_schedule.shape == (
        12,
        4,
        1,
        ScheduleField.PACKED_NUM_FIELDS,
    )
    np.testing.assert_array_equal(schedule.actual_steps,
                                  np.array([8, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(schedule.global_actual_steps,
                                  np.array([8], dtype=np.int32))
    np.testing.assert_array_equal(schedule.req_id[:, 8:, 0],
                                  np.full((4, 4), -1, dtype=np.int32))


def test_generate_schedule_rejects_too_small_step_padding():
    with pytest.raises(ValueError, match="pad_steps_to"):
        generate_pcp_streaming_schedule(
            kv_lens=[16],
            cu_q_lens=[0, 16],
            q_start_offsets=[0],
            block_tables=np.array([[100, 101]], dtype=np.int32),
            page_size=2,
            pcp_size=4,
            interleave_size=2,
            num_lanes=1,
            bq_sz=2,
            pad_kv_pages_to_pcp_group=True,
            pad_steps_to=4,
        )


def test_schedule_packed_fields_match_unpacked_arrays():
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    assert schedule.packed_schedule.shape == (
        6,
        4,
        1,
        ScheduleField.PACKED_NUM_FIELDS,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.REQ_ID),
        schedule.req_id,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.KV_PAGE_RANK),
        schedule.kv_page_rank,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.KV_PAGE_IDX),
        schedule.kv_page_idx,
    )
    np.testing.assert_array_equal(
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.Q_GLOBAL_START),
        schedule.q_global_start,
    )
    np.testing.assert_array_equal(
        schedule.packed_schedule[0, :, 0, ScheduleField.Q_TILE_SIZE],
        schedule.q_tile_size[:, 0, 0],
    )


def test_schedule_packed_field_rejects_invalid_field_index():
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )

    with pytest.raises(ValueError, match="invalid schedule field"):
        unpack_pcp_streaming_schedule_field(schedule.packed_schedule,
                                            ScheduleField.NUM_FIELDS)


def test_validate_schedule_lane_invariant_rejects_q_offset_change_inside_tile(
):
    schedule = generate_pcp_streaming_schedule_reference(
        kv_lens=[12],
        cu_q_lens=[0, 7],
        q_start_offsets=[5],
        block_tables=np.array([[100, 101]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
    )
    q_hbm_offset = schedule.q_hbm_offset.copy()
    q_hbm_offset[0, 1, 0] = 99
    bad_schedule = dataclasses.replace(schedule, q_hbm_offset=q_hbm_offset)

    with pytest.raises(ValueError, match="q_hbm_offset changed"):
        validate_pcp_streaming_schedule(bad_schedule)


def test_validate_schedule_ring_source_invariant_rejects_mixed_source_page():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[16, 16],
        cu_q_lens=[0, 8, 16],
        q_start_offsets=[8, 8],
        block_tables=np.array([[100, 101], [200, 201]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
    )
    kv_page_idx = schedule.kv_page_idx.copy()
    kv_page_idx[1, 0, 0] = 999
    packed_schedule = schedule.packed_schedule.copy()
    packed_schedule[0, 1, 0, ScheduleField.KV_PAGE_IDX] = 999
    bad_schedule = dataclasses.replace(
        schedule,
        kv_page_idx=kv_page_idx,
        packed_schedule=packed_schedule,
    )

    with pytest.raises(ValueError, match="source page changed"):
        validate_pcp_streaming_schedule(
            bad_schedule,
            require_ring_source_invariant=True,
        )


def test_vectorized_schedule_matches_reference_for_aligned_qwen_shape():
    q_len = 4096
    q_start = 4096
    kv_len = q_start + q_len
    pcp_size = 8
    page_size = interleave_size = 32
    block_tables = np.arange(64, dtype=np.int32)[None, :]

    actual = generate_pcp_streaming_schedule(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[q_start],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=256,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=128,
        kv_pages_per_block=8,
    )
    expected = generate_pcp_streaming_schedule_reference(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[q_start],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=256,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=128,
        kv_pages_per_block=8,
    )

    _assert_schedule_equal(actual, expected)
    validate_pcp_streaming_schedule(actual)


def _metadata_schedule_pad_steps(*, global_bucket_tokens,
                                 local_kv_cache_num_blocks, page_size,
                                 pcp_size, interleave_size, q_block_size):
    return estimate_pcp_streaming_schedule_steps_ub(
        np.asarray([global_bucket_tokens], dtype=np.int64),
        capacity_tokens=local_kv_cache_num_blocks * pcp_size * page_size,
        block_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        q_block_size=q_block_size,
        kv_pages_per_block=1,
    )


def _metadata_lockstep_pad_steps(*,
                                 global_bucket_tokens,
                                 local_kv_cache_num_blocks,
                                 page_size,
                                 pcp_size,
                                 q_block_size,
                                 max_num_reqs=1):
    return estimate_pcp_streaming_metadata_schedule_steps_ub(
        global_bucket_tokens=global_bucket_tokens,
        capacity_tokens=local_kv_cache_num_blocks * pcp_size * page_size,
        block_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
        kv_pages_per_block=1,
        max_num_reqs=max_num_reqs,
    )


@pytest.mark.parametrize(
    ("pcp_size", "global_bucket_tokens", "q_start"),
    [
        (2, 1024, 1024),
        (4, 4096, 4096),
    ],
)
def test_metadata_schedule_matches_host_oracle_for_aligned_single_request(
        pcp_size, global_bucket_tokens, q_start):
    page_size = interleave_size = 32
    q_block_size = 256
    q_len = global_bucket_tokens
    kv_len = q_start + q_len
    local_kv_cache_num_blocks = kv_len // (pcp_size * page_size)
    block_tables = np.arange(local_kv_cache_num_blocks,
                             dtype=np.int32)[None, :]
    kv_lens = np.array([kv_len, 0], dtype=np.int32)
    cu_q_lens = np.array([0, q_len, q_len], dtype=np.int32)
    distribution = np.array([0, 0, 1], dtype=np.int32)

    actual = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=kv_lens,
        page_indices=np.vstack([block_tables[0], block_tables[0] + 1000]),
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[q_start],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=_metadata_lockstep_pad_steps(
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            max_num_reqs=kv_lens.size,
        ),
    )

    _assert_schedule_equal(actual, expected)
    validate_pcp_streaming_schedule(actual)


def test_metadata_schedule_matches_host_oracle_for_aligned_multi_request():
    pcp_size = 4
    page_size = interleave_size = 32
    q_block_size = 256
    q_lens = np.array([1024, 1024], dtype=np.int32)
    q_starts = np.array([0, 1024], dtype=np.int32)
    kv_lens_active = q_starts + q_lens
    global_bucket_tokens = int(q_lens.sum())
    local_kv_cache_num_blocks = 64
    page_indices = np.vstack([
        np.arange(local_kv_cache_num_blocks, dtype=np.int32),
        np.arange(local_kv_cache_num_blocks, dtype=np.int32) + 1000,
        np.zeros(local_kv_cache_num_blocks, dtype=np.int32),
    ])
    localized_block_tables = (page_indices[:2].astype(np.int64) %
                              local_kv_cache_num_blocks).astype(np.int32)

    actual = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=np.array([kv_lens_active[0], kv_lens_active[1], 0],
                         dtype=np.int32),
        page_indices=page_indices,
        cu_q_lens=np.array(
            [0, q_lens[0], q_lens.sum(),
             q_lens.sum()], dtype=np.int32),
        distribution=np.array([0, 0, 2], dtype=np.int32),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=kv_lens_active,
        cu_q_lens=np.array([0, q_lens[0], q_lens.sum()], dtype=np.int32),
        q_start_offsets=q_starts,
        block_tables=localized_block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=_metadata_lockstep_pad_steps(
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            max_num_reqs=page_indices.shape[0],
        ),
    )

    _assert_schedule_equal(actual, expected)
    validate_pcp_streaming_schedule(actual, require_ring_source_invariant=True)


def test_metadata_schedule_accepts_decode_only_distribution():
    pcp_size = 2
    page_size = interleave_size = 4
    q_block_size = 4
    q_lens = np.array([1, 1], dtype=np.int32)
    q_starts = np.array([15, 31], dtype=np.int32)
    kv_lens_active = q_starts + q_lens
    global_bucket_tokens = 16
    local_kv_cache_num_blocks = 16
    page_indices = np.vstack([
        np.arange(local_kv_cache_num_blocks, dtype=np.int32),
        np.arange(local_kv_cache_num_blocks, dtype=np.int32) + 1000,
        np.zeros(local_kv_cache_num_blocks, dtype=np.int32),
    ])
    localized_block_tables = (page_indices[:2].astype(np.int64) %
                              local_kv_cache_num_blocks).astype(np.int32)

    actual = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=np.array([kv_lens_active[0], kv_lens_active[1], 0],
                         dtype=np.int32),
        page_indices=page_indices,
        cu_q_lens=np.array([0, 1, 2, 2], dtype=np.int32),
        distribution=np.array([2, 2, 2], dtype=np.int32),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=kv_lens_active,
        cu_q_lens=np.array([0, 1, 2], dtype=np.int32),
        q_start_offsets=q_starts,
        block_tables=localized_block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=_metadata_lockstep_pad_steps(
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            max_num_reqs=page_indices.shape[0],
        ),
    )

    _assert_schedule_equal(actual, expected)
    validate_pcp_streaming_schedule(actual, require_ring_source_invariant=True)


def test_metadata_schedule_accepts_mixed_distribution():
    pcp_size = 2
    page_size = interleave_size = 4
    q_block_size = 4
    q_lens = np.array([1, 9, 17], dtype=np.int32)
    q_starts = np.array([15, 0, 12], dtype=np.int32)
    kv_lens_active = q_starts + q_lens
    global_bucket_tokens = 32
    local_kv_cache_num_blocks = 16
    page_indices = np.vstack([
        np.arange(local_kv_cache_num_blocks, dtype=np.int32),
        np.arange(local_kv_cache_num_blocks, dtype=np.int32) + 1000,
        np.arange(local_kv_cache_num_blocks, dtype=np.int32) + 2000,
        np.zeros(local_kv_cache_num_blocks, dtype=np.int32),
    ])
    localized_block_tables = (page_indices[:3].astype(np.int64) %
                              local_kv_cache_num_blocks).astype(np.int32)

    actual = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=np.array(
            [kv_lens_active[0], kv_lens_active[1], kv_lens_active[2], 0],
            dtype=np.int32),
        page_indices=page_indices,
        cu_q_lens=np.array([0, 1, 10, 27, 27], dtype=np.int32),
        distribution=np.array([1, 2, 3], dtype=np.int32),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=kv_lens_active,
        cu_q_lens=np.array([0, 1, 10, 27], dtype=np.int32),
        q_start_offsets=q_starts,
        block_tables=localized_block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=_metadata_lockstep_pad_steps(
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            max_num_reqs=page_indices.shape[0],
        ),
    )

    _assert_schedule_equal(actual, expected)
    validate_pcp_streaming_schedule(actual, require_ring_source_invariant=True)


def test_metadata_schedule_rejects_invalid_v3_distribution_order():
    with pytest.raises(ValueError, match="decode_end <= prefill_end"):
        generate_pcp_streaming_schedule_from_metadata_host(
            kv_lens=np.array([1], dtype=np.int32),
            page_indices=np.array([[0]], dtype=np.int32),
            cu_q_lens=np.array([0, 1], dtype=np.int32),
            distribution=np.array([1, 0, 1], dtype=np.int32),
            global_bucket_tokens=8,
            local_kv_cache_num_blocks=1,
            page_size=4,
            pcp_size=2,
            interleave_size=4,
            q_block_size=4,
        )


def test_metadata_schedule_shape_comes_from_compile_bucket_not_live_q_len():
    pcp_size = 4
    page_size = interleave_size = 32
    q_block_size = 256
    q_len = 1024
    global_bucket_tokens = 4096
    q_start = 0
    kv_len = q_start + q_len
    local_kv_cache_num_blocks = 128
    block_tables = np.arange(local_kv_cache_num_blocks,
                             dtype=np.int32)[None, :]

    actual = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=np.array([kv_len], dtype=np.int32),
        page_indices=block_tables.reshape(-1),
        cu_q_lens=np.array([0, q_len], dtype=np.int32),
        distribution=np.array([0, 0, 1], dtype=np.int32),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    bucket_steps = _metadata_lockstep_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
    )
    live_steps = _metadata_lockstep_pad_steps(
        global_bucket_tokens=q_len,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
    )

    assert actual.packed_schedule.shape[0] == bucket_steps
    assert bucket_steps > live_steps
    np.testing.assert_array_equal(
        build_pcp_streaming_active_page_groups(actual),
        np.array([int(actual.global_actual_steps[0]) // pcp_size],
                 dtype=np.int32),
    )


def test_metadata_schedule_localizes_standard_block_table_pages():
    pcp_size = 4
    page_size = interleave_size = 32
    q_block_size = 256
    global_bucket_tokens = 1024
    q_start = 1024
    q_len = 1024
    kv_len = q_start + q_len
    local_kv_cache_num_blocks = 64
    block_tables = np.arange(local_kv_cache_num_blocks,
                             dtype=np.int32)[None, :]
    standard_page_indices = block_tables + local_kv_cache_num_blocks

    actual = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=np.array([kv_len], dtype=np.int32),
        page_indices=standard_page_indices.reshape(-1),
        cu_q_lens=np.array([0, q_len], dtype=np.int32),
        distribution=np.array([0, 0, 1], dtype=np.int32),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[q_start],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=_metadata_lockstep_pad_steps(
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            max_num_reqs=1,
        ),
    )

    _assert_schedule_equal(actual, expected)
    np.testing.assert_array_equal(actual.kv_page_idx, expected.kv_page_idx)


def test_metadata_schedule_inputs_return_kernel_arrays():
    packed_schedule, active_page_groups = (
        build_pcp_streaming_schedule_inputs_from_metadata_host(
            kv_lens=np.array([1024], dtype=np.int32),
            page_indices=np.arange(16, dtype=np.int32),
            cu_q_lens=np.array([0, 1024], dtype=np.int32),
            distribution=np.array([0, 0, 1], dtype=np.int32),
            global_bucket_tokens=1024,
            local_kv_cache_num_blocks=16,
            page_size=32,
            pcp_size=2,
            interleave_size=32,
            q_block_size=256,
        ))

    assert packed_schedule.dtype == np.int32
    assert packed_schedule.shape[-1] == ScheduleField.PACKED_NUM_FIELDS
    assert active_page_groups.dtype == np.int32
    assert active_page_groups.shape == (1, )
    assert int(active_page_groups[0]) > 0
    assert packed_schedule.shape[0] >= int(active_page_groups[0]) * 2
    assert packed_schedule.shape[0] % 2 == 0


@functools.partial(
    jax.jit,
    static_argnames=(
        "global_bucket_tokens",
        "local_kv_cache_num_blocks",
        "page_size",
        "pcp_size",
        "interleave_size",
        "q_block_size",
        "max_steps",
    ),
)
def _build_metadata_schedule_inputs_jit(
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    global_bucket_tokens,
    local_kv_cache_num_blocks,
    page_size,
    pcp_size,
    interleave_size,
    q_block_size,
    max_steps,
):
    return build_pcp_streaming_schedule_inputs_from_metadata_jax(
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=max_steps,
    )


def _cpu_i32_array(value):
    return jax.device_put(np.asarray(value, dtype=np.int32),
                          jax.local_devices(backend="cpu")[0])


@pytest.mark.parametrize(
    ("pcp_size", "global_bucket_tokens", "q_start"),
    [
        (2, 1024, 1024),
        (4, 4096, 4096),
    ],
)
def test_jax_metadata_schedule_jit_matches_host_oracle(pcp_size,
                                                       global_bucket_tokens,
                                                       q_start):
    page_size = interleave_size = 32
    q_block_size = 256
    q_len = global_bucket_tokens
    kv_len = q_start + q_len
    local_kv_cache_num_blocks = kv_len // (pcp_size * page_size)
    block_tables = np.arange(local_kv_cache_num_blocks,
                             dtype=np.int32)[None, :]
    standard_page_indices = block_tables + local_kv_cache_num_blocks
    max_steps = _metadata_schedule_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )

    packed_schedule, active_page_groups = _build_metadata_schedule_inputs_jit(
        _cpu_i32_array([kv_len]),
        _cpu_i32_array(standard_page_indices.reshape(-1)),
        _cpu_i32_array([0, q_len]),
        _cpu_i32_array([0, 0, 1]),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=max_steps,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[q_start],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=max_steps,
    )

    np.testing.assert_array_equal(np.asarray(packed_schedule),
                                  expected.packed_schedule)
    np.testing.assert_array_equal(
        np.asarray(active_page_groups),
        build_pcp_streaming_active_page_groups(expected),
    )


def test_jax_metadata_schedule_default_max_steps_matches_host_for_single_request(
):
    pcp_size = 2
    page_size = interleave_size = 4
    q_block_size = 4
    global_bucket_tokens = 16
    q_len = global_bucket_tokens
    q_start = global_bucket_tokens
    kv_len = q_start + q_len
    local_kv_cache_num_blocks = 16
    block_tables = np.arange(local_kv_cache_num_blocks,
                             dtype=np.int32)[None, :]
    standard_page_indices = block_tables + local_kv_cache_num_blocks
    single_request_steps = _metadata_schedule_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    lockstep_steps = _metadata_lockstep_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
    )

    packed_schedule, active_page_groups = _build_metadata_schedule_inputs_jit(
        _cpu_i32_array([kv_len]),
        _cpu_i32_array(standard_page_indices.reshape(-1)),
        _cpu_i32_array([0, q_len]),
        _cpu_i32_array([0, 0, 1]),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=None,
    )
    expected_packed, expected_active_page_groups = (
        build_pcp_streaming_schedule_inputs_from_metadata_host(
            kv_lens=np.array([kv_len], dtype=np.int32),
            page_indices=standard_page_indices.reshape(-1),
            cu_q_lens=np.array([0, q_len], dtype=np.int32),
            distribution=np.array([0, 0, 1], dtype=np.int32),
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            q_block_size=q_block_size,
        ))

    assert single_request_steps < lockstep_steps
    assert expected_packed.shape[0] == lockstep_steps
    np.testing.assert_array_equal(np.asarray(packed_schedule), expected_packed)
    np.testing.assert_array_equal(np.asarray(active_page_groups),
                                  expected_active_page_groups)


def test_jax_metadata_schedule_default_max_steps_matches_host_for_padded_single_request(
):
    pcp_size = 2
    page_size = interleave_size = 4
    q_block_size = 4
    max_num_seqs = 8
    global_bucket_tokens = 16
    q_len = global_bucket_tokens
    q_start = global_bucket_tokens
    kv_len = q_start + q_len
    local_kv_cache_num_blocks = 16
    page_indices = np.zeros((max_num_seqs, local_kv_cache_num_blocks),
                            dtype=np.int32)
    page_indices[0] = (np.arange(local_kv_cache_num_blocks, dtype=np.int32) +
                       local_kv_cache_num_blocks)
    kv_lens = np.zeros(max_num_seqs, dtype=np.int32)
    kv_lens[0] = kv_len
    cu_q_lens = np.full(max_num_seqs + 1, q_len, dtype=np.int32)
    cu_q_lens[0] = 0
    single_request_steps = _metadata_schedule_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    lockstep_steps = _metadata_lockstep_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
        max_num_reqs=max_num_seqs,
    )

    packed_schedule, active_page_groups = _build_metadata_schedule_inputs_jit(
        _cpu_i32_array(kv_lens),
        _cpu_i32_array(page_indices),
        _cpu_i32_array(cu_q_lens),
        _cpu_i32_array([0, 0, 1]),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=None,
    )
    expected_packed, expected_active_page_groups = (
        build_pcp_streaming_schedule_inputs_from_metadata_host(
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=np.array([0, 0, 1], dtype=np.int32),
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            q_block_size=q_block_size,
        ))

    assert single_request_steps < lockstep_steps
    assert expected_packed.shape[0] == lockstep_steps
    np.testing.assert_array_equal(np.asarray(packed_schedule), expected_packed)
    np.testing.assert_array_equal(np.asarray(active_page_groups),
                                  expected_active_page_groups)


def test_jax_metadata_schedule_jit_matches_host_oracle_for_multi_request():
    pcp_size = 4
    page_size = interleave_size = 32
    q_block_size = 256
    q_lens = np.array([1024, 1024], dtype=np.int32)
    q_starts = np.array([0, 1024], dtype=np.int32)
    kv_lens_active = q_starts + q_lens
    global_bucket_tokens = int(q_lens.sum())
    local_kv_cache_num_blocks = 64
    page_indices = np.vstack([
        np.arange(local_kv_cache_num_blocks, dtype=np.int32),
        np.arange(local_kv_cache_num_blocks, dtype=np.int32) + 1000,
        np.zeros(local_kv_cache_num_blocks, dtype=np.int32),
    ])
    localized_block_tables = (page_indices[:2].astype(np.int64) %
                              local_kv_cache_num_blocks).astype(np.int32)
    max_steps = _metadata_lockstep_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
    )

    packed_schedule, active_page_groups = _build_metadata_schedule_inputs_jit(
        _cpu_i32_array([kv_lens_active[0], kv_lens_active[1], 0]),
        _cpu_i32_array(page_indices),
        _cpu_i32_array([0, q_lens[0], q_lens.sum(),
                        q_lens.sum()]),
        _cpu_i32_array([0, 0, 2]),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=max_steps,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=kv_lens_active,
        cu_q_lens=np.array([0, q_lens[0], q_lens.sum()], dtype=np.int32),
        q_start_offsets=q_starts,
        block_tables=localized_block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=max_steps,
    )

    np.testing.assert_array_equal(np.asarray(packed_schedule),
                                  expected.packed_schedule)
    np.testing.assert_array_equal(
        np.asarray(active_page_groups),
        build_pcp_streaming_active_page_groups(expected),
    )


@pytest.mark.parametrize(
    ("q_lens", "q_starts"),
    [
        ([17], [120]),
        ([17, 20], [120, 250]),
        ([1, 63, 97], [0, 33, 128]),
        ([80, 7, 129], [5, 96, 257]),
    ],
)
def test_jax_metadata_schedule_jit_matches_host_for_unaligned_requests(
        q_lens, q_starts):
    pcp_size = 4
    page_size = interleave_size = 32
    q_block_size = 64
    local_kv_cache_num_blocks = 16
    max_num_seqs = 3
    global_bucket_tokens = 512
    q_lens = np.asarray(q_lens, dtype=np.int32)
    q_starts = np.asarray(q_starts, dtype=np.int32)
    num_reqs = int(q_lens.size)
    kv_lens = np.zeros(max_num_seqs, dtype=np.int32)
    kv_lens[:num_reqs] = q_starts + q_lens
    cu_q_lens = np.zeros(max_num_seqs + 1, dtype=np.int32)
    cu_q_lens[1:num_reqs + 1] = np.cumsum(q_lens)
    cu_q_lens[num_reqs + 1:] = cu_q_lens[num_reqs]
    distribution = np.array([0, 0, num_reqs], dtype=np.int32)
    page_indices = np.tile(
        np.arange(local_kv_cache_num_blocks, dtype=np.int32),
        (max_num_seqs, 1),
    )
    max_steps = _metadata_lockstep_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
        max_num_reqs=max_num_seqs,
    )

    packed_schedule, active_page_groups = _build_metadata_schedule_inputs_jit(
        _cpu_i32_array(kv_lens),
        _cpu_i32_array(page_indices),
        _cpu_i32_array(cu_q_lens),
        _cpu_i32_array(distribution),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=max_steps,
    )
    expected_packed, expected_active_page_groups = (
        build_pcp_streaming_schedule_inputs_from_metadata_host(
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            global_bucket_tokens=global_bucket_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            q_block_size=q_block_size,
        ))

    np.testing.assert_array_equal(np.asarray(packed_schedule), expected_packed)
    np.testing.assert_array_equal(np.asarray(active_page_groups),
                                  expected_active_page_groups)


def test_jax_metadata_schedule_jit_shape_comes_from_compile_bucket():
    pcp_size = 4
    page_size = interleave_size = 32
    q_block_size = 256
    q_len = 1024
    global_bucket_tokens = 4096
    kv_len = q_len
    local_kv_cache_num_blocks = 128
    block_tables = np.arange(local_kv_cache_num_blocks,
                             dtype=np.int32)[None, :]
    max_steps = _metadata_schedule_pad_steps(
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    live_steps = _metadata_schedule_pad_steps(
        global_bucket_tokens=q_len,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )

    packed_schedule, active_page_groups = _build_metadata_schedule_inputs_jit(
        _cpu_i32_array([kv_len]),
        _cpu_i32_array(block_tables.reshape(-1)),
        _cpu_i32_array([0, q_len]),
        _cpu_i32_array([0, 0, 1]),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=max_steps,
    )
    expected = generate_pcp_streaming_schedule(
        kv_lens=[kv_len],
        cu_q_lens=[0, q_len],
        q_start_offsets=[0],
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=1,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=max_steps,
    )

    assert packed_schedule.shape[0] == max_steps
    assert max_steps > live_steps
    np.testing.assert_array_equal(np.asarray(packed_schedule),
                                  expected.packed_schedule)
    np.testing.assert_array_equal(
        np.asarray(active_page_groups),
        build_pcp_streaming_active_page_groups(expected),
    )


def test_jax_metadata_schedule_consumes_request_distribution():
    pcp_size = 4
    page_size = interleave_size = 32
    q_block_size = 256
    q_len = 1024
    local_kv_cache_num_blocks = 128
    block_tables = np.arange(local_kv_cache_num_blocks,
                             dtype=np.int32)[None, :]
    max_steps = _metadata_schedule_pad_steps(
        global_bucket_tokens=q_len,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )

    packed_schedule, active_page_groups = _build_metadata_schedule_inputs_jit(
        _cpu_i32_array([q_len]),
        _cpu_i32_array(block_tables.reshape(-1)),
        _cpu_i32_array([0, q_len]),
        _cpu_i32_array([0, 0, 0]),
        global_bucket_tokens=q_len,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        max_steps=max_steps,
    )

    assert int(active_page_groups[0]) == 0
    np.testing.assert_array_equal(
        np.asarray(packed_schedule[..., ScheduleField.REQ_ID]),
        np.full((max_steps, pcp_size, 1), -1, dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({
            "num_lanes": 2
        }, "num_lanes == 1"),
        ({
            "cu_q_lens": [0, 4]
        }, "q_len"),
        ({
            "q_start_offsets": [2],
            "kv_lens": [10]
        }, "q_start_offset"),
        ({
            "pad_kv_pages_to_pcp_group": False
        }, "pad_kv_pages_to_pcp_group=True"),
        ({
            "bq_sz": 3
        }, "bq_sz"),
    ],
)
def test_vectorized_schedule_rejects_unsupported_shapes(kwargs, match):
    args = {
        "kv_lens": [8],
        "cu_q_lens": [0, 8],
        "q_start_offsets": [0],
        "block_tables": np.array([[100]], dtype=np.int32),
        "page_size": 2,
        "pcp_size": 4,
        "interleave_size": 2,
        "num_lanes": 1,
        "bq_sz": 2,
        "pad_kv_pages_to_pcp_group": True,
    }
    args.update(kwargs)

    with pytest.raises(NotImplementedError, match=match):
        generate_pcp_streaming_schedule(**args)


def test_vectorized_schedule_supports_aligned_multiple_active_requests():
    actual = generate_pcp_streaming_schedule(
        kv_lens=[16, 16],
        cu_q_lens=[0, 8, 16],
        q_start_offsets=[8, 8],
        block_tables=np.array([[100, 101], [200, 201]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
    )
    expected = generate_pcp_streaming_schedule_reference(
        kv_lens=[16, 16],
        cu_q_lens=[0, 8, 16],
        q_start_offsets=[8, 8],
        block_tables=np.array([[100, 101], [200, 201]], dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
    )

    _assert_schedule_equal(actual, expected)
    assert actual.global_actual_steps[0] == 16
    np.testing.assert_array_equal(actual.req_id[:, 0, 0],
                                  np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(actual.req_id[:, 8, 0],
                                  np.ones(4, dtype=np.int32))
    np.testing.assert_array_equal(actual.q_hbm_offset[:, 8, 0],
                                  np.full(4, 2, dtype=np.int32))
    validate_pcp_streaming_schedule(actual, require_ring_source_invariant=True)


def test_vectorized_schedule_supports_unaligned_multiple_active_requests():
    schedule = generate_pcp_streaming_schedule(
        kv_lens=[17, 16],
        cu_q_lens=[0, 8, 12],
        q_start_offsets=[9, 8],
        block_tables=np.array([[100, 101, 102], [200, 201, 202]],
                              dtype=np.int32),
        page_size=2,
        pcp_size=4,
        interleave_size=2,
        num_lanes=1,
        bq_sz=2,
        pad_kv_pages_to_pcp_group=True,
    )

    validate_pcp_streaming_schedule(schedule,
                                    require_ring_source_invariant=True)
    assert schedule.global_actual_steps[0] > 0
    assert np.any(schedule.q_tile_size == 1)


def test_generate_schedule_rejects_non_aligned_page_and_interleave_size():
    with pytest.raises(NotImplementedError, match="page_size == interleave"):
        generate_pcp_streaming_schedule(
            kv_lens=[12],
            cu_q_lens=[0, 7],
            q_start_offsets=[5],
            block_tables=np.array([[100, 101]], dtype=np.int32),
            page_size=4,
            pcp_size=4,
            interleave_size=2,
            num_lanes=1,
            bq_sz=2,
        )
