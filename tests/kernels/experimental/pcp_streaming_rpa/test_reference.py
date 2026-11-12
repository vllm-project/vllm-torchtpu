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
"""Host-reference coverage for the production PCP runtime schedule ABI."""

import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.reference import (
    build_runtime_schedule_reference,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField,
    build_pcp_streaming_schedule_inputs_from_metadata_jax,
)

pytestmark = pytest.mark.multichip


def _build_reference(
    *,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    global_bucket_tokens,
    local_kv_cache_num_blocks,
    page_size,
    pcp_size,
    interleave_size,
    q_block_size,
):
    tile_plan, block_tables, *_ = build_pcp_streaming_schedule_inputs_from_metadata_jax(
        kv_lens=np.asarray(kv_lens, dtype=np.int32),
        page_indices=np.asarray(page_indices, dtype=np.int32),
        cu_q_lens=np.asarray(cu_q_lens, dtype=np.int32),
        distribution=np.asarray(distribution, dtype=np.int32),
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
    )
    return build_runtime_schedule_reference(
        tile_plan,
        block_tables,
        pcp_size=pcp_size,
        page_size=page_size,
        interleave_size=interleave_size,
    )


def _active_rows(rows):
    return rows[..., RuntimeScheduleField.REQ_ID] >= 0


def test_runtime_reference_separates_history_boundary_and_fresh_kv():
    reference = _build_reference(
        kv_lens=[19],
        page_indices=np.arange(8, dtype=np.int32) + 5,
        cu_q_lens=[0, 8],
        distribution=[0, 0, 1],
        global_bucket_tokens=8,
        local_kv_cache_num_blocks=8,
        page_size=4,
        pcp_size=2,
        interleave_size=2,
        q_block_size=4,
    )

    active_tile = int(np.flatnonzero(reference.current_num_groups > 0)[0])
    current = reference.current_rows[active_tile]
    history = reference.history_rows[active_tile]
    assert int(reference.current_num_groups[active_tile]) == 3
    assert int(reference.history_num_groups[active_tile]) == 1

    boundary = current[0]
    boundary_active = _active_rows(boundary)
    assert np.all(
        boundary[..., RuntimeScheduleField.KV_HBM_OFFSET][boundary_active] == -1
    )
    assert np.all(boundary[..., RuntimeScheduleField.CAPTURE_LEN] == 0)
    np.testing.assert_array_equal(
        boundary[:, 0, RuntimeScheduleField.KV_PAGE_IDX],
        np.array([6, 6], dtype=np.int32),
    )

    fresh = current[1:3]
    assert np.all(
        fresh[..., RuntimeScheduleField.KV_HBM_OFFSET][_active_rows(fresh)] >= 0
    )
    capture_len = fresh[..., RuntimeScheduleField.CAPTURE_LEN].sum(axis=-1)
    np.testing.assert_array_equal(
        capture_len,
        fresh[..., 0, RuntimeScheduleField.KV_VALID_LEN],
    )

    history_active = _active_rows(history)
    assert np.all(history[..., RuntimeScheduleField.KV_HBM_OFFSET][history_active] == 0)
    np.testing.assert_array_equal(
        history[0, :, 0, RuntimeScheduleField.KV_PAGE_IDX],
        np.array([5, 5], dtype=np.int32),
    )


def test_runtime_reference_routes_one_token_capture_to_absolute_cache_owner():
    pcp_size = 8
    interleave_size = 256
    reference = _build_reference(
        kv_lens=[302],
        page_indices=[7],
        cu_q_lens=[0, 1],
        distribution=[0, 0, 1],
        global_bucket_tokens=pcp_size * interleave_size,
        local_kv_cache_num_blocks=32,
        page_size=interleave_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=interleave_size,
    )

    active_tile = int(np.flatnonzero(reference.current_num_groups > 0)[0])
    current = reference.current_rows[active_tile]
    assert int(reference.current_num_groups[active_tile]) == 2
    fresh_source_0 = current[1, 0]
    assert int(fresh_source_0[0, RuntimeScheduleField.KV_GLOBAL_START]) == 301
    assert int(fresh_source_0[0, RuntimeScheduleField.KV_VALID_LEN]) == 1
    assert int(fresh_source_0[1, RuntimeScheduleField.CAPTURE_SRC_OFFSET]) == 0
    assert int(fresh_source_0[1, RuntimeScheduleField.CAPTURE_DST_OFFSET]) == 0
    assert int(fresh_source_0[1, RuntimeScheduleField.CAPTURE_LEN]) == 1
    assert np.all(
        np.delete(fresh_source_0[..., RuntimeScheduleField.CAPTURE_LEN], 1) == 0
    )


def test_runtime_reference_splits_capture_at_cache_owner_boundary():
    reference = _build_reference(
        kv_lens=[333],
        page_indices=np.arange(8, dtype=np.int32),
        cu_q_lens=[0, 32],
        distribution=[0, 0, 1],
        global_bucket_tokens=64,
        local_kv_cache_num_blocks=8,
        page_size=32,
        pcp_size=2,
        interleave_size=32,
        q_block_size=32,
    )

    active_tile = int(np.flatnonzero(reference.current_num_groups > 0)[0])
    current = reference.current_rows[active_tile]
    assert int(reference.current_num_groups[active_tile]) == 2
    fresh_source_0 = current[1, 0]
    assert int(fresh_source_0[0, RuntimeScheduleField.KV_GLOBAL_START]) == 301
    assert int(fresh_source_0[0, RuntimeScheduleField.KV_VALID_LEN]) == 32
    assert int(fresh_source_0[1, RuntimeScheduleField.CAPTURE_SRC_OFFSET]) == 0
    assert int(fresh_source_0[1, RuntimeScheduleField.CAPTURE_DST_OFFSET]) == 0
    assert int(fresh_source_0[1, RuntimeScheduleField.CAPTURE_LEN]) == 19
    assert int(fresh_source_0[0, RuntimeScheduleField.CAPTURE_SRC_OFFSET]) == 19
    assert int(fresh_source_0[0, RuntimeScheduleField.CAPTURE_DST_OFFSET]) == 0
    assert int(fresh_source_0[0, RuntimeScheduleField.CAPTURE_LEN]) == 13


@pytest.mark.parametrize("history_tokens", [25_600, 128 * 1024])
def test_runtime_reference_preserves_long_absolute_history(history_tokens):
    pcp_size = 4
    page_size = 128
    q_len = 256
    kv_len = history_tokens + q_len
    local_blocks = (kv_len + pcp_size * page_size - 1) // (pcp_size * page_size)
    reference = _build_reference(
        kv_lens=[kv_len],
        page_indices=np.arange(local_blocks, dtype=np.int32),
        cu_q_lens=[0, q_len],
        distribution=[0, 0, 1],
        global_bucket_tokens=q_len,
        local_kv_cache_num_blocks=local_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=32,
        q_block_size=64,
    )

    active_tile = int(np.flatnonzero(reference.current_num_groups > 0)[0])
    assert int(reference.history_num_groups[active_tile]) == (
        history_tokens // (pcp_size * page_size)
    )
    current = reference.current_rows[active_tile]
    current_active = _active_rows(current)
    assert np.all(
        current[..., RuntimeScheduleField.Q_GLOBAL_START][current_active]
        >= history_tokens
    )


def test_runtime_reference_uses_only_runtime_abi_fields():
    reference = _build_reference(
        kv_lens=[8],
        page_indices=[3],
        cu_q_lens=[0, 8],
        distribution=[0, 0, 1],
        global_bucket_tokens=8,
        local_kv_cache_num_blocks=4,
        page_size=4,
        pcp_size=2,
        interleave_size=2,
        q_block_size=4,
    )
    for rows in (reference.current_rows, reference.history_rows):
        assert rows.shape[-1] == RuntimeScheduleField.PACKED_NUM_FIELDS
        np.testing.assert_array_equal(
            rows[..., RuntimeScheduleField.NUM_FIELDS :],
            np.zeros_like(rows[..., RuntimeScheduleField.NUM_FIELDS :]),
        )
