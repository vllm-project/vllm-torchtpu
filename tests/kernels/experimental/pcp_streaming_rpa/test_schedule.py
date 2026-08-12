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
"""Tests for the production PCP JAX tile-plan scheduler."""

import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    TilePlanField, build_pcp_streaming_schedule_inputs_from_metadata_jax)

pytestmark = pytest.mark.multichip


def test_metadata_jax_compact_plan_tracks_group_starts():
    pcp_size = 2
    (tile_plan, block_tables, current_ids, current_starts, current_meta,
     history_ids, history_starts,
     history_meta) = build_pcp_streaming_schedule_inputs_from_metadata_jax(
         kv_lens=np.array([23, 8], dtype=np.int32),
         page_indices=np.arange(32, dtype=np.int32),
         cu_q_lens=np.array([0, 8, 8], dtype=np.int32),
         distribution=np.array([0, 0, 1], dtype=np.int32),
         global_bucket_tokens=8,
         local_kv_cache_num_blocks=16,
         page_size=4,
         pcp_size=pcp_size,
         interleave_size=2,
         q_block_size=4,
     )

    tile_plan = np.asarray(tile_plan)
    logical = tile_plan[:, 0, :pcp_size * TilePlanField.NUM_FIELDS].reshape(
        tile_plan.shape[0], pcp_size, TilePlanField.NUM_FIELDS)
    assert tile_plan.shape == (6, 1, TilePlanField.HBM_ROW_FIELDS)
    assert np.asarray(block_tables).shape == (2, 16)

    for ids, starts, meta, field in (
        (current_ids, current_starts, current_meta,
         TilePlanField.CURRENT_NUM_GROUPS),
        (history_ids, history_starts, history_meta,
         TilePlanField.HISTORY_NUM_GROUPS),
    ):
        groups = logical[:, 0, field]
        expected_ids = np.flatnonzero(groups > 0)
        expected_starts = np.cumsum(
            groups[expected_ids]) - groups[expected_ids]
        count = int(np.asarray(meta)[0])
        np.testing.assert_array_equal(np.asarray(ids)[0, :count], expected_ids)
        np.testing.assert_array_equal(
            np.asarray(starts)[0, :count], expected_starts)
        assert int(np.asarray(meta)[1]) == int(groups.sum())


def test_metadata_jax_moves_partial_history_page_to_current_pass():
    pcp_size = 8
    history_tokens = 25_600
    current_tokens = 32_768
    (tile_plan, _block_tables, _current_ids, _current_starts, current_meta,
     _history_ids, _history_starts,
     history_meta) = build_pcp_streaming_schedule_inputs_from_metadata_jax(
         kv_lens=np.array([history_tokens + current_tokens], dtype=np.int32),
         page_indices=np.arange(136, dtype=np.int32),
         cu_q_lens=np.array([0, current_tokens], dtype=np.int32),
         distribution=np.array([0, 0, 1], dtype=np.int32),
         global_bucket_tokens=current_tokens,
         local_kv_cache_num_blocks=7565,
         page_size=256,
         pcp_size=pcp_size,
         interleave_size=256,
         q_block_size=512,
     )

    logical = np.asarray(tile_plan)[:, 0, :pcp_size *
                                    TilePlanField.NUM_FIELDS].reshape(
                                        tile_plan.shape[0], pcp_size,
                                        TilePlanField.NUM_FIELDS)
    active = np.any(logical[..., TilePlanField.Q_TILE_SIZE] > 0, axis=1)
    logical = logical[active]

    # 25,600 tokens contain 12 complete PCP virtual pages and one half-full
    # boundary page. Only the 12 complete pages use the unmasked history pass;
    # the boundary is the first causal group in every active current tile.
    np.testing.assert_array_equal(
        logical[:, 0, TilePlanField.HISTORY_NUM_GROUPS],
        np.full(8, 12, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        logical[:, 0, TilePlanField.CURRENT_NUM_GROUPS],
        np.arange(3, 18, 2, dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(current_meta),
                                  np.array([8, 80], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(history_meta),
                                  np.array([8, 96], dtype=np.int32))


def test_metadata_jax_default_owner_uses_batch_flat_query_offsets():
    kwargs = {
        "kv_lens": np.array([14, 21], dtype=np.int32),
        "page_indices": np.arange(16, dtype=np.int32),
        "cu_q_lens": np.array([0, 4, 8], dtype=np.int32),
        "distribution": np.array([0, 0, 2], dtype=np.int32),
        "global_bucket_tokens": 8,
        "local_kv_cache_num_blocks": 8,
        "page_size": 4,
        "pcp_size": 2,
        "interleave_size": 2,
        "q_block_size": 2,
    }
    tile_plan, *_ = build_pcp_streaming_schedule_inputs_from_metadata_jax(
        **kwargs)
    logical = np.asarray(tile_plan)[:,
                                    0, :2 * TilePlanField.NUM_FIELDS].reshape(
                                        tile_plan.shape[0], 2,
                                        TilePlanField.NUM_FIELDS)
    active = np.any(logical[..., TilePlanField.Q_TILE_SIZE] > 0, axis=1)
    logical = logical[active]

    np.testing.assert_array_equal(logical[:, 0, TilePlanField.REQ_ID], [0, 1])
    np.testing.assert_array_equal(
        logical[..., TilePlanField.Q_GLOBAL_START],
        np.array([[10, 12], [17, 19]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        logical[..., TilePlanField.Q_HBM_OFFSET],
        np.array([[0, 0], [2, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(logical[..., TilePlanField.Q_TILE_SIZE], 2)
    # Legacy absolute-position ownership would put request 0's first rows on
    # rank 1 and produce [12, 10], rather than the batch-flat [10, 12].
    assert not np.array_equal(
        logical[0, :, TilePlanField.Q_GLOBAL_START],
        np.array([12, 10], dtype=np.int32),
    )


@pytest.mark.parametrize("history_tokens", [25_600, 128 * 1024])
def test_metadata_jax_batch_flat_owner_preserves_long_absolute_history(
        history_tokens):
    pcp_size = 4
    q_len = 256
    page_size = 128
    interleave_size = 32
    q_block_size = 64
    kv_len = history_tokens + q_len
    local_kv_cache_num_blocks = (kv_len + pcp_size * page_size -
                                 1) // (pcp_size * page_size)
    (tile_plan, _block_tables, _current_ids, _current_starts, current_meta,
     _history_ids, _history_starts,
     history_meta) = build_pcp_streaming_schedule_inputs_from_metadata_jax(
         kv_lens=np.array([kv_len], dtype=np.int32),
         page_indices=np.arange(local_kv_cache_num_blocks, dtype=np.int32),
         cu_q_lens=np.array([0, q_len], dtype=np.int32),
         distribution=np.array([0, 0, 1], dtype=np.int32),
         global_bucket_tokens=q_len,
         local_kv_cache_num_blocks=local_kv_cache_num_blocks,
         page_size=page_size,
         pcp_size=pcp_size,
         interleave_size=interleave_size,
         q_block_size=q_block_size,
     )
    logical = np.asarray(tile_plan)[:, 0, :pcp_size *
                                    TilePlanField.NUM_FIELDS].reshape(
                                        tile_plan.shape[0], pcp_size,
                                        TilePlanField.NUM_FIELDS)
    active = np.any(logical[..., TilePlanField.Q_TILE_SIZE] > 0, axis=1)
    logical = logical[active]

    assert logical.shape[0] == 1
    np.testing.assert_array_equal(
        logical[0, :, TilePlanField.Q_GLOBAL_START],
        history_tokens + np.arange(pcp_size) * interleave_size,
    )
    np.testing.assert_array_equal(
        logical[0, :, TilePlanField.Q_TILE_SIZE],
        np.full(pcp_size, q_block_size, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        logical[..., TilePlanField.REQUEST_ABSOLUTE_QUERY_START],
        history_tokens,
    )
    assert np.all(logical[..., TilePlanField.HISTORY_NUM_GROUPS] > 0)
    assert int(np.asarray(current_meta)[1]) > 0
    assert int(np.asarray(history_meta)[1]) > 0


@pytest.mark.parametrize("interleave_size", [0, -2])
def test_metadata_jax_schedule_rejects_nonpositive_interleave_size(
        interleave_size):
    with pytest.raises(ValueError, match="interleave_size must be positive"):
        build_pcp_streaming_schedule_inputs_from_metadata_jax(
            kv_lens=np.array([4], dtype=np.int32),
            page_indices=np.array([0], dtype=np.int32),
            cu_q_lens=np.array([0, 4], dtype=np.int32),
            distribution=np.array([0, 0, 1], dtype=np.int32),
            global_bucket_tokens=4,
            local_kv_cache_num_blocks=1,
            page_size=4,
            pcp_size=1,
            interleave_size=interleave_size,
            q_block_size=4,
        )
