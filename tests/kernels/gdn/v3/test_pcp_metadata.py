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

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.gdn.v3 import config, pcp_metadata

_PROJECTION_TOKEN_BLOCK_SIZE = 1024
_NUM_QKV_OUT_BLOCKS = 72
_NUM_PROJECTION_OUT_BLOCKS = 96


def _cfg(batch_size: int, tile_size: int = 64) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=batch_size,
        dim_size=384,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=128,
        v_head_dim=128,
    )


def _metadata_inputs(lengths, offsets, *, max_num_seqs):
    lengths = np.asarray(lengths, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int32)
    query_start = np.zeros((max_num_seqs + 1, ), dtype=np.int32)
    query_start[1:lengths.size + 1] = np.cumsum(lengths, dtype=np.int32)
    query_start[lengths.size + 1:] = query_start[lengths.size]
    seq_lens = np.zeros((max_num_seqs, ), dtype=np.int32)
    seq_lens[:lengths.size] = lengths + offsets
    state_indices = np.arange(max_num_seqs, dtype=np.int32)
    distribution = np.array([0, lengths.size, lengths.size], dtype=np.int32)
    return tuple(
        jnp.asarray(x)
        for x in (seq_lens, query_start, state_indices, distribution))


def test_schedule_splits_unaligned_requests_at_absolute_pcp_rounds():
    cfg = _cfg(batch_size=128, tile_size=4)
    seq_lens, query_start, state_indices, distribution = _metadata_inputs(
        lengths=[13, 11],
        offsets=[7, 22],
        max_num_seqs=4,
    )

    metadata, schedule = pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        distribution[0],
        distribution[-1],
        pcp_size=4,
        comm_chunk_size=4,
        projection_token_block_size=16,
        num_qkv_out_blocks=6,
        num_projection_out_blocks=8,
    )

    assert int(schedule.num_stages) == 4
    np.testing.assert_array_equal(np.asarray(schedule.num_tokens[:4]),
                                  [9, 4, 10, 1])
    np.testing.assert_array_equal(
        np.asarray(schedule.rank_valid_rows[:4]),
        [
            [0, 1, 4, 4],
            [4, 0, 0, 0],
            [0, 2, 4, 4],
            [1, 0, 0, 0],
        ],
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.rank_recv_start[:4]),
        [
            [0, 4, 8, 12],
            [0, 4, 8, 12],
            [0, 4, 8, 12],
            [0, 4, 8, 12],
        ],
    )
    # Rank-local request segments stay compact across stage boundaries and
    # across requests even though the requests start at different offsets.
    np.testing.assert_array_equal(
        np.asarray(schedule.rank_row_start[:4]),
        [
            [0, 0, 0, 0],
            [0, 1, 4, 4],
            [4, 1, 4, 4],
            [4, 3, 8, 8],
        ],
    )
    assert int(metadata.num_tiles) == 8
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_to_r_base.data[:8]),
        [0, 1, 5, 9, 13, 15, 19, 23],
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_to_r_size.data[:8]),
        [1, 4, 4, 4, 2, 4, 4, 1],
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_is_first_tile.data[:8]),
        [True, False, False, False, True, False, False, False],
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_is_last_tile.data[:8]),
        [False, False, False, True, False, False, False, True],
    )


def test_live_stage_and_tile_counts_do_not_expand_to_compile_bucket():
    cfg = _cfg(batch_size=8 * 4096)
    max_num_seqs = 16

    def build(seq_lens, query_start, state_indices, distribution):
        return pcp_metadata.compute_pcp_stage_metadata(
            cfg,
            seq_lens,
            query_start,
            state_indices,
            distribution[0],
            distribution[-1],
            pcp_size=8,
            comm_chunk_size=256,
            projection_token_block_size=_PROJECTION_TOKEN_BLOCK_SIZE,
            num_qkv_out_blocks=_NUM_QKV_OUT_BLOCKS,
            num_projection_out_blocks=_NUM_PROJECTION_OUT_BLOCKS,
        )

    compiled = jax.jit(build)
    short_inputs = _metadata_inputs(
        lengths=[17],
        offsets=[3],
        max_num_seqs=max_num_seqs,
    )
    long_inputs = _metadata_inputs(
        lengths=[4096] * 8,
        offsets=[0] * 8,
        max_num_seqs=max_num_seqs,
    )

    short_metadata, short_schedule = compiled(*short_inputs)
    long_metadata, long_schedule = compiled(*long_inputs)

    # Both calls share identical compile-time array capacities.
    assert short_schedule.request_id.shape == long_schedule.request_id.shape
    assert short_schedule.tile_stage.shape == long_schedule.tile_stage.shape
    # The kernel grid and all schedule loops consume only the active prefixes.
    assert int(short_schedule.num_stages) == 1
    assert int(short_metadata.num_tiles) == 1
    np.testing.assert_array_equal(
        np.asarray(short_schedule.rank_active_row_end),
        [17, 0, 0, 0, 0, 0, 0, 0],
    )
    assert int(long_schedule.num_stages) == 16
    assert int(long_metadata.num_tiles) == 512
    np.testing.assert_array_equal(
        np.asarray(long_schedule.rank_active_row_end),
        [4096] * 8,
    )
    # The balanced 8x4K workload naturally provides enough GDN tiles to keep
    # every future QKV projection ahead of its communication deadline.
    np.testing.assert_array_equal(
        np.asarray(long_schedule.projection_work_offset_end),
        [0] * 8,
    )


def test_projection_schedule_catches_up_only_the_imbalanced_live_rank():
    cfg = _cfg(batch_size=8 * 4096)
    seq_lens, query_start, state_indices, distribution = _metadata_inputs(
        # Each independent request starts at absolute position zero, so all
        # live rows belong to PCP rank zero.  This deliberately exposes fewer
        # GDN tiles than the next 1024-row projection block needs.
        lengths=[256] * 5,
        offsets=[0] * 5,
        max_num_seqs=16,
    )

    metadata, schedule = pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        distribution[0],
        distribution[-1],
        pcp_size=8,
        comm_chunk_size=256,
        projection_token_block_size=_PROJECTION_TOKEN_BLOCK_SIZE,
        num_qkv_out_blocks=_NUM_QKV_OUT_BLOCKS,
        num_projection_out_blocks=_NUM_PROJECTION_OUT_BLOCKS,
    )

    assert int(schedule.num_stages) == 5
    assert int(metadata.num_tiles) == 20
    np.testing.assert_array_equal(
        np.asarray(schedule.rank_active_row_end),
        [1280, 0, 0, 0, 0, 0, 0, 0],
    )
    # Stage four launches from stage three's first-tile prologue, after only
    # 12 GDN tiles.  Reaching token block one in
    # the projection queue requires block-zero Z (24 tiles) plus block-one QKV
    # (72 tiles), so rank zero performs exactly the missing 84 tiles before
    # launching that stage's QKV DMA.  Empty ranks do no catch-up work.
    np.testing.assert_array_equal(
        np.asarray(schedule.projection_catchup_count[:5, 0]),
        [0, 0, 0, 0, 84],
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.projection_work_offset_end),
        [84, 0, 0, 0, 0, 0, 0, 0],
    )
