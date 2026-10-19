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
import pytest

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
    query_start = np.zeros((max_num_seqs + 1,), dtype=np.int32)
    query_start[1 : lengths.size + 1] = np.cumsum(lengths, dtype=np.int32)
    query_start[lengths.size + 1 :] = query_start[lengths.size]
    seq_lens = np.zeros((max_num_seqs,), dtype=np.int32)
    seq_lens[: lengths.size] = lengths + offsets
    state_indices = np.arange(max_num_seqs, dtype=np.int32)
    distribution = np.array([0, lengths.size, lengths.size], dtype=np.int32)
    return tuple(
        jnp.asarray(x) for x in (seq_lens, query_start, state_indices, distribution)
    )


def _assert_pytree_array_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf), np.asarray(expected_leaf)
        )


def test_schedule_splits_cross_request_batch_flat_pcp_rounds():
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

    assert int(schedule.num_stages) == 3
    np.testing.assert_array_equal(np.asarray(schedule.num_tokens[:3]), [13, 3, 8])
    np.testing.assert_array_equal(
        np.asarray(schedule.rank_valid_rows[:3]),
        [
            [4, 4, 4, 1],
            [0, 0, 0, 3],
            [4, 4, 0, 0],
        ],
    )
    np.testing.assert_array_equal(
        np.asarray(schedule.rank_recv_start[:3]),
        [
            [0, 4, 8, 12],
            [0, 4, 8, 12],
            [0, 4, 8, 12],
        ],
    )
    # Rank-local request segments stay compact across stage boundaries and
    # across requests even though their absolute positions are unrelated.
    np.testing.assert_array_equal(
        np.asarray(schedule.rank_row_start[:3]),
        [
            [0, 0, 0, 0],
            [4, 4, 4, 1],
            [4, 4, 4, 4],
        ],
    )
    assert int(metadata.num_tiles) == 7
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_to_r_base.data[:7]),
        [0, 4, 8, 12, 13, 16, 20],
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_to_r_size.data[:7]),
        [4, 4, 4, 1, 3, 4, 4],
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_is_first_tile.data[:7]),
        [True, False, False, False, True, False, False],
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.p_id_is_last_tile.data[:7]),
        [False, False, False, True, False, False, True],
    )


def test_dual_coordinates_separate_token_ownership_from_absolute_state():
    cfg = _cfg(batch_size=128, tile_size=4)
    seq_lens, query_start, state_indices, distribution = _metadata_inputs(
        lengths=[13, 11],
        offsets=[7, 22],
        max_num_seqs=4,
    )
    request_absolute_starts = jnp.asarray([7, 22, 0, 0], dtype=jnp.int32)
    # Batch-flat request 1 starts inside rank 3's owner chunk [12, 16), while
    # its absolute sequence position remains 22 for GDN state semantics.
    batch_flat_owner_starts = jnp.asarray([0, 13, 0, 0], dtype=jnp.int32)
    fresh_absolute_starts = jnp.zeros((4,), dtype=jnp.int32)

    common_kwargs = dict(
        cfg=cfg,
        query_start_loc=query_start,
        state_indices=state_indices,
        start_seq=distribution[0],
        end_seq=distribution[-1],
        pcp_size=4,
        comm_chunk_size=4,
        projection_token_block_size=16,
        num_qkv_out_blocks=6,
        num_projection_out_blocks=8,
    )
    public_metadata, public_schedule = pcp_metadata.compute_pcp_stage_metadata(
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
    _, request_absolute_schedule = (
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            token_owner_starts=request_absolute_starts,
            request_absolute_starts=request_absolute_starts,
            **common_kwargs,
        )
    )
    owner_metadata, owner_schedule = (
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            token_owner_starts=batch_flat_owner_starts,
            request_absolute_starts=request_absolute_starts,
            **common_kwargs,
        )
    )
    fresh_metadata, fresh_schedule = (
        pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            token_owner_starts=batch_flat_owner_starts,
            request_absolute_starts=fresh_absolute_starts,
            **common_kwargs,
        )
    )

    # The public path now supplies batch-flat owner starts while retaining the
    # independently reconstructed request-absolute starts for state semantics.
    _assert_pytree_array_equal(public_metadata, owner_metadata)
    _assert_pytree_array_equal(public_schedule, owner_schedule)

    assert int(owner_schedule.num_stages) == 3
    np.testing.assert_array_equal(np.asarray(owner_schedule.request_id[:3]), [0, 1, 1])
    np.testing.assert_array_equal(
        np.asarray(owner_schedule.query_row_start[:3]), [0, 13, 16]
    )
    np.testing.assert_array_equal(np.asarray(owner_schedule.num_tokens[:3]), [13, 3, 8])
    np.testing.assert_array_equal(
        np.asarray(owner_schedule.rank_valid_rows[:3]),
        [[4, 4, 4, 1], [0, 0, 0, 3], [4, 4, 0, 0]],
    )
    assert not np.array_equal(
        np.asarray(owner_schedule.num_tokens[:3]),
        np.asarray(request_absolute_schedule.num_tokens[:3]),
    )

    # Changing only absolute starts cannot change ownership, rank rows, or
    # exchange stages, but it does change initial-state handling.
    _assert_pytree_array_equal(owner_schedule, fresh_schedule)
    np.testing.assert_array_equal(
        np.asarray(owner_metadata.s_idx_has_initial_state),
        [True, True, False, False],
    )
    np.testing.assert_array_equal(
        np.asarray(fresh_metadata.s_idx_has_initial_state),
        [False, False, False, False],
    )
    np.testing.assert_array_equal(
        np.asarray(owner_metadata.p_id_to_r_base.data),
        np.asarray(fresh_metadata.p_id_to_r_base.data),
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


def test_projection_schedule_catches_up_only_the_imbalanced_owner_rank():
    cfg = _cfg(batch_size=8 * 4096)
    seq_lens, query_start, state_indices, distribution = _metadata_inputs(
        # The private coordinate helper below explicitly resets ownership for
        # each request, so all live rows belong to PCP rank zero. This retains
        # coverage of the projection catch-up path independently of the public
        # batch-flat ownership policy.
        lengths=[256] * 5,
        offsets=[0] * 5,
        max_num_seqs=16,
    )

    metadata, schedule = pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
        cfg,
        query_start,
        state_indices,
        distribution[0],
        distribution[-1],
        token_owner_starts=jnp.zeros_like(seq_lens),
        request_absolute_starts=jnp.zeros_like(seq_lens),
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


# Production PCP8 geometry: C = 128 (stage of 1024 rows), 64-row GDN tiles.
_CKPT_PCP_SIZE = 8
_CKPT_COMM_CHUNK = 128
_CKPT_NUM_QKV_OUT_BLOCKS = _CKPT_PCP_SIZE
_CKPT_NUM_PROJECTION_OUT_BLOCKS = 2 * _CKPT_PCP_SIZE


def _prefill_ckpt_case(lengths, ckpt_offsets, *, max_num_seqs):
    seq_lens, query_start, state_indices, distribution = _metadata_inputs(
        lengths=lengths,
        offsets=[0] * len(lengths),
        max_num_seqs=max_num_seqs,
    )
    ckpt = np.zeros((max_num_seqs,), dtype=np.int32)
    ckpt[: len(ckpt_offsets)] = ckpt_offsets
    return (seq_lens, query_start, state_indices, distribution), jnp.asarray(ckpt)


def _build_pcp8(cfg, inputs, prefill_ckpt_offsets=None):
    seq_lens, query_start, state_indices, distribution = inputs
    return pcp_metadata.compute_pcp_stage_metadata(
        cfg,
        seq_lens,
        query_start,
        state_indices,
        distribution[0],
        distribution[-1],
        pcp_size=_CKPT_PCP_SIZE,
        comm_chunk_size=_CKPT_COMM_CHUNK,
        projection_token_block_size=_CKPT_COMM_CHUNK,
        num_qkv_out_blocks=_CKPT_NUM_QKV_OUT_BLOCKS,
        num_projection_out_blocks=_CKPT_NUM_PROJECTION_OUT_BLOCKS,
        prefill_ckpt_offsets=prefill_ckpt_offsets,
    )


def _active_tiles(metadata, schedule):
    num_tiles = int(metadata.num_tiles)
    columns = (
        schedule.tile_stage,
        schedule.tile_row_in_stage,
        metadata.p_id_to_s_idx.data,
        metadata.p_id_to_r_base.data,
        metadata.p_id_to_r_size.data,
        metadata.p_id_is_first_tile.data,
        metadata.p_id_is_last_tile.data,
    )
    return [
        tuple(int(column[tile]) for column in map(np.asarray, columns))
        for tile in range(num_tiles)
    ]


def _expected_ckpt_tiles(base_tiles, query_start, ckpt_offsets):
    """Inserts a shadow before each tile that strictly contains row R - 1."""
    query_start = np.asarray(query_start)
    ckpt_offsets = np.asarray(ckpt_offsets)
    ckpt_rows = {}
    for s_idx, offset in enumerate(ckpt_offsets):
        query_len = query_start[s_idx + 1] - query_start[s_idx]
        if 0 < offset < query_len:
            ckpt_rows[s_idx] = int(query_start[s_idx] + offset)
    tiles, modes = [], []
    for tile in base_tiles:
        stage, row, s_idx, r_base, r_size, first, _ = tile
        ckpt_row = ckpt_rows.get(s_idx)
        if ckpt_row is not None and r_base < ckpt_row < r_base + r_size:
            first = int(r_base == query_start[s_idx])
            tiles.append((stage, row, s_idx, r_base, ckpt_row - r_base, first, 0))
            modes.append(pcp_metadata.PREFILL_CKPT_SHADOW)
        tiles.append(tile)
        modes.append(
            pcp_metadata.PREFILL_CKPT_EXPORT
            if ckpt_row == r_base + r_size
            else pcp_metadata.PREFILL_CKPT_NONE
        )
    return tiles, modes, ckpt_rows


def _assert_ckpt_schedule(base, ckpt, query_start, ckpt_offsets, *, max_num_seqs):
    base_metadata, base_schedule = base
    metadata, schedule = ckpt
    assert base_schedule.tile_ckpt_mode is None
    assert schedule.tile_ckpt_mode is not None

    # Shadows take at most one extra tile per request.
    base_capacity = base_schedule.tile_stage.shape[0]
    for tile_array in (
        schedule.tile_stage,
        schedule.tile_row_in_stage,
        schedule.tile_ckpt_mode,
        metadata.p_id_to_s_idx.data,
        metadata.p_id_to_r_base.data,
        metadata.p_id_to_r_size.data,
        metadata.p_id_is_first_tile.data,
        metadata.p_id_is_last_tile.data,
    ):
        assert tile_array.shape == (base_capacity + max_num_seqs,)
    assert int(metadata.num_tiles) <= base_capacity + max_num_seqs

    # Checkpoints never change communication stages.
    for name in (
        "num_stages",
        "rank_active_row_end",
        "request_id",
        "query_row_start",
        "num_tokens",
        "rank_row_start",
        "rank_valid_rows",
        "rank_recv_start",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(schedule, name)),
            np.asarray(getattr(base_schedule, name)),
            err_msg=name,
        )

    base_tiles = _active_tiles(base_metadata, base_schedule)
    tiles = _active_tiles(metadata, schedule)
    expected_tiles, expected_modes, ckpt_rows = _expected_ckpt_tiles(
        base_tiles, query_start, ckpt_offsets
    )
    assert tiles == expected_tiles
    modes = np.asarray(schedule.tile_ckpt_mode)
    np.testing.assert_array_equal(modes[: len(tiles)], expected_modes)
    np.testing.assert_array_equal(modes[len(tiles) :], 0)

    # Every original tile is still present, in order.
    assert [
        tile
        for tile, mode in zip(tiles, modes)
        if mode != pcp_metadata.PREFILL_CKPT_SHADOW
    ] == base_tiles

    num_stages = int(schedule.num_stages)
    first_tile = np.asarray(schedule.first_tile)[:num_stages]
    stage_num_tiles = np.asarray(schedule.num_tiles)[:num_stages]
    stages = np.asarray([tile[0] for tile in tiles])
    for stage in range(num_stages):
        (indices,) = np.nonzero(stages == stage)
        assert first_tile[stage] == indices[0]
        assert stage_num_tiles[stage] == indices.size
        np.testing.assert_array_equal(
            indices, np.arange(indices[0], indices[0] + indices.size)
        )

    # Exactly one checkpoint action per checkpoint row.
    (action_tiles,) = np.nonzero(modes)
    assert sorted(tiles[tile][2] for tile in action_tiles) == sorted(ckpt_rows)
    for tile in action_tiles:
        stage, row, s_idx, r_base, r_size, first, last = tiles[tile]
        assert r_base + r_size == ckpt_rows[s_idx]
        if modes[tile] == pcp_metadata.PREFILL_CKPT_SHADOW:
            real = tiles[tile + 1]
            # Same stage, slot row and query row as the real tile, so the QKV
            # receive addressing is unchanged.
            assert (real[0], real[1], real[2], real[3]) == (stage, row, s_idx, r_base)
            assert r_size < real[4]
            assert not first and not last
            assert tile != first_tile[stage] + stage_num_tiles[stage] - 1
    return modes, tiles


@pytest.mark.parametrize(
    ("lengths", "ckpt_offsets", "expected"),
    [
        # Second row starts at 3000, so R = 3000 + 3072 = 6072 is not on the
        # 64-row tile grid of rank-slot starts: shadow [6016, 6072).
        pytest.param(
            [3000, 8191],
            [0, 3072],
            {6072: (pcp_metadata.PREFILL_CKPT_SHADOW, 6016, 56)},
            id="unaligned-query-start",
        ),
        pytest.param(
            [3072, 8191],
            [0, 3072],
            {6144: (pcp_metadata.PREFILL_CKPT_EXPORT, 6080, 64)},
            id="aligned-query-start",
        ),
        # 8191 tokens with eagle drop checkpoint at 3072; without it at 6144.
        pytest.param(
            [8191, 3000, 8191],
            [3072, 0, 6144],
            {
                3072: (pcp_metadata.PREFILL_CKPT_EXPORT, 3008, 64),
                17335: (pcp_metadata.PREFILL_CKPT_SHADOW, 17280, 55),
            },
            id="mixed",
        ),
    ],
)
def test_prefill_ckpt_emits_shadow_or_export_tile(lengths, ckpt_offsets, expected):
    max_num_seqs = 8
    cfg = _cfg(batch_size=2 * 16384)
    inputs, ckpt = _prefill_ckpt_case(lengths, ckpt_offsets, max_num_seqs=max_num_seqs)
    base = _build_pcp8(cfg, inputs)
    with_ckpt = _build_pcp8(cfg, inputs, ckpt)

    modes, tiles = _assert_ckpt_schedule(
        base, with_ckpt, inputs[1], ckpt, max_num_seqs=max_num_seqs
    )
    actions = {
        tiles[tile][3] + tiles[tile][4]: (
            int(modes[tile]),
            tiles[tile][3],
            tiles[tile][4],
        )
        for tile in np.nonzero(modes)[0]
    }
    assert actions == expected
    num_shadows = sum(
        mode == pcp_metadata.PREFILL_CKPT_SHADOW for mode, _, _ in expected.values()
    )
    assert int(with_ckpt[0].num_tiles) == int(base[0].num_tiles) + num_shadows


def test_prefill_ckpt_none_and_invalid_offsets_keep_the_schedule():
    max_num_seqs = 8
    cfg = _cfg(batch_size=2 * 16384)
    # Offset 0, an offset equal to the query length and an offset in a padded
    # row past end_seq request nothing.
    inputs, ckpt = _prefill_ckpt_case(
        [3000, 1024, 600], [0, 1024, 0], max_num_seqs=max_num_seqs
    )
    ckpt = ckpt.at[5].set(3072)
    base = _build_pcp8(cfg, inputs)
    explicit_none = _build_pcp8(cfg, inputs, None)
    with_ckpt = _build_pcp8(cfg, inputs, ckpt)

    _assert_pytree_array_equal(explicit_none, base)
    modes, _ = _assert_ckpt_schedule(
        base, with_ckpt, inputs[1], ckpt, max_num_seqs=max_num_seqs
    )
    np.testing.assert_array_equal(modes, 0)
    assert int(with_ckpt[0].num_tiles) == int(base[0].num_tiles)


@pytest.mark.parametrize("seed", range(4))
def test_prefill_ckpt_keeps_every_tile_with_a_shadow_in_every_request(seed):
    # Small owner chunks put many rank-slot boundaries inside each request.
    max_num_seqs = 16
    pcp_size = 4
    comm_chunk_size = 16
    cfg = _cfg(batch_size=pcp_size * 1024, tile_size=16)
    rng = np.random.default_rng(seed)
    lengths = rng.integers(70, 250, size=max_num_seqs)
    assert np.sum(lengths) <= cfg.batch_size
    # The builder takes any row, not only the 64-multiples the scheduler
    # sends. Batch-flat rows congruent to 8 mod 16 fall mid-tile on the
    # 16-row tile grid, so every request needs a shadow tile.
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    ckpt_rows = (starts + 8) // 16 * 16 + 8
    ckpt_rows += 16 * rng.integers(1, (starts + lengths - 8 - ckpt_rows) // 16 + 1)
    ckpt_offsets = ckpt_rows - starts
    assert np.all((ckpt_offsets > 0) & (ckpt_offsets < lengths))
    seq_lens, query_start, state_indices, distribution = _metadata_inputs(
        lengths=lengths,
        offsets=[0] * max_num_seqs,
        max_num_seqs=max_num_seqs,
    )

    def build(prefill_ckpt_offsets):
        return pcp_metadata.compute_pcp_stage_metadata(
            cfg,
            seq_lens,
            query_start,
            state_indices,
            distribution[0],
            distribution[-1],
            pcp_size=pcp_size,
            comm_chunk_size=comm_chunk_size,
            projection_token_block_size=comm_chunk_size,
            num_qkv_out_blocks=pcp_size,
            num_projection_out_blocks=2 * pcp_size,
            prefill_ckpt_offsets=prefill_ckpt_offsets,
        )

    base = build(None)
    ckpt = jnp.asarray(ckpt_offsets, dtype=jnp.int32)
    with_ckpt = jax.jit(build)(ckpt)
    modes, _ = _assert_ckpt_schedule(
        base, with_ckpt, query_start, ckpt, max_num_seqs=max_num_seqs
    )
    assert np.count_nonzero(modes == pcp_metadata.PREFILL_CKPT_SHADOW) == max_num_seqs
    assert int(with_ckpt[0].num_tiles) == int(base[0].num_tiles) + max_num_seqs


def _simulate_projection_queue(
    metadata,
    schedule,
    *,
    rank,
    projection_token_block_size,
    num_qkv_out_blocks,
    num_projection_out_blocks,
):
    """Replays the kernel's projection work order for one rank."""
    num_stages = int(schedule.num_stages)
    num_tiles = int(metadata.num_tiles)
    tile_stage = np.asarray(schedule.tile_stage)
    first_tile = np.asarray(schedule.first_tile)
    rank_row_start = np.asarray(schedule.rank_row_start)
    rank_valid_rows = np.asarray(schedule.rank_valid_rows)
    catchup_start = np.asarray(schedule.projection_catchup_start)
    catchup_count = np.asarray(schedule.projection_catchup_count)
    work_offset = np.asarray(schedule.projection_work_offset)
    work_offset_end = np.asarray(schedule.projection_work_offset_end)
    num_z_out_blocks = num_projection_out_blocks - num_qkv_out_blocks

    def cdiv(a, b):
        return -(-int(a) // b)

    active_blocks = cdiv(
        np.asarray(schedule.rank_active_row_end)[rank], projection_token_block_size
    )
    total_work = (
        num_z_out_blocks + max(active_blocks - 1, 0) * num_projection_out_blocks
        if active_blocks > 0
        else 0
    )
    done = []

    def run_catchup(stage):
        start = int(catchup_start[stage, rank])
        done.extend(range(start, start + int(catchup_count[stage, rank])))

    def launch_qkv(stage):
        required_blocks = cdiv(
            rank_row_start[stage, rank] + rank_valid_rows[stage, rank],
            projection_token_block_size,
        )
        required = (
            0
            if required_blocks <= 1
            else num_z_out_blocks
            + (required_blocks - 2) * num_projection_out_blocks
            + num_qkv_out_blocks
        )
        assert done == list(range(len(done)))
        assert len(done) >= required, (stage, len(done), required)

    if num_stages > 0:
        run_catchup(0)
        launch_qkv(0)
    for tile in range(num_tiles):
        stage = int(tile_stage[tile])
        if tile == first_tile[stage] and stage + 1 < num_stages:
            run_catchup(stage + 1)
            launch_qkv(stage + 1)
        work_id = tile + int(work_offset[stage, rank])
        if work_id < total_work:
            done.append(work_id)
    done.extend(range(num_tiles + int(work_offset_end[rank]), total_work))
    assert done == list(range(total_work))
    return int(np.sum(catchup_count[:num_stages, rank]))


def test_prefill_ckpt_shadows_keep_projection_work_exactly_once():
    cfg = _cfg(batch_size=8 * 4096)
    lengths = [312] * 5
    seq_lens, query_start, state_indices, distribution = _metadata_inputs(
        lengths=lengths,
        offsets=[0] * 5,
        max_num_seqs=16,
    )
    # Every request begins 200 rows into rank 0's owner chunk, so its tiles
    # in rank 1 start 56 rows into the request and row 128 is mid-tile.
    owner_starts = jnp.asarray([200] * 5 + [0] * 11, dtype=jnp.int32)
    ckpt = jnp.asarray([128] * 5 + [0] * 11, dtype=jnp.int32)

    def build(prefill_ckpt_offsets):
        return pcp_metadata._compute_pcp_stage_metadata_from_coordinates(
            cfg,
            query_start,
            state_indices,
            distribution[0],
            distribution[-1],
            token_owner_starts=owner_starts,
            request_absolute_starts=jnp.zeros_like(seq_lens),
            pcp_size=8,
            comm_chunk_size=256,
            projection_token_block_size=_PROJECTION_TOKEN_BLOCK_SIZE,
            num_qkv_out_blocks=_NUM_QKV_OUT_BLOCKS,
            num_projection_out_blocks=_NUM_PROJECTION_OUT_BLOCKS,
            prefill_ckpt_offsets=prefill_ckpt_offsets,
        )

    base = build(None)
    with_ckpt = build(ckpt)
    modes, _ = _assert_ckpt_schedule(
        base, with_ckpt, query_start, ckpt, max_num_seqs=16
    )
    assert np.count_nonzero(modes == pcp_metadata.PREFILL_CKPT_SHADOW) == 5

    catchups = {}
    for name, (metadata, schedule) in (("base", base), ("ckpt", with_ckpt)):
        for rank in range(8):
            catchups[name, rank] = _simulate_projection_queue(
                metadata,
                schedule,
                rank=rank,
                projection_token_block_size=_PROJECTION_TOKEN_BLOCK_SIZE,
                num_qkv_out_blocks=_NUM_QKV_OUT_BLOCKS,
                num_projection_out_blocks=_NUM_PROJECTION_OUT_BLOCKS,
            )
    # Rank 1 owns 1280 rows, so stage 4 needs token block 1 and catches up.
    # The shadows add body tiles before that launch, which shortens it.
    assert catchups["base", 1] > catchups["ckpt", 1] > 0


@pytest.mark.parametrize(
    ("lengths", "ckpt_offsets"),
    [
        pytest.param([3000, 8191], [0, 3072], id="unaligned-query-start"),
        pytest.param([8191, 3000, 8191], [3072, 0, 6144], id="mixed"),
    ],
)
def test_prefill_ckpt_pcp8_projection_work_exactly_once(lengths, ckpt_offsets):
    cfg = _cfg(batch_size=2 * 16384)
    inputs, ckpt = _prefill_ckpt_case(lengths, ckpt_offsets, max_num_seqs=8)
    for metadata, schedule in (
        _build_pcp8(cfg, inputs),
        _build_pcp8(cfg, inputs, ckpt),
    ):
        for rank in range(_CKPT_PCP_SIZE):
            _simulate_projection_queue(
                metadata,
                schedule,
                rank=rank,
                projection_token_block_size=_CKPT_COMM_CHUNK,
                num_qkv_out_blocks=_CKPT_NUM_QKV_OUT_BLOCKS,
                num_projection_out_blocks=_CKPT_NUM_PROJECTION_OUT_BLOCKS,
            )
