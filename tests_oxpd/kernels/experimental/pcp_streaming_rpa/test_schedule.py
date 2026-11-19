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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    TilePlanField,
    build_pcp_streaming_schedule_inputs_from_metadata_jax,
    query_row_positions,
)

pytestmark = pytest.mark.multichip


def test_production_schedule_trace_remains_compact():
    """Do not statically unroll every request/tile/rank schedule slot."""

    def build(kv_lens, page_indices, cu_q_lens, distribution):
        return build_pcp_streaming_schedule_inputs_from_metadata_jax(
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            global_bucket_tokens=32_768,
            local_kv_cache_num_blocks=128,
            page_size=768,
            pcp_size=8,
            interleave_size=256,
            q_block_size=512,
        )

    traced = jax.make_jaxpr(build)(
        jnp.zeros((64,), dtype=jnp.int32),
        jnp.zeros((64, 128), dtype=jnp.int32),
        jnp.zeros((65,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
    )

    # The vectorized scheduler is currently about 365 top-level equations.
    # The former 64 x 10 x 8 Python loops produced hundreds of thousands.
    assert len(traced.jaxpr.eqns) < 1_000


def test_metadata_jax_compact_plan_tracks_group_starts():
    pcp_size = 2
    (
        tile_plan,
        block_tables,
        current_ids,
        current_starts,
        current_meta,
        history_ids,
        history_starts,
        history_meta,
    ) = build_pcp_streaming_schedule_inputs_from_metadata_jax(
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
    logical = tile_plan[:, 0, : pcp_size * TilePlanField.NUM_FIELDS].reshape(
        tile_plan.shape[0], pcp_size, TilePlanField.NUM_FIELDS
    )
    assert tile_plan.shape == (6, 1, TilePlanField.HBM_ROW_FIELDS)
    assert np.asarray(block_tables).shape == (2, 16)

    for ids, starts, meta, field in (
        (current_ids, current_starts, current_meta, TilePlanField.CURRENT_NUM_GROUPS),
        (history_ids, history_starts, history_meta, TilePlanField.HISTORY_NUM_GROUPS),
    ):
        groups = logical[:, 0, field]
        expected_ids = np.flatnonzero(groups > 0)
        expected_starts = np.cumsum(groups[expected_ids]) - groups[expected_ids]
        count = int(np.asarray(meta)[0])
        np.testing.assert_array_equal(np.asarray(ids)[0, :count], expected_ids)
        np.testing.assert_array_equal(np.asarray(starts)[0, :count], expected_starts)
        assert int(np.asarray(meta)[1]) == int(groups.sum())


def test_metadata_jax_moves_partial_history_page_to_current_pass():
    pcp_size = 8
    history_tokens = 25_600
    current_tokens = 32_768
    (
        tile_plan,
        _block_tables,
        _current_ids,
        _current_starts,
        current_meta,
        _history_ids,
        _history_starts,
        history_meta,
    ) = build_pcp_streaming_schedule_inputs_from_metadata_jax(
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

    logical = np.asarray(tile_plan)[
        :, 0, : pcp_size * TilePlanField.NUM_FIELDS
    ].reshape(tile_plan.shape[0], pcp_size, TilePlanField.NUM_FIELDS)
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
    np.testing.assert_array_equal(
        np.asarray(current_meta), np.array([8, 80], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(history_meta), np.array([8, 96], dtype=np.int32)
    )


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
    tile_plan, *_ = build_pcp_streaming_schedule_inputs_from_metadata_jax(**kwargs)
    logical = np.asarray(tile_plan)[:, 0, : 2 * TilePlanField.NUM_FIELDS].reshape(
        tile_plan.shape[0], 2, TilePlanField.NUM_FIELDS
    )
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
def test_metadata_jax_batch_flat_owner_preserves_long_absolute_history(history_tokens):
    pcp_size = 4
    q_len = 256
    page_size = 128
    interleave_size = 32
    q_block_size = 64
    kv_len = history_tokens + q_len
    local_kv_cache_num_blocks = (kv_len + pcp_size * page_size - 1) // (
        pcp_size * page_size
    )
    (
        tile_plan,
        _block_tables,
        _current_ids,
        _current_starts,
        current_meta,
        _history_ids,
        _history_starts,
        history_meta,
    ) = build_pcp_streaming_schedule_inputs_from_metadata_jax(
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
    logical = np.asarray(tile_plan)[
        :, 0, : pcp_size * TilePlanField.NUM_FIELDS
    ].reshape(tile_plan.shape[0], pcp_size, TilePlanField.NUM_FIELDS)
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
def test_metadata_jax_schedule_rejects_nonpositive_interleave_size(interleave_size):
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


@pytest.mark.parametrize("length,groups", [(8192, 32), (8191, 39)])
def test_partial_interleaves_share_query_tiles(length, groups):
    """P8191 must not replay the ring for each tiny boundary fragment."""
    build = jax.jit(
        build_pcp_streaming_schedule_inputs_from_metadata_jax,
        static_argnames=(
            "global_bucket_tokens",
            "local_kv_cache_num_blocks",
            "page_size",
            "pcp_size",
            "interleave_size",
            "q_block_size",
        ),
    )
    result = build(
        kv_lens=np.full(8, length, np.int32),
        page_indices=np.zeros((8, 8), np.int32),
        cu_q_lens=np.arange(9, dtype=np.int32) * length,
        distribution=np.array([0, 0, 8], np.int32),
        global_bucket_tokens=65536,
        local_kv_cache_num_blocks=128,
        page_size=128,
        pcp_size=16,
        interleave_size=128,
        q_block_size=512,
    )
    np.testing.assert_array_equal(np.asarray(result[4]), [8, groups])


@pytest.mark.parametrize("pcp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("history", [0, 17, 128 * 1024])
def test_packed_tiles_cover_exact_batch_flat_query_rows(pcp_size, history):
    lengths = np.array([1, 127, 129, 8191, 8192, 8193, 0], np.int32)
    cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    interleave, block = 128, 512
    bucket = (
        ((int(cu[-1]) + pcp_size * block - 1) // (pcp_size * block)) * pcp_size * block
    )
    result = build_pcp_streaming_schedule_inputs_from_metadata_jax(
        kv_lens=lengths + history,
        page_indices=np.zeros((len(lengths), 256), np.int32),
        cu_q_lens=cu,
        distribution=np.array([0, 0, len(lengths) - 1], np.int32),
        global_bucket_tokens=bucket,
        local_kv_cache_num_blocks=256,
        page_size=256,
        pcp_size=pcp_size,
        interleave_size=interleave,
        q_block_size=block,
    )
    plan = np.asarray(result[0])[:, 0, : pcp_size * TilePlanField.NUM_FIELDS].reshape(
        -1, pcp_size, TilePlanField.NUM_FIELDS
    )
    prefixes = np.zeros(pcp_size, np.int32)
    for req, length in enumerate(lengths):
        request_tiles = plan[plan[:, 0, TilePlanField.REQ_ID] == req]
        active = request_tiles[:, 0, TilePlanField.CURRENT_NUM_GROUPS] > 0
        request_tiles = request_tiles[active]
        owned_counts = []
        for rank in range(pcp_size):
            global_rows = np.arange(cu[req], cu[req + 1])
            expected = global_rows[(global_rows // interleave) % pcp_size == rank]
            owned_counts.append(len(expected))
            positions, offsets = [], []
            for tile in request_tiles[:, rank]:
                size = int(tile[TilePlanField.Q_TILE_SIZE])
                offset = int(tile[TilePlanField.Q_HBM_OFFSET])
                rows = np.arange(size)
                positions.extend(
                    query_row_positions(
                        tile[TilePlanField.Q_GLOBAL_START],
                        offset,
                        rows,
                        pcp_size=pcp_size,
                        interleave_size=interleave,
                    )
                )
                offsets.extend(offset + rows)
            np.testing.assert_array_equal(positions, history + expected - cu[req])
            np.testing.assert_array_equal(
                offsets, prefixes[rank] + np.arange(len(expected))
            )
            prefixes[rank] += len(expected)
        assert len(request_tiles) == (max(owned_counts) + block - 1) // block
        if length:
            # The final tile alone captures the full fresh KV, even if some
            # ranks have already exhausted their local query rows.
            np.testing.assert_array_equal(
                request_tiles[:, 0, TilePlanField.CAPTURE_CURRENT_KV],
                np.arange(len(request_tiles)) == len(request_tiles) - 1,
            )
            assert request_tiles[-1, 0, TilePlanField.CURRENT_EFFECTIVE_LEN] == length


@pytest.mark.parametrize("pcp_size", [2, 4])
@pytest.mark.parametrize("history", [0, 17])
def test_packed_query_causality_against_dense_attention(pcp_size, history):
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.reference import (
        build_runtime_schedule_reference,
    )
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
        RuntimeScheduleField as R,
    )

    lengths = np.array([7, 19, 9], np.int32)
    cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    interleave, block, page = 2, 8, 4
    bucket = (
        ((int(cu[-1]) + pcp_size * block - 1) // (pcp_size * block)) * pcp_size * block
    )
    result = build_pcp_streaming_schedule_inputs_from_metadata_jax(
        kv_lens=lengths + history,
        page_indices=np.zeros((3, 16), np.int32),
        cu_q_lens=cu,
        distribution=np.array([0, 0, 3], np.int32),
        global_bucket_tokens=bucket,
        local_kv_cache_num_blocks=16,
        page_size=page,
        pcp_size=pcp_size,
        interleave_size=interleave,
        q_block_size=block,
    )
    reference = build_runtime_schedule_reference(
        np.asarray(result[0]),
        np.asarray(result[1]),
        pcp_size=pcp_size,
        page_size=page,
        interleave_size=interleave,
    )
    plan = np.asarray(result[0])[:, 0, : pcp_size * TilePlanField.NUM_FIELDS].reshape(
        -1, pcp_size, TilePlanField.NUM_FIELDS
    )
    rng = np.random.default_rng(42)
    for tile_idx, tile in enumerate(plan):
        req = int(tile[0, TilePlanField.REQ_ID])
        for rank, row in enumerate(tile):
            size = int(row[TilePlanField.Q_TILE_SIZE])
            if not size:
                continue
            positions = query_row_positions(
                row[TilePlanField.Q_GLOBAL_START],
                row[TilePlanField.Q_HBM_OFFSET],
                np.arange(size),
                pcp_size=pcp_size,
                interleave_size=interleave,
            )
            kv_positions = []
            for schedules in (reference.current_rows, reference.history_rows):
                for group in schedules[tile_idx]:
                    for entry in group[:, rank]:
                        if entry[R.REQ_ID] < 0:
                            continue
                        local = np.arange(entry[R.KV_VALID_LEN])
                        kv_positions.extend(
                            entry[R.KV_GLOBAL_START]
                            + local // interleave * pcp_size * interleave
                            + local % interleave
                        )
            kv_positions = np.asarray(kv_positions, np.int32)
            q = rng.normal(size=(size, 8))
            k = rng.normal(size=(history + lengths[req], 8))
            v = rng.normal(size=(history + lengths[req], 8))
            for i, position in enumerate(positions):
                visible = kv_positions[kv_positions <= position]
                np.testing.assert_array_equal(np.sort(visible), np.arange(position + 1))
                scores = q[i] @ k[visible].T
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                actual = probs @ v[visible]
                scores = q[i] @ k[: position + 1].T
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                np.testing.assert_allclose(
                    actual, probs @ v[: position + 1], atol=1e-12
                )
