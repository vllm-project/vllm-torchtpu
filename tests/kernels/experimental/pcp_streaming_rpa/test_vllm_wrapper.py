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
"""Unit tests for PCP streaming RPA vLLM wrapper, stage plan, and Splash dispatch."""

import dataclasses
import inspect
import math
import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.experimental.pcp_streaming_rpa.test_splash_attention_kernel import (
    _compute_fp64_attention,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import kernel as pcp_kernel
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_kernel as splash_kernel_mod,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_mask_info as splash_mask_info_mod,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    stage_plan as pcp_stage_plan,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as pcp_wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    _attend_multi_vpage_groups,
    _attend_over_vpage_groups,
    _gather_vpage_group_kv_and_seg,
    _gather_vpages_to_head_major,
    _merge_attention_partials,
    _SplashStreamSpec,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.stage_plan import (
    _build_writeback_segment_descriptors,
    _compact_writeback_slot_ids_for_cache_rank,
    _compute_pcp_capture_source_rows_for_cache_rank,
    _compute_pcp_kv_writeback_mapping,
    _compute_pcp_local_query_row_mapping,
    _compute_pcp_local_query_row_mapping_from_metadata,
    _compute_pcp_rank_major_query_row_mapping_from_metadata,
    _LocalQueryRowSpec,
    _PCPKVWritebackMapping,
    _PCPQueryRowMapping,
    _RankMajorQueryRowSpec,
    _SlotMappingSpec,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
    _pack_kv_for_cache,
)


@dataclasses.dataclass(frozen=True)
class _MeshStub:
    """Typed stub for jax.sharding.Mesh."""

    axis_names: tuple[str, ...]
    shape: dict[str, int]


@dataclasses.dataclass(frozen=True)
class _PCPStagePlanStub:
    """Typed stub for PCPStagePlan in wrapper and kernel dispatch unit tests."""

    cache_owner_ranks: jax.Array | None = None
    cache_local_slot_ids: jax.Array | None = None
    capture_offsets: jax.Array | None = None
    writeback_counts: jax.Array | None = None
    compact_slot_ids: jax.Array | None = None
    needs_reroute: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.bool_)
    )
    q_seg_ids: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((128,), dtype=jnp.int32)
    )
    block_tables: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((1, 1), dtype=jnp.int32)
    )
    query_row_valid: jax.Array | None = None
    segment_descriptors: jax.Array | None = None
    num_segments: jax.Array | None = None
    max_writeback_count: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )
    q_sequence: jax.Array | None = None
    sorted_block_ids: jax.Array | None = None
    sorted_seg_ids: jax.Array | None = None
    sorted_vpage_valid_tokens: jax.Array | None = None
    num_active_vpages: jax.Array | None = None
    pcp_rank: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.asarray(0, dtype=jnp.int32)
    )


@dataclasses.dataclass(frozen=True)
class _MultiGroupVpageInputs:
    """Container for multi-group vpage attention test inputs."""

    kv_heads: int
    total_vpages: int
    virtual_page_size: int
    head_dim: int
    k_req0: np.ndarray
    v_req0: np.ndarray
    k_req1: np.ndarray
    v_req1: np.ndarray
    req0_kv_len: int
    req1_kv_len: int


def _build_expected_pcp_writeback_mapping(
    q_lens: np.ndarray,
    seq_lens: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
    local_kv_cache_num_blocks: int,
):
    """Independent NumPy reference for dense ring-capture destinations."""
    q_lens = np.asarray(q_lens, dtype=np.int64)
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int64) % int(
        local_kv_cache_num_blocks
    )
    owner_starts = np.zeros(q_lens.shape, dtype=np.int64)
    if q_lens.size > 1:
        np.cumsum(q_lens[:-1], out=owner_starts[1:])
    token_order, _ = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )

    cache_owners = np.full(padded_num_tokens, -1, dtype=np.int32)
    cache_slots = np.full(padded_num_tokens, -1, dtype=np.int32)
    capture_offsets = np.full(padded_num_tokens, -1, dtype=np.int32)
    writeback_counts = np.zeros(pcp_size, dtype=np.int32)
    compact_slots = np.full((pcp_size, padded_num_tokens), -1, dtype=np.int32)

    req_ends = np.cumsum(q_lens)
    absolute_starts = seq_lens - q_lens
    virtual_block_size = block_size * pcp_size
    for source_row, request_major_row in enumerate(token_order):
        if request_major_row < 0:
            continue
        req_id = int(np.searchsorted(req_ends, request_major_row, side="right"))
        req_start = 0 if req_id == 0 else int(req_ends[req_id - 1])
        absolute_position = (
            int(absolute_starts[req_id]) + int(request_major_row) - req_start
        )
        block_idx = absolute_position // virtual_block_size
        virtual_offset = absolute_position % virtual_block_size
        owner = (virtual_offset // interleave_size) % pcp_size
        local_offset = (
            virtual_offset // (pcp_size * interleave_size)
        ) * interleave_size + virtual_offset % interleave_size
        slot = int(block_tables[req_id, block_idx] * block_size + local_offset)
        cache_owners[source_row] = owner
        cache_slots[source_row] = slot

    for req_id in range(q_lens.size):
        req_rows = np.flatnonzero(
            (token_order >= (0 if req_id == 0 else req_ends[req_id - 1]))
            & (token_order < req_ends[req_id])
        )
        req_rows = req_rows[
            np.argsort(
                np.asarray(
                    [
                        absolute_starts[req_id]
                        + token_order[row]
                        - (0 if req_id == 0 else req_ends[req_id - 1])
                        for row in req_rows
                    ]
                )
            )
        ]
        for source_row in req_rows:
            owner = int(cache_owners[source_row])
            capture_offset = int(writeback_counts[owner])
            writeback_counts[owner] += 1
            capture_offsets[source_row] = capture_offset
            compact_slots[owner, capture_offset] = cache_slots[source_row]

    return _PCPStagePlanStub(
        cache_owner_ranks=jnp.asarray(cache_owners),
        cache_local_slot_ids=jnp.asarray(cache_slots),
        capture_offsets=jnp.asarray(capture_offsets),
        writeback_counts=jnp.asarray(writeback_counts),
        compact_slot_ids=jnp.asarray(compact_slots),
    )


def test_sharded_wrapper_uses_real_metadata_streaming_entry():
    source = inspect.getsource(pcp_wrapper)

    assert "pcp_streaming_attention_page_groups_packed_local_from_metadata" in source


@pytest.mark.parametrize(
    ("kv_dtype", "kv_heads", "q_per_kv", "packed_kv_groups", "k_scale", "v_scale"),
    [
        (jnp.bfloat16, 1, 1, 1, None, None),
        (jnp.bfloat16, 2, 1, 2, None, None),
        (jnp.bfloat16, 2, 2, 2, None, None),
        (jnp.float8_e4m3fn, 1, 2, 1, 0.5, 0.5),
        (jnp.float8_e4m3fn, 2, 1, 1, 0.5, 0.5),
        (jnp.float8_e4m3fn, 3, 2, 2, 0.5, 0.5),
    ],
)
def test_packed_kv_layout_validation_accepts_unified_head_matrix(
    kv_dtype, kv_heads, q_per_kv, packed_kv_groups, k_scale, v_scale
):
    local_tokens = 32
    page_size = 16
    head_dim = 128
    kv_packing = pcp_wrapper.get_dtype_packing(kv_dtype)
    q_local = jnp.zeros(
        (local_tokens, kv_heads, q_per_kv, head_dim), dtype=jnp.bfloat16
    )
    kv_cache_local = jnp.zeros(
        (2, page_size, packed_kv_groups, kv_packing, head_dim),
        dtype=kv_dtype,
    )

    assert pcp_kernel._validate_packed_kv_layout(
        q_local, kv_cache_local, k_scale=k_scale, v_scale=v_scale
    ) == (
        kv_heads,
        packed_kv_groups,
        kv_packing,
    )


def test_packed_kv_layout_validation_rejects_fp8_without_scales():
    q_local = jnp.zeros((32, 1, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((2, 16, 1, 4, 128), dtype=jnp.float8_e4m3fn)

    with pytest.raises(ValueError, match="FP8 KV cache requires"):
        pcp_kernel._validate_packed_kv_layout(q_local, kv_cache_local)


def test_packed_kv_layout_validation_rejects_wrong_physical_group_count():
    q_local = jnp.zeros((32, 3, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((2, 16, 1, 4, 128), dtype=jnp.float8_e4m3fn)

    with pytest.raises(ValueError, match="packed layout does not match"):
        pcp_kernel._validate_packed_kv_layout(
            q_local, kv_cache_local, k_scale=0.5, v_scale=0.5
        )


def test_seq_along_lane_layout_validation_accepts_fp8_multi_head():
    q_local = jnp.zeros((32, 2, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((2, 4, 32, 4, 128), dtype=jnp.float8_e4m3fn)

    assert pcp_kernel._validate_packed_kv_layout(
        q_local,
        kv_cache_local,
        kv_layout=pcp_wrapper.KVLayout.SEQ_ALONG_LANE,
        k_scale=0.5,
        v_scale=0.5,
    ) == (2, 4, 4)


def test_prepare_packed_kv_accepts_equivalent_tail_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _prepare_packed_kv_for_cache reshapes an equivalent packed tail."""
    packed_kv = jnp.arange(3 * 8 * 2 * 8, dtype=jnp.float32).reshape(3, 8, 2, 8)

    monkeypatch.setattr(
        pcp_wrapper, "_pack_kv_for_cache", lambda *_args, **_kwargs: packed_kv
    )

    kv_cache = jnp.zeros((2, 4, 16, 1, 8), dtype=jnp.float32)
    k = jnp.zeros((3, 8, 8), dtype=jnp.float32)
    v = jnp.zeros_like(k)
    prepared = pcp_wrapper._prepare_packed_kv_for_cache(k, v, kv_cache)
    expected = packed_kv.reshape(3, 16, 1, 8)
    np.testing.assert_array_equal(np.asarray(prepared), np.asarray(expected))


def test_seq_along_lane_kv_packing_pads_inside_pcp_wrapper() -> None:
    """Verify SEQ_ALONG_LANE packing pads token extent inside pcp_wrapper."""
    k = jnp.ones((384, 1, 128), dtype=jnp.bfloat16)
    padded = _pack_kv_for_cache(
        k,
        jnp.ones_like(k),
        jnp.float8_e4m3fn,
        kv_layout=pcp_wrapper.KVLayout.SEQ_ALONG_LANE,
        page_size=256,
    )

    assert padded.shape == (2, 32, 4, 512)
    np.testing.assert_array_equal(np.asarray(padded[..., :384]), 1)
    np.testing.assert_array_equal(np.asarray(padded[..., 384:]), 0)


def test_captured_kv_writeback_coalesces_contiguous_cache_slots():
    compact_slots = jnp.asarray([0, 1, 2, 8, 9, 20, -1, -1], dtype=jnp.int32)
    descriptors, num_segments = _build_writeback_segment_descriptors(
        compact_slots,
        jnp.asarray(6),
    )

    assert int(num_segments) == 3
    np.testing.assert_array_equal(
        np.asarray(descriptors),
        np.asarray(
            [
                [0, 3, 5, 0, 0, 0, 0, 0],
                [0, 8, 20, 0, 0, 0, 0, 0],
                [3, 2, 1, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_captured_kv_writeback_preserves_fully_fragmented_capacity():
    compact_slots = jnp.arange(16, dtype=jnp.int32) * 2
    descriptors, num_segments = _build_writeback_segment_descriptors(
        compact_slots,
        jnp.asarray(compact_slots.shape[0]),
    )

    assert int(num_segments) == compact_slots.shape[0]
    np.testing.assert_array_equal(
        np.asarray(descriptors[0]), np.arange(16, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(descriptors[1]), np.arange(16, dtype=np.int32) * 2
    )
    np.testing.assert_array_equal(
        np.asarray(descriptors[2]), np.ones(16, dtype=np.int32)
    )


def test_seq_along_lane_writeback_segments_stop_at_page_boundaries():
    compact_slots = jnp.arange(256, dtype=jnp.int32)
    descriptors, num_segments = _build_writeback_segment_descriptors(
        compact_slots,
        jnp.asarray(256),
        kv_layout=pcp_wrapper.KVLayout.SEQ_ALONG_LANE,
        page_size=128,
    )

    assert int(num_segments) == 2
    np.testing.assert_array_equal(
        np.asarray(descriptors[:, :2]),
        np.asarray([[0, 128], [0, 128], [128, 128]], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.asarray(descriptors[:, 2:]), 0)


def test_metadata_reconstruction_preserves_query_row_coordinates():
    kv_lens = jnp.asarray([8], dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, 5], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    mapping = _compute_pcp_local_query_row_mapping_from_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        spec=_LocalQueryRowSpec(
            local_padded_tokens=4,
            pcp_size=2,
            pcp_rank=0,
            interleave_size=2,
        ),
    )

    np.testing.assert_array_equal(
        np.asarray(mapping.absolute_positions),
        np.array([3, 4, 7, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(mapping.is_valid),
        np.array([True, True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(mapping.request_indices),
        np.array([0, 0, 0, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(mapping.source_ranks),
        np.zeros(4, dtype=np.int32),
    )


def test_query_row_mapping_separates_owner_and_request_absolute_starts():
    q_lens = jnp.asarray([3, 4], dtype=jnp.int32)
    token_owner_starts = jnp.asarray([0, 3], dtype=jnp.int32)
    request_absolute_starts = jnp.asarray([5, 100], dtype=jnp.int32)

    rank_0 = _compute_pcp_local_query_row_mapping(
        q_lens,
        token_owner_starts,
        request_absolute_starts,
        spec=_LocalQueryRowSpec(
            local_padded_tokens=4,
            pcp_size=2,
            pcp_rank=0,
            interleave_size=2,
        ),
    )
    rank_1 = _compute_pcp_local_query_row_mapping(
        q_lens,
        token_owner_starts,
        request_absolute_starts,
        spec=_LocalQueryRowSpec(
            local_padded_tokens=4,
            pcp_size=2,
            pcp_rank=1,
            interleave_size=2,
        ),
    )

    np.testing.assert_array_equal(
        np.asarray(rank_0.absolute_positions), np.array([5, 6, 101, 102])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_0.request_indices), np.array([0, 0, 1, 1])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_1.absolute_positions), np.array([7, 100, 103, -1])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_1.request_indices), np.array([0, 1, 1, -1])
    )
    np.testing.assert_array_equal(
        np.asarray(rank_1.is_valid), np.array([True, True, True, False])
    )
    # Source-rank identity remains available even when the row itself is pad.
    np.testing.assert_array_equal(
        np.asarray(rank_1.source_ranks), np.ones(4, dtype=np.int32)
    )


def test_kv_writeback_mapping_matches_independent_dense_capture_reference():
    pcp_size = 4
    page_size = 4
    interleave_size = 2
    local_padded_tokens = 16
    local_kv_cache_num_blocks = 16
    q_lens = np.array([5, 17, 6], dtype=np.int32)
    absolute_starts = np.array([3, 9, 0], dtype=np.int32)
    seq_lens = absolute_starts + q_lens
    cu_q_lens = np.concatenate(([0], np.cumsum(q_lens))).astype(np.int32)
    distribution = np.array([0, 0, q_lens.size], dtype=np.int32)
    block_tables = (
        np.arange(q_lens.size * 16, dtype=np.int32).reshape(q_lens.size, 16) + 7
    )

    query_rows = _compute_pcp_rank_major_query_row_mapping_from_metadata(
        jnp.asarray(seq_lens),
        jnp.asarray(cu_q_lens),
        jnp.asarray(distribution),
        spec=_RankMajorQueryRowSpec(
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        ),
    )
    slot_spec = _SlotMappingSpec(
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    actual = _compute_pcp_kv_writeback_mapping(
        query_rows,
        jnp.asarray(block_tables),
        spec=slot_spec,
    )
    jitted_actual = jax.jit(
        lambda rows, tables: _compute_pcp_kv_writeback_mapping(
            rows,
            tables,
            spec=slot_spec,
        )
    )(query_rows, jnp.asarray(block_tables))
    expected = _build_expected_pcp_writeback_mapping(
        q_lens,
        seq_lens,
        block_tables,
        page_size,
        pcp_size,
        interleave_size,
        local_padded_tokens * pcp_size,
        local_kv_cache_num_blocks,
    )

    for field in _PCPKVWritebackMapping._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, field)), getattr(expected, field)
        )
        np.testing.assert_array_equal(
            np.asarray(getattr(jitted_actual, field)), getattr(expected, field)
        )
    for cache_rank in range(pcp_size):
        compact_slots = _compact_writeback_slot_ids_for_cache_rank(actual, cache_rank)
        np.testing.assert_array_equal(
            np.asarray(compact_slots), expected.compact_slot_ids[cache_rank]
        )


def test_one_token_writeback_reaches_cache_owner_without_local_q():
    pcp_size = 8
    page_size = interleave_size = 256
    local_padded_tokens = 4
    local_kv_cache_num_blocks = 32
    block_tables = jnp.asarray([[7]], dtype=jnp.int32)
    query_rows = _compute_pcp_rank_major_query_row_mapping_from_metadata(
        kv_lens=jnp.asarray([301], dtype=jnp.int32),
        cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        spec=_RankMajorQueryRowSpec(
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        ),
    )
    mapping = _compute_pcp_kv_writeback_mapping(
        query_rows,
        block_tables,
        spec=_SlotMappingSpec(
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        ),
    )

    valid_row = int(np.flatnonzero(np.asarray(query_rows.is_valid))[0])
    assert int(query_rows.source_ranks[valid_row]) == 0
    assert int(mapping.cache_owner_ranks[valid_row]) == 1
    assert int(mapping.capture_offsets[valid_row]) == 0
    np.testing.assert_array_equal(
        np.asarray(mapping.writeback_counts),
        np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    )
    rank_1_slots = _compact_writeback_slot_ids_for_cache_rank(mapping, 1)
    assert int(rank_1_slots[0]) == 7 * page_size + 44
    assert np.all(np.asarray(rank_1_slots[1:]) == -1)


def test_batch_flat_metadata_32x256_balances_rows_with_cache_owner_skew():
    pcp_size = 8
    interleave_size = page_size = 256
    num_reqs = 32
    q_len = 256
    local_padded_tokens = 1280
    local_kv_cache_num_blocks = 32
    q_lens = jnp.full((num_reqs,), q_len, dtype=jnp.int32)
    token_owner_starts = jnp.arange(num_reqs, dtype=jnp.int32) * q_len
    request_absolute_starts = jnp.zeros((num_reqs,), dtype=jnp.int32)
    page_numbers = np.asarray(
        [
            19,
            3,
            27,
            11,
            0,
            24,
            8,
            30,
            14,
            6,
            22,
            1,
            17,
            29,
            10,
            25,
            5,
            21,
            13,
            31,
            2,
            18,
            28,
            9,
            23,
            7,
            16,
            4,
            26,
            12,
            20,
            15,
        ],
        dtype=np.int32,
    )
    block_tables = jnp.asarray(page_numbers[:, None])
    cu_q_lens = jnp.concatenate(
        (jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_lens, dtype=jnp.int32))
    )
    distribution = jnp.asarray([0, 0, num_reqs], dtype=jnp.int32)

    mappings = [
        _compute_pcp_local_query_row_mapping(
            q_lens,
            token_owner_starts,
            request_absolute_starts,
            spec=_LocalQueryRowSpec(
                local_padded_tokens=local_padded_tokens,
                pcp_size=pcp_size,
                pcp_rank=pcp_rank,
                interleave_size=interleave_size,
            ),
        )
        for pcp_rank in range(pcp_size)
    ]
    metadata_mappings = [
        _compute_pcp_local_query_row_mapping_from_metadata(
            q_lens,
            cu_q_lens,
            distribution,
            spec=_LocalQueryRowSpec(
                local_padded_tokens=local_padded_tokens,
                pcp_size=pcp_size,
                pcp_rank=pcp_rank,
                interleave_size=interleave_size,
            ),
        )
        for pcp_rank in range(pcp_size)
    ]
    for expected, actual in zip(mappings, metadata_mappings):
        for field in _PCPQueryRowMapping._fields:
            np.testing.assert_array_equal(
                np.asarray(getattr(actual, field)), np.asarray(getattr(expected, field))
            )
    rank_major_query_rows = _PCPQueryRowMapping(
        *(
            jnp.concatenate([getattr(mapping, field) for mapping in mappings])
            for field in _PCPQueryRowMapping._fields
        )
    )
    writeback_mapping = _compute_pcp_kv_writeback_mapping(
        rank_major_query_rows,
        block_tables,
        spec=_SlotMappingSpec(
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        ),
    )

    for source_rank, mapping in enumerate(mappings):
        assert int(np.asarray(mapping.is_valid).sum()) == 4 * q_len
        np.testing.assert_array_equal(
            np.asarray(mapping.source_ranks),
            np.full(local_padded_tokens, source_rank, dtype=np.int32),
        )
        assert np.all(np.asarray(mapping.request_indices)[4 * q_len :] == -1)

    cache_owner_ranks = np.asarray(writeback_mapping.cache_owner_ranks)
    cache_local_slot_ids = np.asarray(writeback_mapping.cache_local_slot_ids)
    cache_rank_0_slots = np.where(cache_owner_ranks == 0, cache_local_slot_ids, -1)
    cache_rank_1_slots = np.where(cache_owner_ranks == 1, cache_local_slot_ids, -1)
    assert np.count_nonzero(cache_rank_0_slots >= 0) == num_reqs * q_len
    assert np.all(cache_rank_1_slots < 0)
    np.testing.assert_array_equal(
        np.asarray(writeback_mapping.writeback_counts),
        np.array([num_reqs * q_len, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    )
    capture_offsets = np.asarray(writeback_mapping.capture_offsets)
    assert np.all(capture_offsets[~np.asarray(rank_major_query_rows.is_valid)] == -1)
    np.testing.assert_array_equal(
        np.sort(capture_offsets[capture_offsets >= 0]),
        np.arange(num_reqs * q_len, dtype=np.int32),
    )
    cache_rank_0_compact_slots = _compact_writeback_slot_ids_for_cache_rank(
        writeback_mapping, 0
    )
    np.testing.assert_array_equal(
        np.asarray(cache_rank_0_compact_slots)[: num_reqs * q_len].reshape(
            num_reqs, q_len
        )[:, 0],
        page_numbers * page_size,
    )
    valid = np.asarray(rank_major_query_rows.is_valid)
    request_indices = np.asarray(rank_major_query_rows.request_indices)[valid]
    absolute_positions = np.asarray(rank_major_query_rows.absolute_positions)[valid]
    np.testing.assert_array_equal(
        cache_rank_0_slots[valid],
        page_numbers[request_indices] * page_size + absolute_positions,
    )


def _make_single_rank_shard_map(
    pcp_rank: int,
    local_padded_tokens: int,
    local_kv_cache_num_blocks: int,
) -> typing.Callable[..., typing.Callable[..., tuple[jax.Array, jax.Array]]]:
    """Build a fake shard_map that slices inputs for one simulated PCP rank."""

    def fake_shard_map(
        fn: typing.Callable[..., tuple[jax.Array, jax.Array]],
        **_kwargs: object,
    ) -> typing.Callable[..., tuple[jax.Array, jax.Array]]:
        """Slice sharded Q/K/V/cache inputs for a single simulated PCP rank."""

        def wrapped(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Invoke fn on the simulated local PCP shard."""
            local_args = list(args)
            token_start = pcp_rank * local_padded_tokens
            token_end = token_start + local_padded_tokens
            block_start = pcp_rank * local_kv_cache_num_blocks
            block_end = block_start + local_kv_cache_num_blocks
            local_args[0] = local_args[0][token_start:token_end]
            local_args[1] = local_args[1][token_start:token_end]
            local_args[2] = local_args[2][token_start:token_end]
            local_args[3] = local_args[3][block_start:block_end]
            return fn(*local_args)

        return wrapped

    return fake_shard_map


def test_sharded_wrapper_generates_slot_ids_when_metadata_omits_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sharded wrapper builds stage-plan writeback descriptors from metadata."""
    pcp_size, pcp_rank, interleave_size, page_size = 4, 2, 4, 4
    local_padded_tokens, target_num_reqs, local_kv_cache_num_blocks = 16, 4, 8
    q_lens = np.array([5, 17, 6], dtype=np.int32)
    q_starts = np.array([3, 9, 0], dtype=np.int32)
    seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
    seq_lens[: q_lens.size] = q_starts + q_lens
    cu_q_lens = np.zeros(target_num_reqs + 1, dtype=np.int32)
    cu_q_lens[1 : q_lens.size + 1] = np.cumsum(q_lens)
    cu_q_lens[q_lens.size + 1 :] = cu_q_lens[q_lens.size]
    distribution = np.array([0, 0, q_lens.size], dtype=np.int32)
    block_tables = (
        np.arange(target_num_reqs * 8, dtype=np.int32).reshape(target_num_reqs, 8) + 11
    )
    expected_writeback = _build_expected_pcp_writeback_mapping(
        q_lens,
        q_starts + q_lens,
        block_tables,
        page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=local_padded_tokens * pcp_size,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
    )
    expected = expected_writeback.compact_slot_ids[pcp_rank]

    captured = {}

    def fake_streaming(
        q_streaming: jax.Array,
        _current_kv: jax.Array,
        kv_cache: jax.Array,
        _kv_lens: jax.Array,
        _cu_q_lens: jax.Array,
        _distribution: jax.Array,
        segment_descriptors: jax.Array,
        num_segments: jax.Array,
        *,
        stage_plan: pcp_stage_plan.PCPStagePlan,
        **_kwargs: object,
    ) -> tuple[jax.Array, jax.Array]:
        """Capture stage_plan descriptors and validity mask."""
        captured["segment_descriptors"] = segment_descriptors
        captured["num_segments"] = num_segments
        captured["is_valid"] = stage_plan.query_row_valid
        return jnp.full_like(q_streaming, 5), kv_cache

    def fake_prepare_current(
        k_local: jax.Array,
        _v_local: jax.Array,
        cache: jax.Array,
        **_kwargs: object,
    ) -> jax.Array:
        """Return a zeroed packed current-KV tensor matching cache tail."""
        return jnp.zeros((k_local.shape[0], *cache.shape[2:]), dtype=cache.dtype)

    monkeypatch.setattr(
        pcp_wrapper, "_prepare_packed_kv_for_cache", fake_prepare_current
    )
    monkeypatch.setattr(
        pcp_wrapper,
        "pcp_streaming_attention_page_groups_packed_local_from_metadata",
        fake_streaming,
    )
    monkeypatch.setattr(
        pcp_stage_plan,
        "get_pcp_rank_without_axis_index",
        lambda _axis, _size: jnp.asarray(pcp_rank, dtype=jnp.int32),
    )
    monkeypatch.setattr(
        pcp_wrapper.jax,
        "shard_map",
        _make_single_rank_shard_map(
            pcp_rank, local_padded_tokens, local_kv_cache_num_blocks
        ),
    )

    mesh = _MeshStub(
        axis_names=(pcp_stage_plan.PCP_AXIS_NAME,),
        shape={pcp_stage_plan.PCP_AXIS_NAME: pcp_size},
    )
    global_padded_tokens = local_padded_tokens * pcp_size
    output, new_cache = pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=jnp.ones((global_padded_tokens, 2, 128), dtype=jnp.float32),
        k=jnp.ones((global_padded_tokens, 1, 128), dtype=jnp.float32),
        v=jnp.ones((global_padded_tokens, 1, 128), dtype=jnp.float32),
        kv_cache=jnp.zeros(
            (local_kv_cache_num_blocks * pcp_size, page_size, 2, 1, 128),
            dtype=jnp.float32,
        ),
        kv_lens=jnp.asarray(seq_lens),
        page_indices=jnp.asarray(block_tables.reshape(-1)),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution),
        attention_sink=None,
        sm_scale=0.25,
        cp_kv_cache_interleave_size=interleave_size,
    )

    expected_segment_cap = min(
        local_padded_tokens,
        (local_padded_tokens + interleave_size - 1) // interleave_size
        + 2 * target_num_reqs,
    )
    expected_descriptors, expected_num_segments = _build_writeback_segment_descriptors(
        jnp.asarray(expected),
        jnp.asarray(expected_writeback.writeback_counts[pcp_rank]),
        max_segments=expected_segment_cap,
    )
    np.testing.assert_array_equal(
        np.asarray(captured["segment_descriptors"]),
        np.asarray(expected_descriptors),
    )
    assert int(captured["num_segments"]) == int(expected_num_segments)
    assert np.any(expected < 0)
    expected_output = np.zeros((local_padded_tokens, 2, 128), dtype=np.float32)
    expected_output[np.asarray(captured["is_valid"])] = 5
    np.testing.assert_array_equal(np.asarray(output), expected_output)
    assert new_cache.shape == (local_kv_cache_num_blocks, page_size, 2, 1, 128)


def test_sharded_wrapper_masks_output_with_query_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify sharded wrapper zeroes padded query rows using query_row_valid."""
    query_rows = _PCPQueryRowMapping(
        absolute_positions=jnp.array([0, -1, 2, -1], dtype=jnp.int32),
        request_indices=jnp.array([0, -1, 0, -1], dtype=jnp.int32),
        source_ranks=jnp.zeros(4, dtype=jnp.int32),
        is_valid=jnp.array([True, False, True, False]),
    )
    captured = {}

    def fake_compute_query_rows(
        *_args: object, **_kwargs: object
    ) -> _PCPQueryRowMapping:
        """Return fixed query-row mapping for validity masking test."""
        return query_rows

    def fake_streaming(
        q_streaming: jax.Array,
        _current_kv: jax.Array,
        kv_cache: jax.Array,
        _kv_lens: jax.Array,
        _cu_q_lens: jax.Array,
        _distribution: jax.Array,
        _segment_descriptors: jax.Array,
        _num_segments: jax.Array,
        **_kwargs: object,
    ) -> tuple[jax.Array, jax.Array]:
        """Record metadata entry invocation and return constant 5."""
        captured["writeback_called"] = True
        return jnp.full_like(q_streaming, 5), kv_cache

    def fake_prepare_current(
        k_local: jax.Array,
        _v_local: jax.Array,
        cache: jax.Array,
        **_kwargs: object,
    ) -> jax.Array:
        """Return a zeroed packed current-KV tensor matching cache tail."""
        return jnp.zeros((k_local.shape[0], *cache.shape[2:]), dtype=cache.dtype)

    def fake_shard_map(
        fn: typing.Callable[..., tuple[jax.Array, jax.Array]],
        **_kwargs: object,
    ) -> typing.Callable[..., tuple[jax.Array, jax.Array]]:
        """Return fn unmodified for single-rank unit testing."""
        return fn

    monkeypatch.setattr(
        pcp_stage_plan,
        "get_pcp_rank_without_axis_index",
        lambda _axis, _size: jnp.asarray(0, dtype=jnp.int32),
    )
    monkeypatch.setattr(
        pcp_stage_plan,
        "_compute_pcp_local_query_row_mapping",
        fake_compute_query_rows,
    )
    monkeypatch.setattr(
        pcp_stage_plan,
        "_compute_pcp_local_query_row_mapping_from_metadata",
        fake_compute_query_rows,
    )
    monkeypatch.setattr(
        pcp_wrapper, "_prepare_packed_kv_for_cache", fake_prepare_current
    )
    monkeypatch.setattr(
        pcp_wrapper,
        "pcp_streaming_attention_page_groups_packed_local_from_metadata",
        fake_streaming,
    )
    monkeypatch.setattr(pcp_wrapper.jax, "shard_map", fake_shard_map)

    num_tokens = 4
    page_size = 4
    mesh = _MeshStub(
        axis_names=(pcp_stage_plan.PCP_AXIS_NAME,),
        shape={pcp_stage_plan.PCP_AXIS_NAME: 1},
    )
    q = jnp.ones((num_tokens, 2, 128), dtype=jnp.float32)
    k = jnp.ones((num_tokens, 1, 128), dtype=jnp.float32)
    v = jnp.ones((num_tokens, 1, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros((1, page_size, 2, 1, 128), dtype=jnp.float32)

    output, _ = pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray([num_tokens]),
        page_indices=jnp.asarray([0]),
        cu_q_lens=jnp.asarray([0, num_tokens]),
        distribution=jnp.asarray([0, 0, 1]),
        attention_sink=None,
        sm_scale=0.25,
        cp_kv_cache_interleave_size=page_size,
    )

    assert captured["writeback_called"]
    expected_output = np.zeros((num_tokens, 2, 128), dtype=np.float32)
    expected_output[[0, 2]] = 5
    np.testing.assert_array_equal(np.asarray(output), expected_output)


def test_reshape_packed_kv_cache_for_attention_rejects_ambiguous_layout():
    kv_cache = jnp.arange(2 * 4 * 16 * 1 * 8, dtype=jnp.bfloat16).reshape(
        2, 4, 16, 1, 8
    )

    with pytest.raises(ValueError, match="packing does not match"):
        pcp_wrapper._reshape_packed_kv_cache_for_attention(kv_cache)


@pytest.mark.parametrize(
    ("q_per_kv", "expected_stacked"),
    [(2, 2), (6, 2), (8, 8), (16, 16)],
)
def test_metadata_entry_caps_stacked_q_heads_at_q_per_kv(
    monkeypatch: pytest.MonkeyPatch, q_per_kv: int, expected_stacked: int
) -> None:
    """Verify metadata entry caps num_stacked_q_heads at gcd(16, q_per_kv)."""
    captured = {"num_stacked_q_heads": 0}

    def fake_attend(
        _kv_cache_local: jax.Array,
        q_scaled_head_major: jax.Array,
        _stage_plan: object,
        *,
        spec: _SplashStreamSpec,
    ) -> jax.Array:
        """Capture SplashConfig.num_stacked_q_heads and return zeros."""
        assert spec.splash_config is not None
        captured["num_stacked_q_heads"] = int(spec.splash_config.num_stacked_q_heads)
        return jnp.zeros_like(q_scaled_head_major, dtype=jnp.float32)

    monkeypatch.setattr(
        pcp_kernel,
        "_route_current_kv_to_captured_kv",
        lambda kv_in, *_args, **_kwargs: kv_in,
    )
    monkeypatch.setattr(
        pcp_kernel,
        "write_captured_kv_to_local_cache",
        lambda cache, *_args, **_kwargs: cache,
    )
    monkeypatch.setattr(pcp_kernel, "_attend_over_vpage_groups", fake_attend)

    q_local = jnp.zeros((128, 2, q_per_kv, 128), dtype=jnp.bfloat16)
    current_kv_local = jnp.zeros((128, 2, 2, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((4, 128, 2, 2, 128), dtype=jnp.bfloat16)
    stage_plan = _PCPStagePlanStub()

    pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
        q_local,
        current_kv_local,
        kv_cache_local,
        jnp.asarray([128], dtype=jnp.int32),
        jnp.asarray([0, 128], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        jnp.zeros((3, 8), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        sm_scale=1.0,
        stage_plan=stage_plan,
        vpage_group_size=128,
        pcp_size=8,
        interleave_size=128,
    )

    assert captured["num_stacked_q_heads"] == expected_stacked


@pytest.mark.parametrize(
    ("q_lens", "hist_lens"),
    [
        ([192], [301]),
        ([95, 95], [0, 0]),
        ([96, 96], [0, 0]),
        ([97, 97], [0, 0]),
        ([127, 127], [0, 0]),
    ],
)
def test_capture_source_rows_follow_token_cache_owner(
    q_lens: list[int], hist_lens: list[int]
) -> None:
    """Verify capture source rows partition active tokens by request-absolute owner."""
    pcp_size, interleave_size, local_tokens, max_num_seqs = 2, 32, 128, 2
    pad = max_num_seqs - len(q_lens)
    cu_q_lens = jnp.asarray(
        np.concatenate([[0], np.cumsum(q_lens + [0] * pad)]), dtype=jnp.int32
    )
    kv_lens = jnp.asarray(
        [h + q for h, q in zip(hist_lens, q_lens, strict=True)] + [0] * pad,
        dtype=jnp.int32,
    )
    distribution = jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32)
    rank_major_spec = _RankMajorQueryRowSpec(
        local_padded_tokens=local_tokens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    rows = _compute_pcp_rank_major_query_row_mapping_from_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        spec=rank_major_spec,
    )
    positions = np.asarray(rows.absolute_positions)
    requests = np.asarray(rows.request_indices)
    metadata = pcp_stage_plan.PCPStageMetadata(
        kv_lens=kv_lens,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
    )
    seen: list[int] = []
    for rank in range(pcp_size):
        src, valid = _compute_pcp_capture_source_rows_for_cache_rank(
            metadata,
            rank,
            spec=rank_major_spec,
        )
        src_arr = np.asarray(src)[np.asarray(valid)]
        expected = [
            (r, p)
            for r, (h, q) in enumerate(zip(hist_lens, q_lens, strict=True))
            for p in range(h, h + q)
            if (p // interleave_size) % pcp_size == rank
        ]
        actual = [
            (int(r), int(p))
            for r, p in zip(requests[src_arr], positions[src_arr], strict=True)
        ]
        assert actual == expected
        seen.extend(src_arr.tolist())
    assert len(seen) == len(set(seen)) == sum(q_lens)


def test_gather_vpage_group_zeroes_unwritten_fp8_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unwritten FP8 slots holding NaN bit patterns are zeroed on gather."""
    valid_tokens = 5
    fp8_value = 1.5
    nan = jnp.asarray(0x7F, dtype=jnp.uint8).view(jnp.float8_e4m3fn)
    tokens = jnp.arange(16)[None, :, None]
    raw = jnp.broadcast_to(
        jnp.where(
            tokens < valid_tokens,
            jnp.asarray(fp8_value, dtype=jnp.float8_e4m3fn),
            nan,
        ),
        (1, 16, 128),
    )
    monkeypatch.setattr(
        pcp_kernel, "_gather_vpages_to_head_major", lambda *_a, **_k: (raw, raw)
    )
    plan = _PCPStagePlanStub(
        sorted_block_ids=jnp.asarray([0, 1], dtype=jnp.int32),
        sorted_seg_ids=jnp.asarray([0, -1], dtype=jnp.int32),
        sorted_vpage_valid_tokens=jnp.asarray([valid_tokens, 0], dtype=jnp.int32),
    )
    k_head_major, v_head_major, kv_seg_ids = _gather_vpage_group_kv_and_seg(
        jnp.zeros((2, 4, 1, 4, 128), dtype=jnp.float8_e4m3fn),
        plan,
        0,
        2,
        spec=_SplashStreamSpec(
            kv_heads=1,
            head_dim=128,
            page_size=4,
            pcp_size=2,
            interleave_size=2,
            kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
            pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
        ),
    )
    assert not bool(jnp.isnan(k_head_major.astype(jnp.float32)).any())
    assert not bool(jnp.isnan(v_head_major.astype(jnp.float32)).any())
    assert bool((k_head_major[:, :valid_tokens].astype(jnp.float32) == fp8_value).all())
    assert int((kv_seg_ids == 0).sum()) == valid_tokens


def test_merge_attention_partials_matches_full_softmax() -> None:
    """Verify online LSE partial merging matches full FP64 softmax."""
    rng = np.random.default_rng(2026)
    heads, q_tokens, s1, s2, head_dim = 4, 32, 96, 64, 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = rng.normal(size=(heads, q_tokens, head_dim)).astype(np.float64)
    k = rng.normal(size=(heads, s1 + s2, head_dim)).astype(np.float64)
    v = rng.normal(size=(heads, s1 + s2, head_dim)).astype(np.float64)

    def exact_part(
        k_slice: np.ndarray, v_slice: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact FP64 attention output and natural-log LSE for one slice."""
        scores = np.einsum("htd,hsd->hts", q, k_slice) * sm_scale
        m = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - m)
        denom = exp_s.sum(axis=-1, keepdims=True)
        out = np.einsum("hts,hsd->htd", exp_s / denom, v_slice)
        lse = m[..., 0] + np.log(denom[..., 0])
        return out.astype(np.float32), lse.astype(np.float32)

    o1, lse1 = exact_part(k[:, :s1], v[:, :s1])
    o2, lse2 = exact_part(k[:, s1:], v[:, s1:])
    o_full, lse_full = exact_part(k, v)

    o_merged, lse_merged = _merge_attention_partials(
        jnp.asarray(o1),
        jnp.asarray(lse1),
        jnp.asarray(o2),
        jnp.asarray(lse2),
    )
    o_merged_np = np.asarray(o_merged, dtype=np.float64)
    lse_merged_np = np.asarray(lse_merged, dtype=np.float64)

    u_f32 = 2.0**-24
    max_abs_v = float(np.max(np.abs(v)))
    max_abs_lse = float(np.max(np.abs(lse_full)))
    # Error budget: 8*u_f32 covers float32 exp/log, normalization division,
    # and weighted accumulation across the two partial outputs and LSEs.
    tol_o = 8.0 * u_f32 * max_abs_v
    tol_lse = 8.0 * u_f32 * max(1.0, max_abs_lse)

    assert float(np.max(np.abs(o_merged_np - o_full))) <= tol_o
    assert float(np.max(np.abs(lse_merged_np - lse_full))) <= tol_lse
    assert float(np.max(np.abs(o1.astype(np.float64) - o_full))) > 100.0 * tol_o


def test_merge_attention_partials_ignores_empty_splash_group() -> None:
    """Verify empty Splash group sentinels and -inf rows preserve finite outputs."""
    rng = np.random.default_rng(7)
    o_valid = jnp.asarray(rng.normal(size=(2, 16, 64)).astype(np.float32))
    lse_valid = jnp.asarray(rng.normal(size=(2, 16)).astype(np.float32))
    o_zero = jnp.zeros_like(o_valid)
    sentinel = float(splash_kernel_mod.DEFAULT_MASK_VALUE / splash_kernel_mod.LOG2E)
    lse_sentinel = jnp.full_like(lse_valid, sentinel)

    o_right, lse_right = _merge_attention_partials(
        o_valid, lse_valid, o_zero, lse_sentinel
    )
    np.testing.assert_array_equal(np.asarray(o_right), np.asarray(o_valid))
    np.testing.assert_array_equal(np.asarray(lse_right), np.asarray(lse_valid))

    o_left, lse_left = _merge_attention_partials(
        o_zero, lse_sentinel, o_valid, lse_valid
    )
    np.testing.assert_array_equal(np.asarray(o_left), np.asarray(o_valid))
    np.testing.assert_array_equal(np.asarray(lse_left), np.asarray(lse_valid))

    lse_neginf = jnp.full_like(lse_valid, -jnp.inf)
    o_empty, lse_empty = _merge_attention_partials(
        o_zero, lse_neginf, o_zero, lse_neginf
    )
    assert not bool(jnp.isnan(o_empty).any())
    assert not bool(jnp.isnan(lse_empty).any())
    np.testing.assert_array_equal(
        np.asarray(o_empty), np.zeros_like(np.asarray(o_valid))
    )
    assert bool(jnp.isneginf(lse_empty).all())


def _make_test_splash_cfg(
    block_kv: int = pcp_kernel.SPLASH_BLOCK_KV_MIN,
    num_stacked_q_heads: int = 2,
) -> splash_kernel_mod.SplashConfig:
    """Create a test SplashConfig with base-2 prescaled queries."""
    return splash_kernel_mod.SplashConfig(
        block_q=pcp_kernel.SPLASH_BLOCK_Q,
        block_kv=block_kv,
        block_kv_compute=pcp_kernel.SPLASH_BLOCK_KV_COMPUTE,
        num_stacked_q_heads=num_stacked_q_heads,
        use_base2_exp=True,
        q_prescaled_log2e=True,
    )


def test_attend_over_vpage_groups_routes_rungs_and_multi_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify single-group rung ladder selection and multi-group fallback."""
    calls: list[tuple[str, int]] = []

    def fake_single(
        _kv: jax.Array,
        q_scaled_head_major: jax.Array,
        _plan: object,
        group_size: int,
        **_kwargs: object,
    ) -> jax.Array:
        """Record single-group call and return positive rung sentinel."""
        calls.append(("single", int(group_size)))
        return jnp.full_like(q_scaled_head_major, float(group_size), dtype=jnp.float32)

    def fake_multi(
        _kv: jax.Array,
        q_scaled_head_major: jax.Array,
        _plan: object,
        group_size: int,
        **_kwargs: object,
    ) -> jax.Array:
        """Record multi-group call and return negative group_size sentinel."""
        calls.append(("multi", int(group_size)))
        return jnp.full_like(q_scaled_head_major, -float(group_size), dtype=jnp.float32)

    monkeypatch.setattr(pcp_kernel, "_attend_single_vpage_group", fake_single)
    monkeypatch.setattr(pcp_kernel, "_attend_multi_vpage_groups", fake_multi)

    q_scaled_head_major = jnp.zeros((2, 128, 128), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((8, 128, 1, 2, 128), dtype=jnp.bfloat16)

    def make_stream_spec(vpage_group_size: int) -> _SplashStreamSpec:
        """Build a _SplashStreamSpec for rung-routing tests."""
        return _SplashStreamSpec(
            kv_heads=1,
            head_dim=128,
            page_size=128,
            pcp_size=2,
            interleave_size=32,
            kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
            pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
            vpage_group_size=vpage_group_size,
            splash_config=_make_test_splash_cfg(),
        )

    def make_plan(total_vpages: int, num_active_vpages: int) -> _PCPStagePlanStub:
        """Build stub stage plan with given total and active vpages."""
        return _PCPStagePlanStub(
            sorted_block_ids=jnp.zeros((total_vpages,), dtype=jnp.int32),
            num_active_vpages=jnp.asarray(num_active_vpages, dtype=jnp.int32),
        )

    for active_vpages, expected_rung in [
        (1, 32.0),
        (32, 32.0),
        (33, 64.0),
        (80, 96.0),
        (128, 128.0),
        (257, 512.0),
        (512, 512.0),
    ]:
        out = _attend_over_vpage_groups(
            kv_cache_local,
            q_scaled_head_major,
            make_plan(512, active_vpages),
            spec=make_stream_spec(128),
        )
        assert float(out[0, 0, 0]) == expected_rung

    multi_group_size = 128
    out_513 = _attend_over_vpage_groups(
        kv_cache_local,
        q_scaled_head_major,
        make_plan(640, 513),
        spec=make_stream_spec(multi_group_size),
    )
    assert float(out_513[0, 0, 0]) == -float(multi_group_size)

    calls.clear()
    small_group_size = 2
    out_g2 = _attend_over_vpage_groups(
        kv_cache_local,
        q_scaled_head_major,
        make_plan(6, 6),
        spec=make_stream_spec(small_group_size),
    )
    assert float(out_g2[0, 0, 0]) == -float(small_group_size)
    assert calls == [("multi", small_group_size)]


def _build_multi_group_vpages(
    inputs: _MultiGroupVpageInputs,
) -> tuple[jax.Array, jax.Array]:
    """Construct mock KV vpages with 1e6 poison values in unwritten tails."""
    shape = (
        inputs.kv_heads,
        inputs.total_vpages,
        inputs.virtual_page_size,
        inputs.head_dim,
    )
    k_vpages = np.full(shape, 1e6, dtype=np.float32)
    v_vpages = np.full(shape, 1e6, dtype=np.float32)
    for vp in range(3):
        s = vp * inputs.virtual_page_size
        e = min((vp + 1) * inputs.virtual_page_size, inputs.req0_kv_len)
        k_vpages[:, vp, : e - s, :] = inputs.k_req0[:, s:e, :]
        v_vpages[:, vp, : e - s, :] = inputs.v_req0[:, s:e, :]
    for vp in range(3):
        s = vp * inputs.virtual_page_size
        e = min((vp + 1) * inputs.virtual_page_size, inputs.req1_kv_len)
        k_vpages[:, 3 + vp, : e - s, :] = inputs.k_req1[:, s:e, :]
        v_vpages[:, 3 + vp, : e - s, :] = inputs.v_req1[:, s:e, :]
    return jnp.asarray(k_vpages), jnp.asarray(v_vpages)


class _FakeSplashKernel:
    """CPU reference double matching SplashAttentionKernel base-2 LSE semantics."""

    def __init__(
        self,
        fwd_mask_info: splash_mask_info_mod.MaskInfo,
        **kwargs: object,
    ) -> None:
        """Store forward mask info and causal mask callback."""
        self.fwd_mask_info = fwd_mask_info
        self.kwargs = kwargs
        self.mask_function = kwargs.get("mask_function")

    def __call__(
        self,
        q_scaled: jax.Array,
        k_hm: jax.Array,
        v_hm: jax.Array,
        segment_ids: splash_kernel_mod.SegmentIds,
        *,
        save_residuals: bool = False,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute segmented causal attention and natural-log LSE."""
        assert save_residuals
        assert callable(self.mask_function)
        q_rep_factor = q_scaled.shape[0] // k_hm.shape[0]
        k_rep = jnp.repeat(k_hm.astype(jnp.float32), q_rep_factor, axis=0)
        v_rep = jnp.repeat(v_hm.astype(jnp.float32), q_rep_factor, axis=0)
        scores_b2 = jnp.einsum(
            "hqd,hkd->hqk",
            q_scaled.astype(jnp.float32),
            k_rep,
            precision="highest",
        )
        q_seq = jnp.asarray(self.fwd_mask_info.q_sequence, dtype=jnp.int32)[:, None]
        kv_idx = jnp.arange(k_hm.shape[1], dtype=jnp.int32)[None, :]
        causal_ok = self.mask_function(q_seq, kv_idx)
        seg_ok = (segment_ids.q[:, None] == segment_ids.kv[None, :]) & (
            segment_ids.q[:, None] >= 0
        )
        valid = (causal_ok & seg_ok)[None, :, :]
        any_block_valid = jnp.any(valid)
        sentinel = jnp.asarray(
            splash_kernel_mod.DEFAULT_MASK_VALUE / splash_kernel_mod.LOG2E,
            dtype=jnp.float32,
        )
        masked_b2 = jnp.where(valid, scores_b2, -jnp.inf)
        m_b2 = jnp.max(masked_b2, axis=-1)
        m_safe = jnp.where(jnp.isneginf(m_b2), 0.0, m_b2)
        weights = jnp.where(valid, jnp.exp2(masked_b2 - m_safe[..., None]), 0.0)
        l_sum = jnp.sum(weights, axis=-1)
        inv_l = 1.0 / jnp.where(l_sum > 0.0, l_sum, 1.0)
        out_nonempty = jnp.einsum(
            "hqk,hkd->hqd",
            weights * inv_l[..., None],
            v_rep,
            precision="highest",
        )
        lse_nonempty = (m_b2 + jnp.log2(l_sum)) / jnp.asarray(
            splash_kernel_mod.LOG2E, dtype=jnp.float32
        )
        out = jnp.where(any_block_valid, out_nonempty, jnp.zeros_like(out_nonempty))
        lse = jnp.where(
            any_block_valid,
            lse_nonempty,
            jnp.full_like(lse_nonempty, sentinel),
        )
        return out, {"logsumexp": lse}


def test_attend_multi_vpage_groups_streams_causal_groups_and_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-group Splash streaming across causal groups and empty sentinels."""
    rng = np.random.default_rng(2027)
    kv_heads, q_per_kv, head_dim = 1, 2, 128
    page_size, pcp_size, total_vpages = 128, 2, 8
    virtual_page_size = page_size * pcp_size
    sm_scale = 1.0 / math.sqrt(head_dim)
    q_raw = rng.normal(size=(kv_heads * q_per_kv, 128, head_dim)).astype(np.float32)
    q_scaled_head_major = jnp.asarray(
        q_raw * (sm_scale * splash_kernel_mod.LOG2E), dtype=jnp.float32
    )

    k_req0 = rng.normal(size=(kv_heads, 652, head_dim)).astype(np.float32)
    v_req0 = rng.normal(size=(kv_heads, 652, head_dim)).astype(np.float32)
    k_req1 = rng.normal(size=(kv_heads, 692, head_dim)).astype(np.float32)
    v_req1 = rng.normal(size=(kv_heads, 692, head_dim)).astype(np.float32)
    k_vpages_jnp, v_vpages_jnp = _build_multi_group_vpages(
        _MultiGroupVpageInputs(
            kv_heads=kv_heads,
            total_vpages=total_vpages,
            virtual_page_size=virtual_page_size,
            head_dim=head_dim,
            k_req0=k_req0,
            v_req0=v_req0,
            k_req1=k_req1,
            v_req1=v_req1,
            req0_kv_len=652,
            req1_kv_len=692,
        )
    )

    req0_abs = np.arange(592, 652, dtype=np.int32)
    req1_abs = np.arange(632, 692, dtype=np.int32)
    stage_plan = _PCPStagePlanStub(
        sorted_block_ids=jnp.arange(total_vpages, dtype=jnp.int32),
        sorted_seg_ids=jnp.asarray([0, 0, 0, 1, 1, 1, 2, -2], dtype=jnp.int32),
        sorted_vpage_valid_tokens=jnp.asarray(
            [256, 256, 140, 256, 256, 180, 64, 0], dtype=jnp.int32
        ),
        num_active_vpages=jnp.asarray(7, dtype=jnp.int32),
        q_seg_ids=jnp.asarray(
            np.concatenate(
                [
                    np.zeros(60, dtype=np.int32),
                    np.ones(60, dtype=np.int32),
                    np.full(8, -1, dtype=np.int32),
                ]
            )
        ),
        q_sequence=jnp.asarray(
            np.concatenate(
                [
                    req0_abs,
                    3 * virtual_page_size + req1_abs,
                    np.full(8, -1, dtype=np.int32),
                ]
            )
        ),
    )

    def fake_gather(
        _kv_cache_local: jax.Array, block_ids: jax.Array, **_kwargs: object
    ) -> tuple[jax.Array, jax.Array]:
        """Slice prebuilt per-vpage K and V tensors."""
        return (
            k_vpages_jnp[:, block_ids, :, :].reshape(kv_heads, -1, head_dim),
            v_vpages_jnp[:, block_ids, :, :].reshape(kv_heads, -1, head_dim),
        )

    monkeypatch.setattr(pcp_kernel, "_gather_vpages_to_head_major", fake_gather)
    monkeypatch.setattr(splash_kernel_mod, "SplashAttentionKernel", _FakeSplashKernel)

    actual = np.asarray(
        _attend_multi_vpage_groups(
            jnp.zeros(
                (total_vpages, page_size, kv_heads, 2, head_dim), dtype=jnp.float32
            ),
            q_scaled_head_major,
            stage_plan,
            2,
            spec=_SplashStreamSpec(
                kv_heads=kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=32,
                kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
                vpage_group_size=2,
                splash_config=_make_test_splash_cfg(),
            ),
        ),
        dtype=np.float64,
    )
    assert not np.isnan(actual).any()
    np.testing.assert_array_equal(actual[:, 120:, :], 0.0)

    req0_causal_mask = req0_abs[:, None] >= np.arange(k_req0.shape[1])[None, :]
    req1_causal_mask = req1_abs[:, None] >= np.arange(k_req1.shape[1])[None, :]
    req0_g0_causal_mask = req0_abs[:, None] >= np.arange(512)[None, :]
    ref_req0 = _compute_fp64_attention(
        q_raw[:, :60, :] * sm_scale, k_req0, v_req0, req0_causal_mask
    )[0]
    ref_req1 = _compute_fp64_attention(
        q_raw[:, 60:120, :] * sm_scale, k_req1, v_req1, req1_causal_mask
    )[0]
    ref_req0_g0_only = _compute_fp64_attention(
        q_raw[:, :60, :] * sm_scale,
        k_req0[:, :512, :],
        v_req0[:, :512, :],
        req0_g0_causal_mask,
    )[0]
    # Error budget: 56*u_f32 covers float32 base-2 Q/K dot products, exp2/log2
    # softmax scaling, and 3 online LSE merges across 4 streamed vpage groups.
    tol_stream = (
        56.0
        * (2.0**-24)
        * max(float(np.max(np.abs(v_req0))), float(np.max(np.abs(v_req1))))
    )
    assert float(np.max(np.abs(actual[:, :60, :] - ref_req0))) <= tol_stream
    assert float(np.max(np.abs(actual[:, 60:120, :] - ref_req1))) <= tol_stream
    assert float(np.max(np.abs(ref_req0_g0_only - ref_req0))) > 100.0 * tol_stream


def test_gather_vpages_to_head_major_with_odd_fp8_kv_heads() -> None:
    """Verify FP8 gather slices padded sublane planes after all_gather."""
    pcp_size, page_size, interleave_size = 8, 16, 16
    num_vpages, num_blocks, head_dim, kv_packing = 4, 6, 128, 4
    block_ids = jnp.asarray([1, 3, 0, 5], dtype=jnp.int32)
    virtual_page_size = page_size * pcp_size
    num_tokens = num_vpages * virtual_page_size

    def _run_gather(
        local_caches: jax.Array, num_kvh: int
    ) -> tuple[jax.Array, jax.Array]:
        """Run _gather_vpages_to_head_major across local PCP cache shards."""
        gather_spec = _SplashStreamSpec(
            kv_heads=num_kvh,
            head_dim=head_dim,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
            pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
        )
        return jax.vmap(
            lambda cache_shard: _gather_vpages_to_head_major(
                cache_shard,
                block_ids,
                spec=gather_spec,
            ),
            in_axes=0,
            out_axes=0,
            axis_name=pcp_stage_plan.PCP_AXIS_NAME,
        )(local_caches)

    for kv_heads in (1, 3):
        packed_groups = math.ceil(2 * kv_heads / kv_packing)
        rng = np.random.default_rng(100 + kv_heads)
        k_true = rng.choice(
            np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float32),
            size=(kv_heads, num_blocks * virtual_page_size, head_dim),
        ).astype(jnp.float8_e4m3fn)
        v_true = rng.choice(
            np.array([-1.5, -0.5, 0.25, 0.75, 1.5, 3.0], dtype=np.float32),
            size=(kv_heads, num_blocks * virtual_page_size, head_dim),
        ).astype(jnp.float8_e4m3fn)

        kv_interleaved = np.full(
            (num_blocks, pcp_size, page_size, packed_groups * kv_packing, head_dim),
            -7.0,
            dtype=np.float32,
        )
        k_np = np.asarray(k_true, dtype=np.float32).reshape(
            kv_heads, num_blocks, pcp_size, page_size, head_dim
        )
        v_np = np.asarray(v_true, dtype=np.float32).reshape(
            kv_heads, num_blocks, pcp_size, page_size, head_dim
        )
        for h in range(kv_heads):
            kv_interleaved[:, :, :, 2 * h, :] = k_np[h]
            kv_interleaved[:, :, :, 2 * h + 1, :] = v_np[h]

        local_caches = jnp.asarray(
            np.transpose(kv_interleaved, (1, 0, 2, 3, 4)).reshape(
                pcp_size, num_blocks, page_size, packed_groups, kv_packing, head_dim
            ),
            dtype=jnp.float8_e4m3fn,
        )
        gathered_k, gathered_v = _run_gather(local_caches, kv_heads)
        expected_k = (
            np.asarray(k_true, dtype=np.float32)
            .reshape(kv_heads, num_blocks, virtual_page_size, head_dim)[
                :, np.asarray(block_ids), :, :
            ]
            .reshape(kv_heads, num_tokens, head_dim)
        )
        expected_v = (
            np.asarray(v_true, dtype=np.float32)
            .reshape(kv_heads, num_blocks, virtual_page_size, head_dim)[
                :, np.asarray(block_ids), :, :
            ]
            .reshape(kv_heads, num_tokens, head_dim)
        )
        for rank in range(pcp_size):
            np.testing.assert_allclose(
                np.asarray(gathered_k[rank], dtype=np.float32),
                expected_k,
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                np.asarray(gathered_v[rank], dtype=np.float32),
                expected_v,
                rtol=0.0,
                atol=0.0,
            )


def _run_pure_write_slots(
    kv_cache_local: jax.Array,
    captured_kv_local: jax.Array,
    segment_descriptors: jax.Array,
    num_segments: jax.Array,
    *,
    kv_layout: pcp_kernel.KVLayout = pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    """Execute pure JAX slot writeback matching write_captured_kv_to_local_cache."""
    max_segments = int(segment_descriptors.shape[1])
    capacity = int(
        captured_kv_local.shape[-1]
        if kv_layout == pcp_kernel.KVLayout.SEQ_ALONG_LANE
        else captured_kv_local.shape[0]
    )
    seg_ids = jnp.arange(max_segments, dtype=jnp.int32)[:, None]
    offsets = jnp.arange(capacity, dtype=jnp.int32)[None, :]
    src = segment_descriptors[0, :][:, None] + offsets
    valid = (seg_ids < jnp.reshape(num_segments, ())) & (
        offsets < segment_descriptors[2, :][:, None]
    )
    src_safe = jnp.clip(src, 0, capacity - 1)
    flat_slots = segment_descriptors[1, :][:, None] + offsets
    if kv_layout == pcp_kernel.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache_local.shape[-1]
        total_slots = kv_cache_local.shape[0] * page_size
        flat_cache = kv_cache_local.transpose(0, 4, 1, 2, 3).reshape(
            total_slots, *kv_cache_local.shape[1:4]
        )
        vals = captured_kv_local.transpose(3, 0, 1, 2)[src_safe.reshape(-1)]
        idx = jnp.where(valid.reshape(-1), flat_slots.reshape(-1), total_slots)
        return (
            flat_cache.at[idx]
            .set(vals, mode="drop")
            .reshape(kv_cache_local.shape[0], page_size, *kv_cache_local.shape[1:4])
            .transpose(0, 2, 3, 4, 1)
        )
    page_size = kv_cache_local.shape[1]
    total_slots = kv_cache_local.shape[0] * page_size
    flat_cache = kv_cache_local.reshape(total_slots, *kv_cache_local.shape[2:])
    vals = captured_kv_local[src_safe.reshape(-1)]
    idx = jnp.where(valid.reshape(-1), flat_slots.reshape(-1), total_slots)
    return flat_cache.at[idx].set(vals, mode="drop").reshape(kv_cache_local.shape)


@dataclasses.dataclass(frozen=True)
class _OverflowStepSpec:
    """Batch geometry for multi-window overflow writeback verification."""

    pcp_size: int = 8
    page_size: int = 128
    interleave_size: int = 128
    local_tokens: int = 128
    kv_heads: int = 2
    q_per_kv: int = 2
    head_dim: int = 128
    kv_packing: int = 4
    num_blocks: int = 8
    max_seqs: int = 4


def _make_step_runner_for_layout(
    kv_layout_val: pcp_kernel.KVLayout,
    spec: _OverflowStepSpec,
) -> typing.Callable[..., jax.Array]:
    """Construct a JIT-compiled vmap step runner over PCP ranks."""
    stage_config = pcp_stage_plan.PCPStageConfig(
        local_tokens=spec.local_tokens,
        local_kv_cache_num_blocks=spec.num_blocks,
        page_size=spec.page_size,
        pcp_size=spec.pcp_size,
        interleave_size=spec.interleave_size,
        vpage_group_size=4,
        kv_layout=kv_layout_val,
        pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
    )
    metadata_entry = (
        pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata
    )

    @jax.jit
    def _run_step(
        q_sharded: jax.Array,
        kv_sharded: jax.Array,
        cache_sharded: jax.Array,
        meta_pack: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        """Execute one simulated prefill step across all PCP ranks."""
        kv_lens_jax, page_indices_jax, cu_q_lens_jax, distribution_jax = meta_pack

        def _per_rank(
            q_loc: jax.Array, kv_loc: jax.Array, cache_loc: jax.Array
        ) -> jax.Array:
            """Build stage plan and run metadata entry on one PCP rank."""
            plan = pcp_stage_plan.build_pcp_local_stage_plan(
                kv_lens_jax,
                page_indices_jax,
                cu_q_lens_jax,
                distribution_jax,
                config=stage_config,
            )
            _, next_cache = metadata_entry(
                q_loc,
                kv_loc,
                cache_loc,
                kv_lens_jax,
                cu_q_lens_jax,
                distribution_jax,
                plan.segment_descriptors,
                plan.num_segments,
                pcp_size=spec.pcp_size,
                interleave_size=spec.interleave_size,
                sm_scale=1.0,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=kv_layout_val,
                pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
                stage_plan=plan,
                vpage_group_size=4,
            )
            return next_cache

        return jax.vmap(
            _per_rank,
            in_axes=(0, 0, 0),
            out_axes=0,
            axis_name=pcp_stage_plan.PCP_AXIS_NAME,
        )(q_sharded, kv_sharded, cache_sharded)

    return _run_step


def _fill_rank_step_tokens(
    cur_kv_np: np.ndarray,
    cache_ref: np.ndarray,
    block_tables_np: np.ndarray,
    step_ctx: tuple[int, list[int], list[int], list[int]],
    layout: pcp_kernel.KVLayout,
) -> int:
    """Populate cur_kv_np and cache_ref for one step and return rank-0 write count."""
    spec = _OverflowStepSpec()
    step_idx, added, seq_so_far, q_starts = step_ctx
    cycle = spec.pcp_size * spec.interleave_size
    rank0_writes = 0
    for rank in range(spec.pcp_size):
        owner_rows: list[tuple[int, int]] = []
        for req_idx, add_len in enumerate(added):
            history_start = seq_so_far[req_idx] - add_len
            owner_rows.extend(
                (req_idx, history_start + (owner_pos - q_starts[req_idx]))
                for owner_pos in range(q_starts[req_idx], q_starts[req_idx + 1])
                if (owner_pos % cycle) // spec.interleave_size == rank
            )
        for row_cursor, (req_idx, abs_pos) in enumerate(owner_rows):
            val = float(((step_idx + 1) * 3 + req_idx + abs_pos) % 7 + 1)
            vpage = abs_pos // (spec.pcp_size * spec.page_size)
            page_rem = abs_pos % (spec.pcp_size * spec.page_size)
            cache_rank = (page_rem // spec.interleave_size) % spec.pcp_size
            page_offset = (page_rem // cycle) * spec.interleave_size + (
                page_rem % spec.interleave_size
            )
            block_id = int(block_tables_np[req_idx, vpage])
            if layout == pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE:
                cur_kv_np[rank, row_cursor] = val
                cache_ref[cache_rank, block_id, page_offset] = val
            else:
                cur_kv_np[rank, :, :, :, row_cursor] = val
                cache_ref[cache_rank, block_id, :, :, :, page_offset] = val
            if step_idx == 0 and cache_rank == 0:
                rank0_writes += 1
    return rank0_writes


def test_packed_local_from_metadata_with_overflow_writeback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-window writeback when a cache rank owns > local_tokens rows."""
    spec = _OverflowStepSpec()
    steps_added = [[53, 1, 129], [1, 3, 2], [2, 1, 1]]
    block_tables_np = np.zeros((spec.max_seqs, 2), dtype=np.int32)
    block_tables_np[0], block_tables_np[1], block_tables_np[2] = [0, 1], [2, 3], [4, 5]
    seen_desc_shapes: list[tuple[int, ...]] = []

    def _recording_write(
        kv_cache_local: jax.Array,
        captured_kv_local: jax.Array,
        segment_descriptors: jax.Array,
        num_segments: jax.Array,
        *,
        kv_layout: pcp_kernel.KVLayout = pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
    ) -> jax.Array:
        """Record descriptor shape and execute pure JAX slot writeback."""
        seen_desc_shapes.append(tuple(int(d) for d in segment_descriptors.shape))
        return _run_pure_write_slots(
            kv_cache_local,
            captured_kv_local,
            segment_descriptors,
            num_segments,
            kv_layout=kv_layout,
        )

    monkeypatch.setattr(
        pcp_kernel, "write_captured_kv_to_local_cache", _recording_write
    )
    monkeypatch.setattr(
        pcp_kernel,
        "_attend_over_vpage_groups",
        lambda _kv, q_scaled_head_major, _plan, **_kwargs: jnp.zeros(
            q_scaled_head_major.shape, dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        pcp_stage_plan,
        "get_pcp_rank_without_axis_index",
        lambda axis_name, _pcp_size: jax.lax.axis_index(axis_name),
    )

    for layout in (
        pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
        pcp_kernel.KVLayout.SEQ_ALONG_LANE,
    ):
        is_head_along_sublane = layout == pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE
        cache_shape = (
            (
                spec.pcp_size,
                spec.num_blocks,
                spec.page_size,
                1,
                spec.kv_packing,
                spec.head_dim,
            )
            if is_head_along_sublane
            else (
                spec.pcp_size,
                spec.num_blocks,
                2 * spec.kv_heads,
                spec.head_dim // spec.kv_packing,
                spec.kv_packing,
                spec.page_size,
            )
        )
        cur_shape = (
            (spec.pcp_size, spec.local_tokens, 1, spec.kv_packing, spec.head_dim)
            if is_head_along_sublane
            else (
                spec.pcp_size,
                2 * spec.kv_heads,
                spec.head_dim // spec.kv_packing,
                spec.kv_packing,
                spec.local_tokens,
            )
        )
        run_step_fn = _make_step_runner_for_layout(layout, spec)
        cache_actual = jnp.full(cache_shape, -2.0, dtype=jnp.float8_e4m3fn)
        cache_ref = np.full(cache_shape, -2.0, dtype=np.float32)
        seq_so_far = [0, 0, 0]
        step0_meta: tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None = None
        step0_expected_writes: int | None = None

        for step_idx, added in enumerate(steps_added):
            q_starts = [0, *np.cumsum(added).tolist()]
            cu_q_np = np.full((spec.max_seqs + 1,), q_starts[-1], dtype=np.int32)
            cu_q_np[: len(q_starts)] = q_starts
            for req_idx, add_len in enumerate(added):
                seq_so_far[req_idx] += add_len
            kv_lens_np = np.array([*seq_so_far, 0], dtype=np.int32)
            cur_kv_np = np.zeros(cur_shape, dtype=np.float32)
            rank0_writes = _fill_rank_step_tokens(
                cur_kv_np,
                cache_ref,
                block_tables_np,
                (step_idx, added, seq_so_far, q_starts),
                layout,
            )
            meta_pack = (
                jnp.asarray(kv_lens_np),
                jnp.asarray(block_tables_np.reshape(-1)),
                jnp.asarray(cu_q_np),
                jnp.asarray([0, 0, 3], dtype=jnp.int32),
            )
            if step_idx == 0:
                step0_expected_writes = rank0_writes
                step0_meta = meta_pack
            q_dummy = jnp.zeros(
                (
                    spec.pcp_size,
                    spec.local_tokens,
                    spec.kv_heads,
                    spec.q_per_kv,
                    spec.head_dim,
                ),
                dtype=jnp.bfloat16,
            )
            cache_actual = run_step_fn(
                q_dummy,
                jnp.asarray(cur_kv_np, dtype=jnp.float8_e4m3fn),
                cache_actual,
                meta_pack,
            )

        expected_fp8 = np.asarray(
            jnp.asarray(cache_ref, dtype=jnp.float8_e4m3fn), dtype=np.float32
        )
        np.testing.assert_allclose(
            np.asarray(cache_actual, dtype=np.float32), expected_fp8, rtol=0.0, atol=0.0
        )
        assert step0_meta is not None
        step0_stage_config = pcp_stage_plan.PCPStageConfig(
            local_tokens=spec.local_tokens,
            local_kv_cache_num_blocks=spec.num_blocks,
            page_size=spec.page_size,
            pcp_size=spec.pcp_size,
            interleave_size=spec.interleave_size,
            vpage_group_size=4,
            kv_layout=layout,
            pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
        )
        step0_max_writeback = jax.vmap(
            lambda _, inps=step0_meta, stage_cfg=step0_stage_config: (
                pcp_stage_plan.build_pcp_local_stage_plan(
                    *inps,
                    config=stage_cfg,
                ).max_writeback_count
            ),
            in_axes=0,
            out_axes=0,
            axis_name=pcp_stage_plan.PCP_AXIS_NAME,
        )(jnp.arange(spec.pcp_size, dtype=jnp.int32))
        assert step0_expected_writes is not None
        assert int(np.asarray(step0_max_writeback)[0]) == step0_expected_writes
        assert bool(seen_desc_shapes)
        expected_segment_cap = min(
            spec.local_tokens,
            (spec.local_tokens + spec.interleave_size - 1) // spec.interleave_size
            + 2 * spec.max_seqs,
        )
        assert all(shape == (3, expected_segment_cap) for shape in seen_desc_shapes)


def test_stage_plan_widens_segment_cap_when_interleave_is_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify stage plan widens segment cap above 512 on fine interleave."""
    monkeypatch.setattr(
        pcp_stage_plan,
        "get_pcp_rank_without_axis_index",
        lambda _axis, _size: jnp.asarray(0, dtype=jnp.int32),
    )
    # With pcp_size=2, interleave_size=1, page_size=1 (virtual_page=2 tokens),
    # 600 non-adjacent page numbers produce 600 > 512 writeback segments on rank 0.
    num_vpages = 600
    widened_local_tokens = 1024
    page_numbers = jnp.arange(num_vpages, dtype=jnp.int32) * 2
    plan = pcp_stage_plan.build_pcp_local_stage_plan(
        jnp.asarray([num_vpages * 2, 0], dtype=jnp.int32),
        jnp.pad(page_numbers[None, :], ((0, 1), (0, 0))).reshape(-1),
        jnp.asarray([0, num_vpages * 2, num_vpages * 2], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        config=pcp_stage_plan.PCPStageConfig(
            local_tokens=widened_local_tokens,
            local_kv_cache_num_blocks=2048,
            page_size=1,
            pcp_size=2,
            interleave_size=1,
            vpage_group_size=32,
        ),
    )
    assert plan.segment_descriptors.shape == (3, widened_local_tokens)
    assert int(plan.num_segments) == num_vpages
    assert int(jnp.sum(plan.segment_descriptors[2])) == num_vpages


def test_attend_over_vpage_groups_routes_rung_96_on_nondivisible_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify rung 96 is selected even when total_vpages is not divisible by 96."""

    def _fake_single_group(
        _kv_cache_local: jax.Array,
        q_scaled_head_major: jax.Array,
        _stage_plan: object,
        rung_vpages: int,
        **_kwargs: object,
    ) -> jax.Array:
        """Return rung_vpages as float32 sentinel."""
        return jnp.full_like(q_scaled_head_major, float(rung_vpages), dtype=jnp.float32)

    monkeypatch.setattr(pcp_kernel, "_attend_single_vpage_group", _fake_single_group)
    monkeypatch.setattr(
        pcp_kernel,
        "_attend_multi_vpage_groups",
        lambda _kv, q_scaled_head_major, _plan, _vpg, **_kw: jnp.full_like(
            q_scaled_head_major, -1.0, dtype=jnp.float32
        ),
    )
    splash_config = _make_test_splash_cfg(
        block_kv=pcp_kernel.SPLASH_BLOCK_KV_MAX,
        num_stacked_q_heads=pcp_kernel.SPLASH_MAX_STACKED_Q_HEADS,
    )
    stream_spec = _SplashStreamSpec(
        kv_heads=2,
        head_dim=256,
        page_size=256,
        pcp_size=8,
        interleave_size=256,
        kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
        pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
        vpage_group_size=128,
        splash_config=splash_config,
    )
    for total_vpages in (512, 8192):
        for active_vpages, expected_rung in (
            (8, 32),
            (40, 64),
            (80, 96),
            (96, 96),
            (124, 128),
            (508, 512),
        ):
            plan = _PCPStagePlanStub(
                segment_descriptors=jnp.zeros((3, 8), dtype=jnp.int32),
                num_segments=jnp.asarray(0, dtype=jnp.int32),
                sorted_block_ids=jnp.zeros((total_vpages,), dtype=jnp.int32),
                sorted_seg_ids=jnp.zeros((total_vpages,), dtype=jnp.int32),
                sorted_vpage_valid_tokens=jnp.full(
                    (total_vpages,), 2048, dtype=jnp.int32
                ),
                num_active_vpages=jnp.asarray(active_vpages, dtype=jnp.int32),
                q_sequence=jnp.arange(128, dtype=jnp.int32),
                query_row_valid=jnp.ones((128,), dtype=jnp.bool_),
            )
            out = _attend_over_vpage_groups(
                jnp.zeros((128, 256, 1, 4, 256), dtype=jnp.float8_e4m3fn),
                jnp.zeros((32, 128, 256), dtype=jnp.bfloat16),
                plan,
                spec=stream_spec,
            )
            assert float(out[0, 0, 0]) == float(expected_rung)


def test_pcp_streaming_attention_selects_block_kv_1024_for_aligned_vpages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify metadata entry selects block_kv=1024 iff virtual_page_size % 1024 == 0."""
    seen_block_kv: list[int] = []
    monkeypatch.setattr(
        pcp_kernel,
        "_route_current_kv_to_captured_kv",
        lambda kv_in, *_args, **_kwargs: kv_in,
    )
    monkeypatch.setattr(
        pcp_kernel,
        "write_captured_kv_to_local_cache",
        lambda cache, *_args, **_kwargs: cache,
    )

    def _fake_attend(
        _kv_cache_local: jax.Array,
        q_scaled_head_major: jax.Array,
        _stage_plan: object,
        *,
        spec: _SplashStreamSpec,
    ) -> jax.Array:
        """Record SplashConfig.block_kv and return zeros."""
        assert spec.splash_config is not None
        seen_block_kv.append(int(spec.splash_config.block_kv))
        return jnp.zeros_like(q_scaled_head_major, dtype=jnp.float32)

    monkeypatch.setattr(pcp_kernel, "_attend_over_vpage_groups", _fake_attend)
    stage_plan = _PCPStagePlanStub()
    for page_size, expected_block_kv in ((256, 1024), (128, 1024), (64, 512)):
        seen_block_kv.clear()
        pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
            jnp.zeros((128, 2, 16, 256), dtype=jnp.bfloat16),
            jnp.zeros((128, 1, 4, 256), dtype=jnp.float8_e4m3fn),
            jnp.zeros((4, page_size, 1, 4, 256), dtype=jnp.float8_e4m3fn),
            jnp.asarray([128], dtype=jnp.int32),
            jnp.asarray([0, 128], dtype=jnp.int32),
            jnp.asarray([0, 0, 1], dtype=jnp.int32),
            jnp.zeros((3, 8), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            sm_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            stage_plan=stage_plan,
            vpage_group_size=128,
            pcp_size=8,
            interleave_size=page_size,
        )
        assert seen_block_kv == [expected_block_kv]


def _make_splash_alignment_stage_plan(
    local_tokens: int,
    vpage_group_size: int,
) -> pcp_stage_plan.PCPStagePlan:
    """Build a minimal real PCPStagePlan for metadata alignment checks."""
    num_vpages = max(1, vpage_group_size)
    return pcp_stage_plan.PCPStagePlan(
        segment_descriptors=jnp.zeros((3, 8), dtype=jnp.int32),
        num_segments=jnp.asarray(0, dtype=jnp.int32),
        sorted_block_ids=jnp.zeros((num_vpages,), dtype=jnp.int32),
        sorted_seg_ids=jnp.zeros((num_vpages,), dtype=jnp.int32),
        sorted_vpage_valid_tokens=jnp.zeros((num_vpages,), dtype=jnp.int32),
        num_active_vpages=jnp.asarray(0, dtype=jnp.int32),
        q_seg_ids=jnp.zeros((local_tokens,), dtype=jnp.int32),
        q_sequence=jnp.arange(local_tokens, dtype=jnp.int32),
        query_row_valid=jnp.ones((local_tokens,), dtype=jnp.bool_),
        needs_reroute=jnp.zeros((), dtype=jnp.bool_),
        max_writeback_count=jnp.asarray(0, dtype=jnp.int32),
        block_tables=jnp.zeros((1, 1), dtype=jnp.int32),
        pcp_rank=jnp.asarray(0, dtype=jnp.int32),
    )


def test_metadata_entry_rejects_unaligned_local_tokens_for_splash() -> None:
    """Verify metadata entry rejects non-positive or non-128-aligned local_tokens."""
    page_size = 128
    pcp_size = 8
    kv_heads = 2
    q_per_kv = 2
    head_dim = 128
    vpage_group_size = 128
    kv_cache_local = jnp.zeros((4, page_size, 2, 2, head_dim), dtype=jnp.bfloat16)
    kv_lens = jnp.asarray([128], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    writeback_segment_descriptors = jnp.zeros((3, 8), dtype=jnp.int32)
    writeback_num_segments = jnp.asarray(0, dtype=jnp.int32)

    for bad_local_tokens in (64, 0):
        q_local = jnp.zeros(
            (bad_local_tokens, kv_heads, q_per_kv, head_dim),
            dtype=jnp.bfloat16,
        )
        current_kv_local = jnp.zeros(
            (bad_local_tokens, 2, 2, head_dim),
            dtype=jnp.bfloat16,
        )
        cu_q_lens = jnp.asarray([0, bad_local_tokens], dtype=jnp.int32)
        stage_plan = _make_splash_alignment_stage_plan(
            bad_local_tokens, vpage_group_size
        )
        with pytest.raises(ValueError, match="divisible by 128"):
            pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_local,
                current_kv_local,
                kv_cache_local,
                kv_lens,
                cu_q_lens,
                distribution,
                writeback_segment_descriptors,
                writeback_num_segments,
                pcp_size=pcp_size,
                interleave_size=page_size,
                sm_scale=1.0,
                stage_plan=stage_plan,
                vpage_group_size=vpage_group_size,
            )


def test_metadata_entry_rejects_unaligned_group_tokens_for_splash() -> None:
    """Verify metadata entry rejects non-positive or block_kv-misaligned groups."""
    local_tokens = 128
    page_size = 64
    pcp_size = 4
    kv_heads = 2
    q_per_kv = 2
    head_dim = 128
    q_local = jnp.zeros(
        (local_tokens, kv_heads, q_per_kv, head_dim),
        dtype=jnp.bfloat16,
    )
    current_kv_local = jnp.zeros((local_tokens, 2, 2, head_dim), dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros((4, page_size, 2, 2, head_dim), dtype=jnp.bfloat16)
    kv_lens = jnp.asarray([local_tokens], dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, local_tokens], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    writeback_segment_descriptors = jnp.zeros((3, 8), dtype=jnp.int32)
    writeback_num_segments = jnp.asarray(0, dtype=jnp.int32)

    for bad_vpage_group_size in (1, 0, -128):
        stage_plan = _make_splash_alignment_stage_plan(
            local_tokens, bad_vpage_group_size
        )
        with pytest.raises(ValueError, match="divisible by block_kv"):
            pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_local,
                current_kv_local,
                kv_cache_local,
                kv_lens,
                cu_q_lens,
                distribution,
                writeback_segment_descriptors,
                writeback_num_segments,
                pcp_size=pcp_size,
                interleave_size=page_size,
                sm_scale=1.0,
                stage_plan=stage_plan,
                vpage_group_size=bad_vpage_group_size,
            )


def test_stage_plan_exposes_pcp_rank_and_single_rank_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PCPStagePlan computes pcp_rank once and routing/overflow reuse it."""
    assert "pcp_rank" in pcp_stage_plan.PCPStagePlan._fields

    spec = _OverflowStepSpec()
    added = [53, 1, 129]
    q_starts = [0, *np.cumsum(added).tolist()]
    cu_q_np = np.full((spec.max_seqs + 1,), q_starts[-1], dtype=np.int32)
    cu_q_np[: len(q_starts)] = q_starts
    kv_lens_np = np.array([*added, 0], dtype=np.int32)
    block_tables_np = np.zeros((spec.max_seqs, 2), dtype=np.int32)
    block_tables_np[0], block_tables_np[1], block_tables_np[2] = [0, 1], [2, 3], [4, 5]
    kv_lens_jax = jnp.asarray(kv_lens_np, dtype=jnp.int32)
    page_indices_jax = jnp.asarray(block_tables_np.reshape(-1), dtype=jnp.int32)
    cu_q_lens_jax = jnp.asarray(cu_q_np, dtype=jnp.int32)
    distribution_jax = jnp.asarray([0, 0, 3], dtype=jnp.int32)

    rank_collective_calls = 0

    def _counting_pcp_rank(axis_name: str, _pcp_size: int) -> jax.Array:
        """Count PCP rank collective invocations and return axis_index."""
        nonlocal rank_collective_calls
        rank_collective_calls += 1
        return jax.lax.axis_index(axis_name)

    monkeypatch.setattr(
        pcp_stage_plan,
        "get_pcp_rank_without_axis_index",
        _counting_pcp_rank,
    )
    monkeypatch.setattr(
        pcp_kernel,
        "write_captured_kv_to_local_cache",
        _run_pure_write_slots,
    )
    monkeypatch.setattr(
        pcp_kernel,
        "_attend_over_vpage_groups",
        lambda _kv, q_scaled_head_major, _plan, **_kwargs: jnp.zeros(
            q_scaled_head_major.shape, dtype=jnp.float32
        ),
    )

    stage_config = pcp_stage_plan.PCPStageConfig(
        local_tokens=spec.local_tokens,
        local_kv_cache_num_blocks=spec.num_blocks,
        page_size=spec.page_size,
        pcp_size=spec.pcp_size,
        interleave_size=spec.interleave_size,
        vpage_group_size=4,
        kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
        pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
    )

    def _build_per_rank(_dummy_rank: jax.Array) -> pcp_stage_plan.PCPStagePlan:
        """Build one rank's stage plan inside vmap."""
        return pcp_stage_plan.build_pcp_local_stage_plan(
            kv_lens_jax,
            page_indices_jax,
            cu_q_lens_jax,
            distribution_jax,
            config=stage_config,
        )

    plan_by_rank = jax.vmap(
        _build_per_rank,
        in_axes=0,
        out_axes=0,
        axis_name=pcp_stage_plan.PCP_AXIS_NAME,
    )(jnp.arange(spec.pcp_size, dtype=jnp.int32))
    assert rank_collective_calls == 1
    assert np.array_equal(
        np.asarray(plan_by_rank.pcp_rank),
        np.arange(spec.pcp_size, dtype=np.int32),
    )
    assert bool(np.all(np.asarray(plan_by_rank.needs_reroute)))
    assert int(np.asarray(plan_by_rank.max_writeback_count)[0]) > spec.local_tokens

    rank_collective_calls = 0
    q_sharded = jnp.zeros(
        (spec.pcp_size, spec.local_tokens, spec.kv_heads, spec.q_per_kv, spec.head_dim),
        dtype=jnp.bfloat16,
    )
    current_kv_sharded = jnp.zeros(
        (spec.pcp_size, spec.local_tokens, 1, spec.kv_packing, spec.head_dim),
        dtype=jnp.float8_e4m3fn,
    )
    cache_sharded = jnp.zeros(
        (
            spec.pcp_size,
            spec.num_blocks,
            spec.page_size,
            1,
            spec.kv_packing,
            spec.head_dim,
        ),
        dtype=jnp.float8_e4m3fn,
    )

    def _run_per_rank(
        q_local: jax.Array,
        current_kv_local: jax.Array,
        kv_cache_local: jax.Array,
        stage_plan_rank: pcp_stage_plan.PCPStagePlan,
    ) -> jax.Array:
        """Run metadata entry with prebuilt stage plan on one rank."""
        _, updated_cache = (
            pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_local,
                current_kv_local,
                kv_cache_local,
                kv_lens_jax,
                cu_q_lens_jax,
                distribution_jax,
                stage_plan_rank.segment_descriptors,
                stage_plan_rank.num_segments,
                pcp_size=spec.pcp_size,
                interleave_size=spec.interleave_size,
                sm_scale=1.0,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=pcp_kernel.KVLayout.HEAD_ALONG_SUBLANE,
                pcp_axis_name=pcp_stage_plan.PCP_AXIS_NAME,
                stage_plan=stage_plan_rank,
                vpage_group_size=4,
            )
        )
        return updated_cache

    _ = jax.vmap(
        _run_per_rank,
        in_axes=(0, 0, 0, 0),
        out_axes=0,
        axis_name=pcp_stage_plan.PCP_AXIS_NAME,
    )(q_sharded, current_kv_sharded, cache_sharded, plan_by_rank)
    assert rank_collective_calls == 0


def test_metadata_kernel_requires_sm_scale_and_rejects_removed_kwargs() -> None:
    """Verify metadata entry requires sm_scale and rejects removed kwargs."""
    metadata_entry: typing.Callable[..., typing.Any] = (
        pcp_kernel.pcp_streaming_attention_page_groups_packed_local_from_metadata
    )
    local_tokens = 128
    page_size = 64
    pcp_size = 4
    kv_heads = 2
    q_per_kv = 2
    head_dim = 128
    vpage_group_size = 4
    positional_args = (
        jnp.zeros((local_tokens, kv_heads, q_per_kv, head_dim), dtype=jnp.bfloat16),
        jnp.zeros((local_tokens, 2, 2, head_dim), dtype=jnp.bfloat16),
        jnp.zeros((4, page_size, 2, 2, head_dim), dtype=jnp.bfloat16),
        jnp.asarray([local_tokens], dtype=jnp.int32),
        jnp.asarray([0, local_tokens], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        jnp.zeros((3, 8), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    base_kwargs: dict[str, typing.Any] = {
        "pcp_size": pcp_size,
        "interleave_size": page_size,
        "stage_plan": _PCPStagePlanStub(),
        "vpage_group_size": vpage_group_size,
    }
    with pytest.raises(TypeError, match="sm_scale"):
        metadata_entry(*positional_args, **base_kwargs)
    with pytest.raises(TypeError, match="num_lanes"):
        metadata_entry(*positional_args, sm_scale=1.0, num_lanes=1, **base_kwargs)
    with pytest.raises(TypeError, match="mesh_axis_names"):
        metadata_entry(
            *positional_args,
            sm_scale=1.0,
            mesh_axis_names=(pcp_stage_plan.PCP_AXIS_NAME,),
            **base_kwargs,
        )
