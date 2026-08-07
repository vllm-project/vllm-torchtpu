# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import \
    kernel as pcp_kernel
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import \
    wrapper as pcp_wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_rank_major_token_order

pytestmark = pytest.mark.multichip


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
    block_tables = (np.asarray(block_tables, dtype=np.int64) %
                    int(local_kv_cache_num_blocks))
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
        req_id = int(np.searchsorted(req_ends, request_major_row,
                                     side="right"))
        req_start = 0 if req_id == 0 else int(req_ends[req_id - 1])
        absolute_position = (int(absolute_starts[req_id]) +
                             int(request_major_row) - req_start)
        block_idx = absolute_position // virtual_block_size
        virtual_offset = absolute_position % virtual_block_size
        owner = (virtual_offset // interleave_size) % pcp_size
        local_offset = ((virtual_offset //
                         (pcp_size * interleave_size)) * interleave_size +
                        virtual_offset % interleave_size)
        slot = int(block_tables[req_id, block_idx] * block_size + local_offset)
        cache_owners[source_row] = owner
        cache_slots[source_row] = slot

    for req_id in range(q_lens.size):
        req_rows = np.flatnonzero(
            (token_order >= (0 if req_id == 0 else req_ends[req_id - 1]))
            & (token_order < req_ends[req_id]))
        req_rows = req_rows[np.argsort(
            np.asarray([
                absolute_starts[req_id] + token_order[row] -
                (0 if req_id == 0 else req_ends[req_id - 1])
                for row in req_rows
            ]))]
        for source_row in req_rows:
            owner = int(cache_owners[source_row])
            capture_offset = int(writeback_counts[owner])
            writeback_counts[owner] += 1
            capture_offsets[source_row] = capture_offset
            compact_slots[owner, capture_offset] = cache_slots[source_row]

    return SimpleNamespace(
        cache_owner_ranks=cache_owners,
        cache_local_slot_ids=cache_slots,
        capture_offsets=capture_offsets,
        writeback_counts=writeback_counts,
        compact_slot_ids=compact_slots,
    )


def test_sharded_wrapper_uses_real_metadata_streaming_entry():
    source = inspect.getsource(pcp_wrapper)

    assert "pcp_streaming_attention_page_groups_packed_local_from_metadata" in source


@pytest.mark.parametrize(
    ("kv_dtype", "kv_heads", "q_per_kv", "packed_kv_groups", "k_scale",
     "v_scale"),
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
        kv_dtype, kv_heads, q_per_kv, packed_kv_groups, k_scale, v_scale):
    local_tokens = 32
    page_size = 16
    head_dim = 128
    kv_packing = pcp_wrapper.get_dtype_packing(kv_dtype)
    q_local = jnp.zeros((local_tokens, kv_heads, q_per_kv, head_dim),
                        dtype=jnp.bfloat16)
    kv_cache_local = jnp.zeros(
        (2, page_size, packed_kv_groups, kv_packing, head_dim),
        dtype=kv_dtype,
    )

    assert pcp_kernel._validate_packed_kv_layout(q_local,
                                                 kv_cache_local,
                                                 k_scale=k_scale,
                                                 v_scale=v_scale) == (
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
        pcp_kernel._validate_packed_kv_layout(q_local,
                                              kv_cache_local,
                                              k_scale=0.5,
                                              v_scale=0.5)


def test_prepare_packed_kv_accepts_equivalent_tail_layout(monkeypatch):
    packed_kv = jnp.arange(3 * 8 * 2 * 8,
                           dtype=jnp.float32).reshape(3, 8, 2, 8)

    def fake_prepare_inputs(*_args, **_kwargs):
        return None, packed_kv

    monkeypatch.setattr(pcp_wrapper.batched_rpa_wrapper, "prepare_inputs",
                        fake_prepare_inputs)

    kv_cache = jnp.zeros((2, 4, 16, 1, 8), dtype=jnp.float32)
    k = jnp.zeros((3, 8, 8), dtype=jnp.float32)
    v = jnp.zeros_like(k)
    prepared = pcp_wrapper._prepare_packed_kv_for_cache(k, v, kv_cache)
    expected = packed_kv.reshape(3, 16, 1, 8)
    np.testing.assert_array_equal(np.asarray(prepared), np.asarray(expected))


def test_captured_kv_writeback_coalesces_contiguous_cache_slots():
    compact_slots = jnp.asarray([0, 1, 2, 8, 9, 20, -1, -1], dtype=jnp.int32)
    descriptors, num_segments = \
        pcp_wrapper._build_writeback_segment_descriptors(
            compact_slots,
            jnp.asarray(6),
        )

    assert int(num_segments) == 3
    np.testing.assert_array_equal(
        np.asarray(descriptors),
        np.asarray([
            [0, 3, 5, 0, 0, 0, 0, 0],
            [0, 8, 20, 0, 0, 0, 0, 0],
            [3, 2, 1, 0, 0, 0, 0, 0],
        ],
                   dtype=np.int32),
    )


def test_captured_kv_writeback_preserves_fully_fragmented_capacity():
    compact_slots = jnp.arange(16, dtype=jnp.int32) * 2
    descriptors, num_segments = \
        pcp_wrapper._build_writeback_segment_descriptors(
            compact_slots,
            jnp.asarray(compact_slots.shape[0]),
        )

    assert int(num_segments) == compact_slots.shape[0]
    np.testing.assert_array_equal(np.asarray(descriptors[0]),
                                  np.arange(16, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(descriptors[1]),
                                  np.arange(16, dtype=np.int32) * 2)
    np.testing.assert_array_equal(np.asarray(descriptors[2]),
                                  np.ones(16, dtype=np.int32))


def test_metadata_reconstruction_preserves_query_row_coordinates():
    kv_lens = jnp.asarray([8], dtype=jnp.int32)
    cu_q_lens = jnp.asarray([0, 5], dtype=jnp.int32)
    distribution = jnp.asarray([0, 0, 1], dtype=jnp.int32)

    mapping = pcp_wrapper._compute_pcp_local_query_row_mapping_from_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        local_padded_tokens=4,
        pcp_size=2,
        pcp_rank=0,
        interleave_size=2,
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
    common_kwargs = dict(
        q_lens=jnp.asarray([3, 4], dtype=jnp.int32),
        token_owner_starts=jnp.asarray([0, 3], dtype=jnp.int32),
        request_absolute_starts=jnp.asarray([5, 100], dtype=jnp.int32),
        local_padded_tokens=4,
        pcp_size=2,
        interleave_size=2,
    )

    rank_0 = pcp_wrapper._compute_pcp_local_query_row_mapping(pcp_rank=0,
                                                              **common_kwargs)
    rank_1 = pcp_wrapper._compute_pcp_local_query_row_mapping(pcp_rank=1,
                                                              **common_kwargs)

    np.testing.assert_array_equal(np.asarray(rank_0.absolute_positions),
                                  np.array([5, 6, 101, 102]))
    np.testing.assert_array_equal(np.asarray(rank_0.request_indices),
                                  np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(np.asarray(rank_1.absolute_positions),
                                  np.array([7, 100, 103, -1]))
    np.testing.assert_array_equal(np.asarray(rank_1.request_indices),
                                  np.array([0, 1, 1, -1]))
    np.testing.assert_array_equal(np.asarray(rank_1.is_valid),
                                  np.array([True, True, True, False]))
    # Source-rank identity remains available even when the row itself is pad.
    np.testing.assert_array_equal(np.asarray(rank_1.source_ranks),
                                  np.ones(4, dtype=np.int32))


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
        np.arange(q_lens.size * 16, dtype=np.int32).reshape(q_lens.size, 16) +
        7)

    query_rows = pcp_wrapper \
        ._compute_pcp_rank_major_query_row_mapping_from_metadata(
            jnp.asarray(seq_lens),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
    actual = pcp_wrapper._compute_pcp_kv_writeback_mapping(
        query_rows,
        jnp.asarray(block_tables),
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    jitted_actual = pcp_wrapper.jax.jit(
        lambda rows, tables: pcp_wrapper._compute_pcp_kv_writeback_mapping(
            rows,
            tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        ))(query_rows, jnp.asarray(block_tables))
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

    for field in pcp_wrapper._PCPKVWritebackMapping._fields:
        np.testing.assert_array_equal(np.asarray(getattr(actual, field)),
                                      getattr(expected, field))
        np.testing.assert_array_equal(
            np.asarray(getattr(jitted_actual, field)),
            getattr(expected, field))
    for cache_rank in range(pcp_size):
        compact_slots = pcp_wrapper \
            ._compact_writeback_slot_ids_for_cache_rank(actual, cache_rank)
        np.testing.assert_array_equal(np.asarray(compact_slots),
                                      expected.compact_slot_ids[cache_rank])


def test_one_token_writeback_reaches_cache_owner_without_local_q():
    pcp_size = 8
    page_size = interleave_size = 256
    local_padded_tokens = 4
    local_kv_cache_num_blocks = 32
    block_tables = jnp.asarray([[7]], dtype=jnp.int32)
    query_rows = pcp_wrapper \
        ._compute_pcp_rank_major_query_row_mapping_from_metadata(
            kv_lens=jnp.asarray([301], dtype=jnp.int32),
            cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
            distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
    mapping = pcp_wrapper._compute_pcp_kv_writeback_mapping(
        query_rows,
        block_tables,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )

    valid_row = int(np.flatnonzero(np.asarray(query_rows.is_valid))[0])
    assert int(query_rows.source_ranks[valid_row]) == 0
    assert int(mapping.cache_owner_ranks[valid_row]) == 1
    assert int(mapping.capture_offsets[valid_row]) == 0
    np.testing.assert_array_equal(
        np.asarray(mapping.writeback_counts),
        np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    )
    rank_1_slots = pcp_wrapper._compact_writeback_slot_ids_for_cache_rank(
        mapping, 1)
    assert int(rank_1_slots[0]) == 7 * page_size + 44
    assert np.all(np.asarray(rank_1_slots[1:]) == -1)


def test_batch_flat_metadata_32x256_balances_rows_with_cache_owner_skew():
    pcp_size = 8
    interleave_size = page_size = 256
    num_reqs = 32
    q_len = 256
    local_padded_tokens = 1280
    local_kv_cache_num_blocks = 32
    q_lens = jnp.full((num_reqs, ), q_len, dtype=jnp.int32)
    token_owner_starts = jnp.arange(num_reqs, dtype=jnp.int32) * q_len
    request_absolute_starts = jnp.zeros((num_reqs, ), dtype=jnp.int32)
    page_numbers = np.asarray([
        19, 3, 27, 11, 0, 24, 8, 30, 14, 6, 22, 1, 17, 29, 10, 25, 5, 21, 13,
        31, 2, 18, 28, 9, 23, 7, 16, 4, 26, 12, 20, 15
    ],
                              dtype=np.int32)
    block_tables = jnp.asarray(page_numbers[:, None])
    cu_q_lens = jnp.concatenate((jnp.zeros(
        (1, ), dtype=jnp.int32), jnp.cumsum(q_lens, dtype=jnp.int32)))
    distribution = jnp.asarray([0, 0, num_reqs], dtype=jnp.int32)

    mappings = [
        pcp_wrapper._compute_pcp_local_query_row_mapping(
            q_lens,
            token_owner_starts,
            request_absolute_starts,
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        ) for pcp_rank in range(pcp_size)
    ]
    metadata_mappings = [
        pcp_wrapper._compute_pcp_local_query_row_mapping_from_metadata(
            q_lens,
            cu_q_lens,
            distribution,
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        ) for pcp_rank in range(pcp_size)
    ]
    for expected, actual in zip(mappings, metadata_mappings):
        for field in pcp_wrapper._PCPQueryRowMapping._fields:
            np.testing.assert_array_equal(np.asarray(getattr(actual, field)),
                                          np.asarray(getattr(expected, field)))
    rank_major_query_rows = pcp_wrapper._PCPQueryRowMapping(
        *(jnp.concatenate([getattr(mapping, field) for mapping in mappings])
          for field in pcp_wrapper._PCPQueryRowMapping._fields))
    writeback_mapping = pcp_wrapper._compute_pcp_kv_writeback_mapping(
        rank_major_query_rows,
        block_tables,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )

    for source_rank, mapping in enumerate(mappings):
        assert int(np.asarray(mapping.is_valid).sum()) == 4 * q_len
        np.testing.assert_array_equal(
            np.asarray(mapping.source_ranks),
            np.full(local_padded_tokens, source_rank, dtype=np.int32),
        )
        assert np.all(np.asarray(mapping.request_indices)[4 * q_len:] == -1)

    cache_owner_ranks = np.asarray(writeback_mapping.cache_owner_ranks)
    cache_local_slot_ids = np.asarray(writeback_mapping.cache_local_slot_ids)
    cache_rank_0_slots = np.where(cache_owner_ranks == 0, cache_local_slot_ids,
                                  -1)
    cache_rank_1_slots = np.where(cache_owner_ranks == 1, cache_local_slot_ids,
                                  -1)
    assert np.count_nonzero(cache_rank_0_slots >= 0) == num_reqs * q_len
    assert np.all(cache_rank_1_slots < 0)
    np.testing.assert_array_equal(
        np.asarray(writeback_mapping.writeback_counts),
        np.array([num_reqs * q_len, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    )
    capture_offsets = np.asarray(writeback_mapping.capture_offsets)
    assert np.all(
        capture_offsets[~np.asarray(rank_major_query_rows.is_valid)] == -1)
    np.testing.assert_array_equal(
        np.sort(capture_offsets[capture_offsets >= 0]),
        np.arange(num_reqs * q_len, dtype=np.int32),
    )
    cache_rank_0_compact_slots = pcp_wrapper \
        ._compact_writeback_slot_ids_for_cache_rank(writeback_mapping, 0)
    np.testing.assert_array_equal(
        np.asarray(cache_rank_0_compact_slots)[:num_reqs * q_len].reshape(
            num_reqs, q_len)[:, 0],
        page_numbers * page_size,
    )
    valid = np.asarray(rank_major_query_rows.is_valid)
    request_indices = np.asarray(rank_major_query_rows.request_indices)[valid]
    absolute_positions = np.asarray(
        rank_major_query_rows.absolute_positions)[valid]
    np.testing.assert_array_equal(
        cache_rank_0_slots[valid],
        page_numbers[request_indices] * page_size + absolute_positions,
    )


def test_sharded_wrapper_generates_slot_ids_when_metadata_omits_them(
        monkeypatch):
    pcp_size = 4
    pcp_rank = 2
    interleave_size = page_size = 4
    local_padded_tokens = 16
    target_num_reqs = 4
    local_kv_cache_num_blocks = 8
    q_lens = np.array([5, 17, 6], dtype=np.int32)
    q_starts = np.array([3, 9, 0], dtype=np.int32)
    seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
    seq_lens[:q_lens.size] = q_starts + q_lens
    cu_q_lens = np.zeros(target_num_reqs + 1, dtype=np.int32)
    cu_q_lens[1:q_lens.size + 1] = np.cumsum(q_lens)
    cu_q_lens[q_lens.size + 1:] = cu_q_lens[q_lens.size]
    distribution = np.array([0, 0, q_lens.size], dtype=np.int32)
    block_tables = (np.arange(target_num_reqs * 8, dtype=np.int32).reshape(
        target_num_reqs, 8) + 11)
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

    def fake_streaming(q_streaming, _current_kv, kv_cache, _kv_lens,
                       _page_indices, _cu_q_lens, _distribution,
                       segment_descriptors, num_segments, **_kwargs):
        captured["segment_descriptors"] = segment_descriptors
        captured["num_segments"] = num_segments
        return jnp.full_like(q_streaming, 5), kv_cache

    def fake_prepare_current(k_local, _v_local, cache):
        return jnp.zeros((k_local.shape[0], *cache.shape[2:]),
                         dtype=cache.dtype)

    def fake_select_replicated_shard(tensor, _pcp_axis, _pcp_size, *, axis):
        assert axis == 0
        token_start = pcp_rank * local_padded_tokens
        token_end = token_start + local_padded_tokens
        local = tensor[token_start:token_end]
        if tensor.shape[1] == 4:
            captured["is_valid"] = local[:, 3].astype(jnp.bool_)
        return local

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        del mesh, in_specs, out_specs, check_vma

        def wrapped(*args):
            args = list(args)
            token_start = pcp_rank * local_padded_tokens
            token_end = token_start + local_padded_tokens
            block_start = pcp_rank * local_kv_cache_num_blocks
            block_end = block_start + local_kv_cache_num_blocks
            args[0] = args[0][token_start:token_end]
            args[1] = args[1][token_start:token_end]
            args[2] = args[2][token_start:token_end]
            args[3] = args[3][block_start:block_end]
            return fn(*args)

        return wrapped

    monkeypatch.setattr(pcp_wrapper, "_prepare_packed_kv_for_cache",
                        fake_prepare_current)
    monkeypatch.setattr(
        pcp_wrapper,
        "pcp_streaming_attention_page_groups_packed_local_from_metadata",
        fake_streaming,
    )
    monkeypatch.setattr(pcp_wrapper, "_select_replicated_shard_for_pcp_rank",
                        fake_select_replicated_shard)
    monkeypatch.setattr(pcp_wrapper.jax, "shard_map", fake_shard_map)

    mesh = SimpleNamespace(axis_names=("pcp", ), shape={"pcp": pcp_size})
    global_padded_tokens = local_padded_tokens * pcp_size
    q = jnp.ones((global_padded_tokens, 2, 128), dtype=jnp.float32)
    k = jnp.ones((global_padded_tokens, 1, 128), dtype=jnp.float32)
    v = jnp.ones((global_padded_tokens, 1, 128), dtype=jnp.float32)
    kv_cache = jnp.zeros(
        (local_kv_cache_num_blocks * pcp_size, page_size, 2, 1, 128),
        dtype=jnp.float32)

    output, new_cache = pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(seq_lens),
        page_indices=jnp.asarray(block_tables.reshape(-1)),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution),
        attention_sink=None,
        sm_scale=0.25,
        cp_kv_cache_interleave_size=interleave_size,
    )

    expected_descriptors, expected_num_segments = \
        pcp_wrapper._build_writeback_segment_descriptors(
            jnp.asarray(expected),
            jnp.asarray(expected_writeback.writeback_counts[pcp_rank]),
        )
    np.testing.assert_array_equal(
        np.asarray(captured["segment_descriptors"]),
        np.asarray(expected_descriptors),
    )
    assert int(captured["num_segments"]) == int(expected_num_segments)
    assert np.any(expected < 0)
    expected_output = np.zeros((local_padded_tokens, 2, 128), dtype=np.float32)
    expected_output[np.asarray(captured["is_valid"])] = 5
    np.testing.assert_array_equal(
        np.asarray(output),
        expected_output,
    )
    assert new_cache.shape == (local_kv_cache_num_blocks, page_size, 2, 1, 128)


def test_sharded_wrapper_masks_output_with_query_validity(monkeypatch):
    query_rows = pcp_wrapper._PCPQueryRowMapping(
        absolute_positions=jnp.array([0, -1, 2, -1], dtype=jnp.int32),
        request_indices=jnp.array([0, -1, 0, -1], dtype=jnp.int32),
        source_ranks=jnp.zeros(4, dtype=jnp.int32),
        is_valid=jnp.array([True, False, True, False]),
    )
    captured = {}

    def fake_compute_query_rows(*_args, **_kwargs):
        return query_rows

    def fake_select_query_rows(mapping, _pcp_axis, _pcp_size):
        assert mapping is query_rows
        return mapping

    def fake_streaming(q_streaming, _current_kv, kv_cache, _kv_lens,
                       _page_indices, _cu_q_lens, _distribution,
                       _segment_descriptors, _num_segments, **_kwargs):
        captured["writeback_called"] = True
        return jnp.full_like(q_streaming, 5), kv_cache

    def fake_prepare_current(k_local, _v_local, cache):
        return jnp.zeros((k_local.shape[0], *cache.shape[2:]),
                         dtype=cache.dtype)

    def fake_shard_map(fn, *, mesh, in_specs, out_specs, check_vma):
        del mesh, in_specs, out_specs, check_vma
        return fn

    monkeypatch.setattr(
        pcp_wrapper,
        "_compute_pcp_rank_major_query_row_mapping_from_metadata",
        fake_compute_query_rows,
    )
    monkeypatch.setattr(
        pcp_wrapper,
        "_select_replicated_query_row_mapping_for_pcp_rank",
        fake_select_query_rows,
    )
    monkeypatch.setattr(pcp_wrapper, "_prepare_packed_kv_for_cache",
                        fake_prepare_current)
    monkeypatch.setattr(
        pcp_wrapper,
        "pcp_streaming_attention_page_groups_packed_local_from_metadata",
        fake_streaming,
    )
    monkeypatch.setattr(pcp_wrapper.jax, "shard_map", fake_shard_map)

    num_tokens = 4
    page_size = 4
    mesh = SimpleNamespace(axis_names=("pcp", ), shape={"pcp": 1})
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
    kv_cache = jnp.arange(2 * 4 * 16 * 1 * 8,
                          dtype=jnp.bfloat16).reshape(2, 4, 16, 1, 8)

    with pytest.raises(ValueError, match="packing does not match"):
        pcp_wrapper._reshape_packed_kv_cache_for_attention(kv_cache)
