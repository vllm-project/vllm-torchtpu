# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import \
    wrapper as pcp_wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_rank_major_token_order


def _build_expected_pcp_local_slot_ids(
    q_lens: np.ndarray,
    seq_lens: np.ndarray,
    block_tables: np.ndarray,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    padded_num_tokens: int,
) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("PCP slot ids require block_size > 0.")
    if pcp_size <= 1:
        raise ValueError("PCP slot ids require pcp_size > 1.")
    if interleave_size <= 0:
        raise ValueError("PCP slot ids require interleave_size > 0.")
    if block_size != interleave_size:
        raise NotImplementedError(
            "Native PCP prefill slot ids currently require "
            f"block_size == interleave_size, got {block_size=} "
            f"{interleave_size=}.")
    if padded_num_tokens % pcp_size:
        raise ValueError(
            f"{padded_num_tokens=} must be divisible by {pcp_size=}.")

    q_lens = np.asarray(q_lens, dtype=np.int64)
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    if q_lens.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("q_lens and seq_lens must be 1D arrays.")
    if q_lens.size != seq_lens.size:
        raise ValueError("q_lens and seq_lens must have the same size.")
    if block_tables.ndim != 2 or block_tables.shape[0] < q_lens.size:
        raise ValueError("block_tables must cover every request.")
    if np.any(q_lens < 0):
        raise ValueError("q_lens must be non-negative.")
    if np.any(seq_lens < q_lens):
        raise ValueError("seq_lens must be greater than or equal to q_lens.")

    q_start_offsets = seq_lens - q_lens
    token_order, _ = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_start_offsets_per_req=q_start_offsets,
    )
    slot_ids = np.full(padded_num_tokens, -1, dtype=np.int32)
    valid = token_order >= 0
    if not np.any(valid):
        return slot_ids

    req_ids = []
    req_offsets = []
    for req_id, q_len in enumerate(q_lens):
        req_ids.extend([req_id] * int(q_len))
        req_offsets.extend(range(int(q_len)))
    req_ids_array = np.asarray(req_ids, dtype=np.int64)
    req_offsets_array = np.asarray(req_offsets, dtype=np.int64)

    valid_order = token_order[valid]
    packed_req_ids = req_ids_array[valid_order]
    global_positions = (q_start_offsets[packed_req_ids] +
                        req_offsets_array[valid_order])

    virtual_block_size = block_size * pcp_size
    block_indices = global_positions // virtual_block_size
    if (np.any(block_indices < 0)
            or int(block_indices.max(initial=-1)) >= block_tables.shape[1]):
        raise ValueError("block_tables does not cover requested PCP slots.")

    block_numbers = block_tables[packed_req_ids,
                                 block_indices].astype(np.int64)
    if np.any(block_numbers < 0):
        raise ValueError("block_tables contains invalid PCP slot pages.")
    virtual_offsets = global_positions - block_indices * virtual_block_size
    local_offsets = ((virtual_offsets //
                      (pcp_size * interleave_size)) * interleave_size +
                     (virtual_offsets % interleave_size))
    slot_ids[valid] = (block_numbers * block_size + local_offsets).astype(
        np.int32)
    return slot_ids


def _expected_local_slot_ids(q_lens, q_starts, block_tables, *,
                             local_kv_cache_num_blocks, page_size, pcp_size,
                             interleave_size, local_padded_tokens, pcp_rank,
                             target_num_reqs):
    q_lens = np.asarray(q_lens, dtype=np.int32)
    q_starts = np.asarray(q_starts, dtype=np.int32)
    q_lens_full = np.zeros(target_num_reqs, dtype=np.int32)
    seq_lens_full = np.zeros(target_num_reqs, dtype=np.int32)
    q_lens_full[:q_lens.size] = q_lens
    seq_lens_full[:q_lens.size] = q_starts + q_lens
    block_tables = np.asarray(block_tables, dtype=np.int32)
    localized_tables = block_tables % int(local_kv_cache_num_blocks)
    padded_tokens = int(local_padded_tokens) * int(pcp_size)
    slot_ids = _build_expected_pcp_local_slot_ids(
        q_lens_full,
        seq_lens_full,
        localized_tables,
        page_size,
        pcp_size,
        interleave_size,
        padded_tokens,
    )
    local_start = int(pcp_rank) * int(local_padded_tokens)
    return slot_ids[local_start:local_start + int(local_padded_tokens)]


def test_sharded_wrapper_uses_real_metadata_streaming_entry():
    source = inspect.getsource(pcp_wrapper)

    assert "pcp_streaming_attention_page_groups_packed_local_from_metadata" in source


def test_update_local_paged_kv_cache_accepts_equivalent_tail_layout(
        monkeypatch):
    packed_kv = jnp.arange(3 * 8 * 2 * 8,
                           dtype=jnp.float32).reshape(3, 8, 2, 8)

    def fake_prepare_inputs(*_args, **_kwargs):
        return None, packed_kv

    monkeypatch.setattr(pcp_wrapper.batched_rpa_wrapper, "prepare_inputs",
                        fake_prepare_inputs)

    kv_cache = jnp.zeros((2, 4, 16, 1, 8), dtype=jnp.float32)
    k = jnp.zeros((3, 8, 8), dtype=jnp.float32)
    v = jnp.zeros_like(k)
    slot_ids = jnp.array([0, 1, 2], dtype=jnp.int32)

    out = pcp_wrapper._update_local_paged_kv_cache(kv_cache, k, v, slot_ids)

    expected = packed_kv.reshape(3, 16, 1, 8)
    np.testing.assert_array_equal(np.asarray(out[0, :3]), np.asarray(expected))


def test_compute_pcp_local_slot_ids_from_metadata_matches_runner_reference():
    pcp_size = 4
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

    for pcp_rank in range(pcp_size):
        actual = pcp_wrapper.compute_pcp_local_slot_ids_from_metadata(
            jnp.asarray(seq_lens),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            local_padded_tokens=local_padded_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
        expected = _expected_local_slot_ids(
            q_lens,
            q_starts,
            block_tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            local_padded_tokens=local_padded_tokens,
            pcp_rank=pcp_rank,
            target_num_reqs=target_num_reqs,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_compute_pcp_local_slot_ids_from_decode_metadata_matches_reference():
    pcp_size = 4
    interleave_size = page_size = 4
    local_padded_tokens = 8
    target_num_reqs = 4
    local_kv_cache_num_blocks = 16
    q_lens = np.array([1, 1, 1], dtype=np.int32)
    q_starts = np.array([7, 15, 31], dtype=np.int32)
    seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
    seq_lens[:q_lens.size] = q_starts + q_lens
    cu_q_lens = np.zeros(target_num_reqs + 1, dtype=np.int32)
    cu_q_lens[1:q_lens.size + 1] = np.cumsum(q_lens)
    cu_q_lens[q_lens.size + 1:] = cu_q_lens[q_lens.size]
    distribution = np.array([q_lens.size, q_lens.size, q_lens.size],
                            dtype=np.int32)
    block_tables = (np.arange(target_num_reqs * 16, dtype=np.int32).reshape(
        target_num_reqs, 16) + 3)

    for pcp_rank in range(pcp_size):
        actual = pcp_wrapper.compute_pcp_local_slot_ids_from_metadata(
            jnp.asarray(seq_lens),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            local_padded_tokens=local_padded_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
        expected = _expected_local_slot_ids(
            q_lens,
            q_starts,
            block_tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            local_padded_tokens=local_padded_tokens,
            pcp_rank=pcp_rank,
            target_num_reqs=target_num_reqs,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_compute_pcp_local_slot_ids_from_mixed_metadata_matches_reference():
    pcp_size = 4
    interleave_size = page_size = 4
    local_padded_tokens = 16
    target_num_reqs = 4
    local_kv_cache_num_blocks = 16
    q_lens = np.array([1, 5, 9], dtype=np.int32)
    q_starts = np.array([7, 0, 13], dtype=np.int32)
    seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
    seq_lens[:q_lens.size] = q_starts + q_lens
    cu_q_lens = np.zeros(target_num_reqs + 1, dtype=np.int32)
    cu_q_lens[1:q_lens.size + 1] = np.cumsum(q_lens)
    cu_q_lens[q_lens.size + 1:] = cu_q_lens[q_lens.size]
    distribution = np.array([1, 2, q_lens.size], dtype=np.int32)
    block_tables = (np.arange(target_num_reqs * 16, dtype=np.int32).reshape(
        target_num_reqs, 16) + 5)

    for pcp_rank in range(pcp_size):
        actual = pcp_wrapper.compute_pcp_local_slot_ids_from_metadata(
            jnp.asarray(seq_lens),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            local_padded_tokens=local_padded_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
        expected = _expected_local_slot_ids(
            q_lens,
            q_starts,
            block_tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            local_padded_tokens=local_padded_tokens,
            pcp_rank=pcp_rank,
            target_num_reqs=target_num_reqs,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_compute_pcp_local_slot_ids_ignores_padded_request_slots():
    pcp_size = 4
    interleave_size = page_size = 4
    local_padded_tokens = 8
    target_num_reqs = 4
    local_kv_cache_num_blocks = 16
    active_q_lens = np.array([1, 5], dtype=np.int32)
    active_q_starts = np.array([7, 0], dtype=np.int32)
    all_q_lens = np.array([1, 5, 9], dtype=np.int32)
    all_q_starts = np.array([7, 0, 13], dtype=np.int32)
    seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
    seq_lens[:all_q_lens.size] = all_q_starts + all_q_lens
    cu_q_lens = np.zeros(target_num_reqs + 1, dtype=np.int32)
    cu_q_lens[1:all_q_lens.size + 1] = np.cumsum(all_q_lens)
    cu_q_lens[all_q_lens.size + 1:] = cu_q_lens[all_q_lens.size]
    distribution = np.array([1, 1, active_q_lens.size], dtype=np.int32)
    block_tables = (np.arange(target_num_reqs * 16, dtype=np.int32).reshape(
        target_num_reqs, 16) + 7)

    for pcp_rank in range(pcp_size):
        actual = pcp_wrapper.compute_pcp_local_slot_ids_from_metadata(
            jnp.asarray(seq_lens),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            local_padded_tokens=local_padded_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
        expected = _expected_local_slot_ids(
            active_q_lens,
            active_q_starts,
            block_tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            local_padded_tokens=local_padded_tokens,
            pcp_rank=pcp_rank,
            target_num_reqs=target_num_reqs,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)


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
    expected = _expected_local_slot_ids(
        q_lens,
        q_starts,
        block_tables,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_tokens=local_padded_tokens,
        pcp_rank=pcp_rank,
        target_num_reqs=target_num_reqs,
    )

    captured = {}

    def fake_update(kv_cache, _k, _v, slot_ids):
        captured["slot_ids"] = slot_ids
        return kv_cache

    def fake_streaming(q_streaming, *_args, **_kwargs):
        return jnp.full_like(q_streaming, 5)

    def fake_select_replicated_shard(tensor, _pcp_axis, _pcp_size, *, axis):
        assert axis == 0
        token_start = pcp_rank * local_padded_tokens
        token_end = token_start + local_padded_tokens
        return tensor[token_start:token_end]

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

    monkeypatch.setattr(pcp_wrapper, "_update_local_paged_kv_cache",
                        fake_update)
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

    np.testing.assert_array_equal(np.asarray(captured["slot_ids"]), expected)
    np.testing.assert_array_equal(
        np.asarray(output),
        np.full((local_padded_tokens, 2, 128), 5),
    )
    assert new_cache.shape == (local_kv_cache_num_blocks, page_size, 2, 1, 128)


def test_reshape_packed_kv_cache_for_attention_restores_dtype_packing():
    kv_cache = jnp.arange(2 * 4 * 16 * 1 * 8,
                          dtype=jnp.bfloat16).reshape(2, 4, 16, 1, 8)

    out = pcp_wrapper._reshape_packed_kv_cache_for_attention(kv_cache)

    assert out.shape == (2, 4, 8, 2, 8)
    np.testing.assert_array_equal(np.asarray(out.reshape(kv_cache.shape)),
                                  np.asarray(kv_cache))
