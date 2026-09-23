# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Composite wrappers for PCP streaming RPA."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as batched_rpa_wrapper,
)
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import get_dtype_packing
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    pcp_streaming_attention_page_groups_packed_local_from_metadata,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    _reshape_metadata_page_indices_jax,
)

PCP_AXIS_NAME = "pcp"

KVLayout = batched_rpa_configs.KVLayout


class _PCPQueryRowMapping(NamedTuple):
    """Query-row coordinates in the batch-flat PCP token stream."""

    absolute_positions: jax.Array
    request_indices: jax.Array
    source_ranks: jax.Array
    is_valid: jax.Array


class _PCPKVWritebackMapping(NamedTuple):
    """Unique cache destination and dense capture offset for each source row."""

    cache_owner_ranks: jax.Array
    cache_local_slot_ids: jax.Array
    capture_offsets: jax.Array
    writeback_counts: jax.Array


def _pack_kv_for_cache(
    k: jax.Array,
    v: jax.Array,
    cache_dtype: jnp.dtype,
    *,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int | None = None,
) -> jax.Array:
    """Pack K and V together before converting to the cache dtype."""
    dummy_q = jnp.zeros_like(k)
    _, packed_kv = batched_rpa_wrapper.prepare_inputs(
        dummy_q,
        k,
        v,
        k.dtype,
        cache_dtype,
        kv_layout=kv_layout,
    )
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if page_size is None or page_size <= 0:
            raise ValueError("PCP SEQ_ALONG_LANE KV packing requires page_size > 0.")
        padded_tokens = math.ceil(packed_kv.shape[-1] / page_size) * page_size
        packed_kv = jnp.pad(
            packed_kv,
            ((0, 0), (0, 0), (0, 0), (0, padded_tokens - packed_kv.shape[-1])),
            constant_values=0,
        )
    return packed_kv


def _kv_cache_page_size(kv_cache: jax.Array, kv_layout: KVLayout) -> int:
    if kv_cache.ndim != 5:
        raise ValueError(f"PCP KV cache must be 5D, got {kv_cache.shape}.")
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        return int(kv_cache.shape[-1])
    return int(kv_cache.shape[1])


def _prepare_packed_kv_for_cache(
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    *,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    """Pack K/V once and adapt the packed tail to the cache layout."""
    page_size = _kv_cache_page_size(kv_cache, kv_layout)
    packed_kv = _pack_kv_for_cache(
        k, v, kv_cache.dtype, kv_layout=kv_layout, page_size=page_size
    )
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if packed_kv.shape[:-1] != kv_cache.shape[1:-1]:
            raise ValueError(
                "SEQ_ALONG_LANE fresh KV prefix must match the cache head "
                f"layout: {packed_kv.shape[:-1]} vs {kv_cache.shape[1:-1]}."
            )
        if packed_kv.shape[-1] < k.shape[0]:
            raise ValueError(
                "SEQ_ALONG_LANE fresh KV token extent must cover all local "
                f"tokens: {packed_kv.shape[-1]} vs {k.shape[0]}."
            )
        return packed_kv.astype(kv_cache.dtype)

    if packed_kv.shape[1:] != kv_cache.shape[2:]:
        packed_tail = 1
        for dim in packed_kv.shape[1:]:
            packed_tail *= dim
        cache_tail = 1
        for dim in kv_cache.shape[2:]:
            cache_tail *= dim
        if packed_tail != cache_tail:
            raise ValueError(
                "packed KV update shape is incompatible with KV cache layout: "
                f"{packed_kv.shape[1:]} vs {kv_cache.shape[2:]}."
            )
        packed_kv = packed_kv.reshape((packed_kv.shape[0], *kv_cache.shape[2:]))
    return packed_kv.astype(kv_cache.dtype)


def _reshape_packed_kv_cache_for_attention(
    kv_cache: jax.Array,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    del kv_layout
    expected_packing = get_dtype_packing(kv_cache.dtype)
    if kv_cache.shape[3] == expected_packing:
        return kv_cache
    raise ValueError(
        "KV cache layout packing does not match dtype packing: "
        f"shape={kv_cache.shape}, "
        f"expected_packing={expected_packing}."
    )


def _select_replicated_shard_for_pcp_rank(
    tensor: jax.Array,
    pcp_axis: str,
    pcp_size: int,
    *,
    axis: int,
) -> jax.Array:
    """Select this PCP rank shard from replicated data without axis_index."""
    if axis < 0:
        axis += tensor.ndim
    if tensor.shape[axis] % pcp_size != 0:
        raise ValueError(
            "Replicated PCP tensor axis must be divisible by "
            f"pcp_size: shape={tensor.shape}, axis={axis}, "
            f"pcp_size={pcp_size}."
        )
    shard_size = tensor.shape[axis] // pcp_size
    exchanged = jax.lax.all_to_all(
        tensor, axis_name=pcp_axis, split_axis=axis, concat_axis=axis, tiled=True
    )
    return jax.lax.dynamic_slice_in_dim(exchanged, 0, shard_size, axis=axis)


def compute_pcp_local_mapping(
    positions: jax.Array,
    token_req_indices: jax.Array,
    block_tables: jax.Array,
    block_size: int,
    cp_size: int,
    cp_rank: int,
    interleave_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Compute vLLM-compatible local CP slot mapping for one PCP rank."""
    if block_size <= 0:
        raise ValueError(f"Expected positive block_size, got {block_size}.")
    if cp_size <= 0:
        raise ValueError(f"Expected positive cp_size, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"Expected cp_rank in [0, {cp_size}), got {cp_rank}.")
    if interleave_size <= 0:
        raise ValueError(f"Expected positive interleave_size, got {interleave_size}.")

    positions = positions.astype(jnp.int32)
    token_req_indices = token_req_indices.astype(jnp.int32)
    virtual_block_size = block_size * cp_size
    block_indices = positions // virtual_block_size
    block_numbers = block_tables[token_req_indices, block_indices].astype(jnp.int32)

    virtual_offsets = positions - block_indices * virtual_block_size
    is_local = ((virtual_offsets // interleave_size) % cp_size) == cp_rank
    local_offsets = (
        virtual_offsets // (cp_size * interleave_size)
    ) * interleave_size + (virtual_offsets % interleave_size)
    slot_ids = block_numbers * block_size + local_offsets
    slot_ids = jnp.where(is_local, slot_ids, -1)
    return is_local, slot_ids.astype(jnp.int32)


def _pcp_rank_token_count_before(
    position: jax.Array, pcp_rank: jax.Array, pcp_size: int, interleave_size: int
) -> jax.Array:
    cycle = pcp_size * interleave_size
    full_cycles = position // cycle
    cycle_offset = position - full_cycles * cycle
    rank_start = pcp_rank * interleave_size
    rank_tokens_in_partial = jnp.clip(cycle_offset - rank_start, 0, interleave_size)
    return full_cycles * interleave_size + rank_tokens_in_partial


def _compute_cache_owner_and_slot_ids_for_rows(
    absolute_positions: jax.Array,
    request_indices: jax.Array,
    row_valid: jax.Array,
    block_tables: jax.Array,
    *,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Map arbitrary source rows to their unique PCP cache owner and slot."""
    if local_kv_cache_num_blocks <= 0:
        raise ValueError("local_kv_cache_num_blocks must be positive.")
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive.")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive.")
    if interleave_size > page_size or page_size % interleave_size != 0:
        raise NotImplementedError(
            "PCP slot id reconstruction requires page_size to be divisible "
            f"by interleave_size: {page_size=} {interleave_size=}."
        )

    pages_per_seq = block_tables.shape[1]
    safe_positions = jnp.where(row_valid, absolute_positions, 0)
    safe_request_indices = jnp.where(row_valid, request_indices, 0)

    virtual_block_size = page_size * pcp_size
    block_indices = safe_positions // virtual_block_size
    safe_block_indices = jnp.clip(block_indices, 0, pages_per_seq - 1)
    block_numbers = block_tables[safe_request_indices, safe_block_indices]
    block_numbers = jnp.mod(
        block_numbers,
        jnp.asarray(local_kv_cache_num_blocks, dtype=jnp.int32),
    )
    virtual_offsets = safe_positions - block_indices * virtual_block_size
    cache_owners = (virtual_offsets // interleave_size) % pcp_size
    local_offsets = (
        virtual_offsets // (pcp_size * interleave_size)
    ) * interleave_size + (virtual_offsets % interleave_size)
    slot_ids = block_numbers * page_size + local_offsets
    invalid = jnp.asarray(-1, dtype=jnp.int32)
    return (
        jnp.where(row_valid, cache_owners, invalid).astype(jnp.int32),
        jnp.where(row_valid, slot_ids, invalid).astype(jnp.int32),
    )


def _compute_pcp_local_query_row_mapping(
    q_lens: jax.Array,
    token_owner_starts: jax.Array,
    request_absolute_starts: jax.Array,
    *,
    local_padded_tokens: int,
    pcp_size: int,
    pcp_rank: int | jax.Array,
    interleave_size: int,
) -> _PCPQueryRowMapping:
    """Build one source rank's batch-flat query-row coordinates."""
    q_lens = jnp.asarray(q_lens, dtype=jnp.int32)
    token_owner_starts = jnp.asarray(token_owner_starts, dtype=jnp.int32)
    request_absolute_starts = jnp.asarray(request_absolute_starts, dtype=jnp.int32)
    pcp_rank = jnp.asarray(pcp_rank, dtype=jnp.int32)

    token_owner_ends = token_owner_starts + q_lens
    local_counts = _pcp_rank_token_count_before(
        token_owner_ends, pcp_rank, pcp_size, interleave_size
    ) - _pcp_rank_token_count_before(
        token_owner_starts, pcp_rank, pcp_size, interleave_size
    )
    local_counts = jnp.where(q_lens > 0, local_counts, 0)
    req_ends = jnp.cumsum(local_counts)
    total_local_tokens = req_ends[-1]

    local_token_indices = jnp.arange(local_padded_tokens, dtype=jnp.int32)
    token_req_indices = jnp.sum(
        local_token_indices[:, None] >= req_ends[None, :],
        axis=1,
        dtype=jnp.int32,
    )
    row_valid = local_token_indices < total_local_tokens
    max_num_reqs = int(q_lens.shape[0])
    safe_req_indices = jnp.minimum(token_req_indices, max_num_reqs - 1)
    req_local_starts = req_ends[safe_req_indices] - local_counts[safe_req_indices]
    req_rank_offsets = local_token_indices - req_local_starts

    cycle = pcp_size * interleave_size
    req_owner_starts = token_owner_starts[safe_req_indices]
    req_owner_ends = token_owner_ends[safe_req_indices]
    rank_chunk_offset = pcp_rank * interleave_size
    first_chunk = (req_owner_starts // cycle) * cycle + rank_chunk_offset
    first_chunk = jnp.where(
        first_chunk + interleave_size <= req_owner_starts,
        first_chunk + cycle,
        first_chunk,
    )
    first_overlap_start = jnp.maximum(first_chunk, req_owner_starts)
    first_overlap_end = jnp.minimum(first_chunk + interleave_size, req_owner_ends)
    first_overlap_len = jnp.maximum(first_overlap_end - first_overlap_start, 0)

    after_first = req_rank_offsets - first_overlap_len
    positions_in_first = first_overlap_start + req_rank_offsets
    remaining = jnp.maximum(after_first, 0)
    positions_after_first = (
        first_chunk
        + cycle
        + (remaining // interleave_size) * cycle
        + (remaining % interleave_size)
    )
    owner_positions = jnp.where(
        req_rank_offsets < first_overlap_len, positions_in_first, positions_after_first
    )
    request_offsets = owner_positions - req_owner_starts
    absolute_positions = request_absolute_starts[safe_req_indices] + request_offsets
    absolute_positions = jnp.where(row_valid, absolute_positions, -1).astype(jnp.int32)
    request_indices = jnp.where(row_valid, safe_req_indices, -1).astype(jnp.int32)
    return _PCPQueryRowMapping(
        absolute_positions=absolute_positions,
        request_indices=request_indices,
        source_ranks=jnp.full((local_padded_tokens,), pcp_rank, dtype=jnp.int32),
        is_valid=row_valid,
    )


def _compute_pcp_local_query_row_mapping_from_metadata(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    local_padded_tokens: int,
    pcp_size: int,
    pcp_rank: int | jax.Array,
    interleave_size: int,
) -> _PCPQueryRowMapping:
    """Rebuild local query-row coordinates from PCP attention metadata."""
    if local_padded_tokens <= 0:
        raise ValueError("local_padded_tokens must be positive.")
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive.")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive.")

    kv_lens = jnp.asarray(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(cu_q_lens, dtype=jnp.int32)
    distribution = jnp.asarray(distribution, dtype=jnp.int32)
    max_num_reqs = int(kv_lens.shape[0])

    q_lens_raw = cu_q_lens[1 : max_num_reqs + 1] - cu_q_lens[:max_num_reqs]
    active_num_reqs = jnp.clip(distribution[2], 0, max_num_reqs)
    req_indices = jnp.arange(max_num_reqs, dtype=jnp.int32)
    active_req_mask = jnp.logical_and(req_indices < active_num_reqs, q_lens_raw > 0)
    q_lens = jnp.where(active_req_mask, q_lens_raw, 0)
    absolute_q_starts = jnp.where(active_req_mask, kv_lens - q_lens, 0)
    # The host packs all request-major query rows as one batch-flat stream, so
    # query_start_loc[:-1] is the token-owner coordinate. Absolute positions
    # remain derived from seq_len - q_len and alone determine cache ownership.
    token_owner_starts = jnp.where(active_req_mask, cu_q_lens[:max_num_reqs], 0)
    return _compute_pcp_local_query_row_mapping(
        q_lens,
        token_owner_starts=token_owner_starts,
        request_absolute_starts=absolute_q_starts,
        local_padded_tokens=local_padded_tokens,
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
        interleave_size=interleave_size,
    )


def _compute_pcp_rank_major_query_row_mapping_from_metadata(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    local_padded_tokens: int,
    pcp_size: int,
    interleave_size: int,
) -> _PCPQueryRowMapping:
    """Return query-row mappings shaped like the global sharded Q axis."""
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)
    mapping_by_rank = jax.vmap(
        lambda pcp_rank: _compute_pcp_local_query_row_mapping_from_metadata(
            kv_lens,
            cu_q_lens,
            distribution,
            local_padded_tokens=local_padded_tokens,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
        )
    )(ranks)
    rank_major_tokens = pcp_size * local_padded_tokens
    return _PCPQueryRowMapping(
        absolute_positions=mapping_by_rank.absolute_positions.reshape(
            (rank_major_tokens,)
        ),
        request_indices=mapping_by_rank.request_indices.reshape((rank_major_tokens,)),
        source_ranks=mapping_by_rank.source_ranks.reshape((rank_major_tokens,)),
        is_valid=mapping_by_rank.is_valid.reshape((rank_major_tokens,)),
    )


def _compute_pcp_kv_writeback_mapping(
    rank_major_query_rows: _PCPQueryRowMapping,
    block_tables: jax.Array,
    *,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
) -> _PCPKVWritebackMapping:
    """Build the one-path dense ring-capture destination mapping.

    Capture order is request-major and request-absolute within each cache
    owner. Every valid source row has exactly one cache owner, one cache-local
    slot, and one dense offset in that owner's capture buffer. There is no
    phase or fallback classification in this mapping.
    """
    block_tables = jnp.asarray(block_tables, dtype=jnp.int32)
    cache_owner_ranks, cache_local_slot_ids = (
        _compute_cache_owner_and_slot_ids_for_rows(
            rank_major_query_rows.absolute_positions,
            rank_major_query_rows.request_indices,
            rank_major_query_rows.is_valid,
            block_tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
    )

    max_num_reqs = int(block_tables.shape[0])
    invalid_scatter_index = max_num_reqs
    safe_request_indices = jnp.where(
        rank_major_query_rows.is_valid,
        rank_major_query_rows.request_indices,
        invalid_scatter_index,
    )
    safe_cache_owners = jnp.where(
        rank_major_query_rows.is_valid,
        cache_owner_ranks,
        pcp_size,
    )
    request_starts = jnp.full(
        (max_num_reqs,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32
    )
    request_starts = request_starts.at[safe_request_indices].min(
        rank_major_query_rows.absolute_positions,
        mode="drop",
    )

    request_owner_counts = jnp.zeros((max_num_reqs, pcp_size), dtype=jnp.int32)
    request_owner_counts = request_owner_counts.at[
        safe_request_indices, safe_cache_owners
    ].add(
        rank_major_query_rows.is_valid.astype(jnp.int32),
        mode="drop",
    )
    request_owner_prefixes = (
        jnp.cumsum(request_owner_counts, axis=0) - request_owner_counts
    )

    row_request_starts = request_starts[
        jnp.minimum(safe_request_indices, max_num_reqs - 1)
    ]
    row_cache_owners = jnp.minimum(safe_cache_owners, pcp_size - 1)
    offsets_within_request = _pcp_rank_token_count_before(
        rank_major_query_rows.absolute_positions,
        row_cache_owners,
        pcp_size,
        interleave_size,
    ) - _pcp_rank_token_count_before(
        row_request_starts,
        row_cache_owners,
        pcp_size,
        interleave_size,
    )
    capture_offsets = (
        request_owner_prefixes[
            jnp.minimum(safe_request_indices, max_num_reqs - 1),
            row_cache_owners,
        ]
        + offsets_within_request
    )
    capture_offsets = jnp.where(
        rank_major_query_rows.is_valid, capture_offsets, -1
    ).astype(jnp.int32)
    writeback_counts = jnp.sum(request_owner_counts, axis=0, dtype=jnp.int32)
    return _PCPKVWritebackMapping(
        cache_owner_ranks=cache_owner_ranks,
        cache_local_slot_ids=cache_local_slot_ids,
        capture_offsets=capture_offsets,
        writeback_counts=writeback_counts,
    )


def _compact_writeback_slot_ids_for_cache_rank(
    writeback_mapping: _PCPKVWritebackMapping,
    cache_rank: int | jax.Array,
) -> jax.Array:
    """Compact cache slots into the same dense order as captured KV rows."""
    capacity = writeback_mapping.capture_offsets.shape[0]
    cache_rank = jnp.asarray(cache_rank, dtype=jnp.int32)
    owned = writeback_mapping.cache_owner_ranks == cache_rank
    offsets = jnp.where(owned, writeback_mapping.capture_offsets, capacity)
    slots = jnp.where(owned, writeback_mapping.cache_local_slot_ids, -1)
    compact_slots = jnp.full((capacity,), -1, dtype=jnp.int32)
    return compact_slots.at[offsets].set(slots, mode="drop")


def _build_writeback_segment_descriptors(
    compact_slot_ids: jax.Array,
    writeback_count: jax.Array,
    max_segments: int = 4096,
    *,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Coalesce dense captured rows into contiguous cache-copy segments."""
    if kv_layout == KVLayout.SEQ_ALONG_LANE and (page_size is None or page_size <= 0):
        raise ValueError("SEQ_ALONG_LANE writeback descriptors require page_size > 0.")
    capacity = int(compact_slot_ids.shape[0])
    num_descriptor_slots = min(capacity, max_segments)
    rows = jnp.arange(capacity, dtype=jnp.int32)
    writeback_count = jnp.clip(
        jnp.asarray(writeback_count, dtype=jnp.int32), 0, capacity
    )
    active = jnp.logical_and(rows < writeback_count, compact_slot_ids >= 0)
    previous_active = jnp.concatenate((jnp.asarray([False]), active[:-1]), axis=0)
    previous_slots = jnp.concatenate(
        (jnp.asarray([-2], dtype=jnp.int32), compact_slot_ids[:-1]), axis=0
    )
    discontinuous = compact_slot_ids != previous_slots + 1
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        crossed_page = (compact_slot_ids // page_size) != (previous_slots // page_size)
        discontinuous = jnp.logical_or(discontinuous, crossed_page)
    segment_starts = jnp.logical_and(
        active,
        jnp.logical_or(jnp.logical_not(previous_active), discontinuous),
    )
    segment_ids = jnp.cumsum(segment_starts.astype(jnp.int32)) - 1
    num_segments = jnp.sum(segment_starts, dtype=jnp.int32)
    drop_index = jnp.asarray(num_descriptor_slots, dtype=jnp.int32)

    start_indices = jnp.where(
        jnp.logical_and(segment_starts, segment_ids < num_descriptor_slots),
        segment_ids,
        drop_index,
    )
    source_starts = jnp.zeros((num_descriptor_slots,), dtype=jnp.int32)
    source_starts = source_starts.at[start_indices].set(rows, mode="drop")
    destination_starts = jnp.zeros((num_descriptor_slots,), dtype=jnp.int32)
    destination_starts = destination_starts.at[start_indices].set(
        compact_slot_ids, mode="drop"
    )

    active_segment_ids = jnp.where(
        jnp.logical_and(active, segment_ids < num_descriptor_slots),
        segment_ids,
        drop_index,
    )
    lengths = jnp.zeros((num_descriptor_slots,), dtype=jnp.int32)
    lengths = lengths.at[active_segment_ids].add(active.astype(jnp.int32), mode="drop")
    descriptors = jnp.stack((source_starts, destination_starts, lengths), axis=0)
    return descriptors, num_segments


def _select_replicated_query_row_mapping_for_pcp_rank(
    mapping: _PCPQueryRowMapping,
    pcp_axis: str,
    pcp_size: int,
) -> _PCPQueryRowMapping:
    """Select a PCP rank's query rows with one collective selection."""
    packed_mapping = jnp.stack(
        (
            mapping.absolute_positions,
            mapping.request_indices,
            mapping.source_ranks,
            mapping.is_valid.astype(jnp.int32),
        ),
        axis=1,
    )
    local_mapping = _select_replicated_shard_for_pcp_rank(
        packed_mapping,
        pcp_axis,
        pcp_size,
        axis=0,
    )
    return _PCPQueryRowMapping(
        absolute_positions=local_mapping[:, 0],
        request_indices=local_mapping[:, 1],
        source_ranks=local_mapping[:, 2],
        is_valid=local_mapping[:, 3].astype(jnp.bool_),
    )


def sharded_pcp_ragged_paged_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None,
    sm_scale: float,
    attention_chunk_size: int | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    update_kv_cache: bool = True,
    cp_kv_cache_interleave_size: int = 0,
    q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    q_compute_size: int | None = None,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
):
    """Runs streaming PCP RPA over local Q/K/V shards."""
    if attention_sink is not None:
        raise NotImplementedError("PCP RPA does not support attention sinks.")
    if q.shape[-1] == 64:
        raise NotImplementedError("PCP RPA does not support head_dim==64.")
    if cp_kv_cache_interleave_size <= 0:
        raise ValueError("PCP RPA requires cp_kv_cache_interleave_size > 0.")
    if PCP_AXIS_NAME not in mesh.axis_names:
        raise NotImplementedError("PCP requires a mesh with a pcp axis.")
    pcp_size = mesh.shape[PCP_AXIS_NAME]
    if attention_chunk_size is not None:
        raise NotImplementedError("PCP streaming RPA supports full attention only.")
    if q_scale is not None:
        raise NotImplementedError(
            "PCP streaming RPA does not support quantized Q scales."
        )
    if (k_scale is None) != (v_scale is None):
        raise ValueError(
            "PCP streaming RPA requires k_scale and v_scale to be both "
            "None or both non-None."
        )
    if not update_kv_cache:
        raise NotImplementedError("PCP streaming RPA requires update_kv_cache=True.")
    page_size = _kv_cache_page_size(kv_cache, kv_layout)
    if kv_layout == KVLayout.SEQ_ALONG_LANE and (
        page_size % 128 != 0 or cp_kv_cache_interleave_size % 128 != 0
    ):
        raise NotImplementedError(
            "PCP SEQ_ALONG_LANE UT path requires page_size and "
            "cp_kv_cache_interleave_size to be multiples of 128, got "
            f"{page_size=} {cp_kv_cache_interleave_size=}."
        )
    if (
        cp_kv_cache_interleave_size > page_size
        or page_size % cp_kv_cache_interleave_size != 0
    ):
        raise NotImplementedError(
            "PCP streaming RPA requires page_size to be divisible by "
            "cp_kv_cache_interleave_size: "
            f"page_size={page_size} "
            f"{cp_kv_cache_interleave_size=}."
        )
    if q.shape[0] % pcp_size != 0:
        raise ValueError(
            "PCP streaming RPA requires q tokens to be evenly "
            f"sharded across PCP ranks: {q.shape[0]=} "
            f"{pcp_size=}."
        )
    if kv_cache.shape[0] % pcp_size != 0:
        raise ValueError(
            "PCP streaming RPA requires KV cache blocks to be "
            f"evenly sharded across PCP ranks: "
            f"{kv_cache.shape[0]=} {pcp_size=}."
        )
    qkv_spec = P(PCP_AXIS_NAME)
    metadata_spec = P()
    in_specs = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
        metadata_spec,
    )
    out_specs = (qkv_spec, qkv_spec)
    args = (
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    def _pcp_ragged_paged_attention(
        q_local,
        k_local,
        v_local,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        rank_major_query_rows = _compute_pcp_rank_major_query_row_mapping_from_metadata(
            kv_lens,
            cu_q_lens,
            distribution,
            local_padded_tokens=q_local.shape[0],
            pcp_size=pcp_size,
            interleave_size=cp_kv_cache_interleave_size,
        )
        local_query_rows = _select_replicated_query_row_mapping_for_pcp_rank(
            rank_major_query_rows,
            PCP_AXIS_NAME,
            pcp_size,
        )
        query_row_valid = local_query_rows.is_valid
        cache_rank = local_query_rows.source_ranks[0]
        block_tables = _reshape_metadata_page_indices_jax(
            page_indices, kv_lens.shape[0]
        )
        writeback_mapping = _compute_pcp_kv_writeback_mapping(
            rank_major_query_rows,
            block_tables,
            local_kv_cache_num_blocks=kv_cache.shape[0],
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=cp_kv_cache_interleave_size,
        )
        compact_slot_ids = _compact_writeback_slot_ids_for_cache_rank(
            writeback_mapping, cache_rank
        )
        writeback_count = writeback_mapping.writeback_counts[cache_rank]
        packed_current_kv = _prepare_packed_kv_for_cache(
            k_local, v_local, kv_cache, kv_layout=kv_layout
        )
        if q_local.shape[1] % k_local.shape[1] != 0:
            raise ValueError("Q heads must be divisible by KV heads.")
        q_per_kv = q_local.shape[1] // k_local.shape[1]
        q_streaming = q_local.reshape(
            q_local.shape[0], k_local.shape[1], q_per_kv, q_local.shape[2]
        )
        q_packing = get_dtype_packing(q_local.dtype)
        q_per_kv_padded = math.ceil(q_per_kv / q_packing) * q_packing
        if q_per_kv_padded != q_per_kv:
            q_streaming = jnp.pad(
                q_streaming,
                (
                    (0, 0),
                    (0, 0),
                    (0, q_per_kv_padded - q_per_kv),
                    (0, 0),
                ),
                constant_values=0,
            )
        attention_kv_cache = _reshape_packed_kv_cache_for_attention(kv_cache, kv_layout)
        segment_descriptors, num_segments = _build_writeback_segment_descriptors(
            compact_slot_ids,
            writeback_count,
            kv_layout=kv_layout,
            page_size=page_size,
        )
        output, kv_cache = (
            pcp_streaming_attention_page_groups_packed_local_from_metadata(
                q_streaming,
                packed_current_kv,
                attention_kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                segment_descriptors,
                num_segments,
                pcp_size=pcp_size,
                interleave_size=cp_kv_cache_interleave_size,
                sm_scale=sm_scale,
                q_block_size=q_block_size,
                q_compute_size=q_compute_size,
                k_scale=k_scale,
                v_scale=v_scale,
                kv_layout=kv_layout,
                mesh_axis_names=tuple(mesh.axis_names),
                pcp_axis_name=PCP_AXIS_NAME,
            )
        )
        output = output[:, :, :q_per_kv, :]
        valid_output_rows = query_row_valid.reshape(
            (query_row_valid.shape[0],) + (1,) * (output.ndim - 1)
        )
        output = jnp.where(valid_output_rows, output, jnp.zeros_like(output))
        output = output.reshape(q_local.shape)
        return output, kv_cache

    return jax.shard_map(
        _pcp_ragged_paged_attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)
