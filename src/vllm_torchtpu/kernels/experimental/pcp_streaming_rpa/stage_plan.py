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
"""Stage-entry metadata plan for PCP streaming RPA."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    _reshape_metadata_page_indices_jax,
)

PCP_AXIS_NAME = "pcp"
PCP_STREAMING_RPA_VPAGE_GROUP_SIZE = 128
PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS = (32, 64, 96, 128, 256, 512)

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


class PCPStagePlan(NamedTuple):
    """Precomputed per-PCP-rank stage plan shared across all layers in a stage.

    The sorted_* arrays hold one entry per block-table slot, active vpages
    first, padded with inactive entries to a whole number of vpage groups.
    """

    segment_descriptors: jax.Array
    num_segments: jax.Array
    sorted_block_ids: jax.Array
    sorted_seg_ids: jax.Array
    sorted_vpage_valid_tokens: jax.Array
    num_active_vpages: jax.Array
    q_seg_ids: jax.Array
    q_sequence: jax.Array
    query_row_valid: jax.Array
    needs_reroute: jax.Array
    max_writeback_count: jax.Array | int = 0
    block_tables: jax.Array | None = None


def get_pcp_rank_without_axis_index(pcp_axis_name: str, pcp_size: int) -> jax.Array:
    """Select this PCP rank index without emitting axis_index / partitionId."""
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)
    exchanged = lax.all_to_all(
        ranks,
        axis_name=pcp_axis_name,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )
    return exchanged[0]


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
    position: jax.Array,
    pcp_rank: jax.Array,
    pcp_size: int,
    interleave_size: int,
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
        req_rank_offsets < first_overlap_len,
        positions_in_first,
        positions_after_first,
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
    """Build the one-path dense ring-capture destination mapping."""
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
    *,
    capacity: int | None = None,
    window_offset: int | jax.Array = 0,
) -> jax.Array:
    """Compact cache slots into the same dense order as captured KV rows."""
    if capacity is None:
        capacity = int(writeback_mapping.capture_offsets.shape[0])
    cache_rank = jnp.asarray(cache_rank, dtype=jnp.int32)
    rel_offsets = writeback_mapping.capture_offsets - jnp.asarray(
        window_offset, dtype=jnp.int32
    )
    in_window = jnp.logical_and(
        writeback_mapping.cache_owner_ranks == cache_rank,
        jnp.logical_and(rel_offsets >= 0, rel_offsets < capacity),
    )
    offsets = jnp.where(in_window, rel_offsets, capacity)
    slots = jnp.where(in_window, writeback_mapping.cache_local_slot_ids, -1)
    compact_slots = jnp.full((capacity,), -1, dtype=jnp.int32)
    return compact_slots.at[offsets].set(slots, mode="drop")


def _compute_pcp_capture_source_rows_for_cache_rank(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    cache_rank: int | jax.Array,
    *,
    local_tokens: int,
    pcp_size: int,
    interleave_size: int,
    window_offset: int | jax.Array = 0,
) -> tuple[jax.Array, jax.Array]:
    """Rank-major source row for each capture offset owned by cache_rank.

    Uses the same capture offsets as _compute_misaligned_writeback_slots, so
    captured row i lands in compact writeback slot i.
    """
    rank_major_query_rows = _compute_pcp_rank_major_query_row_mapping_from_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        local_padded_tokens=local_tokens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    # Capture offsets ignore block tables, so one dummy block suffices.
    wb_mapping = _compute_pcp_kv_writeback_mapping(
        rank_major_query_rows,
        jnp.zeros((int(kv_lens.shape[0]), 1), dtype=jnp.int32),
        local_kv_cache_num_blocks=1,
        page_size=interleave_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    num_rows = pcp_size * local_tokens
    rel_offsets = wb_mapping.capture_offsets - jnp.asarray(
        window_offset, dtype=jnp.int32
    )
    in_window = jnp.logical_and(
        wb_mapping.cache_owner_ranks == jnp.asarray(cache_rank, dtype=jnp.int32),
        jnp.logical_and(rel_offsets >= 0, rel_offsets < local_tokens),
    )
    offsets = jnp.where(in_window, rel_offsets, local_tokens)
    rows = jnp.where(in_window, jnp.arange(num_rows, dtype=jnp.int32), -1)
    source_rows = jnp.full((local_tokens,), -1, dtype=jnp.int32)
    source_rows = source_rows.at[offsets].set(rows, mode="drop")
    return jnp.clip(source_rows, 0, num_rows - 1), source_rows >= 0


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
        assert page_size is not None
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


def _compute_aligned_writeback_slots(
    absolute_positions: jax.Array,
    request_indices: jax.Array,
    query_row_valid: jax.Array,
    block_tables: jax.Array,
    *,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Computes cache slots and count for aligned writeback."""
    _, slot_ids = _compute_cache_owner_and_slot_ids_for_rows(
        absolute_positions,
        request_indices,
        query_row_valid,
        block_tables,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    wb_count = jnp.sum(query_row_valid.astype(jnp.int32), dtype=jnp.int32)
    return slot_ids, wb_count


def _compute_misaligned_writeback_slots(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    block_tables: jax.Array,
    cache_rank: jax.Array,
    *,
    local_tokens: int,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    window_offset: int | jax.Array = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes cache slots, window count, and max count for misaligned writeback."""
    rank_major_query_rows = _compute_pcp_rank_major_query_row_mapping_from_metadata(
        kv_lens,
        cu_q_lens,
        distribution,
        local_padded_tokens=local_tokens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    wb_mapping = _compute_pcp_kv_writeback_mapping(
        rank_major_query_rows,
        block_tables,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    slot_ids = _compact_writeback_slot_ids_for_cache_rank(
        wb_mapping,
        cache_rank,
        capacity=local_tokens,
        window_offset=window_offset,
    )
    wb_count = jnp.clip(
        wb_mapping.writeback_counts[cache_rank]
        - jnp.asarray(window_offset, dtype=jnp.int32),
        0,
        local_tokens,
    )
    max_wb_count = jnp.max(wb_mapping.writeback_counts)
    return slot_ids, wb_count, max_wb_count


def build_pcp_local_stage_plan(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    local_tokens: int,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    vpage_group_size: int,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    pcp_axis_name: str = PCP_AXIS_NAME,
) -> PCPStagePlan:
    """Builds the local PCP rank's stage plan inside shard_map without all_to_all."""
    if local_tokens <= 0:
        raise ValueError("local_tokens must be positive.")
    if local_kv_cache_num_blocks <= 0:
        raise ValueError("local_kv_cache_num_blocks must be positive.")
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive.")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive.")
    if vpage_group_size <= 0:
        raise ValueError("vpage_group_size must be positive.")

    kv_lens = jnp.asarray(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(cu_q_lens, dtype=jnp.int32)
    distribution = jnp.asarray(distribution, dtype=jnp.int32)
    max_num_seqs = int(kv_lens.shape[0])
    block_tables = _reshape_metadata_page_indices_jax(
        jnp.asarray(page_indices, dtype=jnp.int32), max_num_seqs
    )
    block_tables = jnp.mod(
        block_tables, jnp.asarray(local_kv_cache_num_blocks, dtype=jnp.int32)
    )
    max_vpages_per_req = int(block_tables.shape[1])
    virtual_page_size = page_size * pcp_size

    active_reqs = jnp.clip(distribution[2], 0, max_num_seqs)
    req_ids = lax.broadcasted_iota(jnp.int32, (max_num_seqs,), 0)
    q_starts = cu_q_lens[:max_num_seqs]
    q_ends = cu_q_lens[1 : max_num_seqs + 1]
    q_len_raw = q_ends - q_starts
    req_active = jnp.logical_and(req_ids < active_reqs, q_len_raw > 0)
    q_lens = jnp.where(req_active, q_len_raw, 0)
    total_kv_lens = jnp.where(req_active, kv_lens[:max_num_seqs], 0)
    hist_tokens = jnp.maximum(total_kv_lens - q_lens, 0)

    cycle = pcp_size * interleave_size
    needs_reroute = jnp.any(
        jnp.logical_and(req_active, jnp.mod(hist_tokens - q_starts, cycle) != 0)
    )

    cache_rank = get_pcp_rank_without_axis_index(pcp_axis_name, pcp_size)
    token_owner_starts = jnp.where(req_active, q_starts, 0)
    local_query_rows = _compute_pcp_local_query_row_mapping(
        q_lens,
        token_owner_starts=token_owner_starts,
        request_absolute_starts=hist_tokens,
        local_padded_tokens=local_tokens,
        pcp_size=pcp_size,
        pcp_rank=cache_rank,
        interleave_size=interleave_size,
    )
    query_row_valid = local_query_rows.is_valid

    def _aligned_writeback(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
        slots, cnt = _compute_aligned_writeback_slots(
            local_query_rows.absolute_positions,
            local_query_rows.request_indices,
            query_row_valid,
            block_tables,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
        return slots, cnt, cnt

    compact_slot_ids, writeback_count, max_writeback_count = lax.cond(
        needs_reroute,
        lambda _: _compute_misaligned_writeback_slots(
            kv_lens,
            cu_q_lens,
            distribution,
            block_tables,
            cache_rank,
            local_tokens=local_tokens,
            local_kv_cache_num_blocks=local_kv_cache_num_blocks,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            window_offset=0,
        ),
        _aligned_writeback,
        operand=None,
    )
    segment_descriptors, num_segments = _build_writeback_segment_descriptors(
        compact_slot_ids,
        writeback_count,
        max_segments=min(local_tokens, 512),
        kv_layout=kv_layout,
        page_size=page_size,
    )

    req_vpages = jnp.where(
        req_active,
        (total_kv_lens + virtual_page_size - 1) // virtual_page_size,
        0,
    )
    req_vpage_starts = jnp.cumsum(req_vpages, axis=0) - req_vpages
    safe_q_req = jnp.clip(local_query_rows.request_indices, 0, max_num_seqs - 1)
    q_sequence = jnp.where(
        query_row_valid,
        req_vpage_starts[safe_q_req] * virtual_page_size
        + local_query_rows.absolute_positions,
        -1,
    )
    q_seg_ids = jnp.where(query_row_valid, local_query_rows.request_indices, -1)

    vpage_iota = lax.broadcasted_iota(jnp.int32, (max_num_seqs, max_vpages_per_req), 1)
    vpage_valid = jnp.logical_and(req_active[:, None], vpage_iota < req_vpages[:, None])
    vpage_seg = jnp.where(
        vpage_valid,
        jnp.broadcast_to(req_ids[:, None], (max_num_seqs, max_vpages_per_req)),
        -2,
    )
    flat_active = vpage_valid.reshape(-1)
    flat_block_ids = jnp.clip(
        block_tables.reshape(-1), 0, local_kv_cache_num_blocks - 1
    )
    flat_seg_ids = vpage_seg.reshape(-1)
    flat_vpage_iota = vpage_iota.reshape(-1)

    _, sorted_block_ids, sorted_seg_ids, sorted_vpage_iota = lax.sort(
        (
            (~flat_active).astype(jnp.int32),
            flat_block_ids,
            flat_seg_ids,
            flat_vpage_iota,
        ),
        num_keys=1,
        is_stable=True,
    )
    sorted_block_ids = jnp.where(sorted_seg_ids >= 0, sorted_block_ids, 0)
    sorted_kv_lens = total_kv_lens[jnp.clip(sorted_seg_ids, 0, max_num_seqs - 1)]
    sorted_vpage_valid_tokens = jnp.where(
        sorted_seg_ids >= 0,
        jnp.clip(
            sorted_kv_lens - sorted_vpage_iota * virtual_page_size,
            0,
            virtual_page_size,
        ),
        0,
    )
    # Inactive padding up to whole vpage groups keeps every group slice in bounds.
    pad_unit = (
        max(vpage_group_size, max(PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS))
        if vpage_group_size == PCP_STREAMING_RPA_VPAGE_GROUP_SIZE
        else vpage_group_size
    )
    group_pad = (0, (-flat_active.shape[0]) % pad_unit)
    sorted_block_ids = jnp.pad(sorted_block_ids, group_pad)
    sorted_seg_ids = jnp.pad(sorted_seg_ids, group_pad, constant_values=-2)
    sorted_vpage_valid_tokens = jnp.pad(sorted_vpage_valid_tokens, group_pad)

    return PCPStagePlan(
        segment_descriptors=segment_descriptors,
        num_segments=num_segments,
        sorted_block_ids=sorted_block_ids,
        sorted_seg_ids=sorted_seg_ids,
        sorted_vpage_valid_tokens=sorted_vpage_valid_tokens,
        num_active_vpages=jnp.sum(flat_active.astype(jnp.int32)),
        q_seg_ids=q_seg_ids,
        q_sequence=q_sequence,
        query_row_valid=query_row_valid,
        needs_reroute=needs_reroute,
        max_writeback_count=max_writeback_count,
        block_tables=block_tables,
    )
