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
"""Runtime schedules for PCP streaming prefill RPA."""

import numpy as np


class RuntimeScheduleField:
    """Minimal per-ring-step ABI consumed by the production Pallas kernels."""

    REQ_ID = 0
    KV_PAGE_IDX = 1
    Q_GLOBAL_START = 2
    KV_GLOBAL_START = 3
    KV_VALID_LEN = 4
    Q_HBM_OFFSET = 5
    Q_TILE_SIZE = 6
    KV_HBM_OFFSET = 7
    CAPTURE_SRC_OFFSET = 8
    CAPTURE_DST_OFFSET = 9
    CAPTURE_LEN = 10
    NUM_FIELDS = 11
    PACKED_NUM_FIELDS = 128


class TilePlanField:
    """Fields stored once per query tile and PCP consumer rank."""

    REQ_ID = 0
    Q_GLOBAL_START = 1
    Q_HBM_OFFSET = 2
    Q_TILE_SIZE = 3
    TOKEN_OWNER_START = 4
    REQUEST_ABSOLUTE_QUERY_START = 5
    CURRENT_EFFECTIVE_LEN = 6
    CURRENT_KV_HBM_START = 7
    WRITEBACK_HBM_PREFIX = 8
    CAPTURE_CURRENT_KV = 9
    CURRENT_NUM_GROUPS = 10
    HISTORY_NUM_GROUPS = 11
    NUM_FIELDS = 12
    HBM_ROW_FIELDS = 128


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _validate_page_interleave_size(page_size: int, interleave_size: int,
                                   context: str) -> None:
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive.")
    if interleave_size > page_size or page_size % interleave_size != 0:
        raise NotImplementedError(
            f"{context} requires page_size to be divisible by "
            f"interleave_size, got {page_size=} {interleave_size=}.")


def _reshape_metadata_page_indices_jax(page_indices, max_num_seqs: int):
    import jax.numpy as jnp

    if max_num_seqs <= 0:
        raise ValueError("metadata must contain at least one sequence slot.")
    if page_indices.ndim == 1:
        if page_indices.shape[0] % max_num_seqs != 0:
            raise ValueError("flat page_indices size must be divisible by "
                             "kv_lens.shape[0].")
        pages_per_seq = page_indices.shape[0] // max_num_seqs
        return jnp.reshape(page_indices, (max_num_seqs, pages_per_seq))
    if page_indices.ndim == 2:
        if page_indices.shape[0] != max_num_seqs:
            raise ValueError("2D page_indices first dimension must match "
                             "kv_lens.shape[0].")
        return page_indices
    raise ValueError("page_indices must be rank 1 or rank 2.")


def build_pcp_streaming_schedule_inputs_from_metadata_jax(
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    global_bucket_tokens: int,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    q_block_size: int,
    num_lanes: int = 1,
):
    """Build the single production current/history tile plan with JAX ops.

    Query and fresh-KV compute ownership follows the batch-flat ``cu_q_lens``
    coordinate. Causality, history, capture ownership, and cache destinations
    remain request-absolute. The result is always the compact tile plan consumed
    by the production Pallas kernels; there is no dense schedule fallback.
    """
    import jax.numpy as jnp

    global_bucket_tokens = int(global_bucket_tokens)
    local_kv_cache_num_blocks = int(local_kv_cache_num_blocks)
    page_size = int(page_size)
    pcp_size = int(pcp_size)
    interleave_size = int(interleave_size)
    q_block_size = int(q_block_size)
    num_lanes = int(num_lanes)

    if num_lanes != 1:
        raise NotImplementedError(
            "PCP streaming JAX metadata schedule only supports num_lanes == 1."
        )
    _validate_page_interleave_size(page_size, interleave_size,
                                   "PCP streaming JAX metadata schedule")
    if q_block_size % interleave_size != 0:
        raise NotImplementedError(
            "PCP streaming JAX metadata schedule requires q_block_size to be "
            "a multiple of interleave_size.")
    if global_bucket_tokens <= 0 or global_bucket_tokens % pcp_size != 0:
        raise ValueError("global_bucket_tokens must be positive and divisible "
                         "by pcp_size.")
    if global_bucket_tokens % (pcp_size * q_block_size) != 0:
        raise ValueError("global_bucket_tokens must be divisible by "
                         "pcp_size * q_block_size for PCP lockstep schedules.")
    if local_kv_cache_num_blocks <= 0:
        raise ValueError("local_kv_cache_num_blocks must be positive.")

    max_num_seqs = int(np.shape(kv_lens)[0])

    kv_lens = jnp.asarray(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(cu_q_lens, dtype=jnp.int32)
    page_indices = jnp.asarray(page_indices, dtype=jnp.int32)
    distribution = jnp.asarray(distribution, dtype=jnp.int32)
    block_tables = _reshape_metadata_page_indices_jax(page_indices,
                                                      max_num_seqs)
    block_tables = jnp.mod(
        block_tables,
        jnp.asarray(local_kv_cache_num_blocks, dtype=jnp.int32),
    )
    active_num_reqs = jnp.clip(distribution[2], 0, max_num_seqs)
    local_bucket_tokens = global_bucket_tokens // pcp_size
    max_q_tiles = _cdiv(local_bucket_tokens, q_block_size) + 2

    page_size_i32 = jnp.asarray(page_size, dtype=jnp.int32)
    pcp_size_i32 = jnp.asarray(pcp_size, dtype=jnp.int32)
    interleave_size_i32 = jnp.asarray(interleave_size, dtype=jnp.int32)
    cycle_i32 = pcp_size_i32 * interleave_size_i32
    virtual_page_size_i32 = page_size_i32 * pcp_size_i32
    chunks_per_page_i32 = page_size_i32 // interleave_size_i32
    group_chunks = max(1, q_block_size // interleave_size)
    group_chunks_i32 = jnp.asarray(group_chunks, dtype=jnp.int32)

    def _local_page_global_start(local_page_idx, src_rank):
        return (local_page_idx * virtual_page_size_i32 +
                src_rank * interleave_size_i32)

    def _local_page_valid_len(kv_len, local_page_idx, src_rank):
        base = _local_page_global_start(local_page_idx, src_rank)
        delta = kv_len - base
        full_chunks = jnp.minimum(
            chunks_per_page_i32,
            jnp.maximum(delta, 0) // cycle_i32,
        )
        partial = jnp.minimum(
            interleave_size_i32,
            jnp.maximum(delta - full_chunks * cycle_i32, 0),
        )
        valid_len = full_chunks * interleave_size_i32 + jnp.where(
            full_chunks < chunks_per_page_i32, partial, 0)
        return jnp.where(kv_len > base, valid_len, 0)

    def _rank_request_layout(q_len, token_owner_start, consumer_rank: int):
        rank_chunk_offset = jnp.asarray(consumer_rank * interleave_size,
                                        dtype=jnp.int32)
        token_owner_end = token_owner_start + q_len
        first_chunk = ((token_owner_start // cycle_i32) * cycle_i32 +
                       rank_chunk_offset)
        first_chunk = jnp.where(
            first_chunk + interleave_size_i32 <= token_owner_start,
            first_chunk + cycle_i32, first_chunk)
        has_chunks = jnp.logical_and(q_len > 0, first_chunk < token_owner_end)

        safe_owner_end_minus_one = jnp.where(q_len > 0, token_owner_end - 1,
                                             token_owner_start)
        last_chunk = ((safe_owner_end_minus_one // cycle_i32) * cycle_i32 +
                      rank_chunk_offset)
        last_chunk = jnp.where(last_chunk > safe_owner_end_minus_one,
                               last_chunk - cycle_i32, last_chunk)
        num_chunks = jnp.where(
            has_chunks,
            (last_chunk - first_chunk) // cycle_i32 + 1,
            0,
        )

        first_overlap_start = jnp.maximum(first_chunk, token_owner_start)
        first_overlap_end = jnp.minimum(first_chunk + interleave_size_i32,
                                        token_owner_end)
        first_overlap_len = jnp.where(
            has_chunks,
            jnp.maximum(first_overlap_end - first_overlap_start, 0),
            0,
        )
        last_overlap_start = jnp.maximum(last_chunk, token_owner_start)
        last_overlap_end = jnp.minimum(last_chunk + interleave_size_i32,
                                       token_owner_end)
        last_overlap_len = jnp.where(
            has_chunks,
            jnp.maximum(last_overlap_end - last_overlap_start, 0),
            0,
        )

        first_partial = jnp.logical_and(
            has_chunks, first_overlap_len < interleave_size_i32)
        last_partial = jnp.logical_and(
            jnp.logical_and(has_chunks, num_chunks > 1),
            last_overlap_len < interleave_size_i32,
        )
        first_partial_i32 = first_partial.astype(jnp.int32)
        last_partial_i32 = last_partial.astype(jnp.int32)
        full_chunk_count = (num_chunks - first_partial_i32 - last_partial_i32)
        num_full_tiles = ((full_chunk_count + group_chunks_i32 - 1) //
                          group_chunks_i32)
        first_full_chunk = first_chunk + jnp.where(first_partial, cycle_i32, 0)
        first_partial_len = jnp.where(first_partial, first_overlap_len, 0)
        local_count = (first_partial_len +
                       full_chunk_count * interleave_size_i32 +
                       jnp.where(last_partial, last_overlap_len, 0))
        num_tiles = first_partial_i32 + num_full_tiles + last_partial_i32
        return (
            has_chunks,
            first_overlap_start,
            first_overlap_len,
            first_partial,
            first_partial_i32,
            first_partial_len,
            first_full_chunk,
            full_chunk_count,
            num_full_tiles,
            last_chunk,
            last_overlap_len,
            last_partial,
            num_tiles,
            local_count,
        )

    def _rank_tile(q_len, token_owner_start, request_absolute_query_start,
                   rank_local_prefix, consumer_rank: int, tile_idx):
        (
            has_chunks,
            first_overlap_start,
            first_overlap_len,
            first_partial,
            first_partial_i32,
            first_partial_len,
            first_full_chunk,
            full_chunk_count,
            num_full_tiles,
            last_chunk,
            last_overlap_len,
            last_partial,
            num_tiles,
            _local_count,
        ) = _rank_request_layout(q_len, token_owner_start, consumer_rank)

        tile_idx_i32 = jnp.asarray(tile_idx, dtype=jnp.int32)
        full_tile_idx = tile_idx_i32 - first_partial_i32
        is_first_partial_tile = jnp.logical_and(first_partial,
                                                tile_idx_i32 == 0)
        is_full_tile = jnp.logical_and(full_tile_idx >= 0, full_tile_idx
                                       < num_full_tiles)
        is_last_partial_tile = jnp.logical_and(
            last_partial,
            tile_idx_i32 == first_partial_i32 + num_full_tiles,
        )
        remaining_full_chunks = full_chunk_count - full_tile_idx * group_chunks_i32
        full_chunks_in_tile = jnp.minimum(
            group_chunks_i32, jnp.maximum(remaining_full_chunks, 0))
        full_tile_len = full_chunks_in_tile * interleave_size_i32

        full_tile_global_start = (first_full_chunk +
                                  full_tile_idx * group_chunks_i32 * cycle_i32)
        full_tile_local_offset = (
            first_partial_len +
            full_tile_idx * group_chunks_i32 * interleave_size_i32)
        last_partial_local_offset = (first_partial_len +
                                     full_chunk_count * interleave_size_i32)

        tile_active = jnp.logical_and(has_chunks, tile_idx_i32 < num_tiles)
        token_owner_tile_start = jnp.where(
            is_first_partial_tile,
            first_overlap_start,
            jnp.where(is_full_tile, full_tile_global_start,
                      jnp.where(is_last_partial_tile, last_chunk, 0)),
        )
        q_tile_size = jnp.where(
            is_first_partial_tile,
            first_overlap_len,
            jnp.where(is_full_tile, full_tile_len,
                      jnp.where(is_last_partial_tile, last_overlap_len, 0)),
        )
        local_offset = jnp.where(
            is_first_partial_tile,
            0,
            jnp.where(
                is_full_tile, full_tile_local_offset,
                jnp.where(is_last_partial_tile, last_partial_local_offset, 0)),
        )
        q_tile_size = jnp.where(tile_active, q_tile_size, 0)
        q_hbm_offset = jnp.where(tile_active, rank_local_prefix + local_offset,
                                 0)
        request_relative_start = token_owner_tile_start - token_owner_start
        q_global_start = jnp.where(
            tile_active,
            request_absolute_query_start + request_relative_start,
            0,
        )
        return q_global_start, q_tile_size, q_hbm_offset

    def _rank_token_count_before(position, rank):
        full_cycles = position // cycle_i32
        cycle_offset = position - full_cycles * cycle_i32
        rank_start = rank * interleave_size_i32
        partial = jnp.clip(cycle_offset - rank_start, 0, interleave_size_i32)
        return full_cycles * interleave_size_i32 + partial

    # Construct the fixed-capacity request/tile/rank plan as one vectorized
    # program.  The previous request x tile x rank Python loops emitted one
    # copy of the scheduling arithmetic for every static slot (64 x 10 x 8 for
    # the production Qwen shape), inflating StableHLO by hundreds of thousands
    # of ops even though the computation is elementwise.  The flattened order
    # below remains request-major, tile-major, rank-minor, so the Pallas ABI and
    # runtime schedule are unchanged.
    request_indices = jnp.arange(max_num_seqs, dtype=jnp.int32)
    tile_indices = jnp.arange(max_q_tiles, dtype=jnp.int32)
    rank_indices = jnp.arange(pcp_size, dtype=jnp.int32)

    q_len_raw = cu_q_lens[1:] - cu_q_lens[:-1]
    req_active = jnp.logical_and(request_indices < active_num_reqs, q_len_raw
                                 > 0)
    q_len = jnp.where(req_active, q_len_raw, 0)
    kv_len = jnp.where(req_active, kv_lens, 0)
    request_absolute_query_start = jnp.where(req_active, kv_len - q_len, 0)
    token_owner_start = cu_q_lens[:-1]
    request_absolute_query_end = request_absolute_query_start + q_len

    # Prefixes are the exclusive cumulative counts of all earlier requests.
    # Computing every request's count first removes the only apparent loop
    # carried state from the original implementation.
    request_q_len = q_len[:, None]
    request_token_owner_start = token_owner_start[:, None]
    request_ranks = rank_indices[None, :]
    *_, local_counts = _rank_request_layout(
        request_q_len,
        request_token_owner_start,
        request_ranks,
    )
    local_counts = jnp.where(req_active[:, None], local_counts, 0)
    rank_local_q_ends = jnp.cumsum(local_counts, axis=0)
    request_rank_local_q_starts = rank_local_q_ends - local_counts

    writeback_counts = (
        _rank_token_count_before(request_absolute_query_end[:, None],
                                 request_ranks) -
        _rank_token_count_before(request_absolute_query_start[:, None],
                                 request_ranks))
    writeback_counts = jnp.where(req_active[:, None], writeback_counts, 0)
    rank_writeback_ends = jnp.cumsum(writeback_counts, axis=0)
    request_writeback_starts = rank_writeback_ends - writeback_counts

    # Broadcast all request/tile/rank coordinates into [request, tile, rank].
    q_global_starts, q_sizes, q_hbm_offsets = _rank_tile(
        q_len[:, None, None],
        token_owner_start[:, None, None],
        request_absolute_query_start[:, None, None],
        request_rank_local_q_starts[:, None, :],
        rank_indices[None, None, :],
        tile_indices[None, :, None],
    )
    q_last_row = jnp.maximum(q_sizes - 1, 0)
    q_global_last = (q_global_starts +
                     (q_last_row // interleave_size_i32) * cycle_i32 +
                     (q_last_row % interleave_size_i32))
    request_relative_end = (q_global_last -
                            request_absolute_query_start[:, None, None] + 1)
    request_relative_end = jnp.where(q_sizes > 0, request_relative_end, 0)
    current_effective_lens = jnp.max(request_relative_end, axis=-1)

    tile_active = current_effective_lens > 0
    owner_cycle_offset = jnp.mod(token_owner_start, cycle_i32)[:, None]
    has_history_boundary = (request_absolute_query_start %
                            virtual_page_size_i32 != 0)[:, None]
    full_history_num_groups = (request_absolute_query_start //
                               virtual_page_size_i32)[:, None]
    current_num_groups = jnp.where(
        tile_active,
        (owner_cycle_offset + current_effective_lens + cycle_i32 - 1) //
        cycle_i32 + has_history_boundary.astype(jnp.int32),
        0,
    )
    history_num_groups = jnp.where(tile_active, full_history_num_groups, 0)
    capture_current_kv = jnp.logical_and(
        tile_active,
        current_effective_lens == q_len[:, None],
    ).astype(jnp.int32)

    num_request_tiles = max_num_seqs * max_q_tiles
    tile_req_ids = jnp.broadcast_to(
        request_indices[:, None],
        (max_num_seqs, max_q_tiles),
    ).reshape(num_request_tiles)
    tile_q_global_starts = q_global_starts.reshape(num_request_tiles, pcp_size)
    tile_q_sizes = q_sizes.reshape(num_request_tiles, pcp_size)
    tile_q_hbm_offsets = q_hbm_offsets.reshape(num_request_tiles, pcp_size)
    tile_token_owner_starts = jnp.broadcast_to(
        token_owner_start[:, None],
        (max_num_seqs, max_q_tiles),
    ).reshape(num_request_tiles)
    tile_request_absolute_query_starts = jnp.broadcast_to(
        request_absolute_query_start[:, None],
        (max_num_seqs, max_q_tiles),
    ).reshape(num_request_tiles)
    tile_current_effective_lens = current_effective_lens.reshape(
        num_request_tiles)
    tile_current_kv_hbm_starts = jnp.broadcast_to(
        request_rank_local_q_starts[:, None, :],
        (max_num_seqs, max_q_tiles, pcp_size),
    ).reshape(num_request_tiles, pcp_size)
    tile_writeback_hbm_prefixes = jnp.broadcast_to(
        request_writeback_starts[:, None, :],
        (max_num_seqs, max_q_tiles, pcp_size),
    ).reshape(num_request_tiles, pcp_size)
    tile_capture_current_kv = capture_current_kv.reshape(num_request_tiles)
    current_num_groups = current_num_groups.reshape(num_request_tiles)
    history_num_groups = history_num_groups.reshape(num_request_tiles)
    num_tiles = tile_req_ids.shape[0]

    def _broadcast_tile(values):
        return jnp.broadcast_to(values[:, None], (num_tiles, pcp_size))

    logical_plan = jnp.stack(
        (
            _broadcast_tile(tile_req_ids),
            tile_q_global_starts,
            tile_q_hbm_offsets,
            tile_q_sizes,
            _broadcast_tile(tile_token_owner_starts),
            _broadcast_tile(tile_request_absolute_query_starts),
            _broadcast_tile(tile_current_effective_lens),
            tile_current_kv_hbm_starts,
            tile_writeback_hbm_prefixes,
            _broadcast_tile(tile_capture_current_kv),
            _broadcast_tile(current_num_groups),
            _broadcast_tile(history_num_groups),
        ),
        axis=-1,
    ).reshape(num_tiles, 1, pcp_size * TilePlanField.NUM_FIELDS)
    packed_plan_fields = _cdiv(
        pcp_size * TilePlanField.NUM_FIELDS,
        TilePlanField.HBM_ROW_FIELDS,
    ) * TilePlanField.HBM_ROW_FIELDS
    tile_plan = jnp.zeros((num_tiles, 1, packed_plan_fields), dtype=jnp.int32)
    tile_plan = tile_plan.at[..., :logical_plan.shape[-1]].set(logical_plan)

    index_capacity = _cdiv(
        num_tiles,
        TilePlanField.HBM_ROW_FIELDS,
    ) * TilePlanField.HBM_ROW_FIELDS

    def _compact_pass(num_groups):
        active = num_groups > 0
        active_count = jnp.sum(active, dtype=jnp.int32)
        tile_ids = jnp.nonzero(active, size=num_tiles,
                               fill_value=0)[0].astype(jnp.int32)
        positions = jnp.arange(num_tiles, dtype=jnp.int32)
        compact_groups = jnp.where(positions < active_count,
                                   num_groups[tile_ids], 0)
        group_starts = jnp.cumsum(compact_groups) - compact_groups

        padded_ids = jnp.zeros((1, index_capacity), dtype=jnp.int32)
        padded_ids = padded_ids.at[0, :num_tiles].set(tile_ids)
        padded_starts = jnp.zeros((1, index_capacity), dtype=jnp.int32)
        padded_starts = padded_starts.at[0, :num_tiles].set(group_starts)
        meta = jnp.stack((active_count, jnp.sum(num_groups, dtype=jnp.int32)))
        return padded_ids, padded_starts, meta

    current_ids, current_starts, current_meta = _compact_pass(
        current_num_groups)
    history_ids, history_starts, history_meta = _compact_pass(
        history_num_groups)
    return (tile_plan, block_tables, current_ids, current_starts, current_meta,
            history_ids, history_starts, history_meta)
