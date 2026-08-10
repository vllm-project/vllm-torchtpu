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
"""Host-side schedules for PCP streaming prefill RPA."""

import dataclasses

import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    pcp_token_owner_chunk_ranges


class ScheduleField:
    REQ_ID = 0
    KV_PAGE_RANK = 1
    KV_PAGE_IDX = 2
    IS_FIRST_KV = 3
    IS_LAST_KV = 4
    LOAD_Q = 5
    Q_GLOBAL_START = 6
    KV_GLOBAL_START = 7
    KV_VALID_LEN = 8
    Q_HBM_OFFSET = 9
    Q_TILE_SIZE = 10
    O_HBM_OFFSET = 11
    NUM_FIELDS = 12
    PACKED_NUM_FIELDS = 128
    KV_PAGE_INDICES_START = NUM_FIELDS
    MAX_KV_PAGES_PER_BLOCK = PACKED_NUM_FIELDS - KV_PAGE_INDICES_START


class TilePlanField:
    """Fields stored once per query tile and PCP consumer rank."""

    REQ_ID = 0
    KV_LEN = 1
    Q_GLOBAL_START = 2
    Q_HBM_OFFSET = 3
    Q_TILE_SIZE = 4
    EFFECTIVE_PAGES = 5
    HISTORY_CUT_PAGES = 6
    HISTORY_EFFECTIVE_PAGES = 7
    CURRENT_NUM_GROUPS = 8
    HISTORY_NUM_GROUPS = 9
    NUM_FIELDS = 10
    HBM_ROW_FIELDS = 128


_PACKED_FIELD_NAMES = (
    "req_id",
    "kv_page_rank",
    "kv_page_idx",
    "is_first_kv",
    "is_last_kv",
    "load_q",
    "q_global_start",
    "kv_global_start",
    "kv_valid_len",
    "q_hbm_offset",
    "q_tile_size",
    "o_hbm_offset",
)


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


def _pcp_kv_steps_for_token_count(token_count: int, *, page_size: int,
                                  pcp_size: int, interleave_size: int) -> int:
    if token_count <= 0:
        return 0
    last_pos = token_count - 1
    virtual_page_size = page_size * pcp_size
    full_virtual_pages = last_pos // virtual_page_size
    rem = last_pos - full_virtual_pages * virtual_page_size
    ranks_in_last_page = min(pcp_size, rem // interleave_size + 1)
    return full_virtual_pages * pcp_size + ranks_in_last_page


def _pcp_kv_steps_for_last_pos(last_pos: int, *, page_size: int, pcp_size: int,
                               interleave_size: int) -> int:
    return _pcp_kv_steps_for_token_count(last_pos + 1,
                                         page_size=page_size,
                                         pcp_size=pcp_size,
                                         interleave_size=interleave_size)


def _pcp_local_page_global_start(local_page_idx: int, src_rank: int, *,
                                 page_size: int, pcp_size: int,
                                 interleave_size: int) -> int:
    return (local_page_idx * pcp_size * page_size + src_rank * interleave_size)


def _pcp_local_page_valid_len(kv_len: int, local_page_idx: int, src_rank: int,
                              *, page_size: int, pcp_size: int,
                              interleave_size: int) -> int:
    base = _pcp_local_page_global_start(local_page_idx,
                                        src_rank,
                                        page_size=page_size,
                                        pcp_size=pcp_size,
                                        interleave_size=interleave_size)
    if kv_len <= base:
        return 0
    delta = kv_len - base
    cycle = pcp_size * interleave_size
    chunks_per_page = page_size // interleave_size
    full_chunks = min(chunks_per_page, delta // cycle)
    if full_chunks >= chunks_per_page:
        return page_size
    partial = min(interleave_size, max(delta - full_chunks * cycle, 0))
    return full_chunks * interleave_size + partial


def _q_global_last_for_strided_tile(q_global_start: int, q_tile_len: int, *,
                                    pcp_size: int,
                                    interleave_size: int) -> int:
    last_row = q_tile_len - 1
    return (q_global_start +
            (last_row // interleave_size) * pcp_size * interleave_size +
            (last_row % interleave_size))


def _iter_pcp_token_owner_tiles(q_len: int, token_owner_start: int,
                                consumer_rank: int, *, pcp_size: int,
                                interleave_size: int, bq_sz: int):
    """Yield local-contiguous Q tiles in token-ownership coordinates.

    The original schedule emitted one tile per PCP interleave chunk. For small
    interleave sizes that forces tiny Q tiles and repeats the same streamed KV
    traffic. When adjacent rank-owned chunks are full interleave chunks, they
    are contiguous in local HBM and can be consumed as one larger tile. The
    kernel reconstructs each row's strided global Q position from the first
    chunk's global start.
    """
    ranges = list(
        pcp_token_owner_chunk_ranges(q_len, token_owner_start, consumer_rank,
                                     pcp_size, interleave_size))
    if bq_sz <= interleave_size:
        for chunk_start, chunk_end in ranges:
            chunk_len = chunk_end - chunk_start
            for tile_start in range(0, chunk_len, bq_sz):
                tile_len = min(bq_sz, chunk_len - tile_start)
                yield chunk_start + tile_start, tile_len
        return

    range_idx = 0
    while range_idx < len(ranges):
        chunk_start, chunk_end = ranges[range_idx]
        chunk_len = chunk_end - chunk_start
        if chunk_len != interleave_size:
            for tile_start in range(0, chunk_len, bq_sz):
                tile_len = min(bq_sz, chunk_len - tile_start)
                yield chunk_start + tile_start, tile_len
            range_idx += 1
            continue

        tile_global_start = chunk_start
        tile_len = 0
        expected_chunk_start = chunk_start
        while range_idx < len(ranges):
            next_start, next_end = ranges[range_idx]
            next_len = next_end - next_start
            if (next_len != interleave_size
                    or next_start != expected_chunk_start
                    or tile_len + next_len > bq_sz):
                break
            tile_len += next_len
            range_idx += 1
            expected_chunk_start += pcp_size * interleave_size

        yield tile_global_start, tile_len


def _iter_pcp_q_tiles(q_len: int, token_owner_start: int,
                      request_absolute_query_start: int, consumer_rank: int, *,
                      pcp_size: int, interleave_size: int, bq_sz: int):
    """Yield rank-owned Q tiles expressed in request-absolute coordinates."""
    for owner_tile_start, tile_len in _iter_pcp_token_owner_tiles(
            q_len,
            token_owner_start,
            consumer_rank,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            bq_sz=bq_sz):
        request_relative_start = owner_tile_start - token_owner_start
        yield request_absolute_query_start + request_relative_start, tile_len


def _build_pcp_rank_query_tiles(
    q_len: int,
    token_owner_start: int,
    request_absolute_query_start: int,
    rank_local_q_offsets: np.ndarray,
    *,
    pcp_size: int,
    interleave_size: int,
    bq_sz: int,
) -> tuple[list[list[tuple[int, int, int]]], np.ndarray]:
    """Build per-rank ``(absolute_start, size, HBM offset)`` Q tiles."""
    next_rank_local_q_offsets = np.asarray(rank_local_q_offsets,
                                           dtype=np.int64).copy()
    if next_rank_local_q_offsets.shape != (pcp_size, ):
        raise ValueError("rank_local_q_offsets must have shape (pcp_size,).")

    rank_tiles: list[list[tuple[int, int, int]]] = []
    for consumer_rank in range(pcp_size):
        q_hbm_offset = int(next_rank_local_q_offsets[consumer_rank])
        tiles = []
        for q_global_start, q_tile_size in _iter_pcp_q_tiles(
                q_len,
                token_owner_start,
                request_absolute_query_start,
                consumer_rank,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                bq_sz=bq_sz):
            tiles.append(
                (int(q_global_start), int(q_tile_size), int(q_hbm_offset)))
            q_hbm_offset += int(q_tile_size)
        next_rank_local_q_offsets[consumer_rank] = q_hbm_offset
        rank_tiles.append(tiles)
    return rank_tiles, next_rank_local_q_offsets


@dataclasses.dataclass(frozen=True)
class PcpStreamingSchedule:
    """PCP streaming schedule arrays.

    All per-step fields have shape [pcp_size, max_steps, num_lanes]. The first
    dimension is the consumer rank. All PCP ranks receive the same replicated
    schedule so source ranks can push their local KV pages to any consumer.
    """

    req_id: np.ndarray
    kv_page_rank: np.ndarray
    kv_page_idx: np.ndarray
    is_first_kv: np.ndarray
    is_last_kv: np.ndarray
    load_q: np.ndarray
    q_global_start: np.ndarray
    kv_global_start: np.ndarray
    kv_valid_len: np.ndarray
    q_hbm_offset: np.ndarray
    q_tile_size: np.ndarray
    o_hbm_offset: np.ndarray
    packed_schedule: np.ndarray
    actual_steps: np.ndarray
    global_actual_steps: np.ndarray
    kv_page_indices: np.ndarray | None = None

    @property
    def pcp_size(self) -> int:
        return self.req_id.shape[0]

    @property
    def max_steps(self) -> int:
        return self.req_id.shape[1]

    @property
    def num_lanes(self) -> int:
        return self.req_id.shape[2]


def build_pcp_streaming_active_page_groups(
    schedule: PcpStreamingSchedule, ) -> np.ndarray:
    """Build kernel active-page-group metadata from a generated schedule."""
    global_actual_steps = np.asarray(schedule.global_actual_steps)
    if global_actual_steps.ndim != 1 or global_actual_steps.size != 1:
        raise ValueError(
            "PCP streaming schedule must have one global_actual_steps value.")
    actual_steps = int(global_actual_steps[0])
    if actual_steps % schedule.pcp_size != 0:
        raise ValueError("PCP streaming schedule steps must be padded to a "
                         "PCP page group.")
    return np.array([actual_steps // schedule.pcp_size], dtype=np.int32)


def _reshape_metadata_page_indices(
    page_indices: np.ndarray,
    max_num_seqs: int,
) -> np.ndarray:
    if max_num_seqs <= 0:
        raise ValueError("metadata must contain at least one sequence slot.")
    if page_indices.ndim == 1:
        if page_indices.size % max_num_seqs != 0:
            raise ValueError("flat page_indices size must be divisible by "
                             "kv_lens.shape[0].")
        pages_per_seq = page_indices.size // max_num_seqs
        return page_indices.reshape(max_num_seqs, pages_per_seq)
    if page_indices.ndim == 2:
        if page_indices.shape[0] != max_num_seqs:
            raise ValueError("2D page_indices first dimension must match "
                             "kv_lens.shape[0].")
        return page_indices
    raise ValueError("page_indices must be rank 1 or rank 2.")


def _metadata_num_reqs(distribution: np.ndarray, max_num_seqs: int,
                       cu_q_lens: np.ndarray) -> int:
    if distribution.ndim != 1 or distribution.size != 3:
        raise ValueError("distribution must have shape (3,).")
    decode_end = int(distribution[0])
    prefill_end = int(distribution[1])
    num_reqs = int(distribution[2])
    if decode_end < 0 or prefill_end < decode_end or num_reqs < prefill_end:
        raise ValueError("distribution must satisfy "
                         "0 <= decode_end <= prefill_end <= mixed_end.")
    if num_reqs <= 0:
        raise NotImplementedError(
            "PCP streaming metadata schedule requires at least one active "
            "request.")
    if num_reqs > max_num_seqs:
        raise ValueError("distribution[2] exceeds kv_lens.shape[0].")
    if cu_q_lens.size < num_reqs + 1:
        raise ValueError("cu_q_lens must cover distribution[2] requests.")
    return num_reqs


def generate_pcp_streaming_schedule_from_metadata_host(
    kv_lens: list[int] | np.ndarray,
    page_indices: np.ndarray,
    cu_q_lens: list[int] | np.ndarray,
    distribution: list[int] | np.ndarray,
    *,
    global_bucket_tokens: int,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    q_block_size: int,
    num_lanes: int = 1,
    kv_pages_per_block: int = 1,
    max_context_tokens: int | None = None,
) -> PcpStreamingSchedule:
    """Build the host/oracle PCP streaming schedule from standard metadata.

    This runtime helper is intentionally narrow for the metadata-driven path:
    active query spans packed from index 0, one lane, one KV page per ring step
    for multi-request chunks, and page_size divisible by interleave_size. The static
    schedule shape is padded from the compile bucket size, not the live query
    length. It converts inputs to NumPy and is not safe for JAX tracers.
    """
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    page_indices = np.asarray(page_indices, dtype=np.int32)
    cu_q_lens = np.asarray(cu_q_lens, dtype=np.int64)
    distribution = np.asarray(distribution, dtype=np.int64)

    if kv_lens.ndim != 1:
        raise ValueError("kv_lens must be a 1D array.")
    if cu_q_lens.ndim != 1:
        raise ValueError("cu_q_lens must be a 1D array.")
    max_num_seqs = int(kv_lens.size)
    block_tables = _reshape_metadata_page_indices(page_indices, max_num_seqs)
    num_reqs = _metadata_num_reqs(distribution, max_num_seqs, cu_q_lens)

    global_bucket_tokens = int(global_bucket_tokens)
    local_kv_cache_num_blocks = int(local_kv_cache_num_blocks)
    if global_bucket_tokens <= 0:
        raise ValueError("global_bucket_tokens must be positive.")
    if local_kv_cache_num_blocks <= 0:
        raise ValueError("local_kv_cache_num_blocks must be positive.")
    if global_bucket_tokens % pcp_size != 0:
        raise ValueError("global_bucket_tokens must be divisible by pcp_size.")
    if num_lanes != 1:
        raise NotImplementedError(
            "PCP streaming metadata schedule only supports num_lanes == 1.")
    if kv_pages_per_block != 1:
        raise NotImplementedError(
            "PCP streaming metadata schedule only supports "
            "kv_pages_per_block == 1.")

    kv_lens_active = kv_lens[:num_reqs]
    cu_q_lens_active = cu_q_lens[:num_reqs + 1]
    q_lens = cu_q_lens_active[1:] - cu_q_lens_active[:-1]
    request_absolute_query_start_offsets = kv_lens_active - q_lens
    if np.any(q_lens <= 0):
        raise NotImplementedError(
            "PCP streaming metadata schedule requires every active request "
            "to have positive q_len.")

    local_block_tables = (block_tables[:num_reqs].astype(np.int64) %
                          local_kv_cache_num_blocks).astype(np.int32)

    capacity_tokens = local_kv_cache_num_blocks * pcp_size * page_size
    too_long = np.flatnonzero(kv_lens_active > capacity_tokens)
    if too_long.size:
        first = int(too_long[0])
        raise ValueError(
            f"kv_lens[{first}] exceeds local KV cache PCP capacity: "
            f"kv_len={int(kv_lens_active[first])} {capacity_tokens=}.")
    schedule_capacity_tokens = capacity_tokens
    if max_context_tokens is not None:
        max_context_tokens = int(max_context_tokens)
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive.")
        schedule_capacity_tokens = min(schedule_capacity_tokens,
                                       max_context_tokens)
    pad_steps_to = estimate_pcp_streaming_metadata_schedule_steps_ub(
        global_bucket_tokens=global_bucket_tokens,
        capacity_tokens=schedule_capacity_tokens,
        block_size=page_size,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
        kv_pages_per_block=kv_pages_per_block,
        max_num_reqs=max_num_seqs,
    )

    return _generate_pcp_streaming_schedule_lockstep_compact(
        kv_lens=kv_lens_active,
        cu_q_lens=cu_q_lens_active,
        token_owner_start_offsets=request_absolute_query_start_offsets,
        request_absolute_query_start_offsets=(
            request_absolute_query_start_offsets),
        block_tables=local_block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=q_block_size,
        pad_kv_pages_to_pcp_group=True,
        pad_steps_to=pad_steps_to,
        kv_pages_per_block=kv_pages_per_block,
    )


def build_pcp_streaming_schedule_inputs_from_metadata_host(
    kv_lens: list[int] | np.ndarray,
    page_indices: np.ndarray,
    cu_q_lens: list[int] | np.ndarray,
    distribution: list[int] | np.ndarray,
    *,
    global_bucket_tokens: int,
    local_kv_cache_num_blocks: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    q_block_size: int,
    num_lanes: int = 1,
    kv_pages_per_block: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return host/oracle packed_schedule and active_page_groups."""
    schedule = generate_pcp_streaming_schedule_from_metadata_host(
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        global_bucket_tokens=global_bucket_tokens,
        local_kv_cache_num_blocks=local_kv_cache_num_blocks,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        q_block_size=q_block_size,
        num_lanes=num_lanes,
        kv_pages_per_block=kv_pages_per_block,
    )
    return (schedule.packed_schedule,
            build_pcp_streaming_active_page_groups(schedule))


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
    max_steps: int | None = None,
    max_context_tokens: int | None = None,
    num_lanes: int = 1,
    kv_pages_per_block: int = 1,
    compact: bool = False,
):
    """Build current/history packed schedules with pure JAX ops.

    Compiled-safe scheduler for PCP query metadata. It supports compact
    per-rank Q tiles with arbitrary live q_len/q_start values while the schedule
    shapes remain static from the compile bucket. The current schedule covers
    KV pages that overlap the active query cycle; the history schedule covers
    earlier KV pages. Callers must keep the narrow metadata path constraints:
    num_lanes=1, kv_pages_per_block=1, and page_size divisible by
    interleave_size.
    """
    import jax.numpy as jnp

    global_bucket_tokens = int(global_bucket_tokens)
    local_kv_cache_num_blocks = int(local_kv_cache_num_blocks)
    page_size = int(page_size)
    pcp_size = int(pcp_size)
    interleave_size = int(interleave_size)
    q_block_size = int(q_block_size)
    num_lanes = int(num_lanes)
    kv_pages_per_block = int(kv_pages_per_block)

    if num_lanes != 1:
        raise NotImplementedError(
            "PCP streaming JAX metadata schedule only supports num_lanes == 1."
        )
    if kv_pages_per_block != 1:
        raise NotImplementedError(
            "PCP streaming JAX metadata schedule only supports "
            "kv_pages_per_block == 1.")
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
    if max_context_tokens is not None:
        max_context_tokens = int(max_context_tokens)
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive.")

    max_num_seqs = int(np.shape(kv_lens)[0])
    if max_steps is None:
        capacity_tokens = local_kv_cache_num_blocks * pcp_size * page_size
        if max_context_tokens is not None:
            capacity_tokens = min(capacity_tokens, max_context_tokens)
        max_steps = estimate_pcp_streaming_metadata_schedule_steps_ub(
            global_bucket_tokens=global_bucket_tokens,
            capacity_tokens=capacity_tokens,
            block_size=page_size,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            kv_pages_per_block=kv_pages_per_block,
            max_num_reqs=max_num_seqs,
        )
    max_steps = int(max_steps)

    kv_lens = jnp.asarray(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.asarray(cu_q_lens, dtype=jnp.int32)
    page_indices = jnp.asarray(page_indices, dtype=jnp.int32)
    distribution = jnp.asarray(distribution, dtype=jnp.int32)
    block_tables = _reshape_metadata_page_indices_jax(page_indices,
                                                      max_num_seqs)
    pages_per_seq = block_tables.shape[1]
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

    def _kv_steps_for_token_count(token_count):
        last_pos = token_count - 1
        full_virtual_pages = last_pos // virtual_page_size_i32
        rem = last_pos - full_virtual_pages * virtual_page_size_i32
        ranks_in_last_page = jnp.minimum(pcp_size_i32,
                                         rem // interleave_size_i32 + 1)
        return jnp.where(
            token_count > 0,
            full_virtual_pages * pcp_size_i32 + ranks_in_last_page, 0)

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

    tile_req_ids = []
    tile_kv_lens = []
    tile_request_absolute_query_starts = []
    tile_q_global_starts = []
    tile_q_sizes = []
    tile_q_hbm_offsets = []
    tile_scheduled_steps = []
    tile_effective_pages_by_rank = []

    rank_local_q_prefixes = jnp.zeros((pcp_size, ), dtype=jnp.int32)
    for req_idx in range(max_num_seqs):
        q_len_raw = cu_q_lens[req_idx + 1] - cu_q_lens[req_idx]
        req_active = jnp.logical_and(req_idx < active_num_reqs, q_len_raw > 0)
        q_len = jnp.where(req_active, q_len_raw, 0)
        kv_len = jnp.where(req_active, kv_lens[req_idx], 0)
        request_absolute_query_start = kv_len - q_len
        # The public metadata path keeps the current ownership policy.
        # Keeping these as distinct values makes the later owner-policy switch
        # independent from causal/history/cache coordinates.
        token_owner_start = request_absolute_query_start
        num_kv_pages = _kv_steps_for_token_count(kv_len)

        for tile_idx in range(max_q_tiles):
            q_global_starts = []
            q_hbm_offsets = []
            q_sizes = []
            effective_by_rank = []
            max_effective_kv_pages = jnp.asarray(0, dtype=jnp.int32)
            for consumer_rank in range(pcp_size):
                q_global_start, q_tile_size, q_hbm_offset = _rank_tile(
                    q_len,
                    token_owner_start,
                    request_absolute_query_start,
                    rank_local_q_prefixes[consumer_rank],
                    consumer_rank,
                    tile_idx,
                )
                q_global_starts.append(q_global_start)
                q_hbm_offsets.append(q_hbm_offset)
                q_sizes.append(q_tile_size)
                q_tile_size_for_last = jnp.maximum(q_tile_size, 1)
                q_global_last = (
                    q_global_start +
                    ((q_tile_size_for_last - 1) // interleave_size_i32) *
                    pcp_size_i32 * interleave_size_i32 +
                    ((q_tile_size_for_last - 1) % interleave_size_i32))
                effective_kv_pages = jnp.minimum(
                    num_kv_pages, _kv_steps_for_token_count(q_global_last + 1))
                effective_kv_pages = jnp.where(
                    jnp.logical_and(req_active, q_tile_size > 0),
                    effective_kv_pages,
                    0,
                )
                effective_by_rank.append(effective_kv_pages)
                max_effective_kv_pages = jnp.maximum(max_effective_kv_pages,
                                                     effective_kv_pages)

            scheduled_steps = ((max_effective_kv_pages + pcp_size_i32 - 1) //
                               pcp_size_i32 * pcp_size_i32)
            tile_req_ids.append(jnp.asarray(req_idx, dtype=jnp.int32))
            tile_kv_lens.append(kv_len)
            tile_request_absolute_query_starts.append(
                request_absolute_query_start)
            tile_q_global_starts.append(jnp.stack(q_global_starts))
            tile_q_sizes.append(jnp.stack(q_sizes))
            tile_q_hbm_offsets.append(jnp.stack(q_hbm_offsets))
            tile_scheduled_steps.append(scheduled_steps)
            tile_effective_pages_by_rank.append(jnp.stack(effective_by_rank))

        local_counts = []
        for consumer_rank in range(pcp_size):
            *_, local_count = _rank_request_layout(q_len, token_owner_start,
                                                   consumer_rank)
            local_counts.append(jnp.where(req_active, local_count, 0))
        rank_local_q_prefixes = rank_local_q_prefixes + jnp.stack(local_counts)

    tile_req_ids = jnp.stack(tile_req_ids)
    tile_kv_lens = jnp.stack(tile_kv_lens)
    tile_request_absolute_query_starts = jnp.stack(
        tile_request_absolute_query_starts)
    tile_q_global_starts = jnp.stack(tile_q_global_starts)
    tile_q_sizes = jnp.stack(tile_q_sizes)
    tile_q_hbm_offsets = jnp.stack(tile_q_hbm_offsets)
    tile_scheduled_steps = jnp.stack(tile_scheduled_steps)
    tile_effective_pages_by_rank = jnp.stack(tile_effective_pages_by_rank)

    def _pack_schedule(tile_page_starts, tile_effective_pages_by_rank,
                       tile_scheduled_steps):
        tile_start_steps = jnp.cumsum(
            tile_scheduled_steps) - tile_scheduled_steps
        global_actual_steps = jnp.sum(tile_scheduled_steps, dtype=jnp.int32)

        step_ids = jnp.arange(max_steps, dtype=jnp.int32)
        in_tile = jnp.logical_and(
            step_ids[None, :] >= tile_start_steps[:, None],
            step_ids[None, :]
            < (tile_start_steps + tile_scheduled_steps)[:, None],
        )
        step_has_tile = jnp.any(in_tile, axis=0)
        step_tile_idx = jnp.argmax(in_tile.astype(jnp.int32), axis=0)

        plan_start = tile_start_steps[step_tile_idx]
        plan_req_id = tile_req_ids[step_tile_idx]
        plan_kv_len = tile_kv_lens[step_tile_idx]
        plan_q_global_start = tile_q_global_starts[step_tile_idx]
        plan_q_size = tile_q_sizes[step_tile_idx]
        plan_q_hbm_offset = tile_q_hbm_offsets[step_tile_idx]
        plan_page_start = tile_page_starts[step_tile_idx]
        plan_effective_by_rank = tile_effective_pages_by_rank[step_tile_idx]

        step_offset = step_ids - plan_start
        global_page = plan_page_start + step_offset
        src_rank = jnp.mod(global_page, pcp_size_i32)
        local_page_idx = global_page // pcp_size_i32
        safe_local_page_idx = jnp.clip(local_page_idx, 0, pages_per_seq - 1)
        page_idx = block_tables[plan_req_id, safe_local_page_idx]

        step_has_tile_2d = step_has_tile[:, None]
        valid_page = jnp.logical_and(
            step_has_tile_2d,
            global_page[:, None] < plan_effective_by_rank,
        )
        page_valid_len = _local_page_valid_len(plan_kv_len, local_page_idx,
                                               src_rank)
        kv_valid_len = jnp.where(valid_page,
                                 jnp.maximum(page_valid_len[:, None], 0), 0)
        is_first = jnp.logical_and(valid_page, step_offset[:, None] == 0)
        is_last = jnp.logical_and(
            valid_page,
            global_page[:, None] == plan_effective_by_rank - 1,
        )
        q_global_start = plan_q_global_start
        kv_global_start = jnp.broadcast_to(
            _local_page_global_start(local_page_idx, src_rank)[:, None],
            (max_steps, pcp_size))
        q_hbm_offset = plan_q_hbm_offset
        q_tile_size = plan_q_size

        req_id_field = jnp.where(valid_page, plan_req_id[:, None], -1)
        kv_page_rank_field = jnp.where(
            step_has_tile_2d,
            jnp.broadcast_to(src_rank[:, None], (max_steps, pcp_size)),
            -1,
        )
        kv_page_idx_field = jnp.where(
            step_has_tile_2d,
            jnp.where(valid_page, page_idx[:, None], 0),
            -1,
        )
        zero_2d = jnp.zeros((max_steps, pcp_size), dtype=jnp.int32)
        logical_fields = (
            req_id_field,
            kv_page_rank_field,
            kv_page_idx_field,
            is_first.astype(jnp.int32),
            is_last.astype(jnp.int32),
            is_first.astype(jnp.int32),
            jnp.where(step_has_tile_2d, q_global_start, zero_2d),
            jnp.where(step_has_tile_2d, kv_global_start, zero_2d),
            kv_valid_len,
            jnp.where(step_has_tile_2d, q_hbm_offset, zero_2d),
            jnp.where(step_has_tile_2d, q_tile_size, zero_2d),
            jnp.where(step_has_tile_2d, q_hbm_offset, zero_2d),
        )
        logical = jnp.stack(logical_fields, axis=-1)[:, :, None, :]
        packed_shape = (max_steps, pcp_size, num_lanes,
                        ScheduleField.PACKED_NUM_FIELDS)
        packed = jnp.zeros(packed_shape, dtype=jnp.int32)
        packed = packed.at[..., :ScheduleField.NUM_FIELDS].set(logical)
        active_page_groups = jnp.asarray([global_actual_steps // pcp_size],
                                         dtype=jnp.int32)
        return packed, active_page_groups

    history_cut_pages = ((tile_request_absolute_query_starts //
                          (page_size_i32 * pcp_size_i32)) * pcp_size_i32)
    history_effective_pages_by_rank = jnp.minimum(tile_effective_pages_by_rank,
                                                  history_cut_pages[:, None])
    history_max_effective_pages = jnp.max(history_effective_pages_by_rank,
                                          axis=1)
    history_scheduled_steps = (
        (history_max_effective_pages + pcp_size_i32 - 1) // pcp_size_i32 *
        pcp_size_i32)
    current_remaining_pages_by_rank = jnp.maximum(
        tile_effective_pages_by_rank - history_cut_pages[:, None], 0)
    current_max_remaining_pages = jnp.max(current_remaining_pages_by_rank,
                                          axis=1)
    current_scheduled_steps = (
        (current_max_remaining_pages + pcp_size_i32 - 1) // pcp_size_i32 *
        pcp_size_i32)

    if compact:
        current_num_groups = current_scheduled_steps // pcp_size_i32
        history_num_groups = history_scheduled_steps // pcp_size_i32
        num_tiles = tile_req_ids.shape[0]

        def _broadcast_tile(values):
            return jnp.broadcast_to(values[:, None], (num_tiles, pcp_size))

        logical_plan = jnp.stack(
            (
                _broadcast_tile(tile_req_ids),
                _broadcast_tile(tile_kv_lens),
                tile_q_global_starts,
                tile_q_hbm_offsets,
                tile_q_sizes,
                tile_effective_pages_by_rank,
                _broadcast_tile(history_cut_pages),
                history_effective_pages_by_rank,
                _broadcast_tile(current_num_groups),
                _broadcast_tile(history_num_groups),
            ),
            axis=-1,
        ).reshape(num_tiles, 1, pcp_size * TilePlanField.NUM_FIELDS)
        packed_plan_fields = _cdiv(
            pcp_size * TilePlanField.NUM_FIELDS,
            TilePlanField.HBM_ROW_FIELDS,
        ) * TilePlanField.HBM_ROW_FIELDS
        tile_plan = jnp.zeros((num_tiles, 1, packed_plan_fields),
                              dtype=jnp.int32)
        tile_plan = tile_plan.at[..., :logical_plan.shape[-1]].set(
            logical_plan)

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
            meta = jnp.stack((active_count, jnp.sum(num_groups,
                                                    dtype=jnp.int32)))
            return padded_ids, padded_starts, meta

        current_ids, current_starts, current_meta = _compact_pass(
            current_num_groups)
        history_ids, history_starts, history_meta = _compact_pass(
            history_num_groups)
        return (tile_plan, block_tables, current_ids, current_starts,
                current_meta, history_ids, history_starts, history_meta)

    current_schedule, current_active_page_groups = _pack_schedule(
        history_cut_pages,
        tile_effective_pages_by_rank,
        current_scheduled_steps,
    )
    history_schedule, history_active_page_groups = _pack_schedule(
        jnp.zeros_like(history_cut_pages),
        history_effective_pages_by_rank,
        history_scheduled_steps,
    )
    return (current_schedule, current_active_page_groups, history_schedule,
            history_active_page_groups)


def _validate_inputs(
    kv_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    token_owner_start_offsets: np.ndarray,
    request_absolute_query_start_offsets: np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    kv_pages_per_block: int,
) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive.")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive.")
    if num_lanes <= 0:
        raise ValueError("num_lanes must be positive.")
    if bq_sz <= 0:
        raise ValueError("bq_sz must be positive.")
    if kv_pages_per_block <= 0:
        raise ValueError("kv_pages_per_block must be positive.")
    if kv_pages_per_block > ScheduleField.MAX_KV_PAGES_PER_BLOCK:
        raise ValueError(
            "kv_pages_per_block exceeds packed schedule capacity: "
            f"{kv_pages_per_block} > "
            f"{ScheduleField.MAX_KV_PAGES_PER_BLOCK}.")
    _validate_page_interleave_size(page_size, interleave_size,
                                   "PCP streaming schedule")
    if cu_q_lens.ndim != 1 or cu_q_lens.size == 0:
        raise ValueError("cu_q_lens must be a non-empty 1D array.")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be a 2D array.")

    num_reqs = int(cu_q_lens.size - 1)
    if kv_lens.size < num_reqs:
        raise ValueError("kv_lens must cover every request in cu_q_lens.")
    if token_owner_start_offsets.size < num_reqs:
        raise ValueError("token_owner_start_offsets must cover every request.")
    if request_absolute_query_start_offsets.size < num_reqs:
        raise ValueError(
            "request_absolute_query_start_offsets must cover every request.")
    if block_tables.shape[0] < num_reqs:
        raise ValueError("block_tables must cover every request.")
    return num_reqs


def pack_pcp_streaming_schedule_fields(
    *,
    req_id: np.ndarray,
    kv_page_rank: np.ndarray,
    kv_page_idx: np.ndarray,
    is_first_kv: np.ndarray,
    is_last_kv: np.ndarray,
    load_q: np.ndarray,
    q_global_start: np.ndarray,
    kv_global_start: np.ndarray,
    kv_valid_len: np.ndarray,
    q_hbm_offset: np.ndarray,
    q_tile_size: np.ndarray,
    o_hbm_offset: np.ndarray,
) -> np.ndarray:
    """Pack schedule fields into [max_steps, pcp_size, lanes, padded_fields]."""
    field_arrays = (
        req_id,
        kv_page_rank,
        kv_page_idx,
        is_first_kv,
        is_last_kv,
        load_q,
        q_global_start,
        kv_global_start,
        kv_valid_len,
        q_hbm_offset,
        q_tile_size,
        o_hbm_offset,
    )
    if len({array.shape for array in field_arrays}) != 1:
        raise ValueError("all schedule fields must have identical shapes.")
    logical = np.stack(field_arrays, axis=-1).astype(np.int32, copy=False)
    padded_shape = logical.shape[:-1] + (ScheduleField.PACKED_NUM_FIELDS, )
    packed = np.zeros(padded_shape, dtype=np.int32)
    packed[..., :ScheduleField.NUM_FIELDS] = logical
    return np.transpose(packed, (1, 0, 2, 3)).copy()


def unpack_pcp_streaming_schedule_field(
    packed_schedule: np.ndarray,
    field: int,
) -> np.ndarray:
    """Unpack one field to [pcp_size, max_steps, num_lanes]."""
    if packed_schedule.ndim != 4:
        raise ValueError("packed_schedule must be rank 4.")
    if field < 0 or field >= ScheduleField.NUM_FIELDS:
        raise ValueError(f"invalid schedule field index: {field}")
    return np.transpose(packed_schedule[..., field], (1, 0, 2)).copy()


def _single_active_aligned_request_params(
    *,
    q_lens: np.ndarray,
    token_owner_start_offsets: np.ndarray,
    block_tables: np.ndarray,
    pcp_size: int,
    interleave_size: int,
    path_name: str,
) -> tuple[int, int]:
    if q_lens.ndim != 1:
        raise ValueError("q_lens must be a 1D array.")
    if token_owner_start_offsets.ndim != 1:
        raise ValueError("token_owner_start_offsets must be a 1D array.")
    if token_owner_start_offsets.size < q_lens.size:
        raise ValueError("token_owner_start_offsets must cover every request.")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be a 2D array.")
    if block_tables.shape[0] < 1 or block_tables.shape[1] == 0:
        raise ValueError("block_tables must cover the active request.")

    active = np.flatnonzero(q_lens > 0)
    if active.size != 1 or int(active[0]) != 0:
        raise NotImplementedError(
            f"{path_name} only supports one active request at index 0, got "
            f"active request indices {active.tolist()}.")

    q_len = int(q_lens[0])
    token_owner_start = int(token_owner_start_offsets[0])
    cycle = pcp_size * interleave_size
    if q_len <= 0:
        raise NotImplementedError(f"{path_name} requires a positive q_len.")
    if token_owner_start < 0:
        raise ValueError("token_owner_start_offset must be non-negative.")
    if q_len % cycle != 0:
        raise NotImplementedError(
            f"{path_name} requires q_len to be a multiple of "
            f"pcp_size * interleave_size, got {q_len=} {pcp_size=} "
            f"{interleave_size=}.")
    if token_owner_start % cycle != 0:
        raise NotImplementedError(
            f"{path_name} requires token_owner_start_offset to be aligned to "
            "pcp_size * interleave_size, got "
            f"{token_owner_start=} {pcp_size=} {interleave_size=}.")
    return q_len, token_owner_start


def _single_aligned_request_params(
    *,
    kv_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    token_owner_start_offsets: np.ndarray,
    request_absolute_query_start_offsets: np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool,
) -> tuple[int, int, int]:
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    if num_lanes != 1:
        raise NotImplementedError(
            "PCP streaming vectorized schedule only supports num_lanes == 1, "
            f"got {num_lanes}.")
    if not pad_kv_pages_to_pcp_group:
        raise NotImplementedError("PCP streaming vectorized schedule requires "
                                  "pad_kv_pages_to_pcp_group=True.")
    _validate_page_interleave_size(page_size, interleave_size,
                                   "PCP streaming vectorized schedule")
    if bq_sz % interleave_size != 0:
        raise NotImplementedError(
            "PCP streaming vectorized schedule requires bq_sz to be a "
            f"multiple of interleave_size, got {bq_sz=} "
            f"{interleave_size=}.")

    q_len, token_owner_start = _single_active_aligned_request_params(
        q_lens=q_lens,
        token_owner_start_offsets=token_owner_start_offsets,
        block_tables=block_tables,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        path_name="PCP streaming vectorized schedule",
    )
    request_absolute_query_start = int(request_absolute_query_start_offsets[0])
    if request_absolute_query_start < 0:
        raise ValueError(
            "request_absolute_query_start_offset must be non-negative.")
    kv_len = int(kv_lens[0])
    if kv_len < request_absolute_query_start + q_len:
        raise ValueError("kv_lens must include all scheduled Q tokens.")
    return q_len, token_owner_start, request_absolute_query_start


def estimate_pcp_streaming_schedule_steps_ub(
    q_lens: np.ndarray,
    capacity_tokens: int,
    block_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    q_block_size: int,
    kv_pages_per_block: int = 1,
) -> int:
    max_global_pages = _cdiv(capacity_tokens, block_size)
    kv_pages_per_block = max(1, int(kv_pages_per_block))
    max_steps = 0
    for consumer_rank in range(pcp_size):
        lane_lengths = np.zeros(num_lanes, dtype=np.int64)
        for q_len in q_lens:
            q_len = int(q_len)
            if q_len <= 0:
                continue
            request_absolute_query_start = max(0, capacity_tokens - q_len)
            token_owner_start = request_absolute_query_start
            for q_global, tile_len in _iter_pcp_q_tiles(
                    q_len,
                    token_owner_start,
                    request_absolute_query_start,
                    consumer_rank,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                    bq_sz=q_block_size):
                q_global_last = _q_global_last_for_strided_tile(
                    q_global,
                    tile_len,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size)
                effective_pages = min(max_global_pages,
                                      q_global_last // block_size + 1)
                scheduled_pages = (
                    _cdiv(effective_pages, pcp_size * kv_pages_per_block) *
                    pcp_size)
                target_lane = int(np.argmin(lane_lengths))
                lane_lengths[target_lane] += scheduled_pages
        max_steps = max(max_steps, int(lane_lengths.max(initial=0)))
    return max_steps


def estimate_pcp_streaming_metadata_schedule_steps_ub(
    *,
    global_bucket_tokens: int,
    capacity_tokens: int,
    block_size: int,
    pcp_size: int,
    q_block_size: int,
    kv_pages_per_block: int = 1,
    max_num_reqs: int = 1,
) -> int:
    """Conservative max_steps bound for compact lockstep metadata schedules."""
    global_bucket_tokens = int(global_bucket_tokens)
    capacity_tokens = int(capacity_tokens)
    block_size = int(block_size)
    pcp_size = int(pcp_size)
    q_block_size = int(q_block_size)
    kv_pages_per_block = max(1, int(kv_pages_per_block))
    max_num_reqs = max(1, int(max_num_reqs))
    if global_bucket_tokens <= 0:
        raise ValueError("global_bucket_tokens must be positive.")
    if capacity_tokens <= 0:
        raise ValueError("capacity_tokens must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive.")
    if q_block_size <= 0:
        raise ValueError("q_block_size must be positive.")
    if global_bucket_tokens % pcp_size != 0:
        raise ValueError("global_bucket_tokens must be divisible by pcp_size: "
                         f"{global_bucket_tokens=} {pcp_size=}.")

    local_bucket_tokens = global_bucket_tokens // pcp_size
    max_total_q_tiles = (_cdiv(local_bucket_tokens, q_block_size) +
                         3 * max_num_reqs)
    max_global_pages = _cdiv(capacity_tokens, block_size)
    max_page_groups_per_tile = _cdiv(max_global_pages,
                                     pcp_size * kv_pages_per_block)
    return max_total_q_tiles * max_page_groups_per_tile * pcp_size


def _last_group_info(effective_kv_pages: int, pcp_size: int,
                     kv_pages_per_block: int) -> tuple[int, int]:
    last_global_page = effective_kv_pages - 1
    last_local_page = last_global_page // pcp_size
    last_block = last_local_page // kv_pages_per_block
    last_local_page_start = last_block * kv_pages_per_block
    last_store_src_rank = 0
    for candidate_src_rank in range(pcp_size):
        for page_offset in range(kv_pages_per_block):
            candidate_global_page = (
                (last_local_page_start + page_offset) * pcp_size +
                candidate_src_rank)
            if candidate_global_page < effective_kv_pages:
                last_store_src_rank = candidate_src_rank
    return last_block, last_store_src_rank


def _fill_pcp_streaming_schedule_rows_vectorized(
    packed_schedule: np.ndarray,
    *,
    start_step: int,
    num_steps: int,
    consumer_rank: int,
    lane: int,
    req_id: int,
    q_global_start: int,
    q_tile_size: int,
    q_hbm_offset: int,
    kv_len: int,
    effective_kv_pages: int,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_pages_per_block: int,
) -> None:
    if num_steps <= 0:
        return

    kv_page_seq_idx = np.arange(num_steps, dtype=np.int32)
    src_rank = kv_page_seq_idx % pcp_size
    local_page_start = (kv_page_seq_idx // pcp_size) * kv_pages_per_block

    page_offsets = np.arange(kv_pages_per_block, dtype=np.int32)
    page_global = (
        (local_page_start[:, None] + page_offsets[None, :]) * pcp_size +
        src_rank[:, None])
    valid_pages = page_global < effective_kv_pages
    local_page_idx = page_global // pcp_size
    if np.any(valid_pages) and int(
            local_page_idx[valid_pages].max()) >= block_tables.shape[1]:
        raise ValueError("block_tables does not cover requested KV page.")
    safe_page_idx = np.minimum(local_page_idx, block_tables.shape[1] - 1)
    page_indices = np.where(valid_pages, block_tables[req_id, safe_page_idx],
                            0).astype(np.int32)

    page_valid = np.zeros_like(page_global, dtype=np.int32)
    for page_offset in range(kv_pages_per_block):
        for row_idx in range(num_steps):
            if valid_pages[row_idx, page_offset]:
                page_valid[row_idx, page_offset] = (_pcp_local_page_valid_len(
                    kv_len,
                    int(local_page_idx[row_idx, page_offset]),
                    int(src_rank[row_idx]),
                    page_size=page_size,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                ))
    kv_valid_len = page_valid.sum(axis=1).astype(np.int32)

    rows = packed_schedule[start_step:start_step + num_steps, consumer_rank,
                           lane, :]
    rows.fill(0)
    rows[:, ScheduleField.REQ_ID] = np.where(kv_valid_len > 0, req_id, -1)
    rows[:, ScheduleField.KV_PAGE_RANK] = src_rank
    rows[:, ScheduleField.KV_PAGE_IDX] = np.where(kv_valid_len > 0,
                                                  page_indices[:, 0], 0)

    last_block, last_store_src_rank = _last_group_info(effective_kv_pages,
                                                       pcp_size,
                                                       kv_pages_per_block)
    cur_block = kv_page_seq_idx // pcp_size
    valid = kv_valid_len > 0
    is_first = valid & (cur_block == 0) & (src_rank == 0)
    is_last = valid & (cur_block == last_block) & (src_rank
                                                   == last_store_src_rank)
    rows[:, ScheduleField.IS_FIRST_KV] = is_first.astype(np.int32)
    rows[:, ScheduleField.IS_LAST_KV] = is_last.astype(np.int32)
    rows[:, ScheduleField.LOAD_Q] = is_first.astype(np.int32)
    rows[:, ScheduleField.Q_GLOBAL_START] = q_global_start
    rows[:, ScheduleField.KV_GLOBAL_START] = [
        _pcp_local_page_global_start(
            int(local_page_start[i]),
            int(src_rank[i]),
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        ) for i in range(num_steps)
    ]
    rows[:, ScheduleField.KV_VALID_LEN] = kv_valid_len
    rows[:, ScheduleField.Q_HBM_OFFSET] = q_hbm_offset
    rows[:, ScheduleField.Q_TILE_SIZE] = q_tile_size
    rows[:, ScheduleField.O_HBM_OFFSET] = q_hbm_offset
    if kv_pages_per_block > 1:
        rows[
            :,
            ScheduleField.
            KV_PAGE_INDICES_START:ScheduleField.KV_PAGE_INDICES_START +
            kv_pages_per_block,
        ] = page_indices


def _packed_field(packed_schedule: np.ndarray, field: int) -> np.ndarray:
    return np.transpose(packed_schedule[..., field], (1, 0, 2)).copy()


def _generate_pcp_streaming_schedule_single_aligned(
    kv_lens: list[int] | np.ndarray,
    cu_q_lens: list[int] | np.ndarray,
    token_owner_start_offsets: list[int] | np.ndarray,
    request_absolute_query_start_offsets: list[int] | np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool = False,
    pad_steps_to: int | None = None,
    kv_pages_per_block: int = 1,
) -> PcpStreamingSchedule:
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    cu_q_lens = np.asarray(cu_q_lens, dtype=np.int64)
    token_owner_start_offsets = np.asarray(token_owner_start_offsets,
                                           dtype=np.int64)
    request_absolute_query_start_offsets = np.asarray(
        request_absolute_query_start_offsets, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    _validate_inputs(
        kv_lens,
        cu_q_lens,
        token_owner_start_offsets,
        request_absolute_query_start_offsets,
        block_tables,
        page_size,
        pcp_size,
        interleave_size,
        num_lanes,
        bq_sz,
        kv_pages_per_block,
    )
    (q_len, token_owner_start,
     request_absolute_query_start) = _single_aligned_request_params(
         kv_lens=kv_lens,
         cu_q_lens=cu_q_lens,
         token_owner_start_offsets=token_owner_start_offsets,
         request_absolute_query_start_offsets=(
             request_absolute_query_start_offsets),
         block_tables=block_tables,
         page_size=page_size,
         pcp_size=pcp_size,
         interleave_size=interleave_size,
         num_lanes=num_lanes,
         bq_sz=bq_sz,
         pad_kv_pages_to_pcp_group=pad_kv_pages_to_pcp_group,
     )

    kv_len = int(kv_lens[0])
    num_kv_pages = _pcp_kv_steps_for_token_count(
        kv_len,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    actual_steps = np.zeros(pcp_size, dtype=np.int32)
    tile_plans: list[list[tuple[int, int, int, int, int, int]]] = []
    actual_max_steps = 0
    for consumer_rank in range(pcp_size):
        rank_q_offset = 0
        rank_plans = []
        rank_steps = 0
        for q_global, q_tile_size in _iter_pcp_q_tiles(
                q_len,
                token_owner_start,
                request_absolute_query_start,
                consumer_rank,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                bq_sz=bq_sz):
            q_global_last = _q_global_last_for_strided_tile(
                q_global,
                q_tile_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size)
            effective_kv_pages = min(
                num_kv_pages,
                _pcp_kv_steps_for_last_pos(
                    q_global_last,
                    page_size=page_size,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                ),
            )
            if kv_pages_per_block == 1:
                scheduled_steps = effective_kv_pages
                if pad_kv_pages_to_pcp_group:
                    scheduled_steps = _cdiv(effective_kv_pages,
                                            pcp_size) * pcp_size
            else:
                scheduled_steps = (
                    _cdiv(effective_kv_pages, pcp_size * kv_pages_per_block) *
                    pcp_size)
            rank_plans.append((rank_steps, int(q_global), int(q_tile_size),
                               int(rank_q_offset), int(effective_kv_pages),
                               int(scheduled_steps)))
            rank_steps += scheduled_steps
            rank_q_offset += int(q_tile_size)
        actual_steps[consumer_rank] = rank_steps
        actual_max_steps = max(actual_max_steps, rank_steps)
        tile_plans.append(rank_plans)

    max_steps = actual_max_steps
    if pad_steps_to is not None:
        pad_steps_to = int(pad_steps_to)
        if pad_steps_to < actual_max_steps:
            raise ValueError(
                "pad_steps_to must be >= generated schedule steps: "
                f"{pad_steps_to=} {actual_max_steps=}.")
        max_steps = pad_steps_to

    packed_schedule = np.zeros(
        (max_steps, pcp_size, num_lanes, ScheduleField.PACKED_NUM_FIELDS),
        dtype=np.int32)
    packed_schedule[..., ScheduleField.REQ_ID] = -1
    packed_schedule[..., ScheduleField.KV_PAGE_RANK] = -1
    packed_schedule[..., ScheduleField.KV_PAGE_IDX] = -1

    for consumer_rank, rank_plans in enumerate(tile_plans):
        for (start_step, q_global, q_tile_size, q_hbm_offset,
             effective_kv_pages, scheduled_steps) in rank_plans:
            _fill_pcp_streaming_schedule_rows_vectorized(
                packed_schedule,
                start_step=start_step,
                num_steps=scheduled_steps,
                consumer_rank=consumer_rank,
                lane=0,
                req_id=0,
                q_global_start=q_global,
                q_tile_size=q_tile_size,
                q_hbm_offset=q_hbm_offset,
                kv_len=kv_len,
                effective_kv_pages=effective_kv_pages,
                block_tables=block_tables,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                kv_pages_per_block=kv_pages_per_block,
            )

    fields = {
        name: _packed_field(packed_schedule,
                            getattr(ScheduleField, name.upper()))
        for name in _PACKED_FIELD_NAMES[1:]
    }
    req_id = _packed_field(packed_schedule, ScheduleField.REQ_ID)
    kv_page_indices = None
    if kv_pages_per_block > 1:
        kv_page_indices = np.transpose(
            packed_schedule[
                ...,
                ScheduleField.
                KV_PAGE_INDICES_START:ScheduleField.KV_PAGE_INDICES_START +
                kv_pages_per_block,
            ],
            (1, 0, 2, 3),
        ).copy()

    return PcpStreamingSchedule(
        req_id=req_id,
        actual_steps=actual_steps,
        global_actual_steps=np.array([actual_max_steps], dtype=np.int32),
        packed_schedule=packed_schedule,
        kv_page_indices=kv_page_indices,
        **fields,
    )


def _validate_lockstep_compact_requests(
    *,
    kv_lens: np.ndarray,
    q_lens: np.ndarray,
    token_owner_start_offsets: np.ndarray,
    request_absolute_query_start_offsets: np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool,
    kv_pages_per_block: int,
) -> None:
    if num_lanes != 1:
        raise NotImplementedError(
            "PCP streaming lockstep schedule only supports num_lanes == 1, "
            f"got {num_lanes}.")
    if not pad_kv_pages_to_pcp_group:
        raise NotImplementedError("PCP streaming lockstep schedule requires "
                                  "pad_kv_pages_to_pcp_group=True.")
    if kv_pages_per_block != 1:
        raise NotImplementedError(
            "PCP streaming lockstep schedule only supports "
            "kv_pages_per_block == 1.")
    _validate_page_interleave_size(page_size, interleave_size,
                                   "PCP streaming lockstep schedule")
    if bq_sz % interleave_size != 0:
        raise NotImplementedError(
            "PCP streaming lockstep schedule requires bq_sz to be a "
            f"multiple of interleave_size, got {bq_sz=} "
            f"{interleave_size=}.")
    if np.any(q_lens <= 0):
        raise NotImplementedError(
            "PCP streaming lockstep schedule requires every request to have "
            "positive q_len.")

    for req_idx, (kv_len, q_len, token_owner_start,
                  request_absolute_query_start) in enumerate(
                      zip(kv_lens, q_lens, token_owner_start_offsets,
                          request_absolute_query_start_offsets)):
        kv_len = int(kv_len)
        q_len = int(q_len)
        token_owner_start = int(token_owner_start)
        request_absolute_query_start = int(request_absolute_query_start)
        if token_owner_start < 0:
            raise ValueError("token_owner_start_offset must be non-negative.")
        if request_absolute_query_start < 0:
            raise ValueError(
                "request_absolute_query_start_offset must be non-negative.")
        if kv_len < request_absolute_query_start + q_len:
            raise ValueError("kv_lens must include all scheduled Q tokens.")
        required_local_pages = _cdiv(
            _pcp_kv_steps_for_token_count(
                kv_len,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
            ),
            pcp_size,
        )
        if required_local_pages > block_tables.shape[1]:
            raise ValueError("block_tables does not cover requested KV page: "
                             f"{req_idx=} {required_local_pages=} "
                             f"available={block_tables.shape[1]}.")


def _generate_pcp_streaming_schedule_lockstep_compact(
    kv_lens: list[int] | np.ndarray,
    cu_q_lens: list[int] | np.ndarray,
    token_owner_start_offsets: list[int] | np.ndarray,
    request_absolute_query_start_offsets: list[int] | np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool = False,
    pad_steps_to: int | None = None,
    kv_pages_per_block: int = 1,
) -> PcpStreamingSchedule:
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    cu_q_lens = np.asarray(cu_q_lens, dtype=np.int64)
    token_owner_start_offsets = np.asarray(token_owner_start_offsets,
                                           dtype=np.int64)
    request_absolute_query_start_offsets = np.asarray(
        request_absolute_query_start_offsets, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    num_reqs = _validate_inputs(
        kv_lens,
        cu_q_lens,
        token_owner_start_offsets,
        request_absolute_query_start_offsets,
        block_tables,
        page_size,
        pcp_size,
        interleave_size,
        num_lanes,
        bq_sz,
        kv_pages_per_block,
    )
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    _validate_lockstep_compact_requests(
        kv_lens=kv_lens[:num_reqs],
        q_lens=q_lens,
        token_owner_start_offsets=token_owner_start_offsets[:num_reqs],
        request_absolute_query_start_offsets=(
            request_absolute_query_start_offsets[:num_reqs]),
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=bq_sz,
        pad_kv_pages_to_pcp_group=pad_kv_pages_to_pcp_group,
        kv_pages_per_block=kv_pages_per_block,
    )

    tile_plans = []
    actual_max_steps = 0
    rank_local_q_offsets = np.zeros(pcp_size, dtype=np.int64)
    for req_idx in range(num_reqs):
        q_len = int(q_lens[req_idx])
        token_owner_start = int(token_owner_start_offsets[req_idx])
        request_absolute_query_start = int(
            request_absolute_query_start_offsets[req_idx])
        kv_len = int(kv_lens[req_idx])
        num_kv_pages = _pcp_kv_steps_for_token_count(
            kv_len,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )

        rank_tiles, rank_local_q_offsets = _build_pcp_rank_query_tiles(
            q_len,
            token_owner_start,
            request_absolute_query_start,
            rank_local_q_offsets,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            bq_sz=bq_sz,
        )

        num_q_tiles = max((len(tiles) for tiles in rank_tiles), default=0)
        for tile_idx in range(num_q_tiles):
            effective_kv_pages_by_rank = []
            q_tiles_by_rank = []
            for consumer_rank in range(pcp_size):
                if tile_idx < len(rank_tiles[consumer_rank]):
                    q_global_start, q_tile_size, q_hbm_offset = rank_tiles[
                        consumer_rank][tile_idx]
                    q_global_last = _q_global_last_for_strided_tile(
                        q_global_start,
                        q_tile_size,
                        pcp_size=pcp_size,
                        interleave_size=interleave_size)
                    effective_kv_pages = min(
                        num_kv_pages,
                        _pcp_kv_steps_for_last_pos(
                            q_global_last,
                            page_size=page_size,
                            pcp_size=pcp_size,
                            interleave_size=interleave_size,
                        ),
                    )
                    q_tiles_by_rank.append(
                        (q_global_start, q_tile_size, q_hbm_offset))
                else:
                    effective_kv_pages = 0
                    q_tiles_by_rank.append((0, 0, 0))
                effective_kv_pages_by_rank.append(effective_kv_pages)

            max_effective_kv_pages = max(effective_kv_pages_by_rank)
            scheduled_steps = _cdiv(max_effective_kv_pages,
                                    pcp_size) * pcp_size
            if scheduled_steps == 0:
                continue
            tile_plans.append((
                actual_max_steps,
                scheduled_steps,
                req_idx,
                kv_len,
                tuple(q_tiles_by_rank),
                tuple(effective_kv_pages_by_rank),
            ))
            actual_max_steps += scheduled_steps

    max_steps = actual_max_steps
    if pad_steps_to is not None:
        pad_steps_to = int(pad_steps_to)
        if pad_steps_to < actual_max_steps:
            raise ValueError(
                "pad_steps_to must be >= generated schedule steps: "
                f"{pad_steps_to=} {actual_max_steps=}.")
        max_steps = pad_steps_to

    packed_schedule = np.zeros(
        (max_steps, pcp_size, num_lanes, ScheduleField.PACKED_NUM_FIELDS),
        dtype=np.int32)
    packed_schedule[..., ScheduleField.REQ_ID] = -1
    packed_schedule[..., ScheduleField.KV_PAGE_RANK] = -1
    packed_schedule[..., ScheduleField.KV_PAGE_IDX] = -1

    for (start_step, scheduled_steps, req_idx, kv_len, q_tiles_by_rank,
         effective_kv_pages_by_rank) in tile_plans:
        for step_offset in range(scheduled_steps):
            src_rank = step_offset % pcp_size
            global_page = step_offset
            local_page_idx = global_page // pcp_size
            kv_global_start = _pcp_local_page_global_start(
                local_page_idx,
                src_rank,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
            )
            for consumer_rank in range(pcp_size):
                q_global_start, q_tile_size, q_hbm_offset = q_tiles_by_rank[
                    consumer_rank]
                effective_kv_pages = effective_kv_pages_by_rank[consumer_rank]
                valid_page = q_tile_size > 0 and global_page < effective_kv_pages
                kv_valid_len = 0
                kv_page_idx = 0
                if valid_page:
                    kv_page_idx = int(block_tables[req_idx, local_page_idx])
                    kv_valid_len = _pcp_local_page_valid_len(
                        kv_len,
                        local_page_idx,
                        src_rank,
                        page_size=page_size,
                        pcp_size=pcp_size,
                        interleave_size=interleave_size,
                    )

                row = packed_schedule[start_step + step_offset, consumer_rank,
                                      0, :]
                row[ScheduleField.REQ_ID] = req_idx if valid_page else -1
                row[ScheduleField.KV_PAGE_RANK] = src_rank
                row[ScheduleField.KV_PAGE_IDX] = kv_page_idx
                row[ScheduleField.IS_FIRST_KV] = int(valid_page
                                                     and global_page == 0)
                row[ScheduleField.IS_LAST_KV] = int(
                    valid_page and global_page == effective_kv_pages - 1)
                row[ScheduleField.LOAD_Q] = row[ScheduleField.IS_FIRST_KV]
                row[ScheduleField.Q_GLOBAL_START] = q_global_start
                row[ScheduleField.KV_GLOBAL_START] = kv_global_start
                row[ScheduleField.KV_VALID_LEN] = kv_valid_len
                row[ScheduleField.Q_HBM_OFFSET] = q_hbm_offset
                row[ScheduleField.Q_TILE_SIZE] = q_tile_size
                row[ScheduleField.O_HBM_OFFSET] = q_hbm_offset

    fields = {
        name: _packed_field(packed_schedule,
                            getattr(ScheduleField, name.upper()))
        for name in _PACKED_FIELD_NAMES[1:]
    }
    req_id = _packed_field(packed_schedule, ScheduleField.REQ_ID)
    actual_steps = np.full(pcp_size, actual_max_steps, dtype=np.int32)

    return PcpStreamingSchedule(
        req_id=req_id,
        actual_steps=actual_steps,
        global_actual_steps=np.array([actual_max_steps], dtype=np.int32),
        packed_schedule=packed_schedule,
        kv_page_indices=None,
        **fields,
    )


def _generate_pcp_streaming_schedule_vectorized_aligned(
    kv_lens: list[int] | np.ndarray,
    cu_q_lens: list[int] | np.ndarray,
    token_owner_start_offsets: list[int] | np.ndarray,
    request_absolute_query_start_offsets: list[int] | np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool = False,
    pad_steps_to: int | None = None,
    kv_pages_per_block: int = 1,
) -> PcpStreamingSchedule:
    cu_q_lens_np = np.asarray(cu_q_lens, dtype=np.int64)
    q_lens = cu_q_lens_np[1:] - cu_q_lens_np[:-1]
    active = np.flatnonzero(q_lens > 0)
    if active.size == 1 and int(active[0]) == 0:
        return _generate_pcp_streaming_schedule_single_aligned(
            kv_lens=kv_lens,
            cu_q_lens=cu_q_lens,
            token_owner_start_offsets=token_owner_start_offsets,
            request_absolute_query_start_offsets=(
                request_absolute_query_start_offsets),
            block_tables=block_tables,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            num_lanes=num_lanes,
            bq_sz=bq_sz,
            pad_kv_pages_to_pcp_group=pad_kv_pages_to_pcp_group,
            pad_steps_to=pad_steps_to,
            kv_pages_per_block=kv_pages_per_block,
        )
    return _generate_pcp_streaming_schedule_lockstep_compact(
        kv_lens=kv_lens,
        cu_q_lens=cu_q_lens,
        token_owner_start_offsets=token_owner_start_offsets,
        request_absolute_query_start_offsets=(
            request_absolute_query_start_offsets),
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=bq_sz,
        pad_kv_pages_to_pcp_group=pad_kv_pages_to_pcp_group,
        pad_steps_to=pad_steps_to,
        kv_pages_per_block=kv_pages_per_block,
    )


def generate_pcp_streaming_schedule(
    kv_lens: list[int] | np.ndarray,
    cu_q_lens: list[int] | np.ndarray,
    q_start_offsets: list[int] | np.ndarray,
    block_tables: np.ndarray,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    num_lanes: int,
    bq_sz: int,
    pad_kv_pages_to_pcp_group: bool = False,
    pad_steps_to: int | None = None,
    kv_pages_per_block: int = 1,
) -> PcpStreamingSchedule:
    """Generate the supported vectorized PCP streaming schedule.

    The production path is intentionally fail-closed: one lane and PCP
    page-group padding. Multi-request chunks use a compact lockstep schedule:
    each rank may contribute arbitrary-sized local Q tiles, while all ranks
    still stream the same KV source page at each ring step.
    """
    return _generate_pcp_streaming_schedule_vectorized_aligned(
        kv_lens=kv_lens,
        cu_q_lens=cu_q_lens,
        token_owner_start_offsets=q_start_offsets,
        request_absolute_query_start_offsets=q_start_offsets,
        block_tables=block_tables,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        num_lanes=num_lanes,
        bq_sz=bq_sz,
        pad_kv_pages_to_pcp_group=pad_kv_pages_to_pcp_group,
        pad_steps_to=pad_steps_to,
        kv_pages_per_block=kv_pages_per_block,
    )


def _validate_pcp_streaming_schedule_ring_source(
    schedule: PcpStreamingSchedule, ) -> None:
    global_actual_steps = np.asarray(schedule.global_actual_steps)
    if global_actual_steps.ndim != 1 or global_actual_steps.size != 1:
        raise ValueError(
            "PCP streaming schedule must have one global_actual_steps value.")
    active_steps = int(global_actual_steps[0])
    if active_steps % schedule.pcp_size != 0:
        raise ValueError("PCP streaming schedule steps must be padded to a "
                         "PCP page group.")

    for step in range(active_steps):
        expected_source_rank = step % schedule.pcp_size
        for lane in range(schedule.num_lanes):
            source_page = None
            for consumer_rank in range(schedule.pcp_size):
                req_id = int(schedule.req_id[consumer_rank, step, lane])
                if req_id == -1:
                    continue
                kv_page_rank = int(schedule.kv_page_rank[consumer_rank, step,
                                                         lane])
                if kv_page_rank != expected_source_rank:
                    raise ValueError(
                        "kv_page_rank must follow ring source rank.")
                candidate = (
                    req_id,
                    kv_page_rank,
                    int(schedule.kv_page_idx[consumer_rank, step, lane]),
                    int(schedule.kv_global_start[consumer_rank, step, lane]),
                    int(schedule.kv_valid_len[consumer_rank, step, lane]),
                )
                if source_page is None:
                    source_page = candidate
                elif candidate != source_page:
                    raise ValueError(
                        "source page changed across consumers in the same "
                        "ring step.")


def validate_pcp_streaming_schedule(
    schedule: PcpStreamingSchedule,
    *,
    require_ring_source_invariant: bool = False,
) -> None:
    """Validate per-lane online softmax state invariants."""
    for consumer_rank in range(schedule.pcp_size):
        for lane in range(schedule.num_lanes):
            in_tile = False
            cur_req_id = -1
            cur_q_offset = -1
            prev_kv_start = -1
            saw_last = False
            for step in range(int(schedule.actual_steps[consumer_rank])):
                req_id = int(schedule.req_id[consumer_rank, step, lane])
                if req_id == -1:
                    continue
                is_first = bool(schedule.is_first_kv[consumer_rank, step,
                                                     lane])
                is_last = bool(schedule.is_last_kv[consumer_rank, step, lane])
                if is_first:
                    if in_tile:
                        raise ValueError(
                            "new q_tile before previous is_last_kv.")
                    in_tile = True
                    saw_last = False
                    cur_req_id = req_id
                    cur_q_offset = int(schedule.q_hbm_offset[consumer_rank,
                                                             step, lane])
                    prev_kv_start = -1
                if not in_tile:
                    raise ValueError("entry outside q_tile boundary.")
                if req_id != cur_req_id:
                    raise ValueError("req_id changed inside q_tile.")
                if int(schedule.q_hbm_offset[consumer_rank, step,
                                             lane]) != cur_q_offset:
                    raise ValueError("q_hbm_offset changed inside q_tile.")
                kv_start = int(schedule.kv_global_start[consumer_rank, step,
                                                        lane])
                if kv_start <= prev_kv_start:
                    raise ValueError(
                        "kv_global_start must increase inside q_tile.")
                prev_kv_start = kv_start
                if is_last:
                    in_tile = False
                    saw_last = True
            if in_tile or not saw_last and schedule.actual_steps[
                    consumer_rank] > 0 and np.any(
                        schedule.req_id[consumer_rank, :, lane] != -1):
                raise ValueError("q_tile not closed at end of schedule.")
    if require_ring_source_invariant:
        _validate_pcp_streaming_schedule_ring_source(schedule)
