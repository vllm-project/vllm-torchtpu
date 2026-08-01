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
"""Test-only reference schedule generator for PCP streaming RPA."""

import dataclasses

import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    pcp_query_chunk_ranges
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    PcpStreamingSchedule, ScheduleField, pack_pcp_streaming_schedule_fields)


@dataclasses.dataclass(frozen=True)
class _Entry:
    req_id: int
    kv_page_rank: int
    kv_page_idx: int
    is_first_kv: int
    is_last_kv: int
    load_q: int
    q_global_start: int
    kv_global_start: int
    kv_valid_len: int
    q_hbm_offset: int
    q_tile_size: int
    o_hbm_offset: int
    kv_page_indices: tuple[int, ...] | None = None


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _q_global_last_for_strided_tile(q_global_start: int, q_tile_len: int, *,
                                    pcp_size: int,
                                    interleave_size: int) -> int:
    last_row = q_tile_len - 1
    return (q_global_start +
            (last_row // interleave_size) * pcp_size * interleave_size +
            (last_row % interleave_size))


def _iter_pcp_q_tiles(q_len: int, q_global_base: int, consumer_rank: int, *,
                      pcp_size: int, interleave_size: int, bq_sz: int):
    ranges = list(
        pcp_query_chunk_ranges(q_len, q_global_base, consumer_rank, pcp_size,
                               interleave_size))
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


def _validate_inputs(
    kv_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    q_start_offsets: np.ndarray,
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
            "kv_pages_per_block exceeds packed schedule capacity.")
    if page_size != interleave_size:
        raise NotImplementedError(
            "reference requires page_size == interleave.")
    if cu_q_lens.ndim != 1 or cu_q_lens.size == 0:
        raise ValueError("cu_q_lens must be a non-empty 1D array.")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be a 2D array.")

    num_reqs = int(cu_q_lens.size - 1)
    if kv_lens.size < num_reqs:
        raise ValueError("kv_lens must cover every request in cu_q_lens.")
    if q_start_offsets.size < num_reqs:
        raise ValueError("q_start_offsets must cover every request.")
    if block_tables.shape[0] < num_reqs:
        raise ValueError("block_tables must cover every request.")
    return num_reqs


def generate_pcp_streaming_schedule_reference(
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
    """Generate a test-only reference schedule without production fast paths."""
    kv_lens = np.asarray(kv_lens, dtype=np.int64)
    cu_q_lens = np.asarray(cu_q_lens, dtype=np.int64)
    q_start_offsets = np.asarray(q_start_offsets, dtype=np.int64)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    num_reqs = _validate_inputs(
        kv_lens,
        cu_q_lens,
        q_start_offsets,
        block_tables,
        page_size,
        pcp_size,
        interleave_size,
        num_lanes,
        bq_sz,
        kv_pages_per_block,
    )
    kv_pages_per_block = int(kv_pages_per_block)

    schedules: list[list[list[_Entry]]] = []
    actual_steps = np.zeros(pcp_size, dtype=np.int32)
    for consumer_rank in range(pcp_size):
        lane_entries: list[list[_Entry]] = [[] for _ in range(num_lanes)]
        lane_lengths = np.zeros(num_lanes, dtype=np.int64)
        rank_q_offset = 0

        for req_idx in range(num_reqs):
            q_len = int(cu_q_lens[req_idx + 1] - cu_q_lens[req_idx])
            if q_len <= 0:
                continue
            q_global_base = int(q_start_offsets[req_idx])
            kv_len = int(kv_lens[req_idx])
            if kv_len < q_global_base + q_len:
                raise ValueError(
                    "kv_lens must include all scheduled Q tokens.")
            num_kv_pages = _cdiv(kv_len, page_size)

            for q_global, tile_len in _iter_pcp_q_tiles(
                    q_len,
                    q_global_base,
                    consumer_rank,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                    bq_sz=bq_sz):
                q_global_last = _q_global_last_for_strided_tile(
                    q_global,
                    tile_len,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size)
                q_hbm_offset = rank_q_offset
                target_lane = int(np.argmin(lane_lengths))
                effective_kv_pages = min(num_kv_pages,
                                         q_global_last // page_size + 1)

                if kv_pages_per_block == 1:
                    scheduled_kv_pages = effective_kv_pages
                    if pad_kv_pages_to_pcp_group:
                        scheduled_kv_pages = _cdiv(effective_kv_pages,
                                                   pcp_size) * pcp_size
                    last_block = -1
                    last_store_src_rank = -1
                else:
                    block_span_pages = pcp_size * kv_pages_per_block
                    scheduled_kv_blocks = _cdiv(effective_kv_pages,
                                                block_span_pages)
                    scheduled_kv_pages = scheduled_kv_blocks * pcp_size
                    last_global_page = effective_kv_pages - 1
                    last_local_page = last_global_page // pcp_size
                    last_block = last_local_page // kv_pages_per_block
                    last_store_src_rank = 0
                    last_local_page_start = last_block * kv_pages_per_block
                    for candidate_src_rank in range(pcp_size):
                        for page_offset in range(kv_pages_per_block):
                            candidate_global_page = (
                                (last_local_page_start + page_offset) *
                                pcp_size + candidate_src_rank)
                            if candidate_global_page < effective_kv_pages:
                                last_store_src_rank = candidate_src_rank

                for kv_page_seq_idx in range(scheduled_kv_pages):
                    if kv_pages_per_block == 1:
                        src_rank = kv_page_seq_idx % pcp_size
                        local_page_start = kv_page_seq_idx // pcp_size
                    else:
                        src_rank = kv_page_seq_idx % pcp_size
                        local_page_start = (kv_page_seq_idx // pcp_size *
                                            kv_pages_per_block)
                    global_page = local_page_start * pcp_size + src_rank
                    global_token_start = global_page * page_size
                    valid_tokens = 0
                    page_indices = []
                    for page_offset in range(kv_pages_per_block):
                        page_global = (
                            (local_page_start + page_offset) * pcp_size +
                            src_rank)
                        if page_global < effective_kv_pages:
                            local_page_index = page_global // pcp_size
                            if local_page_index >= block_tables.shape[1]:
                                raise ValueError(
                                    "block_tables does not cover requested "
                                    "KV page.")
                            physical_page = int(block_tables[req_idx,
                                                             local_page_index])
                            page_valid = min(page_size,
                                             kv_len - page_global * page_size)
                            valid_tokens += max(page_valid, 0)
                        else:
                            physical_page = 0
                        page_indices.append(physical_page)
                    if valid_tokens == 0:
                        lane_entries[target_lane].append(
                            _Entry(
                                req_id=-1,
                                kv_page_rank=src_rank,
                                kv_page_idx=0,
                                is_first_kv=0,
                                is_last_kv=0,
                                load_q=0,
                                q_global_start=q_global,
                                kv_global_start=global_token_start,
                                kv_valid_len=0,
                                q_hbm_offset=q_hbm_offset,
                                q_tile_size=tile_len,
                                o_hbm_offset=q_hbm_offset,
                                kv_page_indices=tuple(page_indices),
                            ))
                        lane_lengths[target_lane] += 1
                        continue
                    if kv_pages_per_block == 1:
                        is_first = int(kv_page_seq_idx == 0)
                        is_last = int(kv_page_seq_idx == effective_kv_pages -
                                      1)
                    else:
                        cur_block = kv_page_seq_idx // pcp_size
                        is_first = int(cur_block == 0 and src_rank == 0)
                        is_last = int(cur_block == last_block
                                      and src_rank == last_store_src_rank)
                    lane_entries[target_lane].append(
                        _Entry(
                            req_id=req_idx,
                            kv_page_rank=src_rank,
                            kv_page_idx=page_indices[0],
                            is_first_kv=is_first,
                            is_last_kv=is_last,
                            load_q=is_first,
                            q_global_start=q_global,
                            kv_global_start=global_token_start,
                            kv_valid_len=valid_tokens,
                            q_hbm_offset=q_hbm_offset,
                            q_tile_size=tile_len,
                            o_hbm_offset=q_hbm_offset,
                            kv_page_indices=tuple(page_indices),
                        ))
                    lane_lengths[target_lane] += 1

                rank_q_offset += tile_len

        actual_steps[consumer_rank] = int(lane_lengths.max(initial=0))
        schedules.append(lane_entries)

    actual_max_steps = int(actual_steps.max(initial=0))
    max_steps = actual_max_steps
    if pad_steps_to is not None:
        pad_steps_to = int(pad_steps_to)
        if pad_steps_to < actual_max_steps:
            raise ValueError(
                "pad_steps_to must be >= generated schedule steps: "
                f"{pad_steps_to=} {actual_max_steps=}.")
        max_steps = pad_steps_to
    shape = (pcp_size, max_steps, num_lanes)

    req_id = np.full(shape, -1, dtype=np.int32)
    kv_page_indices = None
    if kv_pages_per_block > 1:
        kv_page_indices = np.zeros(shape + (kv_pages_per_block, ),
                                   dtype=np.int32)
    fields = {
        "kv_page_rank": np.full(shape, -1, dtype=np.int32),
        "kv_page_idx": np.full(shape, -1, dtype=np.int32),
        "is_first_kv": np.zeros(shape, dtype=np.int32),
        "is_last_kv": np.zeros(shape, dtype=np.int32),
        "load_q": np.zeros(shape, dtype=np.int32),
        "q_global_start": np.zeros(shape, dtype=np.int32),
        "kv_global_start": np.zeros(shape, dtype=np.int32),
        "kv_valid_len": np.zeros(shape, dtype=np.int32),
        "q_hbm_offset": np.zeros(shape, dtype=np.int32),
        "q_tile_size": np.zeros(shape, dtype=np.int32),
        "o_hbm_offset": np.zeros(shape, dtype=np.int32),
    }

    for consumer_rank, lane_entries in enumerate(schedules):
        for lane, entries in enumerate(lane_entries):
            for step, entry in enumerate(entries):
                req_id[consumer_rank, step, lane] = entry.req_id
                fields["kv_page_rank"][consumer_rank, step,
                                       lane] = entry.kv_page_rank
                fields["kv_page_idx"][consumer_rank, step,
                                      lane] = entry.kv_page_idx
                fields["is_first_kv"][consumer_rank, step,
                                      lane] = entry.is_first_kv
                fields["is_last_kv"][consumer_rank, step,
                                     lane] = entry.is_last_kv
                fields["load_q"][consumer_rank, step, lane] = entry.load_q
                fields["q_global_start"][consumer_rank, step,
                                         lane] = entry.q_global_start
                fields["kv_global_start"][consumer_rank, step,
                                          lane] = entry.kv_global_start
                fields["kv_valid_len"][consumer_rank, step,
                                       lane] = entry.kv_valid_len
                fields["q_hbm_offset"][consumer_rank, step,
                                       lane] = entry.q_hbm_offset
                fields["q_tile_size"][consumer_rank, step,
                                      lane] = entry.q_tile_size
                fields["o_hbm_offset"][consumer_rank, step,
                                       lane] = entry.o_hbm_offset
                if kv_page_indices is not None:
                    page_indices = entry.kv_page_indices
                    if page_indices is None:
                        page_indices = (entry.kv_page_idx, )
                    kv_page_indices[consumer_rank, step,
                                    lane, :len(page_indices)] = page_indices

    packed_schedule = pack_pcp_streaming_schedule_fields(req_id=req_id,
                                                         **fields)
    if kv_page_indices is not None:
        packed_schedule[
            ...,
            ScheduleField.
            KV_PAGE_INDICES_START:ScheduleField.KV_PAGE_INDICES_START +
            kv_pages_per_block,
        ] = np.transpose(kv_page_indices, (1, 0, 2, 3))

    return PcpStreamingSchedule(
        req_id=req_id,
        actual_steps=actual_steps,
        global_actual_steps=np.array([actual_max_steps], dtype=np.int32),
        packed_schedule=packed_schedule,
        kv_page_indices=kv_page_indices,
        **fields,
    )
