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
"""Host reference expansion for the production PCP runtime schedule ABI."""

import dataclasses

import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField, TilePlanField)


@dataclasses.dataclass(frozen=True)
class RuntimeScheduleReference:
    """Current/history runtime rows expanded from one production tile plan.

    Row tensors have shape
    ``[tile, group, source_rank, consumer_rank, packed_fields]``. Source rank
    is an implicit ring coordinate in the production kernel and therefore is
    intentionally not duplicated in ``RuntimeScheduleField``.
    """

    current_rows: np.ndarray
    history_rows: np.ndarray
    current_num_groups: np.ndarray
    history_num_groups: np.ndarray


def _rank_token_count_before(position: int, rank: int, *, pcp_size: int,
                             interleave_size: int) -> int:
    cycle = pcp_size * interleave_size
    full_cycles = position // cycle
    cycle_offset = position - full_cycles * cycle
    rank_start = rank * interleave_size
    partial = min(max(cycle_offset - rank_start, 0), interleave_size)
    return full_cycles * interleave_size + partial


def _local_page_valid_len(kv_len: int, local_page_idx: int, source_rank: int,
                          *, page_size: int, pcp_size: int,
                          interleave_size: int) -> int:
    cycle = pcp_size * interleave_size
    base = (local_page_idx * page_size * pcp_size +
            source_rank * interleave_size)
    delta = kv_len - base
    chunks_per_page = page_size // interleave_size
    full_chunks = min(chunks_per_page, max(delta, 0) // cycle)
    partial = min(interleave_size, max(delta - full_chunks * cycle, 0))
    valid_len = full_chunks * interleave_size
    if full_chunks < chunks_per_page:
        valid_len += partial
    return valid_len if kv_len > base else 0


def _new_rows(num_tiles: int, max_groups: int, pcp_size: int) -> np.ndarray:
    rows = np.zeros(
        (
            num_tiles,
            max_groups,
            pcp_size,
            pcp_size,
            RuntimeScheduleField.PACKED_NUM_FIELDS,
        ),
        dtype=np.int32,
    )
    rows[..., RuntimeScheduleField.REQ_ID] = -1
    return rows


def _set_common_fields(row: np.ndarray, plan: np.ndarray, consumer_rank: int,
                       *, req_id: int, kv_page_idx: int, kv_global_start: int,
                       kv_valid_len: int, kv_hbm_offset: int) -> None:
    row[RuntimeScheduleField.REQ_ID] = req_id
    row[RuntimeScheduleField.KV_PAGE_IDX] = kv_page_idx
    row[RuntimeScheduleField.Q_GLOBAL_START] = plan[
        consumer_rank, TilePlanField.Q_GLOBAL_START]
    row[RuntimeScheduleField.KV_GLOBAL_START] = kv_global_start
    row[RuntimeScheduleField.KV_VALID_LEN] = kv_valid_len
    row[RuntimeScheduleField.Q_HBM_OFFSET] = plan[consumer_rank,
                                                  TilePlanField.Q_HBM_OFFSET]
    row[RuntimeScheduleField.Q_TILE_SIZE] = plan[consumer_rank,
                                                 TilePlanField.Q_TILE_SIZE]
    row[RuntimeScheduleField.KV_HBM_OFFSET] = kv_hbm_offset


def build_runtime_schedule_reference(
    tile_plan: np.ndarray,
    block_tables: np.ndarray,
    *,
    pcp_size: int,
    page_size: int,
    interleave_size: int,
) -> RuntimeScheduleReference:
    """Expand production tile metadata into the rows staged by Pallas."""
    tile_plan = np.asarray(tile_plan, dtype=np.int32)
    block_tables = np.asarray(block_tables, dtype=np.int32)
    if tile_plan.ndim != 3 or tile_plan.shape[1] != 1:
        raise ValueError("tile_plan must have shape [tiles, 1, fields].")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must be rank 2.")
    logical_fields = pcp_size * TilePlanField.NUM_FIELDS
    if tile_plan.shape[2] < logical_fields:
        raise ValueError("tile_plan does not cover all PCP rank fields.")
    if page_size <= 0 or pcp_size <= 0 or interleave_size <= 0:
        raise ValueError(
            "page_size, pcp_size, and interleave_size must be positive.")
    if interleave_size > page_size or page_size % interleave_size != 0:
        raise ValueError("page_size must be divisible by interleave_size.")

    logical = tile_plan[:,
                        0, :logical_fields].reshape(tile_plan.shape[0],
                                                    pcp_size,
                                                    TilePlanField.NUM_FIELDS)
    current_num_groups = logical[:, 0, TilePlanField.CURRENT_NUM_GROUPS].copy()
    history_num_groups = logical[:, 0, TilePlanField.HISTORY_NUM_GROUPS].copy()
    current_rows = _new_rows(logical.shape[0],
                             int(current_num_groups.max(initial=0)), pcp_size)
    history_rows = _new_rows(logical.shape[0],
                             int(history_num_groups.max(initial=0)), pcp_size)

    cycle = pcp_size * interleave_size
    virtual_page_size = page_size * pcp_size
    for tile_idx, plan in enumerate(logical):
        req_id = int(plan[0, TilePlanField.REQ_ID])
        token_owner_start = int(plan[0, TilePlanField.TOKEN_OWNER_START])
        absolute_start = int(plan[0,
                                  TilePlanField.REQUEST_ABSOLUTE_QUERY_START])
        current_effective_len = int(plan[0,
                                         TilePlanField.CURRENT_EFFECTIVE_LEN])
        capture_current_kv = bool(plan[0, TilePlanField.CAPTURE_CURRENT_KV])
        block_table = block_tables[req_id]
        history_boundary_present = absolute_start % virtual_page_size != 0

        for group_idx in range(int(current_num_groups[tile_idx])):
            is_history_boundary = history_boundary_present and group_idx == 0
            fresh_group_idx = max(group_idx - int(history_boundary_present), 0)
            cycle_start = (token_owner_start // cycle * cycle +
                           fresh_group_idx * cycle)
            token_owner_end = token_owner_start + current_effective_len

            for source_rank in range(pcp_size):
                source_chunk_start = cycle_start + source_rank * interleave_size
                overlap_start = max(source_chunk_start, token_owner_start)
                overlap_end = min(source_chunk_start + interleave_size,
                                  token_owner_end)
                fresh_valid_len = max(overlap_end - overlap_start, 0)
                fresh_global_start = (absolute_start + overlap_start -
                                      token_owner_start)
                source_rows_before_overlap = (_rank_token_count_before(
                    overlap_start,
                    source_rank,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                ) - _rank_token_count_before(
                    token_owner_start,
                    source_rank,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                ))
                fresh_hbm_offset = int(plan[source_rank,
                                            TilePlanField.CURRENT_KV_HBM_START]
                                       ) + source_rows_before_overlap

                boundary_local_page = absolute_start // virtual_page_size
                safe_boundary_page = min(max(boundary_local_page, 0),
                                         block_table.size - 1)
                boundary_page_idx = int(block_table[safe_boundary_page])
                boundary_global_start = (
                    boundary_local_page * virtual_page_size +
                    source_rank * interleave_size)
                boundary_valid_len = _local_page_valid_len(
                    absolute_start,
                    boundary_local_page,
                    source_rank,
                    page_size=page_size,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                )

                if is_history_boundary:
                    kv_valid_len = boundary_valid_len
                    kv_page_idx = boundary_page_idx
                    kv_global_start = boundary_global_start
                    kv_hbm_offset = -1
                else:
                    kv_valid_len = fresh_valid_len
                    kv_page_idx = 0
                    kv_global_start = fresh_global_start
                    kv_hbm_offset = fresh_hbm_offset
                scheduled_req_id = req_id if kv_valid_len > 0 else -1

                capture_len_0 = 0
                capture_len_1 = 0
                capture_owner_0 = -1
                capture_owner_1 = -1
                capture_offset_0 = 0
                capture_offset_1 = 0
                if (capture_current_kv and not is_history_boundary
                        and fresh_valid_len > 0):
                    capture_owner_0 = (fresh_global_start //
                                       interleave_size) % pcp_size
                    next_owner_boundary = (
                        fresh_global_start // interleave_size +
                        1) * interleave_size
                    capture_len_0 = min(
                        fresh_valid_len,
                        max(next_owner_boundary - fresh_global_start, 0),
                    )
                    capture_len_1 = fresh_valid_len - capture_len_0
                    capture_global_start_1 = (fresh_global_start +
                                              capture_len_0)
                    capture_owner_1 = (capture_global_start_1 //
                                       interleave_size) % pcp_size
                    capture_offset_0 = int(
                        plan[capture_owner_0,
                             TilePlanField.WRITEBACK_HBM_PREFIX]) + (
                                 _rank_token_count_before(
                                     fresh_global_start,
                                     capture_owner_0,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size,
                                 ) - _rank_token_count_before(
                                     absolute_start,
                                     capture_owner_0,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size,
                                 ))
                    capture_offset_1 = int(
                        plan[capture_owner_1,
                             TilePlanField.WRITEBACK_HBM_PREFIX]) + (
                                 _rank_token_count_before(
                                     capture_global_start_1,
                                     capture_owner_1,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size,
                                 ) - _rank_token_count_before(
                                     absolute_start,
                                     capture_owner_1,
                                     pcp_size=pcp_size,
                                     interleave_size=interleave_size,
                                 ))

                for consumer_rank in range(pcp_size):
                    row = current_rows[tile_idx, group_idx, source_rank,
                                       consumer_rank]
                    _set_common_fields(
                        row,
                        plan,
                        consumer_rank,
                        req_id=scheduled_req_id,
                        kv_page_idx=kv_page_idx,
                        kv_global_start=kv_global_start,
                        kv_valid_len=kv_valid_len,
                        kv_hbm_offset=kv_hbm_offset,
                    )
                    owns_0 = (capture_len_0 > 0
                              and capture_owner_0 == consumer_rank)
                    owns_1 = (capture_len_1 > 0
                              and capture_owner_1 == consumer_rank)
                    if owns_0:
                        row[RuntimeScheduleField.CAPTURE_SRC_OFFSET] = 0
                        row[RuntimeScheduleField.
                            CAPTURE_DST_OFFSET] = capture_offset_0
                        row[RuntimeScheduleField.CAPTURE_LEN] = (
                            capture_len_0 +
                            capture_len_1 if owns_1 else capture_len_0)
                    elif owns_1:
                        row[RuntimeScheduleField.
                            CAPTURE_SRC_OFFSET] = capture_len_0
                        row[RuntimeScheduleField.
                            CAPTURE_DST_OFFSET] = capture_offset_1
                        row[RuntimeScheduleField.CAPTURE_LEN] = capture_len_1

        for group_idx in range(int(history_num_groups[tile_idx])):
            local_page_idx = group_idx
            safe_local_page_idx = min(max(local_page_idx, 0),
                                      block_table.size - 1)
            page_idx = int(block_table[safe_local_page_idx])
            for source_rank in range(pcp_size):
                kv_global_start = (local_page_idx * virtual_page_size +
                                   source_rank * interleave_size)
                kv_valid_len = _local_page_valid_len(
                    absolute_start,
                    local_page_idx,
                    source_rank,
                    page_size=page_size,
                    pcp_size=pcp_size,
                    interleave_size=interleave_size,
                )
                scheduled_req_id = req_id if kv_valid_len > 0 else -1
                for consumer_rank in range(pcp_size):
                    row = history_rows[tile_idx, group_idx, source_rank,
                                       consumer_rank]
                    _set_common_fields(
                        row,
                        plan,
                        consumer_rank,
                        req_id=scheduled_req_id,
                        kv_page_idx=page_idx,
                        kv_global_start=kv_global_start,
                        kv_valid_len=kv_valid_len,
                        kv_hbm_offset=0,
                    )

    return RuntimeScheduleReference(
        current_rows=current_rows,
        history_rows=history_rows,
        current_num_groups=current_num_groups,
        history_num_groups=history_num_groups,
    )
