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
"""PCP streaming prefill RPA kernels.

This module currently contains the first production-shaped MVP kernels for the
RingAttention-style PCP path. They intentionally support only page-grouped
schedules. The packed KV-cache production path has a multi-head Pallas kernel
so one ring transfer is consumed by all local query heads instead of repeating
the same schedule and communication once per head pair.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa.utils import \
    get_dtype_packing
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField, build_pcp_streaming_schedule_inputs_from_metadata_jax)

P = jax.sharding.PartitionSpec
AXIS = "pcp"
PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE = 256
TPU_HBM_ROW_TILE = 16


def _consume_scheduled_kv_page(
    q_vmem_ref,
    kv_vmem_ref,
    sched_vmem_ref,
    slot,
    consumer_rank,
    lane,
    m,
    l_state,
    acc,
    *,
    sm_scale,
    pcp_size,
    kv_head_idx,
    kv_packing,
    packed_kv_cache,
    k_scale,
    v_scale,
):
    """Consume one scheduled KV page and update online-softmax state.

    The Q tile is compact in rank-local HBM, so this helper reconstructs each
    row's global query position from the schedule before applying the causal
    mask against the scheduled global KV page.
    """
    q = q_vmem_ref[...].astype(jnp.float32)
    if packed_kv_cache and kv_packing == 4:
        k_lane = (kv_head_idx * 2) % kv_packing
        v_lane = k_lane + 1
    else:
        k_lane = 0
        v_lane = 1
    k = kv_vmem_ref.at[slot, :, k_lane, :][...].astype(jnp.float32)
    v = kv_vmem_ref.at[slot, :, v_lane, :][...].astype(jnp.float32)
    if k_scale is not None:
        k = k * k_scale
    if v_scale is not None:
        v = v * v_scale

    q_global_start = sched_vmem_ref[consumer_rank, lane,
                                    ScheduleField.Q_GLOBAL_START]
    kv_global_start = sched_vmem_ref[consumer_rank, lane,
                                     ScheduleField.KV_GLOBAL_START]
    req_id = sched_vmem_ref[consumer_rank, lane, ScheduleField.REQ_ID]
    kv_valid_len = sched_vmem_ref[consumer_rank, lane,
                                  ScheduleField.KV_VALID_LEN]
    q_tile_size = sched_vmem_ref[consumer_rank, lane,
                                 ScheduleField.Q_TILE_SIZE]

    scores = jnp.matmul(q, k.T, preferred_element_type=jnp.float32) * sm_scale
    q_row = lax.broadcasted_iota(jnp.int32, scores.shape, 0)
    q_interleave = kv_vmem_ref.shape[1]
    q_chunk_idx = lax.div(q_row, q_interleave)
    q_chunk_offset = lax.rem(q_row, q_interleave)
    q_pos = (q_global_start + q_chunk_idx * pcp_size * q_interleave +
             q_chunk_offset)
    kv_pos = kv_global_start + lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    kv_valid = lax.broadcasted_iota(jnp.int32, scores.shape, 1) < kv_valid_len
    q_valid = lax.broadcasted_iota(jnp.int32, scores.shape, 0) < q_tile_size
    entry_valid = req_id != -1
    row_active = jnp.logical_and(entry_valid, q_valid[:, :1])
    mask = jnp.logical_and(jnp.logical_and(q_pos >= kv_pos, kv_valid), q_valid)
    scores = jnp.where(mask, scores, -jnp.inf)
    scores = jnp.where(row_active, scores, 0.0)

    m_curr = jnp.max(scores, axis=1, keepdims=True)
    m_next = jnp.where(row_active, jnp.maximum(m, m_curr), m)
    p = jnp.where(row_active,
                  jnp.exp(scores - jnp.broadcast_to(m_next, scores.shape)),
                  0.0)
    alpha = jnp.where(row_active, jnp.exp(m - m_next), 1.0)
    l_next = alpha * l_state + jnp.sum(p, axis=1, keepdims=True)
    pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
    acc_next = jnp.broadcast_to(alpha, acc.shape) * acc + pv
    return m_next, l_next, acc_next


def _consume_scheduled_kv_page_multi_head(
    q_vmem_ref,
    kv_vmem_ref,
    sched_vmem_ref,
    slot,
    consumer_rank,
    lane,
    m_states,
    l_states,
    acc_states,
    *,
    sm_scale,
    pcp_size,
    k_scale,
    v_scale,
):
    q = q_vmem_ref[...].astype(jnp.float32)

    q_global_start = sched_vmem_ref[consumer_rank, lane,
                                    ScheduleField.Q_GLOBAL_START]
    kv_global_start = sched_vmem_ref[consumer_rank, lane,
                                     ScheduleField.KV_GLOBAL_START]
    req_id = sched_vmem_ref[consumer_rank, lane, ScheduleField.REQ_ID]
    kv_valid_len = sched_vmem_ref[consumer_rank, lane,
                                  ScheduleField.KV_VALID_LEN]
    q_tile_size = sched_vmem_ref[consumer_rank, lane,
                                 ScheduleField.Q_TILE_SIZE]
    q_per_kv = q_vmem_ref.shape[2]
    flat_q_rows = q_vmem_ref.shape[0] * q_per_kv
    page_size = kv_vmem_ref.shape[2]
    kv_block_tokens = kv_vmem_ref.shape[1] * page_size
    next_m_states = []
    next_l_states = []
    next_acc_states = []

    for kv_head_idx in range(q_vmem_ref.shape[1]):
        q_head = q[:, kv_head_idx, :, :].reshape(flat_q_rows,
                                                 q_vmem_ref.shape[-1])
        k = kv_vmem_ref.at[slot, :, :, kv_head_idx,
                           0, :][...].astype(jnp.float32)
        v = kv_vmem_ref.at[slot, :, :, kv_head_idx,
                           1, :][...].astype(jnp.float32)
        if k_scale is not None:
            k = k * k_scale
        if v_scale is not None:
            v = v * v_scale
        k = k.reshape(kv_block_tokens, q_vmem_ref.shape[-1])
        v = v.reshape(kv_block_tokens, q_vmem_ref.shape[-1])

        m_head = m_states[kv_head_idx].reshape(flat_q_rows, 1)
        l_head = l_states[kv_head_idx].reshape(flat_q_rows, 1)
        head_dim = acc_states[kv_head_idx].shape[-1]
        acc_head = acc_states[kv_head_idx].reshape(flat_q_rows, head_dim)

        scores = (jnp.matmul(q_head, k.T, preferred_element_type=jnp.float32) *
                  sm_scale)
        q_row = lax.div(
            lax.broadcasted_iota(jnp.int32, scores.shape, 0),
            q_per_kv,
        )
        q_interleave = page_size
        q_chunk_idx = lax.div(q_row, q_interleave)
        q_chunk_offset = lax.rem(q_row, q_interleave)
        q_pos = (q_global_start + q_chunk_idx * pcp_size * q_interleave +
                 q_chunk_offset)
        kv_local_pos = lax.broadcasted_iota(jnp.int32, scores.shape, 1)
        kv_page_offset = lax.div(kv_local_pos, page_size)
        kv_token_offset = lax.rem(kv_local_pos, page_size)
        kv_pos = (kv_global_start + kv_page_offset * pcp_size * page_size +
                  kv_token_offset)
        kv_valid = kv_local_pos < kv_valid_len
        q_valid = q_row < q_tile_size
        entry_valid = req_id != -1
        row_active = jnp.logical_and(entry_valid, q_valid[:, :1])
        mask = jnp.logical_and(jnp.logical_and(q_pos >= kv_pos, kv_valid),
                               q_valid)
        scores = jnp.where(mask, scores, -jnp.inf)
        scores = jnp.where(row_active, scores, 0.0)

        m_curr = jnp.max(scores, axis=1, keepdims=True)
        m_next = jnp.where(row_active, jnp.maximum(m_head, m_curr), m_head)
        p = jnp.where(row_active,
                      jnp.exp(scores - jnp.broadcast_to(m_next, scores.shape)),
                      0.0)
        alpha = jnp.where(row_active, jnp.exp(m_head - m_next), 1.0)
        l_next = alpha * l_head + jnp.sum(p, axis=1, keepdims=True)
        pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
        acc_next = jnp.broadcast_to(alpha, acc_head.shape) * acc_head + pv

        next_m_states.append(m_next.reshape(q_vmem_ref.shape[0], q_per_kv, 1))
        next_l_states.append(l_next.reshape(q_vmem_ref.shape[0], q_per_kv, 1))
        next_acc_states.append(
            acc_next.reshape(q_vmem_ref.shape[0], q_per_kv, head_dim))

    return tuple(next_m_states), tuple(next_l_states), tuple(next_acc_states)


def _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sem, step):
    load_op = pltpu.make_async_copy(
        src_ref=packed_schedule_ref.at[step],
        dst_ref=sched_vmem_ref.at[:, :, :],
        sem=sem,
    )
    load_op.start()
    load_op.wait()


def _scheduled_q_tile_size(sched_vmem_ref, rank, lane):
    req_id = sched_vmem_ref[rank, lane, ScheduleField.REQ_ID]
    load_q = sched_vmem_ref[rank, lane, ScheduleField.LOAD_Q] != 0
    q_tile_size = sched_vmem_ref[rank, lane, ScheduleField.Q_TILE_SIZE]
    return jnp.where(jnp.logical_and(req_id != -1, load_q), q_tile_size, 0)


def _scheduled_output_tile_size(sched_vmem_ref, rank, lane, group_has_last):
    req_id = sched_vmem_ref[rank, lane, ScheduleField.REQ_ID]
    q_tile_size = sched_vmem_ref[rank, lane, ScheduleField.Q_TILE_SIZE]
    should_store = jnp.logical_and(req_id != -1, group_has_last)
    return jnp.where(should_store, q_tile_size, 0)


def _aligned_row_start(row):
    return pl.multiple_of(row - lax.rem(row, TPU_HBM_ROW_TILE),
                          TPU_HBM_ROW_TILE)


def _validate_q_block_size(q_block_size: int):
    if q_block_size <= 0:
        raise ValueError("q_block_size must be positive.")
    if q_block_size % TPU_HBM_ROW_TILE != 0:
        raise NotImplementedError(
            "page-group MVP requires q_block_size to be a multiple of "
            f"{TPU_HBM_ROW_TILE}.")


def _load_compact_q_tile_single_head(q_ref, q_vmem_ref, q_load_vmem_ref,
                                     local_dma_sem, q_hbm_offset, q_tile_size,
                                     q_block_size):
    q_vmem_ref[...] = jnp.zeros_like(q_vmem_ref)
    aligned_start = _aligned_row_start(q_hbm_offset)
    num_hbm_tiles = q_block_size // TPU_HBM_ROW_TILE + 1
    for tile_idx in range(num_hbm_tiles):
        hbm_start = pl.multiple_of(aligned_start + tile_idx * TPU_HBM_ROW_TILE,
                                   TPU_HBM_ROW_TILE)
        hbm_end = hbm_start + TPU_HBM_ROW_TILE
        tile_active = jnp.logical_and(
            q_tile_size > 0,
            jnp.logical_and(hbm_start < q_hbm_offset + q_tile_size, hbm_end
                            > q_hbm_offset),
        )

        @pl.when(tile_active)
        def _load_aligned_q_tile():
            q_load = pltpu.make_async_copy(
                src_ref=q_ref.at[
                    pl.ds(hbm_start, TPU_HBM_ROW_TILE),
                    :,
                ],
                dst_ref=q_load_vmem_ref.at[:, :],
                sem=local_dma_sem,
            )
            q_load.start()
            q_load.wait()
            q_rows = lax.broadcasted_iota(jnp.int32, q_vmem_ref.shape, 0)
            updated_q = q_vmem_ref[...]
            for row in range(TPU_HBM_ROW_TILE):
                dst_row = hbm_start + row - q_hbm_offset
                row_active = jnp.logical_and(q_rows == dst_row, q_rows
                                             < q_tile_size)
                updated_q = jnp.where(
                    row_active,
                    q_load_vmem_ref.at[row, :][...],
                    updated_q,
                )
            q_vmem_ref[...] = updated_q


def _load_compact_q_tile_multi_head(q_ref, q_vmem_ref, local_dma_sem,
                                    q_hbm_offset, q_tile_size):
    q_vmem_ref[...] = jnp.zeros_like(q_vmem_ref)

    @pl.when(q_tile_size > 0)
    def _load_q_tile():
        q_load = pltpu.make_async_copy(
            src_ref=q_ref.at[
                pl.ds(q_hbm_offset, q_tile_size),
                :,
                :,
                :,
            ],
            dst_ref=q_vmem_ref.at[
                pl.ds(0, q_tile_size),
                :,
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        q_load.start()
        q_load.wait()


def _store_compact_output_tile_single_head(o_ref, o_vmem_ref, o_store_vmem_ref,
                                           local_dma_sem, o_hbm_offset,
                                           q_tile_size, q_block_size):
    offset_aligned = lax.rem(o_hbm_offset, TPU_HBM_ROW_TILE) == 0
    direct_tile = jnp.logical_and(q_tile_size == q_block_size, offset_aligned)
    staged_tile = jnp.logical_and(q_tile_size > 0,
                                  jnp.logical_not(direct_tile))

    @pl.when(direct_tile)
    def _store_output_tile():
        safe_o_hbm_offset = pl.multiple_of(o_hbm_offset, TPU_HBM_ROW_TILE)
        store = pltpu.make_async_copy(
            src_ref=o_vmem_ref.at[
                pl.ds(0, q_block_size),
                :,
            ],
            dst_ref=o_ref.at[
                pl.ds(safe_o_hbm_offset, q_block_size),
                :,
            ],
            sem=local_dma_sem,
        )
        store.start()
        store.wait()

    @pl.when(staged_tile)
    def _store_staged_output_tile():
        aligned_start = _aligned_row_start(o_hbm_offset)
        num_hbm_tiles = q_block_size // TPU_HBM_ROW_TILE + 1
        for tile_idx in range(num_hbm_tiles):
            hbm_start = pl.multiple_of(
                aligned_start + tile_idx * TPU_HBM_ROW_TILE, TPU_HBM_ROW_TILE)
            hbm_end = hbm_start + TPU_HBM_ROW_TILE
            tile_active = jnp.logical_and(
                hbm_start < o_hbm_offset + q_tile_size,
                hbm_end > o_hbm_offset,
            )

            @pl.when(tile_active)
            def _store_aligned_output_tile():
                load_existing = pltpu.make_async_copy(
                    src_ref=o_ref.at[
                        pl.ds(hbm_start, TPU_HBM_ROW_TILE),
                        :,
                    ],
                    dst_ref=o_store_vmem_ref.at[:, :],
                    sem=local_dma_sem,
                )
                load_existing.start()
                load_existing.wait()
                store_rows = lax.broadcasted_iota(jnp.int32,
                                                  o_store_vmem_ref.shape, 0)
                updated_store = o_store_vmem_ref[...]
                for src_row in range(q_block_size):
                    row_active = jnp.logical_and(
                        hbm_start + store_rows == o_hbm_offset + src_row,
                        src_row < q_tile_size,
                    )
                    updated_store = jnp.where(
                        row_active,
                        o_vmem_ref.at[src_row, :][...],
                        updated_store,
                    )
                o_store_vmem_ref[...] = updated_store
                store = pltpu.make_async_copy(
                    src_ref=o_store_vmem_ref.at[:, :],
                    dst_ref=o_ref.at[
                        pl.ds(hbm_start, TPU_HBM_ROW_TILE),
                        :,
                    ],
                    sem=local_dma_sem,
                )
                store.start()
                store.wait()


def _store_compact_output_tile_multi_head(o_ref, o_vmem_ref, local_dma_sem,
                                          o_hbm_offset, q_tile_size):

    @pl.when(q_tile_size > 0)
    def _store_output_tile():
        store = pltpu.make_async_copy(
            src_ref=o_vmem_ref.at[
                pl.ds(0, q_tile_size),
                :,
                :,
                :,
            ],
            dst_ref=o_ref.at[
                pl.ds(o_hbm_offset, q_tile_size),
                :,
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        store.start()
        store.wait()


def _source_page_idx_from_staged_schedule(sched_vmem_ref,
                                          source_rank,
                                          lane,
                                          pcp_size,
                                          page_offset=0):
    page_idx = jnp.array(0, dtype=jnp.int32)
    for consumer_rank in range(pcp_size):
        req_id = sched_vmem_ref[consumer_rank, lane, ScheduleField.REQ_ID]
        kv_page_rank = sched_vmem_ref[consumer_rank, lane,
                                      ScheduleField.KV_PAGE_RANK]
        candidate = jnp.logical_and(req_id != -1, kv_page_rank == source_rank)
        if page_offset == 0:
            page_field = ScheduleField.KV_PAGE_IDX
        else:
            page_field = ScheduleField.KV_PAGE_INDICES_START + page_offset
        page_idx = jnp.where(candidate, sched_vmem_ref[consumer_rank, lane,
                                                       page_field], page_idx)
    return page_idx


def _load_local_kv_page(kv_cache_ref, kv_vmem_ref, sem, local_page_idx, *,
                        kv_head_idx, kv_packing, packed_kv_cache):
    if packed_kv_cache:
        if kv_packing in (2, 4):
            packed_idx = (kv_head_idx * 2) // kv_packing
            load = pltpu.make_async_copy(
                src_ref=kv_cache_ref.at[
                    0,
                    local_page_idx,
                    :,
                    packed_idx,
                    :,
                    :,
                ],
                dst_ref=kv_vmem_ref.at[0],
                sem=sem,
            )
            load.start()
            load.wait()
        elif kv_packing == 1:
            for kv_pair_idx in range(2):
                linear_idx = kv_head_idx * 2 + kv_pair_idx
                packed_idx = linear_idx // kv_packing
                pack_lane = linear_idx % kv_packing
                load = pltpu.make_async_copy(
                    src_ref=kv_cache_ref.at[
                        0,
                        local_page_idx,
                        :,
                        packed_idx,
                        pack_lane,
                        :,
                    ],
                    dst_ref=kv_vmem_ref.at[0, :, kv_pair_idx, :],
                    sem=sem,
                )
                load.start()
                load.wait()
        else:
            raise NotImplementedError(
                "packed PCP streaming KV loads currently support "
                "kv_packing in {1, 2, 4}.")
    else:
        load = pltpu.make_async_copy(
            src_ref=kv_cache_ref.at[0, local_page_idx, :, 0, :, :],
            dst_ref=kv_vmem_ref.at[0],
            sem=sem,
        )
        load.start()
        load.wait()


def _load_local_kv_page_block_all_heads_packed(kv_cache_ref, kv_vmem_ref, sem,
                                               sched_vmem_ref, source_rank,
                                               lane, *, pcp_size, kv_heads,
                                               kv_packing, kv_pages_per_block):
    if kv_packing != 2:
        raise NotImplementedError(
            "multi-head packed PCP streaming loads currently require "
            "kv_packing=2.")
    for page_offset in range(kv_pages_per_block):
        local_page_idx = _source_page_idx_from_staged_schedule(
            sched_vmem_ref,
            source_rank,
            lane,
            pcp_size,
            page_offset=page_offset,
        )
        load = pltpu.make_async_copy(
            src_ref=kv_cache_ref.at[
                0,
                local_page_idx,
                :,
                pl.ds(0, kv_heads),
                :,
                :,
            ],
            dst_ref=kv_vmem_ref.at[0, page_offset],
            sem=sem,
        )
        load.start()
        load.wait()


def _mesh_device_id(mesh_axis_names, pcp_axis_name, pcp_rank):
    return tuple(
        pcp_rank if axis_name == pcp_axis_name else lax.axis_index(axis_name)
        for axis_name in mesh_axis_names)


def _pcp_streaming_attention_page_groups_kernel(
    active_page_groups_ref,
    q_ref,
    kv_cache_ref,
    packed_schedule_ref,
    o_ref,
    sched_dma_sem,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    remote_sync_sems,
    group_slot_free_sems,
    sched_vmem_ref,
    q_vmem_ref,
    kv_vmem_ref,
    o_vmem_ref,
    q_load_vmem_ref,
    o_store_vmem_ref,
    *,
    pcp_size,
    num_lanes,
    q_block_size,
    num_page_groups,
    num_q_blocks,
    sm_scale,
    kv_head_idx,
    kv_packing,
    packed_kv_cache,
    k_scale,
    v_scale,
    mesh_axis_names,
    pcp_axis_name,
):
    """Single-head Pallas body for ring-streaming PCP attention.

    One program instance runs on every rank in the PCP mesh. Each rank computes
    output for its local compact Q rows and streams local KV pages to the next
    PCP rank while receiving pages from the previous rank.
    """
    my_id = lax.axis_index(pcp_axis_name)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)
    next_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, next_rank)
    prev_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, prev_rank)

    o_vmem_ref[...] = jnp.zeros_like(o_vmem_ref)
    for block_idx in range(num_q_blocks):
        zero_store = pltpu.make_async_copy(
            src_ref=o_vmem_ref.at[:, :],
            dst_ref=o_ref.at[
                pl.ds(block_idx * q_block_size, q_block_size),
                :,
            ],
            sem=local_dma_sem,
        )
        zero_store.start()
        zero_store.wait()

    for lane in range(num_lanes):
        # Online-softmax state for the current compact Q tile. A schedule row
        # with IS_FIRST_KV resets these values before consuming the first KV
        # page for that tile.
        m = jnp.full((q_block_size, 1), -jnp.inf, dtype=jnp.float32)
        l_state = jnp.zeros((q_block_size, 1), dtype=jnp.float32)
        acc = jnp.zeros((q_block_size, q_vmem_ref.shape[1]), dtype=jnp.float32)

        def _page_group_loop(group_idx, carry):
            m, l_state, acc = carry
            group_start = group_idx * pcp_size

            # Load the first step of the page group to discover this rank's Q
            # tile and whether this group starts a fresh softmax reduction.
            _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                sched_dma_sem, group_start)
            q_hbm_offset = sched_vmem_ref[my_id, lane,
                                          ScheduleField.Q_HBM_OFFSET]
            group_is_first_kv = sched_vmem_ref[my_id, lane,
                                               ScheduleField.IS_FIRST_KV] != 0
            group_load_q = sched_vmem_ref[my_id, lane,
                                          ScheduleField.LOAD_Q] != 0
            q_tile_size = _scheduled_q_tile_size(sched_vmem_ref, my_id, lane)

            @pl.when(group_load_q)
            def _load_q_tile():
                # Q tiles are compact in rank-local HBM, but may not be aligned
                # to TPU HBM row boundaries. The loader handles aligned HBM
                # row DMA and scatters valid rows into the compact VMEM tile.
                _load_compact_q_tile_single_head(
                    q_ref,
                    q_vmem_ref,
                    q_load_vmem_ref,
                    local_dma_sem,
                    q_hbm_offset,
                    q_tile_size,
                    q_block_size,
                )

            # Stage this rank's local KV page before the ring starts. The
            # schedule is replicated, so each rank can find the page it owns by
            # scanning the staged rows for KV_PAGE_RANK == my_id.
            _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                sched_dma_sem, group_start + my_id)
            local_page_idx = _source_page_idx_from_staged_schedule(
                sched_vmem_ref, my_id, lane, pcp_size)
            _load_local_kv_page(
                kv_cache_ref,
                kv_vmem_ref,
                local_dma_sem,
                local_page_idx,
                kv_head_idx=kv_head_idx,
                kv_packing=kv_packing,
                packed_kv_cache=packed_kv_cache,
            )

            @pl.when(group_idx > 0)
            def _wait_previous_group_slot_free():
                pl.semaphore_wait(group_slot_free_sems.at[lane], 1)

            m = jnp.where(group_is_first_kv, jnp.full_like(m, -jnp.inf), m)
            l_state = jnp.where(group_is_first_kv, jnp.zeros_like(l_state),
                                l_state)
            acc = jnp.where(group_is_first_kv, jnp.zeros_like(acc), acc)
            group_has_last = jnp.array(False)

            for round_idx in range(pcp_size):
                curr_slot = round_idx % 2
                next_slot = 1 - curr_slot
                src_rank = lax.rem(my_id + pcp_size - round_idx, pcp_size)

                # Each ring round consumes the KV page whose schedule row
                # corresponds to the source rank currently resident in
                # curr_slot.
                _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                    sched_dma_sem, group_start + src_rank)
                group_has_last = jnp.logical_or(
                    group_has_last,
                    sched_vmem_ref[my_id, lane, ScheduleField.IS_LAST_KV] != 0,
                )

                if round_idx > 0:
                    wait_count = 1 if round_idx == pcp_size - 1 else 2
                    pl.semaphore_wait(remote_sync_sems.at[lane, round_idx - 1],
                                      wait_count)

                if round_idx < pcp_size - 1:
                    # Send the current KV slot to the next rank while this
                    # rank computes against it. The next round consumes the
                    # slot received from the previous rank.
                    remote_op = pltpu.make_async_remote_copy(
                        src_ref=kv_vmem_ref.at[curr_slot],
                        dst_ref=kv_vmem_ref.at[next_slot],
                        send_sem=remote_send_sems.at[lane, round_idx],
                        recv_sem=remote_recv_sems.at[lane, round_idx],
                        device_id=next_device_id,
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    remote_op.start()

                m, l_state, acc = _consume_scheduled_kv_page(
                    q_vmem_ref,
                    kv_vmem_ref,
                    sched_vmem_ref,
                    curr_slot,
                    my_id,
                    lane,
                    m,
                    l_state,
                    acc,
                    sm_scale=sm_scale,
                    pcp_size=pcp_size,
                    kv_head_idx=kv_head_idx,
                    kv_packing=kv_packing,
                    packed_kv_cache=packed_kv_cache,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

                if round_idx < pcp_size - 1:
                    remote_op.wait()
                    pl.semaphore_signal(
                        remote_sync_sems.at[lane, round_idx],
                        1,
                        device_id=next_device_id,
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    if round_idx < pcp_size - 2:
                        pl.semaphore_signal(
                            remote_sync_sems.at[lane, round_idx],
                            1,
                            device_id=prev_device_id,
                            device_id_type=pl.DeviceIdType.MESH,
                        )

            @pl.when(group_idx < active_page_groups - 1)
            def _signal_next_group_slot_free():
                pl.semaphore_signal(
                    group_slot_free_sems.at[lane],
                    1,
                    device_id=prev_device_id,
                    device_id_type=pl.DeviceIdType.MESH,
                )

            l_broadcast = jnp.broadcast_to(l_state, acc.shape)
            o_vmem_ref[...] = jnp.where(l_broadcast > 0, acc / l_broadcast,
                                        0.0).astype(o_vmem_ref.dtype)
            _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                sched_dma_sem, group_start)
            o_hbm_offset = sched_vmem_ref[my_id, lane,
                                          ScheduleField.O_HBM_OFFSET]
            store_q_tile_size = _scheduled_output_tile_size(
                sched_vmem_ref, my_id, lane, group_has_last)
            _store_compact_output_tile_single_head(
                o_ref,
                o_vmem_ref,
                o_store_vmem_ref,
                local_dma_sem,
                o_hbm_offset,
                store_q_tile_size,
                q_block_size,
            )

            return m, l_state, acc

        active_page_groups = jnp.minimum(active_page_groups_ref[0],
                                         num_page_groups)
        m, l_state, acc = lax.fori_loop(
            0,
            active_page_groups,
            _page_group_loop,
            (m, l_state, acc),
            unroll=False,
        )


def _pcp_streaming_attention_page_groups_multi_head_kernel(
    active_page_groups_ref,
    q_ref,
    kv_cache_ref,
    packed_schedule_ref,
    o_ref,
    sched_dma_sem,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    remote_sync_sems,
    group_slot_free_sems,
    sched_vmem_ref,
    q_vmem_ref,
    kv_vmem_ref,
    o_vmem_ref,
    *,
    pcp_size,
    num_lanes,
    q_block_size,
    num_page_groups,
    num_q_blocks,
    sm_scale,
    kv_heads,
    kv_packing,
    kv_pages_per_block,
    k_scale,
    v_scale,
    mesh_axis_names,
    pcp_axis_name,
):
    """Multi-head Pallas body for packed-KV ring-streaming PCP attention.

    This follows the same page-group ring as the single-head body, but consumes
    the packed batched-RPA KV-cache layout and computes all local KV heads from
    each streamed KV transfer.
    """
    my_id = lax.axis_index(pcp_axis_name)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)
    next_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, next_rank)
    prev_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, prev_rank)

    o_vmem_ref[...] = jnp.zeros_like(o_vmem_ref)
    for block_idx in range(num_q_blocks):
        zero_store = pltpu.make_async_copy(
            src_ref=o_vmem_ref.at[:, :, :, :],
            dst_ref=o_ref.at[
                pl.ds(block_idx * q_block_size, q_block_size),
                :,
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        zero_store.start()
        zero_store.wait()

    for lane in range(num_lanes):
        # Keep independent online-softmax state per KV head. q_per_kv rows are
        # flattened during matmul and restored before returning.
        m_states = tuple(
            jnp.full((q_block_size, q_vmem_ref.shape[2], 1),
                     -jnp.inf,
                     dtype=jnp.float32) for _ in range(kv_heads))
        l_states = tuple(
            jnp.zeros((q_block_size, q_vmem_ref.shape[2], 1),
                      dtype=jnp.float32) for _ in range(kv_heads))
        acc_states = tuple(
            jnp.zeros((q_block_size, q_vmem_ref.shape[2], q_vmem_ref.shape[3]),
                      dtype=jnp.float32) for _ in range(kv_heads))

        def _page_group_loop(group_idx, carry):
            m_states, l_states, acc_states = carry
            group_start = group_idx * pcp_size

            _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                sched_dma_sem, group_start)
            q_hbm_offset = sched_vmem_ref[my_id, lane,
                                          ScheduleField.Q_HBM_OFFSET]
            group_is_first_kv = sched_vmem_ref[my_id, lane,
                                               ScheduleField.IS_FIRST_KV] != 0
            group_req_id = sched_vmem_ref[my_id, lane, ScheduleField.REQ_ID]
            group_o_hbm_offset = sched_vmem_ref[my_id, lane,
                                                ScheduleField.O_HBM_OFFSET]
            group_q_tile_size = sched_vmem_ref[my_id, lane,
                                               ScheduleField.Q_TILE_SIZE]
            group_load_q = sched_vmem_ref[my_id, lane,
                                          ScheduleField.LOAD_Q] != 0
            q_tile_size = _scheduled_q_tile_size(sched_vmem_ref, my_id, lane)

            @pl.when(group_load_q)
            def _load_q_tile():
                _load_compact_q_tile_multi_head(
                    q_ref,
                    q_vmem_ref,
                    local_dma_sem,
                    q_hbm_offset,
                    q_tile_size,
                )

            _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                sched_dma_sem, group_start + my_id)
            _load_local_kv_page_block_all_heads_packed(
                kv_cache_ref,
                kv_vmem_ref,
                local_dma_sem,
                sched_vmem_ref,
                my_id,
                lane,
                pcp_size=pcp_size,
                kv_heads=kv_heads,
                kv_packing=kv_packing,
                kv_pages_per_block=kv_pages_per_block,
            )

            @pl.when(group_idx > 0)
            def _wait_previous_group_slot_free():
                pl.semaphore_wait(group_slot_free_sems.at[lane], 1)

            m_states = tuple(
                jnp.where(group_is_first_kv, jnp.full_like(m, -jnp.inf), m)
                for m in m_states)
            l_states = tuple(
                jnp.where(group_is_first_kv, jnp.zeros_like(l_state), l_state)
                for l_state in l_states)
            acc_states = tuple(
                jnp.where(group_is_first_kv, jnp.zeros_like(acc), acc)
                for acc in acc_states)
            group_has_last = jnp.array(False)

            for round_idx in range(pcp_size):
                curr_slot = round_idx % 2
                next_slot = 1 - curr_slot
                src_rank = lax.rem(my_id + pcp_size - round_idx, pcp_size)

                if round_idx > 0:
                    _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                        sched_dma_sem, group_start + src_rank)
                group_has_last = jnp.logical_or(
                    group_has_last,
                    sched_vmem_ref[my_id, lane, ScheduleField.IS_LAST_KV] != 0,
                )

                if round_idx > 0:
                    wait_count = 1 if round_idx == pcp_size - 1 else 2
                    pl.semaphore_wait(remote_sync_sems.at[lane, round_idx - 1],
                                      wait_count)

                if round_idx < pcp_size - 1:
                    remote_op = pltpu.make_async_remote_copy(
                        src_ref=kv_vmem_ref.at[curr_slot],
                        dst_ref=kv_vmem_ref.at[next_slot],
                        send_sem=remote_send_sems.at[lane, round_idx],
                        recv_sem=remote_recv_sems.at[lane, round_idx],
                        device_id=next_device_id,
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    remote_op.start()

                m_states, l_states, acc_states = (
                    _consume_scheduled_kv_page_multi_head(
                        q_vmem_ref,
                        kv_vmem_ref,
                        sched_vmem_ref,
                        curr_slot,
                        my_id,
                        lane,
                        m_states,
                        l_states,
                        acc_states,
                        sm_scale=sm_scale,
                        pcp_size=pcp_size,
                        k_scale=k_scale,
                        v_scale=v_scale,
                    ))

                if round_idx < pcp_size - 1:
                    remote_op.wait()
                    pl.semaphore_signal(
                        remote_sync_sems.at[lane, round_idx],
                        1,
                        device_id=next_device_id,
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    if round_idx < pcp_size - 2:
                        pl.semaphore_signal(
                            remote_sync_sems.at[lane, round_idx],
                            1,
                            device_id=prev_device_id,
                            device_id_type=pl.DeviceIdType.MESH,
                        )

            @pl.when(group_idx < active_page_groups - 1)
            def _signal_next_group_slot_free():
                pl.semaphore_signal(
                    group_slot_free_sems.at[lane],
                    1,
                    device_id=prev_device_id,
                    device_id_type=pl.DeviceIdType.MESH,
                )

            l_values = jnp.stack(l_states, axis=1)
            acc = jnp.stack(acc_states, axis=1)
            l_broadcast = jnp.broadcast_to(l_values, acc.shape)
            o_vmem_ref[...] = jnp.where(l_broadcast > 0, acc / l_broadcast,
                                        0.0).astype(o_vmem_ref.dtype)
            store_q_tile_size = jnp.where(
                jnp.logical_and(group_req_id != -1, group_has_last),
                group_q_tile_size,
                0,
            )
            _store_compact_output_tile_multi_head(
                o_ref,
                o_vmem_ref,
                local_dma_sem,
                group_o_hbm_offset,
                store_q_tile_size,
            )

            return m_states, l_states, acc_states

        active_page_groups = jnp.minimum(active_page_groups_ref[0],
                                         num_page_groups)
        m_states, l_states, acc_states = lax.fori_loop(
            0,
            active_page_groups,
            _page_group_loop,
            (m_states, l_states, acc_states),
            unroll=False,
        )


def _validate_page_group_inputs(q_by_rank, kv_cache_by_rank, packed_schedule,
                                pcp_size, q_block_size):
    if q_by_rank.ndim != 5:
        raise ValueError("q_by_rank must have shape "
                         "[pcp, local_tokens, kv_heads, q_per_kv, head_dim].")
    if kv_cache_by_rank.ndim != 6:
        raise ValueError("kv_cache_by_rank must have shape "
                         "[pcp, pages, page_size, kv_heads, 2, head_dim].")
    if packed_schedule.ndim != 4:
        raise ValueError("packed_schedule must have shape "
                         "[steps, pcp, lanes, packed_fields].")
    if q_by_rank.shape[0] != pcp_size or kv_cache_by_rank.shape[0] != pcp_size:
        raise ValueError("q_by_rank and kv_cache_by_rank must be sharded over "
                         "pcp_size ranks.")
    _validate_q_block_size(q_block_size)
    if q_by_rank.shape[1] % q_block_size != 0:
        raise NotImplementedError(
            "page-group MVP requires local_tokens to be a multiple of "
            "q_block_size.")
    if packed_schedule.shape[0] % pcp_size != 0:
        raise NotImplementedError(
            "page-group MVP requires schedule steps to be grouped in "
            "pcp_size-step ring groups.")
    if packed_schedule.shape[1] != pcp_size or packed_schedule.shape[2] <= 0:
        raise NotImplementedError("page-group MVP requires at least one lane.")
    if packed_schedule.shape[3] != ScheduleField.PACKED_NUM_FIELDS:
        raise ValueError("packed_schedule must use padded packed fields.")
    if q_by_rank.shape[2] != 1 or q_by_rank.shape[3] != 1:
        raise NotImplementedError(
            "page-group MVP supports kv_heads=1 and q_per_kv=1.")
    if kv_cache_by_rank.shape[3] != 1 or kv_cache_by_rank.shape[4] != 2:
        raise NotImplementedError(
            "page-group MVP expects KV cache shape [..., 1, 2, head_dim].")
    if q_by_rank.shape[-1] != kv_cache_by_rank.shape[-1]:
        raise ValueError("Q and KV head_dim must match.")
    if q_by_rank.shape[-1] % 128 != 0:
        raise NotImplementedError(
            "page-group MVP requires head_dim to be 128-aligned.")


def _validate_common_page_group_inputs(q_by_rank, kv_cache_by_rank,
                                       packed_schedule, pcp_size,
                                       q_block_size):
    if q_by_rank.ndim != 5:
        raise ValueError("q_by_rank must have shape "
                         "[pcp, local_tokens, kv_heads, q_per_kv, head_dim].")
    if kv_cache_by_rank.ndim != 6:
        raise ValueError("kv_cache_by_rank must have shape "
                         "[pcp, pages, page_size, kv_heads, 2, head_dim].")
    if packed_schedule.ndim != 4:
        raise ValueError("packed_schedule must have shape "
                         "[steps, pcp, lanes, packed_fields].")
    if q_by_rank.shape[0] != pcp_size or kv_cache_by_rank.shape[0] != pcp_size:
        raise ValueError("q_by_rank and kv_cache_by_rank must be sharded over "
                         "pcp_size ranks.")
    _validate_q_block_size(q_block_size)
    if q_by_rank.shape[1] % q_block_size != 0:
        raise NotImplementedError(
            "page-group MVP requires local_tokens to be a multiple of "
            "q_block_size.")
    if packed_schedule.shape[0] % pcp_size != 0:
        raise NotImplementedError(
            "page-group MVP requires schedule steps to be grouped in "
            "pcp_size-step ring groups.")
    if packed_schedule.shape[1] != pcp_size or packed_schedule.shape[2] <= 0:
        raise NotImplementedError("page-group MVP requires at least one lane.")
    if packed_schedule.shape[3] != ScheduleField.PACKED_NUM_FIELDS:
        raise ValueError("packed_schedule must use padded packed fields.")
    if kv_cache_by_rank.shape[3] != q_by_rank.shape[2]:
        raise ValueError("Q kv_heads and KV kv_heads must match.")
    if kv_cache_by_rank.shape[4] != 2:
        raise ValueError("KV cache must store K/V pair at axis 4.")
    if q_by_rank.shape[-1] != kv_cache_by_rank.shape[-1]:
        raise ValueError("Q and KV head_dim must match.")
    if q_by_rank.shape[-1] % 128 != 0:
        raise NotImplementedError(
            "page-group MVP requires head_dim to be 128-aligned.")


def _validate_common_page_group_local_inputs(q_local, kv_cache_local,
                                             packed_schedule, pcp_size,
                                             q_block_size):
    if q_local.ndim != 4:
        raise ValueError("q_local must have shape "
                         "[local_tokens, kv_heads, q_per_kv, head_dim].")
    if kv_cache_local.ndim != 5:
        raise ValueError("kv_cache_local must have shape "
                         "[pages, page_size, kv_heads, 2, head_dim].")
    if packed_schedule.ndim != 4:
        raise ValueError("packed_schedule must have shape "
                         "[steps, pcp, lanes, packed_fields].")
    _validate_q_block_size(q_block_size)
    if q_local.shape[0] % q_block_size != 0:
        raise NotImplementedError(
            "page-group MVP requires local_tokens to be a multiple of "
            "q_block_size.")
    if packed_schedule.shape[0] % pcp_size != 0:
        raise NotImplementedError(
            "page-group MVP requires schedule steps to be grouped in "
            "pcp_size-step ring groups.")
    if packed_schedule.shape[1] != pcp_size or packed_schedule.shape[2] <= 0:
        raise NotImplementedError("page-group MVP requires at least one lane.")
    if packed_schedule.shape[3] != ScheduleField.PACKED_NUM_FIELDS:
        raise ValueError("packed_schedule must use padded packed fields.")
    if kv_cache_local.shape[2] != q_local.shape[1]:
        raise ValueError("Q kv_heads and KV kv_heads must match.")
    if kv_cache_local.shape[3] != 2:
        raise ValueError("KV cache must store K/V pair at axis 3.")
    if q_local.shape[-1] != kv_cache_local.shape[-1]:
        raise ValueError("Q and KV head_dim must match.")
    if q_local.shape[-1] % 128 != 0:
        raise NotImplementedError(
            "page-group MVP requires head_dim to be 128-aligned.")


def _validate_packed_page_group_local_inputs(q_local, kv_cache_local,
                                             packed_schedule, pcp_size,
                                             q_block_size):
    if q_local.ndim != 4:
        raise ValueError("q_local must have shape "
                         "[local_tokens, kv_heads, q_per_kv, head_dim].")
    if kv_cache_local.ndim != 5:
        raise ValueError("kv_cache_local must have shape "
                         "[pages, page_size, packed_kv_heads_x2, "
                         "kv_packing, head_dim].")
    if packed_schedule.ndim != 4:
        raise ValueError("packed_schedule must have shape "
                         "[steps, pcp, lanes, packed_fields].")
    _validate_q_block_size(q_block_size)
    if q_local.shape[0] % q_block_size != 0:
        raise NotImplementedError(
            "page-group MVP requires local_tokens to be a multiple of "
            "q_block_size.")
    if packed_schedule.shape[0] % pcp_size != 0:
        raise NotImplementedError(
            "page-group MVP requires schedule steps to be grouped in "
            "pcp_size-step ring groups.")
    if packed_schedule.shape[1] != pcp_size or packed_schedule.shape[2] <= 0:
        raise NotImplementedError("page-group MVP requires at least one lane.")
    if packed_schedule.shape[3] != ScheduleField.PACKED_NUM_FIELDS:
        raise ValueError("packed_schedule must use padded packed fields.")
    if q_local.shape[-1] != kv_cache_local.shape[-1]:
        raise ValueError("Q and KV head_dim must match.")
    if q_local.shape[-1] % 128 != 0:
        raise NotImplementedError(
            "page-group MVP requires head_dim to be 128-aligned.")
    expected_packing = get_dtype_packing(kv_cache_local.dtype)
    if kv_cache_local.shape[3] != expected_packing:
        raise ValueError("kv_cache_local packing axis does not match dtype "
                         f"packing: got {kv_cache_local.shape[3]} vs "
                         f"{expected_packing}.")
    if kv_cache_local.shape[3] not in (1, 2, 4):
        raise NotImplementedError(
            "packed PCP streaming KV loads currently support kv_packing in "
            "{1, 2, 4}.")
    if kv_cache_local.shape[2] * kv_cache_local.shape[3] < q_local.shape[1] * 2:
        raise ValueError(
            "packed KV cache does not contain all local K/V heads.")


def _pcp_streaming_attention_page_groups_single_head_pallas_call(
        q_single_head,
        kv_cache_single_head,
        packed_schedule,
        active_page_groups=None,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        collective_id: int | None,
        kv_head_idx: int = 0,
        kv_packing: int = 1,
        packed_kv_cache: bool = False,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Launch the single-head PCP streaming Pallas kernel.

    Args:
        q_single_head: [local_tokens, head_dim]. Rank-local compact Q rows for
            one query head.
        kv_cache_single_head: [1, pages, page_size, kv_heads_or_packed, 2,
            head_dim] for the current rank. The leading singleton dimension is
            kept to match the Pallas HBM indexing path.
        packed_schedule: [steps, pcp_size, num_lanes, 128]. Replicated schedule
            generated by `schedule.py`; `steps` must be padded to a multiple of
            `pcp_size`.
        active_page_groups: Optional scalar or [1] int32 array limiting how
            many page groups are executed at runtime.
        pcp_size: Number of ranks in the PCP ring.
        q_block_size: Static Q tile size consumed per page group.
        sm_scale: Softmax scale applied to QK scores.
        collective_id: Pallas collective id used for remote-copy semaphores.
            `None` is useful in tests that do not require explicit ids.
        kv_head_idx: KV head index when reading from a packed KV-cache layout.
        kv_packing: Number of packed K/V lanes in the cache dtype layout.
        packed_kv_cache: Whether `kv_cache_single_head` uses the batched-RPA
            packed KV-cache layout.
        k_scale: Optional dequantization scale for K.
        v_scale: Optional dequantization scale for V.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        [local_tokens, head_dim] output for this rank and head.
    """
    page_size = kv_cache_single_head.shape[2]
    head_dim = q_single_head.shape[-1]
    num_page_groups = packed_schedule.shape[0] // pcp_size
    num_lanes = packed_schedule.shape[2]
    num_q_blocks = q_single_head.shape[0] // q_block_size
    kv_vmem_packing = kv_packing if packed_kv_cache and kv_packing == 4 else 2
    if active_page_groups is None:
        active_page_groups = jnp.array([num_page_groups], dtype=jnp.int32)
    else:
        active_page_groups = jnp.asarray(active_page_groups, dtype=jnp.int32)
        if active_page_groups.shape == ():
            active_page_groups = active_page_groups[None]

    return pl.pallas_call(
        functools.partial(
            _pcp_streaming_attention_page_groups_kernel,
            pcp_size=pcp_size,
            num_lanes=num_lanes,
            q_block_size=q_block_size,
            num_page_groups=num_page_groups,
            num_q_blocks=num_q_blocks,
            sm_scale=sm_scale,
            kv_head_idx=kv_head_idx,
            kv_packing=kv_packing,
            packed_kv_cache=packed_kv_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (q_single_head.shape[0], head_dim),
            q_single_head.dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.DMA((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.REGULAR((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.REGULAR((num_lanes, )),
                pltpu.VMEM(
                    (pcp_size, num_lanes, ScheduleField.PACKED_NUM_FIELDS),
                    packed_schedule.dtype),
                pltpu.VMEM((q_block_size, head_dim), q_single_head.dtype),
                pltpu.VMEM((2, page_size, kv_vmem_packing, head_dim),
                           kv_cache_single_head.dtype),
                pltpu.VMEM((q_block_size, head_dim), q_single_head.dtype),
                pltpu.VMEM((TPU_HBM_ROW_TILE, head_dim), q_single_head.dtype),
                pltpu.VMEM((TPU_HBM_ROW_TILE, head_dim), q_single_head.dtype),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
            allow_collective_id_without_custom_barrier=True,
        ),
        name="pcp_streaming_attention_page_groups",
    )(active_page_groups, q_single_head, kv_cache_single_head, packed_schedule)


def _pcp_streaming_attention_page_groups_multi_head_pallas_call(
        q_multi_head,
        kv_cache_local,
        packed_schedule,
        active_page_groups=None,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        collective_id: int | None,
        kv_packing: int,
        kv_pages_per_block: int = 1,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Launch the packed-KV multi-head PCP streaming Pallas kernel.

    Args:
        q_multi_head: [local_tokens, kv_heads, q_per_kv, head_dim].
            Rank-local compact Q rows.
        kv_cache_local: [1, pages, page_size, packed_kv_heads_x2,
            kv_packing, head_dim]. Packed local KV-cache shard. The leading
            singleton dimension matches the Pallas HBM indexing path.
        packed_schedule: [steps, pcp_size, num_lanes, 128]. Replicated
            ring-grouped schedule.
        active_page_groups: Optional scalar or [1] int32 array limiting the
            number of page groups executed at runtime.
        pcp_size: Number of ranks in the PCP ring.
        q_block_size: Static Q tile size consumed per page group.
        sm_scale: Softmax scale applied to QK scores.
        collective_id: Pallas collective id used for remote-copy semaphores.
        kv_packing: Number of packed K/V lanes in the cache dtype layout.
        kv_pages_per_block: Number of consecutive local pages loaded per ring
            step. The production metadata path currently uses 1.
        k_scale: Optional dequantization scale for K.
        v_scale: Optional dequantization scale for V.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        [local_tokens, kv_heads, q_per_kv, head_dim] local output.
    """
    page_size = kv_cache_local.shape[2]
    kv_heads = q_multi_head.shape[1]
    q_per_kv = q_multi_head.shape[2]
    head_dim = q_multi_head.shape[-1]
    num_page_groups = packed_schedule.shape[0] // pcp_size
    num_lanes = packed_schedule.shape[2]
    num_q_blocks = q_multi_head.shape[0] // q_block_size
    if active_page_groups is None:
        active_page_groups = jnp.array([num_page_groups], dtype=jnp.int32)
    else:
        active_page_groups = jnp.asarray(active_page_groups, dtype=jnp.int32)
        if active_page_groups.shape == ():
            active_page_groups = active_page_groups[None]

    return pl.pallas_call(
        functools.partial(
            _pcp_streaming_attention_page_groups_multi_head_kernel,
            pcp_size=pcp_size,
            num_lanes=num_lanes,
            q_block_size=q_block_size,
            num_page_groups=num_page_groups,
            num_q_blocks=num_q_blocks,
            sm_scale=sm_scale,
            kv_heads=kv_heads,
            kv_packing=kv_packing,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ),
        out_shape=jax.ShapeDtypeStruct(
            q_multi_head.shape,
            q_multi_head.dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.DMA((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.REGULAR((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.REGULAR((num_lanes, )),
                pltpu.VMEM(
                    (pcp_size, num_lanes, ScheduleField.PACKED_NUM_FIELDS),
                    packed_schedule.dtype),
                pltpu.VMEM((q_block_size, kv_heads, q_per_kv, head_dim),
                           q_multi_head.dtype),
                pltpu.VMEM(
                    (2, kv_pages_per_block, page_size, kv_heads, 2, head_dim),
                    kv_cache_local.dtype),
                pltpu.VMEM((q_block_size, kv_heads, q_per_kv, head_dim),
                           q_multi_head.dtype),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
            allow_collective_id_without_custom_barrier=True,
        ),
        name="pcp_streaming_attention_page_groups_multi_head",
    )(active_page_groups, q_multi_head, kv_cache_local, packed_schedule)


def pcp_streaming_attention_page_groups_local(
        q_local,
        kv_cache_local,
        packed_schedule,
        active_page_groups=None,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        collective_id: int | None = 13,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Run PCP streaming page groups inside an existing PCP shard_map.

    Args:
        q_local: [local_tokens, kv_heads, q_per_kv, head_dim] for one PCP rank.
        kv_cache_local: [pages, page_size, kv_heads, 2, head_dim] for one rank.
        packed_schedule: Replicated [steps, pcp_size, lanes, 128] schedule.
        active_page_groups: Optional scalar or [1] int32 runtime limit for
            page groups. When omitted, all schedule groups are executed.
        pcp_size: Number of ranks in the PCP ring.
        q_block_size: Static local Q tile size consumed by the Pallas kernel.
        sm_scale: Softmax scale applied to QK scores.
        collective_id: Base Pallas collective id. Per-head calls offset this
            id so each head pair gets a distinct semaphore namespace.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        Local rank output with the same shape as q_local.
    """
    _validate_common_page_group_local_inputs(q_local, kv_cache_local,
                                             packed_schedule, pcp_size,
                                             q_block_size)
    kv_heads = q_local.shape[1]
    q_per_kv = q_local.shape[2]
    head_outputs = []
    for kv_head_idx in range(kv_heads):
        q_head_outputs = []
        kv_slice = kv_cache_local[:, :, kv_head_idx:kv_head_idx + 1]
        for q_head_idx in range(q_per_kv):
            q_slice = q_local[:, kv_head_idx:kv_head_idx + 1,
                              q_head_idx:q_head_idx + 1]
            if collective_id is None:
                head_collective_id = None
            else:
                head_collective_id = (collective_id + kv_head_idx * q_per_kv +
                                      q_head_idx)
            out = _pcp_streaming_attention_page_groups_single_head_pallas_call(
                q_slice[:, 0, 0, :],
                kv_slice[None, ...],
                packed_schedule,
                active_page_groups,
                pcp_size=pcp_size,
                q_block_size=q_block_size,
                sm_scale=sm_scale,
                collective_id=head_collective_id,
                mesh_axis_names=mesh_axis_names,
                pcp_axis_name=pcp_axis_name,
            )
            q_head_outputs.append(out[:, None, None, :])
        head_outputs.append(jnp.concatenate(q_head_outputs, axis=2))
    return jnp.concatenate(head_outputs, axis=1)


def pcp_streaming_attention_page_groups_packed_local(
        q_local,
        kv_cache_local,
        packed_schedule,
        active_page_groups=None,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        collective_id: int | None = 13,
        kv_pages_per_block: int = 1,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Run PCP streaming page groups on batched-RPA packed KV cache layout.

    Args:
        q_local: [local_tokens, kv_heads, q_per_kv, head_dim] for one PCP rank.
        kv_cache_local: [pages, page_size, packed_kv_heads_x2, kv_packing,
            head_dim] for one rank.
        packed_schedule: Replicated [steps, pcp_size, lanes, 128] schedule.
        active_page_groups: Optional scalar or [1] int32 runtime limit for
            page groups. When omitted, all schedule groups are executed.
        pcp_size: Number of ranks in the PCP ring.
        q_block_size: Static local Q tile size consumed by the Pallas kernel.
        sm_scale: Softmax scale applied to QK scores.
        collective_id: Base Pallas collective id. Per-head fallback calls
            offset this id so each head pair gets a distinct semaphore
            namespace.
        kv_pages_per_block: Number of consecutive local pages loaded per ring
            step. The production metadata path currently uses 1.
        k_scale: Optional dequantization scale for K.
        v_scale: Optional dequantization scale for V.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        Local rank output with the same shape as q_local.
    """
    _validate_packed_page_group_local_inputs(q_local, kv_cache_local,
                                             packed_schedule, pcp_size,
                                             q_block_size)
    kv_heads = q_local.shape[1]
    q_per_kv = q_local.shape[2]
    kv_packing = kv_cache_local.shape[3]
    if kv_packing == 2 and kv_heads >= 2 and q_per_kv >= 2:
        return _pcp_streaming_attention_page_groups_multi_head_pallas_call(
            q_local,
            kv_cache_local[None, ...],
            packed_schedule,
            active_page_groups,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            sm_scale=sm_scale,
            collective_id=collective_id,
            kv_packing=kv_packing,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        )

    head_outputs = []
    for kv_head_idx in range(kv_heads):
        q_head_outputs = []
        for q_head_idx in range(q_per_kv):
            q_slice = q_local[:, kv_head_idx:kv_head_idx + 1,
                              q_head_idx:q_head_idx + 1]
            if collective_id is None:
                head_collective_id = None
            else:
                head_collective_id = (collective_id + kv_head_idx * q_per_kv +
                                      q_head_idx)
            out = _pcp_streaming_attention_page_groups_single_head_pallas_call(
                q_slice[:, 0, 0, :],
                kv_cache_local[None, ...],
                packed_schedule,
                active_page_groups,
                pcp_size=pcp_size,
                q_block_size=q_block_size,
                sm_scale=sm_scale,
                collective_id=head_collective_id,
                kv_head_idx=kv_head_idx,
                kv_packing=kv_packing,
                packed_kv_cache=True,
                k_scale=k_scale,
                v_scale=v_scale,
                mesh_axis_names=mesh_axis_names,
                pcp_axis_name=pcp_axis_name,
            )
            q_head_outputs.append(out[:, None, None, :])
        head_outputs.append(jnp.concatenate(q_head_outputs, axis=2))
    return jnp.concatenate(head_outputs, axis=1)


def pcp_streaming_attention_page_groups_packed_local_from_metadata(
        q_local,
        kv_cache_local,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        *,
        pcp_size: int,
        interleave_size: int,
        q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
        sm_scale: float,
        collective_id: int | None = 13,
        max_context_tokens: int | None = None,
        num_lanes: int = 1,
        kv_pages_per_block: int = 1,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Generate the narrow metadata schedule and run packed-local PCP RPA.

    This wrapper keeps the old schedule-taking API as the execution path. The
    runtime schedule shape is derived from the local compile bucket
    ``q_local.shape[0] * pcp_size`` and, when provided, max_context_tokens
    instead of the full KV cache capacity.

    Args:
        q_local: [local_tokens, kv_heads, q_per_kv, head_dim] compact Q rows
            for the current PCP rank.
        kv_cache_local: [pages, page_size, packed_kv_heads_x2, kv_packing,
            head_dim] local packed KV-cache shard.
        kv_lens: [max_num_seqs]. Total KV length for each request, including
            the query tokens in the current prefill chunk.
        page_indices: Flattened or 2D vLLM block table. Entries are reduced
            modulo the local KV-cache block count before local page loads.
        cu_q_lens: [max_num_seqs + 1]. Cumulative query-token offsets for the
            current batch.
        distribution: [3] vLLM request distribution metadata. The third entry
            is the number of active requests considered by the schedule.
        pcp_size: Number of ranks in the PCP ring.
        interleave_size: Number of consecutive global tokens assigned to one
            rank before rotating to the next PCP rank. The production path
            requires this to match `page_size`.
        q_block_size: Static local Q tile size used by the Pallas kernel.
        sm_scale: Softmax scale applied to QK scores.
        collective_id: Base Pallas collective id for ring remote-copy
            semaphores.
        max_context_tokens: Optional cap used when estimating the static
            schedule shape.
        num_lanes: Number of independent schedule lanes. The metadata path
            currently supports one lane.
        kv_pages_per_block: Number of consecutive local pages loaded per ring
            step. The metadata path currently supports one page.
        k_scale: Optional dequantization scale for K.
        v_scale: Optional dequantization scale for V.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        [local_tokens, kv_heads, q_per_kv, head_dim] local attention output.
    """
    page_size = kv_cache_local.shape[1]
    packed_schedule, active_page_groups = (
        build_pcp_streaming_schedule_inputs_from_metadata_jax(
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            global_bucket_tokens=q_local.shape[0] * pcp_size,
            local_kv_cache_num_blocks=kv_cache_local.shape[0],
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            q_block_size=q_block_size,
            max_context_tokens=max_context_tokens,
            num_lanes=num_lanes,
            kv_pages_per_block=kv_pages_per_block,
        ))
    return pcp_streaming_attention_page_groups_packed_local(
        q_local,
        kv_cache_local,
        packed_schedule,
        active_page_groups,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
        sm_scale=sm_scale,
        collective_id=collective_id,
        kv_pages_per_block=kv_pages_per_block,
        k_scale=k_scale,
        v_scale=v_scale,
        mesh_axis_names=mesh_axis_names,
        pcp_axis_name=pcp_axis_name,
    )


def _pcp_streaming_attention_page_groups_single_head(
        q_by_rank,
        kv_cache_by_rank,
        packed_schedule,
        active_page_groups=None,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        collective_id: int | None = 13,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Run RingAttention-style PCP page groups.

    Args:
        q_by_rank: [pcp, local_tokens, 1, 1, head_dim].
        kv_cache_by_rank: [pcp, pages, page_size, 1, 2, head_dim].
        packed_schedule: [page_groups * pcp, pcp, lanes, 128] ring-grouped
            schedule. Within each page group, step order is source-rank order.
        pcp_size: Number of PCP ranks.
        q_block_size: Static Q tile size consumed by each page group.
        sm_scale: Attention softmax scale.
        collective_id: Pallas collective id used by the local ring barrier.

    Returns:
        Rank-local packed output with the same shape as q_by_rank.
    """
    _validate_page_group_inputs(q_by_rank, kv_cache_by_rank, packed_schedule,
                                pcp_size, q_block_size)

    def _call(q, kv_cache, schedule):
        out = _pcp_streaming_attention_page_groups_single_head_pallas_call(
            q[0, :, 0, 0, :],
            kv_cache,
            schedule,
            active_page_groups,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            sm_scale=sm_scale,
            collective_id=collective_id,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        )
        return out[None, :, None, None, :]

    mesh = jax.sharding.Mesh(jax.local_devices()[:pcp_size], (AXIS, ))
    shard_map_kernel = jax.jit(
        jax.shard_map(
            _call,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None, None),
                P(AXIS, None, None, None, None, None),
                P(None, None, None, None),
            ),
            out_specs=P(AXIS, None, None, None, None),
            check_vma=False,
        ))
    return shard_map_kernel(q_by_rank, kv_cache_by_rank, packed_schedule)


def pcp_streaming_attention_page_groups(
        q_by_rank,
        kv_cache_by_rank,
        packed_schedule,
        active_page_groups=None,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        collective_id: int | None = 13,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Run RingAttention-style PCP page groups.

    This wrapper supports multiple KV heads and Q heads per KV head by invoking
    the single-head Pallas kernel for each head pair.

    Args:
        q_by_rank: [pcp_size, local_tokens, kv_heads, q_per_kv, head_dim].
            Rank-major query tensor with an explicit leading PCP dimension.
        kv_cache_by_rank: [pcp_size, pages, page_size, kv_heads, 2, head_dim].
            Unpacked KV-cache shards for all PCP ranks.
        packed_schedule: [steps, pcp_size, lanes, 128] replicated
            ring-grouped schedule.
        active_page_groups: Optional scalar or [1] int32 runtime limit for
            page groups. When omitted, all schedule groups are executed.
        pcp_size: Number of ranks in the PCP ring.
        q_block_size: Static local Q tile size consumed by the Pallas kernel.
        sm_scale: Softmax scale applied to QK scores.
        collective_id: Base Pallas collective id. Per-head calls offset this
            id so each head pair gets a distinct semaphore namespace.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        Rank-major output with the same shape as q_by_rank.
    """
    _validate_common_page_group_inputs(q_by_rank, kv_cache_by_rank,
                                       packed_schedule, pcp_size, q_block_size)
    kv_heads = q_by_rank.shape[2]
    q_per_kv = q_by_rank.shape[3]
    if kv_heads == 1 and q_per_kv == 1:
        return _pcp_streaming_attention_page_groups_single_head(
            q_by_rank,
            kv_cache_by_rank,
            packed_schedule,
            active_page_groups,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            sm_scale=sm_scale,
            collective_id=collective_id,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        )

    head_outputs = []
    for kv_head_idx in range(kv_heads):
        q_head_outputs = []
        kv_slice = kv_cache_by_rank[:, :, :, kv_head_idx:kv_head_idx + 1]
        for q_head_idx in range(q_per_kv):
            q_slice = q_by_rank[:, :, kv_head_idx:kv_head_idx + 1,
                                q_head_idx:q_head_idx + 1]
            if collective_id is None:
                head_collective_id = None
            else:
                head_collective_id = (collective_id + kv_head_idx * q_per_kv +
                                      q_head_idx)
            q_head_outputs.append(
                _pcp_streaming_attention_page_groups_single_head(
                    q_slice,
                    kv_slice,
                    packed_schedule,
                    active_page_groups,
                    pcp_size=pcp_size,
                    q_block_size=q_block_size,
                    sm_scale=sm_scale,
                    collective_id=head_collective_id,
                    mesh_axis_names=mesh_axis_names,
                    pcp_axis_name=pcp_axis_name,
                ))
        head_outputs.append(jnp.concatenate(q_head_outputs, axis=3))
    return jnp.concatenate(head_outputs, axis=2)


def pcp_streaming_attention_single_page_group(
        q_by_rank,
        kv_cache_by_rank,
        packed_schedule,
        *,
        pcp_size: int,
        sm_scale: float,
        collective_id: int | None = 13,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Run one RingAttention-style PCP page group.

    Args:
        q_by_rank: [pcp_size, local_tokens, kv_heads, q_per_kv, head_dim].
        kv_cache_by_rank: [pcp_size, pages, page_size, kv_heads, 2, head_dim].
        packed_schedule: [pcp_size, pcp_size, lanes, 128]. Exactly one
            ring-group worth of schedule rows.
        pcp_size: Number of ranks in the PCP ring.
        sm_scale: Softmax scale applied to QK scores.
        collective_id: Base Pallas collective id.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        Rank-major output with the same shape as q_by_rank.
    """
    if packed_schedule.shape[0] != pcp_size:
        raise NotImplementedError(
            "single-page-group wrapper requires exactly pcp_size schedule "
            "steps.")
    return pcp_streaming_attention_page_groups(
        q_by_rank,
        kv_cache_by_rank,
        packed_schedule,
        pcp_size=pcp_size,
        q_block_size=q_by_rank.shape[1],
        sm_scale=sm_scale,
        collective_id=collective_id,
        mesh_axis_names=mesh_axis_names,
        pcp_axis_name=pcp_axis_name,
    )
