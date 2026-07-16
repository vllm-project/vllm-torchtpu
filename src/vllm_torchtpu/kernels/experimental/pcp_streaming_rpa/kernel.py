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
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa.utils import (
    broadcast_minor, convert_to_target_bitwidth, get_dtype_packing,
    strided_load)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    ScheduleField, build_pcp_streaming_schedule_inputs_from_metadata_jax)

AXIS = "pcp"
PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE = 512
TPU_HBM_ROW_TILE = 16
ONLINE_STATE_PADDED_LANES = 128


def _validate_packed_kv_layout(q_local,
                               kv_cache_local,
                               *,
                               k_scale: float | None = None,
                               v_scale: float | None = None):
    """Validate PCP's physical packed KV layout against logical Q heads."""
    if q_local.ndim != 4:
        raise ValueError("PCP q_local must have shape "
                         "[tokens, kv_heads, q_per_kv, head_dim], got "
                         f"{q_local.shape}.")
    if kv_cache_local.ndim != 5:
        raise ValueError("PCP kv_cache_local must have shape "
                         "[pages, page_size, packed_kv_groups, kv_packing, "
                         f"head_dim], got {kv_cache_local.shape}.")

    kv_heads = q_local.shape[1]
    q_per_kv = q_local.shape[2]
    head_dim = q_local.shape[-1]
    packed_kv_groups = kv_cache_local.shape[2]
    kv_packing = kv_cache_local.shape[3]
    dtype_packing = get_dtype_packing(kv_cache_local.dtype)
    if kv_heads <= 0 or q_per_kv <= 0:
        raise ValueError("PCP requires positive kv_heads and q_per_kv, got "
                         f"{kv_heads=} {q_per_kv=}.")
    if kv_packing not in (2, 4):
        raise NotImplementedError(
            "PCP streaming RPA supports only BF16/FP8 packed KV cache "
            f"layouts with packing 2 or 4, got {kv_packing}.")
    if kv_packing != dtype_packing:
        raise ValueError("PCP KV cache packing does not match dtype: "
                         f"shape_packing={kv_packing}, "
                         f"dtype_packing={dtype_packing}, "
                         f"dtype={kv_cache_local.dtype}.")
    expected_packed_kv_groups = math.ceil((2 * kv_heads) / kv_packing)
    if packed_kv_groups != expected_packed_kv_groups:
        raise ValueError(
            "PCP KV cache packed layout does not match logical KV heads: "
            f"packed_kv_groups={packed_kv_groups}, "
            f"expected={expected_packed_kv_groups}, {kv_heads=}, "
            f"{kv_packing=}.")
    if kv_cache_local.shape[-1] != head_dim:
        raise ValueError("PCP KV cache head_dim must match Q head_dim: "
                         f"{kv_cache_local.shape[-1]} vs {head_dim}.")
    if (k_scale is None) != (v_scale is None):
        raise ValueError("PCP FP8 KV cache requires k_scale and v_scale to "
                         "be both None or both non-None.")
    if kv_packing == 4:
        if k_scale is None or v_scale is None:
            raise ValueError(
                "PCP FP8 KV cache requires positive finite k_scale and "
                "v_scale.")
        if (not math.isfinite(float(k_scale)) or float(k_scale) <= 0
                or not math.isfinite(float(v_scale)) or float(v_scale) <= 0):
            raise ValueError(
                "PCP FP8 KV cache requires positive finite k_scale and "
                "v_scale.")
    return kv_heads, packed_kv_groups, kv_packing


def _strided_load_packed_kv_group(kv_vmem_ref, slot, packed_group_idx):
    page_size = kv_vmem_ref.shape[2]
    kv_block_tokens = kv_vmem_ref.shape[1] * page_size
    packed_kv_groups = kv_vmem_ref.shape[3]
    head_dim = kv_vmem_ref.shape[-1]
    kv_ref = kv_vmem_ref.bitcast(jnp.uint32).at[slot]
    kv_flat_ref = kv_ref.reshape(kv_block_tokens * packed_kv_groups, head_dim)
    kv_group_loaded = strided_load(
        kv_flat_ref,
        packed_group_idx,
        kv_block_tokens * packed_kv_groups,
        packed_kv_groups,
    )
    bitwidth = jax.dtypes.itemsize_bits(kv_vmem_ref.dtype)
    pairs = convert_to_target_bitwidth(
        kv_group_loaded,
        target_bitwidth=bitwidth,
        kv_dtype=kv_vmem_ref.dtype,
    )
    return [(k.reshape(kv_block_tokens,
                       head_dim), v.reshape(kv_block_tokens, head_dim))
            for k, v in pairs]


def _consume_scheduled_kv_page_multi_head(
    q_vmem_ref,
    kv_vmem_ref,
    sched_vmem_ref,
    slot,
    consumer_rank,
    lane,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    sm_scale,
    pcp_size,
    q_compute_size,
    k_scale,
    v_scale,
    causal,
):
    if causal:
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
    page_size = kv_vmem_ref.shape[2]
    kv_block_tokens = kv_vmem_ref.shape[1] * page_size
    num_q_compute_blocks = q_vmem_ref.shape[0] // q_compute_size
    compute_dtype = q_vmem_ref.dtype

    heads_per_group = kv_vmem_ref.shape[4] // 2
    for packed_group_idx in range(kv_vmem_ref.shape[3]):
        kv_pairs = _strided_load_packed_kv_group(kv_vmem_ref, slot,
                                                 packed_group_idx)
        for pair_idx, (k, v) in enumerate(kv_pairs):
            kv_head_idx = packed_group_idx * heads_per_group + pair_idx
            if kv_head_idx >= q_vmem_ref.shape[1]:
                continue
            if k_scale is not None:
                k = (k.astype(jnp.float32) * k_scale).astype(compute_dtype)
            if v_scale is not None:
                v = (v.astype(jnp.float32) * v_scale).astype(compute_dtype)
            if not causal:
                k = k * sm_scale

            def _q_compute_loop(q_compute_idx, carry):
                del carry
                q_start = pl.multiple_of(q_compute_idx * q_compute_size,
                                         q_compute_size)
                q_slice = q_vmem_ref.at[
                    pl.ds(q_start, q_compute_size),
                    kv_head_idx,
                    :,
                    :,
                ][...]
                m_head = m_scratch_ref.at[
                    kv_head_idx,
                    pl.ds(q_start, q_compute_size),
                    :,
                    :,
                ][...]
                l_head = l_scratch_ref.at[
                    kv_head_idx,
                    pl.ds(q_start, q_compute_size),
                    :,
                    :,
                ][...]
                acc_head = acc_scratch_ref.at[
                    kv_head_idx,
                    pl.ds(q_start, q_compute_size),
                    :,
                    :,
                ][...]

                scores = lax.dot_general(
                    q_slice,
                    k,
                    (((2, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32,
                )

                if causal:
                    scores = scores * sm_scale
                    q_row = (
                        q_start +
                        lax.broadcasted_iota(jnp.int32,
                                             (q_compute_size, q_per_kv, 1), 0))
                    q_interleave = page_size
                    q_chunk_idx = lax.div(q_row, q_interleave)
                    q_chunk_offset = lax.rem(q_row, q_interleave)
                    q_pos = (q_global_start +
                             q_chunk_idx * pcp_size * q_interleave +
                             q_chunk_offset)
                    kv_local_pos = lax.broadcasted_iota(
                        jnp.int32, (1, 1, kv_block_tokens), 2)
                    kv_page_offset = lax.div(kv_local_pos, page_size)
                    kv_token_offset = lax.rem(kv_local_pos, page_size)
                    kv_pos = (kv_global_start +
                              kv_page_offset * pcp_size * page_size +
                              kv_token_offset)
                    kv_valid = kv_local_pos < kv_valid_len
                    q_valid = q_row < q_tile_size
                    entry_valid = req_id != -1
                    row_active = jnp.logical_and(entry_valid, q_valid)
                    mask = jnp.logical_and(
                        jnp.logical_and(q_pos >= kv_pos, kv_valid), q_valid)
                    scores = jnp.where(mask, scores, -jnp.inf)
                    scores = jnp.where(row_active, scores, 0.0)

                m_curr = jnp.max(scores, axis=2, keepdims=True)
                if causal:
                    m_next = jnp.where(row_active, jnp.maximum(m_head, m_curr),
                                       m_head)
                else:
                    m_next = jnp.maximum(m_head, m_curr)
                m_next_broadcast = broadcast_minor(m_next, scores.shape)
                p = jnp.exp(scores - m_next_broadcast)
                alpha = jnp.exp(m_head - m_next)
                if causal:
                    p = jnp.where(row_active, p, 0.0)
                    alpha = jnp.where(row_active, alpha, 1.0)
                l_next = alpha * l_head + jnp.sum(p, axis=2, keepdims=True)
                pv = lax.dot_general(
                    p.astype(compute_dtype),
                    v,
                    (((2, ), (0, )), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                alpha_broadcast = broadcast_minor(alpha, acc_head.shape)
                acc_next = alpha_broadcast * acc_head + pv

                m_scratch_ref.at[
                    kv_head_idx,
                    pl.ds(q_start, q_compute_size),
                    :,
                    :,
                ][...] = m_next
                l_scratch_ref.at[
                    kv_head_idx,
                    pl.ds(q_start, q_compute_size),
                    :,
                    :,
                ][...] = l_next
                acc_scratch_ref.at[
                    kv_head_idx,
                    pl.ds(q_start, q_compute_size),
                    :,
                    :,
                ][...] = acc_next
                return jnp.array(0, dtype=jnp.int32)

            lax.fori_loop(
                0,
                num_q_compute_blocks,
                _q_compute_loop,
                jnp.array(0, dtype=jnp.int32),
                unroll=True,
            )


def _reset_multi_head_online_state(m_scratch_ref, l_scratch_ref,
                                   acc_scratch_ref):
    m_scratch_ref[...] = jnp.full(m_scratch_ref.shape,
                                  -jnp.inf,
                                  dtype=jnp.float32)
    l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, dtype=jnp.float32)
    acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, dtype=jnp.float32)


def _write_multi_head_output_from_state(o_vmem_ref, l_scratch_ref,
                                        acc_scratch_ref):
    l_state = l_scratch_ref.at[:, :, :, 0][...]
    l_values = jnp.transpose(l_state, (1, 0, 2))[..., None]
    acc = jnp.transpose(acc_scratch_ref[...], (1, 0, 2, 3))
    l_broadcast = jnp.broadcast_to(l_values, acc.shape)
    o_vmem_ref[...] = jnp.where(l_broadcast > 0, acc / l_broadcast,
                                0.0).astype(o_vmem_ref.dtype)


def _store_multi_head_online_state_tile(m_ref, l_ref, acc_ref, local_dma_sem,
                                        m_scratch_ref, l_scratch_ref,
                                        acc_scratch_ref, q_hbm_offset,
                                        q_tile_size):

    @pl.when(q_tile_size > 0)
    def _store_state_tile():
        m_store = pltpu.make_async_copy(
            src_ref=m_scratch_ref.at[
                :,
                pl.ds(0, q_tile_size),
                :,
                :,
            ],
            dst_ref=m_ref.at[
                :,
                pl.ds(q_hbm_offset, q_tile_size),
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        m_store.start()
        m_store.wait()
        l_store = pltpu.make_async_copy(
            src_ref=l_scratch_ref.at[
                :,
                pl.ds(0, q_tile_size),
                :,
                :,
            ],
            dst_ref=l_ref.at[
                :,
                pl.ds(q_hbm_offset, q_tile_size),
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        l_store.start()
        l_store.wait()
        acc_store = pltpu.make_async_copy(
            src_ref=acc_scratch_ref.at[
                :,
                pl.ds(0, q_tile_size),
                :,
                :,
            ],
            dst_ref=acc_ref.at[
                :,
                pl.ds(q_hbm_offset, q_tile_size),
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        acc_store.start()
        acc_store.wait()


def _load_multi_head_online_state_tile(m_ref, l_ref, acc_ref, local_dma_sem,
                                       m_scratch_ref, l_scratch_ref,
                                       acc_scratch_ref, q_hbm_offset,
                                       q_tile_size):
    _reset_multi_head_online_state(m_scratch_ref, l_scratch_ref,
                                   acc_scratch_ref)

    @pl.when(q_tile_size > 0)
    def _load_state_tile():
        m_load = pltpu.make_async_copy(
            src_ref=m_ref.at[
                :,
                pl.ds(q_hbm_offset, q_tile_size),
                :,
                :,
            ],
            dst_ref=m_scratch_ref.at[
                :,
                pl.ds(0, q_tile_size),
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        m_load.start()
        m_load.wait()
        l_load = pltpu.make_async_copy(
            src_ref=l_ref.at[
                :,
                pl.ds(q_hbm_offset, q_tile_size),
                :,
                :,
            ],
            dst_ref=l_scratch_ref.at[
                :,
                pl.ds(0, q_tile_size),
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        l_load.start()
        l_load.wait()
        acc_load = pltpu.make_async_copy(
            src_ref=acc_ref.at[
                :,
                pl.ds(q_hbm_offset, q_tile_size),
                :,
                :,
            ],
            dst_ref=acc_scratch_ref.at[
                :,
                pl.ds(0, q_tile_size),
                :,
                :,
            ],
            sem=local_dma_sem,
        )
        acc_load.start()
        acc_load.wait()


def _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sem, step):
    load_op = pltpu.make_async_copy(
        src_ref=packed_schedule_ref.at[step],
        dst_ref=sched_vmem_ref.at[:, :, :],
        sem=sem,
    )
    load_op.start()
    load_op.wait()


def _normalize_q_compute_size(q_block_size: int,
                              q_compute_size: int | None) -> int:
    if q_compute_size is None:
        q_compute_size = q_block_size
    q_compute_size = int(q_compute_size)
    if q_compute_size <= 0:
        raise ValueError("q_compute_size must be positive.")
    if q_compute_size > q_block_size:
        raise ValueError("q_compute_size must be <= q_block_size.")
    if q_block_size % q_compute_size != 0:
        raise NotImplementedError(
            "PCP streaming RPA requires q_block_size to be divisible by "
            "q_compute_size.")
    if q_compute_size % TPU_HBM_ROW_TILE != 0:
        raise NotImplementedError(
            "PCP streaming RPA requires q_compute_size to be a multiple of "
            f"{TPU_HBM_ROW_TILE}.")
    return q_compute_size


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


def _load_local_kv_page_block_all_heads_packed(kv_cache_ref, kv_vmem_ref, sem,
                                               sched_vmem_ref, source_rank,
                                               lane, *, pcp_size,
                                               packed_kv_groups,
                                               kv_pages_per_block):
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
                pl.ds(0, packed_kv_groups),
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


def _run_pcp_page_groups_multi_head(
    active_page_groups,
    q_ref,
    kv_cache_ref,
    packed_schedule_ref,
    sched_dma_sem,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    remote_sync_sems,
    group_slot_free_sems,
    sched_vmem_ref,
    q_vmem_ref,
    kv_vmem_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    state_m_ref,
    state_l_ref,
    state_acc_ref,
    o_ref,
    *,
    pcp_size,
    lane,
    q_compute_size,
    sm_scale,
    kv_heads,
    packed_kv_groups,
    kv_packing,
    kv_pages_per_block,
    k_scale,
    v_scale,
    mesh_axis_names,
    pcp_axis_name,
    causal,
    state_mode,
):
    my_id = lax.axis_index(pcp_axis_name)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)
    next_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, next_rank)
    prev_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, prev_rank)

    def _run_page_group(group_idx, kv_group_ref):
        group_start = group_idx * pcp_size

        _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sched_dma_sem,
                            group_start)
        q_hbm_offset = sched_vmem_ref[my_id, lane, ScheduleField.Q_HBM_OFFSET]
        group_req_id = sched_vmem_ref[my_id, lane, ScheduleField.REQ_ID]
        group_o_hbm_offset = sched_vmem_ref[my_id, lane,
                                            ScheduleField.O_HBM_OFFSET]
        group_q_tile_size = sched_vmem_ref[my_id, lane,
                                           ScheduleField.Q_TILE_SIZE]
        group_load_q = jnp.logical_or(
            group_idx == 0,
            sched_vmem_ref[my_id, lane, ScheduleField.LOAD_Q] != 0,
        )
        q_tile_size = jnp.where(
            jnp.logical_and(group_req_id != -1, group_load_q),
            group_q_tile_size,
            0,
        )

        if state_mode == "current":

            @pl.when(group_load_q)
            def _reset_current_state_tile():
                _reset_multi_head_online_state(m_scratch_ref, l_scratch_ref,
                                               acc_scratch_ref)

        elif state_mode == "history":

            @pl.when(group_load_q)
            def _load_history_state_tile():
                _load_multi_head_online_state_tile(
                    state_m_ref,
                    state_l_ref,
                    state_acc_ref,
                    local_dma_sem,
                    m_scratch_ref,
                    l_scratch_ref,
                    acc_scratch_ref,
                    q_hbm_offset,
                    q_tile_size,
                )
        else:
            raise ValueError(f"Unsupported PCP state mode: {state_mode}")

        @pl.when(group_load_q)
        def _load_q_tile():
            _load_compact_q_tile_multi_head(
                q_ref,
                q_vmem_ref,
                local_dma_sem,
                q_hbm_offset,
                q_tile_size,
            )

        _load_schedule_step(packed_schedule_ref, sched_vmem_ref, sched_dma_sem,
                            group_start + my_id)
        _load_local_kv_page_block_all_heads_packed(
            kv_cache_ref,
            kv_group_ref,
            local_dma_sem,
            sched_vmem_ref,
            my_id,
            lane,
            pcp_size=pcp_size,
            packed_kv_groups=packed_kv_groups,
            kv_pages_per_block=kv_pages_per_block,
        )

        @pl.when(group_idx > 0)
        def _wait_previous_group_slot_free():
            pl.semaphore_wait(group_slot_free_sems.at[lane], 1)

        def _round_loop(round_idx, carry):
            group_has_last = carry
            curr_slot = lax.rem(round_idx, 2)
            src_rank = lax.rem(my_id + pcp_size - round_idx, pcp_size)

            @pl.when(round_idx > 0)
            def _load_round_schedule():
                _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                    sched_dma_sem, group_start + src_rank)

            group_has_last = jnp.logical_or(
                group_has_last,
                sched_vmem_ref[my_id, lane, ScheduleField.IS_LAST_KV] != 0,
            )

            next_slot = 1 - curr_slot
            safe_round_idx = jnp.minimum(round_idx, pcp_size - 2)
            safe_prev_round_idx = jnp.minimum(round_idx - 1, pcp_size - 2)

            @pl.when(round_idx > 0)
            def _wait_remote_sync():

                @pl.when(round_idx == pcp_size - 1)
                def _wait_last_round():
                    pl.semaphore_wait(
                        remote_sync_sems.at[lane, safe_prev_round_idx], 1)

                @pl.when(round_idx < pcp_size - 1)
                def _wait_middle_round():
                    pl.semaphore_wait(
                        remote_sync_sems.at[lane, safe_prev_round_idx], 2)

            remote_op = pltpu.make_async_remote_copy(
                src_ref=kv_group_ref.at[curr_slot],
                dst_ref=kv_group_ref.at[next_slot],
                send_sem=remote_send_sems.at[lane, safe_round_idx],
                recv_sem=remote_recv_sems.at[lane, safe_round_idx],
                device_id=next_device_id,
                device_id_type=pl.DeviceIdType.MESH,
            )

            @pl.when(round_idx < pcp_size - 1)
            def _start_remote_copy():
                remote_op.start()

            _consume_scheduled_kv_page_multi_head(
                q_vmem_ref,
                kv_group_ref,
                sched_vmem_ref,
                curr_slot,
                my_id,
                lane,
                m_scratch_ref,
                l_scratch_ref,
                acc_scratch_ref,
                sm_scale=sm_scale,
                pcp_size=pcp_size,
                q_compute_size=q_compute_size,
                k_scale=k_scale,
                v_scale=v_scale,
                causal=causal,
            )

            @pl.when(round_idx < pcp_size - 1)
            def _finish_remote_copy():
                remote_op.wait()
                pl.semaphore_signal(
                    remote_sync_sems.at[lane, safe_round_idx],
                    1,
                    device_id=next_device_id,
                    device_id_type=pl.DeviceIdType.MESH,
                )

                @pl.when(round_idx < pcp_size - 2)
                def _signal_prev_device():
                    pl.semaphore_signal(
                        remote_sync_sems.at[lane, safe_round_idx],
                        1,
                        device_id=prev_device_id,
                        device_id_type=pl.DeviceIdType.MESH,
                    )

            return group_has_last

        group_has_last = lax.fori_loop(
            0,
            pcp_size,
            _round_loop,
            jnp.array(False),
            unroll=False,
        )

        @pl.when(group_idx < active_page_groups - 1)
        def _signal_next_group_slot_free():
            pl.semaphore_signal(
                group_slot_free_sems.at[lane],
                1,
                device_id=prev_device_id,
                device_id_type=pl.DeviceIdType.MESH,
            )

        store_q_tile_size = jnp.where(
            jnp.logical_and(group_req_id != -1, group_has_last),
            group_q_tile_size,
            0,
        )
        if state_mode == "current":
            _store_multi_head_online_state_tile(
                state_m_ref,
                state_l_ref,
                state_acc_ref,
                local_dma_sem,
                m_scratch_ref,
                l_scratch_ref,
                acc_scratch_ref,
                q_hbm_offset,
                store_q_tile_size,
            )
        elif state_mode == "history":

            @pl.when(store_q_tile_size > 0)
            def _write_and_store_history_output():
                _write_multi_head_output_from_state(q_vmem_ref, l_scratch_ref,
                                                    acc_scratch_ref)
                _store_compact_output_tile_multi_head(
                    o_ref,
                    q_vmem_ref,
                    local_dma_sem,
                    group_o_hbm_offset,
                    store_q_tile_size,
                )

        return jnp.array(0, dtype=jnp.int32)

    def _page_group_loop(group_idx, carry):
        del carry
        return _run_page_group(group_idx, kv_vmem_ref)

    lax.fori_loop(
        0,
        active_page_groups,
        _page_group_loop,
        jnp.array(0, dtype=jnp.int32),
        unroll=False,
    )


def _pcp_streaming_attention_current_state_page_groups_multi_head_kernel(
    active_page_groups_ref,
    q_ref,
    kv_cache_ref,
    packed_schedule_ref,
    m_out_ref,
    l_out_ref,
    acc_out_ref,
    sched_dma_sem,
    local_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    remote_sync_sems,
    group_slot_free_sems,
    sched_vmem_ref,
    q_vmem_ref,
    kv_vmem_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    pcp_size,
    num_lanes,
    q_compute_size,
    num_page_groups,
    sm_scale,
    kv_heads,
    packed_kv_groups,
    kv_packing,
    kv_pages_per_block,
    k_scale,
    v_scale,
    mesh_axis_names,
    pcp_axis_name,
):
    """Current-chunk causal PCP body that only materializes online state."""
    for lane in range(num_lanes):
        active_page_groups = jnp.minimum(active_page_groups_ref[0],
                                         num_page_groups)
        _run_pcp_page_groups_multi_head(
            active_page_groups,
            q_ref,
            kv_cache_ref,
            packed_schedule_ref,
            sched_dma_sem,
            local_dma_sem,
            remote_send_sems,
            remote_recv_sems,
            remote_sync_sems,
            group_slot_free_sems,
            sched_vmem_ref,
            q_vmem_ref,
            kv_vmem_ref,
            m_scratch_ref,
            l_scratch_ref,
            acc_scratch_ref,
            m_out_ref,
            l_out_ref,
            acc_out_ref,
            None,
            pcp_size=pcp_size,
            lane=lane,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            kv_heads=kv_heads,
            packed_kv_groups=packed_kv_groups,
            kv_packing=kv_packing,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
            causal=True,
            state_mode="current",
        )


def _pcp_streaming_attention_history_output_page_groups_multi_head_kernel(
    active_page_groups_ref,
    q_ref,
    kv_cache_ref,
    packed_schedule_ref,
    m_in_ref,
    l_in_ref,
    acc_in_ref,
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
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    pcp_size,
    num_lanes,
    q_compute_size,
    num_page_groups,
    sm_scale,
    kv_heads,
    packed_kv_groups,
    kv_packing,
    kv_pages_per_block,
    k_scale,
    v_scale,
    mesh_axis_names,
    pcp_axis_name,
):
    """History no-causal PCP body initialized from current state.

    The output has the same shape as Q and is aliased to the Q input by the
    split-path Pallas call. This body only stores O after all history KV has
    been consumed, so the aliased HBM buffer is not overwritten while Q is
    still needed for QK.
    """
    for lane in range(num_lanes):
        active_page_groups = jnp.minimum(active_page_groups_ref[0],
                                         num_page_groups)
        _run_pcp_page_groups_multi_head(
            active_page_groups,
            q_ref,
            kv_cache_ref,
            packed_schedule_ref,
            sched_dma_sem,
            local_dma_sem,
            remote_send_sems,
            remote_recv_sems,
            remote_sync_sems,
            group_slot_free_sems,
            sched_vmem_ref,
            q_vmem_ref,
            kv_vmem_ref,
            m_scratch_ref,
            l_scratch_ref,
            acc_scratch_ref,
            m_in_ref,
            l_in_ref,
            acc_in_ref,
            o_ref,
            pcp_size=pcp_size,
            lane=lane,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            kv_heads=kv_heads,
            packed_kv_groups=packed_kv_groups,
            kv_packing=kv_packing,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
            causal=False,
            state_mode="history",
        )


def _pcp_streaming_attention_zero_history_output_multi_head_kernel(
    active_page_groups_ref,
    o_in_ref,
    packed_schedule_ref,
    m_in_ref,
    l_in_ref,
    acc_in_ref,
    o_ref,
    sched_dma_sem,
    local_dma_sem,
    sched_vmem_ref,
    o_vmem_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    pcp_size,
    num_lanes,
    num_page_groups,
    pcp_axis_name,
):
    """Write outputs for tiles whose current pass already covered all KV."""
    del o_in_ref
    my_id = lax.axis_index(pcp_axis_name)

    for lane in range(num_lanes):
        active_page_groups = jnp.minimum(active_page_groups_ref[0],
                                         num_page_groups)

        def _page_group_loop(group_idx, carry):
            del carry
            group_start = group_idx * pcp_size
            _load_schedule_step(packed_schedule_ref, sched_vmem_ref,
                                sched_dma_sem, group_start)
            group_req_id = sched_vmem_ref[my_id, lane, ScheduleField.REQ_ID]
            group_load_q = sched_vmem_ref[my_id, lane,
                                          ScheduleField.LOAD_Q] != 0
            group_kv_global_start = sched_vmem_ref[
                my_id, lane, ScheduleField.KV_GLOBAL_START]
            group_q_hbm_offset = sched_vmem_ref[my_id, lane,
                                                ScheduleField.Q_HBM_OFFSET]
            group_o_hbm_offset = sched_vmem_ref[my_id, lane,
                                                ScheduleField.O_HBM_OFFSET]
            group_q_tile_size = sched_vmem_ref[my_id, lane,
                                               ScheduleField.Q_TILE_SIZE]
            write_zero_history = jnp.logical_and(
                jnp.logical_and(group_req_id != -1, group_load_q),
                group_kv_global_start == 0,
            )
            store_q_tile_size = jnp.where(write_zero_history,
                                          group_q_tile_size, 0)

            @pl.when(store_q_tile_size > 0)
            def _write_zero_history_output():
                _load_multi_head_online_state_tile(
                    m_in_ref,
                    l_in_ref,
                    acc_in_ref,
                    local_dma_sem,
                    m_scratch_ref,
                    l_scratch_ref,
                    acc_scratch_ref,
                    group_q_hbm_offset,
                    store_q_tile_size,
                )
                _write_multi_head_output_from_state(o_vmem_ref, l_scratch_ref,
                                                    acc_scratch_ref)
                _store_compact_output_tile_multi_head(
                    o_ref,
                    o_vmem_ref,
                    local_dma_sem,
                    group_o_hbm_offset,
                    store_q_tile_size,
                )

            return jnp.array(0, dtype=jnp.int32)

        lax.fori_loop(
            0,
            active_page_groups,
            _page_group_loop,
            jnp.array(0, dtype=jnp.int32),
            unroll=False,
        )


def _pcp_streaming_attention_current_state_page_groups_multi_head_pallas_call(
        q_multi_head,
        kv_cache_local,
        packed_schedule,
        active_page_groups,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        q_compute_size: int | None = None,
        kv_pages_per_block: int = 1,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    page_size = kv_cache_local.shape[2]
    local_tokens = q_multi_head.shape[0]
    kv_heads = q_multi_head.shape[1]
    q_per_kv = q_multi_head.shape[2]
    head_dim = q_multi_head.shape[-1]
    packed_kv_groups = kv_cache_local.shape[3]
    kv_packing = kv_cache_local.shape[4]
    num_page_groups = packed_schedule.shape[0] // pcp_size
    num_lanes = packed_schedule.shape[2]
    q_compute_size = _normalize_q_compute_size(q_block_size, q_compute_size)
    active_page_groups = jnp.asarray(active_page_groups, dtype=jnp.int32)
    if active_page_groups.shape == ():
        active_page_groups = active_page_groups[None]

    state_lm_hbm_shape = (
        kv_heads,
        local_tokens,
        q_per_kv,
        ONLINE_STATE_PADDED_LANES,
    )
    state_acc_hbm_shape = (kv_heads, local_tokens, q_per_kv, head_dim)
    state_lm_scratch_shape = (
        kv_heads,
        q_block_size,
        q_per_kv,
        ONLINE_STATE_PADDED_LANES,
    )
    state_acc_scratch_shape = (kv_heads, q_block_size, q_per_kv, head_dim)
    return pl.pallas_call(
        functools.partial(
            _pcp_streaming_attention_current_state_page_groups_multi_head_kernel,
            pcp_size=pcp_size,
            num_lanes=num_lanes,
            q_compute_size=q_compute_size,
            num_page_groups=num_page_groups,
            sm_scale=sm_scale,
            kv_heads=kv_heads,
            packed_kv_groups=packed_kv_groups,
            kv_packing=kv_packing,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(state_lm_hbm_shape, jnp.float32),
            jax.ShapeDtypeStruct(state_lm_hbm_shape, jnp.float32),
            jax.ShapeDtypeStruct(state_acc_hbm_shape, jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
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
                pltpu.VMEM((2, kv_pages_per_block, page_size, packed_kv_groups,
                            kv_packing, head_dim), kv_cache_local.dtype),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_acc_scratch_shape, jnp.float32),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes),
        name="pcp_streaming_attention_current_state_page_groups_multi_head",
    )(active_page_groups, q_multi_head, kv_cache_local, packed_schedule)


def _pcp_streaming_attention_history_output_page_groups_multi_head_pallas_call(
        q_multi_head,
        kv_cache_local,
        packed_schedule,
        m_state,
        l_state,
        acc_state,
        active_page_groups,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        q_compute_size: int | None = None,
        kv_pages_per_block: int = 1,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    page_size = kv_cache_local.shape[2]
    kv_heads = q_multi_head.shape[1]
    q_per_kv = q_multi_head.shape[2]
    head_dim = q_multi_head.shape[-1]
    packed_kv_groups = kv_cache_local.shape[3]
    kv_packing = kv_cache_local.shape[4]
    num_page_groups = packed_schedule.shape[0] // pcp_size
    num_lanes = packed_schedule.shape[2]
    q_compute_size = _normalize_q_compute_size(q_block_size, q_compute_size)
    active_page_groups = jnp.asarray(active_page_groups, dtype=jnp.int32)
    if active_page_groups.shape == ():
        active_page_groups = active_page_groups[None]

    state_lm_scratch_shape = (
        kv_heads,
        q_block_size,
        q_per_kv,
        ONLINE_STATE_PADDED_LANES,
    )
    state_acc_scratch_shape = (kv_heads, q_block_size, q_per_kv, head_dim)
    return pl.pallas_call(
        functools.partial(
            _pcp_streaming_attention_history_output_page_groups_multi_head_kernel,
            pcp_size=pcp_size,
            num_lanes=num_lanes,
            q_compute_size=q_compute_size,
            num_page_groups=num_page_groups,
            sm_scale=sm_scale,
            kv_heads=kv_heads,
            packed_kv_groups=packed_kv_groups,
            kv_packing=kv_packing,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ),
        out_shape=jax.ShapeDtypeStruct(q_multi_head.shape, q_multi_head.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
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
                pltpu.VMEM((2, kv_pages_per_block, page_size, packed_kv_groups,
                            kv_packing, head_dim), kv_cache_local.dtype),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_acc_scratch_shape, jnp.float32),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes),
        input_output_aliases={1: 0},
        name="pcp_streaming_attention_history_output_page_groups_multi_head",
    )(active_page_groups, q_multi_head, kv_cache_local, packed_schedule,
      m_state, l_state, acc_state)


def _pcp_streaming_attention_zero_history_output_multi_head_pallas_call(
    o_multi_head,
    packed_schedule,
    m_state,
    l_state,
    acc_state,
    active_page_groups,
    *,
    pcp_size: int,
    q_block_size: int,
    pcp_axis_name: str = AXIS,
):
    kv_heads = o_multi_head.shape[1]
    q_per_kv = o_multi_head.shape[2]
    head_dim = o_multi_head.shape[-1]
    num_page_groups = packed_schedule.shape[0] // pcp_size
    num_lanes = packed_schedule.shape[2]
    active_page_groups = jnp.asarray(active_page_groups, dtype=jnp.int32)
    if active_page_groups.shape == ():
        active_page_groups = active_page_groups[None]

    state_lm_scratch_shape = (
        kv_heads,
        q_block_size,
        q_per_kv,
        ONLINE_STATE_PADDED_LANES,
    )
    state_acc_scratch_shape = (kv_heads, q_block_size, q_per_kv, head_dim)
    return pl.pallas_call(
        functools.partial(
            _pcp_streaming_attention_zero_history_output_multi_head_kernel,
            pcp_size=pcp_size,
            num_lanes=num_lanes,
            num_page_groups=num_page_groups,
            pcp_axis_name=pcp_axis_name,
        ),
        out_shape=jax.ShapeDtypeStruct(o_multi_head.shape, o_multi_head.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(
                    (pcp_size, num_lanes, ScheduleField.PACKED_NUM_FIELDS),
                    packed_schedule.dtype),
                pltpu.VMEM((q_block_size, kv_heads, q_per_kv, head_dim),
                           o_multi_head.dtype),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_acc_scratch_shape, jnp.float32),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes, ),
        input_output_aliases={1: 0},
        name="pcp_streaming_attention_zero_history_output_multi_head",
    )(active_page_groups, o_multi_head, packed_schedule, m_state, l_state,
      acc_state)


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
        q_compute_size: int | None = None,
        max_context_tokens: int | None = None,
        num_lanes: int = 1,
        kv_pages_per_block: int = 1,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Generate the narrow metadata schedule and run split PCP RPA.

    The runtime schedule shape is derived from the local compile bucket
    ``q_local.shape[0] * pcp_size`` and, when provided, max_context_tokens
    instead of the full KV cache capacity. Execution is split into a current
    causal pass that produces online-softmax state and a history no-causal pass
    that consumes that state and writes the final output.

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
        q_compute_size: Static Q rows consumed per multi-head softmax compute
            step. Defaults to `q_block_size`.
        sm_scale: Softmax scale applied to QK scores.
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
    _validate_packed_kv_layout(q_local,
                               kv_cache_local,
                               k_scale=k_scale,
                               v_scale=v_scale)

    page_size = kv_cache_local.shape[1]
    (current_schedule, current_active_page_groups, history_schedule,
     history_active_page_groups) = (
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

    m_state, l_state, acc_state = (
        _pcp_streaming_attention_current_state_page_groups_multi_head_pallas_call(
            q_local,
            kv_cache_local[None, ...],
            current_schedule,
            current_active_page_groups,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ))
    history_output = (
        _pcp_streaming_attention_history_output_page_groups_multi_head_pallas_call(
            q_local,
            kv_cache_local[None, ...],
            history_schedule,
            m_state,
            l_state,
            acc_state,
            history_active_page_groups,
            pcp_size=pcp_size,
            q_block_size=q_block_size,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            kv_pages_per_block=kv_pages_per_block,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ))
    return _pcp_streaming_attention_zero_history_output_multi_head_pallas_call(
        history_output,
        current_schedule,
        m_state,
        l_state,
        acc_state,
        current_active_page_groups,
        pcp_size=pcp_size,
        q_block_size=q_block_size,
        pcp_axis_name=pcp_axis_name,
    )
