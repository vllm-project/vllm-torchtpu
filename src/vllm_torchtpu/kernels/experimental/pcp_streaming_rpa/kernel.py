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

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import (
    broadcast_minor, convert_to_target_bitwidth, get_dtype_packing,
    strided_load)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField, TilePlanField,
    build_pcp_streaming_schedule_inputs_from_metadata_jax)

AXIS = "pcp"
ONLINE_STATE_INIT_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE = 512
TPU_HBM_ROW_TILE = 16
ONLINE_STATE_PADDED_LANES = 128
TPU_KV_DMA_ALIGNMENT = 128
KVLayout = batched_rpa_configs.KVLayout


def _pcp_write_captured_kv_kernel(
    num_segments_ref,
    segment_descriptors_ref,
    captured_kv_ref,
    kv_cache_in_ref,
    kv_cache_out_ref,
    dma_sem,
):
    """Scatter owner-local captured rows into a donated flat cache."""
    del kv_cache_in_ref
    num_segments = jnp.clip(num_segments_ref[0], 0,
                            segment_descriptors_ref.shape[1])

    @pl.loop(0, num_segments)
    def _start_segment(segment_idx):
        source_start = segment_descriptors_ref[0, segment_idx]
        destination_start = segment_descriptors_ref[1, segment_idx]
        segment_len = segment_descriptors_ref[2, segment_idx]
        copy_op = pltpu.make_async_copy(
            src_ref=captured_kv_ref.at[pl.ds(source_start, segment_len)],
            dst_ref=kv_cache_out_ref.at[pl.ds(destination_start, segment_len)],
            sem=dma_sem,
        )
        copy_op.start()

    @pl.loop(0, num_segments)
    def _wait_segment(segment_idx):
        source_start = segment_descriptors_ref[0, segment_idx]
        destination_start = segment_descriptors_ref[1, segment_idx]
        segment_len = segment_descriptors_ref[2, segment_idx]
        copy_op = pltpu.make_async_copy(
            src_ref=captured_kv_ref.at[pl.ds(source_start, segment_len)],
            dst_ref=kv_cache_out_ref.at[pl.ds(destination_start, segment_len)],
            sem=dma_sem,
        )
        copy_op.wait()


def _pcp_write_captured_kv_seq_along_lane_kernel(
    num_segments_ref,
    segment_descriptors_ref,
    captured_kv_ref,
    kv_cache_in_ref,
    kv_cache_out_ref,
    dma_sem,
):
    """Scatter sequence-last captured KV into page-bounded cache slices."""
    del kv_cache_in_ref
    num_segments = jnp.clip(num_segments_ref[0], 0,
                            segment_descriptors_ref.shape[1])
    page_size = kv_cache_out_ref.shape[-1]

    def _copy_op(segment_idx):
        source_start = pl.multiple_of(segment_descriptors_ref[0, segment_idx],
                                      TPU_KV_DMA_ALIGNMENT)
        destination_start = segment_descriptors_ref[1, segment_idx]
        segment_len = pl.multiple_of(segment_descriptors_ref[2, segment_idx],
                                     TPU_KV_DMA_ALIGNMENT)
        destination_page = destination_start // page_size
        destination_offset = pl.multiple_of(destination_start % page_size,
                                            TPU_KV_DMA_ALIGNMENT)
        return pltpu.make_async_copy(
            src_ref=captured_kv_ref.at[
                :,
                :,
                :,
                pl.ds(source_start, segment_len),
            ],
            dst_ref=kv_cache_out_ref.at[
                destination_page,
                :,
                :,
                :,
                pl.ds(destination_offset, segment_len),
            ],
            sem=dma_sem,
        )

    @pl.loop(0, num_segments)
    def _start_segment(segment_idx):
        _copy_op(segment_idx).start()

    @pl.loop(0, num_segments)
    def _wait_segment(segment_idx):
        _copy_op(segment_idx).wait()


def write_captured_kv_to_local_cache(
    kv_cache: jax.Array,
    captured_kv: jax.Array,
    segment_descriptors: jax.Array,
    num_segments: jax.Array,
    *,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    """Write ring-captured KV with one single-output donated Pallas call."""
    if kv_cache.ndim < 2:
        raise ValueError(
            f"KV cache must have at least two dims: {kv_cache.shape}.")
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if captured_kv.shape[:-1] != kv_cache.shape[1:-1]:
            raise ValueError(
                "SEQ_ALONG_LANE captured KV prefix must match the paged cache "
                f"head layout: {captured_kv.shape[:-1]} vs "
                f"{kv_cache.shape[1:-1]}.")
    elif captured_kv.shape[1:] != kv_cache.shape[2:]:
        raise ValueError("Captured KV tail must match the paged cache tail: "
                         f"{captured_kv.shape[1:]} vs {kv_cache.shape[2:]}.")
    if segment_descriptors.ndim != 2 or segment_descriptors.shape[0] != 3:
        raise ValueError(
            "KV writeback segment descriptors must have shape [3, segments], "
            f"got {segment_descriptors.shape}.")

    num_segments = jnp.asarray(num_segments, dtype=jnp.int32).reshape((1, ))
    segment_descriptors = jnp.asarray(segment_descriptors, dtype=jnp.int32)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        write_kernel = _pcp_write_captured_kv_seq_along_lane_kernel
        writeback_cache = kv_cache
    else:
        write_kernel = _pcp_write_captured_kv_kernel
        writeback_cache = kv_cache.reshape((-1, *kv_cache.shape[2:]))
    updated_cache = pl.pallas_call(
        write_kernel,
        out_shape=jax.ShapeDtypeStruct(writeback_cache.shape,
                                       writeback_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(pltpu.SemaphoreType.DMA, ),
            grid=(1, ),
        ),
        input_output_aliases={3: 0},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes),
        name="pcp_write_captured_kv_to_local_cache",
    )(num_segments, segment_descriptors, captured_kv, writeback_cache)
    return updated_cache.reshape(kv_cache.shape)


def _validate_packed_kv_layout(
        q_local,
        kv_cache_local,
        *,
        kv_layout: KVLayout = (KVLayout.HEAD_ALONG_SUBLANE),
        k_scale: float | None = None,
        v_scale: float | None = None):
    """Validate PCP's physical packed KV layout against logical Q heads."""
    if q_local.ndim != 4:
        raise ValueError("PCP q_local must have shape "
                         "[tokens, kv_heads, q_per_kv, head_dim], got "
                         f"{q_local.shape}.")
    if kv_cache_local.ndim != 5:
        raise ValueError("PCP kv_cache_local must be 5D, got "
                         f"{kv_cache_local.shape}.")

    kv_heads = q_local.shape[1]
    q_per_kv = q_local.shape[2]
    head_dim = q_local.shape[-1]
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
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        kv_layout_groups = kv_cache_local.shape[1]
        if kv_layout_groups != 2 * kv_heads:
            raise ValueError(
                "PCP SEQ_ALONG_LANE cache must contain one K and V plane per "
                f"KV head: planes={kv_layout_groups}, expected={2 * kv_heads}."
            )
        if kv_cache_local.shape[2] * kv_packing != head_dim:
            raise ValueError(
                "PCP SEQ_ALONG_LANE cache head_dim must match Q head_dim: "
                f"{kv_cache_local.shape[2]} * {kv_packing} vs {head_dim}.")
        if kv_packing != 4:
            raise NotImplementedError(
                "PCP SEQ_ALONG_LANE currently supports only FP8 KV cache, "
                f"got packing={kv_packing}.")
    else:
        kv_layout_groups = kv_cache_local.shape[2]
        expected_packed_kv_groups = math.ceil((2 * kv_heads) / kv_packing)
        if kv_layout_groups != expected_packed_kv_groups:
            raise ValueError(
                "PCP KV cache packed layout does not match logical KV heads: "
                f"packed_kv_groups={kv_layout_groups}, "
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
    return kv_heads, kv_layout_groups, kv_packing


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
    interleave_size,
    q_compute_size,
    k_scale,
    v_scale,
    causal,
    kv_layout,
):
    if causal:
        q_global_start = sched_vmem_ref[consumer_rank, lane,
                                        RuntimeScheduleField.Q_GLOBAL_START]
        kv_global_start = sched_vmem_ref[consumer_rank, lane,
                                         RuntimeScheduleField.KV_GLOBAL_START]
        q_tile_size = sched_vmem_ref[consumer_rank, lane,
                                     RuntimeScheduleField.Q_TILE_SIZE]
        req_id = sched_vmem_ref[consumer_rank, lane,
                                RuntimeScheduleField.REQ_ID]
        kv_valid_len = sched_vmem_ref[consumer_rank, lane,
                                      RuntimeScheduleField.KV_VALID_LEN]
    q_per_kv = q_vmem_ref.shape[2]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        kv_block_tokens = kv_vmem_ref.shape[-1]
    else:
        page_size = kv_vmem_ref.shape[2]
        kv_block_tokens = kv_vmem_ref.shape[1] * page_size
    num_q_compute_blocks = q_vmem_ref.shape[0] // q_compute_size
    compute_dtype = q_vmem_ref.dtype

    def _consume_kv_head(k, v, kv_head_idx):
        if k_scale is not None:
            k = (k.astype(jnp.float32) * k_scale).astype(compute_dtype)
        if v_scale is not None:
            v = (v.astype(jnp.float32) * v_scale).astype(compute_dtype)
        if not causal:
            k = k * sm_scale

        def _q_compute_loop(q_compute_idx, carry):
            del carry
            k_compute = k
            v_compute = v
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

            if causal:
                q_row = (
                    q_start +
                    lax.broadcasted_iota(jnp.int32,
                                         (q_compute_size, q_per_kv, 1), 0))
                kv_local_pos = lax.broadcasted_iota(jnp.int32,
                                                    (1, 1, kv_block_tokens), 2)
                q_valid = q_row < q_tile_size
                kv_valid = kv_local_pos < kv_valid_len
                q_slice = jnp.where(q_valid, q_slice, 0.0)
                kv_sequence_axis = (1 if kv_layout == KVLayout.SEQ_ALONG_LANE
                                    else 0)
                kv_input_pos = lax.broadcasted_iota(jnp.int32, k.shape,
                                                    kv_sequence_axis)
                k_compute = jnp.where(kv_input_pos < kv_valid_len, k, 0.0)
                v_compute = jnp.where(kv_input_pos < kv_valid_len, v, 0.0)

            if kv_layout == KVLayout.SEQ_ALONG_LANE:
                qk_dimensions = (((2, ), (0, )), ((), ()))
            else:
                qk_dimensions = (((2, ), (1, )), ((), ()))
            scores = lax.dot_general(
                q_slice,
                k_compute,
                qk_dimensions,
                preferred_element_type=jnp.float32,
            )

            if causal:
                scores = scores * sm_scale
                q_interleave = interleave_size
                q_chunk_idx = lax.div(q_row, q_interleave)
                q_chunk_offset = lax.rem(q_row, q_interleave)
                q_pos = (q_global_start +
                         q_chunk_idx * pcp_size * q_interleave +
                         q_chunk_offset)
                kv_chunk_idx = lax.div(kv_local_pos, q_interleave)
                kv_chunk_offset = lax.rem(kv_local_pos, q_interleave)
                kv_pos = (kv_global_start +
                          kv_chunk_idx * pcp_size * q_interleave +
                          kv_chunk_offset)
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
            if kv_layout == KVLayout.SEQ_ALONG_LANE:
                pv_dimensions = (((2, ), (1, )), ((), ()))
            else:
                pv_dimensions = (((2, ), (0, )), ((), ()))
            pv = lax.dot_general(
                p.astype(compute_dtype),
                v_compute,
                pv_dimensions,
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

    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        head_dim = kv_vmem_ref.shape[3] * kv_vmem_ref.shape[4]
        for kv_head_idx in range(q_vmem_ref.shape[1]):
            k = kv_vmem_ref[slot, 0, 2 * kv_head_idx, :, :, :].reshape(
                head_dim, kv_block_tokens)
            v = kv_vmem_ref[slot, 0, 2 * kv_head_idx + 1, :, :, :].reshape(
                head_dim, kv_block_tokens)
            _consume_kv_head(k, v, kv_head_idx)
    else:
        heads_per_group = kv_vmem_ref.shape[4] // 2
        for packed_group_idx in range(kv_vmem_ref.shape[3]):
            kv_pairs = _strided_load_packed_kv_group(kv_vmem_ref, slot,
                                                     packed_group_idx)
            for pair_idx, (k, v) in enumerate(kv_pairs):
                kv_head_idx = packed_group_idx * heads_per_group + pair_idx
                if kv_head_idx >= q_vmem_ref.shape[1]:
                    continue
                _consume_kv_head(k, v, kv_head_idx)


def _reset_multi_head_online_state(m_scratch_ref, l_scratch_ref,
                                   acc_scratch_ref):
    # Keep the hot online-softmax loop identical to origin/main while making
    # an all-masked first ring page well-defined: max(finite_min, -inf) stays
    # finite, so both the probability update and the previous-state scale are
    # exact (0 and 1 respectively) without an extra per-page validity mask.
    m_scratch_ref[...] = jnp.full(m_scratch_ref.shape,
                                  ONLINE_STATE_INIT_VALUE,
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


def _load_hbm_row(src_ref, dst_ref, sem, row):
    load_op = pltpu.make_async_copy(
        src_ref=src_ref.at[row],
        dst_ref=dst_ref.at[:],
        sem=sem,
    )
    load_op.start()
    load_op.wait()


def _vector_lookup(values, index):
    values = jnp.reshape(values, (-1, ))
    positions = jnp.arange(values.shape[0], dtype=jnp.int32)
    return jnp.sum(jnp.where(positions == index, values, 0), dtype=jnp.int32)


def _tile_plan_value(plan_values, consumer_rank, field):
    return _vector_lookup(
        plan_values,
        consumer_rank * TilePlanField.NUM_FIELDS + field,
    )


def _tile_plan_value_static(plan_values, consumer_rank, field):
    return plan_values[0, consumer_rank * TilePlanField.NUM_FIELDS + field]


def _group_tile_position(group_starts, active_tiles, group_idx):
    starts = jnp.reshape(group_starts, (-1, ))
    positions = jnp.arange(starts.shape[0], dtype=jnp.int32)
    before_or_at_group = jnp.logical_and(positions < active_tiles, starts
                                         <= group_idx)
    return jnp.maximum(
        jnp.sum(before_or_at_group.astype(jnp.int32), dtype=jnp.int32) - 1,
        0,
    )


def _local_page_valid_len_from_tile_plan(kv_len, local_page_idx, src_rank, *,
                                         page_size, pcp_size, interleave_size):
    cycle = pcp_size * interleave_size
    base = (local_page_idx * page_size * pcp_size + src_rank * interleave_size)
    delta = kv_len - base
    chunks_per_page = page_size // interleave_size
    full_chunks = jnp.minimum(chunks_per_page, jnp.maximum(delta, 0) // cycle)
    partial = jnp.minimum(
        interleave_size,
        jnp.maximum(delta - full_chunks * cycle, 0),
    )
    valid_len = full_chunks * interleave_size + jnp.where(
        full_chunks < chunks_per_page, partial, 0)
    return jnp.where(kv_len > base, valid_len, 0)


def _rank_token_count_before(position, rank, *, pcp_size, interleave_size):
    cycle = pcp_size * interleave_size
    full_cycles = position // cycle
    cycle_offset = position - full_cycles * cycle
    rank_start = rank * interleave_size
    partial = jnp.clip(cycle_offset - rank_start, 0, interleave_size)
    return full_cycles * interleave_size + partial


def _stage_current_schedule_step_from_tile_plan(
    plan_values,
    block_table_values,
    sched_vmem_ref,
    group_in_tile,
    source_rank,
    *,
    pcp_size,
    page_size,
    interleave_size,
    lane,
):
    req_id = _tile_plan_value_static(plan_values, 0, TilePlanField.REQ_ID)
    token_owner_start = _tile_plan_value_static(
        plan_values, 0, TilePlanField.TOKEN_OWNER_START)
    request_absolute_start = _tile_plan_value_static(
        plan_values, 0, TilePlanField.REQUEST_ABSOLUTE_QUERY_START)
    current_effective_len = _tile_plan_value_static(
        plan_values, 0, TilePlanField.CURRENT_EFFECTIVE_LEN)
    capture_current_kv = _tile_plan_value_static(
        plan_values, 0, TilePlanField.CAPTURE_CURRENT_KV)

    cycle = pcp_size * interleave_size
    virtual_page_size = page_size * pcp_size
    history_boundary_present = (request_absolute_start % virtual_page_size
                                != 0)
    is_history_boundary = jnp.logical_and(history_boundary_present,
                                          group_in_tile == 0)
    fresh_group_in_tile = jnp.maximum(
        group_in_tile - history_boundary_present.astype(jnp.int32), 0)

    cycle_start = (token_owner_start // cycle * cycle +
                   fresh_group_in_tile * cycle)
    source_chunk_start = cycle_start + source_rank * interleave_size
    token_owner_end = token_owner_start + current_effective_len
    overlap_start = jnp.maximum(source_chunk_start, token_owner_start)
    overlap_end = jnp.minimum(source_chunk_start + interleave_size,
                              token_owner_end)
    fresh_kv_valid_len = jnp.maximum(overlap_end - overlap_start, 0)
    valid_fresh_chunk = fresh_kv_valid_len > 0

    request_local_hbm_start = _tile_plan_value(
        plan_values, source_rank, TilePlanField.CURRENT_KV_HBM_START)
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
    fresh_kv_hbm_offset = (request_local_hbm_start +
                           source_rows_before_overlap)
    fresh_kv_global_start = (request_absolute_start + overlap_start -
                             token_owner_start)

    boundary_local_page_idx = request_absolute_start // virtual_page_size
    safe_boundary_local_page_idx = jnp.clip(boundary_local_page_idx, 0,
                                            block_table_values.size - 1)
    boundary_page_idx = _vector_lookup(block_table_values,
                                       safe_boundary_local_page_idx)
    boundary_kv_global_start = (boundary_local_page_idx * virtual_page_size +
                                source_rank * interleave_size)
    boundary_kv_valid_len = _local_page_valid_len_from_tile_plan(
        request_absolute_start,
        boundary_local_page_idx,
        source_rank,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    valid_boundary_page = boundary_kv_valid_len > 0

    valid_scheduled_kv = jnp.where(is_history_boundary, valid_boundary_page,
                                   valid_fresh_chunk)
    kv_page_idx = jnp.where(is_history_boundary, boundary_page_idx, 0)
    kv_global_start = jnp.where(is_history_boundary, boundary_kv_global_start,
                                fresh_kv_global_start)
    kv_valid_len = jnp.where(is_history_boundary, boundary_kv_valid_len,
                             fresh_kv_valid_len)
    # Negative KV_HBM_OFFSET is the single internal source tag: the current
    # causal pass loads this group from the old paged cache. Fresh groups use
    # their compact current-KV HBM row offset.
    kv_hbm_offset = jnp.where(is_history_boundary, -1, fresh_kv_hbm_offset)

    safe_global_start = jnp.where(valid_fresh_chunk, fresh_kv_global_start, 0)
    capture_owner_0 = ((safe_global_start // interleave_size) % pcp_size)
    next_owner_boundary = (safe_global_start // interleave_size +
                           1) * interleave_size
    capture_len_0 = jnp.minimum(
        fresh_kv_valid_len,
        jnp.maximum(next_owner_boundary - safe_global_start, 0),
    )
    capture_len_0 = jnp.where(capture_current_kv != 0, capture_len_0, 0)
    capture_len_1 = jnp.where(capture_current_kv != 0,
                              fresh_kv_valid_len - capture_len_0, 0)
    capture_len_0 = jnp.where(is_history_boundary, 0, capture_len_0)
    capture_len_1 = jnp.where(is_history_boundary, 0, capture_len_1)
    capture_global_start_1 = safe_global_start + capture_len_0
    capture_owner_1 = ((capture_global_start_1 // interleave_size) % pcp_size)

    capture_prefix_0 = _tile_plan_value(plan_values, capture_owner_0,
                                        TilePlanField.WRITEBACK_HBM_PREFIX)
    capture_prefix_1 = _tile_plan_value(plan_values, capture_owner_1,
                                        TilePlanField.WRITEBACK_HBM_PREFIX)
    capture_offset_0 = capture_prefix_0 + (_rank_token_count_before(
        safe_global_start,
        capture_owner_0,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    ) - _rank_token_count_before(
        request_absolute_start,
        capture_owner_0,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    ))
    capture_offset_1 = capture_prefix_1 + (_rank_token_count_before(
        capture_global_start_1,
        capture_owner_1,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    ) - _rank_token_count_before(
        request_absolute_start,
        capture_owner_1,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    ))

    for consumer_rank in range(pcp_size):
        q_global_start = _tile_plan_value_static(plan_values, consumer_rank,
                                                 TilePlanField.Q_GLOBAL_START)
        q_hbm_offset = _tile_plan_value_static(plan_values, consumer_rank,
                                               TilePlanField.Q_HBM_OFFSET)
        q_tile_size = _tile_plan_value_static(plan_values, consumer_rank,
                                              TilePlanField.Q_TILE_SIZE)
        owns_capture_0 = jnp.logical_and(capture_len_0 > 0,
                                         capture_owner_0 == consumer_rank)
        owns_capture_1 = jnp.logical_and(capture_len_1 > 0,
                                         capture_owner_1 == consumer_rank)
        capture_src_offset = jnp.where(
            owns_capture_0, 0, jnp.where(owns_capture_1, capture_len_0, 0))
        capture_dst_offset = jnp.where(
            owns_capture_0, capture_offset_0,
            jnp.where(owns_capture_1, capture_offset_1, 0))
        capture_len = jnp.where(
            owns_capture_0,
            capture_len_0 + jnp.where(owns_capture_1, capture_len_1, 0),
            jnp.where(owns_capture_1, capture_len_1, 0),
        )
        logical_fields = jnp.stack((
            jnp.where(valid_scheduled_kv, req_id, -1),
            jnp.where(valid_scheduled_kv, kv_page_idx, 0),
            q_global_start,
            kv_global_start,
            kv_valid_len,
            q_hbm_offset,
            q_tile_size,
            kv_hbm_offset,
            capture_src_offset,
            capture_dst_offset,
            capture_len,
        ))
        packed_row = jnp.concatenate((
            logical_fields,
            jnp.zeros((RuntimeScheduleField.PACKED_NUM_FIELDS -
                       RuntimeScheduleField.NUM_FIELDS, ),
                      dtype=jnp.int32),
        ))
        sched_vmem_ref[consumer_rank, lane, :] = packed_row


def _stage_history_schedule_step_from_tile_plan(
    plan_values,
    block_table_values,
    sched_vmem_ref,
    group_in_tile,
    source_rank,
    *,
    pcp_size,
    page_size,
    interleave_size,
    lane,
):
    req_id = _tile_plan_value_static(plan_values, 0, TilePlanField.REQ_ID)
    history_token_count = _tile_plan_value_static(
        plan_values, 0, TilePlanField.REQUEST_ABSOLUTE_QUERY_START)
    local_page_idx = group_in_tile
    safe_local_page_idx = jnp.clip(local_page_idx, 0,
                                   block_table_values.size - 1)
    page_idx = _vector_lookup(block_table_values, safe_local_page_idx)
    kv_global_start = (local_page_idx * page_size * pcp_size +
                       source_rank * interleave_size)
    kv_valid_len = _local_page_valid_len_from_tile_plan(
        history_token_count,
        local_page_idx,
        source_rank,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    valid_page = kv_valid_len > 0

    for consumer_rank in range(pcp_size):
        q_global_start = _tile_plan_value_static(plan_values, consumer_rank,
                                                 TilePlanField.Q_GLOBAL_START)
        q_hbm_offset = _tile_plan_value_static(plan_values, consumer_rank,
                                               TilePlanField.Q_HBM_OFFSET)
        q_tile_size = _tile_plan_value_static(plan_values, consumer_rank,
                                              TilePlanField.Q_TILE_SIZE)
        logical_fields = jnp.stack((
            jnp.where(valid_page, req_id, -1),
            jnp.where(valid_page, page_idx, 0),
            q_global_start,
            kv_global_start,
            kv_valid_len,
            q_hbm_offset,
            q_tile_size,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        ))
        packed_row = jnp.concatenate((
            logical_fields,
            jnp.zeros((RuntimeScheduleField.PACKED_NUM_FIELDS -
                       RuntimeScheduleField.NUM_FIELDS, ),
                      dtype=jnp.int32),
        ))
        sched_vmem_ref[consumer_rank, lane, :] = packed_row


def _stage_schedule_step_from_tile_plan(
    plan_values,
    block_table_values,
    sched_vmem_ref,
    group_in_tile,
    source_rank,
    *,
    state_mode,
    pcp_size,
    page_size,
    interleave_size,
    lane,
):
    if state_mode == "current":
        _stage_current_schedule_step_from_tile_plan(
            plan_values,
            block_table_values,
            sched_vmem_ref,
            group_in_tile,
            source_rank,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            lane=lane,
        )
    elif state_mode == "history":
        _stage_history_schedule_step_from_tile_plan(
            plan_values,
            block_table_values,
            sched_vmem_ref,
            group_in_tile,
            source_rank,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            lane=lane,
        )
    else:
        raise ValueError(f"Unsupported PCP state mode: {state_mode}")


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


def _load_local_kv_page_all_heads_packed(kv_cache_ref, kv_vmem_ref, sem,
                                         sched_vmem_ref, lane, *,
                                         packed_kv_groups, kv_layout):
    local_page_idx = sched_vmem_ref[0, lane, RuntimeScheduleField.KV_PAGE_IDX]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        load = pltpu.make_async_copy(
            src_ref=kv_cache_ref.at[0, local_page_idx, :, :, :, :],
            dst_ref=kv_vmem_ref.at[0, 0, :, :, :, :],
            sem=sem,
        )
        load.start()
        load.wait()
        return
    load = pltpu.make_async_copy(
        src_ref=kv_cache_ref.at[
            0,
            local_page_idx,
            :,
            pl.ds(0, packed_kv_groups),
            :,
            :,
        ],
        dst_ref=kv_vmem_ref.at[0, 0],
        sem=sem,
    )
    load.start()
    load.wait()


def _load_local_current_or_boundary_kv_all_heads_packed(
    current_kv_ref,
    kv_cache_ref,
    kv_vmem_ref,
    sem,
    sched_vmem_ref,
    lane,
    *,
    packed_kv_groups,
    kv_layout,
):
    kv_hbm_offset = sched_vmem_ref[0, lane, RuntimeScheduleField.KV_HBM_OFFSET]
    kv_valid_len = sched_vmem_ref[0, lane, RuntimeScheduleField.KV_VALID_LEN]
    load_history_boundary = kv_hbm_offset < 0
    local_page_idx = sched_vmem_ref[0, lane, RuntimeScheduleField.KV_PAGE_IDX]

    @pl.when(load_history_boundary)
    def _load_history_boundary_page():
        if kv_layout == KVLayout.SEQ_ALONG_LANE:
            load = pltpu.make_async_copy(
                src_ref=kv_cache_ref.at[0, local_page_idx, :, :, :, :],
                dst_ref=kv_vmem_ref.at[0, 0, :, :, :, :],
                sem=sem,
            )
            load.start()
            load.wait()
            return
        load = pltpu.make_async_copy(
            src_ref=kv_cache_ref.at[
                0,
                local_page_idx,
                :,
                pl.ds(0, packed_kv_groups),
                :,
                :,
            ],
            dst_ref=kv_vmem_ref.at[0, 0],
            sem=sem,
        )
        load.start()
        load.wait()

    @pl.when(jnp.logical_and(~load_history_boundary, kv_valid_len > 0))
    def _load_current_kv():
        if kv_layout == KVLayout.SEQ_ALONG_LANE:
            seq_hbm_offset = pl.multiple_of(kv_hbm_offset,
                                            TPU_KV_DMA_ALIGNMENT)
            seq_valid_len = pl.multiple_of(kv_valid_len, TPU_KV_DMA_ALIGNMENT)
            load = pltpu.make_async_copy(
                src_ref=current_kv_ref.at[
                    :,
                    :,
                    :,
                    pl.ds(seq_hbm_offset, seq_valid_len),
                ],
                dst_ref=kv_vmem_ref.at[
                    0,
                    0,
                    :,
                    :,
                    :,
                    pl.ds(0, seq_valid_len),
                ],
                sem=sem,
            )
            load.start()
            load.wait()
            return
        load = pltpu.make_async_copy(
            src_ref=current_kv_ref.at[
                pl.ds(kv_hbm_offset, kv_valid_len),
                pl.ds(0, packed_kv_groups),
                :,
                :,
            ],
            dst_ref=kv_vmem_ref.at[
                0,
                0,
                pl.ds(0, kv_valid_len),
                pl.ds(0, packed_kv_groups),
                :,
                :,
            ],
            sem=sem,
        )
        load.start()
        load.wait()


def _mesh_device_id(mesh_axis_names, pcp_axis_name, pcp_rank):
    return tuple(
        pcp_rank if axis_name == pcp_axis_name else lax.axis_index(axis_name)
        for axis_name in mesh_axis_names)


def _run_pcp_page_groups_multi_head(
    active_tiles,
    active_page_groups,
    q_ref,
    current_kv_ref,
    kv_cache_ref,
    captured_kv_ref,
    tile_plan_ref,
    block_tables_ref,
    tile_ids_ref,
    group_starts_ref,
    sched_dma_sem,
    local_dma_sem,
    capture_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    remote_sync_sems,
    group_slot_free_sems,
    plan_vmem_ref,
    block_table_vmem_ref,
    tile_ids_vmem_ref,
    group_starts_vmem_ref,
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
    interleave_size,
    lane,
    q_compute_size,
    sm_scale,
    packed_kv_groups,
    page_size,
    kv_layout,
    k_scale,
    v_scale,
    mesh_axis_names,
    pcp_axis_name,
    state_mode,
):
    if state_mode not in ("current", "history"):
        raise ValueError(f"Unsupported PCP state mode: {state_mode}")
    causal = state_mode == "current"
    my_id = lax.axis_index(pcp_axis_name)
    next_rank = lax.rem(my_id + 1, pcp_size)
    prev_rank = lax.rem(my_id + pcp_size - 1, pcp_size)
    next_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, next_rank)
    prev_device_id = _mesh_device_id(mesh_axis_names, pcp_axis_name, prev_rank)

    ids_load = pltpu.make_async_copy(
        src_ref=tile_ids_ref.at[:, :],
        dst_ref=tile_ids_vmem_ref.at[:, :],
        sem=sched_dma_sem,
    )
    # TODO: Overlap these independent metadata DMAs by starting both copies
    # before waiting when schedule staging becomes a material kernel cost.
    ids_load.start()
    ids_load.wait()
    starts_load = pltpu.make_async_copy(
        src_ref=group_starts_ref.at[:, :],
        dst_ref=group_starts_vmem_ref.at[:, :],
        sem=sched_dma_sem,
    )
    starts_load.start()
    starts_load.wait()
    tile_ids = tile_ids_vmem_ref[...]
    group_starts = group_starts_vmem_ref[...]

    def _run_page_group(group_idx, kv_group_ref):
        tile_pos = _group_tile_position(group_starts, active_tiles, group_idx)
        tile_id = _vector_lookup(tile_ids, tile_pos)
        tile_group_start = _vector_lookup(group_starts, tile_pos)
        group_in_tile = group_idx - tile_group_start

        @pl.when(group_in_tile == 0)
        def _load_tile_metadata():
            _load_hbm_row(tile_plan_ref, plan_vmem_ref, sched_dma_sem, tile_id)
            req_id = _tile_plan_value_static(plan_vmem_ref[...], 0,
                                             TilePlanField.REQ_ID)
            _load_hbm_row(block_tables_ref, block_table_vmem_ref,
                          sched_dma_sem, req_id)

        plan_values = plan_vmem_ref[...]
        block_table_values = block_table_vmem_ref[...]

        _stage_schedule_step_from_tile_plan(
            plan_values,
            block_table_values,
            sched_vmem_ref,
            group_in_tile,
            jnp.array(0, dtype=jnp.int32),
            state_mode=state_mode,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            lane=lane,
        )
        q_hbm_offset = sched_vmem_ref[my_id, lane,
                                      RuntimeScheduleField.Q_HBM_OFFSET]
        group_o_hbm_offset = q_hbm_offset
        group_q_tile_size = sched_vmem_ref[my_id, lane,
                                           RuntimeScheduleField.Q_TILE_SIZE]
        group_load_q = group_in_tile == 0
        q_tile_size = jnp.where(group_load_q, group_q_tile_size, 0)

        if state_mode == "current":

            @pl.when(group_load_q)
            def _reset_current_state_tile():
                _reset_multi_head_online_state(m_scratch_ref, l_scratch_ref,
                                               acc_scratch_ref)

        else:

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

        @pl.when(group_load_q)
        def _load_q_tile():
            _load_compact_q_tile_multi_head(
                q_ref,
                q_vmem_ref,
                local_dma_sem,
                q_hbm_offset,
                q_tile_size,
            )

        _stage_schedule_step_from_tile_plan(
            plan_values,
            block_table_values,
            sched_vmem_ref,
            group_in_tile,
            my_id,
            state_mode=state_mode,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
            lane=lane,
        )
        if state_mode == "current":
            _load_local_current_or_boundary_kv_all_heads_packed(
                current_kv_ref,
                kv_cache_ref,
                kv_group_ref,
                local_dma_sem,
                sched_vmem_ref,
                lane,
                packed_kv_groups=packed_kv_groups,
                kv_layout=kv_layout,
            )
        else:
            _load_local_kv_page_all_heads_packed(
                kv_cache_ref,
                kv_group_ref,
                local_dma_sem,
                sched_vmem_ref,
                lane,
                packed_kv_groups=packed_kv_groups,
                kv_layout=kv_layout,
            )

        @pl.when(group_idx > 0)
        def _wait_previous_group_slot_free():
            pl.semaphore_wait(group_slot_free_sems.at[lane], 1)

        def _round_loop(round_idx, carry):
            del carry
            curr_slot = lax.rem(round_idx, 2)
            src_rank = lax.rem(my_id + pcp_size - round_idx, pcp_size)

            @pl.when(round_idx > 0)
            def _load_round_schedule():
                _stage_schedule_step_from_tile_plan(
                    plan_values,
                    block_table_values,
                    sched_vmem_ref,
                    group_in_tile,
                    src_rank,
                    state_mode=state_mode,
                    pcp_size=pcp_size,
                    page_size=kv_group_ref.shape[2],
                    interleave_size=interleave_size,
                    lane=lane,
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

            if state_mode == "current":
                capture_src_offset = sched_vmem_ref[
                    my_id, lane, RuntimeScheduleField.CAPTURE_SRC_OFFSET]
                capture_dst_offset = sched_vmem_ref[
                    my_id, lane, RuntimeScheduleField.CAPTURE_DST_OFFSET]
                capture_len = sched_vmem_ref[my_id, lane,
                                             RuntimeScheduleField.CAPTURE_LEN]
                capture_active = capture_len > 0
                safe_capture_src_offset = jnp.maximum(capture_src_offset, 0)
                safe_capture_dst_offset = jnp.maximum(capture_dst_offset, 0)
                if kv_layout == KVLayout.SEQ_ALONG_LANE:
                    seq_capture_src_offset = pl.multiple_of(
                        safe_capture_src_offset, TPU_KV_DMA_ALIGNMENT)
                    seq_capture_dst_offset = pl.multiple_of(
                        safe_capture_dst_offset, TPU_KV_DMA_ALIGNMENT)
                    seq_capture_len = pl.multiple_of(capture_len,
                                                     TPU_KV_DMA_ALIGNMENT)
                    capture_op = pltpu.make_async_copy(
                        src_ref=kv_group_ref.at[
                            curr_slot,
                            0,
                            :,
                            :,
                            :,
                            pl.ds(seq_capture_src_offset, seq_capture_len),
                        ],
                        dst_ref=captured_kv_ref.at[
                            :,
                            :,
                            :,
                            pl.ds(seq_capture_dst_offset, seq_capture_len),
                        ],
                        sem=capture_dma_sem,
                    )
                else:
                    capture_op = pltpu.make_async_copy(
                        src_ref=kv_group_ref.at[
                            curr_slot,
                            0,
                            pl.ds(safe_capture_src_offset, capture_len),
                            :,
                            :,
                            :,
                        ],
                        dst_ref=captured_kv_ref.at[
                            pl.ds(safe_capture_dst_offset, capture_len),
                            :,
                            :,
                            :,
                        ],
                        sem=capture_dma_sem,
                    )

                @pl.when(capture_active)
                def _start_capture():
                    capture_op.start()

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
                interleave_size=interleave_size,
                q_compute_size=q_compute_size,
                k_scale=k_scale,
                v_scale=v_scale,
                causal=causal,
                kv_layout=kv_layout,
            )

            if state_mode == "current":

                @pl.when(capture_active)
                def _finish_capture():
                    capture_op.wait()

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

            return jnp.array(0, dtype=jnp.int32)

        lax.fori_loop(
            0,
            pcp_size,
            _round_loop,
            jnp.array(0, dtype=jnp.int32),
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

        if state_mode == "current":
            num_groups_field = TilePlanField.CURRENT_NUM_GROUPS
        else:
            num_groups_field = TilePlanField.HISTORY_NUM_GROUPS
        tile_num_groups = _tile_plan_value_static(plan_values, 0,
                                                  num_groups_field)
        store_q_tile_size = jnp.where(
            group_in_tile == tile_num_groups - 1,
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
        else:

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
    pass_meta_ref,
    q_ref,
    current_kv_ref,
    kv_cache_ref,
    tile_plan_ref,
    block_tables_ref,
    tile_ids_ref,
    group_starts_ref,
    m_out_ref,
    l_out_ref,
    acc_out_ref,
    captured_kv_out_ref,
    sched_dma_sem,
    local_dma_sem,
    capture_dma_sem,
    remote_send_sems,
    remote_recv_sems,
    remote_sync_sems,
    group_slot_free_sems,
    plan_vmem_ref,
    block_table_vmem_ref,
    tile_ids_vmem_ref,
    group_starts_vmem_ref,
    sched_vmem_ref,
    q_vmem_ref,
    kv_vmem_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    pcp_size,
    interleave_size,
    num_lanes,
    q_compute_size,
    num_tiles,
    max_page_groups,
    sm_scale,
    packed_kv_groups,
    page_size,
    kv_layout,
    k_scale,
    v_scale,
    mesh_axis_names,
    pcp_axis_name,
):
    """Current-chunk causal PCP body that only materializes online state."""
    for lane in range(num_lanes):
        active_tiles = jnp.minimum(pass_meta_ref[0], num_tiles)
        active_page_groups = jnp.minimum(pass_meta_ref[1], max_page_groups)
        _run_pcp_page_groups_multi_head(
            active_tiles,
            active_page_groups,
            q_ref,
            current_kv_ref,
            kv_cache_ref,
            captured_kv_out_ref,
            tile_plan_ref,
            block_tables_ref,
            tile_ids_ref,
            group_starts_ref,
            sched_dma_sem,
            local_dma_sem,
            capture_dma_sem,
            remote_send_sems,
            remote_recv_sems,
            remote_sync_sems,
            group_slot_free_sems,
            plan_vmem_ref,
            block_table_vmem_ref,
            tile_ids_vmem_ref,
            group_starts_vmem_ref,
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
            interleave_size=interleave_size,
            lane=lane,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            packed_kv_groups=packed_kv_groups,
            page_size=page_size,
            kv_layout=kv_layout,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
            state_mode="current",
        )


def _pcp_streaming_attention_history_output_page_groups_multi_head_kernel(
    pass_meta_ref,
    q_ref,
    kv_cache_ref,
    tile_plan_ref,
    block_tables_ref,
    tile_ids_ref,
    group_starts_ref,
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
    plan_vmem_ref,
    block_table_vmem_ref,
    tile_ids_vmem_ref,
    group_starts_vmem_ref,
    sched_vmem_ref,
    q_vmem_ref,
    kv_vmem_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    pcp_size,
    interleave_size,
    num_lanes,
    q_compute_size,
    num_tiles,
    max_page_groups,
    sm_scale,
    packed_kv_groups,
    page_size,
    kv_layout,
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
        active_tiles = jnp.minimum(pass_meta_ref[0], num_tiles)
        active_page_groups = jnp.minimum(pass_meta_ref[1], max_page_groups)
        _run_pcp_page_groups_multi_head(
            active_tiles,
            active_page_groups,
            q_ref,
            None,
            kv_cache_ref,
            None,
            tile_plan_ref,
            block_tables_ref,
            tile_ids_ref,
            group_starts_ref,
            sched_dma_sem,
            local_dma_sem,
            None,
            remote_send_sems,
            remote_recv_sems,
            remote_sync_sems,
            group_slot_free_sems,
            plan_vmem_ref,
            block_table_vmem_ref,
            tile_ids_vmem_ref,
            group_starts_vmem_ref,
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
            interleave_size=interleave_size,
            lane=lane,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            packed_kv_groups=packed_kv_groups,
            page_size=page_size,
            kv_layout=kv_layout,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
            state_mode="history",
        )


def _pcp_streaming_attention_zero_history_output_multi_head_kernel(
    pass_meta_ref,
    o_in_ref,
    tile_plan_ref,
    tile_ids_ref,
    group_starts_ref,
    m_in_ref,
    l_in_ref,
    acc_in_ref,
    o_ref,
    sched_dma_sem,
    local_dma_sem,
    plan_vmem_ref,
    tile_ids_vmem_ref,
    group_starts_vmem_ref,
    o_vmem_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    num_tiles,
    pcp_axis_name,
):
    """Write outputs for tiles whose current pass already covered all KV."""
    del o_in_ref
    my_id = lax.axis_index(pcp_axis_name)
    active_tiles = jnp.minimum(pass_meta_ref[0], num_tiles)
    active_page_groups = pass_meta_ref[1]
    ids_load = pltpu.make_async_copy(
        src_ref=tile_ids_ref.at[:, :],
        dst_ref=tile_ids_vmem_ref.at[:, :],
        sem=sched_dma_sem,
    )
    # TODO: Overlap these independent metadata DMAs by starting both copies
    # before waiting when schedule staging becomes a material kernel cost.
    ids_load.start()
    ids_load.wait()
    starts_load = pltpu.make_async_copy(
        src_ref=group_starts_ref.at[:, :],
        dst_ref=group_starts_vmem_ref.at[:, :],
        sem=sched_dma_sem,
    )
    starts_load.start()
    starts_load.wait()
    tile_ids = tile_ids_vmem_ref[...]
    group_starts = group_starts_vmem_ref[...]

    def _page_group_loop(group_idx, carry):
        del carry
        tile_pos = _group_tile_position(group_starts, active_tiles, group_idx)
        tile_id = _vector_lookup(tile_ids, tile_pos)
        tile_group_start = _vector_lookup(group_starts, tile_pos)
        group_in_tile = group_idx - tile_group_start

        @pl.when(group_in_tile == 0)
        def _load_tile_metadata():
            _load_hbm_row(tile_plan_ref, plan_vmem_ref, sched_dma_sem, tile_id)

        plan_values = plan_vmem_ref[...]
        history_num_groups = _tile_plan_value(plan_values, my_id,
                                              TilePlanField.HISTORY_NUM_GROUPS)
        current_num_groups = _tile_plan_value(plan_values, my_id,
                                              TilePlanField.CURRENT_NUM_GROUPS)
        group_q_hbm_offset = _tile_plan_value(plan_values, my_id,
                                              TilePlanField.Q_HBM_OFFSET)
        group_o_hbm_offset = group_q_hbm_offset
        group_q_tile_size = _tile_plan_value(plan_values, my_id,
                                             TilePlanField.Q_TILE_SIZE)
        write_zero_history = jnp.logical_and(
            group_in_tile == 0,
            jnp.logical_and(current_num_groups > 0, history_num_groups == 0),
        )
        store_q_tile_size = jnp.where(write_zero_history, group_q_tile_size, 0)

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
        current_kv_local,
        kv_cache_local,
        tile_plan,
        block_tables,
        tile_ids,
        group_starts,
        pass_meta,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        interleave_size: int,
        kv_layout: KVLayout,
        q_compute_size: int | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    local_tokens = q_multi_head.shape[0]
    kv_heads = q_multi_head.shape[1]
    q_per_kv = q_multi_head.shape[2]
    head_dim = q_multi_head.shape[-1]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        packed_kv_groups = current_kv_local.shape[0]
        page_size = kv_cache_local.shape[-1]
        kv_vmem_shape = (2, 1, *kv_cache_local.shape[2:])
        captured_kv_hbm_shape = (
            *current_kv_local.shape[:-1],
            local_tokens * pcp_size,
        )
    else:
        packed_kv_groups = current_kv_local.shape[1]
        kv_packing = current_kv_local.shape[2]
        page_size = kv_cache_local.shape[2]
        kv_vmem_shape = (2, 1, page_size, packed_kv_groups, kv_packing,
                         head_dim)
        captured_kv_hbm_shape = (
            local_tokens * pcp_size,
            *current_kv_local.shape[1:],
        )
    num_tiles = tile_plan.shape[0]
    # One tile can need an owner-phase partial fresh group plus one old-cache
    # history-boundary group in addition to its full fresh groups.
    max_page_groups = num_tiles * (math.ceil(local_tokens / interleave_size) +
                                   2)
    num_lanes = 1
    q_compute_size = _normalize_q_compute_size(q_block_size, q_compute_size)
    pass_meta = jnp.asarray(pass_meta, dtype=jnp.int32)

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
            interleave_size=interleave_size,
            num_lanes=num_lanes,
            q_compute_size=q_compute_size,
            num_tiles=num_tiles,
            max_page_groups=max_page_groups,
            sm_scale=sm_scale,
            packed_kv_groups=packed_kv_groups,
            page_size=page_size,
            kv_layout=kv_layout,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(state_lm_hbm_shape, jnp.float32),
            jax.ShapeDtypeStruct(state_lm_hbm_shape, jnp.float32),
            jax.ShapeDtypeStruct(state_acc_hbm_shape, jnp.float32),
            jax.ShapeDtypeStruct(captured_kv_hbm_shape,
                                 current_kv_local.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.DMA((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.REGULAR((num_lanes, pcp_size - 1)),
                pltpu.SemaphoreType.REGULAR((num_lanes, )),
                pltpu.VMEM(tile_plan.shape[1:], tile_plan.dtype),
                pltpu.VMEM(block_tables.shape[1:], block_tables.dtype),
                pltpu.VMEM(tile_ids.shape, tile_ids.dtype),
                pltpu.VMEM(group_starts.shape, group_starts.dtype),
                pltpu.VMEM((pcp_size, num_lanes,
                            RuntimeScheduleField.PACKED_NUM_FIELDS),
                           tile_plan.dtype),
                pltpu.VMEM((q_block_size, kv_heads, q_per_kv, head_dim),
                           q_multi_head.dtype),
                pltpu.VMEM(kv_vmem_shape, current_kv_local.dtype),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_acc_scratch_shape, jnp.float32),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes),
        name="pcp_streaming_attention_current_state_page_groups_multi_head",
    )(pass_meta, q_multi_head, current_kv_local, kv_cache_local, tile_plan,
      block_tables, tile_ids, group_starts)


def _pcp_streaming_attention_history_output_page_groups_multi_head_pallas_call(
        q_multi_head,
        kv_cache_local,
        tile_plan,
        block_tables,
        tile_ids,
        group_starts,
        m_state,
        l_state,
        acc_state,
        pass_meta,
        *,
        pcp_size: int,
        q_block_size: int,
        sm_scale: float,
        interleave_size: int,
        kv_layout: KVLayout,
        q_compute_size: int | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    kv_heads = q_multi_head.shape[1]
    q_per_kv = q_multi_head.shape[2]
    head_dim = q_multi_head.shape[-1]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache_local.shape[-1]
        packed_kv_groups = kv_cache_local.shape[2]
        kv_vmem_shape = (2, 1, *kv_cache_local.shape[2:])
    else:
        page_size = kv_cache_local.shape[2]
        packed_kv_groups = kv_cache_local.shape[3]
        kv_packing = kv_cache_local.shape[4]
        kv_vmem_shape = (2, 1, page_size, packed_kv_groups, kv_packing,
                         head_dim)
    num_tiles = tile_plan.shape[0]
    max_page_groups = num_tiles * block_tables.shape[1]
    num_lanes = 1
    q_compute_size = _normalize_q_compute_size(q_block_size, q_compute_size)
    pass_meta = jnp.asarray(pass_meta, dtype=jnp.int32)

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
            interleave_size=interleave_size,
            num_lanes=num_lanes,
            q_compute_size=q_compute_size,
            num_tiles=num_tiles,
            max_page_groups=max_page_groups,
            sm_scale=sm_scale,
            packed_kv_groups=packed_kv_groups,
            page_size=page_size,
            kv_layout=kv_layout,
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
                pltpu.VMEM(tile_plan.shape[1:], tile_plan.dtype),
                pltpu.VMEM(block_tables.shape[1:], block_tables.dtype),
                pltpu.VMEM(tile_ids.shape, tile_ids.dtype),
                pltpu.VMEM(group_starts.shape, group_starts.dtype),
                pltpu.VMEM((pcp_size, num_lanes,
                            RuntimeScheduleField.PACKED_NUM_FIELDS),
                           tile_plan.dtype),
                pltpu.VMEM((q_block_size, kv_heads, q_per_kv, head_dim),
                           q_multi_head.dtype),
                pltpu.VMEM(kv_vmem_shape, kv_cache_local.dtype),
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
    )(pass_meta, q_multi_head, kv_cache_local, tile_plan, block_tables,
      tile_ids, group_starts, m_state, l_state, acc_state)


def _pcp_streaming_attention_zero_history_output_multi_head_pallas_call(
    o_multi_head,
    tile_plan,
    tile_ids,
    group_starts,
    m_state,
    l_state,
    acc_state,
    pass_meta,
    *,
    q_block_size: int,
    pcp_axis_name: str = AXIS,
):
    kv_heads = o_multi_head.shape[1]
    q_per_kv = o_multi_head.shape[2]
    head_dim = o_multi_head.shape[-1]
    num_tiles = tile_plan.shape[0]
    pass_meta = jnp.asarray(pass_meta, dtype=jnp.int32)

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
            num_tiles=num_tiles,
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
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(tile_plan.shape[1:], tile_plan.dtype),
                pltpu.VMEM(tile_ids.shape, tile_ids.dtype),
                pltpu.VMEM(group_starts.shape, group_starts.dtype),
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
    )(pass_meta, o_multi_head, tile_plan, tile_ids, group_starts, m_state,
      l_state, acc_state)


def pcp_streaming_attention_page_groups_packed_local_from_metadata(
        q_local,
        current_kv_local,
        kv_cache_local,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        writeback_segment_descriptors,
        writeback_num_segments,
        *,
        pcp_size: int,
        interleave_size: int,
        q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
        sm_scale: float,
        q_compute_size: int | None = None,
        num_lanes: int = 1,
        k_scale: float | None = None,
        v_scale: float | None = None,
        kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
        mesh_axis_names: tuple[str, ...] = (AXIS, ),
        pcp_axis_name: str = AXIS,
):
    """Generate the narrow metadata schedule and run split PCP RPA.

    The current causal pass consumes freshly packed local KV directly and
    captures passing rows on their request-absolute cache owner. The captured
    KV is written to the local cache immediately after the current pass, ending
    the capture buffer lifetime before the history/output pass.
    The history no-causal pass reads only tokens before the current request
    chunk, so it is unaffected by the newly written current rows.

    Args:
        q_local: [local_tokens, kv_heads, q_per_kv, head_dim] compact Q rows
            for the current PCP rank.
        current_kv_local: Freshly computed packed KV rows. Legacy layout is
            [local_tokens, packed_kv_heads_x2, kv_packing, head_dim];
            SEQ_ALONG_LANE is
            [2 * kv_heads, head_dim / kv_packing, kv_packing, local_tokens].
        kv_cache_local: Local packed KV-cache shard. Legacy layout is
            [pages, page_size, packed_kv_heads_x2, kv_packing, head_dim];
            SEQ_ALONG_LANE is
            [pages, 2 * kv_heads, head_dim / kv_packing, kv_packing,
            page_size].
        kv_lens: [max_num_seqs]. Total KV length for each request, including
            the query tokens in the current prefill chunk.
        page_indices: Flattened or 2D vLLM block table. Entries are reduced
            modulo the local KV-cache block count before local page loads.
        cu_q_lens: [max_num_seqs + 1]. Cumulative query-token offsets for the
            current batch.
        distribution: [3] vLLM request distribution metadata. The third entry
            is the number of active requests considered by the schedule.
        writeback_segment_descriptors: [3, max_segments] dense-source to
            flat-cache copy descriptors for this cache rank.
        writeback_num_segments: Number of active writeback descriptors.
        pcp_size: Number of ranks in the PCP ring.
        interleave_size: Number of consecutive global tokens assigned to one
            rank before rotating to the next PCP rank. Must divide the KV page
            size.
        q_block_size: Static local Q tile size used by the Pallas kernel.
        q_compute_size: Static Q rows consumed per multi-head softmax compute
            step. Defaults to `q_block_size`.
        sm_scale: Softmax scale applied to QK scores.
        num_lanes: Number of independent schedule lanes. The metadata path
            currently supports one lane.
        k_scale: Optional dequantization scale for K.
        v_scale: Optional dequantization scale for V.
        mesh_axis_names: Names of the active JAX mesh axes.
        pcp_axis_name: Mesh axis used as the PCP ring axis.

    Returns:
        A pair of local attention output and updated local KV cache.
    """
    _validate_packed_kv_layout(q_local,
                               kv_cache_local,
                               kv_layout=kv_layout,
                               k_scale=k_scale,
                               v_scale=v_scale)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if (current_kv_local.shape[:-1] != kv_cache_local.shape[1:-1]
                or current_kv_local.shape[-1] < q_local.shape[0]):
            raise ValueError(
                "SEQ_ALONG_LANE fresh KV must match the cache head layout and "
                "cover all local Q rows: "
                f"{current_kv_local.shape=} {q_local.shape[0]=} "
                f"cache_head_layout={kv_cache_local.shape[1:-1]}.")
        page_size = kv_cache_local.shape[-1]
    else:
        if current_kv_local.shape != (q_local.shape[0],
                                      *kv_cache_local.shape[2:]):
            raise ValueError(
                "Fresh packed KV layout must match local Q rows and cache "
                f"tail: {current_kv_local.shape=} {q_local.shape[0]=} "
                f"cache_tail={kv_cache_local.shape[2:]}.")
        page_size = kv_cache_local.shape[1]
    (tile_plan, block_tables, current_tile_ids, current_group_starts,
     current_meta, history_tile_ids, history_group_starts,
     history_meta) = (build_pcp_streaming_schedule_inputs_from_metadata_jax(
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
         num_lanes=num_lanes,
     ))

    m_state, l_state, acc_state, captured_kv_local = (
        _pcp_streaming_attention_current_state_page_groups_multi_head_pallas_call(
            q_local,
            current_kv_local,
            kv_cache_local[None, ...],
            tile_plan,
            block_tables,
            current_tile_ids,
            current_group_starts,
            current_meta,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            q_block_size=q_block_size,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            kv_layout=kv_layout,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ))
    kv_cache_local = write_captured_kv_to_local_cache(
        kv_cache_local,
        captured_kv_local,
        writeback_segment_descriptors,
        writeback_num_segments,
        kv_layout=kv_layout,
    )
    history_output = (
        _pcp_streaming_attention_history_output_page_groups_multi_head_pallas_call(
            q_local,
            kv_cache_local[None, ...],
            tile_plan,
            block_tables,
            history_tile_ids,
            history_group_starts,
            m_state,
            l_state,
            acc_state,
            history_meta,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            q_block_size=q_block_size,
            q_compute_size=q_compute_size,
            sm_scale=sm_scale,
            kv_layout=kv_layout,
            k_scale=k_scale,
            v_scale=v_scale,
            mesh_axis_names=mesh_axis_names,
            pcp_axis_name=pcp_axis_name,
        ))
    output = _pcp_streaming_attention_zero_history_output_multi_head_pallas_call(
        history_output,
        tile_plan,
        current_tile_ids,
        current_group_starts,
        m_state,
        l_state,
        acc_state,
        current_meta,
        q_block_size=q_block_size,
        pcp_axis_name=pcp_axis_name,
    )
    return output, kv_cache_local
