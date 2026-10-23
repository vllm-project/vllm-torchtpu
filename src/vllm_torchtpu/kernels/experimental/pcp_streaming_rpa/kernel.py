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
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import (
    broadcast_minor,
    convert_to_target_bitwidth,
    get_dtype_packing,
    strided_load,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_kernel as _splash_kernel,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_mask as _splash_mask,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField,
    TilePlanField,
    query_row_positions,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.stage_plan import (
    PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS,
    PCP_STREAMING_RPA_VPAGE_GROUP_SIZE,
)

AXIS = "pcp"
ONLINE_STATE_INIT_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE = 512
TPU_HBM_ROW_TILE = 16
ONLINE_STATE_PADDED_LANES = 128
TPU_KV_DMA_ALIGNMENT = 128
KVLayout = batched_rpa_configs.KVLayout


def _copy_hnd_token_slice(
    src_ref,
    dst_ref,
    source_start,
    destination_start,
    length,
    sem,
    *,
    source_hbm,
    destination_hbm,
):
    """Copy HND tokens using aligned DMA and bit-preserving VMEM stitching.

    Both refs have [2*heads, packed_dim, packing, tokens] layout. Partial
    destination tiles are read-modify-written sequentially: other segments
    may share a tile, and bytes outside this slice must survive unchanged.
    Only three 128-token scratch tiles are needed, independent of cache size.
    """
    alignment = TPU_KV_DMA_ALIGNMENT
    aligned = (
        (source_start % alignment == 0)
        & (destination_start % alignment == 0)
        & (length % alignment == 0)
    )

    @pl.when(aligned & (length > 0))
    def _aligned_copy():
        src = pl.multiple_of(source_start, alignment)
        dst = pl.multiple_of(destination_start, alignment)
        size = pl.multiple_of(length, alignment)
        op = pltpu.make_async_copy(
            src_ref.at[:, :, :, pl.ds(src, size)],
            dst_ref.at[:, :, :, pl.ds(dst, size)],
            sem,
        )
        op.start()
        op.wait()

    @pl.when(~aligned & (length > 0))
    def _partial_copy():
        shape = (*src_ref.shape[:-1], alignment)

        def _stitch(src0, src1, dest):
            first = (destination_start // alignment) * alignment
            count = pl.cdiv(destination_start % alignment + length, alignment)

            @pl.loop(0, count)
            def _tile(i):
                dst = pl.multiple_of(first + i * alignment, alignment)
                begin = jnp.maximum(destination_start, dst)
                end = jnp.minimum(destination_start + length, dst + alignment)
                source = source_start + begin - destination_start
                src = pl.multiple_of((source // alignment) * alignment, alignment)
                next_src = pl.multiple_of(
                    jnp.minimum(src + alignment, src_ref.shape[-1] - alignment),
                    alignment,
                )

                def _read(ref, offset, scratch, hbm):
                    if hbm:
                        op = pltpu.make_async_copy(
                            ref.at[:, :, :, pl.ds(offset, alignment)], scratch, sem
                        )
                        op.start()
                        op.wait()
                    else:
                        scratch[...] = ref[:, :, :, pl.ds(offset, alignment)]

                _read(src_ref, src, src0, source_hbm)
                _read(src_ref, next_src, src1, source_hbm)
                _read(dst_ref, dst, dest, destination_hbm)
                # Operate on packed bits, without any FP8 conversion or
                # arithmetic on possibly uninitialized neighbouring tokens.
                a = src0.bitcast(jnp.uint32).reshape(-1, alignment)[...]
                b = src1.bitcast(jnp.uint32).reshape(-1, alignment)[...]
                old = dest.bitcast(jnp.uint32).reshape(-1, alignment)
                lane = lax.broadcasted_iota(jnp.int32, a.shape, 1)
                relative = source - src - (begin - dst)
                a = pltpu.roll(a, -relative, axis=1)
                b = pltpu.roll(b, -relative, axis=1)
                incoming = jnp.where(lane + relative < alignment, a, b)
                valid = (lane >= begin - dst) & (lane < end - dst)
                old[...] = jnp.where(valid, incoming, old[...])
                if destination_hbm:
                    op = pltpu.make_async_copy(
                        dest, dst_ref.at[:, :, :, pl.ds(dst, alignment)], sem
                    )
                    op.start()
                    op.wait()
                else:
                    dst_ref[:, :, :, pl.ds(dst, alignment)] = dest[...]

        pl.run_scoped(
            _stitch,
            pltpu.VMEM(shape, src_ref.dtype),
            pltpu.VMEM(shape, src_ref.dtype),
            pltpu.VMEM(shape, src_ref.dtype),
        )


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
    num_segments = jnp.clip(num_segments_ref[0], 0, segment_descriptors_ref.shape[1])

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
    """Scatter HND segments, preserving partial tiles and earlier segments."""
    del kv_cache_in_ref
    num_segments = jnp.clip(num_segments_ref[0], 0, segment_descriptors_ref.shape[1])
    page_size = kv_cache_out_ref.shape[-1]

    @pl.loop(0, num_segments)
    def _write_segment(segment_idx):
        source_start = segment_descriptors_ref[0, segment_idx]
        destination_start = segment_descriptors_ref[1, segment_idx]
        segment_len = segment_descriptors_ref[2, segment_idx]
        page = destination_start // page_size
        offset = destination_start % page_size
        _copy_hnd_token_slice(
            captured_kv_ref,
            kv_cache_out_ref.at[page],
            source_start,
            offset,
            segment_len,
            dma_sem,
            source_hbm=True,
            destination_hbm=True,
        )


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
        raise ValueError(f"KV cache must have at least two dims: {kv_cache.shape}.")
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if captured_kv.shape[:-1] != kv_cache.shape[1:-1]:
            raise ValueError(
                "SEQ_ALONG_LANE captured KV prefix must match the paged cache "
                f"head layout: {captured_kv.shape[:-1]} vs "
                f"{kv_cache.shape[1:-1]}."
            )
    elif captured_kv.shape[1:] != kv_cache.shape[2:]:
        raise ValueError(
            "Captured KV tail must match the paged cache tail: "
            f"{captured_kv.shape[1:]} vs {kv_cache.shape[2:]}."
        )
    if segment_descriptors.ndim != 2 or segment_descriptors.shape[0] != 3:
        raise ValueError(
            "KV writeback segment descriptors must have shape [3, segments], "
            f"got {segment_descriptors.shape}."
        )

    num_segments = jnp.asarray(num_segments, dtype=jnp.int32).reshape((1,))
    segment_descriptors = jnp.asarray(segment_descriptors, dtype=jnp.int32)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        write_kernel = _pcp_write_captured_kv_seq_along_lane_kernel
        writeback_cache = kv_cache
    else:
        write_kernel = _pcp_write_captured_kv_kernel
        writeback_cache = kv_cache.reshape((-1, *kv_cache.shape[2:]))
    updated_cache = pl.pallas_call(
        write_kernel,
        out_shape=jax.ShapeDtypeStruct(writeback_cache.shape, writeback_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(pltpu.SemaphoreType.DMA,),
            grid=(1,),
        ),
        input_output_aliases={3: 0},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
        ),
        name="pcp_write_captured_kv_to_local_cache",
    )(num_segments, segment_descriptors, captured_kv, writeback_cache)
    return updated_cache.reshape(kv_cache.shape)


def _validate_packed_kv_layout(
    q_local,
    kv_cache_local,
    *,
    kv_layout: KVLayout = (KVLayout.HEAD_ALONG_SUBLANE),
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """Validate PCP's physical packed KV layout against logical Q heads."""
    if q_local.ndim != 4:
        raise ValueError(
            "PCP q_local must have shape "
            "[tokens, kv_heads, q_per_kv, head_dim], got "
            f"{q_local.shape}."
        )
    if kv_cache_local.ndim != 5:
        raise ValueError(f"PCP kv_cache_local must be 5D, got {kv_cache_local.shape}.")

    kv_heads = q_local.shape[1]
    q_per_kv = q_local.shape[2]
    head_dim = q_local.shape[-1]
    kv_packing = kv_cache_local.shape[3]
    dtype_packing = get_dtype_packing(kv_cache_local.dtype)
    if kv_heads <= 0 or q_per_kv <= 0:
        raise ValueError(
            f"PCP requires positive kv_heads and q_per_kv, got {kv_heads=} {q_per_kv=}."
        )
    if kv_packing not in (2, 4):
        raise NotImplementedError(
            "PCP streaming RPA supports only BF16/FP8 packed KV cache "
            f"layouts with packing 2 or 4, got {kv_packing}."
        )
    if kv_packing != dtype_packing:
        raise ValueError(
            "PCP KV cache packing does not match dtype: "
            f"shape_packing={kv_packing}, "
            f"dtype_packing={dtype_packing}, "
            f"dtype={kv_cache_local.dtype}."
        )
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
                f"{kv_cache_local.shape[2]} * {kv_packing} vs {head_dim}."
            )
        if kv_packing != 4:
            raise NotImplementedError(
                "PCP SEQ_ALONG_LANE currently supports only FP8 KV cache, "
                f"got packing={kv_packing}."
            )
    else:
        kv_layout_groups = kv_cache_local.shape[2]
        expected_packed_kv_groups = math.ceil((2 * kv_heads) / kv_packing)
        if kv_layout_groups != expected_packed_kv_groups:
            raise ValueError(
                "PCP KV cache packed layout does not match logical KV heads: "
                f"packed_kv_groups={kv_layout_groups}, "
                f"expected={expected_packed_kv_groups}, {kv_heads=}, "
                f"{kv_packing=}."
            )
        if kv_cache_local.shape[-1] != head_dim:
            raise ValueError(
                "PCP KV cache head_dim must match Q head_dim: "
                f"{kv_cache_local.shape[-1]} vs {head_dim}."
            )
    if (k_scale is None) != (v_scale is None):
        raise ValueError(
            "PCP FP8 KV cache requires k_scale and v_scale to "
            "be both None or both non-None."
        )
    if kv_packing == 4:
        if k_scale is None or v_scale is None:
            raise ValueError(
                "PCP FP8 KV cache requires positive finite k_scale and v_scale."
            )
        if (
            not math.isfinite(float(k_scale))
            or float(k_scale) <= 0
            or not math.isfinite(float(v_scale))
            or float(v_scale) <= 0
        ):
            raise ValueError(
                "PCP FP8 KV cache requires positive finite k_scale and v_scale."
            )
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
    return [
        (k.reshape(kv_block_tokens, head_dim), v.reshape(kv_block_tokens, head_dim))
        for k, v in pairs
    ]


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
        q_global_start = sched_vmem_ref[
            consumer_rank, lane, RuntimeScheduleField.Q_GLOBAL_START
        ]
        q_hbm_offset = sched_vmem_ref[
            consumer_rank, lane, RuntimeScheduleField.Q_HBM_OFFSET
        ]
        kv_global_start = sched_vmem_ref[
            consumer_rank, lane, RuntimeScheduleField.KV_GLOBAL_START
        ]
        q_tile_size = sched_vmem_ref[
            consumer_rank, lane, RuntimeScheduleField.Q_TILE_SIZE
        ]
        req_id = sched_vmem_ref[consumer_rank, lane, RuntimeScheduleField.REQ_ID]
        kv_valid_len = sched_vmem_ref[
            consumer_rank, lane, RuntimeScheduleField.KV_VALID_LEN
        ]
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
            q_start = pl.multiple_of(q_compute_idx * q_compute_size, q_compute_size)
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
                q_row = q_start + lax.broadcasted_iota(
                    jnp.int32, (q_compute_size, q_per_kv, 1), 0
                )
                kv_local_pos = lax.broadcasted_iota(
                    jnp.int32, (1, 1, kv_block_tokens), 2
                )
                q_valid = q_row < q_tile_size
                kv_valid = kv_local_pos < kv_valid_len
                q_slice = jnp.where(q_valid, q_slice, 0.0)
                kv_sequence_axis = 1 if kv_layout == KVLayout.SEQ_ALONG_LANE else 0
                kv_input_pos = lax.broadcasted_iota(
                    jnp.int32, k.shape, kv_sequence_axis
                )
                k_compute = jnp.where(kv_input_pos < kv_valid_len, k, 0.0)
                v_compute = jnp.where(kv_input_pos < kv_valid_len, v, 0.0)

            if kv_layout == KVLayout.SEQ_ALONG_LANE:
                qk_dimensions = (((2,), (0,)), ((), ()))
            else:
                qk_dimensions = (((2,), (1,)), ((), ()))
            scores = lax.dot_general(
                q_slice,
                k_compute,
                qk_dimensions,
                preferred_element_type=jnp.float32,
            )

            if causal:
                scores = scores * sm_scale
                q_interleave = interleave_size
                q_pos = query_row_positions(
                    q_global_start,
                    q_hbm_offset,
                    q_row,
                    pcp_size=pcp_size,
                    interleave_size=q_interleave,
                )
                kv_chunk_idx = lax.div(kv_local_pos, q_interleave)
                kv_chunk_offset = lax.rem(kv_local_pos, q_interleave)
                kv_pos = (
                    kv_global_start
                    + kv_chunk_idx * pcp_size * q_interleave
                    + kv_chunk_offset
                )
                entry_valid = req_id != -1
                row_active = jnp.logical_and(entry_valid, q_valid)
                mask = jnp.logical_and(
                    jnp.logical_and(q_pos >= kv_pos, kv_valid), q_valid
                )
                scores = jnp.where(mask, scores, -jnp.inf)
                scores = jnp.where(row_active, scores, 0.0)

            m_curr = jnp.max(scores, axis=2, keepdims=True)
            if causal:
                m_next = jnp.where(row_active, jnp.maximum(m_head, m_curr), m_head)
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
                pv_dimensions = (((2,), (1,)), ((), ()))
            else:
                pv_dimensions = (((2,), (0,)), ((), ()))
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
                head_dim, kv_block_tokens
            )
            v = kv_vmem_ref[slot, 0, 2 * kv_head_idx + 1, :, :, :].reshape(
                head_dim, kv_block_tokens
            )
            _consume_kv_head(k, v, kv_head_idx)
    else:
        heads_per_group = kv_vmem_ref.shape[4] // 2
        for packed_group_idx in range(kv_vmem_ref.shape[3]):
            kv_pairs = _strided_load_packed_kv_group(
                kv_vmem_ref, slot, packed_group_idx
            )
            for pair_idx, (k, v) in enumerate(kv_pairs):
                kv_head_idx = packed_group_idx * heads_per_group + pair_idx
                if kv_head_idx >= q_vmem_ref.shape[1]:
                    continue
                _consume_kv_head(k, v, kv_head_idx)


def _reset_multi_head_online_state(m_scratch_ref, l_scratch_ref, acc_scratch_ref):
    # Keep the hot online-softmax loop identical to origin/main while making
    # an all-masked first ring page well-defined: max(finite_min, -inf) stays
    # finite, so both the probability update and the previous-state scale are
    # exact (0 and 1 respectively) without an extra per-page validity mask.
    m_scratch_ref[...] = jnp.full(
        m_scratch_ref.shape, ONLINE_STATE_INIT_VALUE, dtype=jnp.float32
    )
    l_scratch_ref[...] = jnp.zeros(l_scratch_ref.shape, dtype=jnp.float32)
    acc_scratch_ref[...] = jnp.zeros(acc_scratch_ref.shape, dtype=jnp.float32)


def _write_multi_head_output_from_state(o_vmem_ref, l_scratch_ref, acc_scratch_ref):
    l_state = l_scratch_ref.at[:, :, :, 0][...]
    l_values = jnp.transpose(l_state, (1, 0, 2))[..., None]
    acc = jnp.transpose(acc_scratch_ref[...], (1, 0, 2, 3))
    l_broadcast = jnp.broadcast_to(l_values, acc.shape)
    o_vmem_ref[...] = jnp.where(l_broadcast > 0, acc / l_broadcast, 0.0).astype(
        o_vmem_ref.dtype
    )


def _store_multi_head_online_state_tile(
    m_ref,
    l_ref,
    acc_ref,
    local_dma_sem,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    q_hbm_offset,
    q_tile_size,
):
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


def _load_multi_head_online_state_tile(
    m_ref,
    l_ref,
    acc_ref,
    local_dma_sem,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    q_hbm_offset,
    q_tile_size,
):
    _reset_multi_head_online_state(m_scratch_ref, l_scratch_ref, acc_scratch_ref)

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
    values = jnp.reshape(values, (-1,))
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
    starts = jnp.reshape(group_starts, (-1,))
    positions = jnp.arange(starts.shape[0], dtype=jnp.int32)
    before_or_at_group = jnp.logical_and(positions < active_tiles, starts <= group_idx)
    return jnp.maximum(
        jnp.sum(before_or_at_group.astype(jnp.int32), dtype=jnp.int32) - 1,
        0,
    )


def _local_page_valid_len_from_tile_plan(
    kv_len, local_page_idx, src_rank, *, page_size, pcp_size, interleave_size
):
    cycle = pcp_size * interleave_size
    base = local_page_idx * page_size * pcp_size + src_rank * interleave_size
    delta = kv_len - base
    chunks_per_page = page_size // interleave_size
    full_chunks = jnp.minimum(chunks_per_page, jnp.maximum(delta, 0) // cycle)
    partial = jnp.minimum(
        interleave_size,
        jnp.maximum(delta - full_chunks * cycle, 0),
    )
    valid_len = full_chunks * interleave_size + jnp.where(
        full_chunks < chunks_per_page, partial, 0
    )
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
        plan_values, 0, TilePlanField.TOKEN_OWNER_START
    )
    request_absolute_start = _tile_plan_value_static(
        plan_values, 0, TilePlanField.REQUEST_ABSOLUTE_QUERY_START
    )
    current_effective_len = _tile_plan_value_static(
        plan_values, 0, TilePlanField.CURRENT_EFFECTIVE_LEN
    )
    capture_current_kv = _tile_plan_value_static(
        plan_values, 0, TilePlanField.CAPTURE_CURRENT_KV
    )

    cycle = pcp_size * interleave_size
    virtual_page_size = page_size * pcp_size
    history_boundary_present = request_absolute_start % virtual_page_size != 0
    is_history_boundary = jnp.logical_and(history_boundary_present, group_in_tile == 0)
    fresh_group_in_tile = jnp.maximum(
        group_in_tile - history_boundary_present.astype(jnp.int32), 0
    )

    cycle_start = token_owner_start // cycle * cycle + fresh_group_in_tile * cycle
    source_chunk_start = cycle_start + source_rank * interleave_size
    token_owner_end = token_owner_start + current_effective_len
    overlap_start = jnp.maximum(source_chunk_start, token_owner_start)
    overlap_end = jnp.minimum(source_chunk_start + interleave_size, token_owner_end)
    fresh_kv_valid_len = jnp.maximum(overlap_end - overlap_start, 0)
    valid_fresh_chunk = fresh_kv_valid_len > 0

    request_local_hbm_start = _tile_plan_value(
        plan_values, source_rank, TilePlanField.CURRENT_KV_HBM_START
    )
    source_rows_before_overlap = _rank_token_count_before(
        overlap_start,
        source_rank,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    ) - _rank_token_count_before(
        token_owner_start,
        source_rank,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    fresh_kv_hbm_offset = request_local_hbm_start + source_rows_before_overlap
    fresh_kv_global_start = request_absolute_start + overlap_start - token_owner_start

    boundary_local_page_idx = request_absolute_start // virtual_page_size
    safe_boundary_local_page_idx = jnp.clip(
        boundary_local_page_idx, 0, block_table_values.size - 1
    )
    boundary_page_idx = _vector_lookup(block_table_values, safe_boundary_local_page_idx)
    boundary_kv_global_start = (
        boundary_local_page_idx * virtual_page_size + source_rank * interleave_size
    )
    boundary_kv_valid_len = _local_page_valid_len_from_tile_plan(
        request_absolute_start,
        boundary_local_page_idx,
        source_rank,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
    )
    valid_boundary_page = boundary_kv_valid_len > 0

    valid_scheduled_kv = jnp.where(
        is_history_boundary, valid_boundary_page, valid_fresh_chunk
    )
    kv_page_idx = jnp.where(is_history_boundary, boundary_page_idx, 0)
    kv_global_start = jnp.where(
        is_history_boundary, boundary_kv_global_start, fresh_kv_global_start
    )
    kv_valid_len = jnp.where(
        is_history_boundary, boundary_kv_valid_len, fresh_kv_valid_len
    )
    # Negative KV_HBM_OFFSET is the single internal source tag: the current
    # causal pass loads this group from the old paged cache. Fresh groups use
    # their compact current-KV HBM row offset.
    kv_hbm_offset = jnp.where(is_history_boundary, -1, fresh_kv_hbm_offset)

    safe_global_start = jnp.where(valid_fresh_chunk, fresh_kv_global_start, 0)
    capture_owner_0 = (safe_global_start // interleave_size) % pcp_size
    next_owner_boundary = (safe_global_start // interleave_size + 1) * interleave_size
    capture_len_0 = jnp.minimum(
        fresh_kv_valid_len,
        jnp.maximum(next_owner_boundary - safe_global_start, 0),
    )
    capture_len_0 = jnp.where(capture_current_kv != 0, capture_len_0, 0)
    capture_len_1 = jnp.where(
        capture_current_kv != 0, fresh_kv_valid_len - capture_len_0, 0
    )
    capture_len_0 = jnp.where(is_history_boundary, 0, capture_len_0)
    capture_len_1 = jnp.where(is_history_boundary, 0, capture_len_1)
    capture_global_start_1 = safe_global_start + capture_len_0
    capture_owner_1 = (capture_global_start_1 // interleave_size) % pcp_size

    capture_prefix_0 = _tile_plan_value(
        plan_values, capture_owner_0, TilePlanField.WRITEBACK_HBM_PREFIX
    )
    capture_prefix_1 = _tile_plan_value(
        plan_values, capture_owner_1, TilePlanField.WRITEBACK_HBM_PREFIX
    )
    capture_offset_0 = capture_prefix_0 + (
        _rank_token_count_before(
            safe_global_start,
            capture_owner_0,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
        - _rank_token_count_before(
            request_absolute_start,
            capture_owner_0,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
    )
    capture_offset_1 = capture_prefix_1 + (
        _rank_token_count_before(
            capture_global_start_1,
            capture_owner_1,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
        - _rank_token_count_before(
            request_absolute_start,
            capture_owner_1,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
        )
    )

    for consumer_rank in range(pcp_size):
        q_global_start = _tile_plan_value_static(
            plan_values, consumer_rank, TilePlanField.Q_GLOBAL_START
        )
        q_hbm_offset = _tile_plan_value_static(
            plan_values, consumer_rank, TilePlanField.Q_HBM_OFFSET
        )
        q_tile_size = _tile_plan_value_static(
            plan_values, consumer_rank, TilePlanField.Q_TILE_SIZE
        )
        owns_capture_0 = jnp.logical_and(
            capture_len_0 > 0, capture_owner_0 == consumer_rank
        )
        owns_capture_1 = jnp.logical_and(
            capture_len_1 > 0, capture_owner_1 == consumer_rank
        )
        capture_src_offset = jnp.where(
            owns_capture_0, 0, jnp.where(owns_capture_1, capture_len_0, 0)
        )
        capture_dst_offset = jnp.where(
            owns_capture_0,
            capture_offset_0,
            jnp.where(owns_capture_1, capture_offset_1, 0),
        )
        capture_len = jnp.where(
            owns_capture_0,
            capture_len_0 + jnp.where(owns_capture_1, capture_len_1, 0),
            jnp.where(owns_capture_1, capture_len_1, 0),
        )
        logical_fields = jnp.stack(
            (
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
            )
        )
        packed_row = jnp.concatenate(
            (
                logical_fields,
                jnp.zeros(
                    (
                        RuntimeScheduleField.PACKED_NUM_FIELDS
                        - RuntimeScheduleField.NUM_FIELDS,
                    ),
                    dtype=jnp.int32,
                ),
            )
        )
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
        plan_values, 0, TilePlanField.REQUEST_ABSOLUTE_QUERY_START
    )
    local_page_idx = group_in_tile
    safe_local_page_idx = jnp.clip(local_page_idx, 0, block_table_values.size - 1)
    page_idx = _vector_lookup(block_table_values, safe_local_page_idx)
    kv_global_start = (
        local_page_idx * page_size * pcp_size + source_rank * interleave_size
    )
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
        q_global_start = _tile_plan_value_static(
            plan_values, consumer_rank, TilePlanField.Q_GLOBAL_START
        )
        q_hbm_offset = _tile_plan_value_static(
            plan_values, consumer_rank, TilePlanField.Q_HBM_OFFSET
        )
        q_tile_size = _tile_plan_value_static(
            plan_values, consumer_rank, TilePlanField.Q_TILE_SIZE
        )
        logical_fields = jnp.stack(
            (
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
            )
        )
        packed_row = jnp.concatenate(
            (
                logical_fields,
                jnp.zeros(
                    (
                        RuntimeScheduleField.PACKED_NUM_FIELDS
                        - RuntimeScheduleField.NUM_FIELDS,
                    ),
                    dtype=jnp.int32,
                ),
            )
        )
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


def _normalize_q_compute_size(q_block_size: int, q_compute_size: int | None) -> int:
    if q_compute_size is None:
        q_compute_size = q_block_size
    q_compute_size = int(q_compute_size)
    if q_compute_size <= 0:
        raise ValueError("q_compute_size must be positive.")
    if q_compute_size > q_block_size:
        raise ValueError("q_compute_size must be <= q_block_size.")
    if q_block_size % q_compute_size != 0:
        raise NotImplementedError(
            "PCP streaming RPA requires q_block_size to be divisible by q_compute_size."
        )
    if q_compute_size % TPU_HBM_ROW_TILE != 0:
        raise NotImplementedError(
            "PCP streaming RPA requires q_compute_size to be a multiple of "
            f"{TPU_HBM_ROW_TILE}."
        )
    return q_compute_size


def _load_compact_q_tile_multi_head(
    q_ref, q_vmem_ref, local_dma_sem, q_hbm_offset, q_tile_size
):
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


def _store_compact_output_tile_multi_head(
    o_ref, o_vmem_ref, local_dma_sem, o_hbm_offset, q_tile_size
):
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


def _load_local_kv_page_all_heads_packed(
    kv_cache_ref, kv_vmem_ref, sem, sched_vmem_ref, lane, *, packed_kv_groups, kv_layout
):
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
            _copy_hnd_token_slice(
                current_kv_ref,
                kv_vmem_ref.at[0, 0],
                kv_hbm_offset,
                0,
                kv_valid_len,
                sem,
                source_hbm=True,
                destination_hbm=False,
            )
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
        for axis_name in mesh_axis_names
    )


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
            req_id = _tile_plan_value_static(
                plan_vmem_ref[...], 0, TilePlanField.REQ_ID
            )
            _load_hbm_row(block_tables_ref, block_table_vmem_ref, sched_dma_sem, req_id)

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
        q_hbm_offset = sched_vmem_ref[my_id, lane, RuntimeScheduleField.Q_HBM_OFFSET]
        group_o_hbm_offset = q_hbm_offset
        group_q_tile_size = sched_vmem_ref[
            my_id, lane, RuntimeScheduleField.Q_TILE_SIZE
        ]
        group_load_q = group_in_tile == 0
        q_tile_size = jnp.where(group_load_q, group_q_tile_size, 0)

        if state_mode == "current":

            @pl.when(group_load_q)
            def _reset_current_state_tile():
                _reset_multi_head_online_state(
                    m_scratch_ref, l_scratch_ref, acc_scratch_ref
                )

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
            curr_slot = lax.rem(round_idx, 3)
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
                    page_size=page_size,
                    interleave_size=interleave_size,
                    lane=lane,
                )

            next_slot = lax.rem(round_idx + 1, 3)
            safe_round_idx = jnp.minimum(round_idx, pcp_size - 2)
            safe_slot_free_round_idx = jnp.clip(round_idx - 2, 0, pcp_size - 2)

            @pl.when(jnp.logical_and(round_idx >= 2, round_idx < pcp_size - 1))
            def _wait_remote_sync():
                pl.semaphore_wait(
                    remote_sync_sems.at[lane, safe_slot_free_round_idx], 1
                )

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
                    my_id, lane, RuntimeScheduleField.CAPTURE_SRC_OFFSET
                ]
                capture_dst_offset = sched_vmem_ref[
                    my_id, lane, RuntimeScheduleField.CAPTURE_DST_OFFSET
                ]
                capture_len = sched_vmem_ref[
                    my_id, lane, RuntimeScheduleField.CAPTURE_LEN
                ]
                capture_active = capture_len > 0
                safe_capture_src_offset = jnp.maximum(capture_src_offset, 0)
                safe_capture_dst_offset = jnp.maximum(capture_dst_offset, 0)
                if kv_layout == KVLayout.SEQ_ALONG_LANE:

                    @pl.when(capture_active)
                    def _capture_hnd():
                        _copy_hnd_token_slice(
                            kv_group_ref.at[curr_slot, 0],
                            captured_kv_ref,
                            safe_capture_src_offset,
                            safe_capture_dst_offset,
                            capture_len,
                            capture_dma_sem,
                            source_hbm=False,
                            destination_hbm=True,
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

            if state_mode == "current" and kv_layout != KVLayout.SEQ_ALONG_LANE:

                @pl.when(capture_active)
                def _finish_capture():
                    capture_op.wait()

            @pl.when(round_idx < pcp_size - 1)
            def _finish_remote_copy():
                remote_op.wait_send()

                @pl.when(round_idx < pcp_size - 3)
                def _signal_prev_device():
                    pl.semaphore_signal(
                        remote_sync_sems.at[lane, safe_round_idx],
                        1,
                        device_id=prev_device_id,
                        device_id_type=pl.DeviceIdType.MESH,
                    )

                remote_op.wait_recv()

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
        tile_num_groups = _tile_plan_value_static(plan_values, 0, num_groups_field)
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
                _write_multi_head_output_from_state(
                    q_vmem_ref, l_scratch_ref, acc_scratch_ref
                )
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
        history_num_groups = _tile_plan_value(
            plan_values, my_id, TilePlanField.HISTORY_NUM_GROUPS
        )
        current_num_groups = _tile_plan_value(
            plan_values, my_id, TilePlanField.CURRENT_NUM_GROUPS
        )
        group_q_hbm_offset = _tile_plan_value(
            plan_values, my_id, TilePlanField.Q_HBM_OFFSET
        )
        group_o_hbm_offset = group_q_hbm_offset
        group_q_tile_size = _tile_plan_value(
            plan_values, my_id, TilePlanField.Q_TILE_SIZE
        )
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
            _write_multi_head_output_from_state(
                o_vmem_ref, l_scratch_ref, acc_scratch_ref
            )
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
    mesh_axis_names: tuple[str, ...] = (AXIS,),
    pcp_axis_name: str = AXIS,
):
    local_tokens = q_multi_head.shape[0]
    kv_heads = q_multi_head.shape[1]
    q_per_kv = q_multi_head.shape[2]
    head_dim = q_multi_head.shape[-1]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        packed_kv_groups = current_kv_local.shape[0]
        page_size = kv_cache_local.shape[-1]
        kv_vmem_shape = (3, 1, *kv_cache_local.shape[2:])
        captured_kv_hbm_shape = (
            *current_kv_local.shape[:-1],
            local_tokens * pcp_size,
        )
    else:
        packed_kv_groups = current_kv_local.shape[1]
        kv_packing = current_kv_local.shape[2]
        page_size = kv_cache_local.shape[2]
        kv_vmem_shape = (3, 1, page_size, packed_kv_groups, kv_packing, head_dim)
        captured_kv_hbm_shape = (
            local_tokens * pcp_size,
            *current_kv_local.shape[1:],
        )
    num_tiles = tile_plan.shape[0]
    # One tile can need an owner-phase partial fresh group plus one old-cache
    # history-boundary group in addition to its full fresh groups.
    max_page_groups = num_tiles * (math.ceil(local_tokens / interleave_size) + 2)
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
            jax.ShapeDtypeStruct(captured_kv_hbm_shape, current_kv_local.dtype),
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
                pltpu.SemaphoreType.REGULAR((num_lanes,)),
                pltpu.VMEM(tile_plan.shape[1:], tile_plan.dtype),
                pltpu.VMEM(block_tables.shape[1:], block_tables.dtype),
                pltpu.VMEM(tile_ids.shape, tile_ids.dtype),
                pltpu.VMEM(group_starts.shape, group_starts.dtype),
                pltpu.VMEM(
                    (pcp_size, num_lanes, RuntimeScheduleField.PACKED_NUM_FIELDS),
                    tile_plan.dtype,
                ),
                pltpu.VMEM(
                    (q_block_size, kv_heads, q_per_kv, head_dim), q_multi_head.dtype
                ),
                pltpu.VMEM(kv_vmem_shape, current_kv_local.dtype),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_acc_scratch_shape, jnp.float32),
            ),
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
        ),
        name="pcp_streaming_attention_current_state_page_groups_multi_head",
    )(
        pass_meta,
        q_multi_head,
        current_kv_local,
        kv_cache_local,
        tile_plan,
        block_tables,
        tile_ids,
        group_starts,
    )


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
    mesh_axis_names: tuple[str, ...] = (AXIS,),
    pcp_axis_name: str = AXIS,
):
    kv_heads = q_multi_head.shape[1]
    q_per_kv = q_multi_head.shape[2]
    head_dim = q_multi_head.shape[-1]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache_local.shape[-1]
        packed_kv_groups = kv_cache_local.shape[2]
        kv_vmem_shape = (3, 1, *kv_cache_local.shape[2:])
    else:
        page_size = kv_cache_local.shape[2]
        packed_kv_groups = kv_cache_local.shape[3]
        kv_packing = kv_cache_local.shape[4]
        kv_vmem_shape = (3, 1, page_size, packed_kv_groups, kv_packing, head_dim)
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
                pltpu.SemaphoreType.REGULAR((num_lanes,)),
                pltpu.VMEM(tile_plan.shape[1:], tile_plan.dtype),
                pltpu.VMEM(block_tables.shape[1:], block_tables.dtype),
                pltpu.VMEM(tile_ids.shape, tile_ids.dtype),
                pltpu.VMEM(group_starts.shape, group_starts.dtype),
                pltpu.VMEM(
                    (pcp_size, num_lanes, RuntimeScheduleField.PACKED_NUM_FIELDS),
                    tile_plan.dtype,
                ),
                pltpu.VMEM(
                    (q_block_size, kv_heads, q_per_kv, head_dim), q_multi_head.dtype
                ),
                pltpu.VMEM(kv_vmem_shape, kv_cache_local.dtype),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_acc_scratch_shape, jnp.float32),
            ),
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
        ),
        input_output_aliases={1: 0},
        name="pcp_streaming_attention_history_output_page_groups_multi_head",
    )(
        pass_meta,
        q_multi_head,
        kv_cache_local,
        tile_plan,
        block_tables,
        tile_ids,
        group_starts,
        m_state,
        l_state,
        acc_state,
    )


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
                pltpu.VMEM(
                    (q_block_size, kv_heads, q_per_kv, head_dim), o_multi_head.dtype
                ),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_lm_scratch_shape, jnp.float32),
                pltpu.VMEM(state_acc_scratch_shape, jnp.float32),
            ),
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        input_output_aliases={1: 0},
        name="pcp_streaming_attention_zero_history_output_multi_head",
    )(
        pass_meta,
        o_multi_head,
        tile_plan,
        tile_ids,
        group_starts,
        m_state,
        l_state,
        acc_state,
    )


def _get_pcp_rank_without_axis_index(pcp_axis_name: str, pcp_size: int) -> jax.Array:
    """Select this PCP rank shard from replicated data without axis_index."""
    ranks = jnp.arange(pcp_size, dtype=jnp.int32)
    exchanged = lax.all_to_all(
        ranks,
        axis_name=pcp_axis_name,
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )
    return exchanged[0]


def _gather_vpages_to_head_major(
    kv_cache_local: jax.Array,
    block_ids: jax.Array,
    *,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> tuple[jax.Array, jax.Array]:
    """Gathers block_ids from all PCP ranks to head-major K and V.

    Each is [kv_heads, num_vpages * page_size * pcp_size, head_dim]; pages are
    transposed before the all-gather, at 1/pcp_size of the gathered volume.
    """
    num_vpages = int(block_ids.shape[0])
    chunks_per_page = page_size // interleave_size
    virtual_page_size = page_size * pcp_size
    num_tokens = num_vpages * virtual_page_size

    local_pages = kv_cache_local[block_ids]
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        local_hm = jnp.transpose(
            local_pages.reshape(
                num_vpages,
                kv_heads,
                2,
                head_dim,
                chunks_per_page,
                interleave_size,
            ),
            (2, 1, 0, 4, 5, 3),
        )
        pages_hm = lax.all_gather(
            local_hm, axis_name=pcp_axis_name, axis=4, tiled=True
        ).reshape(2, kv_heads, num_tokens, head_dim)
        return pages_hm[0], pages_hm[1]

    local_hm = jnp.transpose(
        local_pages.reshape(
            num_vpages,
            chunks_per_page,
            interleave_size,
            2 * kv_heads,
            head_dim,
        ),
        (3, 0, 1, 2, 4),
    )
    pages_hm = lax.all_gather(
        local_hm, axis_name=pcp_axis_name, axis=3, tiled=True
    ).reshape(kv_heads, 2, num_tokens, head_dim)
    return pages_hm[:, 0, :, :], pages_hm[:, 1, :, :]


def _merge_attention_partials(
    o_acc: jax.Array,
    lse_acc: jax.Array,
    o_part: jax.Array,
    lse_part: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Merges normalized attention outputs [h, t, d] by natural-log LSE [h, t] in
    f32.
    """
    lse_max = jnp.maximum(lse_acc, lse_part)
    # A row with no keys in either partial has lse = -inf; a zero max keeps
    # exp() finite.
    lse_max = jnp.where(jnp.isneginf(lse_max), 0.0, lse_max)
    w_acc = jnp.exp(lse_acc - lse_max)
    w_part = jnp.exp(lse_part - lse_max)
    w_sum = w_acc + w_part
    # w_sum is 0 for an empty row and in [1, 2] otherwise, so the floor changes
    # only empty rows.
    inv_w_sum = 1.0 / jnp.maximum(w_sum, 1.0)
    o_merged = (
        o_acc * (w_acc * inv_w_sum)[..., None]
        + o_part.astype(jnp.float32) * (w_part * inv_w_sum)[..., None]
    )
    return o_merged, lse_max + jnp.log(w_sum)


def _gather_vpage_group_kv_and_seg(
    kv_cache_local: jax.Array,
    stage_plan: Any,
    start: int | jax.Array,
    group_size: int,
    *,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gathers unpacked K, V, and token segment IDs for one vpage group."""
    virtual_page_size = page_size * pcp_size
    block_ids = lax.dynamic_slice_in_dim(stage_plan.sorted_block_ids, start, group_size)
    seg_ids = lax.dynamic_slice_in_dim(stage_plan.sorted_seg_ids, start, group_size)
    valid_tokens = lax.dynamic_slice_in_dim(
        stage_plan.sorted_vpage_valid_tokens, start, group_size
    )
    k_hm, v_hm = _gather_vpages_to_head_major(
        kv_cache_local,
        block_ids,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        kv_layout=kv_layout,
        pcp_axis_name=pcp_axis_name,
    )
    token_in_vpage = lax.broadcasted_iota(jnp.int32, (group_size, virtual_page_size), 1)
    kv_seg_ids = jnp.where(
        token_in_vpage < valid_tokens[:, None], seg_ids[:, None], -2
    ).reshape(group_size * virtual_page_size)
    return k_hm, v_hm, kv_seg_ids


def _attend_single_vpage_group(
    kv_cache_local: jax.Array,
    q_scaled_hm: jax.Array,
    stage_plan: Any,
    group_size: int,
    *,
    splash_cfg: _splash_kernel.SplashConfig,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> jax.Array:
    """Runs one Splash call without LSE residuals over a single vpage group."""
    virtual_page_size = page_size * pcp_size
    group_tokens = group_size * virtual_page_size
    splash_single = _splash_kernel.make_splash_mha_single_device(
        _splash_mask.FullMask((q_scaled_hm.shape[1], group_tokens)),
        config=splash_cfg,
        save_residuals=False,
    )
    k_hm, v_hm, kv_seg_ids = _gather_vpage_group_kv_and_seg(
        kv_cache_local,
        stage_plan,
        0,
        group_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        kv_layout=kv_layout,
        pcp_axis_name=pcp_axis_name,
    )
    splash_fn = _splash_kernel.SplashAttentionKernel(
        splash_single.fwd_mask_info._replace(q_sequence=stage_plan.q_sequence),
        splash_single.dkv_mask_info,
        **{**splash_single.kwargs, "mask_function": _causal_mask_function},
    )
    o_group = splash_fn(
        q_scaled_hm,
        k_hm,
        v_hm,
        _splash_kernel.SegmentIds(q=stage_plan.q_seg_ids, kv=kv_seg_ids),
        save_residuals=False,
    )
    return o_group.astype(jnp.float32)


def _attend_vpage_group_with_lse(
    group_idx: jax.Array,
    carry: tuple[jax.Array, jax.Array],
    *,
    kv_cache_local: jax.Array,
    q_scaled_hm: jax.Array,
    stage_plan: Any,
    splash_base: Any,
    group_size: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> tuple[jax.Array, jax.Array]:
    """Accumulates one vpage group's output and log-sum-exp into running state."""
    virtual_page_size = page_size * pcp_size
    start = group_idx * group_size
    k_hm, v_hm, kv_seg_ids = _gather_vpage_group_kv_and_seg(
        kv_cache_local,
        stage_plan,
        start,
        group_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        kv_layout=kv_layout,
        pcp_axis_name=pcp_axis_name,
    )
    splash_fn = _splash_kernel.SplashAttentionKernel(
        splash_base.fwd_mask_info._replace(
            q_sequence=stage_plan.q_sequence - start * virtual_page_size
        ),
        splash_base.dkv_mask_info,
        **{**splash_base.kwargs, "mask_function": _causal_mask_function},
    )
    o_group, stats = splash_fn(
        q_scaled_hm,
        k_hm,
        v_hm,
        _splash_kernel.SegmentIds(q=stage_plan.q_seg_ids, kv=kv_seg_ids),
        save_residuals=True,
    )
    return _merge_attention_partials(*carry, o_group, stats["logsumexp"])


def _attend_multi_vpage_groups(
    kv_cache_local: jax.Array,
    q_scaled_hm: jax.Array,
    stage_plan: Any,
    group_size: int,
    *,
    splash_cfg: _splash_kernel.SplashConfig,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> jax.Array:
    """Streams Splash across multiple vpage groups with online LSE reduction."""
    virtual_page_size = page_size * pcp_size
    group_tokens = group_size * virtual_page_size
    splash_base = _splash_kernel.make_splash_mha_single_device(
        _splash_mask.FullMask((q_scaled_hm.shape[1], group_tokens)),
        config=splash_cfg,
        save_residuals=True,
    )
    start = 0
    k_hm, v_hm, kv_seg_ids = _gather_vpage_group_kv_and_seg(
        kv_cache_local,
        stage_plan,
        start,
        group_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        kv_layout=kv_layout,
        pcp_axis_name=pcp_axis_name,
    )
    splash_fn = _splash_kernel.SplashAttentionKernel(
        splash_base.fwd_mask_info._replace(q_sequence=stage_plan.q_sequence),
        splash_base.dkv_mask_info,
        **{**splash_base.kwargs, "mask_function": _causal_mask_function},
    )
    o_first, stats = splash_fn(
        q_scaled_hm,
        k_hm,
        v_hm,
        _splash_kernel.SegmentIds(q=stage_plan.q_seg_ids, kv=kv_seg_ids),
        save_residuals=True,
    )
    lse_first = stats["logsumexp"]

    num_groups = (stage_plan.num_active_vpages + group_size - 1) // group_size
    body_fn = functools.partial(
        _attend_vpage_group_with_lse,
        kv_cache_local=kv_cache_local,
        q_scaled_hm=q_scaled_hm,
        stage_plan=stage_plan,
        splash_base=splash_base,
        group_size=group_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        kv_layout=kv_layout,
        pcp_axis_name=pcp_axis_name,
    )
    o_acc, _ = lax.fori_loop(
        1,
        num_groups,
        body_fn,
        (o_first.astype(jnp.float32), lse_first),
    )
    return o_acc


def _dispatch_single_group_rungs(
    rem_rungs: tuple[int, ...],
    stage_plan: Any,
    kv_cache_local: jax.Array,
    q_scaled_hm: jax.Array,
    vpage_group_size: int,
    *,
    splash_cfg: _splash_kernel.SplashConfig,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> jax.Array:
    """Dispatches to the smallest single-group rung covering active vpages."""
    r0 = rem_rungs[0]

    def _single_fn(_):
        return _attend_single_vpage_group(
            kv_cache_local,
            q_scaled_hm,
            stage_plan,
            r0,
            splash_cfg=splash_cfg,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            kv_layout=kv_layout,
            pcp_axis_name=pcp_axis_name,
        )

    if len(rem_rungs) == 1:

        def _fallback_fn(_):
            return _attend_multi_vpage_groups(
                kv_cache_local,
                q_scaled_hm,
                stage_plan,
                vpage_group_size,
                splash_cfg=splash_cfg,
                kv_heads=kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                kv_layout=kv_layout,
                pcp_axis_name=pcp_axis_name,
            )

    else:

        def _fallback_fn(_):
            return _dispatch_single_group_rungs(
                rem_rungs[1:],
                stage_plan,
                kv_cache_local,
                q_scaled_hm,
                vpage_group_size,
                splash_cfg=splash_cfg,
                kv_heads=kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                kv_layout=kv_layout,
                pcp_axis_name=pcp_axis_name,
            )

    return lax.cond(
        stage_plan.num_active_vpages <= r0,
        _single_fn,
        _fallback_fn,
        operand=None,
    )


def _attend_over_vpage_groups(
    kv_cache_local: jax.Array,
    q_scaled_hm: jax.Array,
    stage_plan: Any,
    *,
    splash_cfg: _splash_kernel.SplashConfig,
    vpage_group_size: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> jax.Array:
    """Runs Splash over the plan's active vpages, one vpage group per step.

    Returns the f32 attention output [q_heads, local_tokens, head_dim].
    """
    total_vpages = stage_plan.sorted_block_ids.shape[0]
    if total_vpages % vpage_group_size:
        # dynamic_slice clamps a group that runs past the end, which would count its
        # pages twice.
        raise ValueError(
            "The stage plan must hold whole vpage groups: "
            f"{stage_plan.sorted_block_ids.shape=} {vpage_group_size=}."
        )

    if vpage_group_size == PCP_STREAMING_RPA_VPAGE_GROUP_SIZE:
        rungs = tuple(
            r
            for r in PCP_STREAMING_RPA_SINGLE_GROUP_RUNGS
            if r <= total_vpages and total_vpages % r == 0
        )
        if rungs:
            return _dispatch_single_group_rungs(
                rungs,
                stage_plan,
                kv_cache_local,
                q_scaled_hm,
                vpage_group_size,
                splash_cfg=splash_cfg,
                kv_heads=kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                kv_layout=kv_layout,
                pcp_axis_name=pcp_axis_name,
            )

    if vpage_group_size >= 32 and (vpage_group_size & (vpage_group_size - 1)) == 0:
        return lax.cond(
            stage_plan.num_active_vpages <= vpage_group_size,
            lambda _: _attend_single_vpage_group(
                kv_cache_local,
                q_scaled_hm,
                stage_plan,
                vpage_group_size,
                splash_cfg=splash_cfg,
                kv_heads=kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                kv_layout=kv_layout,
                pcp_axis_name=pcp_axis_name,
            ),
            lambda _: _attend_multi_vpage_groups(
                kv_cache_local,
                q_scaled_hm,
                stage_plan,
                vpage_group_size,
                splash_cfg=splash_cfg,
                kv_heads=kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                kv_layout=kv_layout,
                pcp_axis_name=pcp_axis_name,
            ),
            operand=None,
        )
    return _attend_multi_vpage_groups(
        kv_cache_local,
        q_scaled_hm,
        stage_plan,
        vpage_group_size,
        splash_cfg=splash_cfg,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        kv_layout=kv_layout,
        pcp_axis_name=pcp_axis_name,
    )


def _causal_mask_function(q_ids: jax.Array, kv_ids: jax.Array) -> jax.Array:
    return q_ids >= kv_ids


def _rank_chunk_count_before(
    chunk_pos: jax.Array, rank: jax.Array, pcp_size: int
) -> jax.Array:
    full_cycles = chunk_pos // pcp_size
    rem = chunk_pos - full_cycles * pcp_size
    return full_cycles + jnp.clip(rem - rank, 0, 1)


def _route_current_kv_to_captured_kv(
    current_kv_local: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    local_tokens: int,
    max_num_seqs: int,
    pcp_size: int,
    interleave_size: int,
    kv_layout: KVLayout,
    pcp_axis_name: str,
) -> jax.Array:
    """Routes current_kv_local to cache-owner captured_kv_local.

    The caller checks alignment.
    """
    active_reqs = jnp.clip(distribution[2], 0, max_num_seqs)
    req_ids = lax.broadcasted_iota(jnp.int32, (max_num_seqs,), 0)
    q_starts = cu_q_lens[:max_num_seqs]
    q_ends = cu_q_lens[1 : max_num_seqs + 1]
    q_len_raw = q_ends - q_starts
    req_active = jnp.logical_and(req_ids < active_reqs, q_len_raw > 0)
    q_lens = jnp.where(req_active, q_len_raw, 0)
    total_kv_lens = jnp.where(req_active, kv_lens[:max_num_seqs], 0)
    hist_tokens = jnp.maximum(total_kv_lens - q_lens, 0)

    num_local_chunks = local_tokens // interleave_size
    num_total_chunks = pcp_size * num_local_chunks
    ranks = lax.broadcasted_iota(jnp.int32, (1, pcp_size), 1)
    q_starts_c = (q_starts // interleave_size)[:, None]
    q_ends_c = (q_ends // interleave_size)[:, None]
    hist_c = (hist_tokens // interleave_size)[:, None]
    kv_ends_c = (total_kv_lens // interleave_size)[:, None]

    local_c_counts = jnp.where(
        req_active[:, None],
        _rank_chunk_count_before(q_ends_c, ranks, pcp_size)
        - _rank_chunk_count_before(q_starts_c, ranks, pcp_size),
        0,
    )
    local_c_ends = jnp.cumsum(local_c_counts, axis=0)
    local_c_starts = local_c_ends - local_c_counts

    my_rank = _get_pcp_rank_without_axis_index(pcp_axis_name, pcp_size)
    wb_k0 = _rank_chunk_count_before(hist_c[:, 0], my_rank, pcp_size)
    wb_c_counts = jnp.where(
        req_active,
        _rank_chunk_count_before(kv_ends_c[:, 0], my_rank, pcp_size) - wb_k0,
        0,
    )
    wb_c_ends = jnp.cumsum(wb_c_counts, axis=0)
    wb_c_starts = wb_c_ends - wb_c_counts

    d_c = lax.broadcasted_iota(jnp.int32, (num_local_chunks, 1), 0)
    in_req = jnp.logical_and(
        req_active[None, :],
        jnp.logical_and(d_c >= wb_c_starts[None, :], d_c < wb_c_ends[None, :]),
    )
    d_req = jnp.clip(
        jnp.sum(jnp.where(in_req, req_ids[None, :] + 1, 0), axis=1) - 1,
        0,
        max_num_seqs - 1,
    )
    d_valid = jnp.any(in_req, axis=1)
    m_wb = d_c[:, 0] - wb_c_starts[d_req]
    abs_c = (wb_k0[d_req] + m_wb) * pcp_size + my_rank
    g_c = q_starts_c[d_req, 0] + (abs_c - hist_c[d_req, 0])
    src_rank = jnp.mod(g_c, pcp_size)
    src_k0 = _rank_chunk_count_before(q_starts_c[d_req, 0], src_rank, pcp_size)
    req_one_hot = d_req[:, None] == req_ids[None, :]
    rank_one_hot = src_rank[:, None] == ranks
    src_c_start = jnp.sum(
        jnp.where(
            req_one_hot[:, :, None] & rank_one_hot[:, None, :],
            local_c_starts[None, :, :],
            0,
        ),
        axis=(1, 2),
    )
    src_c = src_c_start + (g_c // pcp_size - src_k0)
    flat_src_c = jnp.where(
        d_valid,
        jnp.clip(src_rank * num_local_chunks + src_c, 0, num_total_chunks - 1),
        0,
    )
    onehot = flat_src_c[:, None] == lax.broadcasted_iota(
        jnp.int32, (num_local_chunks, num_total_chunks), 1
    )
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        kv_c = current_kv_local.reshape(
            *current_kv_local.shape[:-1], num_local_chunks, interleave_size
        )
        all_kv_c = lax.all_gather(kv_c, axis_name=pcp_axis_name, axis=-2, tiled=True)
        sel = jnp.sum(
            jnp.where(
                onehot[None, None, None, :, :, None],
                all_kv_c[:, :, :, None, :, :],
                jnp.zeros((1,), dtype=current_kv_local.dtype),
            ),
            axis=-2,
        )
        return sel.reshape(current_kv_local.shape)
    kv_c = current_kv_local.reshape(
        num_local_chunks, interleave_size, *current_kv_local.shape[1:]
    )
    all_kv_c = lax.all_gather(kv_c, axis_name=pcp_axis_name, axis=0, tiled=True)
    sel = jnp.sum(
        jnp.where(
            onehot[:, :, None, None, None, None],
            all_kv_c[None, :, :, :, :, :],
            jnp.zeros((1,), dtype=current_kv_local.dtype),
        ),
        axis=1,
    )
    return sel.reshape(current_kv_local.shape)


def pcp_streaming_attention_page_groups_packed_local_from_metadata(
    q_local,
    current_kv_local,
    kv_cache_local,
    kv_lens,
    cu_q_lens,
    distribution,
    writeback_segment_descriptors,
    writeback_num_segments,
    *,
    pcp_size: int,
    interleave_size: int,
    q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    sm_scale: float | None = 1.0,
    q_compute_size: int | None = None,
    num_lanes: int = 1,
    k_scale: float | None = None,
    v_scale: float | None = None,
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
    mesh_axis_names: tuple[str, ...] = (AXIS,),
    pcp_axis_name: str = AXIS,
    stage_plan: Any,
    vpage_group_size: int,
):
    """Unified local Tokamax Splash over [history + current].

    Runs after local KV cache writeback.
    """
    if num_lanes != 1:
        raise NotImplementedError("Only num_lanes == 1 is supported.")
    if q_block_size <= 0 or (q_compute_size is not None and q_compute_size <= 0):
        raise ValueError("q_block_size and q_compute_size must be positive.")
    if pcp_axis_name not in mesh_axis_names:
        raise ValueError(f"{pcp_axis_name=} must be in {mesh_axis_names=}.")
    _validate_packed_kv_layout(
        q_local, kv_cache_local, kv_layout=kv_layout, k_scale=k_scale, v_scale=v_scale
    )
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        if (
            current_kv_local.shape[:-1] != kv_cache_local.shape[1:-1]
            or current_kv_local.shape[-1] < q_local.shape[0]
        ):
            raise ValueError(
                "SEQ_ALONG_LANE fresh KV must match the cache head layout and "
                "cover all local Q rows: "
                f"{current_kv_local.shape=} {q_local.shape[0]=} "
                f"cache_head_layout={kv_cache_local.shape[1:-1]}."
            )
        page_size = kv_cache_local.shape[-1]
    else:
        if current_kv_local.shape != (q_local.shape[0], *kv_cache_local.shape[2:]):
            raise ValueError(
                "Fresh packed KV layout must match local Q rows and cache "
                f"tail: {current_kv_local.shape=} {q_local.shape[0]=} "
                f"cache_tail={kv_cache_local.shape[2:]}."
            )
        page_size = kv_cache_local.shape[1]

    local_tokens, kv_heads, q_per_kv, head_dim = q_local.shape
    q_heads = kv_heads * q_per_kv
    max_num_seqs = int(kv_lens.shape[0])

    captured_kv_local = lax.cond(
        stage_plan.needs_reroute,
        lambda kv_in: _route_current_kv_to_captured_kv(
            kv_in,
            cu_q_lens,
            kv_lens,
            distribution,
            local_tokens=local_tokens,
            max_num_seqs=max_num_seqs,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            kv_layout=kv_layout,
            pcp_axis_name=pcp_axis_name,
        ),
        lambda kv_in: kv_in,
        current_kv_local,
    )

    kv_cache_local = write_captured_kv_to_local_cache(
        kv_cache_local,
        captured_kv_local,
        writeback_segment_descriptors,
        writeback_num_segments,
        kv_layout=kv_layout,
    )
    eff_q_scale = (
        (1.0 if sm_scale is None else float(sm_scale))
        * (1.0 if k_scale is None else float(k_scale))
        * _splash_kernel.LOG2E
    )
    q_hm = jnp.transpose(q_local, (1, 2, 0, 3)).reshape(q_heads, local_tokens, head_dim)
    q_scaled_hm = q_hm * jnp.asarray(eff_q_scale, dtype=q_hm.dtype)
    splash_cfg = _splash_kernel.SplashConfig(
        block_q=min(128, local_tokens),
        block_kv=512,
        block_kv_compute=256,
        num_stacked_q_heads=math.gcd(16, q_per_kv),
        use_base2_exp=True,
        q_prescaled_log2e=True,
    )
    o_acc = _attend_over_vpage_groups(
        kv_cache_local,
        q_scaled_hm,
        stage_plan,
        splash_cfg=splash_cfg,
        vpage_group_size=vpage_group_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        kv_layout=kv_layout,
        pcp_axis_name=pcp_axis_name,
    )
    o_hm = (o_acc * (1.0 if v_scale is None else float(v_scale))).astype(q_local.dtype)
    output = jnp.transpose(
        o_hm.reshape(kv_heads, q_per_kv, local_tokens, head_dim), (2, 0, 1, 3)
    )
    output = jnp.where((stage_plan.q_seg_ids >= 0)[:, None, None, None], output, 0)
    return output, kv_cache_local
