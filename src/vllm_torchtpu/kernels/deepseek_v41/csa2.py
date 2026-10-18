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
"""Fused BF16 SWA + FP8 CSA + inverse RoPE in request-local BQ groups.

SWA keeps two fixed 128-token blocks and prefetches the following block into
separate staging memory. Causal/window masks advance without shifting KV lanes.
An MTP group that needs a third block is split; overlapping request-tail blocks
are masked to disjoint ranges. MXU RHS loading performs the K transposes.

One program traverses requests; two SWA staging blocks and the next Q/KV bank
prefetch across request boundaries during the current group's computation. Each bank
has independent DMA semaphores; only Q, packed KV, scales and RoPE are doubled.
The BF16 KV workspace and output buffers are reused after their consumers finish.

CSA dequantizes KV to BF16 once for QK/PV reuse. All GEMMs use BF16 operands
and FP32 accumulators, softmax uses FP32, and probabilities are stored in BF16.
Each QK launches before the preceding Q's softmax. CSA PV starts as soon as
its P is ready; SWA PV reuses its RHS for two Qs. Both PV results stay in
independent MRB regions while the next tile runs softmax. QK uses two slots,
and each PV branch uses four. When ten slots exceed MRB capacity, one-Q tiles
use six slots instead. Startup, odd/even tails and final pops are explicit.
CPU interpret uses ordinary dots to check the math; it does not emulate the
asynchronous MXU protocol.

The initial grouped DMA/RoPE organization was adapted from
pallas-samples/kernels/deepseek-v4.1/csa2/csa2.py. Gather/cache publication,
non-causal windows, TP collectives and output projection remain caller-owned.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def main_kernel(
    starts_ref: Any,
    positions_ref: Any,
    main_lengths_ref: Any,
    q_hbm_ref: Any,
    main_kv_hbm_ref: Any,
    main_scale_hbm_ref: Any,
    swa_kv_hbm_ref: Any,
    sink_hbm_ref: Any,
    rope_hbm_ref: Any,
    output_hbm_ref: Any,
    sink_ref: Any,
    q_buffer_ref: Any,
    main_fp8_buffer_ref: Any,
    main_scale_buffer_ref: Any,
    main_bf16_ref: Any,
    swa_first_ref: Any,
    swa_shared_ref: Any,
    swa_cache_ids: Any,
    swa_cache_bases: Any,
    swa_prefetched_id: Any,
    rope_buffer_ref: Any,
    swa_score_ref: Any,
    main_p_ref: Any,
    swa_p_ref: Any,
    output_ref: Any,
    request_swa_ref: Any,
    dma_sem: Any,
    *,
    scale: float,
    compress_ratio: int,
    bq: int,
    interpret: bool,
    swa_metadata_ref: Any = None,
    swa_bounds_ref: Any = None,
) -> None:
    swa_banks = (
        swa_kv_hbm_ref if isinstance(swa_kv_hbm_ref, tuple) else (swa_kv_hbm_ref,)
    )
    swa_kv_hbm_ref = swa_banks[0]
    request_limit = starts_ref.shape[0] - 1
    num_heads = q_hbm_ref.shape[1]
    kv_slots = main_bf16_ref.shape[0]
    mxu_heads = pl.cdiv(num_heads, 16) * 16
    mrb_stride = mxu_heads // 4
    window_size = 128
    sink_dma = pltpu.make_async_copy(sink_hbm_ref, sink_ref, dma_sem.at[0])
    sink_dma.start()
    # Incomplete groups still participate in the grouped SWA QK.
    q_buffer_ref[...] = jnp.zeros(q_buffer_ref.shape, q_buffer_ref.dtype)
    sink_dma.wait()
    sink = jnp.broadcast_to(sink_ref[...][None, :], (128, num_heads)).T[:, :1]

    def next_nonempty(request_index):
        def empty(index):
            safe = jnp.minimum(index, request_limit - 1)
            return (index < request_limit) & (starts_ref[safe] == starts_ref[safe + 1])

        return lax.while_loop(empty, lambda index: index + 1, request_index)

    def request_info(request_index):
        start, end = starts_ref[request_index], starts_ref[request_index + 1]
        first_query_position = positions_ref[jnp.minimum(start, q_hbm_ref.shape[0] - 1)]
        anchor = jnp.maximum(0, first_query_position - 127)

        if swa_metadata_ref is not None:
            # Absolute alignment makes every 128-token block a set of whole pages
            # or an aligned slice of one page, independent of the query position.
            anchor = anchor // 128 * 128
        if swa_bounds_ref is not None:
            anchor = swa_bounds_ref[request_index, 0] // 64 * 64

        return start, end, first_query_position, anchor

    def block_dmas(wanted, request_index, destination, semaphore):
        start, end, first_query_position, anchor = request_info(request_index)
        if swa_metadata_ref is None:
            source_q = jnp.minimum(
                end - 1,
                start
                + jnp.maximum(0, anchor + wanted * 128 + 127 - first_query_position),
            )
            return (
                (
                    pltpu.make_async_copy(
                        swa_kv_hbm_ref.at[source_q], destination, semaphore
                    ),
                    jnp.bool_(True),
                ),
            )
        word_pages = swa_kv_hbm_ref.dtype == jnp.int32
        page_size = swa_kv_hbm_ref.shape[2 if word_pages else 1]
        rows_per_copy = min(page_size, 64 if swa_bounds_ref is not None else 128)
        copies = []
        for offset in range(0, 128, rows_per_copy):
            position = anchor + wanted * 128 + offset
            column = (position - swa_metadata_ref[request_index, 0]) // page_size
            safe_column = jnp.clip(column, 0, swa_metadata_ref.shape[1] - 3)
            page = swa_metadata_ref[request_index, safe_column + 2]
            page = jnp.where(
                (column >= 0) & (column < swa_metadata_ref.shape[1] - 2), page, 0
            )
            for bank, source in enumerate(swa_banks):
                if len(swa_banks) == 1:
                    own, local_page = jnp.bool_(True), page
                else:
                    split = swa_banks[0].shape[0]
                    own = page >= split if bank else page < split
                    local_page = page - split if bank else page
                    # Construct only legal views, even for the skipped bank.
                    local_page = jnp.where(own, local_page, 0)
                if word_pages:
                    for plane in range(2):
                        copies.append(
                            (
                                pltpu.make_async_copy(
                                    source.at[
                                        local_page,
                                        plane,
                                        pl.ds(position % page_size, rows_per_copy),
                                        :,
                                    ],
                                    destination.at[
                                        plane, pl.ds(offset, rows_per_copy), :
                                    ],
                                    semaphore,
                                ),
                                own,
                            )
                        )
                else:
                    copies.append(
                        (
                            pltpu.make_async_copy(
                                source.at[
                                    local_page,
                                    pl.ds(position % page_size, rows_per_copy),
                                    :,
                                ],
                                destination.at[pl.ds(offset, rows_per_copy), :],
                                semaphore,
                            ),
                            own,
                        )
                    )
        return tuple(copies)

    def transfer_block(wanted, request_index, destination, semaphore, *, wait=False):
        for copy, own in block_dmas(wanted, request_index, destination, semaphore):

            @pl.when(own)
            def transfer(*, copy=copy):
                if wait:
                    copy.wait()
                else:
                    copy.start()

    def request_window(request_index, *, wait=False):
        for offset in range(2):
            transfer_block(
                offset,
                request_index,
                request_swa_ref.at[offset],
                dma_sem.at[2 + 9 * bq + offset],
                wait=wait,
            )

    def group_count(request_index, t0):
        start, end, first_query_position, anchor = request_info(request_index)
        if swa_bounds_ref is not None:
            return jnp.minimum(bq, end - t0)
        # Split before a group would need a third SWA block.
        p0 = positions_ref[jnp.minimum(t0, q_hbm_ref.shape[0] - 1)]
        block_id = (jnp.maximum(0, p0 - 127) - anchor) // 128
        return jnp.minimum(
            jnp.minimum(bq, end - t0), anchor + (block_id + 2) * 128 - p0
        )

    def query_dmas(t0, i, bank):
        return tuple(
            pltpu.make_async_copy(src, dst, dma_sem.at[2 + 4 * (bank * bq + i) + k])
            for k, (src, dst) in enumerate(
                (
                    (q_hbm_ref.at[t0 + i], q_buffer_ref.at[bank, i]),
                    (main_kv_hbm_ref.at[t0 + i], main_fp8_buffer_ref.at[bank, i]),
                    (main_scale_hbm_ref.at[t0 + i], main_scale_buffer_ref.at[bank, i]),
                    (rope_hbm_ref.at[t0 + i], rope_buffer_ref.at[bank, i]),
                )
            )
        )

    def bulk_dmas(t0, bank):
        return tuple(
            pltpu.make_async_copy(src, dst, dma_sem.at[2 + 4 * bank * bq + k])
            for k, (src, dst) in enumerate(
                (
                    (q_hbm_ref.at[pl.ds(t0, bq)], q_buffer_ref.at[bank]),
                    (main_kv_hbm_ref.at[pl.ds(t0, bq)], main_fp8_buffer_ref.at[bank]),
                    (
                        main_scale_hbm_ref.at[pl.ds(t0, bq)],
                        main_scale_buffer_ref.at[bank],
                    ),
                    (rope_hbm_ref.at[pl.ds(t0, bq)], rope_buffer_ref.at[bank]),
                )
            )
        )

    def transfer_group(request_index, t0, bank, *, wait=False):
        def per_query():
            @pl.loop(0, group_count(request_index, t0))
            def transfer(i):
                for dma in query_dmas(t0, i, bank):
                    if wait:
                        dma.wait()
                    else:
                        dma.start()

        if q_hbm_ref.shape[0] >= bq:

            def bulk():
                for dma in bulk_dmas(t0, bank):
                    if wait:
                        dma.wait()
                    else:
                        dma.start()

            lax.cond(group_count(request_index, t0) == bq, bulk, per_query)
        else:
            per_query()

    first_request = next_nonempty(jnp.int32(0))
    first_t0 = starts_ref[first_request]

    @pl.when(first_request < request_limit)
    def prime_group():
        request_window(first_request)
        transfer_group(first_request, first_t0, 0)

    def process_group(state):
        request_index, t0, bank = state
        start, end, first_query_position, anchor = request_info(request_index)

        def start_block_dma(wanted):
            transfer_block(wanted, request_index, swa_first_ref, dma_sem.at[1])

        def wait_block_dma(wanted):
            transfer_block(
                wanted, request_index, swa_first_ref, dma_sem.at[1], wait=True
            )

        q_ref = q_buffer_ref.at[bank]
        main_fp8_ref = main_fp8_buffer_ref.at[bank]
        main_scale_ref = main_scale_buffer_ref.at[bank]
        rope_ref = rope_buffer_ref.at[bank]
        p0 = positions_ref[t0]
        block_id = (
            0
            if swa_bounds_ref is not None
            else (jnp.maximum(0, p0 - 127) - anchor) // 128
        )
        count = group_count(request_index, t0)
        next_request = lax.cond(
            t0 + count < end,
            lambda: request_index,
            lambda: next_nonempty(request_index + 1),
        )
        next_t0 = jnp.where(
            next_request == request_index, t0 + count, starts_ref[next_request]
        )

        def store_block(wanted, slot, source_ref):
            source_q = jnp.minimum(
                end - 1,
                start
                + jnp.maximum(0, anchor + wanted * 128 + 127 - first_query_position),
            )
            rows = pl.ds(slot * 128, 128)
            keep = jnp.ones((128, 1), jnp.bool_)
            if swa_metadata_ref is not None:
                left = jnp.maximum(0, first_query_position - 127)
                right = positions_ref[jnp.maximum(start, end - 1)] + 1
                if swa_bounds_ref is not None:
                    left, right = (
                        swa_bounds_ref[request_index, 0],
                        swa_bounds_ref[request_index, 1],
                    )
                left = jnp.maximum(left, swa_metadata_ref[request_index, 1])
                columns = 128 if swa_kv_hbm_ref.dtype == jnp.int32 else 512
                absolute = (
                    anchor
                    + wanted * 128
                    + lax.broadcasted_iota(jnp.int32, (128, columns), 0)
                )
                keep = (absolute >= left) & (absolute < right)
            if swa_kv_hbm_ref.dtype == jnp.int32:
                for plane in range(2):
                    bits = jnp.where(keep, source_ref[plane], 0)
                    low = lax.bitcast_convert_type(bits << 16, jnp.float32)
                    high = lax.bitcast_convert_type(
                        bits & jnp.int32(-65536), jnp.float32
                    )
                    swa_shared_ref[rows, pl.ds(plane * 256, 128)] = jnp.where(
                        keep, low, 0
                    ).astype(jnp.bfloat16)
                    swa_shared_ref[rows, pl.ds(plane * 256 + 128, 128)] = jnp.where(
                        keep, high, 0
                    ).astype(jnp.bfloat16)
            else:
                swa_shared_ref[rows, :] = jnp.where(keep, source_ref[...], 0)
            swa_cache_ids[slot] = wanted
            swa_cache_bases[slot] = (
                jnp.maximum(0, positions_ref[source_q] - 127)
                if swa_metadata_ref is None
                else anchor + wanted * 128
            )

        @pl.when(t0 == start)
        def reset_request_window():
            request_window(request_index, wait=True)
            for offset in range(2):
                store_block(offset, offset, request_swa_ref.at[offset])
            swa_prefetched_id[0] = -1

            @pl.when(
                (swa_bounds_ref is None)
                & (anchor + 2 * 128 - 1 <= positions_ref[end - 1])
            )
            def prefetch_following_block():
                start_block_dma(2)
                swa_prefetched_id[0] = 2

        for offset in range(2):
            wanted = block_id + offset
            slot = wanted % 2

            @pl.when(swa_cache_ids[slot] != wanted)
            def fill_block(*, wanted=wanted, slot=slot):
                # Gathered windows expose a complete block at its last Q.
                # At request end the available window overlaps the previous
                # block; the score mask removes that overlap, including +5 MTP.
                @pl.when(swa_prefetched_id[0] != wanted)
                def start_block():
                    start_block_dma(wanted)

                wait_block_dma(wanted)
                swa_prefetched_id[0] = -1
                store_block(wanted, slot, swa_first_ref)

                @pl.when(
                    (swa_bounds_ref is None)
                    & (wanted >= 1)
                    & (anchor + (wanted + 1) * 128 - 1 <= positions_ref[end - 1])
                )
                def prefetch_block():
                    # Stage the following block without overwriting either live K.
                    start_block_dma(wanted + 1)
                    swa_prefetched_id[0] = wanted + 1

        def dequant(i):
            slot = i if kv_slots == bq else i % kv_slots
            # Byte g of each word holds feature g * 128 + lane.
            words_ref = main_fp8_ref.at[i].bitcast(jnp.uint32).reshape(512, 128)
            for row in range(0, 512, 128):
                rows = pl.ds(row, 128)
                words = words_ref[rows, :]
                scales = main_scale_ref[i, 0, rows][:, None]
                for group in range(4):
                    if interpret:
                        # unpack_elementwise has no CPU lowering.
                        bits = (words >> (8 * group)) & 0xFF
                    else:
                        bits = pltpu.unpack_elementwise(
                            words,
                            index=group,
                            packed_dtype=jnp.uint8,
                            unpacked_dtype=jnp.uint32,
                        )
                    values = lax.bitcast_convert_type(
                        bits.astype(jnp.uint8), jnp.float8_e4m3fn
                    ).astype(jnp.float32)
                    main_bf16_ref[slot, rows, pl.ds(group * 128, 128)] = (
                        values * scales
                    ).astype(jnp.bfloat16)

        transfer_group(request_index, t0, bank, wait=True)

        # Both request windows have been consumed before staging the next request.
        @pl.when(next_request < request_limit)
        def prefetch_group():
            @pl.when(next_request != request_index)
            def next_window():
                request_window(next_request)

            transfer_group(next_request, next_t0, 1 - bank)

        def csa_launch(i, addr, *, pv):
            slot = i if kv_slots == bq else i % kv_slots
            for k in range(2):
                ks = pl.ds(k * 256, 256)
                for mxu in range(2):
                    ns = pl.ds(mxu * 256, 256)
                    rhs = (
                        main_bf16_ref[slot, ks, ns]
                        if pv
                        else main_bf16_ref[slot, ns, ks]
                    )
                    pltpu.matmul_push_rhs(
                        rhs, staging_register=0, mxu_index=mxu, transpose=not pv
                    )
                    lhs = main_p_ref[i, :, ks] if pv else q_ref[i, :, ks]
                    lhs = jnp.pad(lhs, ((0, mxu_heads - num_heads), (0, 0)))
                    pltpu.matmul_acc_lhs(addr, lhs, mxu_index=mxu, load_staged_rhs=0)

        def csa_pop(i, addr, *, pv):
            if interpret:
                lhs = main_p_ref[i] if pv else q_ref[i]
                rhs = main_bf16_ref[i] if pv else main_bf16_ref[i].T
                return jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
            return jnp.concatenate(
                tuple(
                    pltpu.matmul_pop(
                        addr, (mxu_heads, 256), jnp.float32, mxu_index=mxu
                    )[:num_heads]
                    for mxu in range(2)
                ),
                axis=1,
            )

        if interpret:
            swa_score_ref[...] = jnp.dot(
                q_ref[...].reshape(bq * num_heads, 512),
                swa_shared_ref[...].T,
                preferred_element_type=jnp.float32,
            ).reshape(bq, num_heads, 256)
        else:
            # Split legal head domains; a sub-16-row head tail is indivisible.
            first_heads = min(num_heads, ((num_heads + 16) // 32) * 16)
            partitions = [(0, 0, first_heads)] if first_heads else []
            if num_heads > first_heads:
                partitions.append(
                    (len(partitions), first_heads, num_heads - first_heads)
                )
            for k in range(2):
                ks = pl.ds(k * 256, 256)
                rhs = swa_shared_ref[:, ks]
                for mxu, head_start, heads in partitions:
                    rows = pl.cdiv(heads, 16) * 16
                    pltpu.matmul_push_rhs(
                        rhs, staging_register=0, mxu_index=mxu, transpose=True
                    )
                    for i in range(bq):
                        lhs = q_ref[i, pl.ds(head_start, heads), ks]
                        lhs = jnp.pad(lhs, ((0, rows - heads), (0, 0)))
                        pltpu.matmul_acc_lhs(
                            i * rows // 4,
                            lhs,
                            mxu_index=mxu,
                            load_staged_rhs=0 if i == 0 else None,
                        )
            for i in range(bq):
                for mxu, head_start, heads in partitions:
                    rows = pl.cdiv(heads, 16) * 16
                    swa_score_ref[i, pl.ds(head_start, heads), :] = pltpu.matmul_pop(
                        i * rows // 4, (rows, 256), jnp.float32, mxu_index=mxu
                    )[:heads]

        dequant(0)
        if not interpret:
            csa_launch(0, 0, pv=False)

        def softmax(i, addr, *, has_next=True):
            if has_next:
                dequant(i + 1)
            if not interpret and has_next:
                csa_launch(i + 1, mrb_stride - addr, pv=False)
            main_logits = csa_pop(i, addr, pv=False) * scale
            main_count = main_lengths_ref[t0 + i]
            main_logits = jnp.where(
                jnp.arange(512)[None, :] < main_count, main_logits, -jnp.inf
            )
            lanes = jnp.arange(128)
            key_positions = jnp.concatenate(
                (swa_cache_bases[0] + lanes, swa_cache_bases[1] + lanes)
            )
            # Concatenate integer boundaries before constructing the boolean mask.
            unique = key_positions >= jnp.concatenate(
                tuple(
                    jnp.full((128,), anchor + swa_cache_ids[slot] * 128, jnp.int32)
                    for slot in range(2)
                )
            )
            swa_valid = (key_positions >= jnp.maximum(0, p0 + i - window_size + 1)) & (
                key_positions <= p0 + i
            )
            if swa_bounds_ref is not None:
                swa_valid = (key_positions >= swa_bounds_ref[request_index, 0]) & (
                    key_positions < swa_bounds_ref[request_index, 1]
                )
            if swa_metadata_ref is not None:
                swa_valid &= key_positions >= swa_metadata_ref[request_index, 1]
            swa_logits = jnp.where(
                (swa_valid & unique)[None, :], swa_score_ref[i] * scale, -jnp.inf
            )
            # Keep both branches in one FP32 softmax tile; round only P.
            logits = jnp.concatenate((main_logits, swa_logits), axis=1)
            maximum = jnp.maximum(sink, jnp.max(logits, axis=1, keepdims=True))
            exponentials = jnp.exp(logits - maximum)
            denominator = jnp.exp(sink - maximum) + jnp.sum(
                exponentials, axis=1, keepdims=True
            )
            probabilities = (exponentials / denominator).astype(jnp.bfloat16)
            main_p_ref[i] = probabilities[:, :512]
            swa_p_ref[i] = probabilities[:, 512:]

        # Each 256-wide FP32 MRB has 256 addresses, four rows per address.
        if 10 * mrb_stride <= 256:

            def launch_tile(first, slot, *, size=2, has_next=True):
                for row in range(size):
                    softmax(
                        first + row,
                        row * mrb_stride,
                        has_next=row + 1 < size or has_next,
                    )
                    if not interpret:
                        csa_launch(
                            first + row, (2 + 2 * slot + row) * mrb_stride, pv=True
                        )
                if not interpret:
                    for mxu in range(2):
                        pltpu.matmul_push_rhs(
                            swa_shared_ref[:, pl.ds(mxu * 256, 256)],
                            staging_register=0,
                            mxu_index=mxu,
                        )
                        for row in range(size):
                            lhs = jnp.pad(
                                swa_p_ref[first + row],
                                ((0, mxu_heads - num_heads), (0, 0)),
                            )
                            pltpu.matmul_acc_lhs(
                                (6 + 2 * slot + row) * mrb_stride,
                                lhs,
                                mxu_index=mxu,
                                load_staged_rhs=0 if row == 0 else None,
                            )

            def finish_tile_output(i, slot, row):
                main_output = csa_pop(i, (2 + 2 * slot + row) * mrb_stride, pv=True)
                if interpret:
                    swa_output = jnp.dot(
                        swa_p_ref[i],
                        swa_shared_ref[...],
                        preferred_element_type=jnp.float32,
                    )
                else:
                    swa_output = jnp.concatenate(
                        tuple(
                            pltpu.matmul_pop(
                                (6 + 2 * slot + row) * mrb_stride,
                                (mxu_heads, 256),
                                jnp.float32,
                                mxu_index=mxu,
                            )[:num_heads]
                            for mxu in range(2)
                        ),
                        axis=1,
                    )
                output = main_output + swa_output
                tail = output[:, 384:]
                partner = jnp.where(
                    (jnp.arange(128) % 2 == 0)[None, :],
                    pltpu.roll(tail, 127, axis=1),
                    pltpu.roll(tail, 1, axis=1),
                )
                rotated = (
                    tail * rope_ref[i, 0][None, :] + partner * rope_ref[i, 1][None, :]
                )
                output_ref[i, :, :384] = output[:, :384].astype(jnp.bfloat16)
                output_ref[i, :, 384:] = rotated.astype(jnp.bfloat16)
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[2 + 8 * bq + i],
                ).start()

            def finish_tile(first, slot, *, size=2):
                for row in range(size):
                    finish_tile_output(first + row, slot, row)

            def advance_tile(first, slot, *, size=2, has_next=True):
                launch_tile(first, slot, size=size, has_next=has_next)
                finish_tile(first - 2, 1 - slot)

            @pl.when(count == 1)
            def one():
                launch_tile(0, 0, size=1, has_next=False)
                finish_tile(0, 0, size=1)

            @pl.when(count == 2)
            def two():
                launch_tile(0, 0, has_next=False)
                finish_tile(0, 0)

            @pl.when(count > 2)
            def tiled_pipeline():
                full_tiles = (count - 1) // 2
                launch_tile(0, 0)

                @pl.loop(0, (full_tiles - 1) // 2)
                def tile_pairs(pair):
                    advance_tile(4 * pair + 2, 1)
                    advance_tile(4 * pair + 4, 0)

                def tail(slot):
                    @pl.when(count % 2 == 0)
                    def even():
                        advance_tile(2 * full_tiles, slot, has_next=False)
                        finish_tile(2 * full_tiles, slot)

                    @pl.when(count % 2 != 0)
                    def odd():
                        advance_tile(2 * full_tiles, slot, size=1, has_next=False)
                        finish_tile(2 * full_tiles, slot, size=1)

                @pl.when(full_tiles % 2 == 0)
                def even_full():
                    advance_tile(2 * (full_tiles - 1), 1)
                    tail(0)

                @pl.when(full_tiles % 2 != 0)
                def odd_full():
                    tail(1)

        else:

            def pv_launch(i, slot):
                if interpret:
                    return
                # Keep QK, CSA PV and SWA PV in disjoint MRB ranges. Each side
                # owns N=256, so H<=128 needs at most 6*32 of 256 addresses.
                csa_launch(i, (2 + slot) * mrb_stride, pv=True)
                for mxu in range(2):
                    pltpu.matmul_push_rhs(
                        swa_shared_ref[:, pl.ds(mxu * 256, 256)],
                        staging_register=0,
                        mxu_index=mxu,
                    )
                    lhs = jnp.pad(swa_p_ref[i], ((0, mxu_heads - num_heads), (0, 0)))
                    pltpu.matmul_acc_lhs(
                        (4 + slot) * mrb_stride, lhs, mxu_index=mxu, load_staged_rhs=0
                    )

            def finish_output(i, slot):
                main_output = csa_pop(i, (2 + slot) * mrb_stride, pv=True)
                if interpret:
                    swa_output = jnp.dot(
                        swa_p_ref[i],
                        swa_shared_ref[...],
                        preferred_element_type=jnp.float32,
                    )
                else:
                    swa_output = jnp.concatenate(
                        tuple(
                            pltpu.matmul_pop(
                                (4 + slot) * mrb_stride,
                                (mxu_heads, 256),
                                jnp.float32,
                                mxu_index=mxu,
                            )[:num_heads]
                            for mxu in range(2)
                        ),
                        axis=1,
                    )
                output = main_output + swa_output
                tail = output[:, 384:]
                partner = jnp.where(
                    (jnp.arange(128) % 2 == 0)[None, :],
                    pltpu.roll(tail, 127, axis=1),
                    pltpu.roll(tail, 1, axis=1),
                )
                rotated = (
                    tail * rope_ref[i, 0][None, :] + partner * rope_ref[i, 1][None, :]
                )
                output_ref[i, :, :384] = output[:, :384].astype(jnp.bfloat16)
                output_ref[i, :, 384:] = rotated.astype(jnp.bfloat16)
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[2 + 8 * bq + i],
                ).start()

            def advance(i, slot, *, has_next=True):
                softmax(i, slot * mrb_stride, has_next=has_next)
                pv_launch(i, slot)
                # The previous P@V remains in its own MRB while the current Q
                # runs softmax. Output DMA uses a distinct per-Q buffer.
                finish_output(i - 1, 1 - slot)

            @pl.when(count == 1)
            def single_query():
                softmax(0, 0, has_next=False)
                pv_launch(0, 0)
                finish_output(0, 0)

            @pl.when(count > 1)
            def query_pipeline():
                softmax(0, 0)
                pv_launch(0, 0)

                @pl.loop(0, (count - 2) // 2)
                def pairs(pair):
                    advance(2 * pair + 1, 1)
                    advance(2 * pair + 2, 0)

                @pl.when(count % 2 == 0)
                def even_tail():
                    advance(count - 1, 1, has_next=False)
                    finish_output(count - 1, 1)

                @pl.when(count % 2 != 0)
                def odd_tail():
                    advance(count - 2, 1)
                    advance(count - 1, 0, has_next=False)
                    finish_output(count - 1, 0)

        @pl.loop(0, count)
        def wait_output(i):
            pltpu.make_async_copy(
                output_ref.at[i], output_hbm_ref.at[t0 + i], dma_sem.at[2 + 8 * bq + i]
            ).wait()

        @pl.when((next_request != request_index) & (swa_prefetched_id[0] >= 0))
        def drain_request_prefetch():
            wait_block_dma(swa_prefetched_id[0])
            swa_prefetched_id[0] = -1

        return next_request, next_t0, 1 - bank

    lax.while_loop(
        lambda state: state[0] < request_limit,
        process_group,
        (first_request, first_t0, jnp.int32(0)),
    )

    def clear_padding():
        end = starts_ref[request_limit]
        output_ref[...] = jnp.zeros(output_ref.shape, jnp.bfloat16)

        @pl.loop(end, q_hbm_ref.shape[0], step=bq)
        def padding_group(t0):
            count = jnp.minimum(bq, q_hbm_ref.shape[0] - t0)

            @pl.loop(0, count)
            def start_zero(i):
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[2 + 8 * bq + i],
                ).start()

            @pl.loop(0, count)
            def wait_zero(i):
                pltpu.make_async_copy(
                    output_ref.at[i],
                    output_hbm_ref.at[t0 + i],
                    dma_sem.at[2 + 8 * bq + i],
                ).wait()

    clear_padding()


def _paged_main_kernel(starts, positions, lengths, metadata, *refs, **kwargs):
    main_kernel(starts, positions, lengths, *refs, swa_metadata_ref=metadata, **kwargs)


def _bounded_paged_main_kernel(
    starts, positions, lengths, metadata, bounds, *refs, **kwargs
):
    main_kernel(
        starts,
        positions,
        lengths,
        *refs,
        swa_metadata_ref=metadata,
        swa_bounds_ref=bounds,
        **kwargs,
    )


@functools.partial(
    jax.jit, static_argnames=("scale", "compress_ratio", "bq", "interpret")
)
def csa2(
    q: jax.Array,
    main_kv: jax.Array,
    main_kv_scale: jax.Array,
    swa_kv: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope_cache: jax.Array,
    *,
    scale: float = 512**-0.5,
    compress_ratio: int = 2,
    bq: int = 6,
    interpret: bool = False,
    main_lengths: jax.Array | None = None,
    swa_metadata: jax.Array | None = None,
    swa_bounds: jax.Array | None = None,
) -> jax.Array:
    """Fused causal SWA + CSA + inverse RoPE in request-local BQ groups.

    Args:
      q: (T, H, 512) BF16. H is divisible by 8, in [8, 128]. Q/KV
        are forward-rotated.
      main_kv: (T, 512, 4, 128) E4M3FN. The caller has compacted valid
        selections into the first min((positions[t]+1)//compress_ratio, 512)
        rows. Alternatively INT32 (T, 512, 128) stores the same byte planes:
        byte g of word lane holds FP8 feature g*128+lane. This accepts native
        word-row gathers directly, without an external FP8 layout conversion.
      main_kv_scale: (T, 512) FP32. One scale per complete NoPE+RoPE row.
      swa_kv: (T, 128, 512) BF16. Chronological, left-aligned windows:
        slot j is max(0, positions[t]-127)+j, with min(positions[t]+1, 128)
        valid rows. Overlapping rows must match within a request.
      starts: (B_max+1,) INT32, prefetched into SMEM. Nondecreasing request
        boundaries from zero to the valid Q count <= T. Repeats are empty
        requests.
      positions: (T,) INT32, prefetched into SMEM. Nonnegative and consecutive
        within each request.
      sink: (H,) FP32. Denominator-only logits.
      rope_cache: (T, 2, 128) FP32. Plane 0 contains cos, plane 1 signed sin.
        Lanes 0:64 are (1, 0); lanes 64:128 repeat cos for adjacent pairs,
        with sin signs (+, -), giving inverse rotation (x*c+y*s, y*c-x*s).
      main_lengths: Optional INT32 [T] compacted valid-prefix counts in [0,512].
        When supplied, these are authoritative, including shared-cache replay.
      swa_metadata: Optional INT32 [B_max, 2 + page_columns]. If supplied,
        swa_kv is BF16 [pages, page_size, 512], or INT32
        [pages, 2, page_size, 128]. In word plane p, low/high BF16 halves
        encode features p*256+lane and p*256+128+lane. Page 0 is finite zero.
        Metadata columns are (absolute first-page token position, minimum
        visible token position, physical page IDs...). Page IDs cover the
        visible SWA range, including queries in this call; unused columns
        refer to page 0. Page size is 8, 16, 32, 64 or a multiple of 128.
        Request entries
        may describe independent contiguous query segments of one request.
        A pair of banks uses global page IDs over their concatenation; only
        global page 0 is the sentinel. No concatenated cache is materialized.
        Each segment carries its own page metadata and visibility bound.
      swa_bounds: Optional INT32 [B_max,2] absolute [left,right) SWA interval.
        Requires paged input. Every query in a segment shares this interval;
        it replaces the causal window, while metadata's minimum visibility
        bound still applies. Caller guarantees 0 <= left <= right and
        right-left <= 192, covering DSpark's 128 history + short draft block.
        Gather metadata must include all these positions, including future KV.
      scale: Static attention logit scale.
      compress_ratio: Positive static integer compression ratio.
      bq: Static group capacity in [1, 16], or 32 for H=64. BQ=32 reuses
        four dequantized KV slots. Tails use only their valid Qs.
      interpret: Run the Pallas body on CPU without TPU code generation.

    Returns:
      (T, H, 512) BF16 output, already inverse-rotated, with padding zeroed.

    Main KV (including padding), visible SWA and FP32 intermediates must be
    finite; scales are nonnegative. The SC producer zeroes negative main IDs
    before reuse. Paged SWA outside the segment's visibility union is zeroed
    in VMEM before PV, so stale NaN bits in shared-owner padding cannot leak.
    The caller is responsible for metadata values and window ordering.
    """
    if q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("CSA2 q must have shape [T,H,512]")
    num_tokens, num_heads, _ = q.shape
    swa_input = swa_kv
    if isinstance(swa_kv, tuple):
        if len(swa_kv) != 2 or swa_metadata is None:
            raise ValueError("Banked SWA requires two paged cache banks")
        low, high = swa_kv
        if (
            low.shape[1:] != high.shape[1:]
            or low.dtype != high.dtype
            or high.shape[0] < 1
        ):
            raise ValueError("SWA banks require matching nonempty page layouts")
        swa_kv = low
    if num_heads < 8 or num_heads > 128 or num_heads % 8:
        raise NotImplementedError("CSA2 requires H divisible by 8, with 8 <= H <= 128")
    if type(bq) is not int or not (1 <= bq <= 16 or (bq == 32 and num_heads == 64)):
        raise ValueError(
            "CSA2 bq must be a static Python int in [1,16], or 32 for H=64"
        )
    if type(compress_ratio) is not int or compress_ratio < 1:
        raise ValueError("CSA2 compress_ratio must be a positive static Python int")
    if starts.ndim != 1 or starts.shape[0] < 2:
        raise ValueError("CSA2 starts must have shape [B_max+1], B_max >= 1")
    swa_shape = (num_tokens, 128, 512)
    swa_word_pages = swa_metadata is not None and swa_kv.dtype == jnp.int32
    if swa_metadata is not None:
        if (
            swa_metadata.ndim != 2
            or swa_metadata.shape[0] != starts.shape[0] - 1
            or swa_metadata.shape[1] < 3
            or swa_metadata.dtype != jnp.int32
        ):
            raise ValueError("SWA metadata must be INT32 [B_max, 2 + page_columns]")
        if swa_kv.ndim != (4 if swa_word_pages else 3):
            raise ValueError("Paged SWA has an invalid number of dimensions")
        page_size = swa_kv.shape[2 if swa_word_pages else 1]
        valid_shape = (
            swa_kv.ndim == 4 and swa_kv.shape[1] == 2 and swa_kv.shape[-1] == 128
            if swa_word_pages
            else swa_kv.ndim == 3 and swa_kv.shape[-1] == 512
        )
        if (
            not valid_shape
            or swa_kv.shape[0] < 1
            or page_size < 1
            or (page_size not in (8, 16, 32, 64) and page_size % 128)
        ):
            raise ValueError(
                "Paged SWA requires BF16 [pages,S,512] or INT32 [pages,2,S,128], "
                "S=8,16,32,64 or 128*n"
            )
        swa_shape = swa_kv.shape
    if swa_bounds is not None and (
        swa_metadata is None
        or swa_bounds.shape != (starts.shape[0] - 1, 2)
        or swa_bounds.dtype != jnp.int32
    ):
        raise ValueError("SWA bounds require paged inputs and INT32 [B_max,2]")
    main_dtype = jnp.int32 if main_kv.dtype == jnp.int32 else jnp.float8_e4m3fn
    main_shape = (
        (num_tokens, 512, 128) if main_dtype == jnp.int32 else (num_tokens, 512, 4, 128)
    )
    specs = (
        ("q", q, (num_tokens, num_heads, 512), jnp.bfloat16),
        ("main_kv", main_kv, main_shape, main_dtype),
        ("main_kv_scale", main_kv_scale, (num_tokens, 512), jnp.float32),
        ("swa_kv", swa_kv, swa_shape, jnp.int32 if swa_word_pages else jnp.bfloat16),
        ("starts", starts, starts.shape, jnp.int32),
        ("positions", positions, (num_tokens,), jnp.int32),
        ("sink", sink, (num_heads,), jnp.float32),
        ("rope_cache", rope_cache, (num_tokens, 2, 128), jnp.float32),
    )
    for name, value, shape, dtype in specs:
        if value.shape != shape or value.dtype != jnp.dtype(dtype):
            raise ValueError(f"CSA2 {name} must have shape {shape} and dtype {dtype}")
    if num_tokens == 0:
        return jnp.zeros(q.shape, q.dtype)
    if main_lengths is None:
        main_lengths = jnp.minimum((positions + 1) // compress_ratio, 512)
    elif main_lengths.shape != (num_tokens,) or main_lengths.dtype != jnp.int32:
        raise ValueError("Main lengths must be INT32 [T]")
    shared_window_size = 256
    # QK/PV consume KV i before dequant(i+4) reuses its slot; delayed PV
    # pops consume MRB only. Keep Ref reads so the compiler preserves WAR.
    # CPU interpret computes dots at pop time, so it needs all BQ entries.
    kv_slots = bq if interpret or bq <= 16 else 4
    scalars = (
        (starts, positions, main_lengths)
        if swa_metadata is None
        else (starts, positions, main_lengths, swa_metadata)
    )
    kernel = main_kernel if swa_metadata is None else _paged_main_kernel
    if swa_bounds is not None:
        scalars = (*scalars, swa_bounds)
        kernel = _bounded_paged_main_kernel
    return pl.pallas_call(
        functools.partial(
            kernel,
            scale=scale,
            compress_ratio=compress_ratio,
            bq=bq,
            interpret=interpret,
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3
            + [
                tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in swa_input)
                if isinstance(swa_input, tuple)
                else pl.BlockSpec(memory_space=pltpu.HBM)
            ]
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((num_heads,), jnp.float32),
                pltpu.VMEM((2, bq, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM((2, bq, *main_shape[1:]), main_dtype),
                pltpu.VMEM((2, bq, 1, 512), jnp.float32),
                pltpu.VMEM((kv_slots, 512, 512), jnp.bfloat16),
                pltpu.VMEM(
                    (2, 128, 128) if swa_word_pages else (128, 512),
                    jnp.int32 if swa_word_pages else jnp.bfloat16,
                ),
                pltpu.VMEM((shared_window_size, 512), jnp.bfloat16),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.SMEM((1,), jnp.int32),
                pltpu.VMEM((2, bq, 2, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, shared_window_size), jnp.float32),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, num_heads, shared_window_size), jnp.bfloat16),
                pltpu.VMEM((bq, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM(
                    (2, 2, 128, 128) if swa_word_pages else (2, 128, 512),
                    jnp.int32 if swa_word_pages else jnp.bfloat16,
                ),
                pltpu.SemaphoreType.DMA((4 + 9 * bq,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="csa2_swa_csa_inverse_rope",
    )(
        *scalars,
        q,
        main_kv,
        main_kv_scale[:, None, :],
        swa_input,
        sink,
        rope_cache,
    )
