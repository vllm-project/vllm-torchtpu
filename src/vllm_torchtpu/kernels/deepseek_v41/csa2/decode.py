# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode with SWA prefetch and two-MXU RHS lookahead.

Main visibility uses framework gather counts. Matrix operations use kernel-local
accumulator refs allocated with JAX 0.11.2.
Sixteen-row FP8 decode is shared by QK/PV; long sequences use fixed 8-Q tiles.
"""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _decode_kernel(
    # prefetch
    positions_ref: Any,  # SMEM: [total_seqs], int32
    swa_page_ref: Any,  # SMEM: [total_seqs, 2 + page_columns], int32
    main_lengths_ref: Any,  # SMEM: [total_tokens], int32; gathered valid counts
    swa_dma_ref: Any,  # SMEM: flattened [B, tiles, blocks, 2], page and aligned block
    # input (T = block_seq * num_tokens)
    q_ref: Any,  # VMEM: [T, H, 512], bfloat16
    main_cache_ref: Any,  # VMEM: [T, 512, 128], int32; packed FP8
    main_scale_ref: Any,  # VMEM: [T, 1, 512], float32
    swa_kv_ref: Any,  # HBM: one shared [pages,page_size,512] BF16 owner
    sink_ref: Any,  # VMEM: [H, 1], float32
    rope_ref: Any,  # VMEM: [T, 2, 128], float32
    # output
    output_ref: Any,  # VMEM: [T, H, 512], bfloat16
    # scratch
    main_kv_ref: Any,  # VMEM: [T, 512, 512], bfloat16;
    swa_buffer_ref: Any,  # VMEM: [2, block_seq, 256, 512], BF16 windows
    swa_sem: Any,  # DMA: [2, block_seq, num_pages]
    swa_score_ref: Any,  # VMEM: [T, H, 256], float32; scaled, masked SWA logits
    main_p_ref: Any,  # VMEM: [T, H, 512], bfloat16
    swa_p_ref: Any,  # VMEM: [T, H, 256], bfloat16
    *,
    num_tokens: int,
    block_seq: int,
    block_q: int,
    scale: float,
    tokens_per_seq: int | None = None,
    external_accumulators: Any = None,
    sequence_index: Any = None,
    current_swa: Any = None,
    previous_output: Any = None,
    previous_rope: Any = None,
    drain_only: bool = False,
    swa_rhs_preloaded: bool = False,
    preload_swa: Any = None,
) -> None:
    log2_e = math.log2(math.e)
    seq_start = (
        pl.program_id(0) * block_seq if sequence_index is None else sequence_index
    )
    query_offset = 0 if tokens_per_seq is None else pl.program_id(1) * num_tokens
    sequence_stride = num_tokens if tokens_per_seq is None else tokens_per_seq
    cache_page_size = swa_kv_ref.shape[1]
    page_size = min(cache_page_size, 64)
    num_pages = 256 // page_size
    num_heads = q_ref.shape[-2]
    mxu_heads = pl.cdiv(num_heads, 16) * 16
    mrb_stride = mxu_heads // 4
    # Separate QK/PV ranges let the next QK overlap the preceding PV.
    # Each range holds block_q results; together they must fit 256 MRB rows.

    swa_capacity = 256 // (block_seq * mrb_stride)
    accumulator_slots = max(2 * block_q, block_seq * min(num_tokens, swa_capacity))

    accumulators = (
        external_accumulators
        if external_accumulators is not None
        else tuple(
            tuple(
                jax.empty_ref(
                    jax.ShapeDtypeStruct((mxu_heads, 256), jnp.float32),
                    memory_space=pltpu.ACC(mxu),
                )
                for _ in range(accumulator_slots)
            )
            for mxu in range(2)
        )
    )

    def token_ref(ref, i):
        if tokens_per_seq is None:
            return ref.at[i]
        return ref.at[i // num_tokens, i % num_tokens]

    groups = []
    for seq in range(block_seq):
        # A shorter second-sequence group reduces the final PV drain.
        tile = block_q if seq == 0 else min(block_q, 4)
        for first in range(0, num_tokens, tile):
            groups.append(
                tuple(
                    range(
                        seq * num_tokens + first,
                        seq * num_tokens + min(first + tile, num_tokens),
                    )
                )
            )

    # Each group needs KV until its PV RHS is pushed. Adjacent groups occupy
    # separate banks; the preceding PV has consumed a bank before it is reused.
    kv_slots = {
        i: (group % 2) * block_q + r if tokens_per_seq is not None else i
        for group, indices in enumerate(groups)
        for r, i in enumerate(indices)
    }
    probability_slots = {
        i: r if tokens_per_seq is not None else i
        for indices in groups
        for r, i in enumerate(indices)
    }

    if current_swa is not None:
        swa_ref = current_swa
    else:
        tiles_per_seq = 1 if tokens_per_seq is None else tokens_per_seq // num_tokens
        tile = 0 if tokens_per_seq is None else pl.program_id(1)
        step = pl.program_id(0) * tiles_per_seq + tile
        slot = step % 2
        swa_ref = swa_buffer_ref.at[slot]

        def transfer_swa(seq_start, query_offset, slot, *, wait):
            for seq in range(block_seq):
                seq_idx = seq_start + seq
                tile = query_offset // num_tokens
                for block in range(num_pages):
                    descriptor = (
                        (seq_idx * tiles_per_seq + tile) * num_pages + block
                    ) * 2
                    page = swa_dma_ref[descriptor]
                    offset = swa_dma_ref[descriptor + 1] * page_size
                    copy = pltpu.make_async_copy(
                        swa_kv_ref.at[page, pl.ds(offset, page_size), :],
                        swa_buffer_ref.at[
                            slot, seq, pl.ds(block * page_size, page_size), :
                        ],
                        swa_sem.at[slot, seq, block],
                    )
                    if wait:
                        copy.wait()
                    else:
                        copy.start()

        @pl.when(step == 0)
        def prime_swa():
            transfer_swa(seq_start, query_offset, slot, wait=False)

        transfer_swa(seq_start, query_offset, slot, wait=True)

        @pl.when(step + 1 < (positions_ref.shape[0] // block_seq) * tiles_per_seq)
        def prefetch_swa():
            following = step + 1
            transfer_swa(
                following // tiles_per_seq * block_seq,
                following % tiles_per_seq * num_tokens,
                1 - slot,
                wait=False,
            )

    def swa_qk_compute(seq, first, size, slot_base):
        for mxu in range(2):
            ks = pl.ds(mxu * 256, 256)
            rhs = swa_ref[seq, :, ks]
            if not swa_rhs_preloaded:
                pltpu.matmul_push_rhs(rhs, 0, mxu, transpose=True)

            for r in range(size):
                lhs = jnp.pad(
                    token_ref(q_ref, seq * num_tokens + first + r)[:, ks],
                    ((0, mxu_heads - num_heads), (0, 0)),
                )
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][slot_base + r],
                    lhs,
                    load_staged_rhs=(1 if swa_rhs_preloaded else 0) if r == 0 else None,
                )

    def swa_qk_pop(seq, first, size, slot_base):
        position_start = positions_ref[seq_start + seq] + query_offset
        window_start = jnp.maximum(0, position_start - 127)
        page_start = window_start // page_size * page_size
        key_positions = page_start + jnp.arange(num_pages * page_size)
        visible_start = swa_page_ref[seq_start + seq, 1]

        for r in range(size):
            halves = [
                pltpu.matmul_pop(accumulators[mxu][slot_base + r])[:num_heads]
                for mxu in range(2)
            ]
            qk_scores = halves[0] + halves[1]

            position = position_start + first + r
            left = jnp.maximum(jnp.maximum(0, position - 127), visible_start)
            valid = (key_positions >= left) & (key_positions <= position)
            swa_score_ref[seq * num_tokens + first + r] = qk_scores * (
                scale * log2_e
            ) + jnp.where(valid[None, :], 0.0, -jnp.inf)

    def main_unpack(indices):
        # Carry-PV loads 64 rows and converts each 16-row slice once for QK/PV.
        unpack_rows = 64 if external_accumulators is not None else 16
        for i in indices:
            for row in range(0, 512, unpack_rows):
                words = token_ref(main_cache_ref, i)[pl.ds(row, unpack_rows), :]
                for subrow in range(0, unpack_rows, 16):
                    subwords = words[pl.ds(subrow, 16)]
                    for group in range(4):
                        shifted = lax.shift_right_logical(subwords, 8 * group)
                        bits = shifted.astype(jnp.uint8)
                        fp8 = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
                        values = fp8.astype(jnp.bfloat16)
                        main_kv_ref[
                            kv_slots[i],
                            pl.ds(row + subrow, 16),
                            pl.ds(group * 128, 128),
                        ] = values

    def main_qk_compute(indices):
        tiles = [(r, i, block) for r, i in enumerate(indices) for block in range(2)]

        def push(tile_index):
            r, i, block = tiles[tile_index]
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                ns = pl.ds(mxu * 256, 256)
                pltpu.matmul_push_rhs(
                    main_kv_ref[kv_slots[i], ns, ks],
                    tile_index % 2,
                    mxu,
                    transpose=True,
                )

        # Alternate two RHS staging slots across tile and token boundaries.
        push(0)
        for tile_index, (r, i, block) in enumerate(tiles):
            if tile_index + 1 < len(tiles):
                push(tile_index + 1)
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                lhs = jnp.pad(
                    token_ref(q_ref, i)[:, ks], ((0, mxu_heads - num_heads), (0, 0))
                )
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][r],
                    lhs,
                    load_staged_rhs=tile_index % 2,
                )

    def main_qk_pop(indices):
        logits = []
        for r, i in enumerate(indices):
            main_logits = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulators[mxu][r])[:num_heads]
                    for mxu in range(2)
                ],
                axis=1,
            )
            logits.append(main_logits)
        return logits

    def softmax(indices, qk_values):
        for r, i in enumerate(indices):
            main_logits = qk_values[r]
            kv_scales = token_ref(main_scale_ref, i)[...]
            main_logits = main_logits * (kv_scales * (scale * log2_e))

            sink = sink_ref[...] * log2_e
            if tokens_per_seq is None:
                main_count = main_lengths_ref[seq_start * num_tokens + i]
            else:
                main_count = main_lengths_ref[
                    (seq_start + i // num_tokens) * sequence_stride
                    + query_offset
                    + i % num_tokens
                ]
            main_logits = main_logits + jnp.where(
                jnp.arange(512)[None, :] < main_count, 0.0, -jnp.inf
            )
            swa_logits = swa_score_ref[i]
            logits = jnp.concatenate((main_logits, swa_logits), axis=1)
            maximum = jnp.maximum(sink, jnp.max(logits, axis=1, keepdims=True))
            exponentials = jnp.exp2(logits - maximum)
            sink_exponential = jnp.exp2(sink - maximum)
            denominator = sink_exponential + jnp.sum(
                exponentials, axis=1, keepdims=True
            )
            exponentials = exponentials / denominator
            # The denominator uses unscaled weights; fold KV scales into
            # main PV weights and round once to BF16.
            main_p_ref[probability_slots[i]] = (
                exponentials[:, :512] * kv_scales
            ).astype(jnp.bfloat16)
            swa_p_ref[probability_slots[i]] = exponentials[:, 512:].astype(jnp.bfloat16)

    def fused_pv(indices):
        tiles = [(r, i, block) for r, i in enumerate(indices) for block in range(2)]

        def push(tile_index):
            r, i, block = tiles[tile_index]
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                ns = pl.ds(mxu * 256, 256)
                pltpu.matmul_push_rhs(
                    main_kv_ref[kv_slots[i], ks, ns],
                    tile_index % 2,
                    mxu,
                    transpose=False,
                )

        # Alternate two RHS staging slots across tile and token boundaries.
        push(0)
        for tile_index, (r, i, block) in enumerate(tiles):
            if tile_index + 1 < len(tiles):
                push(tile_index + 1)
            ks = pl.ds(block * 256, 256)
            for mxu in range(2):
                lhs = jnp.pad(
                    main_p_ref[probability_slots[i], :, ks],
                    ((0, mxu_heads - num_heads), (0, 0)),
                )
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][block_q + r],
                    lhs,
                    load_staged_rhs=tile_index % 2,
                )

        # A group belongs to one sequence, so its Qs share the SWA RHS.
        seq = indices[0] // num_tokens
        for mxu in range(2):
            pltpu.matmul_push_rhs(
                swa_ref[seq, :, pl.ds(mxu * 256, 256)],
                0,
                mxu,
            )
            for r, i in enumerate(indices):
                lhs = jnp.pad(
                    swa_p_ref[probability_slots[i]],
                    ((0, mxu_heads - num_heads), (0, 0)),
                )
                pltpu.matmul_acc_lhs(
                    accumulators[mxu][block_q + r],
                    lhs,
                    load_staged_rhs=0 if r == 0 else None,
                )

                if r == len(indices) - 1 and preload_swa is not None:
                    preload_swa(mxu)

    def inverse_rope(indices, target=output_ref, rotations=rope_ref):
        for r, i in enumerate(indices):
            output = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulators[mxu][block_q + r])[:num_heads]
                    for mxu in range(2)
                ],
                axis=1,
            )
            tail = output[:, 384:]
            partner = jnp.take_along_axis(
                tail,
                jnp.broadcast_to((jnp.arange(128) ^ 1)[None, :], tail.shape),
                axis=1,
                mode="promise_in_bounds",
            )
            rotated = (
                tail * token_ref(rotations, i)[0][None, :]
                + partner * token_ref(rotations, i)[1][None, :]
            )
            token_ref(target, i)[:, :384] = output[:, :384].astype(jnp.bfloat16)
            token_ref(target, i)[:, 384:] = rotated.astype(jnp.bfloat16)

    if external_accumulators is not None:
        indices = tuple(range(num_tokens))
        if drain_only:
            inverse_rope(indices)
            return
        swa_qk_compute(0, 0, num_tokens, 0)
        main_unpack(indices)
        if previous_output is not None:
            inverse_rope(indices, previous_output, previous_rope)
        swa_qk_pop(0, 0, num_tokens, 0)
        main_qk_compute(indices)
        for i in indices:
            logits = jnp.concatenate(
                [
                    pltpu.matmul_pop(accumulators[mxu][i])[:num_heads]
                    for mxu in range(2)
                ],
                axis=1,
            )
            softmax((i,), (logits,))
        fused_pv(indices)
        return

    for first in range(0, num_tokens, swa_capacity):
        size = min(swa_capacity, num_tokens - first)
        for seq in range(block_seq):
            swa_qk_compute(seq, first, size, seq * size)
        if first == 0:
            main_unpack(groups[0])
        for seq in range(block_seq):
            swa_qk_pop(seq, first, size, seq * size)
    main_qk_compute(groups[0])
    for step, indices in enumerate(groups):
        logits = main_qk_pop(indices)
        if step + 1 < len(groups):
            main_unpack(groups[step + 1])
            main_qk_compute(groups[step + 1])
        softmax(indices, logits)
        if step:
            inverse_rope(groups[step - 1])
        fused_pv(indices)
    inverse_rope(groups[-1])


def _carry_decode(
    q, main_kv, scales, swa, sink, rope, positions, pages, lengths, *, scale
):
    bq = 6
    tokens, heads, _ = q.shape
    tiles = tokens // 6

    def kernel(
        positions_ref,
        pages_ref,
        lengths_ref,
        q_hbm,
        kv_hbm,
        scales_ref,
        swa_hbm,
        sink_ref,
        rope_ref,
        out_hbm,
        q_buf,
        kv_buf,
        swa_buf,
        outputs,
        decoded,
        swa_work,
        swa_scores,
        main_p,
        swa_p,
        input_sems,
        output_sems,
        swa_sems,
    ):
        accumulators = tuple(
            tuple(
                jax.empty_ref(
                    jax.ShapeDtypeStruct((heads, 256), jnp.float32),
                    memory_space=pltpu.ACC(mxu),
                )
                for _ in range(2 * bq)
            )
            for mxu in range(2)
        )

        def input_copies(tile, bank):
            rows = pl.ds(tile * 6, 6)
            return (
                pltpu.make_async_copy(
                    q_hbm.at[rows], q_buf.at[bank, pl.ds(0, 6)], input_sems.at[bank, 0]
                ),
                pltpu.make_async_copy(
                    kv_hbm.at[rows],
                    kv_buf.at[bank, pl.ds(0, 6)],
                    input_sems.at[bank, 1],
                ),
            )

        def start_input(tile, bank):
            for copy in input_copies(tile, bank):
                copy.start()

        def swa_copies(tile, bank):
            position = positions_ref[tile]
            window_start = jnp.maximum(0, position - 127)
            anchor = window_start // 64 * 64
            visible_start = jnp.maximum(window_start, pages_ref[tile, 1])
            copies = []
            for block in range(4):
                key = anchor + block * 64
                column = (key - pages_ref[tile, 0]) // 64
                safe = jnp.clip(column, 0, pages_ref.shape[1] - 3)
                physical = pages_ref[tile, safe + 2]
                valid = (
                    (key < position + 6)
                    & (key + 64 > visible_start)
                    & (column >= 0)
                    & (column < pages_ref.shape[1] - 2)
                    & (physical >= 0)
                    & (physical < swa_hbm.shape[0])
                )
                page = jnp.where(valid, physical, 0)
                copies.append(
                    pltpu.make_async_copy(
                        swa_hbm.at[page],
                        swa_buf.at[bank, block],
                        swa_sems.at[bank, block],
                    )
                )
            return copies

        def start_swa(tile, bank):
            for copy in swa_copies(tile, bank):
                copy.start()

        def wait_swa(tile, bank):
            for copy in swa_copies(tile, bank):
                copy.wait()

        def output_copy(tile, bank):
            return pltpu.make_async_copy(
                outputs.at[bank, pl.ds(0, 6)],
                out_hbm.at[pl.ds(tile * 6, 6)],
                output_sems.at[bank],
            )

        def compute(
            tile,
            bank,
            sb,
            *,
            previous=False,
            drain=False,
            prefetch=True,
            prefetch_swa=True,
            preload_next=True,
        ):
            def preload_next_swa(mxu):
                values = swa_buf[(sb + 1) % 4].reshape(256, 512)
                pltpu.matmul_push_rhs(
                    values[:, pl.ds(mxu * 256, 256)], 1, mxu, transpose=True
                )

            swa_work[...] = swa_buf[sb].reshape(1, 256, 512)
            _decode_kernel(
                positions_ref,
                pages_ref,
                lengths_ref,
                None,
                q_buf.at[sb],
                kv_buf.at[sb],
                scales_ref.at[pl.ds(tile * bq, bq)],
                swa_hbm,
                sink_ref,
                rope_ref.at[pl.ds(tile * bq, bq)],
                outputs.at[bank],
                decoded,
                None,
                None,
                swa_scores,
                main_p,
                swa_p,
                num_tokens=bq,
                block_seq=1,
                block_q=bq,
                scale=scale,
                external_accumulators=accumulators,
                sequence_index=tile,
                current_swa=swa_work,
                previous_output=outputs.at[1 - bank] if previous else None,
                previous_rope=rope_ref.at[pl.ds((tile - 1) * bq, bq)]
                if previous
                else None,
                drain_only=drain,
                swa_rhs_preloaded=previous,
                preload_swa=preload_next_swa if preload_next and not drain else None,
            )
            if previous:
                output_copy(tile - 1, 1 - bank).start()
            if prefetch and not drain:
                start_input(tile + 4, sb)
            if prefetch_swa and not drain:
                start_swa(tile + 4, sb)

        def step(
            tile,
            bank,
            sb,
            *,
            previous=True,
            wait_output=True,
            prefetch=True,
            prefetch_swa=True,
            preload_next=True,
        ):
            for copy in input_copies(tile, sb):
                copy.wait()
            if not previous:
                wait_swa(tile, sb)
            if preload_next:
                wait_swa(tile + 1, (sb + 1) % 4)
            if wait_output:
                output_copy(tile - 3, 1 - bank).wait()
            compute(
                tile,
                bank,
                sb,
                previous=previous,
                prefetch=prefetch,
                prefetch_swa=prefetch_swa,
                preload_next=preload_next,
            )

        for i in range(4):
            start_input(i, i)
        for i in range(4):
            start_swa(i, i)
        step(0, 0, 0, previous=False, wait_output=False)
        step(1, 1, 1, wait_output=False)
        step(2, 0, 2, wait_output=False)
        step(3, 1, 3)
        groups = (tiles - 8) // 4

        def body(group, unused):
            for j in range(4):
                step(4 + group * 4 + j, j % 2, (4 + j) % 4)
            return None

        lax.fori_loop(0, groups, body, None)
        for tile in range(4 + groups * 4, tiles):
            step(
                tile,
                tile % 2,
                tile % 4,
                prefetch=tile < tiles - 4,
                prefetch_swa=tile < tiles - 4,
                preload_next=tile < tiles - 1,
            )
        output_copy(tiles - 3, (tiles - 3) % 2).wait()
        compute(
            tiles - 1,
            (tiles - 1) % 2,
            (tiles - 1) % 4,
            drain=True,
            prefetch=False,
            prefetch_swa=False,
            preload_next=False,
        )
        output_copy(tiles - 1, (tiles - 1) % 2).start()
        output_copy(tiles - 2, (tiles - 2) % 2).wait()
        output_copy(tiles - 1, (tiles - 1) % 2).wait()

    spaces = (pltpu.HBM, pltpu.HBM, pltpu.VMEM, pltpu.HBM, pltpu.VMEM, pltpu.VMEM)
    args = (q, main_kv, scales[:, None, :], swa, sink[:, None], rope)
    args = tuple(pltpu.with_memory_space_constraint(x, s) for x, s in zip(args, spaces))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(),
            in_specs=tuple(pl.BlockSpec(memory_space=s) for s in spaces),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((4, bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((4, bq, 512, 128), jnp.int32),
                pltpu.VMEM((4, 4, 64, 512), jnp.bfloat16),
                pltpu.VMEM((2, bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, 512, 512), jnp.bfloat16),
                pltpu.VMEM((1, 256, 512), jnp.bfloat16),
                pltpu.VMEM((bq, heads, 256), jnp.float32),
                pltpu.VMEM((bq, heads, 512), jnp.bfloat16),
                pltpu.VMEM((bq, heads, 256), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((4, 2)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((4, 4)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, disable_semaphore_checks=True
        ),
        name="csa2_decode",
    )(positions, pages, lengths, *args)


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_tokens",
        "scale",
    ),
)
def csa2_decode(
    q: jax.Array,  # (T,H,512) bf16
    main_kv: jax.Array,  # (T,512,128) int32; packed FP8
    main_kv_scale: jax.Array,  # (T,512) f32
    swa_kv: jax.Array,  # (pages,page_size,512) bf16
    positions: jax.Array,  # (B,) int32; first Q position per sequence
    sink: jax.Array,  # (H,) f32
    rope_cache: jax.Array,  # (T,2,128) f32
    *,
    swa_page: jax.Array,  # [B, 2 + page_columns], int32
    num_tokens: int = 6,
    scale: float = 512**-0.5,
    main_lengths: jax.Array,
) -> jax.Array:
    """Causal CSA2 for consecutive, equal-length sequences.

    Q is BF16 [B*num_tokens,H,512]; positions [B] holds each first Q position.
    Long sequences use fixed eight-query tiles and must be divisible by eight.
    Short sequences use a single tile. Main KV is packed FP8 INT32 [T,512,128]
    with FP32 scales [T,512] and authoritative valid counts [T].
    SWA is finite BF16 [pages,page_size,512], including a reserved zero page.
    swa_page [B,2+C] stores page base, minimum visibility and physical page IDs.
    """
    if q.ndim != 3 or q.shape[2] != 512:
        raise ValueError("CSA2 q must have shape [T,H,512]")
    total_tokens, num_heads, _ = q.shape
    if type(num_tokens) is not int or num_tokens < 1:
        raise ValueError("CSA2 num_tokens must be a positive static Python int")
    if positions.ndim != 1 or positions.shape[0] * num_tokens != total_tokens:
        raise ValueError("CSA2 decode requires positions [B] and T = num_tokens * B")
    batch_size = total_tokens // num_tokens
    if num_heads < 8 or num_heads > 128 or num_heads % 8:
        raise NotImplementedError("CSA2 requires H divisible by 8, with 8 <= H <= 128")
    query_tile = min(num_tokens, 8)
    if num_tokens % query_tile:
        raise ValueError("Long CSA2 sequences must have a query count divisible by 8")
    tiled = num_tokens > 8
    block_q = min(6, query_tile, 256 // (2 * (pl.cdiv(num_heads, 16) * 4)))
    if (
        swa_page is None
        or swa_page.ndim != 2
        or swa_page.shape[0] != batch_size
        or swa_page.shape[1] < 3
        or swa_page.dtype != jnp.int32
    ):
        raise ValueError("SWA page table must be INT32 [B, 2 + page_columns]")
    if (
        not isinstance(swa_kv, jax.Array)
        or swa_kv.ndim != 3
        or swa_kv.dtype != jnp.bfloat16
        or swa_kv.shape[0] < 1
        or swa_kv.shape[2] != 512
    ):
        raise ValueError("SWA must be one BF16 array [pages,page_size,512]")
    page_size = swa_kv.shape[1]
    if page_size not in (8, 16, 32, 64) and (page_size < 128 or page_size % 128):
        raise ValueError(
            "SWA page size must be 8,16,32,64 or a positive multiple of 128"
        )
    specs = (
        ("q", q, (total_tokens, num_heads, 512), jnp.bfloat16),
        ("main_kv", main_kv, (total_tokens, 512, 128), jnp.int32),
        ("main_kv_scale", main_kv_scale, (total_tokens, 512), jnp.float32),
        ("positions", positions, (batch_size,), jnp.int32),
        ("main_lengths", main_lengths, (total_tokens,), jnp.int32),
        ("sink", sink, (num_heads,), jnp.float32),
        ("rope_cache", rope_cache, (total_tokens, 2, 128), jnp.float32),
    )
    for name, value, shape, dtype in specs:
        if value.shape != shape or value.dtype != jnp.dtype(dtype):
            raise ValueError(f"CSA2 {name} must have shape {shape} and dtype {dtype}")
    if total_tokens == 0:
        return jnp.zeros(q.shape, q.dtype)
    # The four-input carry schedule has four prologue and four drain tiles.
    # Larger metadata tables retain the bounded buffered-grid implementation.
    if (
        num_tokens == 6
        and num_heads == 64
        and page_size == 64
        and batch_size >= 8
        and main_kv_scale.nbytes + rope_cache.nbytes
        <= pltpu.get_tpu_info().vmem_capacity_bytes // 4
    ):
        return _carry_decode(
            q,
            main_kv,
            main_kv_scale,
            swa_kv,
            sink,
            rope_cache,
            positions,
            swa_page,
            main_lengths,
            scale=scale,
        )
    block_seq = 2 if batch_size % 2 == 0 else 1
    query_count = query_tile * block_seq

    def query_spec(trailing):
        if tiled:
            shape = (block_seq, query_tile, *trailing)

            def index_map(seq, tile, *_):
                return (
                    jnp.minimum(seq, batch_size // block_seq - 1),
                    jnp.minimum(tile, num_tokens // query_tile - 1),
                    *([0] * len(trailing)),
                )
        else:
            shape = (query_count, *trailing)

            def index_map(seq, *_):
                return jnp.minimum(seq, batch_size // block_seq - 1), *(
                    [0] * len(trailing)
                )

        return pl.BlockSpec(
            shape,
            index_map,
            pipeline_mode=pl.Buffered(2),
            memory_space=pltpu.VMEM,
        )

    kernel = functools.partial(
        _decode_kernel,
        num_tokens=query_tile,
        block_seq=block_seq,
        block_q=block_q,
        scale=scale,
        tokens_per_seq=num_tokens if tiled else None,
    )
    in_specs = (
        query_spec((num_heads, 512)),
        query_spec((512, 128)),
        query_spec((1, 512)),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(
            (num_heads, 1),
            lambda *_: (0, 0),
            pipeline_mode=pl.Buffered(1),
            memory_space=pltpu.VMEM,
        ),
        query_spec((2, 128)),
    )
    kv_count = 2 * block_q if tiled else query_count
    p_count = block_q if tiled else query_count
    grid = (
        (batch_size // block_seq, num_tokens // query_tile)
        if tiled
        else (batch_size // block_seq,)
    )
    prefix = (batch_size, num_tokens) if tiled else (total_tokens,)
    dma_block_size = min(page_size, 64)
    first = positions[:, None, None] + (
        jnp.arange(num_tokens // query_tile, dtype=jnp.int32)[None, :, None]
        * query_tile
    )
    left = jnp.maximum(0, first - 127)
    row = left // dma_block_size * dma_block_size + (
        jnp.arange(256 // dma_block_size, dtype=jnp.int32)[None, None, :]
        * dma_block_size
    )
    column = (row - swa_page[:, 0, None, None]) // page_size
    physical = jnp.take_along_axis(
        swa_page[:, None, 2:], jnp.clip(column, 0, swa_page.shape[1] - 3), axis=2
    )
    needed = (
        (row < first + query_tile)
        & (row + dma_block_size > jnp.maximum(left, swa_page[:, 1, None, None]))
        & (column >= 0)
        & (column < swa_page.shape[1] - 2)
        # A page outside the owner would DMA past it and halt the core.
        & (physical >= 0)
        & (physical < swa_kv.shape[0])
    )
    swa_dma = jnp.stack(
        (
            jnp.where(needed, physical, 0),
            jnp.where(needed, row % page_size // dma_block_size, 0),
        ),
        axis=-1,
    )

    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((*prefix, num_heads, 512), q.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=grid,
            in_specs=in_specs,
            out_specs=query_spec((num_heads, 512)),
            scratch_shapes=(
                pltpu.VMEM((kv_count, 512, 512), jnp.bfloat16),
                pltpu.VMEM((2, block_seq, 256, 512), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2, block_seq, 256 // min(page_size, 64))),
                pltpu.VMEM((query_count, num_heads, 256), jnp.float32),
                pltpu.VMEM((p_count, num_heads, 512), jnp.bfloat16),
                pltpu.VMEM((p_count, num_heads, 256), jnp.bfloat16),
            ),
        ),
        # SWA prefetch carries buffers across consecutive grid iterations.
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * len(grid)
        ),
        name="csa2_decode",
    )(
        positions,
        swa_page,
        main_lengths,
        # Flatten to avoid SMEM padding the small descriptor dimensions.
        swa_dma.reshape(-1),
        q.reshape(*prefix, num_heads, 512),
        main_kv.reshape(*prefix, 512, 128),
        main_kv_scale.reshape(*prefix, 1, 512),
        swa_kv,
        sink[:, None],
        rope_cache.reshape(*prefix, 2, 128),
    )
    return output.reshape(q.shape)
