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
"""TPU7x Pallas sub-row write, FP8+UE8M0 KV pack, and row gather kernels."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TPU7X_GENERATION = 7
WORD_BYTES = 4
BF16_BYTES = 2
BF16_PER_WORD = WORD_BYTES // BF16_BYTES
ROW_WORDS = 128
ROW_BYTES = ROW_WORDS * WORD_BYTES
MIN_SUBLANE_ROWS = 8
DOUBLE_BUFFER = 2

NOPE_DIM = 448
ROPE_DIM = 64
LATENT_DIM = NOPE_DIM + ROPE_DIM
INDEXER_HEAD_DIM = 128
ROPE_WORDS = ROPE_DIM // BF16_PER_WORD
INDEXER_KEY_WORDS = INDEXER_HEAD_DIM // BF16_PER_WORD
VALID_SLICE_WORDS = (ROPE_WORDS, INDEXER_KEY_WORDS, ROW_WORDS)

NOPE_SUBTILE_COUNT = NOPE_DIM // ROW_WORDS
NOPE_TAIL_CHANNELS = NOPE_DIM - NOPE_SUBTILE_COUNT * ROW_WORDS
NOPE_SCALE_GROUP_DIM = 64
FP8_MIN_AMAX = 1e-4
IEEE754_F32_EXP_BIAS = 127
IEEE754_F32_MANTISSA_BITS = 23
BYTE1_SHIFT = 8
BYTE2_SHIFT = 16
BYTE3_SHIFT = 24
ROPE_HIGH_HALF_SHIFT = 16

DEFAULT_PAGE_TOKENS = 16
MAX_WRITE_BLOCK = 128
MAX_PACK_TILE = 512
MAX_GATHER_BLOCK = 2048
GATHER_PAGE_BURST_ROWS = 32
GATHER_SUB_BURST_ROWS = 8
GATHER_SUB_CHUNKS_PER_PAGE = GATHER_PAGE_BURST_ROWS // GATHER_SUB_BURST_ROWS
# Avoid single-block serialization on multi-block gathers: cap block_rows >>= 3.
GATHER_BLOCK_DIVISOR_SHIFT = 3
# Avoid duplicate 8-row sub-bursts inside 32-row page bursts: mode 3 skips sub-chunks.
SUB_CHUNK_COVERED_BY_PAGE_MODE = 3

HBM = pltpu.MemorySpace.HBM
SMEM = pltpu.MemorySpace.SMEM


def _scope(enable_named_scope: bool, scope_name: str) -> AbstractContextManager[Any]:
    return jax.named_scope(scope_name) if enable_named_scope else nullcontext()


def _check_tpu7x() -> Any:
    tpu_info = pltpu.get_tpu_info()
    if tpu_info.generation != TPU7X_GENERATION or tpu_info.num_lanes != ROW_WORDS:
        raise RuntimeError(
            f"Requires TPU7x (generation={TPU7X_GENERATION}, "
            f"num_lanes={ROW_WORDS}), got generation={tpu_info.generation}, "
            f"num_lanes={tpu_info.num_lanes}."
        )
    return tpu_info


def _check_vmem_budget(required_vmem_bytes: int, tpu_info: Any, context: str) -> None:
    capacity_bytes = int(tpu_info.vmem_capacity_bytes)
    if required_vmem_bytes > capacity_bytes:
        raise ValueError(
            f"{context} requires {required_vmem_bytes} B VMEM, exceeding "
            f"capacity {capacity_bytes} B."
        )


def _dma_rows(
    predicate: jax.Array,
    src_ref: Any,
    src_row_offset: Any,
    dst_ref: Any,
    dst_row_offset: Any,
    num_rows: int,
    dma_sem: Any,
    is_wait_phase: bool,
) -> None:
    # Avoid tracing out-of-bounds pl.ds slices on small pools: Mosaic checks
    # static slice extents even inside @pl.when(False).
    if src_ref.shape[0] < num_rows or dst_ref.shape[0] < num_rows:
        return

    @pl.when(predicate)
    def _issue_or_wait() -> None:
        async_copy = pltpu.make_async_copy(
            src_ref.at[pl.ds(src_row_offset, num_rows)],
            dst_ref.at[pl.ds(dst_row_offset, num_rows)],
            dma_sem,
        )
        if is_wait_phase:
            async_copy.wait()
        else:
            async_copy.start()


def _pad_smem_meta(raw_meta_2d: jax.Array) -> tuple[jax.Array, int]:
    # Avoid raw_cols == padded_cols: reserve padded_meta_1d[-1] as an unused SMEM
    # padding slot for the optional after= ordering dependency.
    raw_cols = raw_meta_2d.shape[1]
    padded_cols = max(ROW_WORDS, 1 << raw_cols.bit_length())
    padded_meta_1d = jnp.pad(
        raw_meta_2d, ((0, 0), (0, padded_cols - raw_cols))
    ).reshape(-1)
    return padded_meta_1d, padded_cols


def _deduplicate_slots(
    slot_mapping: jax.Array, total_slots: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_tokens = slot_mapping.shape[0]
    in_range_mask = (slot_mapping >= 0) & (slot_mapping < total_slots)
    local_slots = jnp.where(in_range_mask, slot_mapping, jnp.int32(total_slots))
    # Avoid jnp.argsort + separate 1D array[order] gathers: jax.lax.sort permutes
    # token indices in-place without triggering SparseCore gather_offload ops.
    token_iota = jnp.arange(num_tokens, dtype=jnp.int32)
    sorted_slots, sorted_token_idx = jax.lax.sort(
        (local_slots, token_iota), dimension=0, is_stable=True
    )
    is_last_writer = (sorted_slots < total_slots) & jnp.pad(
        sorted_slots[:-1] != sorted_slots[1:], (0, 1), constant_values=True
    )
    valid_slots = jnp.where(is_last_writer, sorted_slots, jnp.int32(total_slots))
    deduped_slots, deduped_token_idx = jax.lax.sort(
        (valid_slots, sorted_token_idx), dimension=0, is_stable=True
    )
    return deduped_slots, deduped_token_idx, token_iota


def _pack_aligned_sub_row_tiles(
    updates: jax.Array,
    deduped_slots: jax.Array,
    deduped_token_idx: jax.Array,
    token_iota: jax.Array,
    *,
    num_rows: int,
    block_rows: int,
    pad_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array, pl.BlockSpec, jax.Array]:
    num_tokens, slice_words = updates.shape
    sub_slots_per_row = ROW_WORDS // slice_words
    total_slots = num_rows * sub_slots_per_row

    if sub_slots_per_row == 1:
        lead_rows = deduped_slots
        safe_token_idx = jnp.where(lead_rows < num_rows, deduped_token_idx, token_iota)
        aligned_updates = jnp.pad(updates[safe_token_idx], ((0, pad_tokens), (0, 0)))
        aligned_mask = jnp.ones((MIN_SUBLANE_ROWS, ROW_WORDS), dtype=jnp.int32)
        mask_block_spec = pl.BlockSpec(
            (MIN_SUBLANE_ROWS, ROW_WORDS), lambda block_idx: (0, 0)
        )
        all_sub_slots_present = jnp.pad(
            lead_rows < num_rows, (0, pad_tokens), constant_values=False
        )
        return (
            lead_rows,
            aligned_updates,
            aligned_mask,
            mask_block_spec,
            all_sub_slots_present,
        )

    # Avoid post-sort dynamic padded_slots[window_indices] gather and 4D
    # window_updates reduction: static 1D slice shifts before lax.sort and
    # token_iota fallback prevent VMEM gather loops and HBM bank conflicts.
    candidate_rows = deduped_slots // sub_slots_per_row
    is_row_leader = (candidate_rows < num_rows) & jnp.pad(
        candidate_rows[1:] != candidate_rows[:-1], (1, 0), constant_values=True
    )
    leader_candidate_rows = jnp.where(
        is_row_leader, candidate_rows, jnp.int32(num_rows)
    )
    # Avoid jnp.pad(..., (0, offset)): when num_tokens < offset, slice [offset:]
    # is empty and padding by offset overflows length num_tokens.
    shifted_slots = [
        deduped_slots
        if offset == 0
        else jnp.pad(
            deduped_slots[offset:],
            (0, min(offset, num_tokens)),
            constant_values=total_slots,
        )
        for offset in range(sub_slots_per_row)
    ]
    shifted_token_indices = [
        deduped_token_idx
        if offset == 0
        else jnp.pad(
            deduped_token_idx[offset:],
            (0, min(offset, num_tokens)),
            constant_values=num_tokens,
        )
        for offset in range(sub_slots_per_row)
    ]
    sub_slot_token_lists: list[jax.Array] = []
    for sub_slot_idx in range(sub_slots_per_row):
        target_slot_for_sub = candidate_rows * sub_slots_per_row + sub_slot_idx
        matched_token_idx = jnp.full((num_tokens,), num_tokens, dtype=jnp.int32)
        for offset in range(sub_slots_per_row):
            matched_token_idx = jnp.where(
                is_row_leader & (shifted_slots[offset] == target_slot_for_sub),
                shifted_token_indices[offset],
                matched_token_idx,
            )
        sub_slot_token_lists.append(matched_token_idx)

    sorted_leader_bundle = jax.lax.sort(
        (leader_candidate_rows, *sub_slot_token_lists),
        dimension=0,
        is_stable=True,
    )
    lead_rows = sorted_leader_bundle[0]
    sorted_sub_token_lists = sorted_leader_bundle[1:]
    has_sub_slot_list = [tok_idx < num_tokens for tok_idx in sorted_sub_token_lists]
    safe_sub_token_lists = [
        jnp.where(has_sub, tok_idx, token_iota)
        for has_sub, tok_idx in zip(
            has_sub_slot_list, sorted_sub_token_lists, strict=True
        )
    ]
    flat_safe_token_idx = jnp.stack(safe_sub_token_lists, axis=1).reshape(-1)
    merged_sub_slots = updates[flat_safe_token_idx].reshape(num_tokens, ROW_WORDS)
    aligned_updates = jnp.pad(merged_sub_slots, ((0, pad_tokens), (0, 0)))
    mask_slices = [
        jnp.broadcast_to(has_sub[:, None].astype(jnp.int32), (num_tokens, slice_words))
        for has_sub in has_sub_slot_list
    ]
    aligned_mask = jnp.pad(
        jnp.concatenate(mask_slices, axis=1), ((0, pad_tokens), (0, 0))
    )
    mask_block_spec = pl.BlockSpec(
        (block_rows, ROW_WORDS), lambda block_idx: (block_idx, 0)
    )
    all_present_unpadded = jnp.all(jnp.stack(has_sub_slot_list, axis=0), axis=0)
    all_sub_slots_present = jnp.pad(
        all_present_unpadded, (0, pad_tokens), constant_values=False
    )
    return (
        lead_rows,
        aligned_updates,
        aligned_mask,
        mask_block_spec,
        all_sub_slots_present,
    )


def _encode_write_smem_meta(
    lead_rows: jax.Array,
    all_sub_slots_present: jax.Array,
    after: jax.Array | None,
    *,
    num_rows: int,
    num_blocks: int,
    block_rows: int,
    burst_rows: int,
    pad_tokens: int,
) -> tuple[jax.Array, int]:
    num_burst_chunks = block_rows // burst_rows
    block_lead_rows = jnp.pad(
        lead_rows, (0, pad_tokens), constant_values=num_rows
    ).reshape(num_blocks, block_rows)
    block_all_present = all_sub_slots_present.reshape(num_blocks, block_rows)
    block_needs_read = ((block_lead_rows < num_rows) & (~block_all_present)).astype(
        jnp.int32
    )
    burst_chunk_rows = block_lead_rows.reshape(num_blocks, num_burst_chunks, burst_rows)
    burst_start_rows = burst_chunk_rows[:, :, :1]
    is_burst_contiguous = jnp.all(
        block_all_present.reshape(num_blocks, num_burst_chunks, burst_rows),
        axis=2,
    ) & jnp.all(
        burst_chunk_rows
        == burst_start_rows + jnp.arange(burst_rows, dtype=jnp.int32)[None, None, :],
        axis=2,
    )
    burst_chunk_modes = jnp.where(
        is_burst_contiguous,
        2,
        (burst_start_rows[:, :, 0] < num_rows).astype(jnp.int32),
    )
    is_block_all_burst = jnp.all(burst_chunk_modes == 2, axis=1)
    block_modes = jnp.where(
        is_block_all_burst,
        2,
        (block_lead_rows[:, 0] < num_rows).astype(jnp.int32),
    )
    raw_meta = jnp.concatenate(
        [
            block_modes[:, None],
            burst_chunk_modes,
            block_lead_rows,
            block_needs_read,
        ],
        axis=1,
    )
    padded_meta, meta_cols = _pad_smem_meta(raw_meta)
    # Avoid deleting this unread trailing-word write: without a data dependency edge
    # into the opaque pallas_call operand, XLA treats a prior read and this donated
    # write as unordered and keeps a full-pool HBM copy instead of mutating in-place.
    if after is not None and after.size > 0:
        first_after_elem = after.reshape(-1)[0]
        dependency_word = (
            first_after_elem
            if first_after_elem.dtype == jnp.int32
            else jax.lax.bitcast_convert_type(
                first_after_elem.astype(jnp.float32), jnp.int32
            )
        )
        padded_meta = padded_meta.at[-1].set(dependency_word)
    return padded_meta, meta_cols


def _write_sub_rows_kernel(
    meta_ref: Any,
    mask_ref: Any,
    cache_in_ref: Any,
    updates_ref: Any,
    cache_out_ref: Any,
    row_scratch_ref: Any,
    dma_sem: Any,
    *,
    num_rows: int,
    sub_slots_per_row: int,
    block_rows: int,
    burst_rows: int,
    enable_named_scope: bool,
) -> None:
    num_burst_chunks = block_rows // burst_rows
    rows_smem_offset = 1 + num_burst_chunks
    needs_read_smem_offset = rows_smem_offset + block_rows

    @pl.when(meta_ref[0] > 0)
    def _execute_block() -> None:
        if sub_slots_per_row > 1:

            @pl.when(meta_ref[0] == 1)
            def _read_partial_rows() -> None:
                with _scope(enable_named_scope, "dma_read_partial"):
                    for is_wait_phase in (False, True):

                        def _read_row_step(
                            row_in_block: jax.Array,
                            _: None,
                            *,
                            is_wait_phase: bool = is_wait_phase,
                        ) -> None:
                            target_row = meta_ref[rows_smem_offset + row_in_block]
                            needs_read = (
                                meta_ref[needs_read_smem_offset + row_in_block] != 0
                            )
                            _dma_rows(
                                needs_read,
                                cache_in_ref,
                                target_row,
                                row_scratch_ref,
                                row_in_block,
                                1,
                                dma_sem,
                                is_wait_phase,
                            )

                        jax.lax.fori_loop(0, block_rows, _read_row_step, None)

        with _scope(enable_named_scope, "merge_sub_rows"):
            if sub_slots_per_row == 1:
                row_scratch_ref[...] = updates_ref[...]
            else:
                row_scratch_ref[...] = jnp.where(
                    mask_ref[...] != 0,
                    updates_ref[...],
                    row_scratch_ref[...],
                )

        with _scope(enable_named_scope, "dma_write_rows"):
            for is_wait_phase in (False, True):

                def _write_burst_chunk(
                    chunk_idx: jax.Array,
                    _: None,
                    *,
                    is_wait_phase: bool = is_wait_phase,
                ) -> None:
                    chunk_mode = meta_ref[1 + chunk_idx]
                    chunk_row_offset = chunk_idx * burst_rows
                    chunk_start_row = meta_ref[rows_smem_offset + chunk_row_offset]
                    _dma_rows(
                        chunk_mode == 2,
                        row_scratch_ref,
                        chunk_row_offset,
                        cache_out_ref,
                        chunk_start_row,
                        burst_rows,
                        dma_sem,
                        is_wait_phase,
                    )

                    @pl.when(chunk_mode == 1)
                    def _write_scattered_rows(
                        *, is_wait_phase: bool = is_wait_phase
                    ) -> None:
                        def _write_row_step(
                            offset_in_chunk: jax.Array, _: None
                        ) -> None:
                            row_in_block = chunk_row_offset + offset_in_chunk
                            target_row = meta_ref[rows_smem_offset + row_in_block]
                            _dma_rows(
                                target_row < num_rows,
                                row_scratch_ref,
                                row_in_block,
                                cache_out_ref,
                                target_row,
                                1,
                                dma_sem,
                                is_wait_phase,
                            )

                        jax.lax.fori_loop(0, burst_rows, _write_row_step, None)

                jax.lax.fori_loop(0, num_burst_chunks, _write_burst_chunk, None)


def write_sub_rows(
    cache_words: jax.Array,
    updates: jax.Array,
    slot_mapping: jax.Array,
    *,
    after: jax.Array | None = None,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Scatter ``int32[T, 32|64|128]`` updates into donated ``int32[R, 128]`` rows.

    Ignores slots outside ``[0, R * (128 // W))`` and resolves duplicate slots
    last-writer-wins. Pass a prior read's output as ``after`` to enforce XLA
    read-before-write ordering and prevent full-pool HBM clones.
    """
    if cache_words.shape[1:] != (ROW_WORDS,) or cache_words.dtype != jnp.int32:
        raise ValueError("cache_words must be int32[num_rows, 128].")
    is_bad_updates = updates.ndim != 2 or updates.shape[1] not in VALID_SLICE_WORDS
    if is_bad_updates or updates.dtype != jnp.int32:
        raise ValueError("updates must be int32[num_tokens, 32|64|128].")
    if slot_mapping.shape != (updates.shape[0],) or slot_mapping.dtype != jnp.int32:
        raise ValueError("slot_mapping must be int32[num_tokens].")
    if page_tokens <= 0 or (page_tokens & (page_tokens - 1)) != 0:
        raise ValueError(
            f"page_tokens must be a positive power of two; got {page_tokens}."
        )
    tpu_info = _check_tpu7x()
    num_rows = cache_words.shape[0]
    num_tokens, slice_words = updates.shape
    sub_slots_per_row = ROW_WORDS // slice_words
    if num_rows == 0 or num_tokens == 0:
        return cache_words

    raw_burst_rows = max(1, page_tokens // sub_slots_per_row)
    block_rows = min(
        MAX_WRITE_BLOCK,
        max(
            DEFAULT_PAGE_TOKENS,
            raw_burst_rows,
            1 << (num_tokens - 1).bit_length(),
        ),
    )
    burst_rows = min(block_rows, raw_burst_rows)
    mask_rows = MIN_SUBLANE_ROWS if sub_slots_per_row == 1 else block_rows
    required_vmem_bytes = (
        DOUBLE_BUFFER * (mask_rows + block_rows) + block_rows
    ) * ROW_BYTES
    _check_vmem_budget(required_vmem_bytes, tpu_info, "write_sub_rows")

    num_blocks = (num_tokens + block_rows - 1) // block_rows
    pad_tokens = num_blocks * block_rows - num_tokens

    deduped_slots, deduped_token_idx, token_iota = _deduplicate_slots(
        slot_mapping, num_rows * sub_slots_per_row
    )
    (
        lead_rows,
        aligned_updates,
        aligned_mask,
        mask_block_spec,
        all_sub_slots_present,
    ) = _pack_aligned_sub_row_tiles(
        updates,
        deduped_slots,
        deduped_token_idx,
        token_iota,
        num_rows=num_rows,
        block_rows=block_rows,
        pad_tokens=pad_tokens,
    )
    padded_meta, meta_cols = _encode_write_smem_meta(
        lead_rows,
        all_sub_slots_present,
        after,
        num_rows=num_rows,
        num_blocks=num_blocks,
        block_rows=block_rows,
        burst_rows=burst_rows,
        pad_tokens=pad_tokens,
    )

    hbm_block_spec = pl.BlockSpec(memory_space=HBM)
    in_specs = [
        pl.BlockSpec((meta_cols,), lambda block_idx: (block_idx,), memory_space=SMEM),
        mask_block_spec,
        hbm_block_spec,
        pl.BlockSpec((block_rows, ROW_WORDS), lambda block_idx: (block_idx, 0)),
    ]
    scratch_shapes = [
        pltpu.VMEM((block_rows, ROW_WORDS), jnp.int32),
        pltpu.SemaphoreType.DMA,
    ]
    kernel_body = partial(
        _write_sub_rows_kernel,
        num_rows=num_rows,
        sub_slots_per_row=sub_slots_per_row,
        block_rows=block_rows,
        burst_rows=burst_rows,
        enable_named_scope=enable_named_scope,
    )
    return pl.pallas_call(
        kernel_body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            0, (num_blocks,), in_specs, hbm_block_spec, scratch_shapes
        ),
        out_shape=cache_words,
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)),
        interpret=interpret,
    )(padded_meta, aligned_mask, cache_words, aligned_updates)


def _quantize_nope_fp8_words(latent_f32: jax.Array) -> jax.Array:
    fp8_max_val = float(jnp.finfo(jnp.float8_e4m3fn).max)
    subtile3_valid_mask = (jnp.arange(ROW_WORDS, dtype=jnp.int32) < NOPE_TAIL_CHANNELS)[
        None, :
    ]
    subtile0 = latent_f32[:, :ROW_WORDS]
    subtile1 = latent_f32[:, ROW_WORDS : 2 * ROW_WORDS]
    subtile2 = latent_f32[:, 2 * ROW_WORDS : 3 * ROW_WORDS]
    subtile3 = latent_f32[:, 3 * ROW_WORDS : LATENT_DIM]
    group_amax = jnp.maximum(
        jnp.maximum(jnp.abs(subtile0), jnp.abs(subtile1)),
        jnp.maximum(
            jnp.abs(subtile2),
            jnp.where(subtile3_valid_mask, jnp.abs(subtile3), 0.0),
        ),
    )
    group_amax = jnp.clip(
        jnp.maximum(group_amax, pltpu.roll(group_amax, NOPE_SCALE_GROUP_DIM, 1)),
        FP8_MIN_AMAX,
        None,
    )
    exponent_i32 = jnp.ceil(jnp.log2(group_amax / fp8_max_val)).astype(jnp.int32)
    inv_scale_f32 = jax.lax.bitcast_convert_type(
        (IEEE754_F32_EXP_BIAS - exponent_i32) << IEEE754_F32_MANTISSA_BITS,
        jnp.float32,
    )

    def _quantize_subtile_u32(subtile_f32: jax.Array) -> jax.Array:
        quantized_fp8 = (subtile_f32 * inv_scale_f32).astype(jnp.float8_e4m3fn)
        return jax.lax.bitcast_convert_type(quantized_fp8, jnp.uint8).astype(jnp.uint32)

    subtile3_or_scale_u32 = jnp.where(
        subtile3_valid_mask,
        _quantize_subtile_u32(subtile3),
        (exponent_i32 + IEEE754_F32_EXP_BIAS).astype(jnp.uint32),
    )
    packed_words_u32 = (
        _quantize_subtile_u32(subtile0)
        | (_quantize_subtile_u32(subtile1) << BYTE1_SHIFT)
        | (_quantize_subtile_u32(subtile2) << BYTE2_SHIFT)
        | (subtile3_or_scale_u32 << BYTE3_SHIFT)
    )
    return jax.lax.bitcast_convert_type(packed_words_u32, jnp.int32)


def _pack_rope_bf16_words(latent_bf16: jax.Array) -> jax.Array:
    # Avoid slicing latent_bf16[:, 448:512] directly: Mosaic 2D VMEM slices require
    # 128-lane-aligned column offsets, so slice [:, 384:512] and roll lanes [64:128].
    tail_subtile_u16 = jax.lax.bitcast_convert_type(
        latent_bf16[:, 3 * ROW_WORDS : LATENT_DIM], jnp.uint16
    )
    tail_subtile_u32 = tail_subtile_u16.astype(jnp.uint32)
    packed_rope_u32 = pltpu.roll(tail_subtile_u32, NOPE_SCALE_GROUP_DIM, 1) | (
        pltpu.roll(tail_subtile_u32, ROPE_WORDS, 1) << ROPE_HIGH_HALF_SHIFT
    )
    return jax.lax.bitcast_convert_type(packed_rope_u32[:, :ROPE_WORDS], jnp.int32)


def _pack_fp8_tile(
    latent_ref: Any,
    nope_words_ref: Any,
    rope_words_ref: Any,
    *,
    enable_named_scope: bool,
) -> None:
    latent_tile_bf16 = latent_ref[...]
    with _scope(enable_named_scope, "quantize_nope_fp8"):
        nope_words_ref[...] = _quantize_nope_fp8_words(
            latent_tile_bf16.astype(jnp.float32)
        )
    with _scope(enable_named_scope, "pack_rope_bf16"):
        rope_words_ref[...] = _pack_rope_bf16_words(latent_tile_bf16)


def _pack_fp8_pipeline(
    latent_hbm_ref: Any,
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    *,
    tile_tokens: int,
    num_tiles: int,
    enable_named_scope: bool,
) -> None:
    pltpu.emit_pipeline(
        partial(_pack_fp8_tile, enable_named_scope=enable_named_scope),
        grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((tile_tokens, LATENT_DIM), lambda tile_idx: (tile_idx, 0))
        ],
        out_specs=[
            pl.BlockSpec((tile_tokens, ROW_WORDS), lambda tile_idx: (tile_idx, 0)),
            pl.BlockSpec((tile_tokens, ROPE_WORDS), lambda tile_idx: (tile_idx, 0)),
        ],
        dimension_semantics=(pltpu.PARALLEL,),
        trace_scopes=enable_named_scope,
    )(latent_hbm_ref, nope_hbm_ref, rope_hbm_ref)


def pack_fp8_kv_rows(
    latent: jax.Array,
    *,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Pack ``bf16[T, 512]`` latents into ``(int32[T, 128], int32[T, 32])`` words.

    Bytes ``0..2`` of lanes ``0..127`` and byte ``3`` of lanes ``0..63`` hold 448 FP8
    E4M3FN channels; byte ``3`` of lanes ``64..127`` holds 64 UE8M0 scales (lane
    ``64 + j`` scales ``{j + 64k : k = 0..6}``); ``int32[T, 32]`` holds 64 BF16 RoPE.
    """
    if latent.shape[1:] != (LATENT_DIM,) or latent.dtype != jnp.bfloat16:
        raise ValueError("latent must be bfloat16[num_tokens, 512].")
    tpu_info = _check_tpu7x()
    num_tokens = latent.shape[0]
    if num_tokens == 0:
        empty_nope = jnp.zeros((0, ROW_WORDS), jnp.int32)
        return empty_nope, jnp.zeros((0, ROPE_WORDS), jnp.int32)

    aligned_tokens = max(
        MIN_SUBLANE_ROWS,
        ((num_tokens + MIN_SUBLANE_ROWS - 1) // MIN_SUBLANE_ROWS) * MIN_SUBLANE_ROWS,
    )
    num_tiles = (aligned_tokens + MAX_PACK_TILE - 1) // MAX_PACK_TILE
    tile_tokens = (
        ((aligned_tokens + num_tiles - 1) // num_tiles + MIN_SUBLANE_ROWS - 1)
        // MIN_SUBLANE_ROWS
    ) * MIN_SUBLANE_ROWS
    required_vmem_bytes = (
        DOUBLE_BUFFER
        * tile_tokens
        * (LATENT_DIM * BF16_BYTES + (ROW_WORDS + ROPE_WORDS) * WORD_BYTES)
    )
    _check_vmem_budget(required_vmem_bytes, tpu_info, "pack_fp8_kv_rows")

    padded_tokens = num_tiles * tile_tokens
    padded_latent = jnp.pad(latent, ((0, padded_tokens - num_tokens), (0, 0)))

    hbm_block_spec = pl.BlockSpec(memory_space=HBM)
    out_shapes = [
        jax.ShapeDtypeStruct((padded_tokens, ROW_WORDS), jnp.int32),
        jax.ShapeDtypeStruct((padded_tokens, ROPE_WORDS), jnp.int32),
    ]
    pipeline_body = partial(
        _pack_fp8_pipeline,
        tile_tokens=tile_tokens,
        num_tiles=num_tiles,
        enable_named_scope=enable_named_scope,
    )
    nope_words, rope_words = pl.pallas_call(
        pipeline_body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            0, (), [hbm_block_spec], [hbm_block_spec, hbm_block_spec]
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(dimension_semantics=()),
        interpret=interpret,
    )(padded_latent)
    return nope_words[:num_tokens], rope_words[:num_tokens]


def _contig_mode(chunk_indices_3d: jax.Array, num_rows: int) -> jax.Array:
    chunk_width = chunk_indices_3d.shape[2]
    start_indices = chunk_indices_3d[:, :, :1]
    is_contiguous = jnp.all(
        chunk_indices_3d == start_indices + jnp.arange(chunk_width, dtype=jnp.int32),
        axis=2,
    )
    chunk_start_row = start_indices[:, :, 0]
    in_pool = (
        is_contiguous
        & (chunk_start_row >= 0)
        & (chunk_start_row + chunk_width <= num_rows)
    )
    return in_pool.astype(jnp.int32)


def _gather_cache_rows_kernel(
    meta_ref: Any,
    pool_ref: Any,
    out_ref: Any,
    dma_sem: Any,
    *,
    num_rows: int,
    block_rows: int,
    enable_named_scope: bool,
) -> None:
    num_page_chunks = block_rows // GATHER_PAGE_BURST_ROWS
    num_sub_chunks = block_rows // GATHER_SUB_BURST_ROWS
    indices_smem_offset = 1 + num_page_chunks + num_sub_chunks

    def _execute_dma_phase(is_wait_phase: bool) -> None:
        def _gather_page_chunk(page_chunk_idx: jax.Array, _: None) -> None:
            page_mode = meta_ref[1 + page_chunk_idx]
            dst_offset = page_chunk_idx * GATHER_PAGE_BURST_ROWS
            _dma_rows(
                page_mode == 1,
                pool_ref,
                meta_ref[indices_smem_offset + dst_offset],
                out_ref,
                dst_offset,
                GATHER_PAGE_BURST_ROWS,
                dma_sem,
                is_wait_phase,
            )

        jax.lax.fori_loop(0, num_page_chunks, _gather_page_chunk, None)

        @pl.when(meta_ref[0] != 0)
        def _gather_sub_and_scattered() -> None:
            def _gather_sub_chunk(sub_chunk_idx: jax.Array, _: None) -> None:
                sub_mode = meta_ref[1 + num_page_chunks + sub_chunk_idx]
                sub_offset = sub_chunk_idx * GATHER_SUB_BURST_ROWS
                _dma_rows(
                    sub_mode == 1,
                    pool_ref,
                    meta_ref[indices_smem_offset + sub_offset],
                    out_ref,
                    sub_offset,
                    GATHER_SUB_BURST_ROWS,
                    dma_sem,
                    is_wait_phase,
                )

                @pl.when(sub_mode == 0)
                def _gather_scattered_rows() -> None:
                    def _gather_single_row(local_offset: jax.Array, _: None) -> None:
                        query_in_block = sub_offset + local_offset
                        row_idx = meta_ref[indices_smem_offset + query_in_block]
                        in_pool = (row_idx >= 0) & (row_idx < num_rows)
                        _dma_rows(
                            in_pool,
                            pool_ref,
                            row_idx,
                            out_ref,
                            query_in_block,
                            1,
                            dma_sem,
                            is_wait_phase,
                        )

                    jax.lax.fori_loop(
                        0, GATHER_SUB_BURST_ROWS, _gather_single_row, None
                    )

            jax.lax.fori_loop(0, num_sub_chunks, _gather_sub_chunk, None)

    out_ref[...] = jnp.zeros_like(out_ref)
    for is_wait_phase, scope_name in (
        (False, "dma_issue"),
        (True, "dma_wait"),
    ):
        with _scope(enable_named_scope, scope_name):
            _execute_dma_phase(is_wait_phase)


def gather_cache_rows(
    pool: jax.Array,
    row_indices: jax.Array,
    *,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Gather ``int32[Q, 128]`` rows from ``int32[R, 128]`` via 32/8/1-row bursts.

    Fetches indices in ``[0, R)`` asynchronously and zero-fills out-of-bounds rows.
    """
    if pool.shape[1:] != (ROW_WORDS,) or pool.dtype != jnp.int32:
        raise ValueError("pool must be int32[num_rows, 128].")
    if row_indices.ndim != 1 or row_indices.dtype != jnp.int32:
        raise ValueError("row_indices must be a 1D int32 array.")
    tpu_info = _check_tpu7x()

    num_queries = row_indices.shape[0]
    num_rows = pool.shape[0]
    if num_queries == 0 or num_rows == 0:
        return jnp.zeros((num_queries, ROW_WORDS), dtype=jnp.int32)

    block_rows = min(
        MAX_GATHER_BLOCK,
        max(
            GATHER_PAGE_BURST_ROWS,
            1 << max(0, num_queries.bit_length() - GATHER_BLOCK_DIVISOR_SHIFT),
        ),
    )
    _check_vmem_budget(
        DOUBLE_BUFFER * block_rows * ROW_BYTES, tpu_info, "gather_cache_rows"
    )
    num_page_chunks = block_rows // GATHER_PAGE_BURST_ROWS
    num_sub_chunks = block_rows // GATHER_SUB_BURST_ROWS
    num_blocks = (num_queries + block_rows - 1) // block_rows
    padded_queries = num_blocks * block_rows
    padded_indices = jnp.pad(
        row_indices, (0, padded_queries - num_queries), constant_values=-1
    )
    page_chunk_modes = _contig_mode(
        padded_indices.reshape(num_blocks, num_page_chunks, GATHER_PAGE_BURST_ROWS),
        num_rows,
    )
    sub_chunk_modes_raw = _contig_mode(
        padded_indices.reshape(num_blocks, num_sub_chunks, GATHER_SUB_BURST_ROWS),
        num_rows,
    )
    sub_chunk_modes = jnp.where(
        jnp.repeat(page_chunk_modes > 0, GATHER_SUB_CHUNKS_PER_PAGE, axis=1),
        SUB_CHUNK_COVERED_BY_PAGE_MODE,
        sub_chunk_modes_raw,
    )
    has_non_page_chunks = jnp.any(page_chunk_modes == 0, axis=1, keepdims=True).astype(
        jnp.int32
    )
    padded_meta, meta_cols = _pad_smem_meta(
        jnp.concatenate(
            [
                has_non_page_chunks,
                page_chunk_modes,
                sub_chunk_modes,
                padded_indices.reshape(num_blocks, block_rows),
            ],
            axis=1,
        )
    )

    in_specs = [
        pl.BlockSpec((meta_cols,), lambda block_idx: (block_idx,), memory_space=SMEM),
        pl.BlockSpec(memory_space=HBM),
    ]
    out_spec = pl.BlockSpec((block_rows, ROW_WORDS), lambda block_idx: (block_idx, 0))
    kernel_body = partial(
        _gather_cache_rows_kernel,
        num_rows=num_rows,
        block_rows=block_rows,
        enable_named_scope=enable_named_scope,
    )
    gathered_rows = pl.pallas_call(
        kernel_body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            0, (num_blocks,), in_specs, out_spec, [pltpu.SemaphoreType.DMA]
        ),
        out_shape=jax.ShapeDtypeStruct((padded_queries, ROW_WORDS), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.PARALLEL,)),
        interpret=interpret,
    )(padded_meta, pool)
    return gathered_rows[:num_queries]
