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
"""Pallas sub-row cache write, FP8+UE8M0 KV pack, and 2-bank row gather kernels.

Architecture & 3-Layer Decomposition (TPU7x Ironwood):
1. ``write_sub_rows`` (Sub-Row Cache Mutation):
   - Host Sanitization & Last-Writer-Wins Deduplication: Maps global ``slot_mapping``
     to bank-local sub-slots ``local_slot = slot_mapping - slot_offset`` for
     2-bank segmented HBM pools (working around TPU7x SparseCore's 16 GiB single-
     tensor HBM addressing limit), masks out-of-range sub-slots, and runs two stable
     ``jnp.argsort(..., stable=True)`` passes so duplicate writes to the same sub-
     slot resolve deterministically via last-writer-wins before grouping
     ``sub_slots_per_row = 128 // slice_words`` sub-slots per 512-byte row.
   - Layer 2/3 Burst-Coalesced Async DMA: Stages updates through a ``(block_rows,
     128)`` VMEM buffer via ``CMNDE0/1`` (3,207 GB/s peak) instead of ``HBM -> HBM``
     copies on ``HDE0/1`` (32 GB/s ceiling). Optional ``after`` routes a scalar
     dependency into the unused trailing SMEM padding slot ``padded_meta[-1]`` of
     Operand 0 so XLA orders prior ``gather_cache_rows`` reads before donated in-
     place ``write_sub_rows`` without cloning ``cache_words``. Contiguous page runs
     coalesce into ``burst_rows = page_tokens // sub_slots_per_row`` async HBM write
     bursts (supporting both 16-token 1x pages and 8-token 2x-compressed pages).
2. ``pack_fp8_kv_rows`` (Standalone ``csa_cache_layout`` Quantization & Packing):
   - Layer 1 Compute (``_quantize_nope_fp8_words``, ``_pack_rope_bf16_words``):
     Quantizes 448 NoPE channels into 4x128-lane FP8 E4M3FN subtiles with 64-channel
     per-group UE8M0 biased exponents (clamped to ``amax >= 1e-4``) and packs 64
     BF16 RoPE channels into 32 ``int32`` words using 128-lane-aligned slices and
     ``pltpu.roll`` lane shuffles.
   - Layer 2/3 Pipeline (``_pack_fp8_tile``, ``_pack_fp8_pipeline``): Streams up to
     ``MAX_PACK_TILE = 512`` tokens per tile via ``pltpu.emit_pipeline`` (~4.15 VPU
     elem-ops/Byte > 2.44 VPU elem-ops/Byte TPU7x ridge-point; VPU/XLU-pipeline
     bound).
3. ``gather_cache_rows`` (Two-Level Burst 1-Bank / 2-Bank Row Gather):
   - Layer 2/3 Hierarchical Async DMA: Stages up to ``MAX_GATHER_BLOCK = 2048`` rows
     per grid block with split ``dma_issue`` and ``dma_wait`` passes, issuing 32-row
     (16 KiB) page bursts first, then 8-row (4 KiB) sub-bursts and 1-row (512 B)
     scattered DMAs across ``pool_lo`` and optional ``pool_hi``.

VMEM & SMEM Capacity Invariants (P-15, TPU7x 64 MiB VMEM / 1 MiB SMEM):
- ``write_sub_rows`` (``MAX_WRITE_BLOCK = 128``): ``128 * 512 B = 64 KiB`` VMEM
  scratch + ``64 KiB`` updates/mask tiles (``< 0.3%`` of 64 MiB VMEM) and
  ``<= 2 KiB`` SMEM metadata per block.
- ``pack_fp8_kv_rows`` (``MAX_PACK_TILE = 512``): ``2 * 512 * 1024 B = 1.0 MiB``
  double-buffered BF16 input + ``2 * 512 * 640 B = 1.25 MiB`` double-buffered
  ``int32`` outputs = ``2.25 MiB`` VMEM (``3.5%`` of 64 MiB VMEM).
- ``gather_cache_rows`` (``MAX_GATHER_BLOCK = 2048``): ``2 * 2048 * 512 B = 2.0 MiB``
  double-buffered VMEM output (``3.1%`` of 64 MiB VMEM) and ``<= 16 KiB`` SMEM
  metadata per block.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# TPU7x (Ironwood) hardware & word-row geometry constants.
TPU7X_GENERATION = 7
WORD_BYTES = 4
BF16_BYTES = 2
BF16_PER_WORD = WORD_BYTES // BF16_BYTES
ROW_WORDS = 128
ROW_BYTES = ROW_WORDS * WORD_BYTES
MIN_SUBLANE_ROWS = 8
MAX_INT32_EXCLUSIVE = 1 << 31

# DeepSeek-V4.1 MLA / Indexer / csa_cache_layout channel & word widths.
NOPE_DIM = 448
ROPE_DIM = 64
LATENT_DIM = NOPE_DIM + ROPE_DIM
INDEXER_HEAD_DIM = 128
ROPE_WORDS = ROPE_DIM // BF16_PER_WORD
INDEXER_KEY_WORDS = INDEXER_HEAD_DIM // BF16_PER_WORD
VALID_SLICE_WORDS = (ROPE_WORDS, INDEXER_KEY_WORDS, ROW_WORDS)

# csa_cache_layout FP8 E4M3FN + UE8M0 quantization constants.
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

# Page granularity & P-15 VMEM/SMEM tile limits.
DEFAULT_PAGE_TOKENS = 16
COMPRESSED_PAGE_TOKENS = 8
MAX_WRITE_BLOCK = 128
MAX_PACK_TILE = 512
MAX_GATHER_BLOCK = 2048
GATHER_PAGE_BURST_ROWS = 32
GATHER_SUB_BURST_ROWS = 8
GATHER_SUB_CHUNKS_PER_PAGE = GATHER_PAGE_BURST_ROWS // GATHER_SUB_BURST_ROWS
GATHER_BLOCK_DIVISOR_SHIFT = 3
SUB_CHUNK_COVERED_BY_PAGE_MODE = 3

HBM = pltpu.MemorySpace.HBM
SMEM = pltpu.MemorySpace.SMEM


def _scope(enable_named_scope: bool, scope_name: str) -> AbstractContextManager[Any]:
    """Return a ``jax.named_scope`` context when XProf scope tracing is enabled."""
    return jax.named_scope(scope_name) if enable_named_scope else nullcontext()


def _check_tpu7x() -> None:
    """Validate that the active TPU device or abstract mesh is TPU7x (128 lanes).

    Raises:
        RuntimeError: If the active TPU generation is not ``7`` or its vector
            lane count is not ``ROW_WORDS`` (``128``).
    """
    tpu_info = pltpu.get_tpu_info()
    if tpu_info.generation != TPU7X_GENERATION or tpu_info.num_lanes != ROW_WORDS:
        raise RuntimeError(
            f"Requires TPU7x (generation={TPU7X_GENERATION}, "
            f"num_lanes={ROW_WORDS}), got generation={tpu_info.generation}, "
            f"num_lanes={tpu_info.num_lanes}."
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
    """Layer-2 DMA helper: issue or wait on a ``num_rows`` async copy when active.

    Args:
        predicate: Scalar boolean/integer JAX predicate guarding the DMA transfer.
        src_ref: Source HBM or VMEM Pallas ``Ref`` of shape ``(>=num_rows, 128)``.
        src_row_offset: Starting row index in ``src_ref``.
        dst_ref: Destination VMEM or HBM Pallas ``Ref`` of shape ``(>=num_rows, 128)``.
        dst_row_offset: Starting row index in ``dst_ref``.
        num_rows: Static number of contiguous 128-word rows to copy.
        dma_sem: Pallas DMA semaphore tracking the transfer.
        is_wait_phase: When ``False``, calls ``.start()``; when ``True``, calls
            ``.wait()``.
    """
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
    """Pad a 2D SMEM metadata table to a strictly larger power-of-two column width.

    Args:
        raw_meta_2d: ``int32[num_blocks, raw_cols]`` per-block metadata array.

    Returns:
        A 2-tuple ``(padded_meta_1d, padded_cols)`` where ``padded_cols`` is a
        power of two ``>= ROW_WORDS`` (``128``) and strictly greater than
        ``raw_cols`` so ``padded_meta_1d[-1]`` is always an unused padding slot.
    """
    raw_cols = raw_meta_2d.shape[1]
    padded_cols = max(ROW_WORDS, 1 << raw_cols.bit_length())
    padded_meta_1d = jnp.pad(
        raw_meta_2d, ((0, 0), (0, padded_cols - raw_cols))
    ).reshape(-1)
    return padded_meta_1d, padded_cols


def write_sub_rows(
    cache_words: jax.Array,
    updates: jax.Array,
    slot_mapping: jax.Array,
    *,
    slot_offset: int = 0,
    after: jax.Array | None = None,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Write ``32``-, ``64``-, or ``128``-word slices into ``int32[num_rows, 128]``.

    Supports 2-bank HBM windowing via ``slot_offset``, deterministic last-writer-
    wins deduplication on duplicate sub-slots, partial sub-row read-modify-write
    merging in VMEM, and page-coalesced multi-row async HBM write bursts.

    Data Lifecycle:
        - **Created**: Host-preprocessed SMEM metadata ``padded_meta``, aligned
          update/mask tiles ``aligned_updates`` and ``aligned_mask``, VMEM scratch
          ``row_scratch_ref`` of shape ``(block_rows, 128)``, and the returned
          updated ``cache_words`` view.
        - **Used (Read-Only)**: ``updates``, ``slot_mapping``, and optional
          ``after`` (whose first scalar is bitcast into unused SMEM padding
          ``padded_meta[-1]`` of Operand 0 to enforce a zero-copy XLA HLO control
          edge after prior ``gather_cache_rows`` calls).
        - **Changed (In-Place Donated)**: ``cache_words`` (aliased to output 0 via
          ``input_output_aliases={2: 0}``).

    Args:
        cache_words: Target cache bank of shape ``int32[num_rows, 128]``.
        updates: Sub-row word slices of shape ``int32[num_tokens, slice_words]``
            with ``slice_words`` in ``(32, 64, 128)`` (4, 2, or 1 sub-slots/row).
        slot_mapping: Global sub-slot destination indices ``int32[num_tokens]``.
            Entries with ``slot_mapping - slot_offset`` outside
            ``[0, num_rows * (128 // slice_words))`` are ignored (no-op).
        slot_offset: Non-negative starting sub-slot offset of ``cache_words``
            for 2-bank segmented HBM addressing.
        after: Optional tensor from a preceding read on the same cache bank. Its
            first scalar element is bitcast to ``int32`` and stored into unused
            SMEM padding ``padded_meta[-1]`` so XLA schedules the read before this
            in-place write without cloning ``cache_words``.
        page_tokens: Number of sub-slot tokens per physical page (e.g., ``16`` for
            1x full-rate pages, ``8`` for 2x-compressed MLA/Indexer pages, or
            ``32`` for 2-row/token SWA pages). Must be a positive power of two
            (when ``page_tokens < 128 // slice_words``, ``burst_rows`` clamps to
            ``1``).
        interpret: Run ``pl.pallas_call`` in CPU interpret mode when ``True``.
        enable_named_scope: Emit XProf ``jax.named_scope`` markers when ``True``.

    Returns:
        Updated ``int32[num_rows, 128]`` cache bank array.

    Raises:
        ValueError: If tensor ranks, trailing dimensions, dtypes, ``slot_offset``,
            or ``page_tokens`` violate the contract.
        RuntimeError: If the active TPU target or abstract mesh is not TPU7x.
    """
    if cache_words.shape[1:] != (ROW_WORDS,) or cache_words.dtype != jnp.int32:
        raise ValueError("cache_words must be int32[num_rows, 128].")
    is_bad_updates = updates.ndim != 2 or updates.shape[1] not in VALID_SLICE_WORDS
    if is_bad_updates or updates.dtype != jnp.int32:
        raise ValueError("updates must be int32[num_tokens, 32|64|128].")
    if slot_mapping.shape != (updates.shape[0],) or slot_mapping.dtype != jnp.int32:
        raise ValueError("slot_mapping must be int32[num_tokens].")
    if slot_offset < 0 or slot_offset >= MAX_INT32_EXCLUSIVE:
        raise ValueError(f"slot_offset must be in [0, 2**31); got {slot_offset}.")
    sub_slots_per_row = ROW_WORDS // updates.shape[1]
    if page_tokens <= 0 or (page_tokens & (page_tokens - 1)) != 0:
        raise ValueError(
            f"page_tokens must be a positive power of two; got {page_tokens}."
        )
    _check_tpu7x()
    num_rows = cache_words.shape[0]
    num_tokens, slice_words = updates.shape
    if num_rows == 0 or num_tokens == 0:
        return cache_words

    bank_total_slots = num_rows * sub_slots_per_row
    slot_offset_i32 = jnp.int32(slot_offset)
    in_bank_mask = (slot_mapping >= slot_offset_i32) & (
        (slot_mapping - slot_offset_i32) < bank_total_slots
    )
    local_slots = jnp.where(
        in_bank_mask,
        slot_mapping - slot_offset_i32,
        jnp.int32(bank_total_slots),
    )
    sort_by_slot_order = jnp.argsort(local_slots, stable=True)
    sorted_slots = local_slots[sort_by_slot_order]
    is_last_writer = (sorted_slots < bank_total_slots) & jnp.pad(
        sorted_slots[:-1] != sorted_slots[1:], (0, 1), constant_values=True
    )
    compact_valid_order = jnp.argsort(~is_last_writer, stable=True)
    deduped_slots = jnp.where(
        is_last_writer[compact_valid_order],
        sorted_slots[compact_valid_order],
        jnp.int32(bank_total_slots),
    )
    deduped_updates = updates[sort_by_slot_order[compact_valid_order]]

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
    num_burst_chunks = block_rows // burst_rows
    num_blocks = (num_tokens + block_rows - 1) // block_rows
    pad_tokens = num_blocks * block_rows - num_tokens

    if sub_slots_per_row == 1:
        lead_rows = deduped_slots
        aligned_updates = jnp.pad(deduped_updates, ((0, pad_tokens), (0, 0)))
        aligned_mask = jnp.ones((MIN_SUBLANE_ROWS, ROW_WORDS), dtype=jnp.int32)
        mask_block_spec = pl.BlockSpec(
            (MIN_SUBLANE_ROWS, ROW_WORDS), lambda block_idx: (0, 0)
        )
        all_sub_slots_present = jnp.pad(
            lead_rows < num_rows, (0, pad_tokens), constant_values=False
        )
    else:
        padded_slots = jnp.pad(
            deduped_slots, (0, sub_slots_per_row), constant_values=bank_total_slots
        )
        padded_updates = jnp.pad(deduped_updates, ((0, sub_slots_per_row), (0, 0)))
        candidate_rows = deduped_slots // sub_slots_per_row
        is_row_leader = (candidate_rows < num_rows) & jnp.pad(
            candidate_rows[1:] != candidate_rows[:-1],
            (1, 0),
            constant_values=True,
        )
        leader_order = jnp.argsort(~is_row_leader, stable=True)
        lead_rows = jnp.where(
            is_row_leader[leader_order],
            candidate_rows[leader_order],
            jnp.int32(num_rows),
        )
        sub_slot_offsets = jnp.arange(sub_slots_per_row, dtype=jnp.int32)[None, :]
        window_indices = leader_order[:, None] + sub_slot_offsets
        target_slots = (lead_rows[:, None] * sub_slots_per_row + sub_slot_offsets)[
            :, :, None
        ]
        window_slots = padded_slots[window_indices][:, None, :]
        window_updates = padded_updates[window_indices][:, None, :, :]
        sub_slot_matches = (lead_rows[:, None, None] < num_rows) & (
            target_slots == window_slots
        )
        has_sub_slot = jnp.any(sub_slot_matches, axis=2)
        merged_sub_slots = jnp.sum(
            jnp.where(sub_slot_matches[..., None], window_updates, 0), axis=2
        )
        aligned_updates = jnp.pad(
            merged_sub_slots.reshape(num_tokens, ROW_WORDS),
            ((0, pad_tokens), (0, 0)),
        )
        word_write_mask = jnp.repeat(
            has_sub_slot.astype(jnp.int32), slice_words, axis=1
        )
        aligned_mask = jnp.pad(word_write_mask, ((0, pad_tokens), (0, 0)))
        mask_block_spec = pl.BlockSpec(
            (block_rows, ROW_WORDS), lambda block_idx: (block_idx, 0)
        )
        all_sub_slots_present = jnp.pad(
            jnp.all(has_sub_slot, axis=1),
            (0, pad_tokens),
            constant_values=False,
        )

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

    # SMEM meta_ref layout (padded to meta_cols in {128, 256, 512}):
    #   meta_ref[0] = block_mode (0: empty, 1: partial/scattered, 2: all-burst),
    #   meta_ref[1 .. 1+num_burst_chunks] = burst_chunk_modes,
    #   meta_ref[1+num_burst_chunks .. 1+num_burst_chunks+block_rows] = lead_rows,
    #   meta_ref[1+num_burst_chunks+block_rows .. 1+num_burst_chunks+2*block_rows]
    #       = block_needs_read, and
    #   padded_meta[-1] = trailing `after` ordering dependency token slot.
    rows_smem_offset = 1 + num_burst_chunks
    needs_read_smem_offset = rows_smem_offset + block_rows

    def _write_sub_rows_kernel(
        meta_ref: Any,
        mask_ref: Any,
        cache_in_ref: Any,
        updates_ref: Any,
        cache_out_ref: Any,
        row_scratch_ref: Any,
        dma_sem: Any,
    ) -> None:
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
    return pl.pallas_call(
        _write_sub_rows_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            0, (num_blocks,), in_specs, hbm_block_spec, scratch_shapes
        ),
        out_shape=cache_words,
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.ARBITRARY,)),
        interpret=interpret,
    )(padded_meta, aligned_mask, cache_words, aligned_updates)


def _quantize_nope_fp8_words(latent_f32: jax.Array) -> jax.Array:
    """Layer-1 compute: quantize 448 NoPE channels into ``csa_cache_layout`` words.

    Packs 448 ``float8_e4m3fn`` values (bytes 0..2 of all 128 lanes and byte 3 of
    lanes 0..63) plus 64 biased UE8M0 64-channel block scales (byte 3 of lanes
    64..127, where lane ``64 + c`` holds the scale for ``c % 64`` across the 7
    64-channel NoPE groups) into ``int32[tile_tokens, 128]``.

    Args:
        latent_f32: ``float32[tile_tokens, 512]`` VMEM tile whose first 448
            columns hold the absorbed NoPE latent channels.

    Returns:
        Packed ``int32[tile_tokens, 128]`` ``csa_cache_layout`` NoPE words.
    """
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
    """Layer-1 compute: pack 64 BF16 RoPE channels into 32 ``int32`` words.

    Because Mosaic VMEM 2D column slices must start on a multiple of ``ROW_WORDS``
    (``128``), this helper slices ``latent_bf16[:, 384:512]`` (width 128, where
    RoPE occupies columns ``[448:512]``, i.e. local lanes ``[64:128]``) and uses
    ``pltpu.roll`` by 64 and 32 lanes to pack local lanes ``[64:96]`` (low 16 bits)
    and ``[96:128]`` (high 16 bits) into lanes ``[0:32]``.

    Args:
        latent_bf16: ``bfloat16[tile_tokens, 512]`` VMEM tile.

    Returns:
        Packed ``int32[tile_tokens, 32]`` ``csa_cache_layout`` RoPE words.
    """
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
    """Layer-2 per-tile body executed inside ``pltpu.emit_pipeline``."""
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
    """Layer-3 ``pltpu.emit_pipeline`` driver overlapping DMA and VPU packing."""
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
    """Pack ``bf16[num_tokens, 512]`` latents into ``(nope_words, rope_words)``.

    Quantizes ``latent[:, :448]`` into FP8 E4M3FN with 64-channel UE8M0 block
    scales and packs ``latent[:, 448:512]`` into 32 BF16 words matching
    ``vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout`` bit-for-bit.

    Data Lifecycle:
        - **Created**: Padded input buffer (when ``num_tokens`` is not a multiple
          of ``tile_tokens``), double-buffered ``pltpu.emit_pipeline`` VMEM tiles,
          and output arrays ``(nope_words, rope_words)`` in HBM.
        - **Used (Read-Only)**: ``latent`` of shape ``bfloat16[num_tokens, 512]``.
        - **Changed**: None (pure functional return).

    Args:
        latent: Input MLA latent tensor of shape ``bfloat16[num_tokens, 512]``.
        interpret: Run ``pl.pallas_call`` in CPU interpret mode when ``True``.
        enable_named_scope: Emit XProf ``jax.named_scope`` markers when ``True``.

    Returns:
        A 2-tuple ``(nope_words, rope_words)`` with shapes
        ``(int32[num_tokens, 128], int32[num_tokens, 32])``.

    Raises:
        ValueError: If ``latent`` is not a 2D ``bfloat16`` tensor of width ``512``.
        RuntimeError: If the active TPU target or abstract mesh is not TPU7x.
    """
    if latent.shape[1:] != (LATENT_DIM,) or latent.dtype != jnp.bfloat16:
        raise ValueError("latent must be bfloat16[num_tokens, 512].")
    _check_tpu7x()
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


def _contig_mode(chunk_indices_3d: jax.Array, rows_lo: int, rows_hi: int) -> jax.Array:
    """Classify each chunk of ``chunk_width`` row indices for burst DMA gathering.

    Args:
        chunk_indices_3d: ``int32[num_blocks, num_chunks, chunk_width]`` row indices.
        rows_lo: Number of rows in ``pool_lo``.
        rows_hi: Number of rows in ``pool_hi`` (``0`` when ``pool_hi is None``).

    Returns:
        ``int32[num_blocks, num_chunks]`` burst mode codes where ``1`` means a
        contiguous burst inside ``pool_lo``, ``2`` means a contiguous burst inside
        ``pool_hi``, and ``0`` means non-contiguous or out-of-bounds indices.
    """
    chunk_width = chunk_indices_3d.shape[2]
    start_indices = chunk_indices_3d[:, :, :1]
    is_contiguous = jnp.all(
        chunk_indices_3d == start_indices + jnp.arange(chunk_width, dtype=jnp.int32),
        axis=2,
    )
    chunk_start_row = start_indices[:, :, 0]
    in_low_bank = (
        is_contiguous
        & (chunk_start_row >= 0)
        & (chunk_start_row + chunk_width <= rows_lo)
    )
    in_high_bank = (
        is_contiguous
        & (chunk_start_row >= rows_lo)
        & (chunk_start_row - rows_lo + chunk_width <= rows_hi)
    )
    return jnp.where(in_low_bank, 1, in_high_bank.astype(jnp.int32) * 2)


def gather_cache_rows(
    pool_lo: jax.Array,
    row_indices: jax.Array,
    *,
    pool_hi: jax.Array | None = None,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Gather ``128``-word rows from 1-bank or 2-bank HBM pools (OOB zero-filled).

    Uses a two-level async DMA burst schedule (32-row page bursts, 8-row sub-page
    bursts, and 1-row scattered DMAs) across ``pool_lo`` and optional ``pool_hi``
    so pools exceeding TPU7x SparseCore's 16 GiB single-tensor HBM limit are
    addressed seamlessly over ``[0, rows_lo + rows_hi)``.

    Data Lifecycle:
        - **Created**: Host-classified SMEM burst metadata ``padded_meta`` and
          gathered output array ``int32[num_queries, 128]`` in HBM.
        - **Used (Read-Only)**: ``pool_lo``, optional ``pool_hi``, and
          ``row_indices``.
        - **Changed**: None (pure functional return).

    Args:
        pool_lo: Primary HBM cache bank of shape ``int32[rows_lo, 128]``.
        row_indices: 1D query row indices ``int32[num_queries]``. Indices in
            ``[0, rows_lo)`` read ``pool_lo``; indices in
            ``[rows_lo, rows_lo + rows_hi)`` read ``pool_hi[idx - rows_lo]``;
            negative or out-of-bounds indices produce all-zero rows.
        pool_hi: Optional secondary HBM cache bank ``int32[rows_hi, 128]``.
        interpret: Run ``pl.pallas_call`` in CPU interpret mode when ``True``.
        enable_named_scope: Emit XProf ``jax.named_scope`` markers when ``True``.

    Returns:
        Gathered rows of shape ``int32[num_queries, 128]``.

    Raises:
        ValueError: If ``pool_lo``, ``pool_hi``, or ``row_indices`` violate rank,
            width, or ``int32`` dtype constraints.
        RuntimeError: If the active TPU target or abstract mesh is not TPU7x.
    """
    if pool_lo.shape[1:] != (ROW_WORDS,) or pool_lo.dtype != jnp.int32:
        raise ValueError("pool_lo must be int32[rows_lo, 128].")
    if pool_hi is not None and (
        pool_hi.shape[1:] != (ROW_WORDS,) or pool_hi.dtype != jnp.int32
    ):
        raise ValueError("pool_hi must be int32[rows_hi, 128].")
    if row_indices.ndim != 1 or row_indices.dtype != jnp.int32:
        raise ValueError("row_indices must be a 1D int32 array.")
    _check_tpu7x()

    if pool_hi is not None and pool_hi.shape[0] == 0:
        pool_hi = None
    if pool_lo.shape[0] == 0 and pool_hi is not None:
        pool_lo, pool_hi = pool_hi, None
    num_queries = row_indices.shape[0]
    rows_lo = pool_lo.shape[0]
    if num_queries == 0 or rows_lo == 0:
        return jnp.zeros((num_queries, ROW_WORDS), dtype=jnp.int32)

    block_rows = min(
        MAX_GATHER_BLOCK,
        max(
            GATHER_PAGE_BURST_ROWS,
            1 << max(0, num_queries.bit_length() - GATHER_BLOCK_DIVISOR_SHIFT),
        ),
    )
    num_page_chunks = block_rows // GATHER_PAGE_BURST_ROWS
    num_sub_chunks = block_rows // GATHER_SUB_BURST_ROWS
    indices_smem_offset = 1 + num_page_chunks + num_sub_chunks
    num_blocks = (num_queries + block_rows - 1) // block_rows
    padded_queries = num_blocks * block_rows
    padded_indices = jnp.pad(
        row_indices, (0, padded_queries - num_queries), constant_values=-1
    )
    rows_hi = 0 if pool_hi is None else pool_hi.shape[0]
    pool_hi_arg = pool_lo[:1] if pool_hi is None else pool_hi
    page_chunk_modes = _contig_mode(
        padded_indices.reshape(num_blocks, num_page_chunks, GATHER_PAGE_BURST_ROWS),
        rows_lo,
        rows_hi,
    )
    sub_chunk_modes_raw = _contig_mode(
        padded_indices.reshape(num_blocks, num_sub_chunks, GATHER_SUB_BURST_ROWS),
        rows_lo,
        rows_hi,
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

    # SMEM meta_ref layout (padded to meta_cols):
    #   meta_ref[0] = has_non_page_chunks (1 if any 32-row chunk needs sub-8/1-row),
    #   meta_ref[1 .. 1+num_page_chunks] = page_chunk_modes (0: sub-8, 1: lo, 2: hi),
    #   meta_ref[1+num_page_chunks .. indices_smem_offset] = sub_chunk_modes
    #       (0: 1-row, 1: lo, 2: hi, 3: covered by page_chunk_modes),
    #   meta_ref[indices_smem_offset .. indices_smem_offset+block_rows] = indices.
    def _gather_cache_rows_kernel(
        meta_ref: Any,
        pool_lo_ref: Any,
        pool_hi_ref: Any,
        out_ref: Any,
        sem_lo: Any,
        sem_hi: Any,
    ) -> None:
        def _execute_dma_phase(is_wait_phase: bool) -> None:
            def _dispatch_bank_dma(
                in_low_bank: Any,
                in_high_bank: Any,
                global_src_row: Any,
                dst_row_offset: Any,
                num_rows: int,
            ) -> None:
                _dma_rows(
                    in_low_bank,
                    pool_lo_ref,
                    global_src_row,
                    out_ref,
                    dst_row_offset,
                    num_rows,
                    sem_lo,
                    is_wait_phase,
                )
                if rows_hi > 0:
                    _dma_rows(
                        in_high_bank,
                        pool_hi_ref,
                        global_src_row - rows_lo,
                        out_ref,
                        dst_row_offset,
                        num_rows,
                        sem_hi,
                        is_wait_phase,
                    )

            def _gather_page_chunk(page_chunk_idx: jax.Array, _: None) -> None:
                page_mode = meta_ref[1 + page_chunk_idx]
                dst_offset = page_chunk_idx * GATHER_PAGE_BURST_ROWS
                _dispatch_bank_dma(
                    page_mode == 1,
                    page_mode == 2,
                    meta_ref[indices_smem_offset + dst_offset],
                    dst_offset,
                    GATHER_PAGE_BURST_ROWS,
                )

            jax.lax.fori_loop(0, num_page_chunks, _gather_page_chunk, None)

            @pl.when(meta_ref[0] != 0)
            def _gather_sub_and_scattered() -> None:
                def _gather_sub_chunk(sub_chunk_idx: jax.Array, _: None) -> None:
                    sub_mode = meta_ref[1 + num_page_chunks + sub_chunk_idx]
                    sub_offset = sub_chunk_idx * GATHER_SUB_BURST_ROWS
                    _dispatch_bank_dma(
                        sub_mode == 1,
                        sub_mode == 2,
                        meta_ref[indices_smem_offset + sub_offset],
                        sub_offset,
                        GATHER_SUB_BURST_ROWS,
                    )

                    @pl.when(sub_mode == 0)
                    def _gather_scattered_rows() -> None:
                        def _gather_single_row(
                            local_offset: jax.Array, _: None
                        ) -> None:
                            query_in_block = sub_offset + local_offset
                            row_idx = meta_ref[indices_smem_offset + query_in_block]
                            high_bank_row = row_idx - rows_lo
                            in_low_bank = (row_idx >= 0) & (row_idx < rows_lo)
                            in_high_bank = (high_bank_row >= 0) & (
                                high_bank_row < rows_hi
                            )
                            _dispatch_bank_dma(
                                in_low_bank,
                                in_high_bank,
                                row_idx,
                                query_in_block,
                                1,
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

    hbm_block_spec = pl.BlockSpec(memory_space=HBM)
    in_specs = [
        pl.BlockSpec((meta_cols,), lambda block_idx: (block_idx,), memory_space=SMEM),
        hbm_block_spec,
        hbm_block_spec,
    ]
    dma_semaphores = [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA]
    out_spec = pl.BlockSpec((block_rows, ROW_WORDS), lambda block_idx: (block_idx, 0))
    gathered_rows = pl.pallas_call(
        _gather_cache_rows_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            0, (num_blocks,), in_specs, out_spec, dma_semaphores
        ),
        out_shape=jax.ShapeDtypeStruct((padded_queries, ROW_WORDS), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.PARALLEL,)),
        interpret=interpret,
    )(padded_meta, pool_lo, pool_hi_arg)
    return gathered_rows[:num_queries]
