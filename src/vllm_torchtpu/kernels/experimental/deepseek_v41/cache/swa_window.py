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
"""Paged sliding-window gather via whole-page HBM bursts and VMEM window DMA."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.cache.kernel import (
    ROW_BYTES,
    ROW_WORDS,
)

TPU7X_GENERATION = 7
DOUBLE_BUFFER = 2
DEFAULT_SEQS_PER_STEP = 8


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


def _validate_swa_window_inputs(
    pool: jax.Array,
    page_ids: jax.Array,
    phase_rows: jax.Array,
    *,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
    seqs_per_step: int,
) -> Any:
    if pool.ndim != 2 or pool.shape[1] != ROW_WORDS or pool.dtype != jnp.int32:
        raise ValueError("pool must be int32[num_rows, 128].")
    if page_ids.ndim != 2 or page_ids.dtype != jnp.int32:
        raise ValueError("page_ids must be int32[num_seqs, pages_per_window].")
    if phase_rows.shape != page_ids.shape[:1] or phase_rows.dtype != jnp.int32:
        raise ValueError("phase_rows must be int32[num_seqs].")
    if rows_per_page <= 0 or window_rows <= 0 or seqs_per_step <= 0:
        raise ValueError(
            "rows_per_page, window_rows, and seqs_per_step must be positive."
        )
    if page_row_stride < rows_per_page or not (
        0 <= page_row_offset <= page_row_stride - rows_per_page
    ):
        raise ValueError("Each page must fit inside its page_row_stride slot.")
    if pool.shape[0] < page_row_offset + rows_per_page:
        raise ValueError(
            "pool.shape[0] must be at least page_row_offset + rows_per_page."
        )
    # Avoid reading past staged pages for the maximum valid phase (rows_per_page - 1).
    if page_ids.shape[1] * rows_per_page < window_rows + rows_per_page - 1:
        raise ValueError("pages_per_window too small to cover every window phase.")
    return _check_tpu7x()


def _swa_window_kernel(
    pages_ref: Any,
    phase_ref: Any,
    pool_ref: Any,
    out_ref: Any,
    staged_ref: Any,
    page_sem: Any,
    window_sem: Any,
    *,
    seqs_per_step: int,
    pages_per_window: int,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
    enable_named_scope: bool,
) -> None:
    first_seq = pl.program_id(0) * seqs_per_step
    max_pages = (
        pool_ref.shape[0] - page_row_offset - rows_per_page
    ) // page_row_stride + 1

    def _page_copy(local_seq: int, page_slot: int) -> Any:
        page = pages_ref[(first_seq + local_seq) * pages_per_window + page_slot]
        src_row = page * page_row_stride + page_row_offset
        dst_row = page_slot * rows_per_page
        return pltpu.make_async_copy(
            pool_ref.at[pl.ds(src_row, rows_per_page)],
            staged_ref.at[local_seq, pl.ds(dst_row, rows_per_page)],
            page_sem,
        )

    def _page_is_valid(local_seq: int, page_slot: int) -> jax.Array:
        page = pages_ref[(first_seq + local_seq) * pages_per_window + page_slot]
        return (page >= 0) & (page < max_pages)

    def _window_copy(local_seq: int) -> Any:
        phase = phase_ref[first_seq + local_seq]
        return pltpu.make_async_copy(
            staged_ref.at[local_seq, pl.ds(phase, window_rows)],
            out_ref.at[local_seq],
            window_sem,
        )

    slots = [
        (local_seq, page_slot)
        for local_seq in range(seqs_per_step)
        for page_slot in range(pages_per_window)
    ]
    # Avoid DMA from out-of-range page_ids (< 0 or >= max_pages) or checking
    # page * page_row_stride + ... <= R in int32: out-of-range offsets read
    # OOB HBM and multiplying large page_ids overflows int32.
    with _scope(enable_named_scope, "dma_stage_pages"):
        for is_wait_phase in (False, True):
            for local_seq, page_slot in slots:

                @pl.when(_page_is_valid(local_seq, page_slot))
                def _issue_or_wait(
                    *,
                    local_seq: int = local_seq,
                    page_slot: int = page_slot,
                    is_wait_phase: bool = is_wait_phase,
                ) -> None:
                    copy = _page_copy(local_seq, page_slot)
                    if is_wait_phase:
                        copy.wait()
                    else:
                        copy.start()

    # Avoid an XLA take_along_axis epilogue on staged pages: it lowers to a
    # generic gather that re-reads every staged page.
    with _scope(enable_named_scope, "dma_slice_windows"):
        for local_seq in range(seqs_per_step):
            _window_copy(local_seq).start()
        for local_seq in range(seqs_per_step):
            _window_copy(local_seq).wait()


def gather_swa_windows(
    pool: jax.Array,
    page_ids: jax.Array,
    phase_rows: jax.Array,
    *,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int = 0,
    seqs_per_step: int = DEFAULT_SEQS_PER_STEP,
    interpret: bool = False,
    enable_named_scope: bool = False,
) -> jax.Array:
    """Stage ``int32[R, 128]`` pages into VMEM and slice ``int32[S, W, 128]`` windows.

    Preconditions: ``0 <= phase_rows < rows_per_page`` and ``S * P_win`` fits in
    SMEM. Out-of-range ``page_ids`` (``< 0`` or ``>= max_pages``) skip HBM DMA and
    leave uninitialized VMEM rows; callers must mask unmapped positions via select.
    """
    tpu_info = _validate_swa_window_inputs(
        pool,
        page_ids,
        phase_rows,
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
        seqs_per_step=seqs_per_step,
    )
    num_seqs, pages_per_window = page_ids.shape
    staged_rows = pages_per_window * rows_per_page
    required_vmem_bytes = (
        seqs_per_step * (staged_rows + DOUBLE_BUFFER * window_rows) * ROW_BYTES
    )
    _check_vmem_budget(
        required_vmem_bytes,
        tpu_info,
        f"gather_swa_windows(seqs_per_step={seqs_per_step})",
    )
    if num_seqs == 0:
        return jnp.zeros((0, window_rows, ROW_WORDS), dtype=jnp.int32)

    padded_seqs = -(-num_seqs // seqs_per_step) * seqs_per_step
    pad = padded_seqs - num_seqs
    flat_pages = jnp.pad(page_ids, ((0, pad), (0, 0)), constant_values=-1).reshape(-1)
    padded_phase = jnp.pad(phase_rows, (0, pad))

    kernel_body = partial(
        _swa_window_kernel,
        seqs_per_step=seqs_per_step,
        pages_per_window=pages_per_window,
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
        enable_named_scope=enable_named_scope,
    )
    windows = pl.pallas_call(
        kernel_body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(padded_seqs // seqs_per_step,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(
                (seqs_per_step, window_rows, ROW_WORDS),
                lambda step, pages, phases: (step, 0, 0),
            ),
            scratch_shapes=[
                pltpu.VMEM((seqs_per_step, staged_rows, ROW_WORDS), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(
            (padded_seqs, window_rows, ROW_WORDS), jnp.int32
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.PARALLEL,)),
        interpret=interpret,
    )(flat_pages, padded_phase, pool)
    return windows[:num_seqs]
