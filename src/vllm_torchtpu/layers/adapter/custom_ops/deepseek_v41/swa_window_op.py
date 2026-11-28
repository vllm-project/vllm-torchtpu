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
"""Thin ``pallas.jax_op`` bridge for DeepSeek-V4.1 paged SWA window gather."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.cache.swa_window import (
    ROW_WORDS,
    gather_swa_windows,
)


def _gather_swa_windows_jax(
    pool: jax.Array,
    page_ids: jax.Array,
    phase_rows: jax.Array,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
) -> jax.Array:
    # Avoid PyTorch .view(-1, 128) on 3D root pools before custom-op entry:
    # passing view inputs to the compiled TPU backend triggers extra strided-view
    # copy/relayout overhead.
    # Avoid pool.reshape(-1, ROW_WORDS): preserve shape[2] so malformed 3D
    # last dimensions fail kernel validation instead of reinterpreting rows.
    if pool.ndim == 3:
        pool = pool.reshape(pool.shape[0] * pool.shape[1], pool.shape[2])
    return gather_swa_windows(
        pool,
        page_ids,
        phase_rows,
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )


_gather_swa_windows_op = pallas.jax_op(
    "pallas::dsv41_gather_swa_windows",
    _gather_swa_windows_jax,
)


@_gather_swa_windows_op.register_fake
def _fake_gather_swa_windows(
    pool: torch.Tensor,
    page_ids: torch.Tensor,
    phase_rows: torch.Tensor,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
) -> torch.Tensor:
    del phase_rows, rows_per_page, page_row_stride, page_row_offset
    return pool.new_empty(
        (page_ids.shape[0], window_rows, ROW_WORDS), dtype=torch.int32
    )


def dsv41_gather_swa_windows(
    pool: torch.Tensor,
    page_ids: torch.Tensor,
    phase_rows: torch.Tensor,
    *,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int = 0,
) -> torch.Tensor:
    """Gather ``int32[S, window_rows, 128]`` SWA windows from 2D/3D ``int32`` pools.

    Indexes ``pool: int32[R, 128]`` or ``[N, P, 128]`` by ``page_ids: int32[S, P_win]``
    and ``phase_rows: int32[S]``; out-of-range ``page_ids`` (``< 0`` or
    ``>= max_pages``) skip DMA and leave output slots uninitialized.
    """
    return _gather_swa_windows_op(
        pool,
        page_ids,
        phase_rows,
        int(window_rows),
        int(rows_per_page),
        int(page_row_stride),
        int(page_row_offset),
    )


__all__ = ["dsv41_gather_swa_windows"]
