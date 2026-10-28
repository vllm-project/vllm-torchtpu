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
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import config, memory_ref

# Sub-chunks whose index and weight fetches may be in flight at once.
INDEX_FETCH_DEPTH = 8
# Rows per reduce pass and the pass loop's unroll. On tpu7x, four rows unrolled
# by two schedule in ~33 bundles per 16 rows x 16 columns, against ~39 for one
# 16-row pass, whose per-row weights, shifts and predicates overflow registers.
REDUCE_ROWS_PER_PASS = 4
REDUCE_UNROLL = 2


def call_kernel_pipeline(
    row_partition_id: jax.Array,
    refs: memory_ref.KernelRefs,
    col_start: jax.Array,
    cfg: config.Config,
):
    num_rows_per_row_partition = refs.scratch.num_rows_per_row_partition_vmem[...]
    num_rows_current_row_partition = jnp.array(0, jnp.int32)
    for i in range(cfg.num_row_partitions):
        num_rows_current_row_partition = jnp.where(
            row_partition_id == i,
            num_rows_per_row_partition[i],
            num_rows_current_row_partition,
        )

    # Sentinel for the cross-row reduction carry (no previous group). The carry
    # is kernel scratch, so it also persists across window boundaries.
    refs.scratch.prev_dst_row_smem[0] = -1

    # The resident sort scratch holds one window of rows, so the rows are run one
    # window at a time. The window count is a runtime value (it follows the row
    # partition's row count), hence a pl.loop over the window index.
    num_windows = pl.cdiv(num_rows_current_row_partition, cfg.window_size)
    pl.loop(0, num_windows)(
        functools.partial(
            _window_kernel,
            refs=refs,
            row_partition_id=row_partition_id,
            num_rows_current_row_partition=num_rows_current_row_partition,
            col_start=col_start,
            cfg=cfg,
        )
    )


def _window_kernel(
    window_id: jax.Array,
    *,
    refs: memory_ref.KernelRefs,
    row_partition_id: jax.Array,
    num_rows_current_row_partition: jax.Array,
    col_start: jax.Array,
    cfg: config.Config,
):
    """Stages one window of rows, then reduces it in one flat pipeline."""
    num_simd_lanes = cfg.sc_info.num_lanes
    sorted_by_validity = refs.index.sorted_by_validity
    sorted_vmem = refs.scratch.sorted_by_validity_vmem
    recv_sem = refs.scratch.sem.at[0]
    window_words = cfg.window_size
    window_start = (
        row_partition_id * cfg.row_partition_size_padded + window_id * window_words
    )
    window_row_start = window_id * window_words

    # The sort window, followed by the next window's first rows so the window's
    # last row can tell whether its group continues. Clamped, so the last window
    # fetches a harmless past-the-end row instead.
    last_start = sorted_by_validity.shape[0] - num_simd_lanes
    next_window_start = jnp.minimum(window_start + window_words, last_start)
    sorted_dma = pltpu.make_async_copy(
        sorted_by_validity.at[pl.ds(window_start, window_words)],
        sorted_vmem.at[pl.ds(0, window_words)],
        recv_sem,
    )
    next_window_dma = pltpu.make_async_copy(
        sorted_by_validity.at[pl.ds(next_window_start, num_simd_lanes)],
        sorted_vmem.at[pl.ds(window_words, num_simd_lanes)],
        recv_sem,
    )
    sorted_dma.start()
    next_window_dma.start()
    sorted_dma.wait()
    next_window_dma.wait()

    rows_in_window = jnp.clip(
        num_rows_current_row_partition - window_row_start, 0, window_words
    )
    num_subchunks = pl.cdiv(rows_in_window, num_simd_lanes)

    # Gather the window's source rows and weights up front, so the row gather
    # below never waits on an index fetch.
    def index_copies(s):
        sub = pl.ds(pl.multiple_of(s * num_simd_lanes, num_simd_lanes), num_simd_lanes)
        rows = sorted_vmem[sub]
        return (
            pltpu.make_async_copy(
                refs.index.indices.at[rows],
                refs.scratch.src_indices_vmem.at[sub],
                recv_sem,
            ),
            pltpu.make_async_copy(
                refs.data.topk_weights.at[rows],
                refs.scratch.tw_f32_vmem.at[sub],
                recv_sem,
            ),
        )

    # At most INDEX_FETCH_DEPTH sub-chunks' fetches are in flight at once.
    @pl.loop(0, num_subchunks)
    def _(s):
        for copy in index_copies(s):
            copy.start()

        @pl.when(s >= INDEX_FETCH_DEPTH)
        def _():
            for copy in index_copies(s - INDEX_FETCH_DEPTH):
                copy.wait()

    @pl.loop(jnp.maximum(num_subchunks - INDEX_FETCH_DEPTH, 0), num_subchunks)
    def _(s):
        for copy in index_copies(s):
            copy.wait()

    num_steps = num_subchunks * cfg.num_col_chunks
    pltpu.emit_pipeline(
        functools.partial(
            _step_kernel,
            refs=refs,
            num_rows_current_row_partition=num_rows_current_row_partition,
            window_row_start=window_row_start,
            col_start=col_start,
            cfg=cfg,
        ),
        grid=(num_subchunks, cfg.num_col_chunks),
        in_specs=_col_gather_spec(refs.scratch.src_indices_vmem, col_start, cfg),
    )(refs.data.source.bitcast(jnp.uint32))

    # Drain the output writes still in flight: the last step's, and the one
    # before it when there was one.
    any_rows = jax.lax.broadcasted_iota(jnp.int32, (num_simd_lanes,), 0)
    for back in range(2):

        @pl.when(num_steps > back)
        def _():
            slot = (num_steps - 1 - back) % 2
            _out_copy(refs, slot, col_start, any_rows, cfg).wait()

    refs.scratch.prev_dst_row_smem[0] = (
        sorted_vmem[pl.ds(window_words - num_simd_lanes, num_simd_lanes)][
            num_simd_lanes - 1
        ]
        // cfg.reduce_group_size
    )


def _col_gather_spec(
    src_indices_vmem: jax.Ref,
    col_start: jax.Array,
    cfg: config.Config,
) -> pl.BlockSpec:
    """Indirect BlockSpec gathering columns of a 2-D input."""
    num_simd_lanes = cfg.sc_info.num_lanes

    def col_index_map(
        s: int | jax.Array, c: int | jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        row = jnp.bitwise_right_shift(
            src_indices_vmem[pl.ds(s * num_simd_lanes, num_simd_lanes)],
            cfg.row_shift,
        )
        col = col_start // cfg.col_chunk_size + c
        return (row, col)

    return pl.BlockSpec(
        (pl.Indirect(num_simd_lanes), cfg.col_chunk_size),
        col_index_map,
    )


def _out_copy(
    refs: memory_ref.KernelRefs,
    slot: int | jax.Array,
    col_hbm_start: jax.Array,
    dst_rows: jax.Array,
    cfg: config.Config,
):
    """The row scatter of one output slot; waits rebuild it with any rows."""
    cols = pl.ds(col_hbm_start, cfg.col_chunk_size)
    return pltpu.make_async_copy(
        refs.scratch.out_vmem.at[slot],
        refs.data.out.at[dst_rows, cols],
        refs.scratch.sem.at[1 + slot],
    )


def _step_kernel(
    gather_ref: jax.Ref,
    *,
    refs: memory_ref.KernelRefs,
    num_rows_current_row_partition: jax.Array,
    window_row_start: jax.Array,
    col_start: jax.Array,
    cfg: config.Config,
):
    s = pl.program_id(0)
    c = pl.program_id(1)
    step = s * cfg.num_col_chunks + c
    slot = step % 2
    col_hbm_start = col_start + c * cfg.col_chunk_size
    num_simd_lanes = cfg.sc_info.num_lanes
    sorted_vmem = refs.scratch.sorted_by_validity_vmem

    row_start = pl.multiple_of(s * num_simd_lanes, num_simd_lanes)
    row_slice = pl.ds(row_start, num_simd_lanes)
    tw_slice = refs.scratch.tw_f32_vmem[row_slice]
    src_idx_slice = refs.scratch.src_indices_vmem[row_slice]
    dst_slice = sorted_vmem[row_slice] // cfg.reduce_group_size
    prev_dst = jnp.where(
        s == 0,
        refs.scratch.prev_dst_row_smem[0],
        sorted_vmem[pl.ds(jnp.maximum(row_start - num_simd_lanes, 0), num_simd_lanes)][
            num_simd_lanes - 1
        ]
        // cfg.reduce_group_size,
    )

    # Each output row is written once, by its group's last row, after the
    # group's full sum is in that row. Other rows and padding rows go to a
    # throwaway row.
    lane = jax.lax.broadcasted_iota(jnp.int32, (num_simd_lanes,), 0)
    row = window_row_start + row_start + lane
    next_dst = (
        sorted_vmem[pl.ds(row_start + 1, num_simd_lanes)] // cfg.reduce_group_size
    )
    is_valid = row < num_rows_current_row_partition
    next_continues = jnp.logical_and(
        row + 1 < num_rows_current_row_partition, next_dst == dst_slice
    )
    is_final_write = jnp.logical_and(is_valid, jnp.logical_not(next_continues))
    garbage_dst = refs.data.out.shape[0] - 1
    dst_rows = jnp.where(is_final_write, dst_slice, garbage_dst)

    # The slot's previous scatter (two steps back) must land before the slot is
    # overwritten.
    @pl.when(step >= 2)
    def _():
        _out_copy(refs, slot, col_hbm_start, lane, cfg).wait()

    # One copy of the loop per slot: a static slot keeps the row stores at
    # immediate offsets.
    for static_slot in range(2):

        @pl.when(slot == static_slot)
        def _():
            _reduce_rows(
                gather_ref,
                refs.scratch.out_vmem.at[static_slot],
                refs=refs,
                c=c,
                tw_slice=tw_slice,
                src_idx_slice=src_idx_slice,
                dst_slice=dst_slice,
                prev_dst=prev_dst,
                cfg=cfg,
            )

    _out_copy(refs, slot, col_hbm_start, dst_rows, cfg).start()


def _reduce_rows(
    gather_ref: jax.Ref,
    out_vmem: jax.Ref,
    *,
    refs: memory_ref.KernelRefs,
    c: jax.Array,
    tw_slice: jax.Array,
    src_idx_slice: jax.Array,
    dst_slice: jax.Array,
    prev_dst: jax.Array,
    cfg: config.Config,
):
    """Weights one sub-chunk's gathered rows and runs the segmented sum.

    The rows run in passes of REDUCE_ROWS_PER_PASS, each seeded from the previous
    pass's last output row, so a pass keeps only its own rows' weights, shifts
    and group tests in registers.
    """
    num_simd_lanes = cfg.sc_info.num_lanes
    rows_per_pass = min(REDUCE_ROWS_PER_PASS, num_simd_lanes)
    assert num_simd_lanes % rows_per_pass == 0

    for pass_start in range(0, num_simd_lanes, rows_per_pass):

        @plsc.parallel_loop(
            0, cfg.col_chunk_size, step=num_simd_lanes, unroll=REDUCE_UNROLL
        )
        def col_loop(col_compute_offset: jax.Array):
            col_slice = pl.ds(col_compute_offset, num_simd_lanes)
            # Running sum, seeded by the row before the pass.
            if pass_start == 0:
                previous_accumulated_data = refs.scratch.prev_iter_last_row_vmem[
                    c, col_slice
                ]
            else:
                previous_accumulated_data = out_vmem[pass_start - 1, col_slice]
            for row_src in range(pass_start, pass_start + rows_per_pass):
                val_u32 = gather_ref[row_src, col_slice]
                if cfg.in_dtype == jnp.bfloat16:
                    # The two bfloat16 rows packed in one uint32 word sit in the low
                    # (even row) or high (odd row) 16 bits. Shift the wanted half
                    # into the float32 sign/exponent position and clear the rest.
                    is_even_row = jnp.bitwise_and(src_idx_slice[row_src], 1) == 0
                    shift = jnp.where(is_even_row, 16, 0)
                    lower_mask = jnp.uint32(jnp.iinfo(jnp.uint16).max)
                    upper_mask = jnp.left_shift(lower_mask, 16)
                    shifted = jnp.bitwise_and(
                        jnp.left_shift(val_u32, shift), upper_mask
                    )
                    data_f32 = plsc.bitcast(shifted, jnp.float32)
                else:
                    data_f32 = plsc.bitcast(val_u32, jnp.float32)
                data_f32 *= tw_slice[row_src]

                # Reduction: accumulate while the destination group is unchanged,
                # restart otherwise. Sorting guarantees rows of one group are
                # contiguous.
                dst_row_hbm = dst_slice[row_src]
                prev = prev_dst if row_src == 0 else dst_slice[row_src - 1]
                accumulated_data = jnp.where(
                    dst_row_hbm == prev,
                    previous_accumulated_data + data_f32,
                    data_f32,
                )
                previous_accumulated_data = accumulated_data

                # The output buffer stays float32: a bfloat16 output would be
                # (16, 128)-tiled and the row scatter below writes single rows at
                # arbitrary, non-tile-aligned destinations, which is only legal for
                # 32-bit elements. The cast happens in the wrapper.
                out_vmem[row_src, col_slice] = accumulated_data
                if row_src == num_simd_lanes - 1:
                    refs.scratch.prev_iter_last_row_vmem[c, col_slice] = (
                        accumulated_data
                    )
