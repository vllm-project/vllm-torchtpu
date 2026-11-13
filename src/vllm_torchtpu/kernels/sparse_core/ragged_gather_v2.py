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
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper
from vllm_torchtpu.kernels.sparse_core.support import get_sparse_core_info


def calculate_col_size(hidden_size: int, packing: int) -> int:
    """Calculates the max column size bounded by VMEM limits and
    hidden_size divisibility."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    lanes = sc_info.num_lanes

    # v7 packed (bf16) gathers are validated up to 94% of SparseCore VMEM.
    vmem_fraction = 0.94 if (tpu_info.generation == 7 and packing == 2) else 0.8
    target_bytes = sc_info.vmem_capacity_bytes * vmem_fraction

    # Calculate max safe column size based on VMEM budget and aligne to 128.
    num_buffers = 2
    bytes_per_col = (lanes + lanes // packing) * 4 * num_buffers
    max_safe_col = int((target_bytes // bytes_per_col) // 128) * 128

    # Search for the largest divisor of hidden_size bounded by max_safe_col.
    # The first divisor found is the maximum.
    start_col = (min(hidden_size, max_safe_col) // 128) * 128
    for c in range(start_col, 127, -128):
        if hidden_size % c == 0:
            return c
    return max_safe_col


def main_kernel_v2(
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    idx_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int = 1,
):
    del num_row_subchunks
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]
    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    assert isinstance(hidden_size, int), (
        f"hidden_size must be int, got {type(hidden_size)}"
    )
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes

    recv_sem = sem_ref.at[0]

    copy_start = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1], recv_sem)
    copy_end = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], recv_sem)
    copy_start.start()
    copy_end.start()
    copy_start.wait()
    copy_end.wait()

    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    subchunk_start = start // row_subchunk_size
    subchunk_end = pl.cdiv(end, row_subchunk_size)
    total_subchunks = jnp.maximum(0, subchunk_end - subchunk_start)

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    # Partition active 16-row subchunks evenly across all subcores with at most
    # 1 subchunk difference between any two subcores, assigning each subcore a
    # contiguous range [my_start, my_start + my_count).
    q = total_subchunks // num_cores
    rem = total_subchunks - q * num_cores
    my_count = q + jnp.where(core_index < rem, 1, 0)
    my_start = subchunk_start + core_index * q + jnp.minimum(core_index, rem)

    num_cols = pl.cdiv(hidden_size, col_size)

    # SparseCore `.bitcast()` leverages hardware Row-Packing for
    # 16-bit -> 32-bit conversion.
    # The logical row count halves, while physical column dimensions remain unchanged.
    in_hbm_i32 = in_hbm_ref.bitcast(jnp.int32)
    out_hbm_i32 = out_hbm_ref.bitcast(jnp.int32)

    num_phys_cols = col_size
    unpack_col_chunk = 256 if (packing <= 2 and num_phys_cols % 256 == 0) else 128
    shift_amount = packing.bit_length() - 1

    def col_loop(col_base, gather_ref, out_ref, idx_rem, shift_lo, shift_hi):
        col_slice = pl.ds(col_base, unpack_col_chunk)
        if packing == 1:
            gather_dt = gather_ref.bitcast(dtype)
            out_dt = out_ref.bitcast(dtype)
            out_dt[:, col_slice] = gather_dt[:, col_slice]
        elif packing == 2:
            for i in range(num_simd_lanes // 2):
                lo = jnp.bitwise_and(
                    jnp.bitwise_right_shift(
                        gather_ref[pl.ds(2 * i, 1), col_slice], shift_lo[i]
                    ),
                    0xFFFF,
                )
                hi = jnp.bitwise_and(
                    jnp.left_shift(
                        gather_ref[pl.ds(2 * i + 1, 1), col_slice], shift_hi[i]
                    ),
                    jnp.int32(-65536),
                )
                out_ref[pl.ds(i, 1), col_slice] = jnp.bitwise_or(lo, hi)
        else:
            # Manual bitwise extraction and packing for
            # packing >= 4 (int8, int4)
            mask = (1 << dtype_bits) - 1
            shift_multiplier = dtype_bits.bit_length() - 1
            for i in range(num_simd_lanes // packing):
                packed_row = jnp.zeros((1, unpack_col_chunk), dtype=jnp.int32)
                for j in range(packing):
                    k = i * packing + j
                    dynamic_shift = jnp.left_shift(idx_rem[k], shift_multiplier)
                    val = jnp.bitwise_right_shift(
                        gather_ref[pl.ds(k, 1), col_slice], dynamic_shift
                    )
                    val = jnp.bitwise_and(val, mask)
                    pack_shift = j * dtype_bits
                    packed_row = jnp.bitwise_or(
                        packed_row, jnp.left_shift(val, pack_shift)
                    )
                out_ref[pl.ds(i, 1), col_slice] = packed_row

    def inner_pipeline(gather_ref, out_ref):
        r = pl.program_id(0)
        row_slice = pl.ds(r * row_subchunk_size, row_subchunk_size)
        subchunk_idxs = idx_vmem_ref[row_slice]
        if packing == 2:
            idx_rem = jnp.bitwise_and(subchunk_idxs, 1)
            shift_lo = [
                jnp.left_shift(idx_rem[2 * i], 4)
                for i in range(num_simd_lanes // 2)
            ]
            shift_hi = [
                jnp.left_shift(1 - idx_rem[2 * i + 1], 4)
                for i in range(num_simd_lanes // 2)
            ]
        elif packing > 1:
            # Equivalent to `subchunk_idxs % packing`
            idx_rem = jnp.bitwise_and(subchunk_idxs, packing - 1)
            shift_lo = None
            shift_hi = None
        else:
            idx_rem = jnp.zeros_like(subchunk_idxs)
            shift_lo = None
            shift_hi = None

        col_loop_fn = functools.partial(
            col_loop,
            gather_ref=gather_ref,
            out_ref=out_ref,
            idx_rem=idx_rem,
            shift_lo=shift_lo,
            shift_hi=shift_hi,
        )
        plsc.parallel_loop(0, num_phys_cols, step=unpack_col_chunk)(col_loop_fn)

    max_vmem_subchunks = idx_vmem_ref.shape[0] // row_subchunk_size

    def run_chunk(chunk_start, chunk_count):
        dma_offset = pl.multiple_of(
            chunk_start * row_subchunk_size, row_subchunk_size
        )
        dma_len = pl.multiple_of(
            chunk_count * row_subchunk_size, row_subchunk_size
        )
        copy_idx = pltpu.make_async_copy(
            indices_hbm_ref.at[pl.ds(dma_offset, dma_len)],
            idx_vmem_ref.at[pl.ds(0, dma_len)],
            recv_sem,
        )
        copy_idx.start()
        copy_idx.wait()

        pltpu.emit_pipeline(
            inner_pipeline,
            grid=(chunk_count, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), num_phys_cols),
                lambda r, col_id: (
                    jnp.bitwise_right_shift(
                        idx_vmem_ref[
                            pl.ds(r * row_subchunk_size, row_subchunk_size)
                        ],
                        shift_amount,
                    ),
                    col_id,
                ),
            ),
            out_specs=pl.BlockSpec(
                (row_subchunk_size // packing, num_phys_cols),
                lambda r, col_id: (
                    chunk_start + r,
                    col_id,
                ),
            ),
        )(in_hbm_i32, out_hbm_i32)

    @pl.when(my_count > 0)
    def _():
        @pl.when(my_count <= max_vmem_subchunks)
        def _():
            run_chunk(my_start, my_count)

        @pl.when(my_count > max_vmem_subchunks)
        def _():
            num_outer = pl.cdiv(my_count, max_vmem_subchunks)

            def body(step, _):
                c_start = my_start + step * max_vmem_subchunks
                c_count = jnp.minimum(
                    max_vmem_subchunks, my_count - step * max_vmem_subchunks
                )
                run_chunk(c_start, c_count)
                return None

            lax.fori_loop(0, num_outer, body, None)


@jax.jit
def ragged_gather_v2(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """Perform gather on indices within dynamic array start and end using BlockSpec."""

    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4):
        raise ValueError(f"dtype must be f32, bf16, int8, or int4, but got {dtype}")

    sc_info = get_sparse_core_info()
    if sc_info is None:
        return x[indices]

    hidden_size = x.shape[-1]
    out_size = indices.size

    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    row_subchunk_size = num_simd_lanes

    out_pad_size = (
        (out_size + row_subchunk_size - 1) // row_subchunk_size
    ) * row_subchunk_size - out_size
    if out_pad_size > 0:
        indices = jnp.pad(indices, ((0, out_pad_size)))

    total_subchunks = (out_size + out_pad_size) // row_subchunk_size
    max_subchunks_per_core = max(
        1, min(128, (total_subchunks + num_cores - 1) // num_cores)
    )
    idx_vmem_size = max_subchunks_per_core * row_subchunk_size

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out = core_map_helper.kernel(
        functools.partial(
            main_kernel_v2,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
        ),
        out_type=jax.ShapeDtypeStruct(
            (out_size + out_pad_size, aligned_hidden_size), dtype
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=[
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((idx_vmem_size,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        mesh=vector_mesh,
        name="sc_ragged_gather_v2",
    )(start, end, x, indices)
    if out_pad_size == 0 and aligned_hidden_size == hidden_size:
        return out
    return out[:out_size, :hidden_size]
