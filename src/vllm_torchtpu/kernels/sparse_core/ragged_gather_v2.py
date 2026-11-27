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

    target_bytes = sc_info.vmem_capacity_bytes * 0.8

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


# Columns one inner-loop step unpacks.
_UNPACK_COL_CHUNK = 128


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
):
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None
    lanes = sc_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]
    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)
    num_cols = pl.cdiv(hidden_size, col_size)
    assert col_size % _UNPACK_COL_CHUNK == 0

    copy_start = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1], sem_ref.at[0])
    copy_end = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], sem_ref.at[0])
    copy_start.start()
    copy_end.start()
    copy_start.wait()
    copy_end.wait()
    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    # Rows [start, end) go in sub-chunks of `lanes` rows (one index vector),
    # split over all vector subcores as contiguous runs whose lengths differ by
    # at most one. A subcore's sub-chunks run one after another, so the call
    # costs the longest run and follows `end` in steps of lanes * subcores rows.
    num_workers = lax.axis_size((core_axis_name, subcore_axis_name))
    worker = lax.axis_index((core_axis_name, subcore_axis_name))
    first_sub = start // lanes
    last_sub = jnp.minimum(pl.cdiv(end, lanes), out_hbm_ref.shape[0] // lanes)
    num_sub = jnp.where(end <= start, 0, jnp.maximum(last_sub - first_sub, 0))
    base, rem = num_sub // num_workers, num_sub % num_workers
    count = base + (worker < rem).astype(jnp.int32)
    my_first = first_sub + worker * base + jnp.minimum(worker, rem)

    # SparseCore `.bitcast()` leverages hardware Row-Packing for
    # 16-bit -> 32-bit conversion.
    # The logical row count halves, while physical column dimensions remain unchanged.
    in_hbm_i32 = in_hbm_ref.bitcast(jnp.int32)
    out_hbm_i32 = out_hbm_ref.bitcast(jnp.int32)
    shift_amount = packing.bit_length() - 1

    def col_loop(col_base, gather_ref, out_ref, idx_rem):
        col_slice = pl.ds(col_base, _UNPACK_COL_CHUNK)
        if packing == 1:
            gather_dt = gather_ref.bitcast(dtype)
            out_dt = out_ref.bitcast(dtype)
            out_dt[:, col_slice] = gather_dt[:, col_slice]
        else:
            # Manual bitwise extraction and packing for
            # packing >= 2 (bfloat16, int8, int4)
            # bf16: 0xFFFF, int8: 0xFF, int4: 0xF
            mask = (1 << dtype_bits) - 1
            shift_multiplier = dtype_bits.bit_length() - 1
            for i in range(lanes // packing):
                packed_row = jnp.zeros((1, _UNPACK_COL_CHUNK), dtype=jnp.int32)
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

    def body(gather_ref, out_ref):
        subchunk_idxs = idx_vmem_ref[pl.ds(pl.program_id(0) * lanes, lanes)]
        # Equivalent to `subchunk_idxs % packing`
        idx_rem = jnp.bitwise_and(subchunk_idxs, packing - 1)
        plsc.parallel_loop(0, col_size, step=_UNPACK_COL_CHUNK)(
            functools.partial(
                col_loop, gather_ref=gather_ref, out_ref=out_ref, idx_rem=idx_rem
            )
        )

    @pl.when(count > 0)
    def _():
        idx_copy = pltpu.make_async_copy(
            indices_hbm_ref.at[
                pl.ds(pl.multiple_of(my_first * lanes, lanes), count * lanes)
            ],
            idx_vmem_ref.at[pl.ds(0, count * lanes)],
            sem_ref.at[0],
        )
        idx_copy.start()
        idx_copy.wait()
        pltpu.emit_pipeline(
            body,
            grid=(count, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(lanes), col_size),
                lambda r, c: (
                    jnp.bitwise_right_shift(
                        idx_vmem_ref[pl.ds(r * lanes, lanes)], shift_amount
                    ),
                    c,
                ),
            ),
            out_specs=pl.BlockSpec(
                (lanes // packing, col_size), lambda r, c: (my_first + r, c)
            ),
        )(in_hbm_i32, out_hbm_i32)


@jax.jit
def ragged_gather_v2(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """Row ``i`` of the result is ``x[indices[i]]`` for ``i`` in ``[start, end)``.

    Rows outside ``[start, end)``, rounded out to multiples of the SparseCore
    lane count, are not written.
    """
    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
    # The kernel moves bits: e4m3 rows move as int8 ones do.
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4, jnp.float8_e4m3fn):
        raise ValueError(
            f"dtype must be f32, bf16, int8, int4 or e4m3, but got {dtype}"
        )

    sc_info = get_sparse_core_info()
    if sc_info is None:
        return x[indices]

    hidden_size = x.shape[-1]
    out_size = indices.size

    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    col_size = calculate_col_size(hidden_size, packing)

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    lanes = sc_info.num_lanes
    num_workers = sc_info.num_cores * sc_info.num_subcores
    padded_size = pl.cdiv(out_size, lanes) * lanes
    indices = jnp.pad(indices, (0, padded_size - out_size))
    max_run = pl.cdiv(padded_size // lanes, num_workers)
    # Words of subcore VMEM: the double-buffered row pipeline, the run's
    # indices and the two bound vectors.
    vmem_words = 2 * (lanes + lanes // packing) * col_size + max_run * lanes + 32
    if 4 * vmem_words > sc_info.vmem_capacity_bytes:
        raise ValueError(
            f"{out_size} indices need {4 * vmem_words} B of SparseCore VMEM, "
            f"over its {sc_info.vmem_capacity_bytes} B"
        )

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return core_map_helper.kernel(
        functools.partial(
            main_kernel_v2,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
        ),
        out_type=jax.ShapeDtypeStruct((padded_size, aligned_hidden_size), dtype),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=[
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((max_run * lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        mesh=vector_mesh,
        name="sc_ragged_gather_v2",
    )(start, end, x, indices)[:out_size, :hidden_size]
