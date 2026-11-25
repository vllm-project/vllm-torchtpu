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


COPY_MODES = ("vector", "dma")


def max_safe_col_size(packing: int) -> int:
    """The largest 128-aligned column block whose double-buffered gather and
    output blocks fit the per-subcore VMEM target."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    lanes = sc_info.num_lanes

    match tpu_info.generation:
        case 6:
            target_bytes = (256 * 1024) * 0.8
        case 7:
            target_bytes = (512 * 1024) * 0.8
        case _:
            target_bytes = (128 * 1024) * 0.8

    # Calculate max safe column size based on VMEM budget and aligne to 128.
    num_buffers = 2
    bytes_per_col = (lanes + lanes // packing) * 4 * num_buffers
    return int((target_bytes // bytes_per_col) // 128) * 128


def calculate_col_size(hidden_size: int, packing: int) -> int:
    """Calculates the max column size bounded by VMEM limits and
    hidden_size divisibility."""
    max_safe_col = max_safe_col_size(packing)

    # Search for the largest divisor of hidden_size bounded by max_safe_col.
    # The first divisor found is the maximum.
    start_col = (min(hidden_size, max_safe_col) // 128) * 128
    for c in range(start_col, 127, -128):
        if hidden_size % c == 0:
            return c
    return max_safe_col


def gather_kernel_name(
    row_subchunks: int | None = None,
    col_size: int | None = None,
    copy_mode: str = "vector",
) -> str:
    """The kernel name: every diagnosis knob that is set appears in it, so the
    variant that ran can be read from the op names. With no knob set it is the
    served name, sc_ragged_gather_v2."""
    name = "sc_ragged_gather_v2"
    if row_subchunks is not None:
        name += f"-rs_{row_subchunks}"
    if col_size is not None:
        name += f"-cs_{col_size}"
    if copy_mode != "vector":
        name += f"-{copy_mode}"
    return name


def _check_knobs(
    hidden_size: int,
    packing: int,
    row_subchunks: int | None,
    col_size: int | None,
    copy_mode: str,
) -> None:
    if copy_mode not in COPY_MODES:
        raise ValueError(f"copy_mode must be one of {COPY_MODES}, got {copy_mode!r}")
    if row_subchunks is not None and (
        not isinstance(row_subchunks, int)
        or isinstance(row_subchunks, bool)
        or row_subchunks < 1
    ):
        raise ValueError(f"row_subchunks must be a positive int, got {row_subchunks!r}")
    if col_size is not None:
        if (
            not isinstance(col_size, int)
            or isinstance(col_size, bool)
            or col_size < 128
            or col_size % 128
        ):
            raise ValueError(
                f"col_size must be a positive multiple of 128, got {col_size!r}"
            )
        if hidden_size % col_size:
            raise ValueError(
                f"col_size {col_size} must divide the row width {hidden_size}"
            )
        if col_size > max_safe_col_size(packing):
            raise ValueError(
                f"col_size {col_size} exceeds the VMEM budget "
                f"({max_safe_col_size(packing)} columns)"
            )


def main_kernel_v2(
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    out_hbm_ref: jax.Ref,
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    col_size: int,
    copy_mode: str = "vector",
):
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    hidden_size = in_hbm_ref.shape[-1]

    assert isinstance(hidden_size, int), (
        f"hidden_size must be int, got {type(hidden_size)}"
    )
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    recv_sem = sem_ref.at[0]

    copy_start = pltpu.make_async_copy(start_ref, start_vmem_ref.at[:1], recv_sem)
    copy_end = pltpu.make_async_copy(end_ref, end_vmem_ref.at[:1], recv_sem)
    copy_start.start()
    copy_end.start()
    copy_start.wait()
    copy_end.wait()

    start = start_vmem_ref[...][0]
    end = end_vmem_ref[...][0]

    block_start = start // block_size
    block_end = pl.cdiv(end, block_size)
    num_blocks = block_end - block_start
    num_blocks = jnp.where(end <= start, 0, num_blocks)

    num_cols = pl.cdiv(hidden_size, col_size)

    dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    # SparseCore `.bitcast()` leverages hardware Row-Packing for
    # 16-bit -> 32-bit conversion.
    # The logical row count halves, while physical column dimensions remain unchanged.
    in_hbm_i32 = in_hbm_ref.bitcast(jnp.int32)
    out_hbm_i32 = out_hbm_ref.bitcast(jnp.int32)

    num_phys_cols = col_size

    # The outer pipeline runs on the Vector Core, hoisting the integer arithmetic
    # required to decode the row-packed indices. This prevents scalar-core instruction
    # starvation during the execution of the indirect `in_specs` block lambdas.

    # TODO(guoweij): The nested `emit_pipeline` design creates a pipeline bubble
    # (DMA wait) when the inner pipeline empties and restarts with new indices. This is
    # a known high-level API limitation, currently amortized by setting a large
    # num_row_subchunks and shouldn't be an issue for most use cases. We should still
    # monitor the bubble's impact on E2E performance and, if needed, manually
    # reimplement this using primitive operations to eliminate the bubble entirely.

    def col_loop(col_base, gather_ref, out_ref, idx_rem, unpack_col_chunk):
        col_slice = pl.ds(col_base, unpack_col_chunk)
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

    def inner_pipeline(gather_ref, out_ref, idx_ref, unpack_col_chunk):
        row_slice = pl.ds(pl.program_id(0) * row_subchunk_size, row_subchunk_size)
        subchunk_idxs = idx_ref[row_slice]
        if packing > 1:
            # Equivalent to `subchunk_idxs % packing`
            idx_rem = jnp.bitwise_and(subchunk_idxs, packing - 1)
        else:
            idx_rem = jnp.zeros_like(subchunk_idxs)

        col_loop_fn = functools.partial(
            col_loop,
            gather_ref=gather_ref,
            out_ref=out_ref,
            idx_rem=idx_rem,
            unpack_col_chunk=unpack_col_chunk,
        )
        plsc.parallel_loop(0, num_phys_cols, step=unpack_col_chunk)(col_loop_fn)

    def inner_dma(gather_ref, b_global):
        # copy_mode="dma" (diagnosis prototype): the gathered block, 16 rows of
        # the 32-bit view (each holding `packing` source rows), goes from VMEM to
        # the output by DMA, with no vector copy and no unpacking. Output row j
        # of the 32-bit view is x_32bit[indices[j] >> log2(packing)].
        r = pl.program_id(0)
        col_id = pl.program_id(1)
        row0 = (
            (b_global * num_cores + core_index) * num_row_subchunks + r
        ) * row_subchunk_size
        copy = pltpu.make_async_copy(
            gather_ref,
            out_hbm_i32.at[
                pl.ds(pl.multiple_of(row0, row_subchunk_size), row_subchunk_size),
                pl.ds(pl.multiple_of(col_id * num_phys_cols, 128), num_phys_cols),
            ],
            recv_sem,
        )
        copy.start()
        copy.wait()

    def outer_pipeline_dma(idx_ref):
        b_global = pl.program_id(0) + block_start
        shift_amount = packing.bit_length() - 1
        pltpu.emit_pipeline(
            functools.partial(inner_dma, b_global=b_global),
            grid=(num_row_subchunks, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), num_phys_cols),
                lambda r, col_id: (
                    jnp.bitwise_right_shift(
                        idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)],
                        shift_amount,
                    ),
                    col_id,
                ),
            ),
        )(in_hbm_i32)

    def outer_pipeline(idx_ref):
        b = pl.program_id(0)
        b_global = b + block_start

        unpack_col_chunk = 128
        assert num_phys_cols % unpack_col_chunk == 0
        shift_amount = packing.bit_length() - 1
        pltpu.emit_pipeline(
            functools.partial(
                inner_pipeline, idx_ref=idx_ref, unpack_col_chunk=unpack_col_chunk
            ),
            grid=(num_row_subchunks, num_cols),
            in_specs=pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), num_phys_cols),
                lambda r, col_id: (
                    jnp.bitwise_right_shift(
                        idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)],
                        shift_amount,
                    ),
                    col_id,
                ),
            ),
            out_specs=pl.BlockSpec(
                (row_subchunk_size // packing, num_phys_cols),
                lambda r, col_id: (
                    (b_global * num_cores + core_index) * num_row_subchunks + r,
                    col_id,
                ),
            ),
        )(in_hbm_i32, out_hbm_i32)

    pltpu.emit_pipeline(
        outer_pipeline_dma if copy_mode == "dma" else outer_pipeline,
        grid=(num_blocks,),
        in_specs=pl.BlockSpec(
            (row_chunk_size,),
            lambda b: ((b + block_start) * num_cores + core_index,),
        ),
    )(indices_hbm_ref)


@functools.partial(
    jax.jit, static_argnames=("trim_rows", "row_subchunks", "col_size", "copy_mode")
)
def ragged_gather_v2(
    x: jax.Array,
    indices: jax.Array,
    start: jax.Array,
    end: jax.Array,
    trim_rows: bool = True,
    *,
    row_subchunks: int | None = None,
    col_size: int | None = None,
    copy_mode: str = "vector",
) -> jax.Array:
    """Perform gather on indices within dynamic array start and end using BlockSpec.

    The kernel writes whole row blocks. ``trim_rows=False`` returns them as they
    are (rows past ``indices.size`` hold row ``x[0]``), which spares the copy that
    trimming a large output costs when the consumer never reads those rows.

    Diagnosis knobs (default off; each one that is set is named in the kernel
    name, see gather_kernel_name; with none set the kernel is unchanged):
      row_subchunks: 16-row subchunks per outer row block (default 1..4 by size).
      col_size: column block width (default calculate_col_size).
      copy_mode: "vector" (served) or "dma", a DMA-only prototype that writes
        each gathered VMEM block to the output with no vector copy. For a packed
        dtype (``packing = 32 // bits > 1``) it returns the gathered 32-bit rows
        unpacked: ``out[packing * j + h] == x[packing * (indices[j] // packing) + h]``,
        so ``x[indices[j]] == out[packing * j + indices[j] % packing]``.
    """

    assert x.ndim == 2, "Ragged gather only supports 2d inputs."
    assert indices.ndim == 1, "Ragged gather only supports 1d indices."

    if jnp.isscalar(start):
        start = start[None]
    if jnp.isscalar(end):
        end = end[None]

    dtype = x.dtype
    if dtype not in (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4):
        raise ValueError(f"dtype must be f32, bf16, int8, or int4, but got {dtype}")

    if copy_mode not in COPY_MODES:
        raise ValueError(f"copy_mode must be one of {COPY_MODES}, got {copy_mode!r}")
    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        if copy_mode != "vector":
            raise ValueError("copy_mode='dma' needs a SparseCore")
        return x[indices]

    hidden_size = x.shape[-1]
    out_size = indices.size

    dtype_bits = jax.dtypes.itemsize_bits(dtype)
    packing = 32 // dtype_bits
    _check_knobs(hidden_size, packing, row_subchunks, col_size, copy_mode)
    kernel_name = gather_kernel_name(row_subchunks, col_size, copy_mode)
    if col_size is None:
        col_size = calculate_col_size(hidden_size, packing)

    aligned_hidden_size = ((hidden_size + col_size - 1) // col_size) * col_size

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores

    # Calculate ideal num_row_subchunks to avoid too much padding overhead.
    num_row_subchunks = max(
        1, min(4, (out_size + base_block_size - 1) // base_block_size)
    )
    if row_subchunks is not None:
        num_row_subchunks = row_subchunks

    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    indices = jnp.pad(indices, ((0, out_pad_size)))
    # copy_mode="dma" writes every gathered 32-bit row whole: `packing` rows of
    # the output dtype per index.
    out_rows_mult = packing if copy_mode == "dma" else 1

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
            num_row_subchunks=num_row_subchunks,
            col_size=col_size,
            copy_mode=copy_mode,
        ),
        out_type=jax.ShapeDtypeStruct(
            (out_rows_mult * (out_size + out_pad_size), aligned_hidden_size), dtype
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=[
            pltpu.VMEM((16,), jnp.int32),
            pltpu.VMEM((16,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        mesh=vector_mesh,
        name=kernel_name,
    )(start, end, x, indices)
    if copy_mode == "dma":
        rows = out_rows_mult * (out_size if trim_rows else out_size + out_pad_size)
        return out[:rows, :hidden_size]
    return out[:out_size, :hidden_size] if trim_rows else out[:, :hidden_size]
