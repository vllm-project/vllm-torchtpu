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
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    valid_indices_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    nope_sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
):
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)

    # Inputs are 8-bit;
    # nope output stays uint8 (4/int32), rope is unpacked to bf16 (2/int32).
    in_bits = jax.dtypes.itemsize_bits(nope_in_hbm_ref.dtype)
    in_packing = 32 // in_bits
    in_mask = (1 << in_bits) - 1  # 0xFF for 8-bit.
    rope_out_bits = jax.dtypes.itemsize_bits(rope_out_hbm_ref.dtype)
    rope_out_packing = 32 // rope_out_bits
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    # SparseCore gather 32-bit words
    nope_in_i32 = nope_in_hbm_ref.bitcast(jnp.int32)
    rope_in_i32 = rope_in_hbm_ref.bitcast(jnp.int32)
    nope_out_i32 = nope_out_hbm_ref.bitcast(jnp.int32)
    rope_out_i32 = rope_out_hbm_ref.bitcast(jnp.int32)

    nope_in_cols = nope_in_i32.shape[1]
    rope_in_cols = rope_in_i32.shape[1]
    # A `bf16[n, 64]` output is tiled T(8,128), so XLA pads it to 128 lanes and
    # it costs twice its own bytes -- on this kernel's write and on the
    # consumer's read. The output is 128 lanes wide instead, pairing entry `i`
    # of each `top_k` with entry `i + top_k // 2`, which the
    # consumer splits apart with a lane slice and a row concatenate.
    rope_out_cols = rope_out_hbm_ref.shape[1] // 2
    half_streams = num_streams // 2

    def process_rope(gather_ref, out_ref, idx_sub, out_row_base=0, out_col_base=0):
        # one (1, 128) uint8 is one token's rope data, which encodes 64 bf16
        # values. (0, 64) are the high bits for bf16 data, (64, 128) are the low.
        half = rope_out_cols
        col_hi = pl.ds(0, half)
        col_lo = pl.ds(half, half)

        def bf16_bits(k):
            sub = lax.rem(idx_sub[k], in_packing)
            hi = jnp.bitwise_and(
                jnp.bitwise_right_shift(gather_ref[pl.ds(k, 1), col_hi], in_bits * sub),
                in_mask,
            )
            lo = jnp.bitwise_and(
                jnp.bitwise_right_shift(gather_ref[pl.ds(k, 1), col_lo], in_bits * sub),
                in_mask,
            )
            return jnp.bitwise_or(jnp.left_shift(hi, in_bits), lo)

        for t in range(num_simd_lanes // rope_out_packing):
            packed = jnp.zeros((1, half), dtype=jnp.int32)
            for pk in range(rope_out_packing):
                k = t * rope_out_packing + pk
                packed = jnp.bitwise_or(
                    packed, jnp.left_shift(bf16_bits(k), pk * rope_out_bits)
                )
            out_ref[pl.ds(out_row_base + t, 1), pl.ds(out_col_base, half)] = packed

    def outer_pipeline(idx_ref, valid_ref):
        valid_indices = valid_ref[pl.ds(0, row_subchunk_size)][0]
        b = pl.program_id(0)
        out_row_base = (b * num_cores + core_index) * num_row_subchunks

        # Subchunk handled by stream `s` at inner step `r`. `num_streams`
        # independent `pl.Indirect` gathers run concurrently per step, keeping
        # several gather DMAs in flight to raise effective read bandwidth;
        # the first half of the streams serves the low half of the period and
        # the second half the high half.
        def subchunk(r, s):
            return (
                (s // half_streams) * (num_row_subchunks // 2)
                + r * half_streams
                + s % half_streams
            )

        def idx_window(r, s):
            return idx_ref[pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)]

        # int32 rows produced per stream in the rope output (2 tokens per row).
        rope_rows_per_stream = row_subchunk_size // rope_out_packing

        def _body(*refs):
            r = pl.program_id(0)
            nope_g = refs[0 * num_streams : 1 * num_streams]
            rope_g = refs[1 * num_streams : 2 * num_streams]
            rope_o = refs[2 * num_streams]

            # nope needs no vector work at all. The nope output keeps the
            # cache's raw per-token layout. The gathered row *is* the output
            # row.
            #
            # DMA the gather buffer straight to HBM rather than routing it
            # through a pipeline output buffer: staging it there costs a
            # VMEM->VMEM copy.
            nope_copies = []
            for s in range(num_streams):
                out_row = (out_row_base + subchunk(r, s)) * row_subchunk_size
                copy = pltpu.make_async_copy(
                    nope_g[s],
                    nope_out_i32.at[pl.ds(out_row, row_subchunk_size)],
                    nope_sem.at[s],
                )
                copy.start()
                nope_copies.append(copy)

            for s in range(num_streams):
                process_rope(
                    gather_ref=rope_g[s],
                    out_ref=rope_o,
                    idx_sub=idx_window(r, s),
                    out_row_base=(s % half_streams) * rope_rows_per_stream,
                    out_col_base=(s // half_streams) * rope_out_cols,
                )

            # Wait for all nope DMAs to complete.
            for copy in nope_copies:
                copy.wait()

        # Have multiple parallel `pl.Indirect` to hide random access read
        # latency. Output contiguous memory access, multiple output parallel
        # DMAs not help with performance.

        # nope: gather int32 row == index (1 int32 row per entry).
        nope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), nope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )
        # rope: gather int32 row == index // in_packing (in_packing entries/row).
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda r, s=s: (lax.div(idx_window(r, s), in_packing), 0),
            )
            for s in range(num_streams)
        )

        # One merged rope output block, covering all `num_streams` subchunks:
        # `half_streams` subchunks of the low half of the period in lanes
        # [0, rope_out_cols) and the matching high-half ones in
        # [rope_out_cols, 2 * rope_out_cols).
        rope_out_spec = pl.BlockSpec(
            (half_streams * rope_rows_per_stream, 2 * rope_out_cols),
            lambda r: (out_row_base // num_streams + r, 0),
        )
        core_start_idx = out_row_base * row_subchunk_size

        @pl.when(core_start_idx < valid_indices)
        def _run_gather():
            pltpu.emit_pipeline(
                _body,
                grid=(num_row_subchunks // num_streams,),
                in_specs=nope_in_specs + rope_in_specs,
                out_specs=(rope_out_spec,),
            )(
                *([nope_in_i32] * num_streams),
                *([rope_in_i32] * num_streams),
                rope_out_i32,
            )

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks,),
        in_specs=(
            pl.BlockSpec(
                (row_chunk_size,),
                lambda b: (b * num_cores + core_index,),
            ),
            pl.BlockSpec(
                (row_subchunk_size,),
                lambda b: (0,),
            ),
        ),
    )(indices_hbm_ref, valid_indices_ref)


@functools.partial(jax.jit, static_argnames=("top_k",))
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

    Args:
      nope_cache: (total_pages, page_size, 4, 128) uint8. Each (4, 128) uint8 is
        token's nope + nope scales. It encodes 448 fp8 + 7 e8m0 scales + padding.
      rope_cache: (total_pages, page_size // 4, 4, 128) uint8. Each (1, 128) uint8
        is token's rope. It encodes 64 bf16.
      indices: (N,) int32. Token indices into the caches.
      num_valid_indices: Optional (1,) or scalar int32. Number of valid indices to
        gather. Subcores assigned to indices beyond this count skip gathering.
      top_k: the consumer's row block (the attention kernel's top-k). `rope_out`
        pairs entry i of a period with entry i + top_k // 2.

    Returns:
      nope_out: (N, 4, 128) uint8.
        Each (4, 128) uint8 is token's nope. It will be flattened to (1, 512)
        downstream.
      rope_out: (N // 2, 128) bf16.
        Row `period * (top_k // 2) + i` holds entry i of that period in
        lanes 0:64 and entry i + top_k // 2 in lanes 64:128 -- 128 lanes
        so XLA does not pad the buffer to twice its size. The consumer restores
        (top_k, 64) with
        `jnp.concatenate([row[:, :64], row[:, 64:]], axis=0)`.
    """
    assert indices.ndim == 1, "Indices must be 1D."
    assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
    assert nope_cache.dtype == jnp.uint8, "Caches must be uint8."
    assert nope_cache.shape[2] == 4
    assert rope_cache.shape[3] == 128

    # Flatten both caches to 128-wide rows and view as raw bytes.
    nope_cache = nope_cache.reshape(-1, nope_cache.shape[3])
    rope_cache = rope_cache.reshape(-1, rope_cache.shape[3])
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    out_size = indices.size
    nope_subrows = 4
    nope_out_cols = 128
    rope_out_cols = 64
    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    row_subchunk_size = num_simd_lanes

    if num_valid_indices is None:
        valid_indices = jnp.full((row_subchunk_size,), out_size, dtype=jnp.int32)
    else:
        valid_indices = jnp.full(
            (row_subchunk_size,), num_valid_indices, dtype=jnp.int32
        )

    # `num_streams` independent `pl.Indirect` gathers are issued per
    # pipeline step to keep multiple gather DMAs in flight.
    # See `outer_pipeline` for details.
    num_streams = 4
    assert top_k % row_subchunk_size == 0, (
        f"{top_k=} must be a multiple of {row_subchunk_size=}."
    )
    num_row_subchunks = top_k // row_subchunk_size
    assert num_row_subchunks % num_streams == 0, (
        f"{num_streams=} must divide {num_row_subchunks=}."
    )
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    if out_pad_size:
        # spread the padding to avoid hotspots
        num_tokens = nope_cache.shape[0] // nope_subrows
        pad = (jnp.arange(out_pad_size, dtype=indices.dtype) * 104729) % num_tokens
        indices = jnp.concatenate([indices, pad])
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    nope_out, rope_out = pl.kernel(
        functools.partial(
            main_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
            num_streams=num_streams,
        ),
        out_type=(
            jax.ShapeDtypeStruct(
                ((out_size + out_pad_size) * nope_subrows, nope_out_cols),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct(
                ((out_size + out_pad_size) // 2, 2 * rope_out_cols), jnp.bfloat16
            ),
        ),
        # One DMA semaphore per stream for the direct nope gather-buffer -> HBM
        # copies issued in `main_kernel`.
        scratch_types=(pltpu.SemaphoreType.DMA((num_streams,)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name="sc_csa_gather",
    )(nope_cache, rope_cache, indices, valid_indices)
    return (
        nope_out.reshape(-1, nope_subrows, nope_out_cols)[:out_size],
        rope_out[: out_size // 2],
    )
