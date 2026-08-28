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

TILE_SUBROWS = 4
TILE_LANE_BYTES = 128


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
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
    rope_out_cols = rope_out_hbm_ref.shape[1]
    rope_rows_per_stream = row_subchunk_size // rope_out_packing

    def process_rope(gather_ref, out_ref, idx_sub, out_row_base=0):
        # gather_ref: (16, 128) int32
        # out_ref: (16, 128) int32 (tiled block)
        # idx_sub: (16,) int32 (token indices)
        for rg in range(rope_rows_per_stream):
            packed = jnp.zeros((1, rope_out_cols), dtype=jnp.int32)
            for sub in range(rope_out_packing):
                k = rg * rope_out_packing + sub
                sub_in = lax.rem(idx_sub[k], in_packing)
                data = gather_ref[pl.ds(k, 1), :]
                val = jnp.bitwise_and(
                    jnp.bitwise_right_shift(data, 8 * sub_in), in_mask)
                packed = jnp.bitwise_or(packed, jnp.left_shift(val, 8 * sub))
            out_ref[pl.ds(out_row_base + rg, 1), :] = packed

    def outer_pipeline(idx_ref):
        b = pl.program_id(0)
        out_row_base = (b * num_cores + core_index) * num_row_subchunks

        # Subchunk handled by stream `s` at inner step `r`. `num_streams`
        # independent `pl.Indirect` gathers run concurrently per step, keeping
        # several gather DMAs in flight to raise effective read bandwidth.
        def subchunk(r, s):
            return r * num_streams + s

        def idx_window(r, s):
            return idx_ref[pl.ds(
                subchunk(r, s) * row_subchunk_size, row_subchunk_size)]

        def _body(nope_sem, nope_out_i32, *refs):
            r = pl.program_id(0)
            nope_g = refs[0 * num_streams:1 * num_streams]
            rope_g = refs[1 * num_streams:2 * num_streams]
            rope_o = refs[2 * num_streams]

            # nope needs no vector work at all. The nope output keeps the cache's raw
            # per-token layout. The gathered row *is* the output row.
            #
            # DMA the gather buffer straight to HBM rather than routing it through
            # a pipeline output buffer: staging it there costs a VMEM->VMEM copy.
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
                    out_row_base=s * rope_rows_per_stream,
                )

            # Wait for all nope DMAs to complete.
            for copy in nope_copies:
                copy.wait()

        # Have multiple parallel `pl.Indirect` to hide random access read latency.
        # Output contiguous memory access, multiple output parallel DMAs not help
        # with performance.

        # nope: gather int32 row == index (1 int32 row per entry).
        nope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), nope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            ) for s in range(num_streams))
        # rope: gather int32 row == index (1 int32 row per entry).
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda r, s=s: (lax.div(idx_window(r, s), in_packing), 0),
            ) for s in range(num_streams))

        # One merged rope output block, covering all `num_streams` subchunks.
        #
        # nope has no pipeline output block -- `_body` DMAs it to HBM directly (
        # based on microbenchmarks, this approach has better performance than
        # having a pipeline output buffer).
        rope_out_spec = pl.BlockSpec(
            (num_streams * rope_rows_per_stream, rope_out_i32.shape[1]),
            lambda r: (out_row_base // num_streams + r, 0),
        )
        pltpu.emit_pipeline(
            functools.partial(_body, nope_sem, nope_out_i32),
            grid=(num_row_subchunks // num_streams, ),
            in_specs=nope_in_specs + rope_in_specs,
            out_specs=(rope_out_spec, ),
        )(
            *([nope_in_i32] * num_streams),
            *([rope_in_i32] * num_streams),
            rope_out_i32,
        )

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks, ),
        in_specs=pl.BlockSpec(
            (row_chunk_size, ),
            lambda b: (b * num_cores + core_index, ),
        ),
    )(indices_hbm_ref)


@functools.partial(jax.jit)
def dsa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

  Args:
    nope_cache: (total_pages, page_size, TILE_SUBROWS, TILE_LANE_BYTES) uint8.
      Each (TILE_SUBROWS, TILE_LANE_BYTES) uint8 is token's nope. It encodes 512
      fp8 values (per-tensor k_scale applied by the attention kernel; no inline
      scales).
    rope_cache: (total_pages, page_size // TILE_SUBROWS, TILE_SUBROWS,
      TILE_LANE_BYTES) uint8. Each (1, TILE_LANE_BYTES) uint8 is token's rope.
      It encodes 64 bf16.
    indices: (N,) int32. Token indices into the caches.

  Returns:
    nope_out: (N, TILE_SUBROWS, TILE_LANE_BYTES) uint8. Each token's nope tile
      is copied out in the cache's raw layout.
    rope_out: (N, 64) bf16.
      Each (64) bf16 is token's rope.
  """
    assert indices.ndim == 1, "Indices must be 1D."
    assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
    assert nope_cache.dtype == jnp.uint8, "Caches must be uint8."
    assert nope_cache.shape[2:] == (TILE_SUBROWS, TILE_LANE_BYTES), (
        f"nope_cache must be tiled (..., {TILE_SUBROWS}, {TILE_LANE_BYTES}), "
        f"got {nope_cache.shape}")
    assert rope_cache.shape[2:] == (TILE_SUBROWS, TILE_LANE_BYTES), (
        f"rope_cache must be tiled (..., {TILE_SUBROWS}, {TILE_LANE_BYTES}), "
        f"got {rope_cache.shape}")
    rope_out_cols = rope_cache.shape[3]

    # Flatten both caches to 128-wide rows and view as raw bytes.
    nope_cache = nope_cache.reshape(-1, nope_cache.shape[3])
    rope_cache = rope_cache.reshape(-1, rope_cache.shape[3])
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    out_size = indices.size
    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores

    # `num_streams` independent `pl.Indirect` gathers are issued per
    # pipeline step to keep multiple gather DMAs in flight.
    # See `outer_pipeline` for details.
    num_streams = 4
    num_row_subchunks = 32
    assert (
        num_row_subchunks %
        num_streams == 0), f"{num_streams=} must divide {num_row_subchunks=}."
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    out_pad_size = (
        (out_size + block_size - 1) // block_size) * block_size - out_size
    indices = jnp.pad(indices, ((0, out_pad_size)))
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
                ((out_size + out_pad_size) * TILE_SUBROWS, TILE_LANE_BYTES),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct((out_size + out_pad_size, rope_out_cols),
                                 jnp.uint8),
        ),
        # One DMA semaphore per stream for the direct nope gather-buffer -> HBM
        # copies issued in `main_kernel`.
        scratch_types=(pltpu.SemaphoreType.DMA((num_streams, )), ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name="sc_dsa_gather",
    )(nope_cache, rope_cache, indices)
    return (
        nope_out.reshape(-1, TILE_SUBROWS, TILE_LANE_BYTES)[:out_size],
        rope_out[:out_size],
    )
