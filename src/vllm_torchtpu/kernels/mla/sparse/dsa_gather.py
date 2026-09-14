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

from vllm_torchtpu.kernels.mla import kv_cache_utils


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
    skip_padding_windows: bool = False,
    skip_mode: str = "when",
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
    # nope and rope output stays uint8 (4/int32).
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

        # `SKIP_INDEX` in slot 0 means "every index in this window is padding".
        # The caller only sets it for a whole window at once (see
        # `mark_skippable_windows`), so slot 0 decides for all `row_chunk_size`
        # of them and the flag needs no input of its own -- it rides in the
        # index stream that is being DMA'd here regardless.
        #
        # Skipping leaves this window's `nope_out`/`rope_out` rows untouched.
        # That is safe because every consumer already discards them: the
        # attention kernel zeroes each gathered row past its token's `kv_len`
        # (`kernel.py`, `load_bkv`), precisely because the padding tail could
        # always decode to fp8 NaN. Untouched HBM is no worse than the
        # scattered garbage that tail held before.

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
        num_inner_steps = num_row_subchunks // num_streams

        def run_gather_pipeline(steps=num_inner_steps):
            pltpu.emit_pipeline(
                functools.partial(_body, nope_sem, nope_out_i32),
                grid=(steps, ),
                in_specs=nope_in_specs + rope_in_specs,
                out_specs=(rope_out_spec, ),
            )(
                *([nope_in_i32] * num_streams),
                *([rope_in_i32] * num_streams),
                rope_out_i32,
            )

        # The guard must sit *outside* `emit_pipeline` either way: the gather
        # reads are issued by the pipeline's own `pl.Indirect` prefetch, so a
        # predicate inside `_body` would suppress the stores and still pay for
        # every read.
        #
        # Getting the predicate at all takes a detour. `idx_ref` is a VMEM
        # block, and on SparseCore a rank-0 load is legal only out of SMEM --
        # `sc_lowering.py` guards get and swap with the same XOR, so `idx_ref[0]`
        # raises "Get only supports loading scalars from SMEM". The emulation
        # below is the one the compiler's own error text prescribes: load a
        # vector, then extract lane 0.
        #
        # Extract *then* compare, never `(v >= 0)[0]`: the rank-0 squeeze path
        # only accepts 32-bit element types, and a bool is one byte.
        first = idx_ref[pl.ds(0, row_subchunk_size)][0]

        # Two ways to spend that predicate. Neither is lit upstream -- no test
        # anywhere wraps `emit_pipeline` in a `pl.when`, and while a traced grid
        # length is supported and tested, it is tested only on TensorCore. They
        # fail differently, so both are kept until hardware says which lowers.
        if not skip_padding_windows:
            run_gather_pipeline()
        elif skip_mode == "when":
            pl.when(first >= 0)(run_gather_pipeline)
        elif skip_mode == "grid":
            # A zero-length pipeline runs no steps and issues no DMAs: the same
            # statement as `pl.when`, without nesting the buffer and semaphore
            # allocation inside an `scf.if` region.
            run_gather_pipeline(jnp.where(first >= 0, num_inner_steps, 0))
        else:
            raise ValueError(f"unknown {skip_mode=}")

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks, ),
        in_specs=pl.BlockSpec(
            (row_chunk_size, ),
            lambda b: (b * num_cores + core_index, ),
        ),
    )(indices_hbm_ref)


# An index of `SKIP_INDEX` in the first slot of a gather window tells the kernel
# that the whole window is padding and may be skipped. It is never dereferenced.
SKIP_INDEX = -1

_NUM_STREAMS = 4
_NUM_ROW_SUBCHUNKS = 32


def get_row_chunk_size() -> int:
    """Indices handled by one SparseCore core per outer pipeline step.

  This is the granularity at which `skip_padding_windows` can skip work, so a
  caller marking windows with `SKIP_INDEX` must align its marks to it.
  """
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    return sc_info.num_lanes * _NUM_ROW_SUBCHUNKS


@functools.partial(jax.jit,
                   static_argnames=("skip_padding_windows", "skip_mode"))
def dsa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    *,
    skip_padding_windows: bool = False,
    skip_mode: str = "when",
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

  Args:
    nope_cache: (total_pages, page_size, WORD_BYTES, TILE_LANE_BYTES) uint8.
      Each (WORD_BYTES, TILE_LANE_BYTES) uint8 is token's nope. It encodes 512
      fp8 values (per-tensor k_scale applied by the attention kernel; no inline
      scales).
    rope_cache: (total_pages, page_size // WORD_BYTES, WORD_BYTES,
      TILE_LANE_BYTES) uint8. Each (1, TILE_LANE_BYTES) uint8 is token's rope.
      It encodes 128 (rope_out_cols) fp8 values.
    indices: (N,) int32. Token indices into the caches.
    skip_padding_windows: when set, a window of `get_row_chunk_size()` indices
      whose *first* entry is `SKIP_INDEX` is not gathered at all, and the
      corresponding output rows are left untouched. Callers use this to avoid
      moving rows that the consumer is going to mask away anyway; see
      `kernel.mark_skippable_windows`.
    skip_mode: how that skip is expressed -- "when" wraps the inner pipeline in
      `pl.when`, "grid" gives it a traced grid length of 0. Same semantics; they
      exist separately because the two lower differently.

  Returns:
    nope_out: (N, WORD_BYTES, TILE_LANE_BYTES) uint8. Each token's nope tile
      is copied out in the cache's raw layout.
    rope_out: (N, 128) fp8.
      (64) fp8 is token's rope, the rest 64 are padding.
  """
    word_bytes = kv_cache_utils.WORD_BYTES
    tile_lane_bytes = kv_cache_utils.TILE_LANE_BYTES
    assert indices.ndim == 1, "Indices must be 1D."
    assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
    assert nope_cache.dtype == jnp.uint8, "Caches must be uint8."
    assert nope_cache.shape[2:] == (word_bytes, tile_lane_bytes), (
        f"nope_cache must be tiled (..., {word_bytes}, {tile_lane_bytes}), "
        f"got {nope_cache.shape}")
    assert rope_cache.shape[2:] == (word_bytes, tile_lane_bytes), (
        f"rope_cache must be tiled (..., {word_bytes}, {tile_lane_bytes}), "
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
    num_streams = _NUM_STREAMS
    num_row_subchunks = _NUM_ROW_SUBCHUNKS
    assert (
        num_row_subchunks %
        num_streams == 0), f"{num_streams=} must divide {num_row_subchunks=}."
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    out_pad_size = (
        (out_size + block_size - 1) // block_size) * block_size - out_size
    if skip_padding_windows and out_pad_size:
        # Windows lying wholly past `out_size` are marked skippable -- today
        # they gather index 0 `out_pad_size` times, which is the worst possible
        # read-concentration pattern. A window straddling `out_size` still has a
        # real first entry, so its tail keeps a legal (index 0) address.
        pad_pos = out_size + jnp.arange(out_pad_size, dtype=jnp.int32)
        window_start = (pad_pos // row_chunk_size) * row_chunk_size
        indices = jnp.concatenate([
            indices,
            jnp.where(window_start >= out_size, SKIP_INDEX, 0),
        ])
    else:
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
            skip_padding_windows=skip_padding_windows,
            skip_mode=skip_mode,
        ),
        out_type=(
            jax.ShapeDtypeStruct(
                ((out_size + out_pad_size) * word_bytes, tile_lane_bytes),
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
        nope_out.reshape(-1, word_bytes, tile_lane_bytes)[:out_size],
        rope_out[:out_size],
    )
