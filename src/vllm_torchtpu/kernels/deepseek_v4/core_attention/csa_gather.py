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
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout


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
    top_k: int,
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

    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    nope_in_cols = nope_in_hbm_ref.shape[1]
    rope_in_cols = rope_in_hbm_ref.shape[1]
    # A `bf16[n, 64]` output is tiled to 128 lanes, so XLA pads it and it
    # costs twice its own bytes -- on this kernel's write and on the
    # consumer's read. The output is 128 lanes wide instead, pairing entry `i`
    # of each `top_k`-entry period with entry `i + top_k // 2`, which the
    # consumer splits apart with a lane slice and a row concatenate.
    rope_out_cols = rope_out_hbm_ref.shape[1] // 2

    def process_rope(gather_ref, out_ref, out_row_base=0):
        # bf16 output rows 2t and 2t + 1 are the low and high halves of int32
        # row t, so each output word pairs one rope value of an even entry
        # with the same value of the next entry. A cache rope word k holds
        # values k (low) and k + 32 (high).
        for t in range(num_simd_lanes // 2):
            even = gather_ref[pl.ds(2 * t, 1), :]
            odd = gather_ref[pl.ds(2 * t + 1, 1), :]
            row = pl.ds(out_row_base + t, 1)
            out_ref[row, pl.ds(0, rope_in_cols)] = jnp.bitwise_or(
                jnp.bitwise_and(even, 0xFFFF), jnp.left_shift(odd, 16)
            )
            out_ref[row, pl.ds(rope_in_cols, rope_in_cols)] = jnp.bitwise_or(
                lax.shift_right_logical(even, 16),
                jnp.bitwise_and(odd, jnp.int32(-65536)),  # 0xFFFF0000
            )

    def outer_pipeline(idx_ref, valid_ref):
        valid_indices = valid_ref[pl.ds(0, row_subchunk_size)][0]
        b = pl.program_id(0)
        out_row_base = (b * num_cores + core_index) * num_row_subchunks

        # Subchunk handled by stream `s` at inner step `r`. `num_streams`
        # independent `pl.Indirect` gathers run concurrently per step, keeping
        # several gather DMAs in flight to raise effective read bandwidth. A
        # step's subchunks are consecutive, so they lie in one half of a
        # period.
        def subchunk(r, s):
            return r * num_streams + s

        def idx_window(r, s):
            return idx_ref[pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)]

        # int32 rows produced per stream in the rope output (2 tokens per row).
        rope_rows_per_stream = row_subchunk_size // 2

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
                    nope_out_hbm_ref.at[pl.ds(out_row, row_subchunk_size)],
                    nope_sem.at[s],
                )
                copy.start()
                nope_copies.append(copy)

            for s in range(num_streams):
                process_rope(
                    gather_ref=rope_g[s],
                    out_ref=rope_o,
                    out_row_base=s * rope_rows_per_stream,
                )

            # Wait for all nope DMAs to complete.
            for copy in nope_copies:
                copy.wait()

        # Multiple parallel `pl.Indirect` gathers hide random-access read
        # latency. The writes are contiguous and need no extra parallelism:
        # rope writes one block per step, and nope issues one DMA per stream
        # only because each stream has its own gather buffer.

        # nope: gather int32 row == index (1 int32 row per entry).
        nope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), nope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )
        # rope: gather the entry's own 32-word quarter row == index.
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )

        # One rope output block per step, covering all `num_streams`
        # subchunks: a step in the first half of a period fills int32 lanes
        # [0, rope_out_cols) of its rows, a step in the second half lanes
        # [rope_out_cols, 2 * rope_out_cols) of the same rows.
        steps_per_half = top_k // 2 // (num_streams * row_subchunk_size)

        def rope_out_block(r):
            half, step = divmod(out_row_base // num_streams + r, steps_per_half)
            return (half // 2) * steps_per_half + step, half % 2

        rope_out_spec = pl.BlockSpec(
            (num_streams * rope_rows_per_stream, rope_out_cols), rope_out_block
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
                *([nope_in_hbm_ref] * num_streams),
                *([rope_in_hbm_ref] * num_streams),
                rope_out_hbm_ref,
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


def _banked_kernel(
    nope_banks,
    rope_banks,
    indices_hbm,
    valid_hbm,
    nope_out,
    rope_out,
    nope_buf,
    rope_buf,
    rope_packed,
    read_sems,
    write_sems,
    *,
    core_axis_name,
    subcore_axis_name,
    num_row_subchunks,
    num_streams,
    top_k,
):
    # Complementary masked gathers write disjoint rows of the same VMEM
    # buffer. No dummy-bank reads or full-size select buffers reach HBM.
    # Double buffering overlaps the next gather with packing and output DMA.
    lanes = nope_buf.shape[2]
    cores = lax.axis_size((core_axis_name, subcore_axis_name))
    core = lax.axis_index((core_axis_name, subcore_axis_name))
    chunk = num_row_subchunks * lanes
    blocks = pl.cdiv(indices_hbm.shape[0], chunk * cores)
    steps = num_row_subchunks // num_streams
    split = nope_banks[0].shape[0]

    def outer(idx, valid):
        base = (pl.program_id(0) * cores + core) * chunk

        def gather(r, *, wait=False):
            slot = r % 2
            for stream in range(num_streams):
                row = (r * num_streams + stream) * lanes
                ids = idx[pl.ds(row, lanes)]
                for bank in range(2):
                    own = ids >= split if bank else ids < split
                    local = ids - split if bank else ids
                    indices = plsc.Indices(jnp.where(own, local, -1), ignored_value=-1)
                    for kind, (cache, buf) in enumerate(
                        zip(
                            (nope_banks[bank], rope_banks[bank]),
                            (nope_buf, rope_buf),
                            strict=True,
                        )
                    ):
                        copy = pltpu.make_async_copy(
                            cache.at[indices],
                            buf.at[slot, stream],
                            read_sems.at[slot, bank, stream, kind],
                        )
                        if wait:
                            copy.wait()
                        else:
                            copy.start()

        def step(r, _):
            slot = r % 2
            gather(r, wait=True)

            @pl.when(r + 1 < steps)
            def prefetch():
                gather(r + 1)

            copies = []
            for stream in range(num_streams):
                out_row = base + (r * num_streams + stream) * lanes
                copy = pltpu.make_async_copy(
                    nope_buf.at[slot, stream],
                    nope_out.at[pl.ds(out_row, lanes)],
                    write_sems.at[stream, 0],
                )
                copy.start()
                copies.append(copy)
                for t in range(lanes // 2):
                    even = rope_buf[slot, stream, 2 * t : 2 * t + 1, :]
                    odd = rope_buf[slot, stream, 2 * t + 1 : 2 * t + 2, :]
                    rope_packed[stream, t : t + 1, :32] = (even & 0xFFFF) | (odd << 16)
                    rope_packed[stream, t : t + 1, 32:] = lax.shift_right_logical(
                        even, 16
                    ) | (odd & jnp.int32(-65536))
                period, offset = divmod(out_row, top_k)
                half, offset = divmod(offset, top_k // 2)
                rope_row = period * (top_k // 4) + offset // 2
                copy = pltpu.make_async_copy(
                    rope_packed.at[stream],
                    rope_out.at[pl.ds(rope_row, lanes // 2), pl.ds(half * 64, 64)],
                    write_sems.at[stream, 1],
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

        @pl.when(base < valid[:lanes][0])
        def run():
            gather(jnp.int32(0))
            lax.fori_loop(0, steps, step, None)

    pltpu.emit_pipeline(
        outer,
        grid=(blocks,),
        in_specs=(
            pl.BlockSpec((chunk,), lambda b: (b * cores + core,)),
            pl.BlockSpec((lanes,), lambda b: (0,)),
        ),
    )(indices_hbm, valid_hbm)


@functools.partial(jax.jit, static_argnames=("top_k",))
def csa_gather(
    nope_cache: jax.Array | tuple[jax.Array, jax.Array],
    rope_cache: jax.Array | tuple[jax.Array, jax.Array],
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

    Args:
      nope_cache: (total_pages, page_size, 128) int32, one row per token: the
        (4, 128) uint8 nope slab (see `csa_cache_layout`). A pair of arrays
        denotes consecutive low/high banks; indices address their concatenation.
      rope_cache: (total_pages, page_size // 4, 128) int32, 32 words per token
        (see `csa_cache_layout`), with the same banks as nope_cache.
      indices: (N,) int32. Token indices into the caches; N a multiple of
        top_k.
      num_valid_indices: Optional (1,) or scalar int32. Number of valid indices to
        gather, a multiple of top_k. A subcore skips every chunk that
        starts at or past this count, leaving those output rows unwritten.
      top_k: the consumer's row block (the attention kernel's top-k), a
        multiple of 128. `rope_out` pairs entry i of a period with entry
        i + top_k // 2.

    Returns:
      Both outputs are int32 with 128 lanes, which XLA lays out byte-for-byte
      like the uint8 / bf16 arrays they encode, so the consumer views them with
      a free `ref.bitcast`.
      nope_out: (N, 128) int32. Row i is entry i's (4, 128) uint8 nope slab.
      rope_out: (N // 4, 128) int32, i.e. (N // 2, 128) bf16 once bitcast.
        For i < top_k // 2, bf16 row `period * (top_k // 2) + i` holds entry
        i of that period in lanes 0:64 and entry i + top_k // 2 in lanes
        64:128 -- 128 lanes so XLA does not pad the buffer to twice its size.
        The consumer restores a period's (top_k, 64) from its
        (top_k // 2, 128) block `blk` with
        `jnp.concatenate([blk[:, :64], blk[:, 64:]], axis=0)`.
    """
    assert indices.ndim == 1, "Indices must be 1D."
    banked = isinstance(nope_cache, tuple)
    assert banked == isinstance(rope_cache, tuple)
    assert not banked or len(nope_cache) == len(rope_cache) == 2
    nope_banks: tuple[jax.Array, ...] = (
        nope_cache if isinstance(nope_cache, tuple) else (nope_cache,)
    )
    rope_banks: tuple[jax.Array, ...] = (
        rope_cache if isinstance(rope_cache, tuple) else (rope_cache,)
    )
    for nope, rope in zip(nope_banks, rope_banks, strict=True):
        assert nope.dtype == rope.dtype == jnp.int32
        assert nope.shape[0] == rope.shape[0] > 0
        assert nope.shape[1] == rope.shape[1] * 4
        assert nope.shape[2] == rope.shape[2] == csa_cache_layout.ROW_WORDS
    nope_banks = tuple(b.reshape(-1, csa_cache_layout.ROW_WORDS) for b in nope_banks)
    rope_banks = tuple(b.reshape(-1, csa_cache_layout.ROPE_WORDS) for b in rope_banks)
    nope_inputs = nope_banks if banked else nope_banks[0]
    rope_inputs = rope_banks if banked else rope_banks[0]
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    out_size = indices.size
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
    # A banked stream issues four indirect DMAs (two banks x two caches).
    # Two streams keep eight in flight per buffer. The four-stream variant
    # stalls on v7x; the measured configuration limits the DMA concurrency.
    num_streams = 2 if banked else 4
    assert top_k % 128 == 0
    assert out_size % top_k == 0
    max_subchunks = top_k // row_subchunk_size
    target_subchunks = (out_size + num_cores * row_subchunk_size - 1) // (
        num_cores * row_subchunk_size
    )
    num_row_subchunks = num_streams
    while num_row_subchunks < max_subchunks and (
        num_row_subchunks < target_subchunks or max_subchunks % num_row_subchunks != 0
    ):
        num_row_subchunks += num_streams
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    if out_pad_size:
        # spread the padding to avoid hotspots
        num_tokens = sum(b.shape[0] for b in nope_banks)
        pad = (jnp.arange(out_pad_size, dtype=indices.dtype) * 104729) % num_tokens
        indices = jnp.concatenate([indices, pad])
    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    nope_out, rope_out = cast(
        tuple[jax.Array, jax.Array],
        pl.kernel(
            functools.partial(
                _banked_kernel if banked else main_kernel,
                core_axis_name=vector_mesh.core_axis_name,
                subcore_axis_name=vector_mesh.subcore_axis_name,
                num_row_subchunks=num_row_subchunks,
                num_streams=num_streams,
                top_k=top_k,
            ),
            out_type=(
                jax.ShapeDtypeStruct(
                    (out_size + out_pad_size, csa_cache_layout.ROW_WORDS), jnp.int32
                ),
                jax.ShapeDtypeStruct(
                    ((out_size + out_pad_size) // 4, csa_cache_layout.ROW_WORDS),
                    jnp.int32,
                ),
            ),
            # One DMA semaphore per stream for the direct nope gather-buffer -> HBM
            # copies issued in `main_kernel`.
            scratch_types=(
                pltpu.VMEM((2, num_streams, row_subchunk_size, 128), jnp.int32),
                pltpu.VMEM((2, num_streams, row_subchunk_size, 32), jnp.int32),
                pltpu.VMEM((num_streams, row_subchunk_size // 2, 64), jnp.int32),
                pltpu.SemaphoreType.DMA((2, 2, num_streams, 2)),
                pltpu.SemaphoreType.DMA((num_streams, 2)),
            )
            if banked
            else (pltpu.SemaphoreType.DMA((num_streams,)),),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            mesh=vector_mesh,
            name="sc_csa_gather",
        )(nope_inputs, rope_inputs, indices, valid_indices),
    )
    return nope_out[:out_size], rope_out[: out_size // 4]
