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

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout


def main_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    valid_indices_ref: Any,
    max_active_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    valid_vmem_ref: Any,
    nope_vmem: Any,
    nope_in_sem: Any,
    nope_out_sem: Any,
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
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    nope_in_cols = nope_in_hbm_ref.shape[1]
    rope_in_cols = rope_in_hbm_ref.shape[1]
    # A `bf16[n, 64]` output is tiled to 128 lanes, so XLA pads it and it
    # costs twice its own bytes -- on this kernel's write and on the
    # consumer's read. The output is 128 lanes wide instead, pairing entry `i`
    # of each `top_k`-entry period with entry `i + top_k // 2`, which the
    # consumer splits apart with a lane slice and a row concatenate.
    rope_out_cols = rope_out_hbm_ref.shape[1] // 2
    rope_rows_per_stream = row_subchunk_size // 2
    step_size = num_streams * row_subchunk_size

    cp_val = pltpu.make_async_copy(
        valid_indices_ref.at[pl.ds(0, row_subchunk_size)],
        valid_vmem_ref.at[pl.ds(0, row_subchunk_size)],
        nope_out_sem.at[0],
    )
    cp_val.start()
    cp_val.wait()
    valid_indices = valid_vmem_ref[pl.ds(0, row_subchunk_size)][0]
    active_waves = pl.cdiv(pl.cdiv(valid_indices, row_chunk_size), num_cores)

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

    def outer_pipeline(idx_ref, active_ref):
        b = pl.program_id(0)
        period_idx = b * num_cores + core_index
        out_row_base = period_idx * num_row_subchunks
        chunk_active_rows = active_ref[pl.ds(0, row_subchunk_size)][0]
        active_inner_steps = pl.cdiv(chunk_active_rows, step_size)

        # Subchunk handled by stream `s` at inner step `r`.
        def subchunk(r, s):
            return r * num_streams + s

        def idx_window(r, s):
            return idx_ref[
                pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)
            ]

        def _body(*refs):
            r = pl.program_id(0)
            cur_buf = jnp.mod(r, 3)
            prv_buf = jnp.mod(r + 2, 3)  # == (r - 1) % 3 == (r + 2) % 3
            cur_sem = jnp.bitwise_and(r, 1)
            prv_sem = 1 - cur_sem
            rope_g = refs[0 : num_streams]
            rope_o = refs[num_streams]

            # 1. Complete step r - 1's direct VMEM -> HBM DMA on `prv_buf`
            # BEFORE reusing `prv_buf` for step r + 2's indirect HBM -> VMEM gather.
            @pl.when(r > 0)
            def _wait_prev_nope_out():
                for s in range(num_streams):
                    prev_row = (out_row_base + subchunk(r - 1, s)) * row_subchunk_size
                    pltpu.make_async_copy(
                        nope_vmem.at[prv_buf, s],
                        nope_out_hbm_ref.at[pl.ds(prev_row, row_subchunk_size)],
                        nope_out_sem.at[prv_sem],
                    ).wait()

            # 2. Strictly after `prv_buf` is free, prefetch step r + 2 into `prv_buf`.
            @pl.when(r + 2 < active_inner_steps)
            def _prefetch_next_nope_in():
                for s in range(num_streams):
                    pltpu.make_async_copy(
                        nope_in_hbm_ref.at[idx_window(r + 2, s), pl.ds(0, nope_in_cols)],
                        nope_vmem.at[prv_buf, s],
                        nope_in_sem.at[prv_buf, s],
                    ).start()

            # 3. Wait for step r's indirect gather (`cur_buf`, started 2 steps ago).
            for s in range(num_streams):
                pltpu.make_async_copy(
                    nope_in_hbm_ref.at[idx_window(r, s), pl.ds(0, nope_in_cols)],
                    nope_vmem.at[cur_buf, s],
                    nope_in_sem.at[cur_buf, s],
                ).wait()

            # 4. Launch zero-copy direct `cur_buf` -> HBM write overlapped with VPU rope packing.
            for s in range(num_streams):
                out_row = (out_row_base + subchunk(r, s)) * row_subchunk_size
                pltpu.make_async_copy(
                    nope_vmem.at[cur_buf, s],
                    nope_out_hbm_ref.at[pl.ds(out_row, row_subchunk_size)],
                    nope_out_sem.at[cur_sem],
                ).start()

            for s in range(num_streams):
                process_rope(
                    gather_ref=rope_g[s],
                    out_ref=rope_o,
                    out_row_base=s * rope_rows_per_stream,
                )

            # 5. Drain final step's direct VMEM -> HBM write before exiting period.
            @pl.when(r + 1 == active_inner_steps)
            def _wait_last_nope_out():
                for s in range(num_streams):
                    last_row = (out_row_base + subchunk(r, s)) * row_subchunk_size
                    pltpu.make_async_copy(
                        nope_vmem.at[cur_buf, s],
                        nope_out_hbm_ref.at[pl.ds(last_row, row_subchunk_size)],
                        nope_out_sem.at[cur_sem],
                    ).wait()

        # rope: gather the entry's own 32-word quarter row == index.
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda r, s=s: (idx_window(r, s), 0),
            )
            for s in range(num_streams)
        )

        # One rope output block per step, covering all `num_streams` subchunks.
        steps_per_half = top_k // 2 // (num_streams * row_subchunk_size)

        def rope_out_block(r):
            half, step = divmod(out_row_base // num_streams + r, steps_per_half)
            return (half // 2) * steps_per_half + step, half % 2

        rope_out_spec = pl.BlockSpec(
            (num_streams * rope_rows_per_stream, rope_out_cols),
            rope_out_block,
        )
        core_start_idx = period_idx * row_chunk_size
        run_period = (core_start_idx < valid_indices) & (active_inner_steps > 0)
        if not envs.VLLM_DSV41_PURE_20L_PREFILL:
            # [merge] Decode-side (common-base) guard: a period whose first
            # index is negative -- `_build_candidate_union_and_bitmask` marks
            # inactive union chunks with -1 when u_chunk < m_union -- is
            # skipped instead of gathering row -1 with bounds checks disabled.
            # The prefill tree (cd4eedb) has no such guard; keep its kernel
            # body byte-identical on the prefill role.
            first_idx = idx_ref[pl.ds(0, row_subchunk_size)][0]
            run_period = run_period & (first_idx >= 0)

        @pl.when(run_period)
        def _run_gather():
            # Prefetch r = 0 (`buf 0`) and r = 1 (`buf 1`) before entering inner pipeline.
            for s in range(num_streams):
                pltpu.make_async_copy(
                    nope_in_hbm_ref.at[idx_window(0, s), pl.ds(0, nope_in_cols)],
                    nope_vmem.at[0, s],
                    nope_in_sem.at[0, s],
                ).start()

            @pl.when(active_inner_steps > 1)
            def _prefetch_step1():
                for s in range(num_streams):
                    pltpu.make_async_copy(
                        nope_in_hbm_ref.at[idx_window(1, s), pl.ds(0, nope_in_cols)],
                        nope_vmem.at[1, s],
                        nope_in_sem.at[1, s],
                    ).start()

            pltpu.emit_pipeline(
                _body,
                grid=(active_inner_steps,),
                in_specs=rope_in_specs,
                out_specs=(rope_out_spec,),
            )(
                *([rope_in_hbm_ref] * num_streams),
                rope_out_hbm_ref,
            )

    @pl.when(active_waves > 0)
    def _launch_waves():
        pltpu.emit_pipeline(
            outer_pipeline,
            grid=(active_waves,),
            in_specs=(
                pl.BlockSpec(
                    (row_chunk_size,),
                    lambda b: (b * num_cores + core_index,),
                ),
                pl.BlockSpec(
                    (row_subchunk_size,),
                    lambda b: (b * num_cores + core_index,),
                ),
            ),
        )(indices_hbm_ref, max_active_hbm_ref)


@functools.partial(jax.jit, static_argnames=("top_k",))
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    num_valid_indices: jax.Array | None = None,
    *,
    top_k: int = 1024,
    max_active_per_chunk: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fused SparseCore gather of the nope and rope caches.

    Args:
      nope_cache: (total_pages, page_size, 128) int32, one row per token: the
        (4, 128) uint8 nope slab (see `csa_cache_layout`).
      rope_cache: (total_pages, page_size // 4, 128) int32, 32 words per token
        (see `csa_cache_layout`).
      indices: (N,) int32. Token indices into the caches; N a multiple of
        top_k.
      num_valid_indices: Optional (1,) or scalar int32. Number of valid indices to
        gather. A subcore skips every top_k period that starts at or past this
        count, leaving those output rows unwritten.
      top_k: the consumer's row block (the attention kernel's top-k), a
        multiple of 128. `rope_out` pairs entry i of a period with entry
        i + top_k // 2.
      max_active_per_chunk: Optional int32 array of length `N // top_k` giving
        the active token count in each `top_k` period (`[top_k // 2, top_k]`).

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
    assert nope_cache.dtype == rope_cache.dtype, "Caches must share a dtype."
    assert nope_cache.dtype == jnp.int32, "Caches must be int32."
    assert nope_cache.shape[2] == csa_cache_layout.ROW_WORDS
    assert rope_cache.shape[2] == csa_cache_layout.ROW_WORDS

    # Free reshapes: a 128-lane int32 array's T(8,128) tiling is byte-identical
    # to row-major, and the kernel takes its operands untiled
    # (`use_tc_tiling_on_sc=False`), so token t's rope is words
    # [32 t, 32 t + 32).
    nope_cache = nope_cache.reshape(-1, csa_cache_layout.ROW_WORDS)
    rope_cache = rope_cache.reshape(-1, csa_cache_layout.ROPE_WORDS)
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None, "SparseCore info is missing."
    out_size = indices.size
    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    row_subchunk_size = num_simd_lanes

    valid_scalar = (
        jnp.int32(out_size)
        if num_valid_indices is None
        else jnp.reshape(num_valid_indices, ()).astype(jnp.int32)
    )
    valid_indices = jnp.full((row_subchunk_size,), valid_scalar, dtype=jnp.int32)

    # `num_streams` independent indirect gathers are issued per step to keep
    # multiple gather DMAs in flight.
    num_streams = min(4, max(1, (top_k // 2) // row_subchunk_size))
    assert out_size % top_k == 0
    num_row_subchunks = top_k // row_subchunk_size
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    if out_pad_size:
        # spread the padding across valid [0, num_tokens) indices to avoid OOB
        num_tokens = nope_cache.shape[0]
        pad = (jnp.arange(out_pad_size, dtype=indices.dtype) * 104729) % num_tokens
        indices = jnp.concatenate([indices, pad])

    num_chunks = out_size // row_chunk_size
    padded_chunks = (out_size + out_pad_size) // row_chunk_size
    chunk_starts = jnp.arange(num_chunks, dtype=jnp.int32) * row_chunk_size
    rem_rows = jnp.clip(valid_scalar - chunk_starts, 0, row_chunk_size)
    if max_active_per_chunk is None:
        chunk_active = jnp.where(rem_rows > 0, row_chunk_size, 0)
    else:
        chunk_active = jnp.where(
            rem_rows > 0,
            jnp.minimum(
                jnp.maximum(row_chunk_size // 2, max_active_per_chunk),
                row_chunk_size,
            ),
            0,
        )
    if padded_chunks > num_chunks:
        chunk_active = jnp.concatenate(
            [
                chunk_active,
                jnp.zeros((padded_chunks - num_chunks,), dtype=jnp.int32),
            ]
        )
    max_active_vec = jnp.repeat(chunk_active, row_subchunk_size)

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    nope_out, rope_out = cast(
        Any,
        pl.kernel(
            functools.partial(
                main_kernel,
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
            scratch_types=(
                pltpu.VMEM((row_subchunk_size,), jnp.int32),
                pltpu.VMEM(
                    (3, num_streams, row_subchunk_size, csa_cache_layout.ROW_WORDS),
                    jnp.int32,
                ),
                pltpu.SemaphoreType.DMA((3, num_streams)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            mesh=vector_mesh,
            name="sc_csa_gather",
        )(nope_cache, rope_cache, indices, valid_indices, max_active_vec),
    )  # pyrefly: ignore[not-iterable]
    return nope_out[:out_size], rope_out[: out_size // 4]
