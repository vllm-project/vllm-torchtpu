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
"""SparseCore gather of paged FP8 NoPE word rows and interleaved BF16 RoPE pairs."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

TPU7X_GENERATION = 7
ROW_WORDS = 128
ROPE_WORDS = 32
ROPE_TOKENS_PER_ROW = ROW_WORDS // ROPE_WORDS
NUM_SC_CORES = 2
NUM_SUBCORES = 16
BF16_PER_WORD = 2
HALF_WORD_SHIFT_BITS = 16
LOW_HALF_WORD_MASK = 0xFFFF
HIGH_HALF_WORD_MASK = -65536
NUM_GATHER_STREAMS = 4
FALLBACK_STRIDE_PRIME = 104_729


def _check_tpu7x(func_name: str) -> None:
    # Avoid calling pltpu.get_tpu_info() on bare CPU: raises ValueError when no
    # TPU backend or AbstractMesh is active.
    if (
        jax.default_backend() != "tpu"
        and jax.sharding.get_abstract_mesh().abstract_device is None
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x, got backend={jax.default_backend()!r}."
        )
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    if (
        tpu_info.generation != TPU7X_GENERATION
        or sc_info is None
        or sc_info.num_cores != NUM_SC_CORES
        or sc_info.num_subcores != NUM_SUBCORES
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x SparseCore ({NUM_SC_CORES}x{NUM_SUBCORES} "
            f"subcores), got generation={tpu_info.generation}, sparse_core={sc_info!r}."
        )


def _csa_gather_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    nope_sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_simd_lanes: int,
    num_row_subchunks: int,
    num_streams: int,
    top_k: int,
    out_size: int,
) -> None:
    num_cores = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    nope_in_cols = nope_in_hbm_ref.shape[1]
    rope_in_cols = rope_in_hbm_ref.shape[1]
    # Avoid bf16[N, 64] RoPE output: XLA pads 64-lane outputs to 128 lanes (2x HBM
    # bytes); pair entries i and i + top_k // 2 across 128 lanes instead.
    rope_out_cols = rope_out_hbm_ref.shape[1] // BF16_PER_WORD

    def _pack_rope_subchunk(gather_ref: Any, out_ref: Any, out_row_base: int) -> None:
        for pair_idx in range(num_simd_lanes // BF16_PER_WORD):
            even = gather_ref[pl.ds(BF16_PER_WORD * pair_idx, 1), :]
            odd = gather_ref[pl.ds(BF16_PER_WORD * pair_idx + 1, 1), :]
            out_row = pl.ds(out_row_base + pair_idx, 1)
            out_ref[out_row, pl.ds(0, rope_in_cols)] = jnp.bitwise_or(
                jnp.bitwise_and(even, LOW_HALF_WORD_MASK),
                jnp.left_shift(odd, HALF_WORD_SHIFT_BITS),
            )
            out_ref[out_row, pl.ds(rope_in_cols, rope_in_cols)] = jnp.bitwise_or(
                lax.shift_right_logical(even, HALF_WORD_SHIFT_BITS),
                jnp.bitwise_and(odd, jnp.int32(HIGH_HALF_WORD_MASK)),
            )

    def _outer_pipeline(idx_ref: Any) -> None:
        block_idx = pl.program_id(0)
        out_row_base = (block_idx * num_cores + core_index) * num_row_subchunks
        rope_rows_per_stream = row_subchunk_size // BF16_PER_WORD

        def _subchunk_index(step_idx: jax.Array | int, stream_idx: int) -> Any:
            return step_idx * num_streams + stream_idx

        def _idx_window(step_idx: jax.Array | int, stream_idx: int) -> Any:
            return idx_ref[
                pl.ds(
                    _subchunk_index(step_idx, stream_idx) * row_subchunk_size,
                    row_subchunk_size,
                )
            ]

        def _stream_step(*refs: Any) -> None:
            step_idx = pl.program_id(0)
            nope_gather_refs = refs[0:num_streams]
            rope_gather_refs = refs[num_streams : 2 * num_streams]
            rope_out_ref = refs[2 * num_streams]

            # Avoid routing nope gather buffers through an emit_pipeline output buffer:
            # direct VMEM->HBM async DMA eliminates an extra VMEM->VMEM copy.
            nope_copies = []
            for stream_idx in range(num_streams):
                out_row = (
                    out_row_base + _subchunk_index(step_idx, stream_idx)
                ) * row_subchunk_size
                copy = pltpu.make_async_copy(
                    nope_gather_refs[stream_idx],
                    nope_out_hbm_ref.at[pl.ds(out_row, row_subchunk_size)],
                    nope_sem.at[stream_idx],
                )
                copy.start()
                nope_copies.append(copy)

            for stream_idx in range(num_streams):
                _pack_rope_subchunk(
                    gather_ref=rope_gather_refs[stream_idx],
                    out_ref=rope_out_ref,
                    out_row_base=stream_idx * rope_rows_per_stream,
                )

            for copy in nope_copies:
                copy.wait()

        nope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), nope_in_cols),
                lambda step_idx, stream_idx=stream_idx: (
                    _idx_window(step_idx, stream_idx),
                    0,
                ),
            )
            for stream_idx in range(num_streams)
        )
        rope_in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), rope_in_cols),
                lambda step_idx, stream_idx=stream_idx: (
                    _idx_window(step_idx, stream_idx),
                    0,
                ),
            )
            for stream_idx in range(num_streams)
        )

        steps_per_half = (top_k // BF16_PER_WORD) // (num_streams * row_subchunk_size)

        def _rope_out_block(step_idx: jax.Array | int) -> tuple[Any, Any]:
            half, step = divmod(out_row_base // num_streams + step_idx, steps_per_half)
            return (half // BF16_PER_WORD) * steps_per_half + step, half % BF16_PER_WORD

        rope_out_spec = pl.BlockSpec(
            (num_streams * rope_rows_per_stream, rope_out_cols), _rope_out_block
        )
        core_start_idx = out_row_base * row_subchunk_size

        @pl.when(core_start_idx < out_size)
        def _run_gather() -> None:
            pltpu.emit_pipeline(
                _stream_step,
                grid=(num_row_subchunks // num_streams,),
                in_specs=nope_in_specs + rope_in_specs,
                out_specs=(rope_out_spec,),
            )(
                *([nope_in_hbm_ref] * num_streams),
                *([rope_in_hbm_ref] * num_streams),
                rope_out_hbm_ref,
            )

    pltpu.emit_pipeline(
        _outer_pipeline,
        grid=(num_blocks,),
        in_specs=(
            pl.BlockSpec(
                (row_chunk_size,),
                lambda block_idx: (block_idx * num_cores + core_index,),
            ),
        ),
    )(indices_hbm_ref)


@functools.partial(jax.jit, static_argnames=("top_k",))
def csa_gather(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    *,
    top_k: int = 1024,
) -> tuple[jax.Array, jax.Array]:
    """Gather ``(int32[N, 128], int32[N//4, 128])`` NoPE/RoPE rows via SparseCore.

    Reads ``nope_cache: (P, S, 128)`` and ``rope_cache: (P, S//4, 128)`` at
    ``indices: int32[N]``, pairing period entries ``i`` and ``i + top_k // 2`` across
    64-BF16 halves; requires ``S % 4 == 0`` and ``N % top_k == 0``.
    """
    _check_tpu7x("csa_gather")
    sc_info: Any = pltpu.get_tpu_info().sparse_core
    if (
        nope_cache.ndim != 3
        or nope_cache.dtype != jnp.int32
        or nope_cache.shape[0] < 1
        or nope_cache.shape[1] < ROPE_TOKENS_PER_ROW
        or nope_cache.shape[1] % ROPE_TOKENS_PER_ROW != 0
        or nope_cache.shape[2] != ROW_WORDS
    ):
        raise ValueError(
            "Expected int32 nope_cache [pages >= 1, page_size % 4 == 0, 128], "
            f"got {nope_cache.dtype}{nope_cache.shape}."
        )
    expected_rope_shape = (
        nope_cache.shape[0],
        nope_cache.shape[1] // ROPE_TOKENS_PER_ROW,
        ROW_WORDS,
    )
    if rope_cache.shape != expected_rope_shape or rope_cache.dtype != jnp.int32:
        raise ValueError(
            f"Expected int32 rope_cache {expected_rope_shape}, "
            f"got {rope_cache.dtype}{rope_cache.shape}."
        )
    if indices.ndim != 1 or indices.dtype != jnp.int32:
        raise ValueError(
            f"Expected 1D int32 indices, got {indices.dtype}{indices.shape}."
        )

    row_subchunk_size = sc_info.num_lanes
    min_top_k = BF16_PER_WORD * NUM_GATHER_STREAMS * row_subchunk_size
    if top_k < min_top_k or top_k % min_top_k != 0:
        raise ValueError(
            f"Expected top_k positive multiple of {min_top_k}, got {top_k}."
        )
    out_size = indices.shape[0]
    if out_size % top_k != 0:
        raise ValueError(
            f"Expected indices.size ({out_size}) divisible by top_k ({top_k})."
        )
    if out_size == 0:
        empty = jnp.zeros((0, ROW_WORDS), dtype=jnp.int32)
        return empty, empty

    flat_nope_cache = nope_cache.reshape(-1, ROW_WORDS)
    flat_rope_cache = rope_cache.reshape(-1, ROPE_WORDS)
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_row_subchunks = top_k // row_subchunk_size
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    out_pad_size = ((out_size + block_size - 1) // block_size) * block_size - out_size
    if out_pad_size:
        # Avoid constant-zero index padding on tail subcores: prime-strided fallback
        # rows spread dummy gathers across cache banks.
        num_tokens = flat_nope_cache.shape[0]
        pad = (
            jnp.arange(out_pad_size, dtype=indices.dtype) * FALLBACK_STRIDE_PRIME
        ) % num_tokens
        indices = jnp.concatenate([indices, pad])

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    padded_out_size = out_size + out_pad_size
    nope_out, rope_out = cast(
        Callable[..., tuple[jax.Array, jax.Array]],
        pl.kernel(
            functools.partial(
                _csa_gather_kernel,
                core_axis_name=vector_mesh.core_axis_name,
                subcore_axis_name=vector_mesh.subcore_axis_name,
                num_simd_lanes=row_subchunk_size,
                num_row_subchunks=num_row_subchunks,
                num_streams=NUM_GATHER_STREAMS,
                top_k=top_k,
                out_size=out_size,
            ),
            out_type=(
                jax.ShapeDtypeStruct((padded_out_size, ROW_WORDS), jnp.int32),
                jax.ShapeDtypeStruct(
                    (padded_out_size // ROPE_TOKENS_PER_ROW, ROW_WORDS), jnp.int32
                ),
            ),
            scratch_types=(pltpu.SemaphoreType.DMA((NUM_GATHER_STREAMS,)),),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            mesh=vector_mesh,
            name="sc_csa_gather",
        ),
    )(flat_nope_cache, flat_rope_cache, indices)
    return nope_out[:out_size], rope_out[: out_size // ROPE_TOKENS_PER_ROW]
