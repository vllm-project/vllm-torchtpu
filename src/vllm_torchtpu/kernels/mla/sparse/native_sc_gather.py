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
"""Combined native-SparseCore NOPE/ROPE gather for GLM-5.2."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper

NUM_CORES = 2
NUM_SUBCORES = 16
NUM_LANES = 16
ROPE_PACKING = 4
QUADS_PER_ATOM = 16
OCCURRENCES_PER_ATOM = QUADS_PER_ATOM * ROPE_PACKING


def _delta_swap(x, y, shift, mask):
    tmp = jnp.bitwise_and(jnp.bitwise_xor(jnp.right_shift(x, shift), y), mask)
    return (
        jnp.bitwise_xor(x, jnp.left_shift(tmp, shift)),
        jnp.bitwise_xor(y, tmp),
    )


def _kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    sem_ref: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    wait_mode: str,
    serve_pipelined: bool,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))

    def serve_batch(indices_ref):
        owner_batch = pl.program_id(0)
        logical_batch = owner + owner_batch * num_owners
        occurrence_base = (logical_batch * atoms_per_batch *
                           OCCURRENCES_PER_ATOM)

        def values(atom, stream):
            return indices_ref[0, atom, stream, :]

        def destinations(atom, stream):
            return pl.ds(
                occurrence_base + atom * OCCURRENCES_PER_ATOM +
                stream * NUM_LANES,
                NUM_LANES,
            )

        def body(*refs):
            atom = pl.program_id(0)
            nope_inputs = refs[:ROPE_PACKING]
            rope_inputs = refs[ROPE_PACKING:2 * ROPE_PACKING]
            rope_output = refs[-1]

            nope_copies = []
            for stream in range(ROPE_PACKING):
                copy = pltpu.make_async_copy(
                    nope_inputs[stream],
                    nope_out_hbm_ref.at[destinations(atom, stream), :],
                    sem_ref.at[stream],
                )
                copy.start()
                nope_copies.append(copy)
            if wait_mode == "before_rope":
                for copy in nope_copies:
                    copy.wait()

            for row in range(QUADS_PER_ATOM):
                data = []
                for token_slot in range(ROPE_PACKING):
                    logical_rank = row * ROPE_PACKING + token_slot
                    input_stream = logical_rank // NUM_LANES
                    input_lane = logical_rank % NUM_LANES
                    data.append(rope_inputs[input_stream][pl.ds(input_lane, 1),
                                                          pl.ds(0, 32)])

                low_16 = jnp.uint32(0x0000FFFF)
                low_byte_each_half = jnp.uint32(0x00FF00FF)
                data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
                data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
                data[0], data[1] = _delta_swap(data[0], data[1], 8,
                                               low_byte_each_half)
                data[2], data[3] = _delta_swap(data[2], data[3], 8,
                                               low_byte_each_half)
                for band in range(ROPE_PACKING):
                    rope_output[pl.ds(row, 1),
                                pl.ds(band * 32, 32)] = data[band]

            if wait_mode == "overlap_rope":
                for copy in nope_copies:
                    copy.wait()

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 128),
                lambda atom, stream=stream: (values(atom, stream), 0),
            ) for stream in range(ROPE_PACKING))
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 32),
                lambda atom, stream=stream: (values(atom, stream), 0),
            ) for stream in range(ROPE_PACKING))
        rope_out_spec = pl.BlockSpec(
            (QUADS_PER_ATOM, 128),
            lambda atom: (logical_batch * atoms_per_batch + atom, 0),
        )
        pltpu.emit_pipeline(
            body,
            grid=(atoms_per_batch, ),
            in_specs=nope_specs + rope_specs,
            out_specs=(rope_out_spec, ),
            no_pipelining=not serve_pipelined,
        )(
            *([nope_hbm_ref] * ROPE_PACKING),
            *([rope_hbm_ref] * ROPE_PACKING),
            rope_out_hbm_ref,
        )

    indices_spec = pl.BlockSpec(
        (1, atoms_per_batch, ROPE_PACKING, NUM_LANES),
        lambda batch: (owner + batch * num_owners, 0, 0, 0),
    )
    pltpu.emit_pipeline(
        serve_batch,
        grid=(batches_per_owner, ),
        in_specs=(indices_spec, ),
    )(indices_hbm_ref)


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "atoms_per_batch",
        "wait_mode",
        "serve_pipelined",
    ),
)
def dsa_gather_native_sc(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    *,
    out_size: int,
    atoms_per_batch: int = 16,
    wait_mode: str = "overlap_rope",
    serve_pipelined: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Gather explicit native-SC ``uint32`` NOPE and ROPE cache rows.

    ``nope_cache`` has one 128-word row per token. ``rope_cache`` has one
    128-word row per group of four tokens; each token owns a consecutive
    32-word quarter. The returned ROPE is register-transposed into the grouped
    TensorCore format expected by the shared attention consumer.
    """
    if wait_mode not in ("overlap_rope", "before_rope"):
        raise ValueError(f"unknown {wait_mode=}")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError(
            "native-SC ROPE must be quarter-major uint32[...,128]")
    if indices.dtype != jnp.int32 or indices.ndim != 2:
        raise ValueError("indices must be int32[query, topk]")
    if indices.size != out_size:
        raise ValueError(
            f"indices has {indices.size} values, expected {out_size}")

    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    num_cores = sc_info.num_cores if sc_info is not None else NUM_CORES
    num_subcores = (sc_info.num_subcores
                    if sc_info is not None else NUM_SUBCORES)
    num_owners = num_cores * num_subcores

    max_possible_atoms = out_size // (num_owners * OCCURRENCES_PER_ATOM)
    if atoms_per_batch > max_possible_atoms > 0:
        atoms_per_batch = max_possible_atoms
    if atoms_per_batch <= 0:
        raise ValueError(f"atoms_per_batch ({atoms_per_batch}) must be > 0 "
                         f"(out_size={out_size}, num_owners={num_owners})")

    occurrences_per_batch = atoms_per_batch * OCCURRENCES_PER_ATOM
    owner_batch_size = num_owners * occurrences_per_batch
    if out_size % owner_batch_size != 0:
        raise ValueError(
            f"out_size ({out_size}) must be divisible by num_owners * "
            f"occurrences_per_batch ({owner_batch_size})")
    num_batches = out_size // occurrences_per_batch
    batches_per_owner = num_batches // num_owners
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    return core_map_helper.kernel(
        functools.partial(
            _kernel,
            core_axis_name=mesh.core_axis_name,
            subcore_axis_name=mesh.subcore_axis_name,
            num_owners=num_owners,
            atoms_per_batch=atoms_per_batch,
            batches_per_owner=batches_per_owner,
            wait_mode=wait_mode,
            serve_pipelined=serve_pipelined,
        ),
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct((out_size // ROPE_PACKING, 128), jnp.uint32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types={"sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING, ))},
        mesh=mesh,
        name=(f"sc_native_sc_gather_a{atoms_per_batch}_{wait_mode}_"
              f"{'pipe' if serve_pipelined else 'nopipe'}"),
    )(
        nope_cache.reshape(-1, 128),
        rope_cache.reshape(-1, 32),
        indices.reshape(num_batches, atoms_per_batch, ROPE_PACKING, NUM_LANES),
    )
