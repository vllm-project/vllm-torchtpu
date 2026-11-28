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
"""Combined native-SparseCore NOPE/ROPE gather for GLM-5.2.

An optional traced live-row count (``num_live_indices``) bounds the outer
emit_pipeline grid so only the logical batches holding live rows are gathered;
rows beyond are left unwritten (the attention grid never reads them). Without
it every row is gathered.
"""

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
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

NUM_CORES = 2
NUM_SUBCORES = 16
NUM_LANES = 16
ROPE_PACKING = 4
QUADS_PER_ATOM = 16
OCCURRENCES_PER_ATOM = QUADS_PER_ATOM * ROPE_PACKING
ROPE_WORDS = 32  # max words per rope token (128B padded view)


def _delta_swap(x, y, shift, mask):
    tmp = jnp.bitwise_and(jnp.bitwise_xor(jnp.right_shift(x, shift), y), mask)
    return (
        jnp.bitwise_xor(x, jnp.left_shift(tmp, shift)),
        jnp.bitwise_xor(y, tmp),
    )


def transpose_quad(data: list[Any]) -> list[Any]:
    """Four tokens' banded words -> four bands of token-interleaved words.

    In: `data[t]` word `w` byte `b` is token `t`'s byte `b * n + w` (the
    `pack_tokens` banding, `n` words per token: 32 padded, 16 packed). Out:
    `data[b]` word `w` byte `t` is that same byte, which bitcasts to the
    four-token uint8 tile the TensorCore consumer reads. Pure jnp, so tests
    run it on CPU too.
    """
    data = list(data)
    low_16 = jnp.uint32(0x0000FFFF)
    low_byte_each_half = jnp.uint32(0x00FF00FF)
    data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
    data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
    data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
    data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
    return data


def _kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    sem_ref: Any,
    live_vmem_ref: Any = None,
    live_sem_ref: Any = None,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    wait_mode: str,
    serve_pipelined: bool,
    live_hbm_ref: Any = None,
    guard_tail: bool = False,
    rope_words: int = ROPE_WORDS,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    live_batches = None
    if live_hbm_ref is not None:
        # The live logical-batch count arrives as a broadcast i32[NUM_LANES]
        # vector. On SparseCore a scalar load is legal only from SMEM and SMEM
        # only at rank 0 (jax/_src/pallas/mosaic/sc_lowering.py), so the count is
        # DMA'd into a VMEM vector and lane 0 extracted -- the same shape as the
        # in-repo dsa_gather.py:90 (`valid_ref[pl.ds(0, 16)][0]`).
        cp = pltpu.make_async_copy(live_hbm_ref, live_vmem_ref, live_sem_ref.at[0])
        cp.start()
        cp.wait()
        # Lanes [0,16) broadcast the live logical-batch count, lanes [16,32)
        # the per-owner trip count; only lane 0 of a vector is ever extracted.
        live_batches = live_vmem_ref[pl.ds(0, NUM_LANES)][0]

    def serve_batch(indices_ref):
        owner_batch = pl.program_id(0)
        logical_batch = owner + owner_batch * num_owners
        occurrence_base = logical_batch * atoms_per_batch * OCCURRENCES_PER_ATOM

        def values(atom, stream):
            return indices_ref[0, atom, stream, :]

        def destinations(atom, stream):
            return pl.ds(
                occurrence_base + atom * OCCURRENCES_PER_ATOM + stream * NUM_LANES,
                NUM_LANES,
            )

        def body(*refs):
            atom = pl.program_id(0)
            nope_inputs = refs[:ROPE_PACKING]
            rope_inputs = refs[ROPE_PACKING : 2 * ROPE_PACKING]
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
                    data.append(
                        rope_inputs[input_stream][
                            pl.ds(input_lane, 1), pl.ds(0, rope_words)
                        ]
                    )

                data = transpose_quad(data)
                for band in range(ROPE_PACKING):
                    rope_output[pl.ds(row, 1), pl.ds(band * rope_words, rope_words)] = (
                        data[band]
                    )
                # Packed rope fills only the first ROPE_PACKING * rope_words
                # lanes; zero the rest, which the consumer reads as the
                # (zero-padded) rope tail, as the padded cache stores it.
                # Garbage there could decode to fp8 NaN, and NaN * 0 poisons
                # the score against q's zero pad.
                for band in range(ROPE_PACKING, 128 // rope_words):
                    rope_output[pl.ds(row, 1), pl.ds(band * rope_words, rope_words)] = (
                        jnp.zeros((1, rope_words), jnp.uint32)
                    )

            if wait_mode == "overlap_rope":
                for copy in nope_copies:
                    copy.wait()

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 128),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(ROPE_PACKING)
        )
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), rope_words),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(ROPE_PACKING)
        )
        rope_out_spec = pl.BlockSpec(
            (QUADS_PER_ATOM, 128),
            lambda atom: (logical_batch * atoms_per_batch + atom, 0),
        )

        def run_inner():
            pltpu.emit_pipeline(
                body,
                grid=(atoms_per_batch,),
                in_specs=nope_specs + rope_specs,
                out_specs=(rope_out_spec,),
                no_pipelining=not serve_pipelined,
            )(
                *([nope_hbm_ref] * ROPE_PACKING),
                *([rope_hbm_ref] * ROPE_PACKING),
                rope_out_hbm_ref,
            )

        if live_batches is not None and guard_tail:
            # Owners share a uniform traced grid of cdiv(live, num_owners);
            # the ragged last round is skipped per owner.
            pl.when(logical_batch < live_batches)(run_inner)
        else:
            run_inner()

    indices_spec = pl.BlockSpec(
        (1, atoms_per_batch, ROPE_PACKING, NUM_LANES),
        lambda batch: (owner + batch * num_owners, 0, 0, 0),
    )
    if live_batches is None:
        pltpu.emit_pipeline(
            serve_batch,
            grid=(batches_per_owner,),
            in_specs=(indices_spec,),
        )(indices_hbm_ref)
    else:
        # Traced grid length for emit_pipeline on a VectorSubcoreMesh, and
        # pl.when around it for the zero-trip case -- both lower on v7x SC.
        # No upstream SparseCore test traces an emit_pipeline grid (the
        # tpu_sparsecore_pallas_test.py grids are static); the nearest
        # upstream form, tests/pallas/tpu_pallas_pipeline_test.py:2420-2452
        # @ce57049, is a TensorCore test. This SC form is measured in-repo.
        # Uniform per-owner trip count; no scalar division on SC: the wrapper
        # sends live_batches as a multiple-of-num_owners-rounded pair.
        owner_batches = live_vmem_ref[pl.ds(NUM_LANES, NUM_LANES)][0]

        @pl.when(owner_batches > 0)
        def _serve_live():
            pltpu.emit_pipeline(
                serve_batch,
                grid=(owner_batches,),
                in_specs=(indices_spec,),
            )(indices_hbm_ref)


def _rope_transpose_atom(rope_src: Any, rope_dst: Any, rope_words: int):
    """Writes one atom's (64, rope_words) gathered ROPE rows into bands 0..3
    of its (16, 128) tile; any tail bands >= 4 are pre-zeroed on `rope_dst`."""
    for row in range(QUADS_PER_ATOM):
        data = [
            rope_src[pl.ds(row * ROPE_PACKING + slot, 1), pl.ds(0, rope_words)]
            for slot in range(ROPE_PACKING)
        ]
        data = transpose_quad(data)
        for band in range(ROPE_PACKING):
            rope_dst[pl.ds(row, 1), pl.ds(band * rope_words, rope_words)] = data[band]


def _ring_kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    indices_hbm_ref: Any,  # i32[num_batches, atoms_per_batch * 64]
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    nope_buf: Any,  # u32[ring_slots, 64, 128]
    rope_buf: Any,  # u32[ring_slots, 64, rope_words]
    rope_out_buf: Any,  # u32[2, 16, 128]
    idx_buf: Any,  # i32[2 * atoms_per_batch * 64]
    gather_sem: Any,  # DMA[2, ring_slots]
    nope_out_sem: Any,  # DMA[ring_slots]
    rope_out_sem: Any,  # DMA[2]
    idx_sem: Any,  # DMA[2]
    live_vmem_ref: Any = None,
    live_sem_ref: Any = None,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    ring_slots: int,
    live_hbm_ref: Any = None,
    guard_tail: bool = False,
    rope_words: int = ROPE_WORDS,
):
    """`_kernel` with a hand-rolled `ring_slots`-deep DMA ring.

    Same owner -> logical batch assignment and output bytes as `_kernel`, but
    each owner walks its atoms in one flat loop: atom j's 64 rows arrive as one
    indirect NOPE and one indirect ROPE DMA (indices read from VMEM), and while
    atom j is written back the gathers of atoms j+1 .. j+ring_slots-1 stay in
    flight across owner-batch boundaries (no per-batch pipeline restart). Index
    blocks are double-buffered per owner batch.
    """
    rows = OCCURRENCES_PER_ATOM
    batch_rows = atoms_per_batch * rows
    log_atoms = atoms_per_batch.bit_length() - 1
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    if live_hbm_ref is not None:
        cp = pltpu.make_async_copy(live_hbm_ref, live_vmem_ref, live_sem_ref.at[0])
        cp.start()
        cp.wait()
        live_batches = live_vmem_ref[pl.ds(0, NUM_LANES)][0]
        owner_batches = live_vmem_ref[pl.ds(NUM_LANES, NUM_LANES)][0]
        if guard_tail:
            # Drop this owner's ragged last round (see `_kernel`).
            last = owner + (owner_batches - 1) * num_owners
            owner_batches = owner_batches - jnp.where(last >= live_batches, 1, 0)
    else:
        owner_batches = jnp.int32(batches_per_owner)
    num_atoms = owner_batches * atoms_per_batch

    def batch_of(j):
        return lax.shift_right_logical(j, log_atoms)

    def atom_in_batch(j):
        return jnp.bitwise_and(j, atoms_per_batch - 1)

    def global_atom(j):
        return (owner + batch_of(j) * num_owners) * atoms_per_batch + atom_in_batch(j)

    def slot(j):
        return jnp.bitwise_and(j, ring_slots - 1)

    def idx_copy(b):
        half = jnp.bitwise_and(b, 1)
        return pltpu.make_async_copy(
            indices_hbm_ref.at[owner + b * num_owners],
            idx_buf.at[pl.ds(half * batch_rows, batch_rows)],
            idx_sem.at[half],
        )

    def gather_copies(j):
        half = jnp.bitwise_and(batch_of(j), 1)
        idx = idx_buf.at[pl.ds(half * batch_rows + atom_in_batch(j) * rows, rows)]
        return (
            pltpu.make_async_copy(
                nope_hbm_ref.at[idx], nope_buf.at[slot(j)], gather_sem.at[0, slot(j)]
            ),
            pltpu.make_async_copy(
                rope_hbm_ref.at[idx], rope_buf.at[slot(j)], gather_sem.at[1, slot(j)]
            ),
        )

    def nope_write(j):
        return pltpu.make_async_copy(
            nope_buf.at[slot(j)],
            nope_out_hbm_ref.at[pl.ds(global_atom(j) * rows, rows)],
            nope_out_sem.at[slot(j)],
        )

    def rope_write(j):
        half = jnp.bitwise_and(j, 1)
        return pltpu.make_async_copy(
            rope_out_buf.at[half],
            rope_out_hbm_ref.at[pl.ds(global_atom(j) * QUADS_PER_ATOM, QUADS_PER_ATOM)],
            rope_out_sem.at[half],
        )

    def start_gather(j):
        a = atom_in_batch(j)
        b = batch_of(j)

        @pl.when((a == 0) & (b > 0))
        def _():
            idx_copy(b).wait()

        # Prefetch the next batch's indices into the other half once every
        # gather still reading it (the previous batch's) has been waited.
        # Atom b*A + K is started inside step(b*A + 1), after the waits on
        # atoms up to b*A + 1, so batch b-1's gathers are all done. This needs
        # A > K (the caller's `atoms_per_batch > ring_slots` gate): otherwise
        # a == K never occurs within a batch and the prefetch is never issued.
        @pl.when((a == ring_slots) & (b + 1 < owner_batches))
        def _():
            idx_copy(b + 1).start()

        for copy in gather_copies(j):
            copy.start()

    @pl.when(num_atoms > 0)
    def _run():
        first = idx_copy(jnp.int32(0))
        first.start()
        for half in range(2):
            for row in range(QUADS_PER_ATOM):
                for band in range(ROPE_PACKING, 128 // rope_words):
                    rope_out_buf[
                        half, pl.ds(row, 1), pl.ds(band * rope_words, rope_words)
                    ] = jnp.zeros((1, rope_words), jnp.uint32)
        first.wait()
        for j in range(ring_slots - 1):
            pl.when(j < num_atoms)(functools.partial(start_gather, jnp.int32(j)))

        def step(j, carry):
            for copy in gather_copies(j):
                copy.wait()
            nope_write(j).start()

            @pl.when(j >= 2)
            def _():
                rope_write(j - 2).wait()

            _rope_transpose_atom(
                rope_buf.at[slot(j)],
                rope_out_buf.at[jnp.bitwise_and(j, 1)],
                rope_words,
            )
            rope_write(j).start()

            # Atom j-1's slot is refilled next: its write-back must be done.
            @pl.when(j >= 1)
            def _():
                nope_write(j - 1).wait()

            pl.when(j + ring_slots - 1 < num_atoms)(
                functools.partial(start_gather, j + ring_slots - 1)
            )
            return carry

        lax.fori_loop(0, num_atoms, step, 0)
        nope_write(num_atoms - 1).wait()
        rope_write(num_atoms - 1).wait()

        @pl.when(num_atoms >= 2)
        def _():
            rope_write(num_atoms - 2).wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "atoms_per_batch",
        "wait_mode",
        "serve_pipelined",
        "live_multiple",
        "ring_slots",
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
    num_live_indices: jax.Array | None = None,
    live_multiple: int = 1,
    ring_slots: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Gather explicit native-SC ``uint32`` NOPE and ROPE cache rows.

    ``nope_cache`` has one 128-word row per token. ``rope_cache`` packs
    ``128 // rope_words`` tokens into each 128-word row (e.g. eight 16-word
    64B tokens or four 32-word 128B tokens), with ``rope_words`` derived from
    the cache shapes. The returned ROPE is register-transposed into the grouped
    TensorCore format expected by the shared attention consumer, with any lanes
    past ``4 * rope_words`` zeroed.

    ``num_live_indices`` (traced i32 scalar, optional) gathers only the logical
    batches covering the first ``num_live_indices`` rows; later rows are left
    unwritten. ``live_multiple`` is a static promise that ``num_live_indices``
    is a multiple of it; when that multiple covers whole owner rounds the
    per-owner tail guard is dropped.

    ``ring_slots`` > 0 serves each owner with a hand-rolled DMA ring of that
    many atoms (`_ring_kernel`) instead of the nested ``emit_pipeline``: one
    64-row indirect DMA per cache per atom and ``ring_slots - 1`` atoms of
    gathers in flight across owner batches. Bit-identical output; ``wait_mode``
    and ``serve_pipelined`` are then unused. ``ring_slots`` must be ``0``
    (disabled) or a power of two ``>= 2``; when ``atoms_per_batch <= ring_slots``
    (e.g. small decode chunks), ``atoms_per_batch`` is not a power of two after
    the divisor fit below, the chip is not a generation-7 TPU, or ring scratch
    exceeds 80% of subcore VMEM, the ``emit_pipeline`` path is used (logging a
    one-time warning for unexpected fallbacks).
    """
    if wait_mode not in ("overlap_rope", "before_rope"):
        raise ValueError(f"unknown {wait_mode=}")
    if ring_slots != 0 and (ring_slots < 2 or (ring_slots & (ring_slots - 1)) != 0):
        raise ValueError(f"ring_slots ({ring_slots}) must be 0 or a power of two >= 2")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError("native-SC ROPE must be banded uint32[...,128]")
    if indices.dtype != jnp.int32 or indices.ndim != 2:
        raise ValueError("indices must be int32[query, topk]")
    if indices.size != out_size:
        raise ValueError(f"indices has {indices.size} values, expected {out_size}")

    nope_rows_view = nope_cache.reshape(-1, 128)
    num_tokens = nope_rows_view.shape[0]
    if num_tokens <= 0 or rope_cache.size % num_tokens != 0:
        raise ValueError(
            f"ROPE cache ({rope_cache.size} words) must divide evenly across "
            f"NOPE tokens ({num_tokens})"
        )
    rope_words = rope_cache.size // num_tokens
    if (
        rope_words <= 0
        or rope_words % NUM_LANES != 0
        or ROPE_PACKING * rope_words > 128
        or 128 % rope_words != 0
    ):
        raise ValueError(
            f"derived rope_words ({rope_words}) must be a positive multiple of "
            f"{NUM_LANES} dividing 128 with {ROPE_PACKING} * rope_words <= 128"
        )
    rope_rows_view = rope_cache.reshape(-1, rope_words)

    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    if sc_info is not None and sc_info.num_lanes != NUM_LANES:
        raise ValueError(
            f"dsa_gather_native_sc requires a {NUM_LANES}-lane SparseCore, "
            f"got {sc_info.num_lanes}"
        )
    num_cores = sc_info.num_cores if sc_info is not None else NUM_CORES
    num_subcores = sc_info.num_subcores if sc_info is not None else NUM_SUBCORES
    num_owners = num_cores * num_subcores

    max_possible_atoms = out_size // (num_owners * OCCURRENCES_PER_ATOM)
    if atoms_per_batch > max_possible_atoms > 0:
        atoms_per_batch = max_possible_atoms
    if atoms_per_batch <= 0:
        raise ValueError(
            f"atoms_per_batch ({atoms_per_batch}) must be > 0 "
            f"(out_size={out_size}, num_owners={num_owners})"
        )

    occurrences_per_batch = atoms_per_batch * OCCURRENCES_PER_ATOM
    owner_batch_size = num_owners * occurrences_per_batch
    if out_size % owner_batch_size != 0:
        raise ValueError(
            f"out_size ({out_size}) must be divisible by num_owners * "
            f"occurrences_per_batch ({owner_batch_size})"
        )
    num_batches = out_size // occurrences_per_batch
    batches_per_owner = num_batches // num_owners
    live = num_live_indices is not None
    ring_scratch_bytes = 4 * (
        ring_slots * OCCURRENCES_PER_ATOM * (128 + rope_words)
        + 2 * QUADS_PER_ATOM * 128
        + 2 * occurrences_per_batch
    )
    ring_vmem_budget = (
        int(sc_info.vmem_capacity_bytes * 0.8) if sc_info is not None else None
    )
    use_ring = False
    if ring_slots >= 2 and atoms_per_batch > ring_slots:
        if atoms_per_batch & (atoms_per_batch - 1) != 0:
            logger.warning_once(
                "dsa_gather_native_sc falling back to emit_pipeline: "
                "clamped atoms_per_batch=%d is not a power of two "
                "(ring_slots=%d, out_size=%d).",
                atoms_per_batch,
                ring_slots,
                out_size,
            )
        elif tpu_info is not None and tpu_info.generation != 7:
            logger.warning_once(
                "dsa_gather_native_sc falling back to emit_pipeline: "
                "DMA ring is only validated on generation-7 TPUs "
                "(got generation=%s, ring_slots=%d, atoms_per_batch=%d).",
                tpu_info.generation,
                ring_slots,
                atoms_per_batch,
            )
        elif ring_vmem_budget is not None and ring_scratch_bytes > ring_vmem_budget:
            logger.warning_once(
                "dsa_gather_native_sc falling back to emit_pipeline: "
                "ring_scratch_bytes=%d exceeds 80%% subcore VMEM budget=%d "
                "(ring_slots=%d, atoms_per_batch=%d, rope_words=%d).",
                ring_scratch_bytes,
                ring_vmem_budget,
                ring_slots,
                atoms_per_batch,
                rope_words,
            )
        else:
            use_ring = True
    # Logged once per gather configuration (not per token bucket), so a bug
    # report shows which gather served it.
    logger.info_once(
        "dsa_gather_native_sc: %s gather (ring_slots=%d, atoms_per_batch=%d, "
        "rope_words=%d).",
        "DMA-ring" if use_ring else "emit_pipeline",
        ring_slots,
        atoms_per_batch,
        rope_words,
    )
    if use_ring:
        scratch_types: dict[str, Any] = {
            "nope_buf": pltpu.VMEM((ring_slots, OCCURRENCES_PER_ATOM, 128), jnp.uint32),
            "rope_buf": pltpu.VMEM(
                (ring_slots, OCCURRENCES_PER_ATOM, rope_words), jnp.uint32
            ),
            "rope_out_buf": pltpu.VMEM((2, QUADS_PER_ATOM, 128), jnp.uint32),
            "idx_buf": pltpu.VMEM((2 * occurrences_per_batch,), jnp.int32),
            "gather_sem": pltpu.SemaphoreType.DMA((2, ring_slots)),
            "nope_out_sem": pltpu.SemaphoreType.DMA((ring_slots,)),
            "rope_out_sem": pltpu.SemaphoreType.DMA((2,)),
            "idx_sem": pltpu.SemaphoreType.DMA((2,)),
        }
        operands = [
            nope_rows_view,
            rope_rows_view,
            indices.reshape(num_batches, occurrences_per_batch),
        ]
    else:
        scratch_types = {"sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING,))}
        operands = [
            nope_rows_view,
            rope_rows_view,
            indices.reshape(num_batches, atoms_per_batch, ROPE_PACKING, NUM_LANES),
        ]
    guard_tail = False
    if live:
        # Live logical batches = cdiv(num_live_indices, occurrences/batch),
        # clamped to the static count; per-owner trip = cdiv(live, num_owners).
        live_b = jnp.clip(
            (jnp.asarray(num_live_indices, jnp.int32) + occurrences_per_batch - 1)
            // occurrences_per_batch,
            0,
            num_batches,
        )
        owner_b = (live_b + num_owners - 1) // num_owners
        # Lanes [0,16) = live logical batches, [16,32) = per-owner grid length.
        live_vec = jnp.concatenate(
            [
                jnp.full((NUM_LANES,), live_b, jnp.int32),
                jnp.full((NUM_LANES,), owner_b, jnp.int32),
            ]
        )
        operands.append(live_vec)
        scratch_types["live_vmem_ref"] = pltpu.VMEM((2 * NUM_LANES,), jnp.int32)
        scratch_types["live_sem_ref"] = pltpu.SemaphoreType.DMA((1,))
        # Per-owner guard is needed only if a live count can leave a ragged
        # last round, i.e. unless the caller promises num_live_indices is a
        # multiple of num_owners * occurrences_per_batch (at topk=2048,
        # atoms=1: 32 * 64 = 2048 = one token, so decode needs no guard).
        guard_tail = live_multiple % owner_batch_size != 0
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    if use_ring:
        body: Any = functools.partial(
            _ring_kernel,
            core_axis_name=mesh.core_axis_name,
            subcore_axis_name=mesh.subcore_axis_name,
            num_owners=num_owners,
            atoms_per_batch=atoms_per_batch,
            batches_per_owner=batches_per_owner,
            ring_slots=ring_slots,
            guard_tail=guard_tail,
            rope_words=rope_words,
        )
    else:
        body = functools.partial(
            _kernel,
            core_axis_name=mesh.core_axis_name,
            subcore_axis_name=mesh.subcore_axis_name,
            num_owners=num_owners,
            atoms_per_batch=atoms_per_batch,
            batches_per_owner=batches_per_owner,
            wait_mode=wait_mode,
            serve_pipelined=serve_pipelined,
            guard_tail=guard_tail,
            rope_words=rope_words,
        )
    if live:
        inner = body

        def body(nope, rope, idx, live_hbm, nope_o, rope_o, **scr):
            return inner(nope, rope, idx, nope_o, rope_o, live_hbm_ref=live_hbm, **scr)

    mode_tag = (
        f"ring{ring_slots}"
        if use_ring
        else f"{wait_mode}_{'pipe' if serve_pipelined else 'nopipe'}"
    )
    return core_map_helper.kernel(
        body,
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct((out_size // ROPE_PACKING, 128), jnp.uint32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=(
            f"sc_native_sc_gather_a{atoms_per_batch}_rw{rope_words}_{mode_tag}"
            f"{'_live' if live else ''}"
        ),
    )(*operands)
