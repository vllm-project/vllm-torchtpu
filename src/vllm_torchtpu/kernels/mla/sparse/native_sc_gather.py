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

NUM_CORES = 2
NUM_SUBCORES = 16
NUM_LANES = 16
ROPE_PACKING = 4
COMPACT_ROPE_PACKING = 8
QUADS_PER_ATOM = 16
OCCURRENCES_PER_ATOM = QUADS_PER_ATOM * ROPE_PACKING
COMPACT_ROWS_PER_ATOM = OCCURRENCES_PER_ATOM // COMPACT_ROPE_PACKING
ROPE_WORDS = 32  # one rope token on the SparseCore view of the rope cache
PACKED_ROPE_WORDS = 16
WRITE_NOPE_PART, WRITE_ROPE_PART, WRITE_DST_PART, WRITE_PARTS = 0, 1, 2, 3


def _delta_swap(x, y, shift, mask):
    tmp = jnp.bitwise_and(jnp.bitwise_xor(jnp.right_shift(x, shift), y), mask)
    return (
        jnp.bitwise_xor(x, jnp.left_shift(tmp, shift)),
        jnp.bitwise_xor(y, tmp),
    )


def live_atom_lanes(
    row_live_atoms: jax.Array, num_batches: int, atoms_per_batch: int, row_atoms: int
) -> jax.Array:
    """Per logical batch: [live atoms, exclusive live start of rows 1..R-1]."""
    per_row = jnp.clip(row_live_atoms.astype(jnp.int32), 0, row_atoms)
    if atoms_per_batch % row_atoms == 0:
        rows_per_batch = atoms_per_batch // row_atoms
        ends = jnp.cumsum(per_row.reshape(num_batches, rows_per_batch), axis=1)
        return jnp.concatenate([ends[:, -1:], ends[:, :-1]], axis=1)
    if row_atoms % atoms_per_batch == 0:
        split = row_atoms // atoms_per_batch
        first = (jnp.arange(num_batches, dtype=jnp.int32) % split) * atoms_per_batch
        live = jnp.clip(jnp.repeat(per_row, split) - first, 0, atoms_per_batch)
        return live[:, None]
    raise ValueError(
        f"atoms_per_batch ({atoms_per_batch}) and row atoms ({row_atoms}) "
        "must divide one another"
    )


def live_atom_of(step: Any, starts: list[Any], row_atoms: int) -> Any:
    """Maps a packed live step to its batch atom, skipping each row's dead tail."""
    shift = 0
    for r, start in enumerate(starts, 1):
        shift = jnp.where(step >= start, r * row_atoms - start, shift)
    return step + shift


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


def _wide_atom_factor(
    atoms_per_batch: int,
    *,
    wait_mode: str,
    serve_pipelined: bool,
    is_dcp_mode: bool,
    has_row_live_atoms: bool,
) -> int:
    # Avoid narrow 64-row indirect gathers on multi-atom batches: grouping up
    # to 4 base atoms (256 rows) per pipeline step increases HBM gather
    # concurrency without changing logical-batch or live_vec geometry.
    if (
        not serve_pipelined
        or wait_mode != "overlap_rope"
        or is_dcp_mode
        or has_row_live_atoms
    ):
        return 1
    if atoms_per_batch % 4 == 0:
        return 4
    if atoms_per_batch % 2 == 0:
        return 2
    return 1


def _kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    sem_ref: Any,
    live_vmem_ref: Any = None,
    live_sem_ref: Any = None,
    wide_nope_vm: Any = None,
    wide_rope_vm: Any = None,
    wide_rope_out_vm: Any = None,
    wide_in_nope_sem: Any = None,
    wide_in_rope_sem: Any = None,
    wide_out_sem: Any = None,
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
    is_dcp_mode: bool = False,
    row_atoms_hbm_ref: Any = None,
    row_atoms: int = 0,
    rows_per_batch: int = 1,
    linear_indices: bool = False,
    write_refs: tuple[Any, ...] | None = None,
    write_scratch: tuple[Any, ...] | None = None,
    write_tokens: int = 0,
    rope_words: int = ROPE_WORDS,
    wide_factor: int = 1,
    compact_rope: bool = False,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    if write_refs is not None:
        assert write_scratch is not None
        _write_phase(
            write_refs,
            write_scratch,
            nope_hbm_ref,
            rope_hbm_ref,
            subcore=lax.axis_index(subcore_axis_name),
            num_tokens=write_tokens,
        )
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

    def serve_batch(indices_ref, row_atoms_ref=None):
        owner_batch = pl.program_id(0)
        logical_batch = owner + owner_batch * num_owners
        occurrence_base = logical_batch * atoms_per_batch * OCCURRENCES_PER_ATOM

        num_atoms = atoms_per_batch
        starts = []
        if row_atoms_ref is not None:
            num_atoms = row_atoms_ref[pl.ds(0, NUM_LANES)][0]
            starts = [
                row_atoms_ref[pl.ds(r * NUM_LANES, NUM_LANES)][0]
                for r in range(1, rows_per_batch)
            ]

        def atom_of(step):
            return live_atom_of(step, starts, row_atoms)

        def values(atom, stream):
            if linear_indices:
                return indices_ref[
                    pl.ds(atom * OCCURRENCES_PER_ATOM + stream * NUM_LANES, NUM_LANES)
                ]
            return indices_ref[0, atom, stream, :]

        def destinations(atom, stream):
            return pl.ds(
                occurrence_base + atom * OCCURRENCES_PER_ATOM + stream * NUM_LANES,
                NUM_LANES,
            )

        def body(*refs):
            atom = atom_of(pl.program_id(0))
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

            if compact_rope:
                # Pack two quads (8 packed-rope tokens = 128 words, 0 pad) per
                # output row: lanes 0..63 hold tokens 0..31, lanes 64..127 hold
                # tokens 32..63 so TC unpacks each atom of 64 tokens as two (8,128)
                # tiles without sub-tile slicing.
                half_atom = OCCURRENCES_PER_ATOM // 2
                for row in range(COMPACT_ROWS_PER_ATOM):
                    data_l = []
                    data_r = []
                    for token_slot in range(ROPE_PACKING):
                        rank_l = row * ROPE_PACKING + token_slot
                        rank_r = half_atom + rank_l
                        data_l.append(
                            rope_inputs[rank_l // NUM_LANES][
                                pl.ds(rank_l % NUM_LANES, 1), pl.ds(0, rope_words)
                            ]
                        )
                        data_r.append(
                            rope_inputs[rank_r // NUM_LANES][
                                pl.ds(rank_r % NUM_LANES, 1), pl.ds(0, rope_words)
                            ]
                        )
                    data_l = transpose_quad(data_l)
                    data_r = transpose_quad(data_r)
                    for band in range(ROPE_PACKING):
                        rope_output[
                            pl.ds(row, 1), pl.ds(band * rope_words, rope_words)
                        ] = data_l[band]
                        rope_output[
                            pl.ds(row, 1),
                            pl.ds((ROPE_PACKING + band) * rope_words, rope_words),
                        ] = data_r[band]
            else:
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
                        rope_output[
                            pl.ds(row, 1), pl.ds(band * rope_words, rope_words)
                        ] = data[band]
                    # Packed rope fills only the first ROPE_PACKING * rope_words
                    # lanes; zero the rest, which the consumer reads as the
                    # (zero-padded) rope tail, as the padded cache stores it.
                    # Garbage there could decode to fp8 NaN, and NaN * 0 poisons
                    # the score against q's zero pad.
                    for band in range(ROPE_PACKING, 128 // rope_words):
                        rope_output[
                            pl.ds(row, 1), pl.ds(band * rope_words, rope_words)
                        ] = jnp.zeros((1, rope_words), jnp.uint32)

            if wait_mode == "overlap_rope":
                for copy in nope_copies:
                    copy.wait()

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), 128),
                lambda atom, stream=stream: (values(atom_of(atom), stream), 0),
            )
            for stream in range(ROPE_PACKING)
        )
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(NUM_LANES), rope_words),
                lambda atom, stream=stream: (values(atom_of(atom), stream), 0),
            )
            for stream in range(ROPE_PACKING)
        )
        rope_rows_per_atom = COMPACT_ROWS_PER_ATOM if compact_rope else QUADS_PER_ATOM
        rope_out_spec = pl.BlockSpec(
            (rope_rows_per_atom, 128),
            lambda atom: (logical_batch * atoms_per_batch + atom_of(atom), 0),
        )

        def run_inner_wide():
            super_atoms = atoms_per_batch // wide_factor
            occ_per_step = wide_factor * OCCURRENCES_PER_ATOM
            subchunks = occ_per_step // NUM_LANES
            rope_rows_per_step = wide_factor * rope_rows_per_atom
            rope_row_base = logical_batch * atoms_per_batch * rope_rows_per_atom

            def idx_lanes(step, sub):
                if linear_indices:
                    return indices_ref[
                        pl.ds(step * occ_per_step + sub * NUM_LANES, NUM_LANES)
                    ]
                return indices_ref[0, step, sub, :]

            # Avoid pl.Indirect(>16) offsets: older libtpu requires SparseCore
            # indirect DMA offsets to match the 16-lane hardware vector width.
            def start_fetch(step, slot):
                for sub in range(subchunks):
                    idx_v = idx_lanes(step, sub)
                    pltpu.make_async_copy(
                        nope_hbm_ref.at[idx_v, :],
                        wide_nope_vm.at[slot, pl.ds(sub * NUM_LANES, NUM_LANES), :],
                        wide_in_nope_sem.at[slot],
                    ).start()
                    pltpu.make_async_copy(
                        rope_hbm_ref.at[idx_v, :],
                        wide_rope_vm.at[slot, pl.ds(sub * NUM_LANES, NUM_LANES), :],
                        wide_in_rope_sem.at[slot],
                    ).start()

            def wait_fetch(step, slot):
                for sub in range(subchunks):
                    idx_v = idx_lanes(step, sub)
                    pltpu.make_async_copy(
                        nope_hbm_ref.at[idx_v, :],
                        wide_nope_vm.at[slot, pl.ds(sub * NUM_LANES, NUM_LANES), :],
                        wide_in_nope_sem.at[slot],
                    ).wait()
                    pltpu.make_async_copy(
                        rope_hbm_ref.at[idx_v, :],
                        wide_rope_vm.at[slot, pl.ds(sub * NUM_LANES, NUM_LANES), :],
                        wide_in_rope_sem.at[slot],
                    ).wait()

            zeros_band = jnp.zeros((1, rope_words), jnp.uint32)
            start_fetch(jnp.int32(0), jnp.int32(0))

            def run_step(step, carry):
                slot = lax.rem(step, jnp.int32(2))
                nxt = step + jnp.int32(1)
                nslot = jnp.int32(1) - slot

                wait_fetch(step, slot)

                @pl.when(nxt < jnp.int32(super_atoms))
                def _prefetch_next():
                    start_fetch(nxt, nslot)

                nope_cp = pltpu.make_async_copy(
                    wide_nope_vm.at[slot],
                    nope_out_hbm_ref.at[
                        pl.ds(occurrence_base + step * occ_per_step, occ_per_step), :
                    ],
                    wide_out_sem.at[0],
                )
                nope_cp.start()

                if compact_rope:

                    def transpose_compact_row(r_idx, _c):
                        atom_i = lax.div(r_idx, jnp.int32(COMPACT_ROWS_PER_ATOM))
                        in_atom = lax.rem(r_idx, jnp.int32(COMPACT_ROWS_PER_ATOM))
                        base_l = atom_i * jnp.int32(
                            OCCURRENCES_PER_ATOM
                        ) + in_atom * jnp.int32(ROPE_PACKING)
                        base_r = base_l + jnp.int32(OCCURRENCES_PER_ATOM // 2)
                        data_l = transpose_quad(
                            [
                                wide_rope_vm[
                                    slot, pl.ds(base_l + t, 1), pl.ds(0, rope_words)
                                ]
                                for t in range(ROPE_PACKING)
                            ]
                        )
                        data_r = transpose_quad(
                            [
                                wide_rope_vm[
                                    slot, pl.ds(base_r + t, 1), pl.ds(0, rope_words)
                                ]
                                for t in range(ROPE_PACKING)
                            ]
                        )
                        for band in range(ROPE_PACKING):
                            wide_rope_out_vm[
                                0, pl.ds(r_idx, 1), pl.ds(band * rope_words, rope_words)
                            ] = data_l[band]
                            wide_rope_out_vm[
                                0,
                                pl.ds(r_idx, 1),
                                pl.ds((ROPE_PACKING + band) * rope_words, rope_words),
                            ] = data_r[band]
                        return _c

                    lax.fori_loop(
                        0, rope_rows_per_step, transpose_compact_row, jnp.int32(0)
                    )
                else:

                    def transpose_row(q_idx, _c):
                        base_t = q_idx * ROPE_PACKING
                        data = [
                            wide_rope_vm[
                                slot, pl.ds(base_t + t, 1), pl.ds(0, rope_words)
                            ]
                            for t in range(ROPE_PACKING)
                        ]
                        data = transpose_quad(data)
                        for band in range(ROPE_PACKING):
                            wide_rope_out_vm[
                                0, pl.ds(q_idx, 1), pl.ds(band * rope_words, rope_words)
                            ] = data[band]
                        for band in range(ROPE_PACKING, 128 // rope_words):
                            wide_rope_out_vm[
                                0, pl.ds(q_idx, 1), pl.ds(band * rope_words, rope_words)
                            ] = zeros_band
                        return _c

                    # Avoid unrolling 64 quads in Python: lax.fori_loop keeps
                    # compile time ~0.2s and eliminates code-size bloat on SC.
                    lax.fori_loop(0, rope_rows_per_step, transpose_row, jnp.int32(0))

                rope_cp = pltpu.make_async_copy(
                    wide_rope_out_vm.at[0],
                    rope_out_hbm_ref.at[
                        pl.ds(
                            rope_row_base + step * rope_rows_per_step,
                            rope_rows_per_step,
                        ),
                        :,
                    ],
                    wide_out_sem.at[1],
                )
                rope_cp.start()
                nope_cp.wait()
                rope_cp.wait()
                return carry

            lax.fori_loop(0, super_atoms, run_step, jnp.int32(0))

        def run_inner():
            if wide_factor > 1:
                run_inner_wide()
                return
            pltpu.emit_pipeline(
                body,
                grid=(num_atoms,),
                in_specs=nope_specs + rope_specs,
                out_specs=(rope_out_spec,),
                no_pipelining=not serve_pipelined,
            )(
                *([nope_hbm_ref] * ROPE_PACKING),
                *([rope_hbm_ref] * ROPE_PACKING),
                rope_out_hbm_ref,
            )

        pred = None
        if is_dcp_mode:
            pred = values(0, 0)[0] >= 0
        if live_batches is not None and guard_tail:
            # Owners share a uniform traced grid of cdiv(live, num_owners);
            # the ragged last round is skipped per owner.
            tail_ok = logical_batch < live_batches
            pred = tail_ok if pred is None else (pred & tail_ok)
        if row_atoms_ref is not None:
            pred = num_atoms > 0 if pred is None else (pred & (num_atoms > 0))
        if pred is None:
            run_inner()
        else:
            pl.when(pred)(run_inner)

    if linear_indices:
        indices_spec = pl.BlockSpec(
            (atoms_per_batch * OCCURRENCES_PER_ATOM,),
            lambda batch: (owner + batch * num_owners,),
        )
    elif wide_factor > 1:
        indices_spec = pl.BlockSpec(
            (
                1,
                atoms_per_batch // wide_factor,
                wide_factor * ROPE_PACKING,
                NUM_LANES,
            ),
            lambda batch: (owner + batch * num_owners, 0, 0, 0),
        )
    else:
        indices_spec = pl.BlockSpec(
            (1, atoms_per_batch, ROPE_PACKING, NUM_LANES),
            lambda batch: (owner + batch * num_owners, 0, 0, 0),
        )
    in_specs = (indices_spec,)
    in_refs = (indices_hbm_ref,)
    if row_atoms_hbm_ref is not None:
        in_specs += (
            pl.BlockSpec(
                (rows_per_batch * NUM_LANES,),
                lambda batch: (owner + batch * num_owners,),
            ),
        )
        in_refs += (row_atoms_hbm_ref,)
    if live_batches is None:
        pltpu.emit_pipeline(
            serve_batch,
            grid=(batches_per_owner,),
            in_specs=in_specs,
        )(*in_refs)
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
                in_specs=in_specs,
            )(*in_refs)


def write_slots(num_tokens: int) -> int:
    """Write-phase scratch slots (dst vectors) for a padded token count."""
    return num_tokens // NUM_SUBCORES


def _write_phase(
    write_refs: tuple[Any, ...],
    write_scratch: tuple[Any, ...],
    nope_cache_ref: Any,
    rope_cache_ref: Any,
    *,
    subcore: jax.Array,
    num_tokens: int,
) -> None:
    (rows_src,) = write_refs
    dst_vmem, dst_sem, write_sem = write_scratch
    num_rows = nope_cache_ref.shape[0]
    rope_words = rope_cache_ref.shape[-1]
    rounds = pl.cdiv(num_tokens, NUM_SUBCORES)
    copies = []
    for k in range(rounds):
        token = subcore + k * NUM_SUBCORES
        cp = pltpu.make_async_copy(
            rows_src.bitcast(jnp.int32).at[
                pl.ds(WRITE_DST_PART * num_tokens + token, 1), pl.ds(0, NUM_LANES)
            ],
            dst_vmem.at[pl.ds(k, 1), :],
            dst_sem.at[k],
        )
        cp.start()
        cp.wait()
        dst = dst_vmem[k, :][0] - 1  # stored as dst + 1 (< 2**31), 0 = skip
        write = jnp.logical_and(dst >= 0, dst < num_rows)
        row = jnp.clip(dst, 0, num_rows - 1)
        nope_cp = pltpu.make_async_copy(
            rows_src.at[pl.ds(WRITE_NOPE_PART * num_tokens + token, 1)],
            nope_cache_ref.at[pl.ds(row, 1)],
            write_sem.at[2 * k],
        )
        rope_cp = pltpu.make_async_copy(
            rows_src.at[
                pl.ds(WRITE_ROPE_PART * num_tokens + token, 1), pl.ds(0, rope_words)
            ],
            rope_cache_ref.at[pl.ds(row, 1)],
            write_sem.at[2 * k + 1],
        )

        @pl.when(write)
        def _(nope_cp=nope_cp, rope_cp=rope_cp):
            nope_cp.start()
            rope_cp.start()

        copies.append((write, nope_cp, rope_cp))

    for write, nope_cp, rope_cp in copies:

        @pl.when(write)
        def _(nope_cp=nope_cp, rope_cp=rope_cp):
            nope_cp.wait()
            rope_cp.wait()

    plsc.subcore_barrier()


def pack_write_rows(
    dst_rows: jax.Array, nope_rows: jax.Array, rope_rows: jax.Array
) -> jax.Array:
    """The write phase's single operand, u32[3 * Tp, 128] (see `_write_phase`)."""
    t = dst_rows.shape[0]
    if not 0 < t <= 2 * NUM_SUBCORES:
        raise ValueError(f"kv_write takes 1..{2 * NUM_SUBCORES} tokens")
    if nope_rows.shape != (t, 128) or nope_rows.dtype != jnp.uint32:
        raise ValueError(f"nope_rows must be uint32[{t}, 128]")
    rope_widths = ((t, PACKED_ROPE_WORDS), (t, ROPE_WORDS), (t, 128))
    if rope_rows.shape not in rope_widths or rope_rows.dtype != jnp.uint32:
        raise ValueError(f"rope_rows must be uint32[{t}, 16, 32 or 128]")
    if rope_rows.shape[1] != 128:
        rope_rows = jnp.pad(rope_rows, ((0, 0), (0, 128 - rope_rows.shape[1])))
    tp = pl.cdiv(t, NUM_SUBCORES) * NUM_SUBCORES
    enc = jnp.where(dst_rows >= 0, dst_rows + 1, 0).astype(jnp.uint32)
    parts = [nope_rows, rope_rows, jnp.broadcast_to(enc[:, None], (t, 128))]
    return jnp.concatenate([jnp.pad(x, ((0, tp - t), (0, 0))) for x in parts], 0)


def live_vec_geometry(out_size: int, atoms_per_batch: int) -> tuple[int, ...]:
    """Static gather geometry exactly as `dsa_gather_native_sc` resolves it."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
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
    return num_cores, num_subcores, atoms_per_batch, occurrences_per_batch, num_batches


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "atoms_per_batch",
        "wait_mode",
        "serve_pipelined",
        "live_multiple",
        "is_dcp_mode",
        "linear_indices",
        "live_vec_offset",
        "write_fault",
        "packed_rope",
        "compact_rope",
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
    is_dcp_mode: bool = False,
    row_live_atoms: jax.Array | None = None,
    live_vec: jax.Array | None = None,
    linear_indices: bool = False,
    live_vec_offset: int | None = None,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    write_fault: str | None = None,
    packed_rope: bool | None = None,
    compact_rope: bool = False,
) -> tuple[jax.Array, ...]:
    """Gather native-SC ``uint32`` NOPE and ROPE cache rows with optional KV write."""
    if wait_mode not in ("overlap_rope", "before_rope"):
        raise ValueError(f"unknown {wait_mode=}")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError("native-SC ROPE must be quarter-major uint32[...,128]")
    if indices.dtype != jnp.int32 or indices.ndim != 2:
        raise ValueError("indices must be int32[query, topk]")
    if live_vec_offset is None:
        if indices.size != out_size:
            raise ValueError(f"indices has {indices.size} values, expected {out_size}")
    else:
        if not linear_indices:
            raise ValueError("live_vec_offset requires linear_indices")
        if live_vec_offset < out_size or indices.size < live_vec_offset + 2 * NUM_LANES:
            raise ValueError(
                f"live_vec_offset ({live_vec_offset}) must place the live vector "
                f"past the {out_size} indices, inside {indices.size} values"
            )

    if sum(x is not None for x in (num_live_indices, live_vec, live_vec_offset)) > 1:
        raise ValueError("pass one of num_live_indices, live_vec, live_vec_offset")
    nope_rows_view: Any = nope_cache.reshape(-1, 128)
    num_tokens_in_cache = nope_rows_view.shape[0]
    if num_tokens_in_cache <= 0 or rope_cache.size % num_tokens_in_cache != 0:
        raise ValueError(
            f"rope_cache ({rope_cache.shape}) must divide evenly across "
            f"{num_tokens_in_cache} NOPE tokens"
        )
    rope_words = rope_cache.size // num_tokens_in_cache
    if (
        rope_words <= 0
        or rope_words % NUM_LANES != 0
        or 128 % rope_words != 0
        or ROPE_PACKING * rope_words > 128
    ):
        raise ValueError(
            f"derived rope_words ({rope_words}) must be a positive multiple of "
            f"{NUM_LANES} dividing 128 with {ROPE_PACKING} * rope_words <= 128"
        )
    if packed_rope is False and compact_rope:
        raise ValueError("compact_rope requires packed_rope=True")
    if packed_rope is not None:
        expected_rw = PACKED_ROPE_WORDS if packed_rope else ROPE_WORDS
        if rope_words != expected_rw:
            raise ValueError(
                f"packed_rope={packed_rope} expects {expected_rw}-word rope tokens, "
                f"got {rope_words}"
            )
    if compact_rope and rope_words != PACKED_ROPE_WORDS:
        raise ValueError("compact_rope requires packed_rope=True")
    rope_rows_view: Any = rope_cache.reshape(-1, rope_words)
    num_cores, num_subcores, atoms_per_batch, occurrences_per_batch, num_batches = (
        live_vec_geometry(out_size, atoms_per_batch)
    )
    num_owners = num_cores * num_subcores
    owner_batch_size = num_owners * occurrences_per_batch
    batches_per_owner = num_batches // num_owners
    live = (
        num_live_indices is not None
        or live_vec is not None
        or live_vec_offset is not None
    )
    scratch_types = {"sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING,))}
    rope_out_packing = COMPACT_ROPE_PACKING if compact_rope else ROPE_PACKING
    write_tokens = 0
    if kv_write is not None:
        nope_rows_view = jax.new_ref(nope_rows_view)
        rope_rows_view = jax.new_ref(rope_rows_view)
    wide_factor = _wide_atom_factor(
        atoms_per_batch,
        wait_mode=wait_mode,
        serve_pipelined=serve_pipelined,
        is_dcp_mode=is_dcp_mode,
        has_row_live_atoms=row_live_atoms is not None,
    )
    if wide_factor > 1:
        occ_per_step = wide_factor * OCCURRENCES_PER_ATOM
        rope_rows_per_step = occ_per_step // rope_out_packing
        scratch_types["wide_nope_vm"] = pltpu.VMEM((2, occ_per_step, 128), jnp.uint32)
        scratch_types["wide_rope_vm"] = pltpu.VMEM(
            (2, occ_per_step, rope_words), jnp.uint32
        )
        scratch_types["wide_rope_out_vm"] = pltpu.VMEM(
            (1, rope_rows_per_step, 128), jnp.uint32
        )
        scratch_types["wide_in_nope_sem"] = pltpu.SemaphoreType.DMA((2,))
        scratch_types["wide_in_rope_sem"] = pltpu.SemaphoreType.DMA((2,))
        scratch_types["wide_out_sem"] = pltpu.SemaphoreType.DMA((2,))
    if linear_indices:
        indices_operand = indices.reshape(-1)
    elif wide_factor > 1:
        indices_operand = indices.reshape(
            num_batches,
            atoms_per_batch // wide_factor,
            wide_factor * ROPE_PACKING,
            NUM_LANES,
        )
    else:
        indices_operand = indices.reshape(
            num_batches, atoms_per_batch, ROPE_PACKING, NUM_LANES
        )
    operands = [
        nope_rows_view,
        rope_rows_view,
        indices_operand,
    ]
    guard_tail = False
    if live_vec_offset is not None:
        scratch_types["live_vmem_ref"] = pltpu.VMEM((2 * NUM_LANES,), jnp.int32)
        scratch_types["live_sem_ref"] = pltpu.SemaphoreType.DMA((1,))
        guard_tail = live_multiple % owner_batch_size != 0
    elif live_vec is not None:
        if live_vec.shape != (2 * NUM_LANES,) or live_vec.dtype != jnp.int32:
            raise ValueError(f"live_vec must be int32[{2 * NUM_LANES}]")
        operands.append(live_vec)
        scratch_types["live_vmem_ref"] = pltpu.VMEM((2 * NUM_LANES,), jnp.int32)
        scratch_types["live_sem_ref"] = pltpu.SemaphoreType.DMA((1,))
        guard_tail = live_multiple % owner_batch_size != 0
    elif live:
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
    row_kwargs = {}
    if row_live_atoms is not None:
        num_rows = row_live_atoms.shape[0] if row_live_atoms.ndim == 1 else 0
        row_len = out_size // num_rows if num_rows else 0
        if (
            not num_rows
            or num_rows * row_len != out_size
            or row_len % OCCURRENCES_PER_ATOM
        ):
            raise ValueError("row_live_atoms must be i32[query] with 64-slot rows")
        row_atoms = row_len // OCCURRENCES_PER_ATOM
        vals = live_atom_lanes(row_live_atoms, num_batches, atoms_per_batch, row_atoms)
        rows_per_batch = vals.shape[1]
        operands.append(
            jnp.broadcast_to(
                vals[:, :, None], (num_batches, rows_per_batch, NUM_LANES)
            ).reshape(-1)
        )
        row_kwargs = dict(row_atoms=row_atoms, rows_per_batch=rows_per_batch)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    num_base = len(operands)  # nope, rope, indices[, live_vec][, row atoms]
    if kv_write is not None:
        if not isinstance(kv_write, jax.Array) and kv_write[2].shape[-1] not in (
            rope_words,
            128,
        ):
            raise ValueError(
                f"kv_write rope rows are {kv_write[2].shape[-1]} words; this rope "
                f"cache takes {rope_words}-word tokens (or 128-lane rows)"
            )
        write_rows = (
            kv_write if isinstance(kv_write, jax.Array) else pack_write_rows(*kv_write)
        )
        if num_subcores != NUM_SUBCORES:
            raise ValueError(f"kv_write assumes {NUM_SUBCORES} subcores per core")
        padded = write_rows.shape[0] // WRITE_PARTS
        if (
            write_rows.shape != (WRITE_PARTS * padded, 128)
            or write_rows.dtype != jnp.uint32
            or not 0 < padded <= 2 * NUM_SUBCORES
            or padded % NUM_SUBCORES
        ):
            raise ValueError(
                f"write_rows must be uint32[3 * Tp, 128], Tp in (16, 32); got "
                f"{write_rows.shape} {write_rows.dtype}"
            )
        if row_kwargs and padded > NUM_SUBCORES:
            raise ValueError(
                f"row_live_atoms with a {padded}-token kv_write (more than "
                f"{NUM_SUBCORES}) hangs the kernel; drop one of them"
            )
        operands.append(write_rows)
        rounds = write_slots(padded)
        scratch_types["write_dst_vmem"] = pltpu.VMEM((rounds, NUM_LANES), jnp.int32)
        scratch_types["write_dst_sem"] = pltpu.SemaphoreType.DMA((rounds,))
        scratch_types["write_sem"] = pltpu.SemaphoreType.DMA((2 * rounds,))
        write_tokens = padded
    inner = functools.partial(
        _kernel,
        core_axis_name=mesh.core_axis_name,
        subcore_axis_name=mesh.subcore_axis_name,
        num_owners=num_owners,
        atoms_per_batch=atoms_per_batch,
        batches_per_owner=batches_per_owner,
        wait_mode=wait_mode,
        serve_pipelined=serve_pipelined,
        guard_tail=guard_tail,
        is_dcp_mode=is_dcp_mode,
        linear_indices=linear_indices,
        write_tokens=write_tokens,
        **row_kwargs,
        rope_words=rope_words,
        wide_factor=wide_factor,
        compact_rope=compact_rope,
    )

    def body(*refs, write_dst_vmem=None, write_dst_sem=None, write_sem=None, **scr):
        ins, (nope_o, rope_o) = refs[:-2], refs[-2:]
        nope, rope, idx = ins[:3]
        extras = list(ins[3:num_base])  # [live_vec][, row atoms]
        kw = {}
        if live_vec_offset is not None:
            kw["live_hbm_ref"] = idx.at[pl.ds(live_vec_offset, 2 * NUM_LANES)]
        elif live:
            kw["live_hbm_ref"] = extras.pop(0)
        if row_kwargs:
            kw["row_atoms_hbm_ref"] = extras.pop(0)
        if kv_write is not None:
            kw["write_refs"] = tuple(ins[num_base:])
            kw["write_scratch"] = (write_dst_vmem, write_dst_sem, write_sem)
        return inner(nope, rope, idx, nope_o, rope_o, **kw, **scr)

    kernel_fn: Any = pl.kernel if kv_write is not None else core_map_helper.kernel
    outs: tuple[jax.Array, jax.Array] = kernel_fn(
        body,
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct((out_size // rope_out_packing, 128), jnp.uint32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=(
            f"sc_native_sc_gather_a{atoms_per_batch}_rw{rope_words}_{wait_mode}_"
            f"{'pipe' if serve_pipelined else 'nopipe'}{'_live' if live else ''}"
            f"{'_rowlive' if row_kwargs else ''}"
            f"{'_kvw' if kv_write is not None else ''}"
            f"{'_compactrope' if compact_rope else ''}"
            f"{f'_w{wide_factor}' if wide_factor > 1 else ''}"
        ),
    )(*operands)
    if kv_write is None:
        return outs
    return (
        *outs,
        jax.freeze(nope_rows_view).reshape(nope_cache.shape),
        jax.freeze(rope_rows_view).reshape(rope_cache.shape),
    )
