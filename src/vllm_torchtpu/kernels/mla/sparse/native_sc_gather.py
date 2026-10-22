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
it every row is gathered. An optional per-row live-atom count
(``row_live_atoms``) further skips each index row's dead 64-slot atoms.

An optional write phase (``kv_write``, TPU_MLA_KV_WRITE_IN_GATHER) first DMAs
this step's packed new tokens into both caches in place, then fences, so the
gather reads them: the decode step's KV-cache write rides the gather's launch
instead of a TensorCore kernel of its own.
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
QUADS_PER_ATOM = 16
OCCURRENCES_PER_ATOM = QUADS_PER_ATOM * ROPE_PACKING
ROPE_WORDS = 32  # one rope token on the SparseCore view of the rope cache
# One rope token under packed rope (`TPU_SPARSE_MLA_PACKED_ROPE`): its 64 fp8
# bytes alone, eight tokens per 128-word cache row instead of four.
PACKED_ROPE_WORDS = 16
# `write_rows` (the write phase's one operand), u32[3 * T, 128]: row blocks.
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
        # A row spans several batches; each keeps its share of the live prefix.
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
    linear_indices: bool = False,
    write_refs: tuple[Any, ...] | None = None,
    write_scratch: tuple[Any, ...] | None = None,
    write_tokens: int = 0,
    write_fault: str | None = None,
    rope_words: int = ROPE_WORDS,
    row_atoms_hbm_ref: Any = None,
    row_atoms: int = 0,
    rows_per_batch: int = 1,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    late_drain = None
    if write_refs is not None and write_fault != "late_write":
        assert write_scratch is not None
        late_drain = _write_phase(
            write_refs,
            write_scratch,
            nope_hbm_ref,
            rope_hbm_ref,
            subcore=lax.axis_index(subcore_axis_name),
            num_tokens=write_tokens,
            fault=write_fault,
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
            # Lane block k broadcasts [live atoms, start of row 1, ...]; a scalar
            # comes only from a rank-1 (16,) load + element extract on SC.
            num_atoms = row_atoms_ref[pl.ds(0, NUM_LANES)][0]
            starts = [
                row_atoms_ref[pl.ds(r * NUM_LANES, NUM_LANES)][0]
                for r in range(1, rows_per_batch)
            ]

        def atom_of(step):
            return live_atom_of(step, starts, row_atoms)

        def values(atom, stream):
            if linear_indices:
                # 1-D operand, one logical batch per block (dsa_gather.py's form).
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
        rope_out_spec = pl.BlockSpec(
            (QUADS_PER_ATOM, 128),
            lambda atom: (logical_batch * atoms_per_batch + atom_of(atom), 0),
        )

        def run_inner():
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

        run = None
        if live_batches is not None and guard_tail:
            # Owners share a uniform traced grid of cdiv(live, num_owners);
            # the ragged last round is skipped per owner.
            run = logical_batch < live_batches
        if row_atoms_ref is not None:
            run = num_atoms > 0 if run is None else jnp.logical_and(run, num_atoms > 0)
        if run is None:
            run_inner()
        else:
            pl.when(run)(run_inner)

    if linear_indices:
        indices_spec = pl.BlockSpec(
            (atoms_per_batch * OCCURRENCES_PER_ATOM,),
            lambda batch: (owner + batch * num_owners,),
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

    if late_drain is not None:
        late_drain()
    if write_refs is not None and write_fault == "late_write":
        # Positive control: the write lands after the gather (stale reads).
        assert write_scratch is not None
        _write_phase(
            write_refs,
            write_scratch,
            nope_hbm_ref,
            rope_hbm_ref,
            subcore=lax.axis_index(subcore_axis_name),
            num_tokens=write_tokens,
            fault=None,
        )


# Test-only faults of the write phase (positive controls).
WRITE_FAULTS = (
    None,
    "skip_write",
    "wrong_row",
    "ignore_owner",
    # Subcore 0 of each core writes every token and nobody waits or fences
    # before gathering (a single-writer design without its fence).
    "no_fence",
    # The writes are issued only after the gather (write/gather misordered).
    "late_write",
)


def write_slots(num_tokens: int, fault: str | None) -> int:
    """Write-phase scratch slots (dst vectors) for a padded token count."""
    return num_tokens if fault == "no_fence" else num_tokens // NUM_SUBCORES


def _write_phase(
    write_refs: tuple[Any, ...],
    write_scratch: tuple[Any, ...],
    nope_cache_ref: Any,
    rope_cache_ref: Any,
    *,
    subcore: jax.Array,
    num_tokens: int,
    fault: str | None,
) -> Any:
    """Write this step's packed tokens into both caches, then fence.

    Every SparseCore core writes every token: its `NUM_SUBCORES` subcores
    split the tokens (subcore s takes tokens s, s + 16, ...), wait their own
    DMAs, and meet at `subcore_barrier`. So every row a core's gather reads
    was written -- and completed -- by that core before its first gather DMA
    (the write-then-barrier-then-read pattern of upstream
    tests/pallas/tpu_sparsecore_pallas_test.py:2227-2240 @ce57049). The other
    core writes the same rows with identical bytes, so the race between the
    two cores' writes and either core's reads cannot change a byte, and no
    cross-core fence is needed.

    On a SparseCore view a rope token is its own 32-word row (the gather's
    `rope_cache.reshape(-1, 32)`; 16 words under packed rope), so unlike the
    TensorCore writer there is no read-modify-write of the shared 512 B row.

    Everything arrives in one operand, `write_rows` u32[3 * tokens, 128] (see
    `pack_write_rows`): row `t` the packed nope token, row `tokens + t` the
    packed rope token in words [0, 32) ([0, 16) under packed rope: the width
    of `rope_cache_ref`'s rows), row `2 * tokens + t` `dst + 1`
    broadcast (0 = skip: padding, or a row another DCP rank owns). One operand
    because every TensorCore-produced SparseCore operand costs the producer
    ~0.6 us at decode sizes (twin j-488b9087 / j-44074daf: one such op per
    operand per layer); 128-lane rows so the TensorCore (8, 128) tiling is
    already the SparseCore's linear layout (a [T, 384] operand cost a 0.6 us
    relayout, j-241c6f15). A SparseCore scalar comes from lane 0 of a VMEM
    vector.
    """
    (rows_src,) = write_refs
    dst_vmem, dst_sem, write_sem = write_scratch
    num_rows = nope_cache_ref.shape[0]
    # One rope token per row of the SparseCore rope view (ROPE_WORDS, or
    # PACKED_ROPE_WORDS under packed rope).
    rope_words = rope_cache_ref.shape[-1]
    single_writer = fault == "no_fence"
    rounds = num_tokens if single_writer else pl.cdiv(num_tokens, NUM_SUBCORES)
    copies = []
    for k in range(rounds):
        token = jnp.int32(k) if single_writer else subcore + k * NUM_SUBCORES
        cp = pltpu.make_async_copy(
            rows_src.bitcast(jnp.int32).at[
                pl.ds(WRITE_DST_PART * num_tokens + token, 1), pl.ds(0, NUM_LANES)
            ],
            dst_vmem.at[pl.ds(k, 1), :],
            # The single-writer control shares one semaphore set (SparseCore
            # has ~59 words of semaphores); each wait names its own copy.
            dst_sem.at[0 if single_writer else k],
        )
        cp.start()
        cp.wait()
        dst = dst_vmem[k, :][0] - 1  # stored as dst + 1 (< 2**31), 0 = skip
        if fault == "wrong_row":
            dst = dst + 1
        write = jnp.logical_and(dst >= 0, dst < num_rows)
        if fault == "ignore_owner":
            # A non-owned token written anyway (to row 0 of this shard).
            write = jnp.logical_and(token < num_tokens, dst < num_rows)
        if single_writer:
            write = jnp.logical_and(write, subcore == 0)
        row = jnp.clip(dst, 0, num_rows - 1)
        nope_cp = pltpu.make_async_copy(
            rows_src.at[pl.ds(WRITE_NOPE_PART * num_tokens + token, 1)],
            nope_cache_ref.at[pl.ds(row, 1)],
            write_sem.at[0 if single_writer else 2 * k],
        )
        rope_cp = pltpu.make_async_copy(
            rows_src.at[
                pl.ds(WRITE_ROPE_PART * num_tokens + token, 1), pl.ds(0, rope_words)
            ],
            rope_cache_ref.at[pl.ds(row, 1)],
            write_sem.at[1 if single_writer else 2 * k + 1],
        )
        if fault == "skip_write":
            continue

        @pl.when(write)
        def _(nope_cp=nope_cp, rope_cp=rope_cp):
            nope_cp.start()
            rope_cp.start()

        copies.append((write, nope_cp, rope_cp))

    def drain():
        for write, nope_cp, rope_cp in copies:

            @pl.when(write)
            def _(nope_cp=nope_cp, rope_cp=rope_cp):
                nope_cp.wait()
                rope_cp.wait()

    if fault == "no_fence":
        # Gather while the writes are still in flight: no wait, no barrier
        # before the gather; the caller drains at the end of the kernel.
        return drain
    drain()
    plsc.subcore_barrier()
    return None


def pack_write_rows(
    dst_rows: jax.Array, nope_rows: jax.Array, rope_rows: jax.Array
) -> jax.Array:
    """The write phase's single operand, u32[3 * Tp, 128] (see `_write_phase`).

    `dst_rows` i32[T] (negative = skip), `nope_rows` u32[T, 128] and
    `rope_rows` u32[T, 32 or 128] (token in words [0, 32), rest zero), packed
    for the SparseCore cache (`kv_cache_utils.pack_tokens`). Under packed rope
    the token is 16 words: u32[T, 16 or 128], words [0, 16). Tokens are padded
    to `Tp`, a whole number of subcore rounds (16); padding rows are skipped.
    """
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
    """Static gather geometry exactly as `dsa_gather_native_sc` resolves it.

    Returns ``(num_cores, num_subcores, atoms_per_batch, occurrences_per_batch,
    num_batches)``: `atoms_per_batch` after the clamp to what `out_size` can
    feed every owner, and the logical-batch count. A caller building
    ``live_vec`` itself uses these, so its live bound cannot drift from the
    one this module would derive from ``num_live_indices``.
    """
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
        "linear_indices",
        "live_vec_offset",
        "write_fault",
        "packed_rope",
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
    live_vec: jax.Array | None = None,
    linear_indices: bool = False,
    live_vec_offset: int | None = None,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    write_fault: str | None = None,
    packed_rope: bool = False,
    row_live_atoms: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Gather explicit native-SC ``uint32`` NOPE and ROPE cache rows.

    ``nope_cache`` has one 128-word row per token. ``rope_cache`` has one
    128-word row per group of four tokens; each token owns a consecutive
    32-word quarter. The returned ROPE is register-transposed into the grouped
    TensorCore format expected by the shared attention consumer.

    ``num_live_indices`` (traced i32 scalar, optional) gathers only the logical
    batches covering the first ``num_live_indices`` rows; later rows are left
    unwritten. ``live_multiple`` is a static promise that ``num_live_indices``
    is a multiple of it; when that multiple covers whole owner rounds the
    per-owner tail guard is dropped.

    ``live_vec`` (traced i32[2 * NUM_LANES], optional; exclusive with
    ``num_live_indices``) is that live bound already in the kernel's own
    operand form -- lanes [0, 16) the live logical-batch count, [16, 32) the
    per-owner trip count, as `live_vec_geometry` defines them -- for a caller
    that computes it inside its own kernel (`prepare_capped_window_fused`).

    ``linear_indices`` hands the indices to the kernel as one flat i32[out_size]
    operand instead of [batches, atoms, 4, 16]. XLA lays a 1-D i32 array out
    linearly, which is the SparseCore operand layout, and a 128-lane 2-D
    [n, 128] producer output reshapes to it as a bitcast; the 4-D reshape
    otherwise costs two relayout copies (T(4,128), then linear). Same rows.

    ``live_vec_offset`` (static, with ``linear_indices``; exclusive with the
    other live forms): ``live_vec`` is read from the flat ``indices`` operand
    itself, at ``[live_vec_offset, live_vec_offset + 2 * NUM_LANES)``, past
    the ``out_size`` gathered indices -- the layout
    `prepare_capped_window_fused` writes, so its single output carries both.

    ``kv_write`` (TPU_MLA_KV_WRITE_IN_GATHER) = ``(dst_rows, nope_rows,
    rope_rows)``: i32[T] flattened cache rows (`page * page_size + slot`,
    negative = skip), u32[T, 128] and u32[T, 32 or 128] (16 or 128 under
    ``packed_rope``) tokens already packed for the SparseCore cache
    (`kv_cache_utils.pack_tokens`; a 128-wide rope operand carries the token
    in words [0, 32), or [0, 16), and avoids a relayout), or
    those three already in the kernel's one-operand form, `pack_write_rows`'s
    u32[3 * Tp, 128] (what the serving path hands over: one TensorCore
    producer). They are written into
    both caches in place before anything is gathered, and the call then also
    returns the updated ``(nope_cache, rope_cache)``. Bytes identical to
    `scatter.scatter` / `tc_scatter.scatter_caches` with the same rows.

    ``packed_rope`` reads (and ``kv_write`` writes) a ROPE cache whose tokens
    own 16-word eighths of a row (`SparseMLAKVCacheSpec.packed_rope`) instead
    of 32-word quarters. The gathered output is the same: the rope lanes past
    each token's 64 bytes are zeroed, exactly as the padded cache stores them.

    ``row_live_atoms`` (traced i32[query], optional) gathers only the first
    ``row_live_atoms[i]`` 64-slot atoms of index row ``i``; the other atoms'
    output rows are left unwritten. Composes with ``num_live_indices``; needs
    the 2-D (non-linear) index operand.
    """
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
    if write_fault not in WRITE_FAULTS:
        raise ValueError(f"unknown {write_fault=}")
    rope_words = PACKED_ROPE_WORDS if packed_rope else ROPE_WORDS
    nope_rows_view = nope_cache.reshape(-1, 128)
    rope_rows_view = rope_cache.reshape(-1, rope_words)
    if packed_rope and rope_rows_view.shape[0] != nope_rows_view.shape[0]:
        raise ValueError(
            f"packed rope cache holds {rope_rows_view.shape[0]} tokens, the NOPE "
            f"cache {nope_rows_view.shape[0]}"
        )
    write_tokens = 0
    if kv_write is not None:
        # The caches become refs the kernel writes in place (as
        # `scatter_utils.scatter_rows` does); frozen back after the call.
        nope_rows_view = jax.new_ref(nope_rows_view)
        rope_rows_view = jax.new_ref(rope_rows_view)
    operands = [
        nope_rows_view,
        rope_rows_view,
        (
            indices.reshape(-1)
            if linear_indices
            else indices.reshape(num_batches, atoms_per_batch, ROPE_PACKING, NUM_LANES)
        ),
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
        num_rows, row_len = indices.shape
        if linear_indices:
            raise ValueError("row_live_atoms needs the 2-D index operand")
        if row_live_atoms.shape != (num_rows,) or row_len % OCCURRENCES_PER_ATOM:
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
    num_base = len(operands)  # nope, rope, indices[, live_vec]
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
        operands.append(write_rows)
        rounds = write_slots(padded, write_fault)
        scratch_types["write_dst_vmem"] = pltpu.VMEM((rounds, NUM_LANES), jnp.int32)
        sems = 1 if write_fault == "no_fence" else rounds
        scratch_types["write_dst_sem"] = pltpu.SemaphoreType.DMA((sems,))
        scratch_types["write_sem"] = pltpu.SemaphoreType.DMA((2 * sems,))
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
        linear_indices=linear_indices,
        write_tokens=write_tokens,
        write_fault=write_fault,
        rope_words=rope_words,
        **row_kwargs,
    )

    def body(*refs, write_dst_vmem=None, write_dst_sem=None, write_sem=None, **scr):
        ins, (nope_o, rope_o) = refs[:-2], refs[-2:]
        nope, rope, idx = ins[:3]
        kw = {}
        extra = 3
        if live_vec_offset is not None:
            kw["live_hbm_ref"] = idx.at[pl.ds(live_vec_offset, 2 * NUM_LANES)]
        elif live:
            kw["live_hbm_ref"] = ins[extra]
            extra += 1
        if row_kwargs:
            kw["row_atoms_hbm_ref"] = ins[extra]
        if kv_write is not None:
            kw["write_refs"] = tuple(ins[num_base:])
            kw["write_scratch"] = (write_dst_vmem, write_dst_sem, write_sem)
        return inner(nope, rope, idx, nope_o, rope_o, **kw, **scr)

    outs = core_map_helper.kernel(
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
            f"sc_native_sc_gather_a{atoms_per_batch}_{wait_mode}_"
            f"{'pipe' if serve_pipelined else 'nopipe'}{'_live' if live else ''}"
            f"{'_kvw' if kv_write is not None else ''}"
            f"{'_rowlive' if row_kwargs else ''}"
            f"{'_packedrope' if packed_rope else ''}"
        ),
    )(*operands)
    if kv_write is None:
        return outs
    return (
        *outs,
        jax.freeze(nope_rows_view).reshape(nope_cache.shape),
        jax.freeze(rope_rows_view).reshape(rope_cache.shape),
    )


# ---------------------------------------------------------------------------
# Union read (TPU_DCP_MLA_MIXED_UNION_GATHER): the live-group gather of one
# chunk with each half grid step's distinct cache rows read from HBM once.
# ---------------------------------------------------------------------------
# Rows of the per-subcore union region in Spmem (VMEM_SHARED). The 16 regions
# of a SparseCore core share its 2,097,151 allocatable words, of which 262,143
# are taken before any kernel scratch (Mosaic-SC allocation report, tpu7x):
# 1,835,008 words = 16 x 896 rows of nope (128 words), or 16 x 704 rows of
# nope + rope (160 words; `rope_union`).
UNION_CAP_ROWS = 736
UNION_CAP_ROWS_ROPE = 576
# TPU_DCP_MLA_MIXED_UNION_GATHER=1: the measured default mode
# (t4-mla-mixed probes 12-14).
UNION_GATHER_DEFAULT = ("pair", "rope")
# `pair_union`: a pair's two regions hold one union (2 x these rows), less the
# table + list hand-off area in Spmem.
UNION_CAP_ROWS_PAIR = 704
UNION_CAP_ROWS_ROPE_PAIR = 576
# Rows per step of the union load pipeline (two 16-row indirect streams).
UNION_LOAD_ROWS = 32
# Entries of the per-subcore open-addressing hash table (TileSPMEM, i32).
UNION_HASH_SIZE = 2048
# Rows of one expansion unit: one token's 64-slot atom, the incumbent's unit
# (4 streams of 16 rows; 16 grouped rope rows).
UNION_UNIT_ROWS = QUADS_PER_ATOM * ROPE_PACKING
# Test-only faults of the union gather (positive controls).
UNION_FAULTS = (
    None,
    # hit lanes read a neighbouring union row
    "map_off",
    # a hash hit is trusted without comparing the stored row (aliasing)
    "noverify",
    # units with a lane past the union cap are not re-read from HBM
    "skip_ovf",
    # each token's last live slot is treated as padding
    "kv_lo",
    # the union path runs even when the chunk overflows window 0
    "no_fallback",
    # timing-only removal arms (wrong rows): skip the union load / the expansion
    "rm_load",
    "rm_expand",
)


def _rope_group(rope_inputs, rope_output):
    """The incumbent body's rope register transpose (`_kernel` body) in
    16-lane vectors (a kernel without Mosaic-SC layout passes -- the union
    gather's vector gathers need them off -- loads only (16,) vectors): four
    16-row x 32-word rope streams -> 16 grouped 128-word rows. Same bytes."""
    low_16 = jnp.uint32(0x0000FFFF)
    low_byte_each_half = jnp.uint32(0x00FF00FF)
    for row in range(QUADS_PER_ATOM):
        for half in range(32 // NUM_LANES):
            cols = pl.ds(half * NUM_LANES, NUM_LANES)
            data = []
            for token_slot in range(ROPE_PACKING):
                logical_rank = row * ROPE_PACKING + token_slot
                input_stream = logical_rank // NUM_LANES
                input_lane = logical_rank % NUM_LANES
                data.append(rope_inputs[input_stream][input_lane, cols])
            data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
            data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
            data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
            data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
            for band in range(ROPE_PACKING):
                rope_output[row, pl.ds(band * 32 + half * NUM_LANES, NUM_LANES)] = data[
                    band
                ]


def _expand_units(
    src_nope,
    src_rope,
    nidx_ref,
    nidx_off,
    ridx_ref,
    ridx_off,
    pos_of,
    nope_out_ref,
    rope_out_ref,
    sem_ref,
    num_units,
):
    """Gather `num_units` 64-row units and write each at its own output rows.

    Unit k reads nope rows `nidx_ref[nidx_off(k) + [0, 64)]` of `src_nope`
    and rope rows `ridx_ref[ridx_off(k) + [0, 64)]` of `src_rope` (HBM, or
    the subcore's Spmem union region), and writes them at nope rows
    `pos_of(k) + [0, 64)` and grouped rope rows `/ 4` -- the incumbent body
    (`_kernel`) with the unit's rows and destination as data.
    """

    def nvalues(k, stream):
        return nidx_ref[pl.ds(nidx_off(k) + stream * NUM_LANES, NUM_LANES)]

    def rvalues(k, stream):
        return ridx_ref[pl.ds(ridx_off(k) + stream * NUM_LANES, NUM_LANES)]

    def body(*refs):
        k = pl.program_id(0)
        nope_inputs = refs[:ROPE_PACKING]
        rope_inputs = refs[ROPE_PACKING : 2 * ROPE_PACKING]
        rope_output = refs[-1]
        dst = pos_of(k)
        copies = []
        for stream in range(ROPE_PACKING):
            copy = pltpu.make_async_copy(
                nope_inputs[stream],
                nope_out_ref.at[pl.ds(dst + stream * NUM_LANES, NUM_LANES), :],
                sem_ref.at[stream],
            )
            copy.start()
            copies.append(copy)
        _rope_group(rope_inputs, rope_output)
        for copy in copies:
            copy.wait()

    nope_specs = tuple(
        pl.BlockSpec(
            (pl.Indirect(NUM_LANES), 128),
            lambda k, stream=stream: (nvalues(k, stream), 0),
        )
        for stream in range(ROPE_PACKING)
    )
    rope_specs = tuple(
        pl.BlockSpec(
            (pl.Indirect(NUM_LANES), 32),
            lambda k, stream=stream: (rvalues(k, stream), 0),
        )
        for stream in range(ROPE_PACKING)
    )
    rope_out_spec = pl.BlockSpec(
        (QUADS_PER_ATOM, 128), lambda k: (jnp.right_shift(pos_of(k), 6), 0)
    )
    pltpu.emit_pipeline(
        body,
        grid=(num_units,),
        in_specs=nope_specs + rope_specs,
        out_specs=(rope_out_spec,),
    )(
        *([src_nope] * ROPE_PACKING),
        *([src_rope] * ROPE_PACKING),
        rope_out_ref,
    )


def _union_kernel(
    nope_hbm_ref,
    rope_hbm_ref,
    indices_hbm_ref,
    tab_hbm_ref,
    kv_hbm_ref,
    live_hbm_ref,
    nope_out_hbm_ref,
    rope_out_hbm_ref,
    sem_ref,
    load_sem_ref,
    vec_ref,
    tab_smem,
    kv_smem,
    pos_smem,
    uid_smem,
    misc_smem,
    ovf_smem,
    ibuf,
    cbuf,
    mtok,
    ulist,
    htab,
    nope_sh,
    rope_sh=None,
    hand_sh=None,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_subcores: int,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    chunk_index: int,
    chunk_size: int,
    bz: int,
    num_groups: int,
    num_chunks: int,
    atoms_w: int,
    cap: int,
    rope_union: bool,
    pair: bool,
    fault: str | None,
):
    gpc = chunk_size // bz
    atom = UNION_UNIT_ROWS
    block = bz * atom  # one atom block of a group: bz tokens x 64 slots
    half = bz // 2  # tokens per subcore
    max_units = half * atoms_w
    occ = atoms_per_batch * OCCURRENCES_PER_ATOM
    iota = lax.broadcasted_iota(jnp.int32, (NUM_LANES,), 0)

    # The chunk's group table (A | off), its live-index bound, and its tokens'
    # kv_lens arrive as HBM vectors; SparseCore scalars come from SMEM.
    # (three small DMAs in flight together: each is latency-bound; on semaphores
    # no pipeline holds yet -- a SparseCore has few)
    nl_base = chunk_index // NUM_LANES * NUM_LANES
    nl_size = min(NUM_LANES, num_chunks - nl_base)
    meta = [
        pltpu.make_async_copy(
            tab_hbm_ref.at[pl.ds(chunk_index * gpc, gpc)],
            vec_ref.at[pl.ds(0, gpc)],
            sem_ref.at[0],
        ),
        pltpu.make_async_copy(
            tab_hbm_ref.at[pl.ds(num_groups + chunk_index * gpc, gpc)],
            vec_ref.at[pl.ds(gpc, gpc)],
            sem_ref.at[1],
        ),
        pltpu.make_async_copy(
            live_hbm_ref.at[pl.ds(nl_base, nl_size)],
            vec_ref.at[pl.ds(2 * gpc, nl_size)],
            sem_ref.at[2],
        ),
    ]
    for cp in meta:
        cp.start()
    for cp in meta:
        cp.wait()
    a_vec = vec_ref[pl.ds(0, NUM_LANES)]
    off_vec = vec_ref[pl.ds(gpc, NUM_LANES)]
    num_live = vec_ref[pl.ds(2 * gpc, NUM_LANES)][chunk_index - nl_base]
    for i in range(gpc):
        tab_smem[i] = a_vec[i]
        tab_smem[gpc + i] = off_vec[i]
    live_blocks = off_vec[gpc - 1] + a_vec[gpc - 1]
    use_union = num_live == live_blocks * block
    if fault == "no_fallback":
        use_union = jnp.bool_(True)

    misc_smem[0] = jnp.int32(0)  # units this subcore reads from HBM directly
    misc_smem[2] = jnp.int32(0)  # live units expanded from the union
    core = lax.axis_index(core_axis_name)
    sub = lax.axis_index(subcore_axis_name)
    # Subcore pair (2i, 2i + 1) of core c serves group c * (subcores / 2) + i:
    # tokens [0, bz/2) and [bz/2, bz) of it.
    g = core * (num_subcores // 2) + jnp.right_shift(sub, 1)
    h = jnp.bitwise_and(sub, 1)
    a_live = tab_smem[g]
    goff = tab_smem[gpc + g]

    work = use_union & (a_live > 0)
    whole = a_live == atoms_w
    base_out = goff * block
    mask_h = htab.shape[0] - 1
    streams = UNION_LOAD_ROWS // NUM_LANES
    # With `pair`, the two subcores of a pair share one union (their 16 tokens,
    # ids in [0, 2 cap), rows in the pair's two adjacent Spmem regions); else
    # each subcore has its own (8 tokens, ids in [0, cap)).
    cap_u = 2 * cap if pair else cap
    region = (sub - h) * cap if pair else sub * cap

    def prep_slots():
        tok0 = chunk_index * chunk_size + g * bz
        kv_cp = pltpu.make_async_copy(
            kv_hbm_ref.at[pl.ds(pl.multiple_of(tok0, bz), bz)],
            vec_ref.at[pl.ds(0, bz)],
            load_sem_ref.at[1, 1],
        )
        kv_cp.start()

        # This subcore's slots, in the layout the prep wrote: atom blocks of
        # [bz tokens, 64] (A < atoms_w), or the whole group token-major
        # [bz, atoms_w * 64]. Either way the ibuf holds this subcore's tokens'
        # 64-slot units back to back. All copies in flight together.
        def blk(a):
            return pltpu.make_async_copy(
                indices_hbm_ref.at[
                    pl.ds(
                        pl.multiple_of((goff + a) * block + h * half * atom, atom),
                        half * atom,
                    )
                ],
                ibuf.at[pl.ds(pl.multiple_of(a * half * atom, atom), half * atom)],
                load_sem_ref.at[0, 0],
            )

        span = half * atoms_w * atom
        whole_cp = pltpu.make_async_copy(
            indices_hbm_ref.at[
                pl.ds(pl.multiple_of(goff * block + h * span, atom), span)
            ],
            ibuf.at[pl.ds(0, span)],
            load_sem_ref.at[0, 0],
        )

        @pl.when(jnp.logical_not(whole))
        def _blocks():
            def start(a, c):
                blk(a).start()
                return c

            def wait(a, c):
                blk(a).wait()
                return c

            lax.fori_loop(0, a_live, start, 0)
            lax.fori_loop(0, a_live, wait, 0)

        @pl.when(whole)
        def _whole():
            whole_cp.start()
            whole_cp.wait()

        kv_cp.wait()
        kv_vec = vec_ref[pl.ds(0, NUM_LANES)]
        for i in range(bz):
            kv_smem[i] = kv_vec[i]

    def clear_table():
        def clear(i):
            htab[pl.ds(i * NUM_LANES, NUM_LANES)] = jnp.full(
                (NUM_LANES,), -1, jnp.int32
            )

        def clear_list(i):
            ulist[pl.ds(i * NUM_LANES, NUM_LANES)] = jnp.full(
                (NUM_LANES,), -1, jnp.int32
            )

        plsc.parallel_loop(0, htab.shape[0] // NUM_LANES, unroll=8)(clear)
        plsc.parallel_loop(0, ulist.shape[0] // NUM_LANES, unroll=8)(clear_list)

    def hash_of(rows):
        return jnp.bitwise_and(
            jnp.bitwise_xor(
                jnp.bitwise_xor(rows, jnp.right_shift(rows, 11)),
                jnp.right_shift(rows, 22),
            ),
            mask_h,
        )

    # Tokens in order (rows past the cap are assigned last, so they land in the
    # last tokens' units). Within a token, three passes with no loop-carried
    # dependence inside a pass: (A) look every live slot up in the table, in
    # parallel -- a token's rows are distinct, so its own slots can only hit
    # entries of earlier tokens; (B) a scalar prefix of each vector's misses;
    # (C) assign ids and insert, in parallel (two of the token's vectors meet
    # only on a colliding entry of a different row; whichever insert lands,
    # its id and stored row agree). Then its units in order (keep / HBM-direct
    # / skip).
    def token_body(t, carry):
        base, nlu, novf = carry
        kv_t = kv_smem[h * half + t]
        if fault == "kv_lo":
            kv_t = kv_t - 1
        nvec = (jnp.clip(kv_t, 0, a_live * atom) + NUM_LANES - 1) // NUM_LANES

        def unit_of(a):
            return jnp.where(whole, t * atoms_w + a, a * half + t)

        def rows_of(jv):
            a = jnp.right_shift(jv, 2)
            v16 = jnp.bitwise_and(jv, 3) * NUM_LANES
            live = v16 + iota < kv_t - a * atom
            rows = ibuf[
                pl.ds(pl.multiple_of(unit_of(a) * atom + v16, NUM_LANES), NUM_LANES)
            ]
            return rows, live

        def look(jv):
            rows, live = rows_of(jv)
            uo = plsc.load_gather(htab, [hash_of(rows)], mask=live)
            known = live & (uo >= 0)
            r = plsc.load_gather(ulist, [jnp.where(known, uo, 0)], mask=known)
            hit = known if fault == "noverify" else known & (r == rows)
            miss = live & jnp.logical_not(hit)
            # hit: its id; miss: -1; padding: -2
            mtok[pl.ds(pl.multiple_of(jv * NUM_LANES, NUM_LANES), NUM_LANES)] = (
                jnp.where(hit, uo, jnp.where(live, -1, -2))
            )
            ovf_smem[jv] = plsc.all_reduce_population_count(miss)[0]

        plsc.parallel_loop(0, nvec)(look)

        def prefix(jv, run):
            c = ovf_smem[jv]
            ovf_smem[jv] = run
            return run + c

        total = lax.fori_loop(0, nvec, prefix, jnp.int32(0))

        def insert(jv):
            rows, _ = rows_of(jv)
            m0 = mtok[pl.ds(pl.multiple_of(jv * NUM_LANES, NUM_LANES), NUM_LANES)]
            miss = m0 == -1
            newu = base + ovf_smem[jv] + plsc.cumsum(miss.astype(jnp.int32)) - 1
            ok = miss & (newu < cap_u)
            plsc.store_scatter(ulist, [jnp.where(ok, newu, 0)], rows, mask=ok)
            plsc.store_scatter(htab, [hash_of(rows)], newu, mask=ok)
            m_hit = m0
            if fault == "map_off":
                m_hit = jnp.minimum(jnp.bitwise_xor(m0, 1), jnp.maximum(base - 1, 0))
            mtok[pl.ds(pl.multiple_of(jv * NUM_LANES, NUM_LANES), NUM_LANES)] = (
                jnp.where(m0 >= 0, m_hit, jnp.where(ok, newu, 0))
            )
            ovf_smem[jv] = plsc.all_reduce_population_count(miss & jnp.logical_not(ok))[
                0
            ]

        plsc.parallel_loop(0, nvec)(insert)
        base = jnp.minimum(base + total, cap_u)

        def unit_step(a, c):
            nlu, novf = c
            u = unit_of(a)
            pos = jnp.where(
                whole,
                base_out + h * half * atoms_w * atom + u * atom,
                base_out + a * block + h * half * atom + t * atom,
            )
            nlive = jnp.clip(kv_t - a * atom, 0, atom)
            over = jnp.int32(0)
            for v in range(atom // NUM_LANES):
                over = over + jnp.where(v * NUM_LANES < nlive, ovf_smem[4 * a + v], 0)
            unit_live = nlive > 0
            to_union = unit_live & (over == 0)
            to_hbm = unit_live & (over > 0)

            @pl.when(to_union)
            def _keep():
                for v in range(atom // NUM_LANES):
                    lane = v * NUM_LANES + iota
                    # Pad lanes read a distinct already-loaded row (the attention
                    # masks them); a repeated index is the degenerate bank conflict.
                    fill = jnp.where(lane < base, lane, 0)
                    m = mtok[
                        pl.ds(
                            pl.multiple_of((4 * a + v) * NUM_LANES, NUM_LANES),
                            NUM_LANES,
                        )
                    ]
                    m = jnp.where(lane < nlive, m, fill)
                    cbuf[
                        pl.ds(
                            pl.multiple_of(nlu * atom + v * NUM_LANES, NUM_LANES),
                            NUM_LANES,
                        )
                    ] = m + region
                pos_smem[nlu] = pos
                uid_smem[nlu] = u

            @pl.when(to_hbm)
            def _direct():
                pos_smem[max_units + novf] = pos
                uid_smem[max_units + novf] = u

            return nlu + to_union.astype(jnp.int32), novf + to_hbm.astype(jnp.int32)

        nlu, novf = lax.fori_loop(0, a_live, unit_step, (nlu, novf))
        return base, nlu, novf

    def dedup(base0):
        """Assign union ids to this subcore's live slots from `base0`; leaves
        (ids used, live units, HBM-direct units) in misc_smem[1, 2, 0]."""
        num_u, n_units, n_ovf = lax.fori_loop(
            0, half, token_body, (base0, jnp.int32(0), jnp.int32(0))
        )
        misc_smem[1] = num_u
        misc_smem[2] = n_units
        if fault != "skip_ovf":
            misc_smem[0] = n_ovf

    def round_up(x):
        return (x + UNION_LOAD_ROWS - 1) // UNION_LOAD_ROWS * UNION_LOAD_ROWS

    def load_rows(lo, hi):
        """ulist[lo:hi] nope (and rope) rows HBM -> TileSPMEM (indirect) ->
        Spmem rows region + [lo, hi). `lo` is a multiple of the load step; the
        tail up to the next step is filled with distinct loaded rows."""
        nld = (hi - lo + UNION_LOAD_ROWS - 1) // UNION_LOAD_ROWS

        @pl.when((hi > lo) & (fault != "rm_load"))
        def _load():
            first = ulist[pl.ds(0, NUM_LANES)]
            filler = jnp.where(
                iota < hi, first, jnp.broadcast_to(first[0], (NUM_LANES,))
            )
            for q in range(streams):
                p = pl.multiple_of(
                    lo + (nld - 1) * UNION_LOAD_ROWS + q * NUM_LANES, NUM_LANES
                )
                cur = ulist[pl.ds(p, NUM_LANES)]
                ulist[pl.ds(p, NUM_LANES)] = jnp.where(p + iota >= hi, filler, cur)

            def lvalues(p, stream):
                return ulist[
                    pl.ds(
                        pl.multiple_of(
                            lo + p * UNION_LOAD_ROWS + stream * NUM_LANES, NUM_LANES
                        ),
                        NUM_LANES,
                    )
                ]

            def lbody(*refs):
                p = pl.program_id(0)
                cps = []
                for stream in range(streams):
                    dst = pl.ds(
                        pl.multiple_of(
                            region + lo + p * UNION_LOAD_ROWS + stream * NUM_LANES,
                            NUM_LANES,
                        ),
                        NUM_LANES,
                    )
                    c1 = pltpu.make_async_copy(
                        refs[stream], nope_sh.at[dst], load_sem_ref.at[0, stream]
                    )
                    c1.start()
                    cps.append(c1)
                    if rope_union:
                        assert rope_sh is not None
                        c2 = pltpu.make_async_copy(
                            refs[streams + stream],
                            rope_sh.at[dst],
                            load_sem_ref.at[1, stream],
                        )
                        c2.start()
                        cps.append(c2)
                for c in cps:
                    c.wait()

            nspecs = tuple(
                pl.BlockSpec(
                    (pl.Indirect(NUM_LANES), 128), lambda p, st=st: (lvalues(p, st), 0)
                )
                for st in range(streams)
            )
            rspecs = (
                tuple(
                    pl.BlockSpec(
                        (pl.Indirect(NUM_LANES), 32),
                        lambda p, st=st: (lvalues(p, st), 0),
                    )
                    for st in range(streams)
                )
                if rope_union
                else ()
            )
            pltpu.emit_pipeline(
                lbody, grid=(nld,), in_specs=nspecs + rspecs, out_specs=()
            )(*([nope_hbm_ref] * streams), *([rope_hbm_ref] * len(rspecs)))

    if not pair:

        @pl.when(work)
        def _solo():
            prep_slots()
            clear_table()
            dedup(jnp.int32(0))
            load_rows(jnp.int32(0), misc_smem[1])

    else:
        # Subcore 2i dedups its tokens, hands its table and union list to
        # subcore 2i + 1 through Spmem, and loads its rows while 2i + 1
        # continues the same union (ids from a load-step boundary) and loads
        # the rows it added; a barrier then makes every row visible to both.
        pair_idx = jnp.right_shift(sub, 1)
        ht = htab.shape[0]
        assert hand_sh is not None

        @pl.when(work & (h == 0))
        def _first():
            prep_slots()
            clear_table()
            dedup(jnp.int32(0))
            u0 = misc_smem[1]
            u0r = round_up(u0)
            # pad [u0, u0r) with distinct loaded rows (2i's load fills it too)
            first = ulist[pl.ds(0, NUM_LANES)]
            filler = jnp.where(
                iota < u0, first, jnp.broadcast_to(first[0], (NUM_LANES,))
            )
            for q in range(streams):
                p = pl.multiple_of(
                    jnp.maximum(u0r - UNION_LOAD_ROWS, 0) + q * NUM_LANES, NUM_LANES
                )
                cur = ulist[pl.ds(p, NUM_LANES)]
                ulist[pl.ds(p, NUM_LANES)] = jnp.where(p + iota >= u0, filler, cur)
            vec_ref[pl.ds(0, NUM_LANES)] = jnp.broadcast_to(u0r, (NUM_LANES,))
            pltpu.sync_copy(htab, hand_sh.at[pair_idx, pl.ds(0, ht)])
            pltpu.sync_copy(
                ulist.at[pl.ds(0, cap_u)], hand_sh.at[pair_idx, pl.ds(ht, cap_u)]
            )
            pltpu.sync_copy(
                vec_ref.at[pl.ds(0, NUM_LANES)],
                hand_sh.at[pair_idx, pl.ds(ht + cap_u, NUM_LANES)],
            )

        plsc.subcore_barrier()

        @pl.when(work & (h == 1))
        def _second():
            prep_slots()
            pltpu.sync_copy(hand_sh.at[pair_idx, pl.ds(0, ht)], htab)
            pltpu.sync_copy(
                hand_sh.at[pair_idx, pl.ds(ht, cap_u)], ulist.at[pl.ds(0, cap_u)]
            )
            pltpu.sync_copy(
                hand_sh.at[pair_idx, pl.ds(ht + cap_u, NUM_LANES)],
                vec_ref.at[pl.ds(0, NUM_LANES)],
            )
            u0r = vec_ref[pl.ds(0, NUM_LANES)][0]
            dedup(u0r)
            load_rows(u0r, misc_smem[1])

        @pl.when(work & (h == 0))
        def _first_load():
            load_rows(jnp.int32(0), misc_smem[1])

        plsc.subcore_barrier()

    n_units = misc_smem[2]
    unit_off = lambda k: k * atom  # noqa: E731
    own_rows = lambda k: uid_smem[k] * atom  # noqa: E731

    # Expansion: every live unit's nope rows from the union in Spmem (rope from
    # the union too, or from HBM by its own rows).
    @pl.when(work & (n_units > 0) & (fault != "rm_expand"))
    def _expand():
        _expand_units(
            nope_sh,
            rope_sh if rope_union else rope_hbm_ref,
            cbuf,
            unit_off,
            cbuf if rope_union else ibuf,
            unit_off if rope_union else own_rows,
            lambda k: pos_smem[k],
            nope_out_hbm_ref,
            rope_out_hbm_ref,
            sem_ref,
            n_units,
        )

    # Units read from HBM directly, through one pipeline: this subcore's
    # units past the union cap, or -- when the chunk reaches past window 0 --
    # the incumbent gather's own work, one logical batch (atoms_per_batch
    # units, the incumbent's owner interleave) per round.
    live_b = jnp.right_shift(num_live, occ.bit_length() - 1)
    owner_b = jnp.right_shift(live_b + num_owners - 1, num_owners.bit_length() - 1)
    owner = core * num_subcores + sub
    n_rounds = jnp.where(use_union, jnp.minimum(misc_smem[0], 1), owner_b)

    def hbm_round(r, c):
        batch = owner + r * num_owners
        fb = jnp.logical_not(use_union) & (batch < live_b)

        @pl.when(fb)
        def _fetch():
            start = pl.multiple_of(batch * occ, occ)
            pltpu.sync_copy(
                indices_hbm_ref.at[pl.ds(start, occ)], ibuf.at[pl.ds(0, occ)]
            )
            for j in range(atoms_per_batch):
                pos_smem[max_units + j] = start + j * atom
                uid_smem[max_units + j] = jnp.int32(j)

        n = jnp.where(use_union, misc_smem[0], jnp.where(fb, atoms_per_batch, 0))

        @pl.when(n > 0)
        def _run():
            rows_of = lambda k: uid_smem[max_units + k] * atom  # noqa: E731
            _expand_units(
                nope_hbm_ref,
                rope_hbm_ref,
                ibuf,
                rows_of,
                ibuf,
                rows_of,
                lambda k: pos_smem[max_units + k],
                nope_out_hbm_ref,
                rope_out_hbm_ref,
                sem_ref,
                n,
            )

        return c

    lax.fori_loop(0, n_rounds, hbm_round, 0)


def union_gather_supported(
    chunk_size: int, kernel_batch_size: int, topk_cap: int
) -> bool:
    """Whether `dsa_gather_native_sc_union` serves a live-group chunk: one
    subcore pair per grid step (chunk / bz == cores * subcores / 2), whole
    64-slot units per subcore, power-of-two atoms per window."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    num_cores = sc_info.num_cores if sc_info is not None else NUM_CORES
    num_subcores = sc_info.num_subcores if sc_info is not None else NUM_SUBCORES
    atoms_w = topk_cap // UNION_UNIT_ROWS
    bz = kernel_batch_size
    return (
        chunk_size % bz == 0
        and chunk_size // bz == num_cores * num_subcores // 2
        and bz % 2 == 0
        and (bz // 2) & (bz // 2 - 1) == 0
        and bz == NUM_LANES
        and topk_cap % UNION_UNIT_ROWS == 0
        and atoms_w & (atoms_w - 1) == 0
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "chunk_index",
        "chunk_size",
        "kernel_batch_size",
        "topk_cap",
        "atoms_per_batch",
        "cap_rows",
        "rope_union",
        "pair_union",
        "fault",
    ),
)
def dsa_gather_native_sc_union(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    indices: jax.Array,
    group_tab: jax.Array,
    kv_lens: jax.Array,
    group_num_live: jax.Array,
    *,
    out_size: int,
    chunk_index: int,
    chunk_size: int,
    kernel_batch_size: int,
    topk_cap: int,
    atoms_per_batch: int = 16,
    cap_rows: int | None = None,
    rope_union: bool = False,
    pair_union: bool = False,
    fault: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """The live-group chunk gather of `dsa_gather_native_sc` (linear indices,
    `num_live_indices`) with each half grid step's distinct rows read once.

    The two subcores of a pair split a grid step's ``bz`` tokens. Each dedups
    its tokens' live slots (slot < kv_len) with an open-addressing table in
    TileSPMEM, gathers the distinct rows' nope (``U`` rows, up to
    ``cap_rows``) from HBM into its own Spmem region, then writes every live
    64-slot unit of its tokens from Spmem at exactly the rows the incumbent
    writes; the unit's rope rows come from HBM by their own rows, or with
    ``rope_union`` from the union in Spmem too (fewer rows fit). A unit with a
    row past the cap is read from HBM directly; a chunk whose gather bound
    reaches past window 0 (a token over ``topk_cap``) runs the incumbent body.

    Every live slot gets the bytes the incumbent writes there. Padding slots
    (slot >= kv_len) inside a live unit get some already-loaded row instead of
    a hashed one, and wholly padded units are not written: the attention masks
    and zeroes both, as it does for the incumbent's stale hashed rows.

    ``indices``: the chunk's prep rows (any shape, ``out_size`` values in the
    live-group layout). ``group_tab`` i32[2 G] and ``group_num_live``
    i32[num_chunks]: `live_group_table` for the whole bucket; ``kv_lens``
    i32[T] for the whole bucket. ``fault``: test-only, see `UNION_FAULTS`.
    """
    if fault not in UNION_FAULTS:
        raise ValueError(f"unknown {fault=}")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError("native-SC ROPE must be quarter-major uint32[...,128]")
    if indices.size != out_size:
        raise ValueError(f"indices has {indices.size} values, expected {out_size}")
    bz = kernel_batch_size
    if not union_gather_supported(chunk_size, bz, topk_cap):
        raise ValueError(
            f"union gather unsupported: chunk {chunk_size}, bz {bz}, cap {topk_cap}"
        )
    if cap_rows is None:
        cap_rows = (
            (UNION_CAP_ROWS_ROPE_PAIR if rope_union else UNION_CAP_ROWS_PAIR)
            if pair_union
            else (UNION_CAP_ROWS_ROPE if rope_union else UNION_CAP_ROWS)
        )
    if cap_rows % UNION_LOAD_ROWS:
        raise ValueError(f"cap_rows {cap_rows} must be a multiple of {UNION_LOAD_ROWS}")
    num_cores, num_subcores, atoms_per_batch, occurrences_per_batch, num_batches = (
        live_vec_geometry(out_size, atoms_per_batch)
    )
    if occurrences_per_batch & (occurrences_per_batch - 1):
        raise ValueError(
            f"occurrences per batch {occurrences_per_batch} not a power of two"
        )
    num_owners = num_cores * num_subcores
    num_groups = group_tab.shape[0] // 2
    num_chunks = group_num_live.shape[0]
    atoms_w = topk_cap // UNION_UNIT_ROWS
    half = bz // 2
    max_units = half * atoms_w
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    scratch_types = {
        "sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING,)),
        "load_sem_ref": pltpu.SemaphoreType.DMA((2, UNION_LOAD_ROWS // NUM_LANES)),
        "vec_ref": pltpu.VMEM((3 * NUM_LANES + NUM_LANES,), jnp.int32),
        "tab_smem": pltpu.SMEM((2 * (chunk_size // bz),), jnp.int32),
        "kv_smem": pltpu.SMEM((bz,), jnp.int32),
        "pos_smem": pltpu.SMEM((2 * max_units,), jnp.int32),
        "uid_smem": pltpu.SMEM((2 * max_units,), jnp.int32),
        "misc_smem": pltpu.SMEM((8,), jnp.int32),
        "ovf_smem": pltpu.SMEM((atoms_w * UNION_UNIT_ROWS // NUM_LANES,), jnp.int32),
        "mtok": pltpu.VMEM((atoms_w * UNION_UNIT_ROWS,), jnp.int32),
        "ibuf": pltpu.VMEM(
            (max(max_units * UNION_UNIT_ROWS, occurrences_per_batch),), jnp.int32
        ),
        "cbuf": pltpu.VMEM((max_units * UNION_UNIT_ROWS,), jnp.int32),
        "ulist": pltpu.VMEM(
            ((2 if pair_union else 1) * cap_rows + UNION_LOAD_ROWS,), jnp.int32
        ),
        "htab": pltpu.VMEM((UNION_HASH_SIZE,), jnp.int32),
        "nope_sh": pltpu.VMEM_SHARED((num_subcores * cap_rows, 128), jnp.uint32),
    }
    if pair_union:
        scratch_types["hand_sh"] = pltpu.VMEM_SHARED(
            (num_subcores // 2, UNION_HASH_SIZE + 2 * cap_rows + NUM_LANES), jnp.int32
        )
    if rope_union:
        scratch_types["rope_sh"] = pltpu.VMEM_SHARED(
            (num_subcores * cap_rows, ROPE_WORDS), jnp.uint32
        )
    inner = functools.partial(
        _union_kernel,
        core_axis_name=mesh.core_axis_name,
        subcore_axis_name=mesh.subcore_axis_name,
        num_subcores=num_subcores,
        num_owners=num_owners,
        atoms_per_batch=atoms_per_batch,
        batches_per_owner=num_batches // num_owners,
        chunk_index=chunk_index,
        chunk_size=chunk_size,
        bz=bz,
        num_groups=num_groups,
        num_chunks=num_chunks,
        atoms_w=atoms_w,
        cap=cap_rows,
        rope_union=rope_union,
        pair=pair_union,
        fault=fault,
    )

    def body(nope, rope, idx, tab, kv, live, nope_o, rope_o, **scr):
        return inner(nope, rope, idx, tab, kv, live, nope_o, rope_o, **scr)

    return core_map_helper.kernel(
        body,
        out_type=(
            jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
            jax.ShapeDtypeStruct((out_size // ROPE_PACKING, 128), jnp.uint32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=False,
            disable_bounds_checks=True,
        ),
        scratch_types=scratch_types,
        mesh=mesh,
        name=(
            f"sc_native_sc_gather_union{'_rope' if rope_union else ''}"
            f"{'_pair' if pair_union else ''}_c{chunk_index}"
        ),
    )(
        nope_cache.reshape(-1, 128),
        rope_cache.reshape(-1, ROPE_WORDS),
        indices.reshape(-1),
        group_tab.astype(jnp.int32),
        kv_lens.reshape(-1).astype(jnp.int32),
        group_num_live.astype(jnp.int32),
    )
