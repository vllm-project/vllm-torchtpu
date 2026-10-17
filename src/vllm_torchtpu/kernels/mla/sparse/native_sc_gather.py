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
    is_dcp_mode: bool = False,
    row_atoms_hbm_ref: Any = None,
    row_atoms: int = 0,
    rows_per_batch: int = 1,
    linear_indices: bool = False,
    write_refs: tuple[Any, ...] | None = None,
    write_scratch: tuple[Any, ...] | None = None,
    write_tokens: int = 0,
    write_fault: str | None = None,
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
                        rope_inputs[input_stream][pl.ds(input_lane, 1), pl.ds(0, 32)]
                    )

                low_16 = jnp.uint32(0x0000FFFF)
                low_byte_each_half = jnp.uint32(0x00FF00FF)
                data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
                data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
                data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
                data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
                for band in range(ROPE_PACKING):
                    rope_output[pl.ds(row, 1), pl.ds(band * 32, 32)] = data[band]

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
                (pl.Indirect(NUM_LANES), 32),
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

        pred = None
        if is_dcp_mode:
            # Skip a batch whose first index is the -1 pad.
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
    `rope_cache.reshape(-1, 32)`), so unlike the TensorCore writer there is no
    read-modify-write of the shared 512 B row.

    Everything arrives in one operand, `write_rows` u32[3 * tokens, 128] (see
    `pack_write_rows`): row `t` the packed nope token, row `tokens + t` the
    packed rope token in words [0, 32), row `2 * tokens + t` `dst + 1`
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
                pl.ds(WRITE_ROPE_PART * num_tokens + token, 1), pl.ds(0, ROPE_WORDS)
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
    for the SparseCore cache (`kv_cache_utils.pack_tokens`). Tokens are padded
    to `Tp`, a whole number of subcore rounds (16); padding rows are skipped.
    """
    t = dst_rows.shape[0]
    if not 0 < t <= 2 * NUM_SUBCORES:
        raise ValueError(f"kv_write takes 1..{2 * NUM_SUBCORES} tokens")
    if nope_rows.shape != (t, 128) or nope_rows.dtype != jnp.uint32:
        raise ValueError(f"nope_rows must be uint32[{t}, 128]")
    if rope_rows.shape not in ((t, ROPE_WORDS), (t, 128)) or (
        rope_rows.dtype != jnp.uint32
    ):
        raise ValueError(f"rope_rows must be uint32[{t}, 32 or 128]")
    if rope_rows.shape[1] != 128:
        rope_rows = jnp.pad(rope_rows, ((0, 0), (0, 128 - ROPE_WORDS)))
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
        "is_dcp_mode",
        "linear_indices",
        "live_vec_offset",
        "write_fault",
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

    ``row_live_atoms`` (traced i32[query], optional) gathers only the first
    ``row_live_atoms[i]`` 64-slot atoms of index row ``i``; the other atoms'
    output rows are left unwritten. Composes with ``num_live_indices``.

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
    negative = skip), u32[T, 128] and u32[T, 32 or 128] tokens already packed
    for the SparseCore cache (`kv_cache_utils.pack_tokens`; a 128-wide rope
    operand carries the token in words [0, 32) and avoids a relayout), or
    those three already in the kernel's one-operand form, `pack_write_rows`'s
    u32[3 * Tp, 128] (what the serving path hands over: one TensorCore
    producer). They are written into
    both caches in place before anything is gathered, and the call then also
    returns the updated ``(nope_cache, rope_cache)``. Bytes identical to
    `scatter.scatter` / `tc_scatter.scatter_caches` with the same rows.

    Args:
      nope_cache: explicit native-SC uint32[..., 128] cache.
      rope_cache: quarter-major uint32[..., 128] cache.
      indices: int32[query, topk] token indices.
      out_size: total number of gathered tokens (indices.size).
      atoms_per_batch: number of atoms per batch.
      wait_mode: "overlap_rope" or "before_rope".
      serve_pipelined: whether to pipeline the gather serving loop.
      is_dcp_mode: whether to enable DCP mode (skips gather for padding atoms).
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
    # Arrays, or refs the kernel writes in place under `kv_write`.
    nope_rows_view: Any = nope_cache.reshape(-1, 128)
    rope_rows_view: Any = rope_cache.reshape(-1, 32)
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
        # Rows of the first `out_size` indices; a `live_vec_offset` operand
        # carries its live vector past them, so the width is not its shape.
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
            # Deterministic device hang on v7x (both atom sizes, either live
            # form): a two-round write phase together with the per-row live
            # gather never completes. One round (<= 16 tokens) runs correctly.
            raise ValueError(
                f"row_live_atoms with a {padded}-token kv_write (more than "
                f"{NUM_SUBCORES}) hangs the kernel; drop one of them"
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
        is_dcp_mode=is_dcp_mode,
        linear_indices=linear_indices,
        write_tokens=write_tokens,
        write_fault=write_fault,
        **row_kwargs,
    )

    def body(*refs, write_dst_vmem=None, write_dst_sem=None, write_sem=None, **scr):
        ins, (nope_o, rope_o) = refs[:-2], refs[-2:]
        nope, rope, idx = ins[:3]
        extras = list(ins[3:num_base])  # [live_vec][, row atoms]
        kw = {}
        if live_vec_offset is not None:
            # The fused prep's live vector rides in the indices operand itself.
            kw["live_hbm_ref"] = idx.at[pl.ds(live_vec_offset, 2 * NUM_LANES)]
        elif live:
            kw["live_hbm_ref"] = extras.pop(0)
        if row_kwargs:
            kw["row_atoms_hbm_ref"] = extras.pop(0)
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
            f"{'_rowlive' if row_kwargs else ''}"
            f"{'_kvw' if kv_write is not None else ''}"
        ),
    )(*operands)
    if kv_write is None:
        return outs
    return (
        *outs,
        jax.freeze(nope_rows_view).reshape(nope_cache.shape),
        jax.freeze(rope_rows_view).reshape(rope_cache.shape),
    )
