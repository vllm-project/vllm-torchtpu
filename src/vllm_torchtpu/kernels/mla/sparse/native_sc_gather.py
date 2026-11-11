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

An optional write phase (``kv_write``) first writes this step's new tokens into
both caches in place, then fences, so the gather reads them. The decode step's
KV-cache write then needs no kernel launch of its own.
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
ROPE_WORDS = 32  # max words per rope token (128B padded view)
# Row blocks of the write phase's operand, u32[WRITE_PARTS * tokens, 128].
WRITE_NOPE_PART, WRITE_ROPE_PART, WRITE_DST_PART, WRITE_PARTS = 0, 1, 2, 3


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
    linear_indices: bool = False,
    write_rows_ref: Any = None,
    write_scratch: tuple[Any, ...] | None = None,
    write_tokens: int = 0,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))
    if write_rows_ref is not None:
        _write_phase(
            write_rows_ref,
            write_scratch,
            nope_hbm_ref,
            rope_hbm_ref,
            subcore=lax.axis_index(subcore_axis_name),
            num_tokens=write_tokens,
            rope_words=rope_words,
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

    def serve_batch(indices_ref):
        owner_batch = pl.program_id(0)
        logical_batch = owner + owner_batch * num_owners
        occurrence_base = logical_batch * atoms_per_batch * OCCURRENCES_PER_ATOM

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


def _write_phase(
    rows_ref: Any,
    write_scratch: tuple[Any, ...],
    nope_cache_ref: Any,
    rope_cache_ref: Any,
    *,
    subcore: jax.Array,
    num_tokens: int,
    rope_words: int,
) -> None:
    """Writes this step's tokens into both caches, then fences.

    Every SparseCore core writes every token: its `NUM_SUBCORES` subcores split
    the tokens (subcore s takes tokens s, s + 16, ...), wait for their own DMAs
    and meet at `subcore_barrier`. So every row that a core gathers was written
    by that core before its first gather DMA. The other core writes the same
    bytes to the same rows, so the race between the two cores cannot change a
    byte, and no cross-core fence is needed.

    `rows_ref` is `pack_write_rows`' u32[3 * num_tokens, 128] operand. A
    destination row is read from lane 0 of a VMEM vector, because a SparseCore
    scalar load from HBM is not supported.
    """
    dst_vmem, dst_sem, write_sem = write_scratch
    num_rows = nope_cache_ref.shape[0]
    copies = []
    for k in range(num_tokens // NUM_SUBCORES):
        token = subcore + k * NUM_SUBCORES
        cp = pltpu.make_async_copy(
            rows_ref.bitcast(jnp.int32).at[
                pl.ds(WRITE_DST_PART * num_tokens + token, 1), pl.ds(0, NUM_LANES)
            ],
            dst_vmem.at[pl.ds(k, 1), :],
            dst_sem.at[k],
        )
        cp.start()
        cp.wait()
        dst = dst_vmem[k, :][0] - 1  # stored as dst + 1, so 0 means skip
        write = jnp.logical_and(dst >= 0, dst < num_rows)
        row = jnp.clip(dst, 0, num_rows - 1)
        nope_cp = pltpu.make_async_copy(
            rows_ref.at[pl.ds(WRITE_NOPE_PART * num_tokens + token, 1)],
            nope_cache_ref.at[pl.ds(row, 1)],
            write_sem.at[2 * k],
        )
        rope_cp = pltpu.make_async_copy(
            rows_ref.at[
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
    """Packs a decode step's KV write into the gather's one write operand.

    Args:
      dst_rows: i32[T] flattened cache row per token (`page * page_size +
        slot`); negative skips the token (padding, or a row another DCP rank
        owns).
      nope_rows: u32[T, 128] nope tokens packed for the SparseCore cache
        (`kv_cache_utils.pack_tokens`).
      rope_rows: u32[T, 128] rope tokens packed the same way, each token in
        its first `rope_words` words and zeros after.

    Returns:
      u32[3 * Tp, 128], where Tp is T rounded up to a multiple of
      `NUM_SUBCORES`: rows [0, Tp) the nope tokens, [Tp, 2 * Tp) the rope
      tokens, and [2 * Tp, 3 * Tp) `dst + 1` broadcast over the row (0 =
      skip). Padding tokens are skipped. One 128-lane operand needs a single
      TensorCore producer and no relayout to the SparseCore's linear layout.
    """
    t = dst_rows.shape[0]
    if not 0 < t <= 2 * NUM_SUBCORES:
        raise ValueError(f"kv_write takes 1 to {2 * NUM_SUBCORES} tokens, got {t}")
    for name, rows in (("nope_rows", nope_rows), ("rope_rows", rope_rows)):
        if rows.shape != (t, 128) or rows.dtype != jnp.uint32:
            raise ValueError(f"{name} must be uint32[{t}, 128], got {rows.shape}")
    tp = pl.cdiv(t, NUM_SUBCORES) * NUM_SUBCORES
    enc = jnp.where(dst_rows >= 0, dst_rows + 1, 0).astype(jnp.uint32)
    parts = [nope_rows, rope_rows, jnp.broadcast_to(enc[:, None], (t, 128))]
    return jnp.concatenate([jnp.pad(x, ((0, tp - t), (0, 0))) for x in parts], 0)


def live_vec_geometry(out_size: int, atoms_per_batch: int) -> tuple[int, ...]:
    """Static gather geometry exactly as `dsa_gather_native_sc` resolves it.

    Returns ``(num_cores, num_subcores, atoms_per_batch, occurrences_per_batch,
    num_batches)``: `atoms_per_batch` after the clamp to what `out_size` can
    feed every owner, and the logical-batch count. A caller that computes the
    live bound itself (``live_vec_offset``) uses these, so its bound cannot
    drift from the one this module derives from ``num_live_indices``.
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
    linear_indices: bool = False,
    live_vec_offset: int | None = None,
    kv_write: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
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

    ``linear_indices`` hands the indices to the kernel as one flat i32[out_size]
    operand instead of [batches, atoms, 4, 16]. XLA lays a 1-D i32 array out
    linearly, which is the SparseCore operand layout, and a 128-lane 2-D
    [n, 128] producer output reshapes to it as a bitcast; the 4-D reshape
    otherwise costs two relayout copies (T(4,128), then linear). Same rows.

    ``live_vec_offset`` (static, with ``linear_indices``; exclusive with
    ``num_live_indices``): the live bound is read from the flat ``indices``
    operand itself, at ``[live_vec_offset, live_vec_offset + 2 * NUM_LANES)``,
    past the ``out_size`` gathered indices. Lanes [0, 16) hold the live
    logical-batch count and lanes [16, 32) the per-owner trip count, as
    `live_vec_geometry` defines them. `prepare_capped_window_fused` writes
    this layout, so its single output carries both the indices and the bound.

    ``kv_write`` (optional) is `pack_write_rows`' u32[3 * Tp, 128] operand:
    this step's new tokens and their destination rows. They are written into
    both caches in place before anything is gathered, with the same bytes as
    `scatter.scatter` writes for the same rows. The call then also returns the
    updated ``(nope_cache, rope_cache)`` after the gathered rows.
    """
    if wait_mode not in ("overlap_rope", "before_rope"):
        raise ValueError(f"unknown {wait_mode=}")
    if nope_cache.dtype != jnp.uint32 or nope_cache.shape[-1] != 128:
        raise ValueError("native-SC NOPE must be explicit uint32[...,128]")
    if rope_cache.dtype != jnp.uint32 or rope_cache.shape[-1] != 128:
        raise ValueError("native-SC ROPE must be banded uint32[...,128]")
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
    if kv_write is not None:
        # The caches become refs that the kernel writes in place (as
        # `scatter_utils.scatter_rows` does); they are frozen after the call.
        nope_rows_view = jax.new_ref(nope_rows_view)
        rope_rows_view = jax.new_ref(rope_rows_view)

    if num_live_indices is not None and live_vec_offset is not None:
        raise ValueError("pass one of num_live_indices, live_vec_offset")
    num_cores, num_subcores, atoms_per_batch, occurrences_per_batch, num_batches = (
        live_vec_geometry(out_size, atoms_per_batch)
    )
    num_owners = num_cores * num_subcores
    owner_batch_size = num_owners * occurrences_per_batch
    batches_per_owner = num_batches // num_owners
    live = num_live_indices is not None or live_vec_offset is not None
    scratch_types = {"sem_ref": pltpu.SemaphoreType.DMA((ROPE_PACKING,))}
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
    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    num_read_operands = len(operands)  # nope, rope, indices[, live_vec]
    write_tokens = 0
    if kv_write is not None:
        if num_subcores != NUM_SUBCORES:
            raise ValueError(f"kv_write assumes {NUM_SUBCORES} subcores per core")
        write_tokens = kv_write.shape[0] // WRITE_PARTS
        if (
            kv_write.shape != (WRITE_PARTS * write_tokens, 128)
            or kv_write.dtype != jnp.uint32
            or write_tokens not in (NUM_SUBCORES, 2 * NUM_SUBCORES)
        ):
            raise ValueError(
                f"kv_write must be uint32[3 * Tp, 128] with Tp in "
                f"({NUM_SUBCORES}, {2 * NUM_SUBCORES}), got "
                f"{kv_write.shape} {kv_write.dtype}"
            )
        operands.append(kv_write)
        rounds = write_tokens // NUM_SUBCORES
        scratch_types["write_dst_vmem"] = pltpu.VMEM((rounds, NUM_LANES), jnp.int32)
        scratch_types["write_dst_sem"] = pltpu.SemaphoreType.DMA((rounds,))
        scratch_types["write_sem"] = pltpu.SemaphoreType.DMA((2 * rounds,))
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
        rope_words=rope_words,
        linear_indices=linear_indices,
        write_tokens=write_tokens,
    )

    def body(*refs, write_dst_vmem=None, write_dst_sem=None, write_sem=None, **scr):
        ins, (nope_o, rope_o) = refs[:-2], refs[-2:]
        nope, rope, idx = ins[:3]
        kwargs = {}
        if live_vec_offset is not None:
            kwargs["live_hbm_ref"] = idx.at[pl.ds(live_vec_offset, 2 * NUM_LANES)]
        elif live:
            kwargs["live_hbm_ref"] = ins[3]
        if kv_write is not None:
            kwargs["write_rows_ref"] = ins[num_read_operands]
            kwargs["write_scratch"] = (write_dst_vmem, write_dst_sem, write_sem)
        return inner(nope, rope, idx, nope_o, rope_o, **kwargs, **scr)

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
            f"sc_native_sc_gather_a{atoms_per_batch}_rw{rope_words}_{wait_mode}_"
            f"{'pipe' if serve_pipelined else 'nopipe'}{'_live' if live else ''}"
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
