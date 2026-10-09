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

"""Where the fused kernels' operands live, and the streamed weights' DMAs."""

from __future__ import annotations

from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# QkvWeights or GdnWeights, of arrays, ShapeDtypeStructs or refs.
Weights = Any


# Memory spaces.


def _spec(space: pltpu.MemorySpace) -> pl.BlockSpec:
    return pl.BlockSpec(memory_space=space)


def _weight_spaces(
    weights: Weights, streamed: tuple[str, ...]
) -> dict[str, pltpu.MemorySpace]:
    return {
        name: pltpu.HBM if name in streamed else pltpu.VMEM
        for name, a in weights._asdict().items()
        if a is not None
    }


def in_specs(
    tiled_in: tuple[jax.Array | None, ...],
    weights: Weights,
    num_tiles: int,
    streamed: tuple[str, ...],
) -> tuple[pl.BlockSpec | None, ...]:
    """pallas_call in_specs of (*tiled_in, weights).

    NOTE: With a single tile, the tiled inputs' blocks are the whole arrays, so
    they stay in VMEM and the pipeline aliases them instead of copying;
    otherwise they are HBM and the pipeline streams them.
    """
    tiled = _spec(pltpu.VMEM if num_tiles == 1 else pltpu.HBM)
    spaces = _weight_spaces(weights, streamed)
    return (
        *jax.tree.map(lambda _: tiled, tiled_in),
        weights._replace(**{n: _spec(s) for n, s in spaces.items()}),
    )


def out_specs(
    out_shapes: tuple[jax.ShapeDtypeStruct | None, ...],
) -> tuple[pl.BlockSpec | None, ...]:
    """pallas_call out_specs: always HBM.

    NOTE: VMEM outputs would make XLA evict them after the kernel, one serial
    copy each, which takes longer than the pipeline's parallel writeback.
    """
    hbm = _spec(pltpu.HBM)
    return jax.tree.map(lambda _: hbm, out_shapes)


def pin_weights(weights: Weights, streamed: tuple[str, ...]) -> Weights:
    """Constrains the `streamed` weight fields to HBM and the rest to VMEM.

    NOTE: The BlockSpec only sets the kernel's view; XLA picks the operand's
    memory space from this constraint. Without it, W lands in HBM and the kernel
    copies it synchronously. XLA copies VMEM weights in before the kernel,
    slowing whatever runs then, so the large ones are streamed by the kernel.
    """
    spaces = _weight_spaces(weights, streamed)
    return weights._replace(
        **{
            name: pltpu.with_memory_space_constraint(getattr(weights, name), space)
            for name, space in spaces.items()
        }
    )


# Streamed weights.

# Column blocks per chunk of a streamed weight. One block per chunk is slower
# (each wait is a scheduling barrier); one chunk for all of it overlaps
# nothing.
STREAM_BLOCKS_PER_CHUNK = 2


def stream_chunks(
    block_cols: list[int] | range, block: int
) -> tuple[tuple[int, int], ...]:
    """(start, size) column chunks of a streamed weight, in read order.

    block_cols are the start columns of the `block`-wide blocks, in read order.
    A chunk joins up to STREAM_BLOCKS_PER_CHUNK blocks adjacent both in read
    order and in columns, so it starts at the column first read from it.
    """
    chunks = []
    for col in block_cols:
        if chunks:
            start, size = chunks[-1]
            if start + size == col and size < STREAM_BLOCKS_PER_CHUNK * block:
                chunks[-1] = (start, size + block)
                continue
        chunks.append((col, block))
    return tuple(chunks)


def stream_scratch(
    w: jax.Array | jax.ShapeDtypeStruct, chunks: tuple[tuple[int, int], ...]
) -> list[Any]:
    """Scratch shapes of a streamed weight: its VMEM copy, a DMA sem per chunk."""
    return [pltpu.VMEM(w.shape, w.dtype), pltpu.SemaphoreType.DMA((len(chunks),))]


def stream_vmem_bytes(weights: Weights, streamed: tuple[str, ...]) -> int:
    """VMEM bytes of the streamed weights' copies, on top of the tiling's."""
    return sum(
        getattr(weights, n).size * getattr(weights, n).dtype.itemsize for n in streamed
    )


def start_streams(
    streams: list[tuple[jax.Ref, jax.Ref, jax.Ref, tuple[tuple[int, int], ...]]],
    num_tiles: int,
) -> list[dict[int, Any]]:
    """Starts copying the streamed weights HBM -> VMEM, chunk by chunk.

    With one tile, the tile waits for each chunk just before its first read, so
    the copies overlap Step 0 and the earlier chunks' matmuls. With more tiles,
    this waits for all copies: a wait in the pipeline body is a scheduling
    barrier in every tile.

    `streams` holds (hbm_ref, vmem_ref, sems, chunks) per weight, in first-read
    order. Returns, per stream, {chunk start column: copy} of the copies not yet
    waited on.
    """
    pending = []
    for hbm_ref, vmem_ref, sems, chunks in streams:
        copies = {}
        for i, (start, size) in enumerate(chunks):
            cols = pl.ds(start, size)
            copies[start] = pltpu.make_async_copy(
                hbm_ref.at[:, cols], vmem_ref.at[:, cols], sems.at[i]
            )
            copies[start].start()
        pending.append(copies)
    if num_tiles > 1:
        for copies in pending:
            for copy in copies.values():
                copy.wait()
            copies.clear()
    return pending


def wait_chunk(pending: dict[int, Any], col: int) -> None:
    """Waits for the chunk starting at column `col` unless already waited.

    Call before each read of a block at `col`, in read order: a chunk's first
    read is at its start, and earlier chunks are already waited.
    The bookkeeping is at trace time, so later calls emit nothing.
    """
    copy = pending.pop(col, None)
    if copy is not None:
        copy.wait()
