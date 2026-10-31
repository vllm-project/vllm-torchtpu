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
"""LM-head logits ``x @ w.T`` for a few rows, streaming the head weight.

With a handful of rows the head is a pass over its whole ``[vocab, hidden]``
weight. XLA's dot reads it in narrow hidden-dim windows (8-16 KiB pieces);
here each grid step moves one whole contiguous vocab block, several DMAs in
flight, and contracts it over the full hidden dim in one MXU dot with f32
accumulation, as the dot does.

The kernel reads the head as ``pack_head`` stores it, ``[blocks, hidden,
rows]``: each vocab block transposed, so the dot's weight operand is already
``[hidden, rows]`` and loads into the MXU without a transposing push.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Headroom on top of the buffers for Mosaic's internal scratch.
LOGITS_VMEM_SLACK_BYTES = 8 * 1024 * 1024


class LogitsTiling(NamedTuple):
    """How the head weight is streamed."""

    # Most vocab rows per grid step.
    block_rows: int
    # DMAs a block's weight slab is split into.
    block_dmas: int
    # Weight slabs resident in VMEM: the one being contracted plus prefetches.
    buffers: int


# A block's rows are the lane dim of its transposed slab.
_ROW_GRANULE = 128


def block_rows(vocab: int, tiling: LogitsTiling) -> int:
    """The largest block <= tiling.block_rows that tiles ``vocab``, or 0."""
    for rows in range(
        tiling.block_rows // _ROW_GRANULE * _ROW_GRANULE, 0, -_ROW_GRANULE
    ):
        if vocab % rows == 0:
            return rows
    return 0


def pack_head(w: jax.Array, rows: int) -> jax.Array:
    """``[vocab, hidden]`` head -> the ``[vocab // rows, hidden, rows]`` blocks
    the kernel streams."""
    vocab, hidden = w.shape
    return w.reshape(vocab // rows, rows, hidden).transpose(0, 2, 1)


def _kernel(
    x_ref,
    w_hbm,
    o_ref,
    w_vmem,
    sems,
    *,
    num_blocks: int,
    tiling: LogitsTiling,
    first_block: jax.Array | None = None,
):
    i = pl.program_id(0)
    dmas, buffers = tiling.block_dmas, tiling.buffers
    piece = w_hbm.shape[1] // dmas

    def copies(block, slot):
        if first_block is not None:
            block = first_block + block
        return [
            pltpu.make_async_copy(
                w_hbm.at[block, pl.ds(d * piece, piece)],
                w_vmem.at[slot, pl.ds(d * piece, piece)],
                sems.at[slot, d],
            )
            for d in range(dmas)
        ]

    @pl.when(i == 0)
    def _prologue():
        for block in range(min(buffers - 1, num_blocks)):
            for c in copies(block, block):
                c.start()

    ahead = i + buffers - 1

    @pl.when(ahead < num_blocks)
    def _prefetch():
        for c in copies(ahead, ahead % buffers):
            c.start()

    slot = i % buffers
    for c in copies(i, slot):
        c.wait()
    o_ref[...] = jax.lax.dot_general(
        x_ref[...],
        w_vmem[slot],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(o_ref.dtype)


def _shard_kernel(first_ref, *refs, num_blocks: int, tiling: LogitsTiling):
    _kernel(*refs, num_blocks=num_blocks, tiling=tiling, first_block=first_ref[0])


def _pallas_logits(
    x: jax.Array,
    w: jax.Array,
    first_block: jax.Array | None,
    num_blocks: int,
    tiling: LogitsTiling,
    interpret: bool,
) -> jax.Array:
    """Logits of ``num_blocks`` consecutive vocab blocks of ``w`` (packed),
    starting at block ``first_block[0]`` (block 0 when None)."""
    m, hidden = x.shape
    rows = w.shape[2]
    if hidden % (16 * tiling.block_dmas):
        raise ValueError(
            f"hidden {hidden} does not split into {tiling.block_dmas} DMAs"
        )
    cols = num_blocks * rows
    sharded = first_block is not None
    w_bytes = tiling.buffers * rows * hidden * w.dtype.itemsize
    io_bytes = 2 * m * (hidden + rows) * x.dtype.itemsize
    # Index maps get the scalar-prefetch ref as a trailing argument.
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=int(sharded),
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((m, hidden), lambda i, *_: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=pl.BlockSpec((m, rows), lambda i, *_: (0, i)),
        scratch_shapes=[
            pltpu.VMEM((tiling.buffers, hidden, rows), w.dtype),
            pltpu.SemaphoreType.DMA((tiling.buffers, tiling.block_dmas)),
        ],
    )
    operands = (first_block, x, w) if sharded else (x, w)
    return pl.pallas_call(
        functools.partial(
            _shard_kernel if sharded else _kernel,
            num_blocks=num_blocks,
            tiling=tiling,
        ),
        out_shape=jax.ShapeDtypeStruct((m, cols), x.dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=w_bytes + io_bytes + LOGITS_VMEM_SLACK_BYTES,
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * m * hidden * cols,
            bytes_accessed=(cols * hidden + m * hidden + m * cols) * x.dtype.itemsize,
            transcendentals=0,
        ),
        name="streamed_logits",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*operands)


@functools.partial(jax.jit, static_argnames=("tiling", "interpret"))
def _streamed_logits(
    x: jax.Array, w: jax.Array, tiling: LogitsTiling, interpret: bool = False
) -> jax.Array:
    return _pallas_logits(x, w, None, w.shape[0], tiling, interpret)


def streamed_logits(x: jax.Array, w: jax.Array, tiling: LogitsTiling) -> jax.Array:
    """``x @ head.T`` in ``x.dtype``, f32-accumulated, for ``w = pack_head(head,
    block_rows(vocab, tiling))``."""
    return _streamed_logits(x, w, tiling)


def vocab_shard_layout(
    vocab: int, num_shards: int, tiling: LogitsTiling
) -> tuple[int, int, tuple[int, ...]]:
    """How ``num_shards`` ranks split the head's vocab blocks between them.

    Returns ``(rows, blocks_per_shard, first_blocks)``: every shard streams the
    same number of whole ``rows``-row blocks, so it computes equal-width logits
    an all-to-all can exchange. When the blocks do not divide evenly, the last
    shard starts early enough to end on the last block and overlaps the one
    before it; ``vocab_shard_columns`` says which of its columns are new.
    Returns ``blocks_per_shard == 0`` when the head cannot be split this way.
    """
    rows = block_rows(vocab, tiling)
    if rows == 0:
        return 0, 0, ()
    num_blocks = vocab // rows
    per_shard = -(-num_blocks // num_shards)
    first = tuple(min(r * per_shard, num_blocks - per_shard) for r in range(num_shards))
    return rows, per_shard, first


def vocab_shard_columns(
    vocab: int, num_shards: int, tiling: LogitsTiling
) -> tuple[tuple[int, int], ...]:
    """For each shard, the ``[start, stop)`` columns of its logits that
    concatenate, shard by shard, into the full vocab."""
    rows, per_shard, first = vocab_shard_layout(vocab, num_shards, tiling)
    num_blocks = vocab // rows
    spans = []
    for r, f in enumerate(first):
        lo = min(r * per_shard, num_blocks)
        hi = min((r + 1) * per_shard, num_blocks)
        spans.append(((lo - f) * rows, (max(lo, hi) - f) * rows))
    return tuple(spans)


@functools.partial(jax.jit, static_argnames=("num_blocks", "tiling", "interpret"))
def _streamed_logits_shard(
    x: jax.Array,
    w: jax.Array,
    first_block: jax.Array,
    num_blocks: int,
    tiling: LogitsTiling,
    interpret: bool = False,
) -> jax.Array:
    return _pallas_logits(x, w, first_block, num_blocks, tiling, interpret)


def streamed_logits_shard(
    x: jax.Array,
    w: jax.Array,
    first_block: jax.Array,
    num_shards: int,
    tiling: LogitsTiling,
) -> jax.Array:
    """``x @ head[shard].T`` for this rank's shard of ``vocab_shard_layout``,
    ``w`` packed as for `streamed_logits`: ``first_block`` is int32 ``[1]``
    holding the shard's first block, data rather than a constant so every rank
    compiles the same program."""
    blocks, _, rows = w.shape
    layout_rows, per_shard, _ = vocab_shard_layout(blocks * rows, num_shards, tiling)
    if layout_rows != rows:
        raise ValueError(f"head packed in {rows}-row blocks, tiling has {layout_rows}")
    return _streamed_logits_shard(
        x, w, first_block, num_blocks=per_shard, tiling=tiling
    )
