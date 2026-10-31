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
here each grid step moves whole contiguous vocab-row slabs, several DMAs in
flight, and contracts them over the full hidden dim in one MXU dot with f32
accumulation, as the dot does.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Vocab rows per grid step.
LOGITS_BLOCK_ROWS = 1280
# DMAs a block's weight slab is split into.
LOGITS_BLOCK_DMAS = 2
# Weight slabs resident in VMEM: the one being contracted plus prefetches.
LOGITS_BUFFERS = 3
# Headroom on top of the buffers for Mosaic's internal scratch.
LOGITS_VMEM_SLACK_BYTES = 8 * 1024 * 1024


def block_rows(vocab: int) -> int:
    """The largest block <= LOGITS_BLOCK_ROWS that tiles ``vocab``, or 0."""
    granule = 16 * LOGITS_BLOCK_DMAS
    for rows in range(LOGITS_BLOCK_ROWS, 0, -granule):
        if vocab % rows == 0:
            return rows
    return 0


def _kernel(x_ref, w_hbm, o_ref, w_vmem, sems, *, rows: int, num_blocks: int):
    i = pl.program_id(0)
    piece = rows // LOGITS_BLOCK_DMAS

    def copies(block, slot):
        return [
            pltpu.make_async_copy(
                w_hbm.at[pl.ds(block * rows + d * piece, piece)],
                w_vmem.at[slot, pl.ds(d * piece, piece)],
                sems.at[slot, d],
            )
            for d in range(LOGITS_BLOCK_DMAS)
        ]

    @pl.when(i == 0)
    def _prologue():
        for block in range(min(LOGITS_BUFFERS - 1, num_blocks)):
            for c in copies(block, block):
                c.start()

    ahead = i + LOGITS_BUFFERS - 1

    @pl.when(ahead < num_blocks)
    def _prefetch():
        for c in copies(ahead, ahead % LOGITS_BUFFERS):
            c.start()

    slot = i % LOGITS_BUFFERS
    for c in copies(i, slot):
        c.wait()
    o_ref[...] = jax.lax.dot_general(
        x_ref[...],
        w_vmem[slot],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("rows", "interpret"))
def _streamed_logits(
    x: jax.Array, w: jax.Array, rows: int, interpret: bool = False
) -> jax.Array:
    m, hidden = x.shape
    vocab = w.shape[0]
    num_blocks = vocab // rows
    w_bytes = LOGITS_BUFFERS * rows * hidden * w.dtype.itemsize
    io_bytes = 2 * m * (hidden + rows) * x.dtype.itemsize
    return pl.pallas_call(
        functools.partial(_kernel, rows=rows, num_blocks=num_blocks),
        out_shape=jax.ShapeDtypeStruct((m, vocab), x.dtype),
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((m, hidden), lambda i: (0, 0)),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=pl.BlockSpec((m, rows), lambda i: (0, i)),
        scratch_shapes=[
            pltpu.VMEM((LOGITS_BUFFERS, rows, hidden), w.dtype),
            pltpu.SemaphoreType.DMA((LOGITS_BUFFERS, LOGITS_BLOCK_DMAS)),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=w_bytes + io_bytes + LOGITS_VMEM_SLACK_BYTES,
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * m * hidden * vocab,
            bytes_accessed=(vocab * hidden + m * hidden + m * vocab) * x.dtype.itemsize,
            transcendentals=0,
        ),
        name="streamed_logits",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(x, w)


def streamed_logits(x: jax.Array, w: jax.Array) -> jax.Array:
    """``x @ w.T`` in ``x.dtype``, f32-accumulated; needs block_rows(vocab) > 0."""
    rows = block_rows(w.shape[0])
    if rows == 0:
        raise ValueError(f"no logits block tiles vocab {w.shape[0]}")
    return _streamed_logits(x, w, rows)
