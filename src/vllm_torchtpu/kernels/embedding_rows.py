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
"""Embedding lookup ``table[ids]`` from a table stored without row pairing.

In a bf16 ``[vocab, hidden]`` array every 32-bit word holds two adjacent rows,
so a lookup has to read each row's neighbour too. ``pack_table`` stores the
table as int32 ``[vocab, hidden // 2]`` instead: word ``k`` of a row holds
columns ``k`` (low half) and ``k + hidden // 2`` (high half), so a row is its
own 32-bit row. The lookup DMAs even- and odd-numbered ids into two VMEM
buffers and re-pairs each pair's halves into the bf16 layout with bit ops. It
only moves bits, so the output equals ``table[ids]`` exactly.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Ids gathered per grid step; the next step's rows are fetched while this
# step's are re-paired.
EMBED_BLOCK_ROWS = 128
# Row DMAs issued per unrolled loop iteration.
EMBED_ISSUE_UNROLL = 8

# A block is a multiple of this so each parity buffer fills whole (8, 128)
# vregs.
_ROW_GRANULE = 16
# Lanes re-paired per vector op.
_REPAIR_LANES = 512


def packed_shape(vocab: int, hidden: int) -> tuple[int, int]:
    return vocab, hidden // 2


def can_pack(hidden: int) -> bool:
    half = hidden // 2
    return hidden % 256 == 0 and half % min(_REPAIR_LANES, half) == 0


def pack_table(table: np.ndarray, chunk_rows: int = 16384) -> np.ndarray:
    """bf16 ``[vocab, hidden]`` (as uint16 bits) -> packed int32 table."""
    vocab, hidden = table.shape
    bits = table.view(np.uint16)
    half = hidden // 2
    packed = np.empty((vocab, half), np.uint32)
    for r in range(0, vocab, chunk_rows):
        rows = slice(r, r + chunk_rows)
        packed[rows] = bits[rows, :half]
        packed[rows] |= bits[rows, half:].astype(np.uint32) << 16
    return packed.view(np.int32)


def _block_rows(tokens: int) -> int:
    """The largest multiple of _ROW_GRANULE <= EMBED_BLOCK_ROWS dividing tokens."""
    for rows in range(EMBED_BLOCK_ROWS, 0, -_ROW_GRANULE):
        if tokens % rows == 0:
            return rows
    raise ValueError(f"{tokens} tokens is not a multiple of {_ROW_GRANULE}")


def _kernel(ids_ref, table_hbm, out_ref, buf, sems, *, rows: int, vocab: int):
    i = pl.program_id(0)
    half = table_hbm.shape[1]

    def issue(block, slot):
        @pl.loop(0, rows // 2, unroll=EMBED_ISSUE_UNROLL)
        def _(j):
            for parity in range(2):
                row = ids_ref[block * rows + 2 * j + parity]
                # torch's index semantics: negative ids wrap, then clamp.
                row = jnp.where(row < 0, row + vocab, row)
                row = jnp.minimum(jnp.maximum(row, 0), vocab - 1)
                pltpu.make_async_copy(
                    table_hbm.at[pl.ds(row, 1)],
                    buf.at[slot, parity, pl.ds(j, 1)],
                    sems.at[slot],
                ).start()

    @pl.when(i == 0)
    def _():
        issue(0, 0)

    @pl.when(i + 1 < pl.num_programs(0))
    def _():
        issue(i + 1, (i + 1) % 2)

    slot = i % 2
    whole = buf.at[slot]
    # One wait for the bytes of every row copy into the slot.
    pltpu.make_async_copy(whole, whole, sems.at[slot]).wait()

    chunk = min(_REPAIR_LANES, half)
    for c in range(0, half, chunk):
        lanes = pl.ds(c, chunk)
        even = whole[0, :, lanes]
        odd = whole[1, :, lanes]
        # bf16 rows 2j and 2j+1 share a word, the even row in the low half.
        lo = (even & 0xFFFF) | (odd << 16)
        hi = jax.lax.shift_right_logical(even, 16) | (odd & -65536)
        out_ref[:, lanes] = pltpu.bitcast(lo, jnp.bfloat16)
        out_ref[:, pl.ds(half + c, chunk)] = pltpu.bitcast(hi, jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("interpret",))
def _embedding_rows(
    ids: jax.Array, packed: jax.Array, interpret: bool = False
) -> jax.Array:
    (tokens,) = ids.shape
    vocab, half = packed.shape
    hidden = 2 * half
    padded = -(-tokens // _ROW_GRANULE) * _ROW_GRANULE
    ids = ids.astype(jnp.int32)
    if padded != tokens:
        ids = jnp.pad(ids, (0, padded - tokens))
    rows = _block_rows(padded)
    out = pl.pallas_call(
        functools.partial(_kernel, rows=rows, vocab=vocab),
        out_shape=jax.ShapeDtypeStruct((padded, hidden), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(padded // rows,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec((rows, hidden), lambda i, ids: (i, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, 2, rows // 2, half), packed.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        # Every id is clamped into the table before its copy is issued.
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), disable_bounds_checks=True
        ),
        cost_estimate=pl.CostEstimate(
            flops=0, bytes_accessed=4 * padded * hidden, transcendentals=0
        ),
        name="embedding_rows",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(ids, packed)
    return out[:tokens] if padded != tokens else out


def embedding_rows(ids: jax.Array, packed: jax.Array) -> jax.Array:
    """bf16 ``table[ids]`` for ``packed = pack_table(table)``."""
    return _embedding_rows(ids, packed)
