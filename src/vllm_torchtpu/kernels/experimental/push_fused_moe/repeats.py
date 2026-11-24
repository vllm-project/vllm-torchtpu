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
"""Which quantized rows repeat the row before them, bit for bit.

The routed FP8 rows q [t, hidden / 128, 128] keep their rows on the major
axis, above the (sublane, lane) tiling, so a row and the one before it are
different vregs and comparing them needs no relayout. XLA cannot see that: a
shifted compare of q's 32-bit words lowered to a 1.44 ms slice on the DP16
2048-row step. This kernel walks q's row blocks in order, compares each row
with the one before (the previous block's last row carried in VMEM), and
writes one flag per row.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_TILES = (256, 128, 96, 64, 48, 32, 16, 8)


def row_block(t: int) -> int:
    """Rows per grid step, or 0 when none divides `t`."""
    return next((r for r in _TILES if t % r == 0), 0)


def supports(q: jax.Array) -> bool:
    """[t, blocks, 128] one-byte rows, blocks a multiple of 4 (32-bit words),
    and a row block that divides t."""
    return (
        q.ndim == 3
        and jnp.dtype(q.dtype).itemsize == 1
        and q.shape[1] % 4 == 0
        and q.shape[2] == 128
        and row_block(q.shape[0]) > 0
    )


def _kernel(q_ref, out_ref, last_ref, *, rows):
    step = pl.program_id(0)
    # Four one-byte values per 32-bit word, packed along the sublanes.
    words = pltpu.bitcast(q_ref[...], jnp.uint32)
    prev = jnp.concatenate([last_ref[...], words[:-1]], axis=0)
    eq = (words == prev).astype(jnp.int32)
    same = jnp.min(jnp.min(eq, axis=2), axis=1)
    # Row 0 of the first block has no row before it.
    first = (lax.broadcasted_iota(jnp.int32, (rows,), 0) == 0) & (step == 0)
    out_ref[...] = jnp.where(first, 0, same)[None, None, :]
    last_ref[...] = words[rows - 1 :]


def repeats_previous_row(q: jax.Array, *, interpret=False) -> jax.Array:
    """[t] bool: row r of q equals row r - 1 bit for bit; row 0 False.

    q is [t, lanes / 128, 128] with one-byte elements (`supports`);
    `interpret` is pallas_call's.
    """
    if not supports(q):
        raise ValueError(f"repeats_previous_row takes {q.shape} {q.dtype.name}")
    t, blocks, lanes = q.shape
    rows = row_block(t)
    steps = t // rows
    same = pl.pallas_call(
        functools.partial(_kernel, rows=rows),
        grid=(steps,),
        in_specs=[pl.BlockSpec((rows, blocks, lanes), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((1, 1, rows), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((steps, 1, rows), jnp.int32),
        scratch_shapes=[pltpu.VMEM((1, blocks // 4, lanes), jnp.uint32)],
        # In order: each step reads the row the step before left.
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"moe_repeated_q_rows_t{t}",
        interpret=interpret,
    )(q)
    return same.reshape(t) != 0
