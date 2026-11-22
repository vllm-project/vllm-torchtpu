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
"""Dequantize an SP step's gathered e4m3 row pieces into one bf16 block.

Piece ``k`` (``q [n_k, H]`` e4m3, per-row f32 scales ``[n_k, 1]``) lands as
``bf16(q * scale)`` in its rows of the ``[sum n_k, H]`` output: one HBM pass
over the pieces, where XLA dequantizes each piece and then concatenates them,
re-reading and rewriting every row once more (and, fused into one pass, runs
the concatenate at a third of the HBM rate).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MAX_BLOCK_ROWS = 256


def _block_rows(n: int) -> int:
    for rows in (_MAX_BLOCK_ROWS, 128, 64, 32):
        if n % rows == 0:
            return rows
    return n


def _dequant_block(q_ref, s_ref, out_ref):
    out_ref[...] = (q_ref[...].astype(jnp.float32) * s_ref[...]).astype(out_ref.dtype)


def _kernel(*refs):
    *pieces, out_ref = refs
    offset = 0
    for q_ref, s_ref in zip(pieces[::2], pieces[1::2]):
        n, h = q_ref.shape
        rows = _block_rows(n)
        block = lambda i: (i, 0)  # noqa: E731
        pltpu.emit_pipeline(
            _dequant_block,
            grid=(n // rows,),
            in_specs=[pl.BlockSpec((rows, h), block), pl.BlockSpec((rows, 1), block)],
            out_specs=[pl.BlockSpec((rows, h), block)],
        )(q_ref, s_ref, out_ref.at[pl.ds(offset, n)])
        offset += n


@functools.partial(jax.jit, static_argnames=("interpret",))
def join_pieces(
    pieces: tuple[tuple[jax.Array, jax.Array], ...], interpret: bool = False
) -> jax.Array:
    """The pieces' bf16 rows ``bf16(q * scale)``, in order, as one block."""
    h = pieces[0][0].shape[1]
    operands = []
    for q, scale in pieces:
        assert q.shape[1] == h and scale.shape == (q.shape[0], 1), (
            q.shape,
            scale.shape,
        )
        operands += [q, scale.astype(jnp.float32)]
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        _kernel,
        in_specs=[any_spec] * len(operands),
        out_specs=any_spec,
        out_shape=jax.ShapeDtypeStruct(
            (sum(q.shape[0] for q, _ in pieces), h), jnp.bfloat16
        ),
        name="minimax_m3_rows_join",
        interpret=interpret,
    )(*operands)
