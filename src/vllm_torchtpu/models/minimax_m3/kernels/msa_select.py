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
"""Pallas top-k block selection for MiniMax Sparse Attention.

A row tile of block scores stays in VMEM while `topk` rounds of max /
first-index-of-max / mask-out pick the blocks, so the selection costs one
read of the scores instead of a sort (or `topk` passes) over HBM.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import tight_vmem


DEFAULT_BLOCK_ROWS = 256
_NEG_INF = float("-inf")
_FORCED_LOCAL_SCORE = 1e29
_FORCED_INIT_SCORE = 1e30
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024


def _kernel(
    scores_ref, pos_ref, out_ref, *, block_size, topk, init_blocks, local_blocks
):
    s = scores_ref[...]
    rows, num_blocks = s.shape
    blocks = jax.lax.broadcasted_iota(jnp.int32, (rows, num_blocks), 1)
    num_visible = pos_ref[...] // block_size + 1
    visible = blocks < num_visible
    s = jnp.where(visible, s, _NEG_INF)
    s = jnp.where(
        visible & (blocks >= num_visible - local_blocks), _FORCED_LOCAL_SCORE, s
    )
    s = jnp.where(visible & (blocks < init_blocks), _FORCED_INIT_SCORE, s)
    picked = []
    for _ in range(topk):
        best_score = jnp.max(s, axis=1, keepdims=True)
        best = jnp.min(
            jnp.where(s == best_score, blocks, num_blocks), axis=1, keepdims=True
        )
        picked.append(jnp.where(best_score > _NEG_INF, best, -1))
        s = jnp.where(blocks == best, _NEG_INF, s)
    out_ref[...] = jnp.concatenate(picked, axis=1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "topk",
        "init_blocks",
        "local_blocks",
        "block_rows",
        "interpret",
    ),
)
def select_blocks(
    scores: jax.Array,
    pos: jax.Array,
    *,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    interpret: bool = False,
) -> jax.Array:
    """Selected logical blocks ``[T, 1, topk]``, ``-1`` padded at the end.

    Blocks come in selection order (forced blocks, then by score), not sorted:
    the attention matches pages by value, so ordering would only add a sort.

    Args:
      scores: ``[T, 1, P]`` block scores (one index head per rank).
      pos: ``[T]`` absolute position of each token.
    """
    num_tokens, num_heads, num_blocks = scores.shape
    if num_heads != 1:
        raise NotImplementedError(
            f"MSA block selection expects one index head per rank, got {num_heads}."
        )
    pad = -num_tokens % block_rows
    s = jnp.pad(scores[:, 0], ((0, pad), (0, 0)))
    p = jnp.pad(pos, (0, pad))[:, None]
    picked = tight_vmem.pallas_call(
        functools.partial(
            _kernel,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        ),
        grid=((num_tokens + pad) // block_rows,),
        in_specs=[
            pl.BlockSpec((block_rows, num_blocks), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, 1), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((num_tokens + pad, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT_BYTES
        ),
        interpret=interpret,
        name="minimax_m3_msa_select_blocks",
    )(s, p)[:num_tokens]
    return picked[:, None, :]
