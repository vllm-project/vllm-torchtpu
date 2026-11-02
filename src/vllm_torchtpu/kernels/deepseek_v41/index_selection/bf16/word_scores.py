# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index scores directly from gathered packed halfplane key rows.

Each ``int32[128]`` row packs two BF16 records: low16 holds the even
record and high16 the odd record. Each record uses split feature order
(even64 then odd64). A native TPU bitcast exposes the records in order for
the dot and FP32 ReLU-weighted head sum.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS, SLOTS

MAX_BLOCK_ROWS = 4096


def _kernel(q_ref, w_ref, words_ref, out_ref):
    key = pltpu.bitcast(words_ref[0], jnp.bfloat16)
    for t in range(q_ref.shape[0]):
        logits = jax.lax.dot_general(
            q_ref[t], key, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        out_ref[t, 0] = (jnp.maximum(logits, 0) * w_ref[t]).sum(0)


# Jitted: the memory-space constraint below cannot run eagerly.
@functools.partial(jax.jit, static_argnames=("interpret",))
def score_word_rows(
    q: jax.Array,
    weights: jax.Array,
    words: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Scores ``[tokens, SLOTS * rows]`` of the keys in ``words``.

    ``q``: BF16 ``[tokens, heads, DIM]`` in the split element order.
    ``weights``: FP32 ``[tokens, heads, 1]``. ``words``: int32
    ``[groups, rows, ROW_WORDS]`` where ``tokens % groups == 0``; each group of
    ``tokens // groups`` consecutive queries shares one ``[rows, ROW_WORDS]``
    word tile in canonical halfplane layout. ``rows`` must be a power of two >= 8.
    """
    groups, rows, row_words = words.shape
    tokens, heads = q.shape[:2]
    if (
        row_words != ROW_WORDS
        or groups < 1
        or tokens < groups
        or tokens % groups != 0
        or q.shape != (tokens, heads, DIM)
        or weights.shape != (tokens, heads, 1)
    ):
        raise ValueError("Expected BF16 [tokens, heads, 128] keys in 128-word rows")
    if rows < 8 or rows & (rows - 1):
        raise ValueError("Word rows per token must be a power of two >= 8")
    group_tokens = tokens // groups
    block = min(rows, MAX_BLOCK_ROWS)
    # Left unconstrained, XLA stages the whole (gathered) tile into VMEM with
    # a serialized copy before the call: 3.55 us per 8 MB tile in the 40L
    # decode profile. Pipelined HBM reads overlap the scoring instead: 106K
    # keys at 2 rows 0.264 -> 0.254 ms.
    words = pltpu.with_memory_space_constraint(words, pltpu.MemorySpace.HBM)
    keys = SLOTS * block
    return pl.pallas_call(
        _kernel,
        grid=(groups, rows // block),
        in_specs=[
            pl.BlockSpec((group_tokens, heads, DIM), lambda i, j: (i, 0, 0)),
            pl.BlockSpec((group_tokens, heads, 1), lambda i, j: (i, 0, 0)),
            pl.BlockSpec((1, block, ROW_WORDS), lambda i, j: (i, j, 0)),
        ],
        # A unit middle axis keeps the block's last two dims TPU-legal.
        out_specs=pl.BlockSpec((group_tokens, 1, keys), lambda i, j: (i, 0, j)),
        out_shape=jax.ShapeDtypeStruct((tokens, 1, SLOTS * rows), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
        name="dsv41_index_word_scores",
    )(q, weights, words).reshape(tokens, SLOTS * rows)
