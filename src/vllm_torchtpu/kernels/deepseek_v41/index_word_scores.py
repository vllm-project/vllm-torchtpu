# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index scores straight from gathered BF16 index-cache word rows.

A native pool word row (``int32[128]``) holds ``128 // (dim // 2)`` BF16
index keys in the split element order (`cache.split_halves`): the low halves
of a key's words are its even elements, the high halves its odd ones. XLA
scores such rows by splitting the words into two BF16 arrays and relaying
them out as ``[keys, dim]`` (two HBM copies per tile). This kernel instead
lays the keys out in VMEM, one per sublane in record order (two stride-2
stores of the word rows), rebuilds each key with a half-lane roll and a
select, then runs the same ``K = dim`` dot and FP32 ReLU-weighted head sum,
so the scores are bitwise those of ``einsum("thd,tcd->thc")`` +
``(max(., 0) * w).sum(heads)`` and come out in record order.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROW_WORDS = 128
DIM = 128  # BF16 elements per key: two keys per word row
SLOTS = ROW_WORDS // (DIM // 2)
MAX_BLOCK_ROWS = 4096


def _bf16(bits: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def unpack_word_rows(words, keys_ref):
    """Unpack BF16 keys into split element order, preserving record order."""
    rows, half = words.shape[0], ROW_WORDS // 2
    # Record SLOTS * r + s (slot s of row r) on sublane SLOTS * r + s, its
    # words rolled to lanes [0, 64): scores then leave in record order, with
    # no [slots, rows] -> [rows, slots] relayout in XLA (0.27 ms/step).
    keys_ref[pl.ds(0, rows, stride=SLOTS), :] = words
    keys_ref[pl.ds(1, rows, stride=SLOTS), :] = pltpu.roll(words, half, 1)
    words = keys_ref[...]
    lo, hi = _bf16(words << 16), _bf16(words & jnp.int32(-65536))
    first = jax.lax.broadcasted_iota(jnp.int32, words.shape, 1) < half
    # Lanes [0, 64) of lo (even elements), then the same lanes of hi (odd
    # elements), i.e. q's split element order.
    return jnp.where(first, lo, pltpu.roll(hi, half, 1))


def _kernel(q_ref, w_ref, words_ref, out_ref, keys_ref):
    key = unpack_word_rows(words_ref[0], keys_ref)
    for t in range(q_ref.shape[0]):
        logits = jax.lax.dot_general(
            q_ref[t], key, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        out_ref[t, 0] = (jnp.maximum(logits, 0) * w_ref[t]).sum(0)


# Jitted: the memory-space constraint below cannot run eagerly.
@functools.partial(jax.jit, static_argnames=("interpret",))
def score_word_rows(
    q: jax.Array, weights: jax.Array, words: jax.Array, *, interpret: bool = False
) -> jax.Array:
    """Scores ``[tokens, SLOTS * rows]`` of the keys in ``words``.

    ``q``: BF16 ``[tokens, heads, DIM]`` in the split element order.
    ``weights``: FP32 ``[tokens, heads, 1]``. ``words``: int32
    ``[groups, rows, ROW_WORDS]`` where ``tokens % groups == 0``; each group of
    ``tokens // groups`` consecutive queries shares one ``[rows, ROW_WORDS]``
    word tile. ``rows`` must be a power of two >= 8.
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
        scratch_shapes=[pltpu.VMEM((keys, ROW_WORDS), jnp.int32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=interpret,
        name="dsv41_index_word_scores",
    )(q, weights, words).reshape(tokens, SLOTS * rows)
