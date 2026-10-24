# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index scores directly from gathered BF16 index-cache word rows.

A native pool word row (`int32[128]`) holds `128 // (dim // 2)` BF16 index
keys in split-element order: the low 16 bits of a key's words hold its even
elements and the high 16 bits hold its odd elements. Instead of splitting and
relayouting gathered word rows into two BF16 HBM arrays, this Pallas kernel
lays out the keys in VMEM (one per sublane in record order via two stride-2
stores), reconstructs each key with a half-lane roll and select, and runs the
`K = dim` MXU dot product and FP32 ReLU-weighted head reduction in VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROW_WORDS = 128
DIM = 128  # BF16 elements per key: two keys per word row
SLOTS = ROW_WORDS // (DIM // 2)
MAX_BLOCK_ROWS = 4096  # 2 MiB words_ref + 4 MiB keys_ref scratch per VMEM block
_COMPILER_PARAMS = pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))


def _bf16(bits: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def _kernel(q_ref, w_ref, words_ref, out_ref, keys_ref) -> None:
    words = words_ref[0]
    rows, half = words.shape[0], ROW_WORDS // 2
    if q_ref.shape[0] == 1:
        # Place record SLOTS * r + s on sublane SLOTS * r + s with its words
        # rolled to lanes [0, 64) so scores exit in record order without a
        # post-kernel [slots, rows] -> [rows, slots] relayout.
        keys_ref[pl.ds(0, rows, stride=SLOTS), :] = words
        keys_ref[pl.ds(1, rows, stride=SLOTS), :] = pltpu.roll(words, half, 1)
        words = keys_ref[...]
        lo, hi = _bf16(words << 16), _bf16(words & jnp.int32(-65536))
        first = jax.lax.broadcasted_iota(jnp.int32, words.shape, 1) < half
        key = jnp.where(first, lo, pltpu.roll(hi, half, 1))
        logits = jax.lax.dot_general(
            q_ref[0], key, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        out_ref[0, 0] = (jnp.maximum(logits, 0) * w_ref[0]).sum(0)
    else:
        rolled = pltpu.roll(words, half, 1)
        first = jax.lax.broadcasted_iota(jnp.int32, words.shape, 1) < half
        keys_ref[pl.ds(0, rows, stride=SLOTS), :] = jnp.where(
            first, words << 16, rolled & jnp.int32(-65536)
        )
        keys_ref[pl.ds(1, rows, stride=SLOTS), :] = jnp.where(
            first, rolled << 16, words & jnp.int32(-65536)
        )
        key = _bf16(keys_ref[...])
        heads = q_ref.shape[1]
        all_logits = jax.lax.dot_general(
            q_ref[...].reshape(q_ref.shape[0] * heads, DIM),
            key,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        for t in range(q_ref.shape[0]):
            logits = all_logits[t * heads : (t + 1) * heads]
            out_ref[t, 0] = (jnp.maximum(logits, 0) * w_ref[t]).sum(0)


@functools.partial(jax.jit, static_argnames=("interpret",))
def score_word_rows(
    q: jax.Array, weights: jax.Array, words: jax.Array, *, interpret: bool = False
) -> jax.Array:
    """Compute ReLU-weighted index scores for BF16 keys packed in `words`.

    Args:
        q: BF16 query tensor of shape `[tokens, heads, DIM]` in split-element
            order.
        weights: FP32 per-head weights of shape `[tokens, heads, 1]`.
        words: Packed `int32` key rows of shape `[groups, rows, ROW_WORDS]`,
            where `tokens % groups == 0` and each group of `tokens // groups`
            consecutive queries shares one `[rows, ROW_WORDS]` word tile.
        interpret: Whether to execute `pallas_call` in CPU interpret mode.

    Returns:
        FP32 index scores of shape `[tokens, SLOTS * rows]` in record order.

    Raises:
        ValueError: If input shapes or dtypes violate kernel contracts, or if
            `rows` is not a power of two `>= 8`.
    """
    if q.ndim != 3 or weights.ndim != 3 or words.ndim != 3:
        raise ValueError(
            "Expected 3D tensors: q [tokens, heads, 128], "
            "weights [tokens, heads, 1], words [groups, rows, 128]"
        )
    groups, rows, row_words = words.shape
    tokens, heads, q_dim = q.shape
    if (
        row_words != ROW_WORDS
        or groups < 1
        or tokens < groups
        or tokens % groups != 0
        or q_dim != DIM
        or weights.shape != (tokens, heads, 1)
        or q.dtype != jnp.bfloat16
        or weights.dtype != jnp.float32
        or words.dtype != jnp.int32
    ):
        raise ValueError(
            "Expected BF16 q [tokens, heads, 128], FP32 weights [tokens, heads, 1], "
            "and int32 words [groups, rows, 128] with tokens % groups == 0"
        )
    if rows < 8 or rows & (rows - 1):
        raise ValueError("Word rows per group must be a power of two >= 8")
    group_tokens = tokens // groups
    block = min(rows, MAX_BLOCK_ROWS)
    # Constrain `words` to HBM so XLA does not insert a serialized whole-tile
    # HBM->VMEM copy before the Pallas call; DMA reads then pipeline across
    # the `rows // block` grid.
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
        compiler_params=_COMPILER_PARAMS,
        interpret=interpret,
        name="dsv41_index_word_scores",
    )(q, weights, words).reshape(tokens, SLOTS * rows)
