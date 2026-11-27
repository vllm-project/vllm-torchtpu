# SPDX-License-Identifier: Apache-2.0
"""LM heads with the vocabulary sharded over the DP ranks of an EP mesh.

Under DP attention every rank streams the whole [vocab, hidden] head for its
own few rows. Here each rank holds one ``shard``-row slice of the head, the
ranks' rows are exchanged first, and each rank multiplies every rank's rows by
its slice, so a step streams the head once across the mesh instead of once per
rank. The functions run inside ``shard_map`` over ``AXIS``; every rank passes
equal shapes.

- ``target_logits``: the logits come back to their rows' rank by an all-to-all,
  so each rank ends with full-vocabulary logits for its own rows, the values
  the replicated head gives (each logit is the same full-K dot).
- ``dspark_samples``: the DSpark draft head plus its greedy Markov chain. Each
  rank runs the replicated sampler's Markov kernel on its vocabulary slice of
  the base logits and Markov head; each slot's per-tile winners are reduced
  over ranks with the kernel's own rule (NaN first, then the largest BF16
  score, then the lowest token id), so the tokens equal the replicated
  sampler's.

Rows are exchanged as all-to-alls of copies rather than all-gathers, as the
Engram HBM lookup does (`deepseek_v41/common/engram_hbm.py`), to stay off the
SparseCore all-gather offload that serving configures.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.logits_head import SLICE_TILE, logits_head_slice
from vllm_torchtpu.kernels.markov_sampling import reduce_candidates, slot_candidates

# Single character: torch_tpu's local-shape helper iterates a spec entry.
AXIS = "d"


def shard_rows(vocab_size: int, world: int) -> int:
    """Rows of each rank's head slice: the vocabulary split ``world`` ways,
    rounded up to whole ``SLICE_TILE`` tiles (the last slices are padded)."""
    per_rank = -(-vocab_size // world)
    return -(-per_rank // SLICE_TILE) * SLICE_TILE


def markov_rows(shard: int) -> int:
    """Rows of each rank's Markov head slice: the head slice rounded up to the
    Markov kernel's 512-column chunks, so one kernel tile covers it."""
    return -(-shard // 512) * 512


def _gather_rows(x: jax.Array, world: int) -> jax.Array:
    """[n, ...] on every rank -> [world, n, ...], rank-ordered, on every rank."""
    copies = jnp.broadcast_to(x[None], (world, *x.shape))
    return jax.lax.all_to_all(copies, AXIS, 0, 0, tiled=True)


def target_logits(
    hidden: jax.Array,
    weight_shard: jax.Array,
    *,
    world: int,
    vocab_size: int,
    chunks: int = 1,
) -> jax.Array:
    """[R, K] rows and this rank's [S, K] head slice -> [R, vocab] logits.

    ``chunks`` splits the slice's tiles so each chunk's exchange can overlap
    the next chunk's matmul."""
    rows, width = hidden.shape
    shard = weight_shard.shape[0]
    tiles = shard // SLICE_TILE
    every_row = _gather_rows(hidden, world).reshape(world * rows, width)
    parts = []
    start = 0
    for c in range(chunks):
        count = (tiles - start) // (chunks - c)
        part = logits_head_slice(
            every_row, weight_shard, tile_start=start, num_tiles=count
        ).reshape(world, rows, count * SLICE_TILE)
        # [destination rank, R, cols] -> [source slice, R, cols]
        parts.append(jax.lax.all_to_all(part, AXIS, 0, 0, tiled=True))
        start += count
    logits = parts[0] if chunks == 1 else jnp.concatenate(parts, axis=2)
    logits = jnp.transpose(logits, (1, 0, 2)).reshape(rows, world * shard)
    return logits[:, :vocab_size]


def dspark_samples(
    hidden: jax.Array,
    anchors: jax.Array,
    markov_embed: jax.Array,
    markov_shard: jax.Array,
    weight_shard: jax.Array,
    rank: jax.Array,
    *,
    world: int,
    vocab_size: int,
) -> jax.Array:
    """Greedy DSpark block tokens for this rank's requests.

    hidden: [slots * n, K] normed head inputs, slot-major (slot s holds rows
        s*n .. s*n+n-1), n requests per slot (zero-padded);
    anchors: [n] token ids seeding the Markov chain;
    markov_embed: the full [vocab, r] Markov embedding;
    markov_shard / weight_shard: this rank's [markov_rows(S), r] Markov head
        and [S, K] LM head slices (vocabulary rows from rank*S; zero rows past
        S or the vocabulary);
    rank: [1, 1] this rank's mesh index.
    Returns [n, slots] int32 token ids.
    """
    n = anchors.shape[0]
    width = hidden.shape[1]
    slots = hidden.shape[0] // n
    shard = weight_shard.shape[0]
    # [rank, slot, n, K] -> [slot, rank, n, K]: slot-major over every rank.
    every_row = jnp.transpose(
        _gather_rows(hidden.reshape(slots, n, width), world),
        (1, 0, 2, 3),
    ).reshape(slots * world * n, width)
    base = logits_head_slice(every_row, weight_shard, num_tiles=shard // SLICE_TILE)
    me = rank[0, 0]
    # Widened to the Markov slice; columns past the vocabulary score -inf, and
    # a real token at -inf still wins their tie, being lower.
    cols = markov_shard.shape[0]
    col = jnp.arange(cols, dtype=jnp.int32)
    valid = (col < shard) & (me * shard + col < vocab_size)
    base = jnp.pad(base, ((0, 0), (0, cols - shard)))
    base = jnp.where(valid[None, :], base, -jnp.inf).astype(base.dtype)
    prev = _gather_rows(anchors.astype(jnp.int32), world).reshape(world * n)
    no_index = jnp.iinfo(jnp.int32).max
    tokens = []
    for slot in range(slots):
        # The replicated sampler's kernel on this slice: the same per-token
        # BF16 scores.
        value, index = slot_candidates(
            markov_embed[prev], markov_shard, base, slot=slot, group=cols
        )
        index = jnp.where(index == no_index, no_index, index + me * shard)
        both = jnp.stack([jax.lax.bitcast_convert_type(value, jnp.int32), index])
        every = _gather_rows(both, world)  # [world, 2, tiles, W*n]
        prev = reduce_candidates(
            jax.lax.bitcast_convert_type(every[:, 0], jnp.float32).reshape(
                -1, world * n
            ),
            every[:, 1].reshape(-1, world * n),
        )
        tokens.append(prev)
    every = jnp.stack(tokens, axis=1)  # [W*n, slots]
    return jax.lax.dynamic_slice_in_dim(every, me * n, n, axis=0)
