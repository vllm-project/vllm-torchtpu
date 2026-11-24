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
"""DSpark's greedy Markov chain over vocab-parallel draft logits, in one call.

Per draft slot: embed the previous token (a row of the replicated
``markov_w1``), add its rank-r bias on this rank's vocab shard
(``emb @ markov_w2_shard.T``) to the slot's logits, reduce the shard to one
(max, global id) pair per row, exchange the pairs between the n ranks, and take
the first rank's id among the maxima as the next previous token. That is
``spec_decode/vocab_parallel_argmax``'s reduction with the gathered path's bf16
roundings of the bias and of the sum.

As XLA ops a slot is a chain of ~15 small dependent ops around an 8-byte
all-gather. Here the chain runs inside one Pallas call: the Markov shard and
the logits stream through VMEM in double-buffered chunks (the next slot's
first chunk loads while a slot exchanges its pairs), the embedding row is a
DMA, and each slot's pairs go to every peer by remote DMA into a receive slot
of its own, so one barrier at entry covers the whole chain.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32, I32 = jnp.bfloat16, jnp.float32, jnp.int32
# Not tp_all_reduce's, so the two kernels never share a barrier semaphore.
COLLECTIVE_ID = 1
_MESH = pl.DeviceIdType.MESH
_LANES = 128
_ROWS = 16  # bf16 sublane tile; the logits, embeddings and pair tiles pad to it
# Rows and slots are unrolled; beyond the decode batches a draft serves, the
# per-slot XLA loop is as good.
MAX_ROWS = 32
# The vocab shard streams through VMEM in chunks of at most this many 128-lane
# tiles, so the kernel's VMEM stays small beside what XLA keeps there for the
# rest of the draft step.
_CHUNK_TILES = 32


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def padded_vocab(vocab_shard: int) -> int:
    """Lanes of the Markov shard (and of the logits inside the kernel)."""
    return _round_up(vocab_shard, _LANES)


def chunk_lanes(vocab_shard: int) -> int:
    """Lanes of the vocab chunk the kernel streams: the widest whole number of
    128-lane tiles, at most _CHUNK_TILES, that divides the padded shard."""
    tiles = padded_vocab(vocab_shard) // _LANES
    return _LANES * max(t for t in range(1, _CHUNK_TILES + 1) if tiles % t == 0)


def vmem_bytes(n: int, rows: int, slots: int, rank_dim: int, vocab_shard: int):
    """VMEM one call declares: two buffers of a chunk of the Markov shard and
    of the logits, one chunk's f32 temporaries, the embedding, pair and id
    tiles, and headroom."""
    rows_p = _round_up(rows, _ROWS)
    chunk = chunk_lanes(vocab_shard)
    return (
        2 * (rank_dim + rows_p) * chunk * 2  # double-buffered shard and logits
        + 6 * rows_p * chunk * 4  # f32 temporaries of one chunk
        + rows * _ROWS * rank_dim * 2  # embedding tiles
        + (slots * (n + 1) + 1) * rows_p * _LANES * 4  # pair and id tiles
        + (2 << 20)  # headroom for Mosaic's own scratch
    )


def max_rows(n: int, slots: int, rank_dim: int, vocab_shard: int, budget: int) -> int:
    """The most rows (a multiple of 16, at most MAX_ROWS) one call takes
    within ``budget`` bytes of VMEM; 0 when none fit."""
    rows = 0
    while rows < MAX_ROWS and (
        vmem_bytes(n, rows + _ROWS, slots, rank_dim, vocab_shard) <= budget
    ):
        rows += _ROWS
    return rows


@functools.cache
def _call(n: int, rows: int, slots: int, rank_dim: int, vocab_shard: int):
    if rows > MAX_ROWS:
        raise ValueError(f"{rows} rows: the kernel takes at most {MAX_ROWS}")
    rows_p = _round_up(rows, _ROWS)
    chunk = chunk_lanes(vocab_shard)
    chunks = padded_vocab(vocab_shard) // chunk
    vmem_limit = vmem_bytes(n, rows, slots, rank_dim, vocab_shard)
    big = jnp.iinfo(I32).max

    def kernel(
        scalars_ref,  # SMEM [3]: mesh index, first id of the shard, its ids
        prev0_ref,  # SMEM [rows]
        logits_hbm,  # [slots, rows_p, lanes] bf16
        w2t_hbm,  # [rank_dim, lanes] bf16
        w1_hbm,  # [vocab, rank_dim] bf16
        out_ref,  # VMEM [rows_p, 128] int32
        w2_buf,
        logits_buf,
        emb_tiles,
        send,
        recv,
        ids_v,
        ids_s,
        prev_s,
        load_sem,
        emb_sem,
        ids_sem,
        send_sem,
        recv_sem,
    ):
        me, first_id, valid = scalars_ref[0], scalars_ref[1], scalars_ref[2]

        def loads(s, c, buf):
            """The chunk-c DMAs of slot s (the Markov shard's columns do not
            depend on the slot) into buffer buf."""
            cols = pl.ds(pl.multiple_of(c * chunk, _LANES), chunk)
            return (
                pltpu.make_async_copy(
                    w2t_hbm.at[:, cols], w2_buf.at[buf], load_sem.at[0, buf]
                ),
                pltpu.make_async_copy(
                    logits_hbm.at[s, :, cols], logits_buf.at[buf], load_sem.at[1, buf]
                ),
            )

        # The Markov shard and logits stream through two buffers, one chunk
        # ahead, across slot boundaries: slot s + 1's first chunk loads while
        # slot s exchanges its pairs. Step g = s * chunks + c uses buffer g % 2.
        for copy in loads(0, 0, 0):
            copy.start()

        # Every peer is in the kernel (its receive slots are live) before any
        # pair is sent; the first chunk loads meanwhile.
        barrier = pltpu.get_barrier_semaphore()
        for d in range(n):
            pl.semaphore_signal(barrier, inc=1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, n)

        for r in range(rows):
            prev_s[r] = prev0_ref[r]
        tile_row = lax.broadcasted_iota(I32, (_ROWS, rank_dim), 0)
        emb_row = lax.broadcasted_iota(I32, (rows_p, rank_dim), 0)
        lane = lax.broadcasted_iota(I32, (rows_p, chunk), 1)
        pair_lane = lax.broadcasted_iota(I32, (rows_p, _LANES), 1)
        ids = jnp.zeros((rows_p, _LANES), I32)
        sends = []

        for s in range(slots):
            # A DMA moves whole sublane tiles: fetch each previous token's
            # tile of embedding rows, then pick its row.
            embeds = [
                pltpu.make_async_copy(
                    w1_hbm.at[pl.ds(pl.multiple_of(prev_s[r] & -_ROWS, _ROWS), _ROWS)],
                    emb_tiles.at[r],
                    emb_sem.at[r],
                )
                for r in range(rows)
            ]
            for copy in embeds:
                copy.start()
            emb = jnp.zeros((rows_p, rank_dim), F32)
            for r, copy in enumerate(embeds):
                copy.wait()
                tile = emb_tiles[r].astype(F32)
                row = jnp.sum(
                    jnp.where(tile_row == (prev_s[r] & (_ROWS - 1)), tile, 0.0),
                    axis=0,
                    keepdims=True,
                )
                emb = jnp.where(emb_row == r, row, emb)
            emb = emb.astype(BF16)

            def chunk_step(c, carry, s=s, emb=emb):
                best, at, first_nan = carry
                g = s * chunks + c
                buf = g % 2
                for copy in loads(s, c, buf):
                    copy.wait()

                @pl.when(c + 1 < chunks)
                def _():
                    for copy in loads(s, c + 1, 1 - buf):
                        copy.start()

                if s + 1 < slots:

                    @pl.when(c + 1 == chunks)
                    def _():
                        for copy in loads(s + 1, 0, 1 - buf):
                            copy.start()

                # The gathered path's roundings: the bias and the sum are bf16.
                bias = jnp.dot(emb, w2_buf[buf], preferred_element_type=F32)
                logits = logits_buf[buf].astype(F32)
                scores = logits + bias.astype(BF16).astype(F32)
                scores = scores.astype(BF16).astype(F32)
                col = lane + c * chunk
                scores = jnp.where(col < valid, scores, -jnp.inf)
                nan = scores != scores
                top = jnp.max(jnp.where(nan, -jnp.inf, scores), axis=1, keepdims=True)
                hit = (scores == top) & ~nan
                top_at = jnp.min(jnp.where(hit, col, big), axis=1, keepdims=True)
                nan_at = jnp.min(jnp.where(nan, col, big), axis=1, keepdims=True)
                # Strictly greater: an earlier chunk keeps a tie (first index).
                better = top > best
                return (
                    jnp.where(better, top, best),
                    jnp.where(better, top_at, at),
                    jnp.minimum(first_nan, nan_at),
                )

            init = (
                jnp.full((rows_p, 1), -jnp.inf, F32),
                jnp.zeros((rows_p, 1), I32),
                jnp.full((rows_p, 1), big, I32),
            )
            best, at, first_nan = lax.fori_loop(0, chunks, chunk_step, init)
            # As torch.argmax: a NaN is the max, and the first NaN wins.
            has_nan = first_nan < big
            best = jnp.where(has_nan, jnp.nan, best)
            gid = (jnp.where(has_nan, first_nan, at) + first_id).astype(F32)
            send[s] = jnp.where(
                pair_lane == 0, best, jnp.where(pair_lane == 1, gid, 0.0)
            )

            for step in range(1, n):
                t = lax.rem(me + step, n)
                copy = pltpu.make_async_remote_copy(
                    send.at[s],
                    recv.at[s, me],
                    send_sem.at[s, step],
                    recv_sem.at[s, me],
                    device_id=(t,),
                    device_id_type=_MESH,
                )
                copy.start()
                sends.append(copy)
            recv[s, me] = send[s]
            for step in range(1, n):
                t = lax.rem(me + n - step, n)
                pltpu.make_async_copy(
                    recv.at[s, t], recv.at[s, t], recv_sem.at[s, t]
                ).wait_recv()

            # The lowest id among the maxima (a NaN max first): the shards
            # hold increasing ids by TP rank, so this is the first rank
            # torch.argmax picks over the rank-major gathered pairs, whatever
            # the mesh order.
            pairs = recv[s]
            maxima = pairs[:, :, 0:1]
            nan = maxima != maxima
            has_nan = jnp.max(nan.astype(F32), axis=0) > 0
            top = jnp.max(jnp.where(nan, -jnp.inf, maxima), axis=0)
            at = (nan & has_nan) | ((maxima == top) & ~has_nan)
            gid = jnp.min(jnp.where(at, pairs[:, :, 1:2], jnp.inf), axis=0)
            gid = gid.astype(I32)
            ids = jnp.where(pair_lane == s, gid, ids)
            if s + 1 < slots:
                ids_v[...] = jnp.broadcast_to(gid, (rows_p, _LANES))
                copy = pltpu.make_async_copy(ids_v, ids_s, ids_sem.at[0])
                copy.start()
                copy.wait()
                for r in range(rows):
                    prev_s[r] = ids_s[r, 0]

        out_ref[...] = ids
        for copy in sends:
            copy.wait_send()

    any_space = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows_p, _LANES), I32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[any_space, any_space, any_space],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((2, rank_dim, chunk), BF16),
                pltpu.VMEM((2, rows_p, chunk), BF16),
                pltpu.VMEM((rows, _ROWS, rank_dim), BF16),
                pltpu.VMEM((slots, rows_p, _LANES), F32),
                pltpu.VMEM((slots, n, rows_p, _LANES), F32),
                pltpu.VMEM((rows_p, _LANES), I32),
                pltpu.SMEM((rows_p, _LANES), I32),
                pltpu.SMEM((rows,), I32),
                pltpu.SemaphoreType.DMA((2, 2)),
                pltpu.SemaphoreType.DMA((rows,)),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((slots, n)),
                pltpu.SemaphoreType.DMA((slots, n)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID, vmem_limit_bytes=vmem_limit
        ),
        name=f"markov_argmax_chain-r_{rows}-s_{slots}-n_{n}",
    )


def markov_argmax_chain(
    logits: jax.Array,
    w2t: jax.Array,
    w1: jax.Array,
    prev: jax.Array,
    scalars: jax.Array,
    n: int,
) -> jax.Array:
    """Greedy Markov draft ids ``[rows, slots]`` (int32, global vocab ids),
    the same on every rank. Call under shard_map over the 1-D mesh of the n
    vocab shards.

    logits: this rank's bf16 ``[rows, slots, vocab_shard]`` draft logits.
    w2t: this rank's Markov shard transposed, bf16 ``[rank_dim,
    padded_vocab(vocab_shard)]``, zero past ``vocab_shard``. w1: the replicated
    bf16 ``[vocab, rank_dim]`` Markov embedding, vocab a multiple of 16 (it is
    fetched by whole sublane tiles). prev: int ``[rows]``, the
    token before each row's first slot. scalars (int32, three elements): this
    device's mesh index, the first global id of its shard, and how many of the
    shard's ``vocab_shard`` columns are real ids (the rest are padding).
    """
    rows, slots, vocab_shard = logits.shape
    rank_dim, lanes = w2t.shape
    if (
        lanes != padded_vocab(vocab_shard)
        or w1.shape[1] != rank_dim
        or w1.shape[0] % _ROWS
    ):
        raise ValueError(
            f"Markov shard {w2t.shape} / embedding {w1.shape} do not fit "
            f"logits {logits.shape}"
        )
    rows_p = _round_up(rows, _ROWS)
    # Slot-major, padded with -inf rows and lanes (the lanes are also masked).
    logits = jnp.pad(
        logits.transpose(1, 0, 2),
        ((0, 0), (0, rows_p - rows), (0, lanes - vocab_shard)),
        constant_values=-jnp.inf,
    )
    ids = _call(n, rows, slots, rank_dim, vocab_shard)(
        scalars.reshape(3).astype(I32),
        # In range for the embedding DMA, as an XLA gather would clamp it.
        jnp.clip(prev, 0, w1.shape[0] - 1).astype(I32),
        logits,
        w2t,
        w1,
    )
    return ids[:rows, :slots]
