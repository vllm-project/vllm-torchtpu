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
"""The DSpark greedy Markov chain over a vocab-parallel LM head, in one call.

The chain drafts K tokens per request, one after the other: token i is
``argmax(logits[i] + markov_w2 @ markov_w1[token i-1])`` over the whole vocab,
and the vocab is split over the n TP ranks. Done with XLA ops, every step is
a bias matvec, a local argmax, an all-gather of one (max, id) pair per rank and
a pick: seven latency-bound all-gathers in a row, ~21 us per step at decode.

This kernel runs the K steps in one Pallas call. The rank's Markov rows stay
in VMEM for all steps, the next step's logits are fetched while a step runs,
and the pairs go to the peers by remote DMA (one barrier for the whole call,
not one per step). Per step and rank:

  e      = markov_w1[prev]                 (one DMA per row, from HBM)
  bias   = bf16(e @ w2_shard.T)            (f32 accumulation, rounded once)
  scores = bf16(logits[i] + bias)          (pad columns masked to -inf)
  pair   = (max score, first global id of the max)
  prev   = id of the largest max over the ranks (the lowest id on ties)

which is the arithmetic of `markov_shard_pair` + `reduce_shard_argmax_pairs`.
A NaN score is not given torch.argmax's NaN-wins semantics here.

Step i's pairs land in a receive slot of parity i % 2: a rank sends step i + 1
only after it holds every rank's step-i pair, so a peer is at most one step
ahead and two slots are enough.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32 = jnp.bfloat16, jnp.float32
COLLECTIVE_ID = 0
_MESH = pl.DeviceIdType.MESH
_LANES = 128
# Rows padded to whole bf16 sublane tiles.
_SUBLANES = 16
_SUBLANES_32 = 8
# Vocab columns per bias tile (a multiple of 128 lanes).
TILE_COLS = 3584
# Most rows (requests) one call takes: two logits slots of this many rows plus
# the Markov rows stay well inside VMEM. Larger draft buckets keep XLA's loop.
MAX_ROWS = 64


def padded_vocab(vs: int) -> int:
    """Shard width padded to a whole number of bias tiles."""
    return -(-vs // TILE_COLS) * TILE_COLS


def padded_rows(rows: int) -> int:
    return -(-rows // _SUBLANES) * _SUBLANES


def vmem_bytes(rows: int, vp: int, rank_dim: int) -> int:
    rp = padded_rows(rows)
    return (
        vp * rank_dim * 2  # Markov rows
        + 2 * rp * vp * 2  # two logits slots
        + rp * 16 * rank_dim * 4  # embedding row blocks
        + 6 * rp * _LANES * 4  # pairs
        + 2 * 8 * rp * _LANES * 4  # receive slots (n <= 8)
    )


@functools.cache
def _call(
    n: int,
    k: int,
    rp: int,
    vp: int,
    vs: int,
    rank_dim: int,
    vocab: int,
    w1_dtype,
    rows: int,
    interpret=None,
    ablate: frozenset = frozenset(),
):
    # `ablate` (timing attribution only; wrong ids): "gather" skips the
    # embedding DMAs, "dot" the bias matmul, "score" the whole tile loop,
    # "exchange" the remote DMAs, "w2load" the Markov row load, "barrier"
    # the entry barrier (only with "exchange"). "T" (same ids) streams the
    # Markov rows through the MXU against stationary embeddings.
    # Only the `rows` real rows gather embeddings and hand ids to the scalar
    # core; pad rows up to rp keep zero embeddings and are never read.
    nt = vp // TILE_COLS
    # Rows per (sublane) tile of markov_w1: 8 for 32-bit, 16 for bf16.
    blk = _SUBLANES_32 * 4 // jnp.dtype(w1_dtype).itemsize
    assert vocab % blk == 0, (vocab, blk)

    def kernel(
        rank_ref,  # SMEM [1] mesh index (remote DMA addressing)
        start_ref,  # SMEM [1] global id of this rank's first vocab column
        prev0_ref,  # SMEM [rp]
        logits_hbm,  # ANY [k, rp, vp] bf16
        w1_hbm,  # ANY [vocab, rank_dim] bf16
        w2_hbm,  # ANY [vp, rank_dim] bf16
        out_ref,  # SMEM [k, rp] int32
        w2_v,
        lg_v,
        e_v,
        send_v,
        recv_v,
        prev_s,
        load_sem,
        lg_sem,
        e_sem,
        send_sem,
        recv_sem,
    ):
        me = rank_ref[0]

        # The Markov rows load one tile per DMA; step 0 waits for each tile
        # just before it uses it, so the load overlaps the barrier, the first
        # gathers and step 0's own tiles.
        def w2_copy(t):
            c0 = pl.multiple_of(t * TILE_COLS, TILE_COLS)
            return pltpu.make_async_copy(
                w2_hbm.at[pl.ds(c0, TILE_COLS)],
                w2_v.at[pl.ds(c0, TILE_COLS)],
                load_sem.at[t],
            )

        if "w2load" not in ablate:
            for t in range(nt):
                w2_copy(t).start()

        def lg_copy(i):
            return pltpu.make_async_copy(
                logits_hbm.at[i], lg_v.at[i % 2], lg_sem.at[i % 2]
            )

        lg_copy(0).start()
        for r in range(rows):
            prev_s[r] = prev0_ref[r]

        if "barrier" not in ablate:
            barrier = pltpu.get_barrier_semaphore()
            for d in range(n):
                pl.semaphore_signal(
                    barrier, inc=1, device_id=(d,), device_id_type=_MESH
                )
            pl.semaphore_wait(barrier, n)

        lane = lax.broadcasted_iota(jnp.int32, (rp, _LANES), 1)
        col = lax.broadcasted_iota(jnp.int32, (rp, TILE_COLS), 1)
        start = start_ref[0]

        for i in range(k):
            par = i % 2
            # A DMA offset along the tiled row axis must be tile aligned:
            # fetch the aligned block of `blk` rows holding each row.
            gathers = []
            for r in range(0 if "gather" in ablate else rows):
                b0 = pl.multiple_of((prev_s[r] // blk) * blk, blk)
                g = pltpu.make_async_copy(
                    w1_hbm.at[pl.ds(b0, blk)], e_v.at[r], e_sem.at[0]
                )
                g.start()
                gathers.append(g)
            if i + 1 < k:
                lg_copy(i + 1).start()
            lg_copy(i).wait()
            for g in gathers:
                g.wait()
            # Pick row prev % blk of each block and assemble [rp, r] in f32
            # (exact for bf16 values).
            erow = lax.broadcasted_iota(jnp.int32, (rp, rank_dim), 0)
            bsub = lax.broadcasted_iota(jnp.int32, (blk, rank_dim), 0)
            e = jnp.zeros((rp, rank_dim), F32)
            for r in range(rows):
                hit = bsub == lax.rem(prev_s[r], blk)
                row = jnp.sum(
                    jnp.where(hit, e_v[r].astype(F32), 0.0), axis=0, keepdims=True
                )
                e = jnp.where(erow == r, row, e)
            e = e.astype(BF16)
            if "T" in ablate:
                # Markov rows as the streamed operand, embeddings stationary.
                e = jnp.concatenate([e, jnp.zeros((_LANES - rp, rank_dim), BF16)], 0)

            def tile(t, carry, par=par, e=e, first=(i == 0)):
                best_m, best_i = carry
                if first and "w2load" not in ablate:
                    w2_copy(t).wait()
                c0 = pl.multiple_of(t * TILE_COLS, TILE_COLS)
                w = w2_v[pl.ds(c0, TILE_COLS), :]
                if "dot" in ablate:
                    bias = jnp.zeros((rp, TILE_COLS), BF16) + w[0:1, 0:1]
                elif "T" in ablate:
                    bias_t = lax.dot_general(
                        w, e, (((1,), (1,)), ((), ())), preferred_element_type=F32
                    )
                    bias = bias_t.T[:rp].astype(BF16)
                else:
                    bias = lax.dot_general(
                        e, w, (((1,), (1,)), ((), ())), preferred_element_type=F32
                    ).astype(BF16)
                lg = lg_v[par, :, pl.ds(c0, TILE_COLS)]
                s = (lg.astype(F32) + bias.astype(F32)).astype(BF16).astype(F32)
                # Column ids in f32 (exact below 2**24): the first max by min.
                gcol = (col + c0).astype(F32)
                s = jnp.where(gcol < vs, s, -jnp.inf)
                m = jnp.max(s, axis=1, keepdims=True)
                idx = jnp.min(jnp.where(s == m, gcol, 2.0**30), axis=1, keepdims=True)
                take = m > best_m
                return jnp.where(take, m, best_m), jnp.where(take, idx, best_i)

            best_m, best_i = lax.fori_loop(
                0,
                0 if "score" in ablate else nt,
                tile,
                (
                    jnp.full((rp, 1), -jnp.inf, F32),
                    jnp.zeros((rp, 1), F32),
                ),
            )
            if i == 0 and "score" in ablate and "w2load" not in ablate:
                for t in range(nt):
                    w2_copy(t).wait()
            gid = best_i + (jnp.zeros((rp, 1), jnp.int32) + start).astype(F32)
            pair = jnp.where(lane == 0, best_m, jnp.where(lane == 1, gid, 0.0))
            send_v[par] = pair
            recv_v[par, me] = pair
            sends = []
            for step in range(1, 1 if "exchange" in ablate else n):
                t = lax.rem(me + step, n)
                cp = pltpu.make_async_remote_copy(
                    send_v.at[par],
                    recv_v.at[par, me],
                    send_sem.at[par, step],
                    recv_sem.at[par, me],
                    device_id=(t,),
                    device_id_type=_MESH,
                )
                cp.start()
                sends.append(cp)
            for step in range(1, 1 if "exchange" in ablate else n):
                t = lax.rem(me + n - step, n)
                pltpu.make_async_copy(
                    recv_v.at[par, t], recv_v.at[par, t], recv_sem.at[par, t]
                ).wait_recv()
            # Largest max; equal maxima -> the lower global id, whatever the
            # mesh order of the ranks.
            win = recv_v[par, 0]
            for s_ in range(1, n):
                cand = recv_v[par, s_]
                cv, wv = cand[:, 0:1], win[:, 0:1]
                better = (cv > wv) | ((cv == wv) & (cand[:, 1:2] < win[:, 1:2]))
                win = jnp.where(better, cand, win)
            ids = win[:, 1:2].astype(jnp.int32)  # [rp, 1]
            row = lax.broadcasted_iota(jnp.int32, (rp, 1), 0)
            for r in range(rows):
                v = jnp.sum(jnp.where(row == r, ids, 0))
                v = jnp.clip(v, 0, vocab - 1)
                prev_s[r] = v
                out_ref[i, r] = v
            for cp in sends:
                cp.wait_send()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((k, rp), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[hbm, hbm, hbm],
            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
            scratch_shapes=[
                pltpu.VMEM((vp, rank_dim), BF16),
                pltpu.VMEM((2, rp, vp), BF16),
                pltpu.VMEM((rp, blk, rank_dim), w1_dtype),
                pltpu.VMEM((2, rp, _LANES), F32),
                pltpu.VMEM((2, n, rp, _LANES), F32),
                pltpu.SMEM((rp,), jnp.int32),
                pltpu.SemaphoreType.DMA((nt,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((2, n)),
                pltpu.SemaphoreType.DMA((2, n)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=vmem_bytes(rp, vp, rank_dim) + (8 << 20),
        ),
        name=f"dspark_markov_argmax-k_{k}-r_{rows}-v_{vp}-n_{n}",
        **({} if interpret is None else {"interpret": interpret}),
    )


def markov_argmax(
    logits: jax.Array,
    prev0: jax.Array,
    w1: jax.Array,
    w2p: jax.Array,
    rank: jax.Array,
    start: jax.Array,
    n: int,
    vs: int,
    interpret=None,
    ablate: frozenset = frozenset(),
) -> jax.Array:
    """Greedy Markov drafts ``[rows, k]`` int32 (global ids).

    Call under shard_map over the 1-D TP mesh, per rank:
      logits [rows, k, vs] bf16  this rank's vocab shard of the draft logits
      prev0  [rows] int          each row's first ``prev`` (the anchor id)
      w1     [vocab, r] bf16/f32 markov_w1 (replicated); vocab a multiple of
                                 the row tile (16 for bf16)
      w2p    [vp, r] bf16        this rank's markov_w2 rows, zero-padded to
                                 ``padded_vocab(vs)``
      rank   [1] int32           this rank's mesh index
      start  [1] int32           global id of this rank's first vocab column
    """
    rows, k, width = logits.shape
    assert rows <= MAX_ROWS, (rows, MAX_ROWS)
    assert width == vs, (width, vs)
    vp, rank_dim = w2p.shape
    assert vp == padded_vocab(vs), (vp, vs)
    vocab = w1.shape[0]
    rp = padded_rows(rows)
    lg = jnp.transpose(logits, (1, 0, 2))
    lg = jnp.pad(lg, ((0, 0), (0, rp - rows), (0, vp - vs)))
    p0 = jnp.pad(prev0.astype(jnp.int32), (0, rp - rows))
    out = _call(
        n, k, rp, vp, vs, rank_dim, vocab, jnp.dtype(w1.dtype), rows, interpret, ablate
    )(
        rank.reshape(1).astype(jnp.int32),
        start.reshape(1).astype(jnp.int32),
        p0,
        lg,
        w1,
        w2p,
    )
    return jnp.transpose(out)[:rows]
