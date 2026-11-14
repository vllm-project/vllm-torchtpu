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

Step i of the chain drafts ``argmax(logits_i + markov_w2 @ markov_w1[prev])``
over the whole vocabulary, ``prev`` the previous step's draft. Each of the n
ranks holds a vocab shard of ``logits_i`` and ``markov_w2``; the XLA loop runs
per step a bias matmul, a shard argmax, an 8-way all-gather of one (max, id)
pair per row and the next embedding gather as separate ops, ~26 us a step of
which ~11 us is idle (c4 b1, served j-9986c92e). Here the n ranks run the K
steps in one kernel: the shard of ``markov_w2`` stays in VMEM, the pairs go to
every peer by remote DMA, and the next step's ``markov_w1`` rows are fetched
by DMA at the winning ids.

The scores are kept in f32 (bf16 logits plus the f32-accumulated bias), and
ties go to the lowest vocabulary id (the XLA path's first index of the first
rank). The XLA loop's bf16 `F.linear` and add round where XLA does not elide
the round trip: rounding every score to bf16 here turned near-ties into exact
ties won by the lowest id and differed from the XLA loop on 3-27 % of drafts
at served shapes, the f32 scores on 0-6 % (kernel benches j-b346c6bb,
j-43f953a9). NaN scores: the XLA path picks the first NaN, the kernel skips
them.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32 = jnp.bfloat16, jnp.float32
COLLECTIVE_ID = 0
LANES = 128
# Columns per bias-matmul chunk: bounds the f32 [rows, chunk] intermediate.
CHUNK = 3584
_MESH = pl.DeviceIdType.MESH
_BIG_ID = float(2**30)
# Scalar-prefetch header: this rank's mesh index, its shard's first id, and
# its number of real (unpadded) columns; the anchor ids follow.
_HEADER = 3


# VMEM the kernel may ask for (tpu7x has 64 MiB per core).
MAX_VMEM_BYTES = 48 << 20


def padded_width(width: int) -> int:
    """Shard columns padded to whole `CHUNK`s."""
    return -(-width // CHUNK) * CHUNK


def vmem_bytes(n: int, steps: int, rows: int, width: int, half: int) -> int:
    """VMEM of one call: the markov_w2 halves, two steps of logits, a chunk's
    f32 intermediates, the exchange slots, and headroom."""
    w2 = 2 * half * width * 2
    logits = 2 * rows * width * 2
    chunk = 4 * rows * CHUNK * 4
    exchange = (steps * n + steps) * rows * LANES * 4
    return w2 + logits + chunk + exchange + (8 << 20)


def fits(n: int, steps: int, rows: int, width: int, half: int) -> bool:
    """Whether a call of this shape fits `MAX_VMEM_BYTES` (and its output's
    ``LANES`` columns)."""
    return steps <= LANES and vmem_bytes(n, steps, rows, width, half) <= MAX_VMEM_BYTES


@functools.cache
def _call(n: int, steps: int, rows: int, width: int, vocab: int, half: int, interpret):
    def kernel(
        scal,
        logits_hbm,
        w1_hbm,
        w2_even,
        w2_odd,
        out_ref,
        ids_smem,
        ids_vmem,
        emb,
        lg,
        send,
        recv,
        lg_sem,
        emb_sem,
        id_sem,
        send_sem,
        recv_sem,
    ):
        me = scal[0]
        start = scal[1]
        valid = scal[2]
        barrier = pltpu.get_barrier_semaphore()
        for d in range(n):
            pl.semaphore_signal(barrier, inc=1, device_id=(d,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, n)

        def logits_copy(step):
            return pltpu.make_async_copy(
                logits_hbm.at[step], lg.at[step % 2], lg_sem.at[step % 2]
            )

        def embed_rows(index_of):
            def copy(r):
                return pltpu.make_async_copy(
                    w1_hbm.at[pl.ds(index_of(r), 1)], emb.at[pl.ds(r, 1)], emb_sem
                )

            def start_one(r, c):
                copy(r).start()
                return c

            def wait_one(r, c):
                copy(r).wait()
                return c

            lax.fori_loop(0, rows, start_one, 0)
            lax.fori_loop(0, rows, wait_one, 0)

        def bf16_halves(words):
            """The even and odd bf16 elements packed in int32 words."""
            even = lax.bitcast_convert_type(jnp.left_shift(words, 16), F32)
            odd = lax.bitcast_convert_type(words & jnp.int32(-65536), F32)
            return even.astype(BF16), odd.astype(BF16)

        logits_copy(0).start()
        embed_rows(lambda r: scal[_HEADER + r])
        lane = lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)
        sends = []
        for step in range(steps):
            if step + 1 < steps:
                logits_copy(step + 1).start()
            logits_copy(step).wait()
            even, odd = bf16_halves(emb[...])

            def chunk(j, carry, step=step, even=even, odd=odd):
                best, best_col = carry
                c0 = j * CHUNK  # static: dynamic lane offsets lower to slow loads
                cols = pl.ds(c0, CHUNK)
                bias = jnp.dot(
                    even, w2_even[:, cols], preferred_element_type=F32
                ) + jnp.dot(odd, w2_odd[:, cols], preferred_element_type=F32)
                logits = lg[step % 2, :, pl.ds(c0, CHUNK)].astype(F32)
                scores = logits + bias
                for g in range(CHUNK // LANES):
                    col = c0 + g * LANES + lane
                    v = scores[:, g * LANES : (g + 1) * LANES]
                    v = jnp.where(col < valid, v, -jnp.inf)
                    take = v > best
                    best = jnp.where(take, v, best)
                    best_col = jnp.where(take, col, best_col)
                return best, best_col

            init = (
                jnp.full((rows, LANES), -jnp.inf, F32),
                jnp.zeros((rows, LANES), jnp.int32),
            )
            best, best_col = init
            for j in range(width // CHUNK):
                best, best_col = chunk(j, (best, best_col))
            m = jnp.max(best, axis=1, keepdims=True)
            col = jnp.min(jnp.where(best == m, best_col, width), axis=1, keepdims=True)
            gid = (col + start).astype(F32)
            send[step] = jnp.where(lane == 0, m, jnp.where(lane == 1, gid, 0.0))
            recv[step, me] = send[step]
            for i in range(1, n):
                t = lax.rem(me + i, n)
                copy = pltpu.make_async_remote_copy(
                    send.at[step],
                    recv.at[step, me],
                    send_sem.at[step, i],
                    recv_sem.at[step, me],
                    device_id=(t,),
                    device_id_type=_MESH,
                )
                copy.start()
                sends.append(copy)
            for i in range(1, n):
                t = lax.rem(me + n - i, n)
                pltpu.make_async_copy(
                    recv.at[step, t], recv.at[step, t], recv_sem.at[step, t]
                ).wait_recv()
            win_v = jnp.full((rows, 1), -jnp.inf, F32)
            win_id = jnp.full((rows, 1), _BIG_ID, F32)
            for s in range(n):
                pair = recv[step, s]
                v, d = pair[:, 0:1], pair[:, 1:2]
                take = (v > win_v) | ((v == win_v) & (d < win_id))
                win_v = jnp.where(take, v, win_v)
                win_id = jnp.where(take, d, win_id)
            ids = win_id.astype(jnp.int32)
            out_ref[:, step : step + 1] = ids
            if step + 1 < steps:
                ids_vmem[...] = jnp.broadcast_to(ids, (rows, LANES))
                to_smem = pltpu.make_async_copy(ids_vmem, ids_smem, id_sem)
                to_smem.start()
                to_smem.wait()
                embed_rows(lambda r: ids_smem[r, 0])
        for copy in sends:
            copy.wait_send()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    scratch = [
        pltpu.SMEM((rows, LANES), jnp.int32),
        pltpu.VMEM((rows, LANES), jnp.int32),
        pltpu.VMEM((rows, half), jnp.int32),
        pltpu.VMEM((2, rows, width), BF16),
        pltpu.VMEM((steps, rows, LANES), F32),
        pltpu.VMEM((steps, n, rows, LANES), F32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA(()),
        pltpu.SemaphoreType.DMA(()),
        pltpu.SemaphoreType.DMA((steps, n)),
        pltpu.SemaphoreType.DMA((steps, n)),
    ]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, LANES), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm, hbm, vmem, vmem],
            out_specs=vmem,
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=vmem_bytes(n, steps, rows, width, half),
        ),
        name=f"markov_argmax-k_{steps}-r_{rows}-n_{n}",
        interpret=interpret,
    )


def split_halves(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``markov_w2`` rows ``[W, 2h]`` -> its even and odd columns, transposed
    to ``[h, W]`` each: the order `markov_argmax` unpacks the int32 words of
    ``markov_w1`` in, laid out for a plain (not transposed) MXU matmul."""
    return w[:, 0::2].T, w[:, 1::2].T


def markov_argmax(
    scalars: jax.Array,
    logits: jax.Array,
    markov_w1_words: jax.Array,
    w2_even: jax.Array,
    w2_odd: jax.Array,
    n: int,
    interpret=False,
) -> jax.Array:
    """Draft ids int32 ``[rows, 128]`` (column i = step i) of the greedy
    Markov chain; call under shard_map over the 1-D mesh of the n ranks.

    ``scalars`` int32 ``[3 + rows]``: this device's mesh index, the first
    vocabulary id of its shard, its count of real columns, then each row's
    anchor id. ``logits`` bf16 ``[steps, rows, W]`` this rank's shard of each
    step's logits, W = `padded_width` of the shard. ``markov_w1_words`` int32
    ``[vocab, D / 2]``: the whole bf16 ``markov_w1 [vocab, D]`` viewed as
    int32 (a row's bf16 pairs per word, so a row is a DMA of whole words).
    ``w2_even`` / ``w2_odd`` bf16 ``[D / 2, W]``: this rank's rows of
    ``markov_w2`` (zero past the real columns) split by `split_halves`.
    """
    steps, rows, width = logits.shape
    vocab, half = markov_w1_words.shape
    assert w2_even.shape == w2_odd.shape == (half, width), w2_even.shape
    if not fits(n, steps, rows, width, half):
        raise ValueError(f"markov_argmax: {rows} rows of {width} columns exceed VMEM")
    return _call(n, steps, rows, width, vocab, half, interpret)(
        scalars, logits, markov_w1_words, w2_even, w2_odd
    )
