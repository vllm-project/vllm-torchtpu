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
"""A wave-major ragged all-to-all as a pair of split-phase Pallas kernels.

XLA builds an async ragged all-to-all, but its scheduler places start and done
back to back and `AsyncCollectiveMerger` turns the pair back into one sync op,
so nothing can run under the exchange. Splitting it into custom calls -- a
start kernel that only ISSUES the remote DMAs and returns their semaphores, and
consumers that wait on them -- puts the overlap under the caller's control
(jax async design note, third_party/py/jax/docs/pallas/design/async_note.md).

The exchange is issued in `waves` groups: the caller lays the dispatch plan out
by expert group, and the start kernel issues group 0 to every destination, then
group 1, and so on. Every row of group w lands on the receiver's semaphore
`recv_sems[w]`, so a receiver can start computing on group 0
while later groups are still in flight. The fused MoE kernel does exactly that:
it takes the receive semaphores as an operand and waits for group w only before
its first expert in group w.

Table: the [4 * ep, waves] int32 dispatch plan. Rows (k * ep + d, w): k=0 send
offset, k=1 send rows, k=2 offset in d's receive buffer; row 3 * ep holds the
rows this shard receives per group, row 3 * ep + 1 col 0 the total rows it
sends to OTHER shards. A sender signals the receiver's semaphore for group w,
so the receiver's count for group w is the sum over sources -- exactly row
3 * ep -- and its waits are byte-exact (every semaphore returns to zero).

Invariants:
  * Senders only `wait_send` (never `wait`), once, after the consumer kernel
    (`ragged_a2a_send_done`): every outgoing DMA had the whole kernel to
    finish, and `wait_send` never blocks on a peer.
  * The start kernel opens with an all-pairs barrier, so no shard writes a
    peer's receive buffer before that peer is executing the same program
    point (its buffer is live there; SPMD buffer assignment is identical).
  * `send` and `recv` are threaded through the kernels with
    input_output_aliases, and the send drain threads a value the caller uses,
    so XLA can neither free, reuse nor dead-code-eliminate them while DMAs
    are in flight.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Distinct from the fused kernel's COLLECTIVE_ID (0): its own barrier semaphore.
A2A_COLLECTIVE_ID = 1

_MESH = pl.DeviceIdType.MESH


def _all_pairs_barrier(ep):
    sem = pltpu.get_barrier_semaphore()
    for i in range(ep):
        pl.semaphore_signal(sem, inc=1, device_id=(jnp.int32(i),), device_id_type=_MESH)
    pl.semaphore_wait(sem, ep)


def _start_waves_kernel(
    tab,
    me_ref,
    send,
    recv_in,
    send_sem,
    recv_sems,
    send_out,
    recv_out,
    *,
    ep,
    waves,
    chunk_rows,
):
    del recv_in, send_out  # aliased to recv_out / send
    me = me_ref[0]
    _all_pairs_barrier(ep)
    cs = jnp.int32(chunk_rows)
    for w in range(waves):
        # The first chunk_rows rows of every destination, then the rest of
        # each destination's run in one copy: every peer's first piece is in
        # flight early, and a wave costs at most 2 * ep copies however
        # skewed it is (chunking every run cost a loop body per chunk of the
        # busiest destination, times ep).
        for head in (True, False):
            # Destination order rotated by `me` (as XLA's own a2a does) so the
            # shards do not all hit peer 0 first.
            for i in range(ep):
                d = lax.rem(me + i, ep)
                n = tab[ep + d, w]
                lo = jnp.int32(0) if head else cs
                m = jnp.minimum(cs, n) if head else n - cs
                src = send.at[pl.ds(tab[d, w] + lo, m)]
                dst = recv_out.at[pl.ds(tab[2 * ep + d, w] + lo, m)]
                if i == 0:  # d == me: a local copy on this shard's own slot

                    @pl.when(m > 0)
                    def _(src=src, dst=dst, w=w):
                        pltpu.make_async_copy(src, dst, recv_sems.at[w]).start()
                else:

                    @pl.when(m > 0)
                    def _(src=src, dst=dst, d=d, w=w):
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst,
                            send_sem=send_sem,
                            recv_sem=recv_sems.at[w],
                            device_id=(d,),
                            device_id_type=_MESH,
                        ).start()


def ragged_a2a_start_waves(send, recv, disp_tab, me, *, ep, waves, chunk_rows):
    """Issue the group-major exchange.

    send [rows, ...] holds this shard's packed send rows and recv (e.g.
    uninitialized HBM) the receive buffer; disp_tab is the [4 * ep, waves]
    plan described in the module docstring and me this shard's index.
    Returns (send_sem, recv_sems [waves], send, recv): a consumer waits
    recv_sems[w] for group w's rows in recv,
    and `ragged_a2a_send_done` drains send_sem afterwards.
    """
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    anyspec = pl.BlockSpec(memory_space=pl.ANY)
    semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    return pl.pallas_call(
        functools.partial(
            _start_waves_kernel, ep=ep, waves=waves, chunk_rows=chunk_rows
        ),
        out_shape=(
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA((waves,)),
            jax.ShapeDtypeStruct(send.shape, send.dtype),
            jax.ShapeDtypeStruct(recv.shape, recv.dtype),
        ),
        in_specs=[smem, smem, anyspec, anyspec],
        out_specs=(semspec, semspec, anyspec, anyspec),
        input_output_aliases={2: 2, 3: 3},
        compiler_params=pltpu.CompilerParams(collective_id=A2A_COLLECTIVE_ID),
        name=f"sparse_a2a_start_ep{ep}_w{waves}",
    )(disp_tab.astype(jnp.int32), jnp.reshape(me, (1,)).astype(jnp.int32), send, recv)


def _send_done_kernel(tab, send, send_sem, carry, carry_out, *, ep):
    del carry, carry_out  # aliased pass-through (orders + keeps this alive)
    total = tab[3 * ep + 1, 0]

    @pl.when(total > 0)
    def _():
        # One byte-exact wait for every row this shard sent to other shards
        # (all on the one send semaphore); `wait_send` never blocks on peers.
        pltpu.make_async_remote_copy(
            src_ref=send.at[pl.ds(0, total)],
            dst_ref=send.at[pl.ds(0, total)],
            send_sem=send_sem,
            recv_sem=send_sem,
            device_id=(jnp.int32(0),),
            device_id_type=_MESH,
        ).wait_send()


def ragged_a2a_send_done(send_sem, send, disp_tab, carry, *, ep):
    """Drain the start kernel's outgoing DMAs; returns `carry` (aliased).

    `carry` must be a value the caller USES (e.g. the fused kernel's
    arrivals): threading it through makes the drain impossible to
    dead-code-eliminate and orders it after `carry`'s producer. Without it
    the drain was eliminated, so the send semaphore and send buffer died
    right after the start kernel and XLA reused them while the DMAs were
    still in flight (the fused kernel's semaphores ended nonzero and garbage
    rows were sent).
    """
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    anyspec = pl.BlockSpec(memory_space=pl.ANY)
    semspec = pl.BlockSpec(memory_space=pltpu.SEMAPHORE)
    return pl.pallas_call(
        functools.partial(_send_done_kernel, ep=ep),
        out_shape=jax.ShapeDtypeStruct(carry.shape, carry.dtype),
        in_specs=[smem, anyspec, semspec, anyspec],
        out_specs=anyspec,
        input_output_aliases={3: 0},
        name=f"sparse_a2a_send_done_ep{ep}",
    )(disp_tab.astype(jnp.int32), send, send_sem, carry)
