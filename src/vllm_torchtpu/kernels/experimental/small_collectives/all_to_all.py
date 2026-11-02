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
"""All-to-all of small row blocks as one Pallas kernel (one round of remote DMAs).

``x`` is ``[n * block, V]``: block ``d`` is this core's message for core ``d``.
Every core pushes rows ``[row0, block)`` of block ``d`` straight into rows
``[me * block + row0, (me + 1) * block)`` of core ``d``'s output in HBM, copies
its own block into its own slot, and waits for the ``n - 1`` arrivals. Nothing
is staged in VMEM and no vector unit touches the words, so every row at or
past ``row0`` of every output block is bitwise ``lax.all_to_all(x, axis, 0, 0,
tiled=True)``'s; rows before ``row0`` are left unwritten (the caller never
reads them).

It is the DCP LSE merge's exchange at decode (TPU_DCP_MLA_SEND_SMALL_A2A: 8
ranks x a few-KB int16 send block per destination), which XLA runs as a
ragged all-to-all of 9-11 us on the served mirror; the same bytes in one
round of remote DMAs need about one barrier plus one 7-peer round.

With ``live_rows`` (data, the same on every core) only rows ``[row0, end)``
of every block move, ``end`` being ``live_row0 + live_rows`` rounded up to a
whole 8-row tile, in one copy per peer: one ``pl.when`` branch per multiple
of 8 rows up to the block, each a copy of that static size, the last one the
whole block, and the waits count the same size. Rows past ``end`` are left
unwritten. The DCP merge's padding tokens fill the end of the block, so their
partial rows are not sent. In merge mode the copies land in the VMEM
receive scratch at the same rows; the merge reads only rows below
``merge_row + merge_live`` and masks the rest of each chunk, so the
unwritten rows never reach the result.

As in ``all_gather``, one barrier per call is enough: a core leaves call k
only after every peer's block of call k has landed and its own sends are
done, and a peer sends call k + 1 only after the barrier of call k + 1, which
this core enters only after it left call k. The barrier is also what makes a
peer's output buffer safe to write: the peer has entered the kernel that owns
it.

Peer addresses: inside a ``(lane, axis)`` mesh a peer is ``(my lane, j)``;
under the torch bridge both indices arrive as data.

Merge mode (``merge_live``, TPU_DCP_MLA_SEND_SMALL_A2A_MERGE): the blocks are
the DCP merge's send blocks -- row ``merge_row - 1`` the lse word row (the
hi halves of the block's ``tl`` lse bit patterns at lanes ``[0, tl)``, the
lo halves at ``[tl, 2 * tl)``), rows ``[merge_row, block)`` the bf16
partials as 16-bit words. Peers push into a VMEM receive scratch instead of
the output, and after the arrival wait the kernel runs the unpack and
``_merge_received`` on the live rows only, ``MERGE_CHUNK_ROWS`` at a time:
the weights of every sender and row at once (senders in sublanes, rows in
lanes, transposed once), then each live chunk's partials streamed into two
f32 accumulators, senders in order. Chunks past the live count are written
as zeros while the blocks are in flight. It returns the bf16 ``[tl, V]``
merge, each chunk DMA'd out as soon as it is done. The VMEM destination is
safe for the same reason the output is: a peer writes it only after the
barrier, which this core passes only inside the kernel that owns it.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    MESH,
    wait_bytes_of,
)

# Barrier semaphore group, distinct from the fused MoE kernels (0), the
# all-reduces (5), the reduce-scatter (7) and the all-gather (9).
COLLECTIVE_ID = 11
# Merge mode: rows per merged chunk (one bf16 sublane tile).
MERGE_CHUNK_ROWS = 16


def reference(blocks, row0=0):
    """[cores, cores, block, V] (sender, destination, ...) -> what each core
    gets, [cores, cores * block, V], with rows before ``row0`` zero."""
    out = blocks.swapaxes(0, 1).copy()
    out[:, :, :row0] = 0
    n = out.shape[0]
    return out.reshape(n, -1, out.shape[-1])


def live_sizes(count, rows):
    """``(size, taken)`` per multiple of 8 up to ``rows``: the one copy of
    rows ``[0, count)`` of a span of ``rows``, ``count`` a multiple of 8 in
    ``[8, rows]``. Sizes are static; ``taken`` (``size == count``) is data."""
    return [(size, count == size) for size in range(8, rows + 1, 8)]


def merge_sender_rows(num_cores):
    """Merge mode's senders, padded to whole 8-row sublane tiles."""
    return -(-num_cores // 8) * 8


def merge_weights(recv_ref, w_ref, *, num_cores, word_row, tl, rolled=False):
    """``_merge_received``'s weights of every sender and row: the lse words of
    sender ``w`` from row ``word_row`` of slot ``w`` of ``recv_ref``, as f32
    rows (senders in sublanes, rows in lanes), then transposed once into
    ``w_ref`` [LANES, LANES]: row r holds ``w_hi`` of sender w at lane w and
    ``w_lo`` at lane ``senders + w``. ``rolled``: the senders in a loop, not
    unrolled (same values)."""
    n = num_cores
    vdim = recv_ref.shape[-1]
    senders = merge_sender_rows(n)
    head0 = word_row // MERGE_CHUNK_ROWS * MERGE_CHUNK_ROWS
    sub = lax.broadcasted_iota(jnp.int32, (senders, LANES), 0)
    lse = jnp.full((senders, LANES), -jnp.inf, jnp.float32)

    def take_sender(w, lse):
        head = recv_ref[w, pl.ds(head0, MERGE_CHUNK_ROWS), :].astype(jnp.int32)
        words = head[word_row - head0 : word_row - head0 + 1]
        lo = pltpu.roll(words, vdim - tl, 1)
        bits = lax.shift_left(words, jnp.int32(16)) | (lo & 0xFFFF)
        row = lax.bitcast_convert_type(bits, jnp.float32)[:, :LANES]
        return jnp.where(sub == w, jnp.broadcast_to(row, lse.shape), lse)

    if rolled:
        lse = lax.fori_loop(0, n, take_sender, lse)
    else:
        for w in range(n):
            lse = take_sender(w, lse)
    finite = jnp.isfinite(lse)
    shift = jnp.max(jnp.where(finite, lse, -jnp.inf), axis=0, keepdims=True)
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = jnp.where(finite, jnp.exp(lse - shift), 0.0)
    denom = jnp.sum(weights, axis=0, keepdims=True)
    weights = weights / jnp.where(denom > 0, denom, 1.0)
    w_hi = weights.astype(jnp.bfloat16).astype(jnp.float32)
    w_lo = weights - w_hi
    pad = jnp.zeros((LANES - 2 * senders, LANES), jnp.float32)
    w_ref[...] = jnp.concatenate([w_hi, w_lo, pad], axis=0).T


def merge_chunk(recv_ref, w_ref, chunk, live, *, num_cores, part_row, rolled=False):
    """Rows ``[chunk * MERGE_CHUNK_ROWS, ...)`` of the merge, bf16: each
    sender's partials (NaN zeroed, then f32) times its ``w_hi`` and ``w_lo``
    columns, summed in order, rows at or past ``live`` zero. ``rolled``: the
    senders in a loop, each one's columns rolled to lane 0 (same values);
    ``chunk`` and ``part_row`` may then be data, ``part_row`` a whole tile."""
    n = num_cores
    senders = merge_sender_rows(n)
    start = chunk * MERGE_CHUNK_ROWS
    vdim = recv_ref.shape[-1]
    dynamic = not isinstance(chunk, int)
    if rolled or dynamic:
        start = pl.multiple_of(start, MERGE_CHUNK_ROWS)
    cols = w_ref[pl.ds(start, MERGE_CHUNK_ROWS), :]

    def column(lane):
        if rolled:
            return pltpu.roll(cols, lax.rem(LANES - lane, LANES), 1)[:, :1]
        return cols[:, lane : lane + 1]

    def add_sender(k, acc):
        acc_hi, acc_lo = acc
        first = part_row + start
        if rolled or dynamic:
            first = pl.multiple_of(first, MERGE_CHUNK_ROWS)
        words = recv_ref[k, pl.ds(first, MERGE_CHUNK_ROWS), :]
        part = pltpu.bitcast(words, jnp.bfloat16)
        part = jnp.where(jnp.isnan(part), jnp.bfloat16(0), part).astype(jnp.float32)
        acc_hi = acc_hi + part * column(k)
        acc_lo = acc_lo + part * column(senders + k)
        return acc_hi, acc_lo

    acc = (
        jnp.zeros((MERGE_CHUNK_ROWS, vdim), jnp.float32),
        jnp.zeros((MERGE_CHUNK_ROWS, vdim), jnp.float32),
    )
    if rolled:
        acc_hi, acc_lo = lax.fori_loop(0, n, add_sender, acc)
    else:
        for k in range(n):
            acc = add_sender(k, acc)
        acc_hi, acc_lo = acc
    row = lax.broadcasted_iota(jnp.int32, acc_hi.shape, 0) + start
    merged = jnp.where(row < live, acc_hi + acc_lo, 0.0)
    return merged.astype(jnp.bfloat16)


def small_all_to_all(
    x,
    *,
    num_cores,
    axis,
    row0=0,
    rank=None,
    lane_axis=None,
    lane=None,
    collective_id=COLLECTIVE_ID,
    hbm_operand=False,
    merge_live=None,
    merge_row=None,
    live_rows=None,
    live_row0=0,
):
    """``lax.all_to_all(x, axis, 0, 0, tiled=True)`` of ``[n * block, V]``
    for rows ``[row0, block)`` of every block, inside a shard_map over a mesh
    whose ``axis`` has ``num_cores`` cores.

    Args:
      x: [n * block, V], this core's blocks, destination-major (any dtype; the
        bytes are copied). ``block`` and ``row0`` multiples of 8.
      row0: the first row of each block to move (static).
      rank: int32 [1], this core's index on ``axis`` as data, or None to read
        it from the axis.
      lane_axis: the mesh's other axis, if it has one (at most one).
      lane: int32 [1], this core's index on ``lane_axis`` as data, or None to
        read it from the axis. Ignored without ``lane_axis``.
      hbm_operand: constrain ``x`` to HBM (``pltpu.with_memory_space_constraint``).
        With the operand in ``pl.ANY``, XLA's memory-space assignment stages the
        send blocks into VMEM with a copy-start/-done right before the call,
        exposed because the producer runs just before; the kernel only DMAs
        them, so HBM costs nothing. The in_spec's memory space alone does not
        stop the staging (it only types the kernel ref).
      merge_live: int32 [1], the number of live partial rows, or None. With
        it the call runs merge mode (see the module docstring) on 16-bit
        DCP send blocks and returns the bf16 merge ``[tl, V]``,
        ``tl = block - merge_row``.
      merge_row: merge mode's first partial row of a block (static); the row
        before it is the lse word row.
      live_rows: int32 [1], the same on every core, or None. With it, rows of
        every block from ``live_row0 + live_rows`` on, rounded up to a whole
        8-row tile, are not moved (see the module docstring); the call is
        named ``..._live``. Composes with merge mode.
      live_row0: the block row ``live_rows`` counts from (static).
    """
    if x.ndim != 2:
        raise ValueError(f"small_all_to_all takes [n * block, V], got {x.shape}")
    n = num_cores
    total, vdim = x.shape
    if total % n:
        raise ValueError(f"{total} rows do not split into {n} blocks")
    block = total // n
    if block % 8 or row0 % 8 or not 0 <= row0 < block:
        raise ValueError(
            f"block {block} / row0 {row0}: need multiples of 8, row0 < block"
        )
    rows = block - row0
    data_rank = rank is not None
    data_lane = lane_axis is not None and lane is not None
    merge = merge_live is not None
    sized = live_rows is not None
    if merge:
        tl = block - (merge_row or 0)
        if not (
            x.dtype.itemsize == 2
            and row0 < merge_row < block
            and (merge_row - row0) % 8 == 0
            and tl % MERGE_CHUNK_ROWS == 0
            and tl <= LANES
            and 2 * tl <= vdim
            and 2 * merge_sender_rows(n) <= LANES
        ):
            raise ValueError(
                f"merge mode: {x.dtype} block {block} / row0 {row0} / merge_row "
                f"{merge_row} / {n} cores / V {vdim} has no in-kernel merge layout"
            )
        chunks = tl // MERGE_CHUNK_ROWS

    def kernel(*refs):
        refs = list(refs)
        me = refs.pop(0)[0] if data_rank else lax.axis_index(axis)
        if lane_axis is None:
            my_lane = None
        elif data_lane:
            my_lane = refs.pop(0)[0]
        else:
            my_lane = lax.axis_index(lane_axis)
        if merge:
            live = jnp.clip(refs.pop(0)[0], 0, tl)
        if sized:
            end = live_row0 + refs.pop(0)[0]
            sizes = live_sizes(jnp.clip((end - row0 + 7) & -8, 8, rows), rows)
        if merge:
            x_ref, o_ref, recv_ref, w_ref, out_ref, *sems = refs
            send_sem, recv_sem, local_sem, out_sem = sems
        else:
            x_ref, o_ref, send_sem, recv_sem, local_sem = refs

        def peer(j):
            ids = {axis: jnp.int32(j)}
            if lane_axis is not None:
                ids[lane_axis] = my_lane
            return ids

        def chunk_out(c):
            part = pl.ds(c * MERGE_CHUNK_ROWS, MERGE_CHUNK_ROWS)
            return pltpu.make_async_copy(out_ref.at[part], o_ref.at[part], out_sem)

        def slot(size):
            """Rows ``[row0, row0 + size)`` of slot ``me``."""
            if merge:
                if size == rows:
                    return recv_ref.at[me]
                return recv_ref.at[me, pl.ds(0, size)]
            return o_ref.at[pl.ds(pl.multiple_of(me * block + row0, 8), size)]

        def send(size):
            """``size`` rows of block j from row ``row0`` into slot ``me`` of
            core j; returns the copy of this core's own block."""
            mine = slot(size)
            for j in range(n):

                @pl.when(jnp.int32(j) != me)
                def _(j=j):
                    pltpu.make_async_remote_copy(
                        src_ref=x_ref.at[pl.ds(j * block + row0, size)],
                        dst_ref=mine,
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=peer(j),
                        device_id_type=MESH,
                    ).start()

            own = x_ref.at[pl.ds(pl.multiple_of(me * block + row0, 8), size)]
            return pltpu.make_async_copy(own, mine, local_sem)

        def span(copies, size):
            """A ref of ``copies`` copies of ``size`` rows, for byte counts."""
            if merge:
                if size == rows:
                    return recv_ref.at[pl.ds(0, copies)]
                return recv_ref.at[pl.ds(0, copies), pl.ds(0, size)]
            return o_ref.at[pl.ds(0, copies * size)]

        def wait(wait_sem, copies):
            """Block until ``wait_sem`` has counted ``copies`` copies of the
            rows sent."""
            if not sized:
                wait_bytes_of(span(copies, rows), wait_sem)
                return
            for size, taken in sizes:

                @pl.when(taken)
                def _(size=size):
                    wait_bytes_of(span(copies, size), wait_sem)

        sem = pltpu.get_barrier_semaphore()
        for j in range(n):
            pl.semaphore_signal(sem, 1, device_id=peer(j), device_id_type=MESH)
        pl.semaphore_wait(sem, n)

        if sized:
            for size, taken in sizes:

                @pl.when(taken)
                def _(size=size):
                    send(size).start()

        else:
            local = send(rows)
            local.start()
        if merge:
            for c in range(chunks):

                @pl.when(c * MERGE_CHUNK_ROWS >= live)
                def _(c=c):
                    out_ref[pl.ds(c * MERGE_CHUNK_ROWS, MERGE_CHUNK_ROWS), :] = (
                        jnp.zeros((MERGE_CHUNK_ROWS, vdim), jnp.bfloat16)
                    )
                    chunk_out(c).start()

        wait(recv_sem, n - 1)
        if sized:
            wait(local_sem, 1)
        else:
            local.wait()
        if merge:
            merge_weights(
                recv_ref, w_ref, num_cores=n, word_row=merge_row - 1 - row0, tl=tl
            )
            for c in range(chunks):

                @pl.when(c * MERGE_CHUNK_ROWS < live)
                def _(c=c):
                    out_ref[pl.ds(c * MERGE_CHUNK_ROWS, MERGE_CHUNK_ROWS), :] = (
                        merge_chunk(
                            recv_ref,
                            w_ref,
                            c,
                            live,
                            num_cores=n,
                            part_row=merge_row - row0,
                        )
                    )
                    chunk_out(c).start()

        wait(send_sem, n - 1)
        if merge:
            for c in range(chunks):
                chunk_out(c).wait()

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    any_ = pl.BlockSpec(memory_space=pl.ANY)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scalars = []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
    if data_lane:
        scalars.append(jnp.reshape(lane, (-1,))[:1].astype(jnp.int32))
    if merge:
        scalars.append(jnp.reshape(merge_live, (-1,))[:1].astype(jnp.int32))
    if sized:
        scalars.append(jnp.reshape(live_rows, (-1,))[:1].astype(jnp.int32))
    if hbm_operand:
        x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    scratch = [pltpu.SemaphoreType.DMA] * 3
    name = f"small_all_to_all_n{n}_b{block}_r{row0}" + ("_live" if sized else "")
    if merge:
        out_shape = jax.ShapeDtypeStruct((tl, vdim), jnp.bfloat16)
        scratch = [
            pltpu.VMEM((n, rows, vdim), x.dtype),
            pltpu.VMEM((LANES, LANES), jnp.float32),
            pltpu.VMEM((tl, vdim), jnp.bfloat16),
            *[pltpu.SemaphoreType.DMA] * 4,
        ]
        name += "_merge"
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[smem] * len(scalars) + [hbm if hbm_operand else any_],
        out_specs=hbm,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=name,
    )(*scalars, x)
