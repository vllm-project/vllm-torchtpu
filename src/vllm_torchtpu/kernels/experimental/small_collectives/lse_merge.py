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
"""DCP merge of attention partials over their log-sum-exps, as one Pallas kernel.

Under decode context parallelism (DCP) every core of the group attends over its
own share of the context, for all ``H`` heads, and holds a partial output
``o_j [T, H, D]`` with its log-sum-exp ``lse_j [T, H]``. Core ``r`` owns heads
``[r * hl, (r + 1) * hl)``, ``hl = H / n``, and needs

    out[h, t] = sum_j o_j[t, h] * exp(lse_j[t, h] - log(sum_k exp(lse_k[t, h])))

The two-collective merge (``dcp_comm.cp_lse_ag_out_rs``) all-gathers the
log-sum-exps, rescales the partial on the sender and reduce-scatters it: two
collectives in series, each with its own fixed cost. This kernel needs one
round of remote DMAs. Every core sends core ``r`` the unscaled partial of
core ``r``'s heads and its whole log-sum-exp, and core ``r`` computes the
weights and the weighted sum itself, in float32, rounding once to the
output dtype.

The kernel, per core, in order:

1. Starts the copies of its partial into VMEM, ``chunks`` token chunks of
   ``tc = T / chunks`` tokens, and signals the barrier.
2. Relayouts each chunk head-major (``[tc, H, D] -> [H, tc, D]``), so core
   ``r``'s heads of a chunk are one contiguous ``[hl, tc, D]`` block, and pushes
   that block into slot (chunk, me) of core ``r``'s receive buffer. The
   relayout of chunk c + 1 overlaps the sends of chunk c.
3. Pushes its log-sum-exp ``[T, H]`` into slot ``me`` of every core.
4. Computes the weights of every sender, token and head from the gathered
   log-sum-exps, then selects the columns of its own ``hl`` heads with one
   small matmul against a one-hot matrix (the head index is data, the rank).
5. Merges each chunk as soon as the ``n`` blocks of that chunk have arrived,
   so the merge of chunk c overlaps the arrival of chunk c + 1.

The weights are those of ``cp_lse_ag_out_rs``: ``exp(lse_j - lse_max) / sum``
with non-finite log-sum-exps weighted 0, and a sender whose weight is 0
contributes 0 even if its partial is NaN. The column selection splits each
float32 weight into two bfloat16 terms (``hi + lo``), so the selected weight
differs from the float32 weight by at most about 2**-16 of it, far below the
bfloat16 rounding of the result.

One barrier per call is enough, by the argument in ``all_gather``: a core
writes into a peer's receive buffer only after the barrier of the same call,
which the peer passes only inside the kernel that owns that buffer, and a core
leaves a call only after all of its blocks have arrived and all of its sends
are done.

Example (GLM-5.2 decode, DCP 16, a 64-token bucket): ``o_j`` bf16
``[64, 64, 512]`` (4 MiB) and ``lse_j`` f32 ``[64, 64]`` per core; with
``chunks = 4`` each core sends 15 x 4 blocks of bf16 ``[4, 16, 512]`` (64 KB)
and 15 log-sum-exps of 16 KB, and returns bf16 ``[4, 64, 512]``.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    peer,
    wait_bytes_of,
)

# Barrier semaphore group, distinct from every other collective kernel in the
# tree (0, 5, 7, 9, 11, 13, 15, 17, 19).
COLLECTIVE_ID = 21
# Most token chunks a call is split into (one DMA semaphore per chunk each for
# the input copies, the arrivals and the own block).
MAX_CHUNKS = 4
# Test-only faults (``small_lse_merge(_fault=...)``), one per fault class the
# accuracy test guards: a sender left out of the sum, a chunk merged from the
# wrong chunk's blocks, the weights of the neighbouring head, weights not
# divided by their sum.
FAULTS = ("drop_sender", "stale_chunk", "wrong_head", "unnormalized")


def sublane_tile(dtype) -> int:
    """Rows of one native VMEM tile of ``dtype``: 8 sublanes of 32-bit words,
    so 8 rows of float32 and 16 rows of bfloat16."""
    return 8 * 4 // jnp.dtype(dtype).itemsize


def default_chunks(tokens: int, dtype) -> int:
    """The most chunks, up to MAX_CHUNKS, that split ``tokens`` into whole
    sublane tiles of ``dtype``; 0 when ``tokens`` is not a whole tile."""
    tile = sublane_tile(dtype)
    for chunks in range(MAX_CHUNKS, 0, -1):
        if tokens % (chunks * tile) == 0:
            return chunks
    return 0


def check_shapes(output_shape, output_dtype, lse_shape, lse_dtype, num_cores):
    """Raise ValueError when the kernel has no layout for these operands."""
    if len(output_shape) != 3:
        raise ValueError(f"output must be [T, H, D], got {output_shape}")
    tokens, heads, dim = output_shape
    if jnp.dtype(output_dtype).itemsize != 2:
        raise ValueError(f"output must be a 2-byte float, got {output_dtype}")
    if tuple(lse_shape) != (tokens, heads) or jnp.dtype(lse_dtype) != jnp.float32:
        raise ValueError(f"lse must be f32 [{tokens}, {heads}], got {lse_dtype}{lse_shape}")
    if heads % num_cores:
        raise ValueError(f"{heads} heads do not split over {num_cores} cores")
    if dim % LANES:
        raise ValueError(f"head dim {dim} is not a multiple of {LANES} lanes")
    if not default_chunks(tokens, output_dtype):
        raise ValueError(f"{tokens} tokens are not a whole sublane tile")


def reference(outputs, lses, core):
    """float64 oracle. ``outputs`` [n, T, H, D], ``lses`` [n, T, H] (NumPy or
    JAX, any float dtype) -> core ``core``'s merged heads, float64
    [hl, T, D]."""
    import numpy as np

    o = np.asarray(outputs, dtype=np.float64)
    lse = np.asarray(lses, dtype=np.float64)
    n, _, heads, _ = o.shape
    hl = heads // n
    with np.errstate(invalid="ignore", over="ignore"):
        finite = np.isfinite(lse)
        shift = np.max(np.where(finite, lse, -np.inf), axis=0)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        e = np.where(finite, np.exp(lse - shift), 0.0)
        s = e.sum(axis=0)
        w = e / np.where(s > 0, s, 1.0)
        terms = np.where(w[..., None] > 0, o * w[..., None], 0.0)
    merged = terms.sum(axis=0)  # [T, H, D]
    return merged[:, core * hl : (core + 1) * hl].transpose(1, 0, 2)


def small_lse_merge(
    output,
    lse,
    *,
    num_cores,
    axis,
    rank=None,
    chunks=None,
    collective_id=COLLECTIVE_ID,
    out_dtype=None,
    interpret=None,
    _fault=None,
):
    """Merge of the DCP partials of ``num_cores`` cores, inside a shard_map
    over mesh axis ``axis``: core r returns its heads, head-major.

    Args:
      output: [T, H, D] bf16 (any 2-byte float), this core's partial for all
        heads. T a multiple of 16 rows, H a multiple of ``num_cores``, D of 128.
      lse: [T, H] f32, its log-sum-exp; -inf (or NaN) where the core saw no
        keys.
      num_cores: cores on ``axis`` (the DCP group size).
      axis: the mesh axis the cores sit on, in DCP rank order.
      rank: int32 [1], this core's index on ``axis`` as data (the torch
        bridge's form, see ``common``), or None to read it from the axis.
      chunks: token chunks the exchange is split into; None picks the most
        (up to MAX_CHUNKS) that keep each chunk a whole sublane tile.
      out_dtype: dtype of the result; None keeps ``output.dtype``.
      interpret: passed to ``pl.pallas_call`` (``pltpu.InterpretParams()`` runs
        the kernel on CPU devices for tests).
      _fault: tests only. One of FAULTS: a deliberately wrong kernel that the
        accuracy test must reject (its positive control).

    Returns:
      [H / num_cores, T, D] in ``out_dtype``: heads [r * hl, (r + 1) * hl) of
      the merge on core r, float32 accumulation rounded once.
    """
    if _fault is not None and _fault not in FAULTS:
        raise ValueError(f"unknown fault {_fault}")
    n = num_cores
    check_shapes(output.shape, output.dtype, lse.shape, lse.dtype, n)
    tokens, heads, dim = output.shape
    hl = heads // n
    chunks = chunks or default_chunks(tokens, output.dtype)
    if tokens % (chunks * sublane_tile(output.dtype)) or chunks > MAX_CHUNKS:
        raise ValueError(f"{chunks} chunks do not split {tokens} tokens into tiles")
    tc = tokens // chunks
    out_dtype = jnp.dtype(out_dtype or output.dtype)
    data_rank = rank is not None

    def kernel(*refs):
        refs = list(refs)
        me = refs.pop(0)[0] if data_rank else lax.axis_index(axis)
        (
            x_hbm,
            lse_ref,
            o_ref,
            xbuf,
            send,
            recv,
            lse_recv,
            cols,
            in_sems,
            send_sem,
            recv_sems,
            local_sems,
            lse_send_sem,
            lse_recv_sem,
            lse_local_sem,
        ) = refs

        def to_peer(j):
            return peer(j, axis, data_rank)

        def in_copy(c):
            return pltpu.make_async_copy(
                x_hbm.at[pl.ds(c * tc, tc)], xbuf.at[c], in_sems.at[c]
            )

        def relayout(c):
            in_copy(c).wait()
            send[c] = pltpu.einshape("thd->htd", xbuf[c])

        def send_chunk(c):
            for j in range(n):

                @pl.when(jnp.int32(j) != me)
                def _(j=j):
                    pltpu.make_async_remote_copy(
                        src_ref=send.at[c, pl.ds(j * hl, hl)],
                        dst_ref=recv.at[c, me],
                        send_sem=send_sem,
                        recv_sem=recv_sems.at[c],
                        device_id=to_peer(j),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

            own = send.at[c, pl.ds(pl.multiple_of(me * hl, hl), hl)]
            pltpu.make_async_copy(own, recv.at[c, me], local_sems.at[c]).start()

        for c in range(chunks):
            in_copy(c).start()
        barrier = pltpu.get_barrier_semaphore()
        for j in range(n):
            pl.semaphore_signal(
                barrier, 1, device_id=to_peer(j), device_id_type=pl.DeviceIdType.MESH
            )
        # Chunk 0's relayout is local work: it overlaps the barrier's latency.
        relayout(0)
        pl.semaphore_wait(barrier, n)

        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pltpu.make_async_remote_copy(
                    src_ref=lse_ref,
                    dst_ref=lse_recv.at[me],
                    send_sem=lse_send_sem,
                    recv_sem=lse_recv_sem,
                    device_id=to_peer(j),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        lse_own = pltpu.make_async_copy(lse_ref, lse_recv.at[me], lse_local_sem)
        lse_own.start()
        send_chunk(0)
        for c in range(1, chunks):
            relayout(c)
            send_chunk(c)

        # Weights of every (sender, token, head), then the columns of this
        # core's heads: cols[j * T + t, k] = w_j[t, me * hl + k] for k < hl.
        wait_bytes_of(lse_recv.at[pl.ds(0, n - 1)], lse_recv_sem)
        lse_own.wait()
        lse_all = lse_recv[...]
        finite = jnp.isfinite(lse_all)
        shift = jnp.max(jnp.where(finite, lse_all, -jnp.inf), axis=0, keepdims=True)
        shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
        e = jnp.where(finite, jnp.exp(lse_all - shift), 0.0)
        s = jnp.sum(e, axis=0, keepdims=True)
        if _fault != "unnormalized":
            e = e / jnp.where(s > 0, s, 1.0)
        w = e.reshape(n * tokens, heads)
        head = lax.broadcasted_iota(jnp.int32, (heads, LANES), 0)
        lane = lax.broadcasted_iota(jnp.int32, (heads, LANES), 1)
        mine = me * hl + lane + (1 if _fault == "wrong_head" else 0)
        onehot = jnp.where((lane < hl) & (head == mine), 1.0, 0.0).astype(
            jnp.bfloat16
        )
        w_hi = w.astype(jnp.bfloat16)
        w_lo = (w - w_hi.astype(jnp.float32)).astype(jnp.bfloat16)
        cols[...] = jnp.dot(w_hi, onehot, preferred_element_type=jnp.float32) + jnp.dot(
            w_lo, onehot, preferred_element_type=jnp.float32
        )

        for c in range(chunks):
            wait_bytes_of(recv.at[c, pl.ds(0, n - 1)], recv_sems.at[c])
            wait_bytes_of(recv.at[c, 0], local_sems.at[c])
            for k in range(hl):
                acc = jnp.zeros((tc, dim), jnp.float32)
                senders = n - 1 if _fault == "drop_sender" else n
                src_chunk = 0 if _fault == "stale_chunk" else c
                for j in range(senders):
                    col = cols[pl.ds(j * tokens + c * tc, tc), :][:, k : k + 1]
                    part = recv[src_chunk, j, k].astype(jnp.float32)
                    acc = acc + jnp.where(col > 0, part * col, 0.0)
                o_ref[k, pl.ds(c * tc, tc), :] = acc.astype(o_ref.dtype)

        for c in range(chunks):
            wait_bytes_of(recv.at[c, pl.ds(0, n - 1)], send_sem)
        wait_bytes_of(lse_recv.at[pl.ds(0, n - 1)], lse_send_sem)

    scalars = []
    in_specs = []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    in_specs += [pl.BlockSpec(memory_space=pl.ANY), vmem]
    scratch = [
        pltpu.VMEM((chunks, tc, heads, dim), output.dtype),  # xbuf
        pltpu.VMEM((chunks, heads, tc, dim), output.dtype),  # send, head-major
        pltpu.VMEM((chunks, n, hl, tc, dim), output.dtype),  # recv
        pltpu.VMEM((n, tokens, heads), jnp.float32),  # lse_recv
        pltpu.VMEM((n * tokens, LANES), jnp.float32),  # cols
        pltpu.SemaphoreType.DMA((chunks,)),  # in_sems
        pltpu.SemaphoreType.DMA,  # send_sem
        pltpu.SemaphoreType.DMA((chunks,)),  # recv_sems
        pltpu.SemaphoreType.DMA((chunks,)),  # local_sems
        pltpu.SemaphoreType.DMA,  # lse_send_sem
        pltpu.SemaphoreType.DMA,  # lse_recv_sem
        pltpu.SemaphoreType.DMA,  # lse_local_sem
    ]
    params = {"collective_id": collective_id}
    if interpret is None:
        # Three [T, H, D] buffers plus the weights and the temporaries of one
        # chunk's relayout: 13.5 MiB at T=64, H=64, D=512, n=16.
        params["vmem_limit_bytes"] = pltpu.get_tpu_info().vmem_capacity_bytes * 3 // 4
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((hl, tokens, dim), out_dtype),
        in_specs=in_specs,
        out_specs=vmem,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(**params),
        interpret=interpret,
        name=f"small_lse_merge_n{n}_t{tokens}_c{chunks}",
    )(*scalars, output, lse)
