# Copyright 2026 The vLLM-TorchTPU Authors
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

"""Column-window FP8 decode linear fused with the all-gather of the windows.

Under DCP query replication every rank needs every head's q, so each rank streams
the whole replicated q_b weight ([2048, 16384] fp8, 33.5 MB) at decode. Here each
rank streams only its own column window (its heads: 1/n of the bytes), computes it
with fp8_skinny's arithmetic, and pushes it into the same window of every other
rank's output -- one round of remote copies, whose barrier is signalled at entry and
hides under the weight stream:

    entry   start the window's weight tiles (fp8_skinny's 'n' ring); signal the
            barrier on every rank, self included; quantize x
    tile j  wait W_j; out[:, window tile j] = bf16(x_q . W_j * x_scale * w_scale)
    push    wait the barrier; copy out[:, window] into out[:, window] of every
            other rank
    wait    the n - 1 inbound windows (one byte-counted semaphore); drain the sends

Each output column is fp8_skinny's expression for that column (the same x_q, scale
and f32 dot over the whole K), so the gathered output is bitwise the replicated
kernel's.

Safety, as `small_collectives.common.barrier`: a rank writes into a peer only after
the barrier of this call, which the peer signals at entry, so the peer's output is
live; a peer leaves only after every inbound window landed and its own sends
drained, so no write of call k reaches a peer in call k + 1.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

MESH = pl.DeviceIdType.MESH
# Barrier semaphore group, distinct from the fused MoE kernels (0), the
# all-reduces (5), the reduce-scatter (7), the all-gather (9), the all-to-all (11)
# and the sparse MLA kernel's exchange (13).
COLLECTIVE_ID = 15
# Row slices a weight tile is copied in, all in flight at once: the window kernel's
# single 2 MB strided copy took 3.6 us in situ (j-6482bc6e), where the replicated
# kernel's four tiles in flight stream 2.5 TB/s. The tile's dot still runs once
# over the whole K, so the arithmetic is unchanged.
_TILE_COPY_PARTS = 4


def supports(m: int, k: int, n: int, num_cores: int) -> bool:
    """Whether `fp8_skinny.window_supports` serves the rank windows of a `(k, n)`
    weight split over `num_cores` ranks, for `m <= fp8_skinny.MAX_M` rows."""
    return (
        m <= fp8_skinny.MAX_M
        and num_cores >= 2
        and n % num_cores == 0
        and fp8_skinny.window_supports(k, n, n // num_cores)
    )


def _kernel(
    rank_ref,
    *refs,
    n,
    axis,
    lane_axis,
    t,
    n_tiles,
    nbuf,
    q_dtype,
):
    lane_ref, *refs = refs if lane_axis is not None else (None, *refs)
    x_ref, ws_ref, w_hbm, o_ref, xq_ref, xs_ref, wbuf, wsem, send_sem, recv_sem = refs
    me = rank_ref[0]
    width = n_tiles * t
    col0 = pl.multiple_of(me * width, width)
    bsem = pltpu.get_barrier_semaphore()

    def peer(j):
        """Rank j on `axis`, in this core's lane of the mesh's other axis."""
        ids = {axis: jnp.int32(j)}
        if lane_ref is not None:
            ids[lane_axis] = lane_ref[0]
        return ids

    k = w_hbm.shape[0]
    parts = _TILE_COPY_PARTS if k % (_TILE_COPY_PARTS * 32) == 0 else 1
    rows = k // parts

    def tile(j):
        return w_hbm.at[:, pl.ds(col0 + j * t, t)], wbuf.at[j % nbuf], wsem.at[j % nbuf]

    def start(j):
        src, dst, sem = tile(j)
        for p in range(parts):
            part = pl.ds(p * rows, rows)
            pltpu.make_async_copy(src.at[part], dst.at[part], sem).start()

    def wait(j):
        # The parts signal one semaphore; wait for the whole tile's bytes.
        pltpu.make_async_copy(*tile(j)).wait()

    for j in range(min(nbuf, n_tiles)):
        start(j)
    for j in range(n):
        pl.semaphore_signal(bsem, 1, device_id=peer(j), device_id_type=MESH)

    # Activation quant: fp8_skinny._kernel's formula (bit-identical x_q and scale).
    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(
        jnp.float32
    ) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        wait(j)
        y = jax.lax.dot_general(
            xq_ref[...], wbuf[j % nbuf], dn, preferred_element_type=jnp.float32
        )
        cols = pl.ds(col0 + j * t, t)
        o_ref[:, cols] = (y * xs_ref[...] * ws_ref[:, cols]).astype(o_ref.dtype)
        if j + nbuf < n_tiles:
            start(j + nbuf)

    mine = o_ref.at[:, pl.ds(col0, width)]
    pl.semaphore_wait(bsem, n)
    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            pltpu.make_async_remote_copy(
                src_ref=mine,
                dst_ref=mine,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=peer(j),
                device_id_type=MESH,
            ).start()

    peers = o_ref.at[:, pl.ds(0, (n - 1) * width)]  # the byte count of n - 1 windows
    pltpu.make_async_copy(peers, peers, recv_sem).wait()
    pltpu.make_async_copy(peers, peers, send_sem).wait()


@functools.cache
def _build(m, k, n_out, w_dtype, x_dtype, num_cores, axis, lane_axis):
    _, t, nbuf = fp8_skinny.TUNED[(k, n_out)]
    width = n_out // num_cores
    n_tiles = width // t
    nbuf = max(1, min(nbuf, n_tiles))
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),  # x_q (fp8 values held as bf16)
        pltpu.VMEM((m, 1), jnp.float32),  # x scale
        pltpu.VMEM((nbuf, k, t), w_dtype),  # weight ring
        pltpu.SemaphoreType.DMA((nbuf,)),
        pltpu.SemaphoreType.DMA,  # window sends
        pltpu.SemaphoreType.DMA,  # window receives
    ]
    kern = functools.partial(
        _kernel,
        n=num_cores,
        axis=axis,
        lane_axis=lane_axis,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1 if lane_axis is None else 2,
        in_specs=[vmem, vmem, pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=vmem,
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n_out), x_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"fp8_window_all_gather_m{m}_k{k}_n{n_out}_w{width}",
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained(call, scalars, x, w_s2, w_q):
    # The weight stays in HBM (the kernel streams only its window), as
    # fp8_skinny's hbm_pin.
    return call(*scalars, x, w_s2, pltpu.with_memory_space_constraint(w_q, pltpu.HBM))


def fp8_window_all_gather(
    x, w_q, w_s, rank, *, num_cores, axis, lane_axis=None, lane=None
):
    """`fp8_skinny(x, w_q, w_s)` on every rank of mesh axis `axis`, each rank
    computing only its column window `rank` and all-gathering the rest, inside a
    shard_map over that axis.

    x bf16 [m <= 16, k] (the same on every rank); w_q fp8_e4m3fn [k, n] (K-major,
    replicated); w_s f32 [n]; rank int32 [1] (this core's index on the axis, as
    data) -> bf16 [m, n]. A mesh with a second axis names it `lane_axis` and
    passes this core's index on it as `lane` (int32 [1]): peers are this lane's
    ranks.
    """
    m, k = x.shape
    n_out = w_q.shape[1]
    if not supports(m, k, n_out, num_cores):
        raise ValueError(
            f"[{m}, {k}] x [{k}, {n_out}] windows over {num_cores} ranks: unsupported"
        )
    if (lane_axis is None) != (lane is None):
        raise ValueError("lane_axis and lane go together")
    call = _build(m, k, n_out, w_q.dtype, x.dtype, num_cores, axis, lane_axis)
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    scalars = (rank.astype(jnp.int32),)
    if lane is not None:
        scalars += (lane.astype(jnp.int32),)
    return _call_constrained(call, scalars, x, w_s2, w_q)
