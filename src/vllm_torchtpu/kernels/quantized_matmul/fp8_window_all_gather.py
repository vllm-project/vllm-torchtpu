# SPDX-License-Identifier: Apache-2.0
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

With `w_uk` (MLA query absorption, TPU_MLA_LOCAL_Q_FUSED_ABSORB) each rank also
multiplies its heads' q_nope by their W_UK, after pushing the q window and while it
travels, and all-gathers that window of the absorbed q the same way: every rank
absorbs 1/n of the heads, and the absorbed q comes out token-major, the layout the
MLA kernels read.

Safety, as `small_collectives.common.barrier`: a rank writes into a peer only after
the barrier of this call, which the peer signals at entry, so the peer's outputs are
live; a peer leaves only after every inbound window landed and its own sends
drained, so no write of call k reaches a peer in call k + 1.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

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
# Row slices this kernel copies a weight tile in, all in flight at once: its single
# 2 MB strided copy of the window took 3.6 us in situ (j-6482bc6e), where the
# replicated kernel's four tiles in flight stream 2.5 TB/s (fp8_skinny's own window
# kernel still copies the window whole). The tile's dot still runs once over the
# whole K, so the arithmetic is unchanged.
_TILE_COPY_PARTS = 4
# Rows of one VMEM tile of an 8-bit weight: each row slice starts on a whole tile.
_FP8_TILE_ROWS = 32


class Absorb(NamedTuple):
    """Static shape of the in-kernel query absorption: this rank's `heads` heads
    (each `width // heads` columns of the window, q_nope first), W_UK [heads,
    nope, rank]."""

    heads: int
    nope: int
    rank: int


def supports(m: int, k: int, n: int, num_cores: int) -> bool:
    """Whether `fp8_skinny.window_supports` serves the rank windows of a `(k, n)`
    weight split over `num_cores` ranks, for `m <= fp8_skinny.MAX_M` rows."""
    return (
        m <= fp8_skinny.MAX_M
        and num_cores >= 2
        and n % num_cores == 0
        and fp8_skinny.window_supports(k, n, n // num_cores)
    )


def absorb_supports(n: int, num_cores: int, absorb: Absorb) -> bool:
    """Whether the window of a `(k, n)` weight over `num_cores` ranks splits into
    `absorb.heads` lane-aligned heads whose first `absorb.nope` columns (whole bf16
    tiles of W_UK rows) meet a lane-aligned W_UK."""
    width = n // num_cores
    head = width // absorb.heads
    return (
        width % absorb.heads == 0
        and head % 128 == 0
        and 0 < absorb.nope <= head
        and absorb.nope % 16 == 0
        and absorb.rank % 128 == 0
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
    absorb,
):
    lane_ref, *refs = refs if lane_axis is not None else (None, *refs)
    n_in, n_out = (3, 1) if absorb is None else (4, 2)
    x_ref, ws_ref, w_hbm, *uk_hbm = refs[:n_in]
    o_ref, *ql_ref = refs[n_in : n_in + n_out]
    xq_ref, xs_ref, wbuf, wsem, send_sem, recv_sem, *uk_scratch = refs[n_in + n_out :]
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

    def push(ref, c0, w, send, recv):
        """This rank's columns [c0, c0 + w) of `ref` into the same columns of every
        other rank's `ref`."""
        mine = ref.at[:, pl.ds(c0, w)]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j):
                pltpu.make_async_remote_copy(
                    src_ref=mine,
                    dst_ref=mine,
                    send_sem=send,
                    recv_sem=recv,
                    device_id=peer(j),
                    device_id_type=MESH,
                ).start()

    def drain(ref, w, send, recv):
        """Wait for the n - 1 inbound w-column windows of `ref`, then for the sends."""
        peers = ref.at[:, pl.ds(0, (n - 1) * w)]  # the byte count of n - 1 windows
        pltpu.make_async_copy(peers, peers, recv).wait()
        pltpu.make_async_copy(peers, peers, send).wait()

    k = w_hbm.shape[0]
    parts = _TILE_COPY_PARTS if k % (_TILE_COPY_PARTS * _FP8_TILE_ROWS) == 0 else 1
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
    if absorb is not None:
        ((uk_hbm,), (ql_ref,)) = uk_hbm, ql_ref
        uk_buf, uk_sem, ql_send_sem, ql_recv_sem = uk_scratch
        # This rank's heads of W_UK fill the first `nope` rows of each head's slot;
        # the rows past them are zero, so a head's (finite) rope columns add exact
        # zeros.
        head = width // absorb.heads
        uk_copy = pltpu.make_async_copy(
            uk_hbm, uk_buf.at[:, pl.ds(0, absorb.nope), :], uk_sem
        )
        uk_copy.start()
        if absorb.nope < head:
            uk_buf[:, pl.ds(absorb.nope, head - absorb.nope), :] = jnp.zeros(
                (absorb.heads, head - absorb.nope, absorb.rank), uk_buf.dtype
            )
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

    pl.semaphore_wait(bsem, n)
    push(o_ref, col0, width, send_sem, recv_sem)
    if absorb is not None:
        # ql[:, head] = bf16(q_nope[head] @ W_UK[head]) in f32, under the q pushes.
        q_width = absorb.heads * absorb.rank
        ql0 = pl.multiple_of(me * q_width, q_width)
        uk_copy.wait()
        for h in range(absorb.heads):
            q_head = o_ref[:, pl.ds(pl.multiple_of(col0 + h * head, head), head)]
            y = jax.lax.dot_general(
                q_head, uk_buf[h], dn, preferred_element_type=jnp.float32
            )
            ql_ref[:, pl.ds(ql0 + h * absorb.rank, absorb.rank)] = y.astype(
                ql_ref.dtype
            )
        push(ql_ref, ql0, q_width, ql_send_sem, ql_recv_sem)
    drain(o_ref, width, send_sem, recv_sem)
    if absorb is not None:
        drain(ql_ref, q_width, ql_send_sem, ql_recv_sem)


@functools.cache
def _build(m, k, n_out, w_dtype, x_dtype, num_cores, axis, lane_axis, absorb):
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
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [vmem, vmem, hbm]
    out_specs = vmem
    out_shape = jax.ShapeDtypeStruct((m, n_out), x_dtype)
    name = f"fp8_window_all_gather_m{m}_k{k}_n{n_out}_w{width}"
    if absorb is not None:
        head = width // absorb.heads
        scratch += [
            pltpu.VMEM((absorb.heads, head, absorb.rank), jnp.bfloat16),  # W_UK
            pltpu.SemaphoreType.DMA,  # its copy
            pltpu.SemaphoreType.DMA,  # absorbed window sends
            pltpu.SemaphoreType.DMA,  # absorbed window receives
        ]
        in_specs.append(hbm)
        out_specs = (vmem, vmem)
        ql_cols = num_cores * absorb.heads * absorb.rank
        out_shape = (out_shape, jax.ShapeDtypeStruct((m, ql_cols), x_dtype))
        name += f"_uk{absorb.heads}x{absorb.nope}x{absorb.rank}"
    kern = functools.partial(
        _kernel,
        n=num_cores,
        axis=axis,
        lane_axis=lane_axis,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        absorb=absorb,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1 if lane_axis is None else 2,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=name,
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained(call, scalars, x, w_s2, w_q, *w_uk):
    # The weights stay in HBM (the kernel streams only its window), as
    # fp8_skinny's hbm_pin.
    hbm = [pltpu.with_memory_space_constraint(w, pltpu.HBM) for w in (w_q, *w_uk)]
    return call(*scalars, x, w_s2, *hbm)


def fp8_window_all_gather(
    x, w_q, w_s, rank, *, num_cores, axis, lane_axis=None, lane=None, w_uk=None
):
    """`fp8_skinny(x, w_q, w_s)` on every rank of mesh axis `axis`, each rank
    computing only its column window `rank` and all-gathering the rest, inside a
    shard_map over that axis.

    x bf16 [m <= 16, k] (the same on every rank); w_q fp8_e4m3fn [k, n] (K-major,
    replicated); w_s f32 [n]; rank int32 [1] (this core's index on the axis, as
    data) -> bf16 [m, n]. A mesh with a second axis names it `lane_axis` and
    passes this core's index on it as `lane` (int32 [1]): peers are this lane's
    ranks.

    w_uk bf16 [h, nope, r] (this rank's h heads of the MLA W_UK, in window order):
    also absorb each of the window's h heads -- its first `nope` columns times its
    W_UK, in f32, rounded to bf16 -- and all-gather that too; returns (q [m, n],
    ql [m, num_cores * h * r]), ql token-major by head.
    """
    m, k = x.shape
    n_out = w_q.shape[1]
    if not supports(m, k, n_out, num_cores):
        raise ValueError(
            f"[{m}, {k}] x [{k}, {n_out}] windows over {num_cores} ranks: unsupported"
        )
    if (lane_axis is None) != (lane is None):
        raise ValueError("lane_axis and lane go together")
    absorb = None
    if w_uk is not None:
        absorb = Absorb(*w_uk.shape)
        if w_uk.dtype != jnp.bfloat16 or not absorb_supports(n_out, num_cores, absorb):
            raise ValueError(
                f"{w_uk.dtype} W_UK {tuple(w_uk.shape)} over {num_cores} windows of "
                f"{n_out} columns: unsupported"
            )
    call = _build(m, k, n_out, w_q.dtype, x.dtype, num_cores, axis, lane_axis, absorb)
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    scalars = (rank.astype(jnp.int32),)
    if lane is not None:
        scalars += (lane.astype(jnp.int32),)
    return _call_constrained(
        call, scalars, x, w_s2, w_q, *(() if w_uk is None else (w_uk,))
    )
