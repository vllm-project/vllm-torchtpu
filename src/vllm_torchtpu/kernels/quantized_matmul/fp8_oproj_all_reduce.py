# SPDX-License-Identifier: Apache-2.0
"""Row-parallel FP8 decode linear fused with its TP all-reduce: one Pallas kernel.

A row-parallel layer at decode (o_proj: [M <= 16, 1024] x [1024, 6144] per rank on 16
ranks) runs today as two calls -- `fp8_skinny` (the weight stream) then the small
two-shot all-reduce (`small_collectives.all_reduce`). The all-reduce starts with a
barrier round (~2.2 us on a 2x2x2 tpu7x slice) that cannot begin until the matmul's
call has ended, so it is exposed every layer. This kernel signals the barrier at entry,
lets the 6 MB weight stream hide it, and runs the same two-shot exchange on the
finished tiles in the same call:

    entry   start the scale DMA and the weight ring (all tiles in flight, fp8_skinny's
            ring); signal the barrier on every rank, self included; quantize x
    tile j  wait W_j; y = x_q . W_j (f32); out[:, tile j] = bf16(y * x_scale * w_scale)
            (`waves`) after tile 0: wait the barrier and push every owner chunk that is
            complete after each tile
    RS      after the last tile: wait the barrier; push out[:, chunk t] to rank t's slot
            `me` for every t != me; a local DMA of chunk me into slot me
    sum     wait the 15 inbound chunks (one byte-counted semaphore); sum the 16 slots in
            float32 in rank order, round once to bf16
    AG      push the sum into chunk me of every other rank's out; a local DMA into my
            own; wait the 15 inbound chunks; drain both send semaphores

Numerics are the two production kernels': the partial is fp8_skinny's expression and
the f32 sum runs over the same slots in the same (rank) order as the two-shot kernel,
so the output is bitwise the two-call path's. Every row moves (the two-shot full path).

Safety, as `small_collectives.common.barrier`: a rank writes into a peer only after the
barrier of this call, which the peer signals at entry, i.e. after it finished the
previous call; a peer's out[:, chunk me] is overwritten by my all-gather only after I
received that chunk from it, so its reduce-scatter read is complete.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MESH = pl.DeviceIdType.MESH
# The barrier group of the small all-reduce this kernel replaces
# (small_collectives.all_reduce.COLLECTIVE_ID).
COLLECTIVE_ID = 5
LANES = 128
MAX_M = 16
# The largest block any caller may allow (VMEM: the 6 MB weight ring plus 16 receive
# slots and the output through 96 rows fit the scoped limit).
LIMIT_M = 96
# (k, n) -> (tile, nbuf): fp8_skinny's measured o_proj ring ("n", 2048, 3).
TUNED = {(1024, 6144): (2048, 3)}


def supports(m: int, k: int, n: int, num_cores: int, max_m: int = MAX_M) -> bool:
    """Whether a [m, k] x [k, n] row-parallel block over num_cores ranks takes the
    kernel: a measured (k, n), m <= max_m (16 unless the caller allows up to 96),
    and owner chunks of whole lane groups."""
    return (
        m <= min(max_m, LIMIT_M)
        and (k, n) in TUNED
        and num_cores >= 2
        and n % num_cores == 0
        and (n // num_cores) % LANES == 0
    )


def owner_waves(num_cores: int, n: int, tile: int) -> list[list[int]]:
    """The owner chunks first complete after each tile (owner j after the tile
    holding its last column)."""
    own = n // num_cores
    waves: list[list[int]] = [[] for _ in range(n // tile)]
    for j in range(num_cores):
        waves[((j + 1) * own - 1) // tile].append(j)
    return waves


def _kernel(
    idx_ref,
    rank_ref,
    *refs,
    n,
    axis,
    t,
    n_tiles,
    nbuf,
    q_dtype,
    waves,
    uv_heads,
):
    if uv_heads:
        (x_ref, wuv_hbm, ws_hbm, w_hbm, o_ref, xq_ref, xs_ref, wbuf, wsem, ws_vmem,
         ws_sem, recv, red, rs_send, rs_recv, ag_send, ag_recv, loc, wuv_vmem,
         wuv_sem, xv_ref) = refs  # fmt: skip
        wuv_copy = pltpu.make_async_copy(wuv_hbm, wuv_vmem, wuv_sem)
        wuv_copy.start()
    else:
        (x_ref, ws_hbm, w_hbm, o_ref, xq_ref, xs_ref, wbuf, wsem, ws_vmem, ws_sem,
         recv, red, rs_send, rs_recv, ag_send, ag_recv, loc) = refs  # fmt: skip
    s = idx_ref[0]
    me = rank_ref[0]
    own = o_ref.shape[1] // n
    bsem = pltpu.get_barrier_semaphore()
    dn = (((1,), (0,)), ((), ()))

    ws_copy = pltpu.make_async_copy(ws_hbm, ws_vmem, ws_sem)
    ws_copy.start()

    def copy(j):
        return pltpu.make_async_copy(
            w_hbm.at[s, :, pl.ds(j * t, t)], wbuf.at[j % nbuf], wsem.at[j % nbuf]
        )

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()
    for j in range(n):
        pl.semaphore_signal(
            bsem, 1, device_id={axis: jnp.int32(j)}, device_id_type=MESH
        )

    if uv_heads:
        # The value up-projection of each head, as the bmm it replaces: the
        # latent in its own dtype times W_UV, f32 accumulation, rounded to the
        # activation dtype.
        wuv_copy.wait()
        c = x_ref.shape[1] // uv_heads
        v = wuv_vmem.shape[2]
        for h in range(uv_heads):
            y_h = jax.lax.dot_general(
                x_ref[:, h * c : (h + 1) * c],
                wuv_vmem[h].astype(x_ref.dtype),
                dn,
                preferred_element_type=jnp.float32,
            )
            xv_ref[:, h * v : (h + 1) * v] = y_h.astype(xv_ref.dtype)
        x = xv_ref[...]
    else:
        x = x_ref[...]

    # Activation quant: fp8_skinny._kernel's formula (bit-identical x_q and scale).
    q_max = float(jnp.finfo(q_dtype).max)
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(
        jnp.float32
    ) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    def chunk(j):
        return o_ref.at[:, pl.ds(j * own, own)]

    def push_owner(j):
        @pl.when(jnp.int32(j) != me)
        def _():
            pltpu.make_async_remote_copy(
                src_ref=chunk(j),
                dst_ref=recv.at[me],
                send_sem=rs_send,
                recv_sem=rs_recv,
                device_id={axis: jnp.int32(j)},
                device_id_type=MESH,
            ).start()

        @pl.when(jnp.int32(j) == me)
        def _():
            pltpu.make_async_copy(chunk(j), recv.at[j], loc.at[0]).start()

    wave_of = owner_waves(n, o_ref.shape[1], t) if waves else None
    for j in range(n_tiles):
        copy(j).wait()
        y = jax.lax.dot_general(
            xq_ref[...], wbuf[j % nbuf], dn, preferred_element_type=jnp.float32
        )
        if j == 0:
            ws_copy.wait()
        cols = slice(j * t, (j + 1) * t)
        o_ref[:, cols] = (y * xs_ref[...] * ws_vmem[:, cols]).astype(o_ref.dtype)
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()
        if wave_of is not None:
            if j == 0:
                pl.semaphore_wait(bsem, n)
            for o in wave_of[j]:
                push_owner(o)
    if wave_of is None:
        pl.semaphore_wait(bsem, n)
        for o in range(n):
            push_owner(o)

    slots = recv.at[pl.ds(0, n - 1)]  # the byte count of 15 chunks
    pltpu.make_async_copy(recv.at[0], recv.at[0], loc.at[0]).wait()
    pltpu.make_async_copy(slots, slots, rs_recv).wait()
    acc = recv[0].astype(jnp.float32)
    for r in range(1, n):
        acc = acc + recv[r].astype(jnp.float32)
    red[...] = acc.astype(red.dtype)

    dst_me = o_ref.at[:, pl.ds(pl.multiple_of(me * own, LANES), own)]
    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            pltpu.make_async_remote_copy(
                src_ref=red,
                dst_ref=dst_me,
                send_sem=ag_send,
                recv_sem=ag_recv,
                device_id={axis: jnp.int32(j)},
                device_id_type=MESH,
            ).start()

    own_ag = pltpu.make_async_copy(red, dst_me, loc.at[1])
    own_ag.start()
    own_ag.wait()
    pltpu.make_async_copy(slots, slots, ag_recv).wait()
    pltpu.make_async_copy(slots, slots, rs_send).wait()
    pltpu.make_async_copy(slots, slots, ag_send).wait()


@functools.cache
def _build(m, k, n_out, n_stack, w_dtype, x_dtype, num_cores, axis, waves, uv=None):
    """`uv` = (heads, latent width, latent dtype, W_UV dtype): the kernel takes the
    latent [m, heads * latent width] and W_UV [heads, latent width, k / heads]
    and projects it first (x_dtype is then the activation dtype)."""
    t, nbuf = TUNED[(k, n_out)]
    n_tiles = n_out // t
    nbuf = max(1, min(nbuf, n_tiles))
    own = n_out // num_cores
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),  # x_q (fp8 values held as bf16)
        pltpu.VMEM((m, 1), jnp.float32),  # x scale
        pltpu.VMEM((nbuf, k, t), w_dtype),  # weight ring
        pltpu.SemaphoreType.DMA((nbuf,)),
        pltpu.VMEM((1, n_out), jnp.float32),  # weight scale
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((num_cores, m, own), x_dtype),  # receive slots, by sender rank
        pltpu.VMEM((m, own), x_dtype),  # my summed chunk
        pltpu.SemaphoreType.DMA,  # reduce-scatter send
        pltpu.SemaphoreType.DMA,  # reduce-scatter receive
        pltpu.SemaphoreType.DMA,  # all-gather send
        pltpu.SemaphoreType.DMA,  # all-gather receive
        pltpu.SemaphoreType.DMA((2,)),  # local copies
    ]
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [vmem, hbm, hbm]
    heads = 0
    if uv is not None:
        heads, latent, _, wuv_dtype = uv
        scratch += [
            pltpu.VMEM((heads, latent, k // heads), wuv_dtype),  # W_UV
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((m, k), x_dtype),  # the projected values
        ]
        in_specs = [vmem, hbm, hbm, hbm]
    kern = functools.partial(
        _kernel,
        n=num_cores,
        axis=axis,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        waves=waves,
        uv_heads=heads,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=in_specs,
        out_specs=vmem,
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n_out), x_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=(
            f"fp8_{'uv_' if uv else ''}oproj_all_reduce{'_w' if waves else ''}"
            f"_m{m}_k{k}_n{n_out}"
        ),
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained(call, idx, rank, x, w_s2, w_stack):
    # Scale and weight constrained to HBM, as fp8_skinny's scale_dma + hbm_pin.
    return call(
        idx,
        rank,
        x,
        pltpu.with_memory_space_constraint(w_s2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_stack, pltpu.HBM),
    )


def fp8_matmul_all_reduce_stacked(
    x, w_stack, idx, w_s, rank, *, num_cores, axis, waves=False, max_m=MAX_M
):
    """sum over the num_cores ranks of mesh axis `axis` of
    fp8_skinny(x, w_stack[idx[0]], w_s), inside a shard_map over that axis.

    x bf16 [m <= max_m, k]; w_stack fp8_e4m3fn [L, k, n] (K-major); idx, rank int32 [1]
    (rank = this core's index on the axis, as data); w_s f32 [n] -> bf16 [m, n].
    """
    m, k = x.shape
    n_stack, _, n_out = w_stack.shape
    if not supports(m, k, n_out, num_cores, max_m):
        raise ValueError(
            f"[{m}, {k}] x [{k}, {n_out}] over {num_cores} ranks: unsupported"
        )
    call = _build(m, k, n_out, n_stack, w_stack.dtype, x.dtype, num_cores, axis, waves)
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    return _call_constrained(
        call, idx.astype(jnp.int32), rank.astype(jnp.int32), x, w_s2, w_stack
    )


def fp8_matmul_all_reduce(
    x, w_q, w_s, rank, *, num_cores, axis, waves=False, max_m=MAX_M
):
    """`fp8_matmul_all_reduce_stacked` on one [k, n] weight."""
    return fp8_matmul_all_reduce_stacked(
        x,
        w_q[None],
        jnp.zeros((1,), jnp.int32),
        w_s,
        rank,
        num_cores=num_cores,
        axis=axis,
        waves=waves,
        max_m=max_m,
    )


def supports_uv(m, heads, latent, k, n, num_cores, max_m=MAX_M) -> bool:
    """Whether a latent [m, heads * latent] through W_UV [heads, latent, k / heads]
    into a [k, n] row-parallel block takes the kernel: `supports` for the block,
    and lane-aligned head slices on both sides."""
    return (
        supports(m, k, n, num_cores, max_m)
        and heads > 0
        and k % heads == 0
        and latent % LANES == 0
        and (k // heads) % LANES == 0
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_uv_constrained(call, idx, rank, x_lat, w_uv, w_s2, w_stack):
    return call(
        idx,
        rank,
        x_lat,
        pltpu.with_memory_space_constraint(w_uv, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_s2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_stack, pltpu.HBM),
    )


def fp8_uv_matmul_all_reduce(
    x_lat, w_uv, w_q, w_s, rank, *, num_cores, axis, out_dtype, waves=False,
    max_m=MAX_M,
):  # fmt: skip
    """The sum over the num_cores ranks of mesh axis `axis` of fp8_skinny(v, w_q, w_s)
    with v = the per-head bmm of x_lat [m, heads * latent] and w_uv [heads, latent,
    k / heads] rounded to out_dtype, inside a shard_map over that axis: the value
    up-projection, the output projection and its all-reduce in one kernel."""
    m, width = x_lat.shape
    heads, latent, v = w_uv.shape
    k, n_out = w_q.shape
    if width != heads * latent or heads * v != k:
        raise ValueError(f"latent {x_lat.shape}, W_UV {w_uv.shape}, w {w_q.shape}")
    if not supports_uv(m, heads, latent, k, n_out, num_cores, max_m):
        raise ValueError(
            f"[{m}, {width}] -> [{k}, {n_out}] over {num_cores} ranks: unsupported"
        )
    uv = (heads, latent, jnp.dtype(x_lat.dtype).name, w_uv.dtype)
    call = _build(
        m, k, n_out, 1, w_q.dtype, jnp.dtype(out_dtype), num_cores, axis, waves, uv
    )
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    return _call_uv_constrained(
        call,
        jnp.zeros((1,), jnp.int32),
        rank.astype(jnp.int32),
        x_lat,
        w_uv,
        w_s2,
        w_q[None],
    )
