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

One-row build (`one_row`): the kernel also takes the step's
valid-row count. With one valid row (a batch-1 decode step) it skips RS and AG: after
the last tile it folds row 0 into one row tile (16 x n / 16), pushes that to slot `me`
of every peer and sums the 16 slots itself -- one round instead of two. Each element is
the same f32 sum in rank order, so row 0 is bitwise the two-shot's; rows >= 1 (padding)
come out +0. Other counts run the two-shot unchanged.

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
# The one-row path folds a row into this many rows (the bf16 sublane tiling).
ROW_TILE = 16
# The barrier group of the small all-reduce this kernel replaces
# (small_collectives.all_reduce.COLLECTIVE_ID).
COLLECTIVE_ID = 5
LANES = 128
MAX_M = 64
D2D_MIN_M = 32
# (k, n) -> (tile, nbuf): fp8_skinny's measured o_proj ring ("n", 2048, 3).
TUNED = {(1024, 6144): (2048, 3)}


def supports(m: int, k: int, n: int, num_cores: int) -> bool:
    """Whether a [m, k] x [k, n] row-parallel block over num_cores ranks takes the
    kernel: a measured (k, n), m <= 64, and owner chunks of whole lane groups."""
    return (
        0 < m <= MAX_M
        and (k, n) in TUNED
        and num_cores >= 2
        and n % num_cores == 0
        and (n // num_cores) % LANES == 0
    )


def d2d_first_supported(m: int, n: int, num_cores: int) -> bool:
    return (
        m >= D2D_MIN_M
        and num_cores >= 4
        and num_cores % 2 == 0
        and n % num_cores == 0
        and (n // num_cores) % LANES == 0
    )


def one_row_supported(n: int) -> bool:
    """Whether one output row of n columns folds into ROW_TILE rows of whole lane
    groups (the one-row build)."""
    return n % (ROW_TILE * LANES) == 0


def owner_waves(num_cores: int, n: int, tile: int) -> list[list[int]]:
    """The owner chunks first complete after each tile (owner j after the tile
    holding its last column)."""
    own = n // num_cores
    waves: list[list[int]] = [[] for _ in range(n // tile)]
    for j in range(num_cores):
        waves[((j + 1) * own - 1) // tile].append(j)
    return waves


def _one_row_exchange(o_ref, one, me, rs_send, rs_recv, *, n, axis):
    """All-reduce row 0 of o_ref in one round (module docstring): fold it into
    one[me], push that to every peer's slot `me`, sum the n slots in rank order."""
    m, n_out = o_ref.shape
    w = n_out // ROW_TILE
    row0 = o_ref[pl.ds(0, 1), :].astype(jnp.float32)
    rows = jax.lax.broadcasted_iota(jnp.int32, (ROW_TILE, w), 0)
    q1 = jnp.zeros((ROW_TILE, w), jnp.float32)
    for j in range(ROW_TILE):
        part = jnp.broadcast_to(row0[:, j * w : (j + 1) * w], (ROW_TILE, w))
        q1 = jnp.where(rows == j, part, q1)
    one.at[me][...] = q1.astype(one.dtype)
    # Avoid fixed rank-0..n-1 loop: rotating peer order eliminates link incast.
    for step in range(1, n):
        dst = jax.lax.rem(me + step, jnp.int32(n))
        pltpu.make_async_remote_copy(
            src_ref=one.at[me],
            dst_ref=one.at[me],
            send_sem=rs_send,
            recv_sem=rs_recv,
            device_id={axis: dst},
            device_id_type=MESH,
        ).start()

    peers = one.at[pl.ds(0, n - 1)]  # the byte count of n - 1 slots
    pltpu.make_async_copy(peers, peers, rs_recv).wait()
    acc = one[0].astype(jnp.float32)
    for r in range(1, n):
        acc = acc + one[r].astype(jnp.float32)
    total = acc.astype(o_ref.dtype).astype(jnp.float32)
    first = jax.lax.broadcasted_iota(jnp.int32, (m, w), 0) == 0
    for j in range(ROW_TILE):
        part = jnp.broadcast_to(total[j : j + 1, :], (m, w))
        o_ref[:, pl.ds(j * w, w)] = jnp.where(first, part, 0.0).astype(o_ref.dtype)
    pltpu.make_async_copy(peers, peers, rs_send).wait()


def _d2d_first_exchange(
    o_ref,
    d2d_recv,
    rs_recv,
    red_buf,
    sems,
    me,
    *,
    n,
    axis,
    single=None,
):
    """1-barrier pair-first reduce-scatter + half-world RS/AG + pair all-gather."""
    (
        d2d_send,
        d2d_recv_sem,
        d2d_ag_send,
        d2d_ag_recv,
        rs_send,
        rs_recv_sem,
        ag_send,
        ag_recv_sem,
        loc,
    ) = sems
    m, n_out = o_ref.shape
    half_n = n_out // 2
    n_pairs = n // 2
    chunk_w = half_n // n_pairs

    bit = jax.lax.bitwise_and(me, jnp.int32(1))
    pair = jax.lax.shift_right_logical(me, jnp.int32(1))
    partner = jax.lax.bitwise_xor(me, jnp.int32(1))
    my_off = bit * half_n
    peer_off = (1 - bit) * half_n
    my_slice = o_ref.at[:, pl.ds(pl.multiple_of(my_off, half_n), half_n)]
    peer_slice = o_ref.at[:, pl.ds(pl.multiple_of(peer_off, half_n), half_n)]

    pair_rs = pltpu.make_async_remote_copy(
        src_ref=peer_slice,
        dst_ref=d2d_recv,
        send_sem=d2d_send.at[0],
        recv_sem=d2d_recv_sem.at[0],
        device_id={axis: partner},
        device_id_type=MESH,
    )
    pair_rs.start()
    pltpu.make_async_copy(d2d_recv, d2d_recv, d2d_recv_sem.at[0]).wait_recv()
    my_slice[...] = (
        my_slice[...].astype(jnp.float32) + d2d_recv[...].astype(jnp.float32)
    ).astype(o_ref.dtype)

    def chunk_half(p_idx):
        return o_ref.at[
            :, pl.ds(pl.multiple_of(my_off + p_idx * chunk_w, chunk_w), chunk_w)
        ]

    def slot_half(p_idx):
        return rs_recv.at[p_idx, pl.ds(0, m), pl.ds(0, chunk_w)]

    own_c = pltpu.make_async_copy(chunk_half(pair), slot_half(pair), loc.at[0])
    own_c.start()
    rs_copies = []
    for idx, step in enumerate(range(1, n_pairs)):
        dst_p = jax.lax.rem(pair + step, jnp.int32(n_pairs))
        dst_r = dst_p * 2 + bit
        cp = pltpu.make_async_remote_copy(
            src_ref=chunk_half(dst_p),
            dst_ref=slot_half(pair),
            send_sem=rs_send.at[idx],
            recv_sem=rs_recv_sem.at[pair],
            device_id={axis: dst_r},
            device_id_type=MESH,
        )
        cp.start()
        rs_copies.append(cp)
    own_c.wait()
    for step in range(1, n_pairs):
        src_p = jax.lax.rem(pair + (n_pairs - step), jnp.int32(n_pairs))
        pltpu.make_async_copy(
            slot_half(src_p), slot_half(src_p), rs_recv_sem.at[src_p]
        ).wait_recv()
    acc = rs_recv[0].astype(jnp.float32)
    for q in range(1, n_pairs):
        acc = acc + rs_recv[q].astype(jnp.float32)
    red_buf[...] = acc.astype(red_buf.dtype)

    own_a = pltpu.make_async_copy(red_buf, chunk_half(pair), loc.at[1])
    own_a.start()
    ag_copies = []
    for idx, step in enumerate(range(1, n_pairs)):
        dst_p = jax.lax.rem(pair + step, jnp.int32(n_pairs))
        dst_r = dst_p * 2 + bit
        cp = pltpu.make_async_remote_copy(
            src_ref=red_buf,
            dst_ref=chunk_half(pair),
            send_sem=ag_send.at[idx],
            recv_sem=ag_recv_sem.at[pair],
            device_id={axis: dst_r},
            device_id_type=MESH,
        )
        cp.start()
        ag_copies.append(cp)
    own_a.wait()
    for step in range(1, n_pairs):
        src_p = jax.lax.rem(pair + (n_pairs - step), jnp.int32(n_pairs))
        pltpu.make_async_copy(
            chunk_half(src_p), chunk_half(src_p), ag_recv_sem.at[src_p]
        ).wait_recv()

    pair_rs.wait_send()
    pair_ag = pltpu.make_async_remote_copy(
        src_ref=my_slice,
        dst_ref=my_slice,
        send_sem=d2d_ag_send.at[0],
        recv_sem=d2d_ag_recv.at[0],
        device_id={axis: partner},
        device_id_type=MESH,
    )
    pair_ag.start()
    pltpu.make_async_copy(peer_slice, peer_slice, d2d_ag_recv.at[0]).wait_recv()
    pair_ag.wait_send()
    for cp in (*rs_copies, *ag_copies):
        cp.wait_send()
    if single is not None:

        @pl.when(single)
        def _zero_pad():
            row_idx = jax.lax.broadcasted_iota(jnp.int32, (m, n_out), 0)
            o_ref[...] = jnp.where(
                row_idx == 0, o_ref[...].astype(jnp.float32), 0.0
            ).astype(o_ref.dtype)


def _apply_add_rmsnorm(o_ref, res_ref, nw_ref, norm_ref, eps: float):
    s = o_ref[...].astype(jnp.float32) + res_ref[...].astype(jnp.float32)
    o_ref[...] = s.astype(o_ref.dtype)
    inv = jax.lax.rsqrt(jnp.mean(s * s, axis=-1, keepdims=True) + jnp.float32(eps))
    # Avoid intermediate bf16 cast on `s * inv`: XLA TPU keeps f32 excess precision
    # across the weight multiply, while Mosaic Pallas materializes explicit bf16
    # rounding (~25.9% 1-ULP precision loss per layer vs unfused RMSNorm).
    y = (s * inv) * nw_ref[...].astype(jnp.float32)
    norm_ref[...] = y.astype(norm_ref.dtype)


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
    one_row,
    d2d_first,
    has_rmsnorm,
    eps,
):
    nv_ref, *refs = refs if one_row else (None, *refs)
    if has_rmsnorm:
        (
            x_ref,
            ws_hbm,
            w_hbm,
            res_ref,
            nw_ref,
            norm_ref,
            o_ref,
            xq_ref,
            xs_ref,
            wbuf,
            wsem,
            ws_vmem,
            ws_sem,
            *ar_refs,
        ) = refs
    else:
        res_ref = nw_ref = norm_ref = None
        (
            x_ref,
            ws_hbm,
            w_hbm,
            o_ref,
            xq_ref,
            xs_ref,
            wbuf,
            wsem,
            ws_vmem,
            ws_sem,
            *ar_refs,
        ) = refs
    s = idx_ref[0]
    me = rank_ref[0]
    own = o_ref.shape[1] // n
    bsem = pltpu.get_barrier_semaphore()
    # The one-row path's step: one valid row (data, the same on every rank).
    single = None if nv_ref is None else nv_ref[0] == 1

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
    if d2d_first:
        (
            d2d_recv,
            rs_recv_buf,
            red_buf,
            d2d_send,
            d2d_recv_sem,
            d2d_ag_send,
            d2d_ag_recv,
            rs_send,
            rs_recv,
            ag_send,
            ag_recv,
            loc,
        ) = ar_refs
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
        pl.semaphore_wait(bsem, n)
        _d2d_first_exchange(
            o_ref,
            d2d_recv,
            rs_recv_buf,
            red_buf,
            (
                d2d_send,
                d2d_recv_sem,
                d2d_ag_send,
                d2d_ag_recv,
                rs_send,
                rs_recv,
                ag_send,
                ag_recv,
                loc,
            ),
            me,
            n=n,
            axis=axis,
            single=single,
        )
        if has_rmsnorm:
            _apply_add_rmsnorm(o_ref, res_ref, nw_ref, norm_ref, eps)
        return

    recv, red, rs_send, rs_recv, ag_send, ag_recv, loc, *one = ar_refs

    def chunk(j):
        return o_ref.at[:, pl.ds(pl.multiple_of(j * own, LANES), own)]

    def two_shot(cond):
        """cond, and not the one-row path's step."""
        return (
            jnp.logical_and(jnp.logical_not(single), cond)
            if one_row and single is not None
            else cond
        )

    def push_owner(dst):
        @pl.when(two_shot(dst != me))
        def _():
            pltpu.make_async_remote_copy(
                src_ref=chunk(dst),
                dst_ref=recv.at[me],
                send_sem=rs_send,
                recv_sem=rs_recv,
                device_id={axis: dst},
                device_id_type=MESH,
            ).start()

        @pl.when(two_shot(dst == me))
        def _():
            pltpu.make_async_copy(chunk(dst), recv.at[dst], loc.at[0]).start()

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
            wlist = wave_of[j]
            nw_len = len(wlist)
            for idx_w in range(nw_len):
                # Avoid fixed owner-order pushes: rotating inside each contiguous wave
                # spreads remote copies across peers without changing rank summation.
                dst = jnp.int32(wlist[0]) + jax.lax.rem(me + idx_w, jnp.int32(nw_len))
                push_owner(dst)
    if wave_of is None:
        pl.semaphore_wait(bsem, n)
        for step in range(n):
            push_owner(jax.lax.rem(me + step, jnp.int32(n)))

    tail = functools.partial(
        _two_shot_tail,
        o_ref,
        recv,
        red,
        (rs_send, rs_recv, ag_send, ag_recv, loc),
        me,
        n=n,
        axis=axis,
    )
    if not one_row:
        tail()
    else:
        assert single is not None
        pl.when(single)(
            lambda: _one_row_exchange(
                o_ref, one[0], me, rs_send, rs_recv, n=n, axis=axis
            )
        )
        pl.when(jnp.logical_not(single))(tail)
    if has_rmsnorm:
        _apply_add_rmsnorm(o_ref, res_ref, nw_ref, norm_ref, eps)


def _two_shot_tail(o_ref, recv, red, sems, me, *, n, axis):
    """Wait the reduce-scatter's chunks, sum them, all-gather the sum."""
    rs_send, rs_recv, ag_send, ag_recv, loc = sems
    own = o_ref.shape[1] // n
    slots = recv.at[pl.ds(0, n - 1)]  # the byte count of 15 chunks
    pltpu.make_async_copy(recv.at[0], recv.at[0], loc.at[0]).wait()
    pltpu.make_async_copy(slots, slots, rs_recv).wait()
    acc = recv[0].astype(jnp.float32)
    for r in range(1, n):
        acc = acc + recv[r].astype(jnp.float32)
    red[...] = acc.astype(red.dtype)

    dst_me = o_ref.at[:, pl.ds(pl.multiple_of(me * own, LANES), own)]
    for step in range(1, n):
        dst = jax.lax.rem(me + step, jnp.int32(n))
        pltpu.make_async_remote_copy(
            src_ref=red,
            dst_ref=dst_me,
            send_sem=ag_send,
            recv_sem=ag_recv,
            device_id={axis: dst},
            device_id_type=MESH,
        ).start()

    own_ag = pltpu.make_async_copy(red, dst_me, loc.at[1])
    own_ag.start()
    own_ag.wait()
    pltpu.make_async_copy(slots, slots, ag_recv).wait()
    pltpu.make_async_copy(slots, slots, rs_send).wait()
    pltpu.make_async_copy(slots, slots, ag_send).wait()


@functools.cache
def _build(
    m,
    k,
    n_out,
    n_stack,
    w_dtype,
    x_dtype,
    num_cores,
    axis,
    waves,
    one_row,
    nw_dtype=None,
    eps=1e-6,
):
    t, nbuf = TUNED[(k, n_out)]
    n_tiles = n_out // t
    nbuf = max(1, min(nbuf, n_tiles))
    own = n_out // num_cores
    d2d_first = d2d_first_supported(m, n_out, num_cores)
    has_rmsnorm = nw_dtype is not None
    base_scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),  # x_q (fp8 values held as bf16)
        pltpu.VMEM((m, 1), jnp.float32),  # x scale
        pltpu.VMEM((nbuf, k, t), w_dtype),  # weight ring
        pltpu.SemaphoreType.DMA((nbuf,)),
        pltpu.VMEM((1, n_out), jnp.float32),  # weight scale
        pltpu.SemaphoreType.DMA,
    ]
    if d2d_first:
        n_pairs = num_cores // 2
        scratch = [
            *base_scratch,
            pltpu.VMEM((m, n_out // 2), x_dtype),  # intra-pair receive
            pltpu.VMEM((n_pairs, m, own), x_dtype),  # half-world RS slots
            pltpu.VMEM((m, own), x_dtype),  # reduced chunk
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((n_pairs,)),
            pltpu.SemaphoreType.DMA((n_pairs,)),
            pltpu.SemaphoreType.DMA((n_pairs,)),
            pltpu.SemaphoreType.DMA((n_pairs,)),
            pltpu.SemaphoreType.DMA((2,)),
        ]
    else:
        scratch = [
            *base_scratch,
            pltpu.VMEM((num_cores, m, own), x_dtype),  # receive slots, by sender rank
            pltpu.VMEM((m, own), x_dtype),  # my summed chunk
            pltpu.SemaphoreType.DMA,  # reduce-scatter send
            pltpu.SemaphoreType.DMA,  # reduce-scatter receive
            pltpu.SemaphoreType.DMA,  # all-gather send
            pltpu.SemaphoreType.DMA,  # all-gather receive
            pltpu.SemaphoreType.DMA((2,)),  # local copies
        ]
        if one_row:
            # The one-row path's slots, by sender rank: row 0 folded into ROW_TILE rows.
            scratch.append(
                pltpu.VMEM((num_cores, ROW_TILE, n_out // ROW_TILE), x_dtype)
            )
    kern = functools.partial(
        _kernel,
        n=num_cores,
        axis=axis,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        waves=waves,
        one_row=one_row,
        d2d_first=d2d_first,
        has_rmsnorm=has_rmsnorm,
        eps=float(eps),
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [vmem, hbm, hbm, vmem, vmem] if has_rmsnorm else [vmem, hbm, hbm]
    out_specs = [vmem, vmem] if has_rmsnorm else vmem
    out_shape = (
        (
            jax.ShapeDtypeStruct((m, n_out), x_dtype),
            jax.ShapeDtypeStruct((m, n_out), x_dtype),
        )
        if has_rmsnorm
        else jax.ShapeDtypeStruct((m, n_out), x_dtype)
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3 if one_row else 2,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=(
            f"fp8_oproj_all_reduce{'_d2d' if d2d_first else ''}"
            f"{'_w' if waves and not d2d_first else ''}{'_1r' if one_row else ''}"
            f"{'_rn' if has_rmsnorm else ''}_m{m}_k{k}_n{n_out}"
        ),
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained(call, scalars, x, w_s2, w_stack):
    # Scale and weight constrained to HBM, as fp8_skinny's scale_dma + hbm_pin.
    return call(
        *scalars,
        x,
        pltpu.with_memory_space_constraint(w_s2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_stack, pltpu.HBM),
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained_rmsnorm(call, scalars, x, w_s2, w_stack, residual, norm_w2):
    return call(
        *scalars,
        x,
        pltpu.with_memory_space_constraint(w_s2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_stack, pltpu.HBM),
        residual,
        norm_w2,
    )


def fp8_matmul_all_reduce_stacked(
    x,
    w_stack,
    idx,
    w_s,
    rank,
    *,
    num_cores,
    axis,
    waves=False,
    n_valid=None,
    residual=None,
    norm_w=None,
    eps: float = 1e-6,
):
    """sum over the num_cores ranks of mesh axis `axis` of
    fp8_skinny(x, w_stack[idx[0]], w_s), inside a shard_map over that axis.

    x bf16 [m <= 64, k]; w_stack fp8_e4m3fn [L, k, n] (K-major); idx, rank int32 [1]
    (rank = this core's index on the axis, as data); w_s f32 [n] -> bf16 [m, n], or
    (normed [m, n], new_residual [m, n]) when residual and norm_w are provided.
    n_valid int32 [1] (the same on every rank) builds the one-row kernel: rows
    >= n_valid are padding, and with n_valid == 1 they come out +0 before add+norm.
    """
    m, k = x.shape
    n_stack, _, n_out = w_stack.shape
    if not supports(m, k, n_out, num_cores):
        raise ValueError(
            f"[{m}, {k}] x [{k}, {n_out}] over {num_cores} ranks: unsupported"
        )
    if (residual is None) != (norm_w is None):
        raise ValueError("residual and norm_w must both be provided or both None")
    if residual is not None and norm_w is not None:
        if residual.shape != (m, n_out) or residual.dtype != x.dtype:
            raise ValueError(
                f"residual {residual.shape}/{residual.dtype} != "
                f"({m}, {n_out})/{x.dtype}"
            )
        if norm_w.shape != (n_out,):
            raise ValueError(f"norm_w {norm_w.shape} != ({n_out},)")
    one_row = n_valid is not None
    if one_row and not one_row_supported(n_out):
        raise ValueError(f"the one-row build needs n = {n_out} to fill {ROW_TILE} rows")
    call = _build(
        m,
        k,
        n_out,
        n_stack,
        w_stack.dtype,
        x.dtype,
        num_cores,
        axis,
        waves,
        one_row,
        nw_dtype=norm_w.dtype if norm_w is not None else None,
        eps=float(eps),
    )
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    scalars = (idx.astype(jnp.int32), rank.astype(jnp.int32))
    if n_valid is not None:
        scalars += (n_valid.astype(jnp.int32),)
    if residual is not None and norm_w is not None:
        return _call_constrained_rmsnorm(
            call, scalars, x, w_s2, w_stack, residual, norm_w.reshape(1, n_out)
        )
    return _call_constrained(call, scalars, x, w_s2, w_stack)


def fp8_matmul_all_reduce(
    x,
    w_q,
    w_s,
    rank,
    *,
    num_cores,
    axis,
    waves=False,
    n_valid=None,
    residual=None,
    norm_w=None,
    eps: float = 1e-6,
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
        n_valid=n_valid,
        residual=residual,
        norm_w=norm_w,
        eps=eps,
    )
