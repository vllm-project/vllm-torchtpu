# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""In-kernel collectives of the megakernel (all barrier-free after the one entry barrier).

* `all_reduce`: main's moe_tp_decode all-reduce protocol (all_reduce.py `_exchange` @ 9ffdb1eb4: reduce-scatter of
  R column chunks, f32 sum of the R bf16 slots in rank order rounded to bf16, all-gather; per-sender receive
  semaphores; rank r's i-th copy to (r + i) % R). Copied rather than imported so the fault variants (a dropped
  partial) and the chunk-width fallback live next to it; the protocol and its numerics are unchanged.
* `exchange`: glm5_mk_dcp's one-round all-peer exchange (kernel.py `exchange` @ d41e70f6d, :896-920): q head
  all-gather and the partial + lse all-to-all.
"""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32, I32 = jnp.bfloat16, jnp.float32, jnp.int32
MESH = pl.DeviceIdType.MESH


def ar_scratch(R, M, H):
    """Scratch of one all-reduce slot set: recv [R, M, H/R] bf16, red [M, H/R] bf16, 4 x DMA((R,)), DMA((2,))."""
    cw = H // R
    return (
        [pltpu.VMEM((R, M, cw), BF16), pltpu.VMEM((M, cw), BF16)]
        + [pltpu.SemaphoreType.DMA((R,)) for _ in range(4)]
        + [pltpu.SemaphoreType.DMA((2,))]
    )


AR_NAMES = ("recv", "red", "rs_send", "rs_recv", "ag_send", "ag_recv", "loc")


def _cols(ref, t, size):
    return ref.at[:, pl.ds(pl.multiple_of(t * size, size), size)]


def _send_all(R, me, src_of, dst_of, send_sem, recv_sem):
    sends = []
    for step in range(1, R):
        t = lax.rem(me + step, R)
        c = pltpu.make_async_remote_copy(
            src_of(t),
            dst_of(me),
            send_sem.at[step],
            recv_sem.at[me],
            device_id=(t,),
            device_id_type=MESH,
        )
        c.start()
        sends.append(c)
    return sends


def _recv_all(R, me, dst_of, recv_sem):
    for step in range(1, R):
        t = lax.rem(me + (R - step), R)
        pltpu.make_async_copy(dst_of(t), dst_of(t), recv_sem.at[t]).wait_recv()


def all_reduce(R, me, buf, rows, cols, s, drop_last=False, early_read=False, mids=()):
    """In-place all-reduce of buf ([rows, cols] bf16 VMEM). s: dict(zip(AR_NAMES, scratch)).
    drop_last (FAULT control `missing_rank_partial`): rank R-1's partial is left out of the f32 sum.
    early_read (FAULT control `f_ar_earlyread`, a delay-exposed race): the reduce-scatter slots are summed BEFORE
    their receives are waited, so a late peer's slot still holds the previous layer's partial (the barrier-free
    slot reuse this protocol must never expose); the waits follow, keeping the semaphores balanced.
    mids (stage-B L2 `gearly`): up to two callables run while the reduce-scatter (mids[0]) and the all-gather
    (mids[1]) are in flight, i.e. in the TC's idle wait windows; they must not touch buf or the slot sets."""
    if R == 1:
        if drop_last:
            buf[...] = jnp.zeros(buf.shape, buf.dtype)
        return
    cw = cols // R

    def slot(q):
        return s["recv"].at[q, pl.ds(0, rows), pl.ds(0, cw)]

    def chunk(t):
        return _cols(buf, t, cw)

    red = s["red"].at[pl.ds(0, rows), pl.ds(0, cw)]
    own = pltpu.make_async_copy(chunk(me), slot(me), s["loc"].at[0])
    own.start()
    rs = _send_all(R, me, chunk, slot, s["rs_send"], s["rs_recv"])
    if len(mids) > 0 and mids[0] is not None:
        mids[0]()
    own.wait()
    if not early_read:
        _recv_all(R, me, slot, s["rs_recv"])
    acc = s["recv"][0, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
    for q in range(1, R):
        v = s["recv"][q, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
        if drop_last and q == R - 1:
            v = jnp.zeros_like(v)
        acc = acc + v
    red[...] = acc.astype(red.dtype)
    if early_read:  # the sum is materialised before the receives are waited
        _recv_all(R, me, slot, s["rs_recv"])
    own_ag = pltpu.make_async_copy(red, chunk(me), s["loc"].at[1])
    own_ag.start()
    ag = _send_all(R, me, lambda t: red, chunk, s["ag_send"], s["ag_recv"])
    if len(mids) > 1 and mids[1] is not None:
        mids[1]()
    own_ag.wait()
    _recv_all(R, me, chunk, s["ag_recv"])
    for c in (*rs, *ag):
        c.wait_send()


def row_switch(n_dyn, options, fn):
    """LV (live-row collectives): run fn(n) for the one static row count n in `options` equal to the runtime n_dyn
    (DMA shapes are static, so every option is its own issue site; exactly one fires, rank-uniformly)."""
    for n in options:

        @pl.when(n_dyn == n)
        def _(n=n):
            fn(n)


# LV all-reduce rows. Mosaic hazard (lv-prod, local AOT on libtpu 0.0.50 + the arlv8 oracle on 0.0.47): a row slice of
# a (16, 128)-tiled bf16 VMEM buffer that starts on a 16-row boundary and is 8 (mod 16) rows long KEEPS the (16, 128)
# tiling, so its DMA moves whole tiles -- 8 rows move 16, 24 move 32 (rows past the slice are overwritten). 4- and
# 12-row slices are re-tiled to (4, 128) and move exactly their rows; 20 / 28 do not compile. So an n-row transfer is
# issued as pieces: (0, 16k) whole tiles, then the remainder r = n mod 16 as (16k, 4), (16k, 12) or -- r = 8 --
# either two 4-row pieces ("split") or rounded up to 12 rows ("round", resolved in lv_ar_rows).
AR_SAFE = {4: "any 4-aligned offset", 12: "16-aligned offset", 16: "16-aligned offset (and any 16k)"}


def ar_pieces(n):
    """[(row offset, rows)] of an n-row LV all-reduce transfer, every piece Mosaic-exact (AR_SAFE). n % 4 == 0."""
    assert n > 0 and n % 4 == 0, n
    base, r = n // 16 * 16, n % 16
    ps = [(0, base)] if base else []
    if r == 4 or r == 12:
        ps.append((base, r))
    elif r == 8:
        ps += [(base, 4), (base + 4, 4)]
    for o, m in ps:
        assert (m % 16 == 0 and o == 0) or (m == 12 and o % 16 == 0) or (m == 4 and o % 4 == 0), (n, ps)
    return ps


def lv_ar_options(rows, floor=0, eight="split"):
    """The static row counts an LV all-reduce of a [rows, .] buffer can move: ceil(T / 4) * 4 clamped to >= floor
    (the next-smaller bucket: a step below it never runs this program), r = 8 (mod 16) rounded to 12 if
    eight == "round", and rows itself."""
    assert rows % 16 == 0 and eight in ("split", "round"), (rows, eight)
    out = set()
    for k in range(4, rows + 1, 4):
        n = max(k, floor)
        if eight == "round" and n % 16 == 8:
            n += 4
        out.add(min(n, rows))
    return tuple(sorted(out))


def lv_ar_rows(T, rows, floor=0, eight="split"):
    """Runtime (traced) row count of an LV all-reduce for T live rows: one of lv_ar_options(rows, floor, eight)."""
    n = jnp.maximum((T + 3) // 4 * 4, floor)
    if eight == "round":
        n = jnp.where(n % 16 == 8, n + 4, n)
    return jnp.minimum(n, rows)


def _send_all_loop(R, me, src_of, dst_of, send_sem, recv_sem):
    """_send_all as a fori_loop over the R - 1 steps (one issue site instead of R - 1: LV bundles)."""

    def body(step, c):
        t = lax.rem(me + step, R)
        pltpu.make_async_remote_copy(
            src_of(t), dst_of(me), send_sem.at[step], recv_sem.at[me], device_id=(t,), device_id_type=MESH
        ).start()
        return c

    lax.fori_loop(1, R, body, 0)


def _recv_all_loop(R, me, dst_of, recv_sem):
    def body(step, c):
        t = lax.rem(me + (R - step), R)
        pltpu.make_async_copy(dst_of(t), dst_of(t), recv_sem.at[t]).wait_recv()
        return c

    lax.fori_loop(1, R, body, 0)


def all_reduce_lv(R, me, buf, rows, cols, s, n_dyn, options, drop_last=False, mids=(), loop=0):
    """LV form of `all_reduce`: the same protocol (reduce-scatter of R column chunks, f32 rank-order sum, all-gather;
    the same semaphores and peer order), moving only the first n rows, n = n_dyn (traced, rank-uniform) in `options`
    (lv_ar_options), each transfer issued as Mosaic-exact pieces (ar_pieces). Rows >= n of buf are
    neither sent nor overwritten: the caller zeroes rows >= T, so buf ends bit-identical to `all_reduce`'s. Slot / red
    rows >= n hold stale bytes and are never sent (no mask needed). n == rows is `all_reduce`'s transfer.
    loop: 0 = one issue site per (row count, piece), per-peer sites unrolled; 1 = the same with fori_loops over the
    peers (bundles: stageB/lv-prod D1 measured a never-executed copy of the unrolled b32 LV code at ~0.1 ms/step);
    2 = loops except the n == rows transfer (unrolled as `all_reduce` issues it)."""
    if R == 1:
        if drop_last:
            buf[...] = jnp.zeros(buf.shape, buf.dtype)
        return
    cw = cols // R
    assert loop in (0, 1, 2), loop

    def slot(q, o, m):
        return s["recv"].at[q, pl.ds(o, m), pl.ds(0, cw)]

    def chunk(t, o, m):
        return buf.at[pl.ds(o, m), pl.ds(pl.multiple_of(t * cw, cw), cw)]

    def red(o, m):
        return s["red"].at[pl.ds(o, m), pl.ds(0, cw)]

    def piece(kind, o, m, lp):
        send_all = _send_all_loop if lp else _send_all
        recv_all = _recv_all_loop if lp else _recv_all
        if kind == "rs_start":
            pltpu.make_async_copy(chunk(me, o, m), slot(me, o, m), s["loc"].at[0]).start()
            send_all(R, me, lambda t: chunk(t, o, m), lambda q: slot(q, o, m), s["rs_send"], s["rs_recv"])
        elif kind == "rs_wait":
            pltpu.make_async_copy(chunk(me, o, m), slot(me, o, m), s["loc"].at[0]).wait()
            recv_all(R, me, lambda q: slot(q, o, m), s["rs_recv"])
        elif kind == "ag_start":
            pltpu.make_async_copy(red(o, m), chunk(me, o, m), s["loc"].at[1]).start()
            send_all(R, me, lambda t: red(o, m), lambda q: chunk(q, o, m), s["ag_send"], s["ag_recv"])
        else:  # ag_wait: the all-gather receives, then every send of this piece (reduce-scatter and all-gather)
            pltpu.make_async_copy(red(o, m), chunk(me, o, m), s["loc"].at[1]).wait()
            recv_all(R, me, lambda q: chunk(q, o, m), s["ag_recv"])

            def wsend(step):
                t = lax.rem(me + step, R)
                for src, dst, ss, rs_ in (
                    (chunk(t, o, m), slot(me, o, m), s["rs_send"], s["rs_recv"]),
                    (red(o, m), chunk(me, o, m), s["ag_send"], s["ag_recv"]),
                ):
                    pltpu.make_async_remote_copy(
                        src, dst, ss.at[step], rs_.at[me], device_id=(t,), device_id_type=MESH
                    ).wait_send()

            if lp:
                lax.fori_loop(1, R, lambda step, c: (wsend(step), c)[1], 0)
            else:
                for step in range(1, R):
                    wsend(step)

    def phase(kind):
        def one(n):
            for o, m in ar_pieces(n):
                piece(kind, o, m, loop == 1 or (loop == 2 and n != rows))

        row_switch(n_dyn, options, one)

    phase("rs_start")
    if len(mids) > 0 and mids[0] is not None:
        mids[0]()
    phase("rs_wait")
    acc = s["recv"][0, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
    for q in range(1, R):
        v = s["recv"][q, pl.ds(0, rows), pl.ds(0, cw)].astype(F32)
        if drop_last and q == R - 1:
            v = jnp.zeros_like(v)
        acc = acc + v
    s["red"][pl.ds(0, rows), pl.ds(0, cw)] = acc.astype(BF16)
    phase("ag_start")
    if len(mids) > 1 and mids[1] is not None:
        mids[1]()
    phase("ag_wait")


def entry_barrier(R):
    """The one barrier of the call: every rank signals every rank, then waits for R signals."""
    if R == 1:
        return
    b = pltpu.get_barrier_semaphore()
    for d in range(R):
        pl.semaphore_signal(b, inc=1, device_id=(jnp.int32(d),), device_id_type=MESH)
    pl.semaphore_wait(b, R)


def exchange(R, me, src_fn, dst_fn, send_sem, recv_sem, wait=True, scope=None):
    """Start src_fn(d) -> peer d's dst_fn(me) for every peer d != me; finish() waits the R-1 receives and sends
    (byte-counting semaphores; every exchange carries the same bytes per peer)."""
    if R == 1:
        return None if wait else (lambda: None)
    for d in range(R):

        @pl.when(me != d)
        def _(d=d):
            pltpu.make_async_remote_copy(
                src_ref=src_fn(d),
                dst_ref=dst_fn(me),
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(jnp.int32(d),),
                device_id_type=MESH,
            ).start()

    def finish():
        desc = pltpu.make_async_remote_copy(
            src_ref=src_fn(0),
            dst_ref=dst_fn(0),
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(me,),
            device_id_type=MESH,
        )

        def waits():
            for _ in range(R - 1):
                desc.wait_recv()
            for _ in range(R - 1):
                desc.wait_send()

        if scope is not None:
            with scope("X_wait"):
                waits()
        else:
            waits()

    if not wait:
        return finish
    finish()


def start_all(R, me, src_fn, dst_fn, send_sem, recv_sem):
    """Start src_fn(d) -> peer d's dst_fn(me) for every peer d != me (no wait). L4 C1 / C2: the chunked A2A and the
    lse tiles start pieces of one exchange at different times on one (send, recv) semaphore pair."""
    if R == 1:
        return
    for d in range(R):

        @pl.when(me != d)
        def _(d=d):
            pltpu.make_async_remote_copy(
                src_ref=src_fn(d),
                dst_ref=dst_fn(me),
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=(jnp.int32(d),),
                device_id_type=MESH,
            ).start()


def wait_pieces(R, me, pieces, send_sem, recv_sem, scope=None):
    """Wait every piece started by start_all on (send_sem, recv_sem): pieces = [(src_ref, dst_ref, n_starts)], each
    started n_starts times towards every peer. Byte-counting semaphores: all receives are waited before any send, and
    nothing is read before every wait returned, so the arrival order of pieces does not matter."""
    if R == 1:
        return

    def waits():
        descs = [
            (
                pltpu.make_async_remote_copy(
                    src_ref=s_,
                    dst_ref=d_,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id=(me,),
                    device_id_type=MESH,
                ),
                n,
            )
            for s_, d_, n in pieces
        ]
        for kind in ("recv", "send"):
            for desc, n in descs:
                tot = n * (R - 1)
                fn = desc.wait_recv if kind == "recv" else desc.wait_send
                if isinstance(tot, int) and tot <= 16:
                    for _ in range(tot):
                        fn()
                else:

                    def body(i, c, fn=fn):
                        fn()
                        return c

                    lax.fori_loop(0, tot, body, 0)

    if scope is not None:
        with scope("X_wait"):
            waits()
    else:
        waits()


def spin(n, ref):
    """F-DELAY: a data-dependent busy loop of n iterations on a (8, 128) f32 VMEM ref (csq_proto spin, :538)."""
    ref[...] = lax.fori_loop(
        0, n, lambda it, a: a * 0.9990234375 + 0.0009765625, ref[...]
    )
