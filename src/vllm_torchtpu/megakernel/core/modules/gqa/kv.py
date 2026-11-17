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
"""GQA KV streaming: the per-rank KV worklist and the paged / dense KV ring."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...dma import pool_alias
from ...ops import U32, bswitch
from ...pool.config import cache_exec, resolved_max_pages_per_seq
from .context import GQAEmit
from .params import GQAParams


class KVRing(NamedTuple):
    """KV ring of one layer: `wait(i)` before reading worklist item `i`,
    `done(i)` once it is consumed.

    `stage(i)` is item `i`'s paged stage: a `(2, HD // PK, BLK)` u32 view when
    `u32` (expert pool `KVE` or dense ring `KVS` banks on TPU), else typed
    `(2, HD // PK, PK, BLK)` pages. `kbuf` / `vbuf` are the dense (non-paged)
    rings. `start_next_layer()` (dedicated rings only, else None) starts the
    module's next layer's first items once this layer's are consumed.
    """

    wait: Any
    done: Any
    stage: Any
    u32: bool
    kbuf: Any
    vbuf: Any
    start_next_layer: Any


def build_worklist(ctx: Any, ap: GQAParams) -> None:
    """Build this rank's KV worklist (`wl_b`, `wl_t`, `wl_kp`, `wl_cnt[0]`)."""
    ex, scr, ST, pos_ref = ctx.ex, ctx.scr, ctx.ST, ctx.pos_ref
    NS, T = ex.NS, ex.T
    BLK, TS, SPLIT = ap.BLK, ap.TS, ap.SPLIT
    gi = lax.rem(ctx.me, ap.G) if SPLIT else jnp.int32(0)
    eff_ctx = (
        max(ex.ctx, resolved_max_pages_per_seq(ex) * cache_exec(ex).page_size)
        if cache_exec(ex).paged_kv
        else ex.ctx
    )
    max_nb = -(-eff_ctx // BLK)
    max_p = resolved_max_pages_per_seq(ex)

    n = jnp.int32(0)
    for b in range(NS):
        nb = jnp.minimum(jnp.maximum((pos_ref[b * T] + BLK - 1) // BLK, 0), max_nb)
        t0 = lax.rem(gi + (TS - b % TS), TS) if SPLIT else jnp.int32(0)
        cnt = jnp.maximum(nb - t0 + TS - 1, 0) // TS

        def put(j, carry, b=b, t0=t0, n=n):
            scr["wl_b"][n + j] = jnp.int32(b)
            scr["wl_t"][n + j] = t0 + j * TS
            if cache_exec(ex).paged_kv:
                pt_ref_ = ST["page_table"]
                pb = BLK // cache_exec(ex).page_size
                p0 = (t0 + j * TS) * pb
                k0 = (n + j) * pb
                for p in range(pb):
                    col = jnp.minimum(p0 + p, max_p - 1)
                    scr["wl_kp"][k0 + p] = jnp.clip(
                        pt_ref_[b, col], 0, cache_exec(ex).num_pages - 1
                    )
            return carry

        lax.fori_loop(0, cnt, put, 0)
        n = n + cnt
    scr["wl_cnt"][0] = n


def _kvs_item(ctx: Any, ap: GQAParams, bank: Any, j: Any) -> Any:
    """KV item `j` of dense-ring bank `bank` (`KVS`)."""
    if ctx.ex.interpret:
        return ctx.scr["kvs_ring"].at[bank, j]
    rows = ap.KV_ITEM // 128
    return pool_alias.view(
        ctx.scr["dpool"].at[pl.ds(bank * ctx.stream.sp.BROWS + j * rows, rows)],
        (2, ap.HD // ap.KV_PK, ap.BLK),
        U32,
    )


def _kvs_page(ctx: Any, item: Any, p: int) -> Any:
    """Page `p` of KV item view `item`."""
    ps = cache_exec(ctx.ex).page_size
    if ctx.ex.interpret:
        return item.at[:, :, :, pl.ds(p * ps, ps)]
    return item.at[:, :, pl.ds(p * ps, ps)]


def add_stream_kv(ctx: Any, ap: GQAParams, name: str) -> None:
    """Stream every layer's KV worklist through the dense ring (`KVS`): its
    `ceil(items / KVS_IPT)` bank tiles follow the layer's `att_in` tiles."""
    ex, scr, skip, stream = ctx.ex, ctx.scr, ctx.skip, ctx.stream
    ps = cache_exec(ex).page_size
    P_BLK, IPT = ap.BLK // ps, ap.KVS_IPT
    n_items = scr["wl_cnt"][0]

    def src(r, kp):
        if ex.interpret:
            return r.at[kp]
        return r.at[kp].bitcast(U32).reshape(2, ap.HD // ap.KV_PK, ps)

    def issue(layer, j, bank):
        if "kv" in skip:
            return
        i0 = j * IPT

        # One pool switch per tile, not per item.
        def start_for_ref(r):
            def one(k, carry):
                item = _kvs_item(ctx, ap, bank, k)
                for p in range(P_BLK):
                    kp = scr["wl_kp"][(i0 + k) * P_BLK + p]
                    pltpu.make_async_copy(
                        src(r, kp), _kvs_page(ctx, item, p), scr["kv_sem"].at[bank, k]
                    ).start()
                return carry

            lax.fori_loop(0, jnp.minimum(IPT, n_items - i0), one, 0)

        a = stream.sp.module_ord(layer, name)
        bswitch(a, [lambda r=r: start_for_ref(r) for r in ctx.raw_refs])

    stream.add_runtime_segment("att_in", (n_items + IPT - 1) // IPT, issue)


def _stream_kv_ring(cx: GQAEmit) -> KVRing:
    """The layer's `KVS` items: bank tiles from `stream.runtime_start()`."""
    ctx, ap, stream, n_items = cx.ctx, cx.ap, cx.stream, cx.n_items
    IPT, P_BLK = ap.KVS_IPT, ap.BLK // cache_exec(cx.ex).page_size
    g0 = stream.runtime_start()

    def bank(i):
        return lax.rem(g0 + i // IPT, ap.NBD)

    def wait(i):
        if "kv" in cx.skip:
            return
        b, k = bank(i), lax.rem(i, IPT)
        item, sem = _kvs_item(ctx, ap, b, k), cx.ksem.at[b, k]
        if cx.interpret:
            for p in range(P_BLK):
                dst = _kvs_page(ctx, item, p)
                pltpu.make_async_copy(ctx.raw_refs[0].at[0], dst, sem).wait()
        else:
            pltpu.make_async_copy(item, item, sem).wait()

    def done(i):
        last = jnp.logical_or(lax.rem(i, IPT) == IPT - 1, i == n_items - 1)
        pl.when(last)(lambda: stream.dgive(g0 + i // IPT))

    return KVRing(
        wait,
        done,
        lambda i: _kvs_item(ctx, ap, bank(i), lax.rem(i, IPT)),
        not cx.interpret,
        None,
        None,
        None,
    )


def make_kv_ring(cx: GQAEmit) -> KVRing:
    """Define the layer's KV ring and start its first `KR` items."""
    if cx.ap.KVS_IPT:
        return _stream_kv_ring(cx)
    ctx, ex, ap, scr, ST, skip = cx.ctx, cx.ex, cx.ap, cx.scr, cx.ST, cx.skip
    stream, interpret, a, ksem = cx.stream, cx.interpret, cx.a, cx.ksem
    n_items, KR = cx.n_items, cx.KR
    NKV, KVE, KV_ITEM, EBROWS, BLK = ap.NKV, ap.KVE, ap.KV_ITEM, ap.EBROWS, ap.BLK
    HD = ap.HD
    kv_slot: Any = None
    kvbuf: Any = None
    kbuf: Any = None
    vbuf: Any = None
    if cache_exec(ex).paged_kv:
        raw_refs = ctx.raw_refs
        P_BLK = BLK // cache_exec(ex).page_size

        if KVE:
            epool_ref = stream.borrow_epool()
            if interpret:

                def kv_slot(s):
                    return epool_ref.at[s]

                def kv_src(r, kp):
                    return r.at[kp]

                def kv_dst(s, p):
                    return kv_slot(s).at[
                        :,
                        :,
                        :,
                        pl.ds(p * cache_exec(ex).page_size, cache_exec(ex).page_size),
                    ]

            else:

                def kv_slot(s):
                    return pool_alias.view(
                        epool_ref.at[pl.ds(s * EBROWS, KV_ITEM // 128)],
                        (2, HD // 4, BLK),
                        U32,
                    )

                def kv_src(r, kp):
                    return (
                        r.at[kp]
                        .bitcast(U32)
                        .reshape(2, HD // 4, cache_exec(ex).page_size)
                    )

                def kv_dst(s, p):
                    return kv_slot(s).at[
                        :,
                        :,
                        pl.ds(p * cache_exec(ex).page_size, cache_exec(ex).page_size),
                    ]

        else:
            kvbuf = scr["kvbuf"]

            def kv_src(r, kp):
                return r.at[kp]

            def kv_dst(s, p):
                return kvbuf.at[
                    s,
                    :,
                    :,
                    :,
                    pl.ds(p * cache_exec(ex).page_size, cache_exec(ex).page_size),
                ]

        def kv_stage(s):
            return kv_slot(s) if KVE else kvbuf.at[s]

        def kv_start(i, la=a):
            @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
            def _():
                s = i % KR
                k0 = i * P_BLK
                sem = ksem.at[s]
                kp_tab = scr["wl_kp"]

                def start_for_ref(r):
                    for p in range(P_BLK):
                        kp = kp_tab[k0 + p]
                        pltpu.make_async_copy(kv_src(r, kp), kv_dst(s, p), sem).start()

                bswitch(la, [lambda r=r: start_for_ref(r) for r in raw_refs])

        def kv_wait(i):
            if "kv" not in skip:
                s = i % KR
                if interpret:
                    for p in range(P_BLK):
                        pltpu.make_async_copy(
                            raw_refs[0].at[0], kv_dst(s, p), ksem.at[s]
                        ).wait()
                else:
                    st_ = kv_stage(s)
                    pltpu.make_async_copy(st_, st_, ksem.at[s]).wait()

        n_layers = len(raw_refs)  # one pool per layer of the module
    else:
        kc, vc = ST["kc"], ST["vc"]
        kbuf, vbuf = scr["kbuf"], scr["vbuf"]

        def kv_item(i, b, t, la=a):
            s = i % NKV
            t0 = pl.multiple_of(t * BLK, BLK)
            return (
                pltpu.make_async_copy(
                    kc.at[la, b, pl.ds(t0, BLK)], kbuf.at[s], ksem.at[0, s]
                ),
                pltpu.make_async_copy(
                    vc.at[la, b, pl.ds(t0, BLK)], vbuf.at[s], ksem.at[1, s]
                ),
            )

        def kv_start(i, la=a):
            @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
            def _():
                for d in kv_item(i, scr["wl_b"][i], scr["wl_t"][i], la):
                    d.start()

        def kv_wait(i):
            if "kv" not in skip:
                zero = jnp.int32(0)
                for d in kv_item(i, zero, zero):
                    d.wait()

        n_layers = kc.shape[0]

    def start_layer(la):
        lax.fori_loop(0, KR, lambda i, _: (kv_start(i, la), 0)[1], 0)

    def done(i):
        kv_start(i + KR)

    def stage(i):
        s = i % KR
        return kv_slot(s) if KVE else kvbuf.at[s]

    u32 = KVE and not interpret
    if KVE:  # the expert pool is busy outside attention
        start_layer(a)
        return KVRing(kv_wait, done, stage, u32, kbuf, vbuf, None)
    # A dedicated ring is idle between layers, so each layer starts the next
    # one's first items after its own attention: started at that layer's start
    # they queue behind ~NBD in-flight weight tiles (6 x 8 MiB, 15 us at
    # 3.2 TB/s) and stall it.
    pl.when(a == 0)(lambda: start_layer(a))
    return KVRing(
        kv_wait,
        done,
        stage,
        u32,
        kbuf,
        vbuf,
        lambda: pl.when(a + 1 < n_layers)(lambda: start_layer(a + 1)),
    )
