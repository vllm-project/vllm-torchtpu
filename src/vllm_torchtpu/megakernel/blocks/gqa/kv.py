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

from ... import pool_alias
from ...ops import U32, bswitch
from ...pool.config import cache_exec, resolved_max_pages_per_seq
from .context import GQAEmit
from .params import GQAParams


class KVRing(NamedTuple):
    """KV ring of one layer: `start(i)` / `wait(i)` stream worklist item `i`.

    `slot(s)` is ring stage `s` when it lives in the expert pool (`KVE`);
    `kvbuf` (paged) or `kbuf` / `vbuf` (dense) are the dedicated VMEM rings.
    """

    start: Any
    wait: Any
    slot: Any
    kvbuf: Any
    kbuf: Any
    vbuf: Any


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
        t0 = (
            gi
            if ap.DCP_SHARDED
            else (lax.rem(gi + (TS - b % TS), TS) if SPLIT else jnp.int32(0))
        )
        cnt = jnp.maximum(nb - t0 + TS - 1, 0) // TS

        def put(j, carry, b=b, t0=t0, n=n):
            scr["wl_b"][n + j] = jnp.int32(b)
            scr["wl_t"][n + j] = t0 + j * TS
            if cache_exec(ex).paged_kv:
                pt_ref_ = ST["page_table"]
                pb = BLK // cache_exec(ex).page_size
                p0 = (j if ap.DCP_SHARDED else (t0 + j * TS)) * pb
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


def make_kv_ring(cx: GQAEmit) -> KVRing:
    """Define the layer's KV ring and start its first `KR` items."""
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

        def kv_start(i):
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

                bswitch(a, [lambda r=r: start_for_ref(r) for r in raw_refs])

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

        lax.fori_loop(0, KR, lambda i, _: (kv_start(i), 0)[1], 0)
    else:
        kc, vc = ST["kc"], ST["vc"]
        kbuf, vbuf = scr["kbuf"], scr["vbuf"]

        def kv_item(i, b, t):
            s = i % NKV
            t0 = pl.multiple_of(t * BLK, BLK)
            return (
                pltpu.make_async_copy(
                    kc.at[a, b, pl.ds(t0, BLK)], kbuf.at[s], ksem.at[0, s]
                ),
                pltpu.make_async_copy(
                    vc.at[a, b, pl.ds(t0, BLK)], vbuf.at[s], ksem.at[1, s]
                ),
            )

        def kv_start(i):
            @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
            def _():
                for d in kv_item(i, scr["wl_b"][i], scr["wl_t"][i]):
                    d.start()

        def kv_wait(i):
            if "kv" not in skip:
                zero = jnp.int32(0)
                for d in kv_item(i, zero, zero):
                    d.wait()

        for i in range(NKV):
            kv_start(jnp.int32(i))
    return KVRing(kv_start, kv_wait, kv_slot, kvbuf, kbuf, vbuf)
