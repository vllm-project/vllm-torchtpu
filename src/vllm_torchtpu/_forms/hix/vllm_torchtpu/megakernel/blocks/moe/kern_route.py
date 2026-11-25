# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-KERN K3 RC-VEC (VLLM_TPU_MK_KERN_ROUTE_VEC=1, default off): the MoE hit list from an order-preserving vector pass.

Flag off, `route()` copies the top-k picks `hitv` (VMEM -> SMEM, synchronous: route_dma) and `dedup_local` walks all B x TOPK picks on the
scalar unit (route_dedup), appending this rank's expert at its first LIVE pick in (row b, k) order: 38.0 + 28.1 us per 16-row step at every
live count (T62z j-0bb8a641 arm t, M). With the flag the same list is computed on the vector unit from `hitv` before any copy:

  F[e]   = the smallest pick index b * TOPK + k over live rows b, k < TOPK, whose pick is local expert e (this rank's e - me * ER)
  hit[e] = F[e] is finite;  rank[e] = #{e' hit : F[e'] < F[e]}  (picks are distinct indices, so ranks are distinct and dense)
  elist[rank[e]] = e + me * ER for every hit e;  n_hit = #hit

which is `dedup_local`'s list exactly: an expert enters at its first live pick and keeps that order; a pad row (X-K-LIVE / the row-mask
debug input) adds nothing and does not mark the expert, so a later live pick still counts; another rank's expert never enters. The list
row and n_hit are written into `hitv` rows 0 / 1 and copied to SMEM (`hits` rows 0..7) by ONE copy (the route_dma slot), and n_hit
entries are moved into `elist` (about 10-16 scalar stores against 160 picks). Entries at or past n_hit are never read (every reader is
guarded by n_hit or st[0] = n_hit * TE: egive, route_fetch, the expert loops). `seen` is not used on this path.
All integer arithmetic on int32 vectors: exact. Refused with redundant copies (ECOPY), route_vec (VEC), PREGATE, moe_tp or ER > 128.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...ops import I32
from .context import MoEEmit

BIG = 1 << 12  # > every pick index (B * TOPK <= 64 * 10 at the buckets this runs at)
KERN_ROUTE_TRACES: list = []  # (B, MR, ER, TOPK) per traced flag-on route (tests / engagement)


def kern_route_vec(m: MoEEmit) -> bool:
    from vllm_torchtpu.megakernel_runtime.qwen38_kern import route_vec_on

    if not route_vec_on():
        return False
    mp = m.mp
    if mp.ECOPY or mp.route_vec or mp.PG or mp.moe_tp or mp.ER > 128 or m.ex.B * mp.TOPK >= BIG or m.ex.MR < 8:
        raise ValueError(
            "q38 HIX-KERN VLLM_TPU_MK_KERN_ROUTE_VEC=1 needs EP routing without ECOPY / route_vec / PREGATE, ER <= 128, MR >= 8 "
            f"(ECOPY={mp.ECOPY} route_vec={mp.route_vec} PG={mp.PG} moe_tp={mp.moe_tp} ER={mp.ER} B={m.ex.B} MR={m.ex.MR})")
    return True


def vec_list(ids: Any, live: Any, me: Any, ER: int, TOPK: int) -> tuple[Any, Any]:
    """(elist row [1, 128] int32, n_hit [1, 128] int32 broadcast) from the picks `ids` [MR, 128] (lanes < TOPK valid) and the row mask
    `live` [MR, 128] int32, LANE-DENSE (row b live <=> live[b, :] != 0). Pure jnp: also the reference the interpret check compares
    against the scalar loop. K3-MOSAIC fix: the mask was a [MR, 1] bool, whose broadcast Mosaic lowers through an i1 -> i8 extsi on a
    vector<16x1xi1> with a padded (8, 128) layout and refuses ('Invalid input layout ... arith.extsi', HIX-STACK2 dev4 j-4c543d9a,
    deviceless tpu7x at b8 / b16); every mask here is now produced by full-tile int32 compares."""
    MR = ids.shape[0]
    row = lax.broadcasted_iota(I32, (MR, 128), 0)
    lane = lax.broadcasted_iota(I32, (MR, 128), 1)
    loc = ids - me * ER
    ok = jnp.logical_and(jnp.logical_and(live != 0, lane < TOPK), jnp.logical_and(loc >= 0, loc < ER))
    key = jnp.where(ok, row * TOPK + lane, BIG)
    acc = jnp.full((MR, 128), BIG, I32)
    for k in range(TOPK):  # column k of loc / key by a masked lane sum (no lane slicing), compared with the expert lanes
        lk = jnp.sum(jnp.where(lane == k, loc, 0), axis=1, keepdims=True)
        kk = jnp.sum(jnp.where(lane == k, key, 0), axis=1, keepdims=True)
        acc = jnp.minimum(acc, jnp.where(lane == lk, kk, BIG))
    F = jnp.min(acc, axis=0, keepdims=True)  # [1, 128]: first pick index of local expert e (lane e); BIG = no live pick
    A = jnp.broadcast_to(F, (128, 128))      # A[i, j] = F[j]
    AT = jnp.transpose(A)                    # AT[i, j] = F[i]
    rank = jnp.sum(jnp.where(jnp.logical_and(A < BIG, A < AT), 1, 0), axis=1, keepdims=True)  # [128, 1]
    rr = lax.broadcasted_iota(I32, (128, 128), 1)
    gid = lax.broadcasted_iota(I32, (128, 128), 0) + me * ER
    el = jnp.sum(jnp.where(jnp.logical_and(AT < BIG, rank == rr), gid, 0), axis=0, keepdims=True)
    n = jnp.sum(jnp.where(F < BIG, 1, 0), axis=1, keepdims=True)  # [1, 1]
    return el.astype(I32), jnp.broadcast_to(n, (1, 128)).astype(I32)


def kern_route_list(m: MoEEmit, row_on: Any) -> Any:
    """The flag-on replacement of route_dma + dedup_local in `route()`; returns n_hit (scalar)."""
    ex, mp, scr, sc, me = m.ex, m.mp, m.scr, m.sc, m.me
    B, MR, ER, TOPK = ex.B, ex.MR, mp.ER, mp.TOPK
    KERN_ROUTE_TRACES.append((B, MR, ER, TOPK))
    with sc("route_dedup", 2):
        r2 = lax.broadcasted_iota(I32, (MR, 128), 0)
        live = jnp.where(r2 < B, 1, 0).astype(I32)  # K3-MOSAIC: lane-dense int32 row mask (no [MR, 1] bool vector)
        if ex.live_rows or m.dbg.row_mask:
            for b in range(B):
                on_b = jnp.where(row_on(m, b), 1, 0).astype(I32)  # scalar
                live = jnp.where(r2 == b, live * on_b, live)
        el, n = vec_list(scr["hitv"][...], live, me, ER, TOPK)
        scr["hitv"][0:1, :] = el
        scr["hitv"][1:2, :] = n
    with sc("route_dma", 2):
        cp = pltpu.make_async_copy(scr["hitv"].at[pl.ds(0, 8)], scr["hits"].at[pl.ds(0, 8)], scr["misc_sem"])
        cp.start()
        cp.wait()
    n_hit = scr["hits"][1, 0]

    def mv(k: Any, c: Any) -> Any:
        scr["elist"][k] = scr["hits"][0, k]
        return c

    lax.fori_loop(0, n_hit, mv, 0)
    return n_hit
