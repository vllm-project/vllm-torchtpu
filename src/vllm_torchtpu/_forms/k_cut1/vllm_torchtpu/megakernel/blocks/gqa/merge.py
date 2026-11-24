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
"""GQA sequence-split (`SPLIT`) partial-softmax exchange and merge."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ...fused_io_hook import fused_io
from ...ops import I32, rnd
from ...pool.config import cache_exec
from .context import GQAEmit


def gq_merge_start(cx: GQAEmit) -> list[Any]:
    """Send packed-query (`GQ`) partial outputs and `(m | l)` to the group."""
    ex, ap, scr, sc, comm, gi, am = cx.ex, cx.ap, cx.scr, cx.sc, cx.comm, cx.gi, cx.am
    al, aacc = cx.al, cx.aacc
    NS = ex.NS
    G, QG, MLW = ap.G, ap.QG, ap.MLW
    psend, pri = scr["psend"], scr["pri"]
    with sc("att_merge"):
        lane = lax.broadcasted_iota(I32, (QG, 128), 1)
        for g_idx in range(G):
            rs = slice(g_idx * QG, (g_idx + 1) * QG)
            for s_ in range(NS):
                psend[g_idx, s_] = aacc[s_, rs]
            xm, xl = am[0, rs], al[0, rs]
            for s_ in range(1, NS):
                hi = lane >= s_ * MLW
                xm = jnp.where(hi, am[s_, rs], xm)
                xl = jnp.where(hi, al[s_, rs], xl)
            psend[g_idx, NS] = jnp.concatenate((xm, xl), axis=-1)
        pri[gi] = psend[gi]
        mx = comm.group_copies(
            lambda off: psend.at[gi ^ off],
            pri.at[gi],
            scr["sp_ssem"],
            scr["sp_rsem"],
            G,
        )
        for d in mx:
            d.start()
    return mx


def slab_merge_start(cx: GQAEmit) -> list[Any]:
    """Send per-row partial outputs and `(m | l)` slabs to the group."""
    ex, ap, scr, sc, comm, gi, am = cx.ex, cx.ap, cx.scr, cx.sc, cx.comm, cx.gi, cx.am
    al, aacc = cx.al, cx.aacc
    B, T = ex.B, ex.T
    G, nh, HD, QR, PW = ap.G, ap.nh, ap.HD, ap.QR, ap.PW
    with sc("att_merge"):
        pso, pri = scr["pso"], scr["pri"]
        lane_pw = lax.broadcasted_iota(I32, (nh, 128), 1)
        for b in range(B):
            t_, o_ = ap.qrow(b)
            sq, q0 = b // T, (b % T) * QR
            mb, lb = am[sq, q0 : q0 + QR], al[sq, q0 : q0 + QR]
            ab = aacc[sq, q0 : q0 + QR]
            for g_idx in range(G):
                r0 = g_idx * nh
                pso[g_idx, t_, o_ : o_ + nh, 0:HD] = ab[r0 : r0 + nh]
                pso[g_idx, t_, o_ : o_ + nh, HD:PW] = jnp.where(
                    lane_pw < 64, mb[r0 : r0 + nh], lb[r0 : r0 + nh]
                )
        pri[gi] = pso[gi]
        mx = comm.group_copies(
            lambda off: pso.at[gi ^ off],
            pri.at[gi],
            scr["sp_ssem"],
            scr["sp_rsem"],
            G,
        )
        for d in mx:
            d.start()
    return mx


def split_merge(cx: GQAEmit) -> None:
    """Exchange partial softmax state; merge it into `am` / `al` / `aacc`."""
    ctx, ex, ap, scr, sc, a, am = cx.ctx, cx.ex, cx.ap, cx.scr, cx.sc, cx.a, cx.am
    al, aacc = cx.al, cx.aacc
    B, T = ex.B, ex.T
    G, nh, HD, QR, PACK, PB, NBP = ap.G, ap.nh, ap.HD, ap.QR, ap.PACK, ap.PB, ap.NBP
    GQ = ap.GQ
    mx = gq_merge_start(cx) if GQ else slab_merge_start(cx)
    pri = scr["pri"]
    if cache_exec(ex).FUSED:
        with sc("att_kvwr"):
            fused_io().gqa_fused_kv_rmw(ctx, ap, a)
    with sc("att_gate"):
        scr["gate"][0:B, :] = rnd(jax.nn.sigmoid(scr["gate"][0:B, :]))
    with sc("att_merge"):
        for d in mx:
            d.wait_recv()
        for d in mx:
            d.wait_send()
        for t in range(NBP if not GQ else 0):
            parts = [pri[g_idx, t] for g_idx in range(G)]
            mg = [x[:, HD : HD + 1] for x in parts]
            m = mg[0]
            for x in mg[1:]:
                m = jnp.maximum(m, x)
            wg = [jnp.exp(x - m) for x in mg]
            lsum: Any = sum(w * x[:, HD + 64 : HD + 65] for w, x in zip(wg, parts))
            asum: Any = sum(w * x[:, :HD] for w, x in zip(wg, parts))
            if not PACK:
                sq, q0 = t // T, (t % T) * QR
                am[sq, q0 : q0 + 8] = jnp.broadcast_to(m, (8, 128))
                al[sq, q0 : q0 + 8] = jnp.broadcast_to(lsum, (8, 128))
                aacc[sq, q0 : q0 + 8] = asum
                continue
            for b in range(t * PB, min(B, (t + 1) * PB)):
                o_ = ap.qrow(b)[1]
                sq, q0 = b // T, (b % T) * QR
                am[sq, q0 : q0 + nh] = jnp.broadcast_to(m[o_ : o_ + nh], (nh, 128))
                al[sq, q0 : q0 + nh] = jnp.broadcast_to(lsum[o_ : o_ + nh], (nh, 128))
                aacc[sq, q0 : q0 + nh] = asum[o_ : o_ + nh]
