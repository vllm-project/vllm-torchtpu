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
"""GQA current-token update: fold this step's K/V in, normalize, apply the gate."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ...ops import F32, I32, rnd
from .context import GQAEmit


def att_new(cx: GQAEmit) -> None:
    """Attend to the step's own K/V rows and write the gated output to `mix`."""
    ex, ap, scr, sc, gi, am, al = cx.ex, cx.ap, cx.scr, cx.sc, cx.gi, cx.am, cx.al
    aacc = cx.aacc
    B, T, NS, VER = ex.B, ex.T, ex.NS, ex.VER
    G, nh, HD, QR, SPLIT, PACK, QV = ap.G, ap.nh, ap.HD, ap.QR, ap.SPLIT, ap.PACK, ap.QV
    GQ, NHH, SR, QG, MLW = ap.GQ, ap.NHH, ap.SR, ap.QG, ap.MLW
    with sc("att_new"):
        if GQ:
            pri = scr["pri"]
            mlt = [pri[g_idx, NS] for g_idx in range(G)]
            row_q = lax.broadcasted_iota(I32, (QG, 1), 0)
            for sq in range(NS):
                f = sq * MLW
                parts = [pri[g_idx, sq] for g_idx in range(G)]
                mg = [x[:, f : f + 1] for x in mlt]
                lg = [x[:, 128 + f : 128 + f + 1] for x in mlt]
                m = mg[0]
                for x in mg[1:]:
                    m = jnp.maximum(m, x)
                wg = [jnp.exp(x - m) for x in mg]
                lsum_gq: Any = sum(w * x for w, x in zip(wg, lg))
                asum_gq: Any = sum(w * x for w, x in zip(wg, parts))
                r0 = sq * QG
                qblk = jnp.concatenate(
                    [scr["qg"][0, j, r0 : r0 + QG, :] for j in range(NHH)],
                    axis=-1,
                )
                m_old, l_old, a_old = m, lsum_gq, asum_gq
                for j in range(T):
                    bj = sq * T + j
                    s_new = jnp.sum(
                        qblk * scr["knew"][bj : bj + 1, :],
                        axis=-1,
                        keepdims=True,
                    )
                    m_new = jnp.maximum(m_old, s_new)
                    alpha = jnp.exp(m_old - m_new)
                    p_new = jnp.exp(s_new - m_new)
                    l_new = l_old * alpha + p_new
                    a_new = a_old * alpha + p_new * scr["vnew"][bj : bj + 1, :]
                    if j == 0:
                        m_old, l_old, a_old = m_new, l_new, a_new
                    else:
                        upd = row_q >= j * nh
                        m_old = jnp.where(upd, m_new, m_old)
                        l_old = jnp.where(upd, l_new, l_old)
                        a_old = jnp.where(upd, a_new, a_old)
                ob = rnd(a_old / l_old)
                for j in range(NHH):
                    scr["ost"][j, r0 : r0 + QG, :] = ob[:, j * 128 : (j + 1) * 128]
        else:
            NR = nh if PACK else 8
            for b in range(B):
                sq, tb = divmod(b, T)
                q0 = tb * QR
                m_old = am[sq, q0 : q0 + NR][:, 0:1]
                l_old = al[sq, q0 : q0 + NR][:, 0:1]
                a_old = aacc[sq, q0 : q0 + NR]
                if QV:
                    qf = jnp.concatenate(
                        [scr["qg"][gi, j, b * SR : b * SR + NR, :] for j in range(NHH)],
                        axis=-1,
                    )
                elif PACK:
                    t_, o_ = ap.qrow(b)
                    qf = scr["qg"][gi, t_, o_ : o_ + nh]
                else:
                    qf = scr["qg"][gi, b] if SPLIT or VER else scr["qb"][b].astype(F32)
                if VER:
                    for j in range(tb + 1):
                        bj = sq * T + j
                        s_new = jnp.sum(
                            qf * scr["knew"][bj : bj + 1, :],
                            axis=-1,
                            keepdims=True,
                        )
                        m_new = jnp.maximum(m_old, s_new)
                        alpha = jnp.exp(m_old - m_new)
                        p_new = jnp.exp(s_new - m_new)
                        l_old = l_old * alpha + p_new
                        a_old = a_old * alpha + p_new * scr["vnew"][bj : bj + 1, :]
                        m_old = m_new
                    outb = a_old / l_old
                else:
                    s_new = jnp.sum(
                        qf * scr["knew"][b : b + 1, :],
                        axis=-1,
                        keepdims=True,
                    )
                    m_new = jnp.maximum(m_old, s_new)
                    alpha = jnp.exp(m_old - m_new)
                    p_new = jnp.exp(s_new - m_new)
                    l_tot = l_old * alpha + p_new
                    outb = (a_old * alpha + p_new * scr["vnew"][b : b + 1, :]) / l_tot
                if QV:
                    ob = rnd(outb[:nh])
                    for j in range(NHH):
                        scr["ost"][j, b * SR : b * SR + nh, :] = ob[
                            :, j * 128 : (j + 1) * 128
                        ]
                    continue
                for h in range(nh):
                    oh = rnd(outb[h : h + 1, :])
                    gh = scr["gate"][b : b + 1, h * HD : (h + 1) * HD]
                    if not SPLIT:
                        gh = rnd(jax.nn.sigmoid(gh))
                    scr["mix"][b : b + 1, h * HD : (h + 1) * HD] = rnd(oh * gh)
        if QV:
            for h in range(nh):
                oh = jnp.concatenate(
                    [
                        scr["ost"][jnp.int32(j), pl.ds(h, B, stride=SR), :]
                        for j in range(NHH)
                    ],
                    axis=-1,
                )
                gh = scr["gate"][:B, h * HD : (h + 1) * HD]
                if not SPLIT:
                    gh = rnd(jax.nn.sigmoid(gh))
                scr["mix"][:B, h * HD : (h + 1) * HD] = rnd(oh * gh)
