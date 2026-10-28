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
"""MoE routed experts: rolled / unrolled, `PIPE` and `PIPE_EP` schedules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ...ops import BF16, F32, rnd
from .context import MoEEmit
from .ring import egive, egive_group, es_wait, etake


def gate_up(m: MoEEmit, i, w13):
    mp, scr, stream = m.mp, m.scr, m.stream
    IR, NSS = mp.IR, mp.NSS
    h = stream.mm(scr["xb"][...], w13[...])
    if "es" in m.W:
        h = h * scr["es"][i % NSS, :, : 2 * IR]
    return rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)


def down(m: MoEEmit, i, mid, w2):
    ex, lane_e8, mp, scr, stream = m.ex, m.lane_e8, m.mp, m.scr, m.stream
    MR = ex.MR
    CH, H, IR, NSS = mp.CH, mp.H, mp.IR, mp.NSS
    e, s, part = scr["elist"][i], i % NSS, scr["part"]
    has_es = "es" in m.W
    wcol = jnp.sum(jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True)
    for j in range(H // CH):
        lo = 2 * IR + j * CH
        y = stream.mm(mid, w2[:, j * CH : (j + 1) * CH])[0:MR]
        y = rnd(y * scr["es"][s, :, lo : lo + CH]) if has_es else rnd(y)
        part[0:MR, j * CH : (j + 1) * CH] += wcol * y


def expert(m: MoEEmit, l_idx, i, carry):
    ex, lane_e8, mp, scr, stream, tk = m.ex, m.lane_e8, m.mp, m.scr, m.stream, m.tk
    MB, MR = ex.MB, ex.MR
    CH, H, IR, NSS, T13, T2, TE = mp.CH, mp.H, mp.IR, mp.NSS, mp.T13, mp.T2, mp.TE
    e = scr["elist"][i]
    s = i % NSS
    base = i * TE
    acc, xb, xb2, part = scr["acc"], scr["xb"], scr["xb2"], scr["part"]
    DIRECT0 = mp.r2_direct_store
    if T13 == 1:
        h = stream.mm(xb[...], etake(m, "e13", base)[...])
        egive(m, base, l_idx)
    elif not mp.unroll:
        # Default path: rolled tile loop with per-tile refill.
        acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)

        def b13(t, cc):
            u = base + t
            w = etake(m, "e13", u)
            k0 = pl.multiple_of(t * tk["e13"], tk["e13"])
            acc[:, : 2 * IR] += stream.mm(xb[:, pl.ds(k0, tk["e13"])], w[...])
            egive(m, u, l_idx)
            return cc

        lax.fori_loop(0, T13, b13, 0)
        h = acc[:, : 2 * IR]
    else:
        TK13 = tk["e13"]
        if not DIRECT0:
            acc[:, : 2 * IR] = jnp.zeros((MB, 2 * IR), F32)
        for t in range(T13):
            u = base + t
            w = etake(m, "e13", u)
            m13 = stream.mm(xb[:, t * TK13 : (t + 1) * TK13], w[...])
            if DIRECT0 and t == 0:
                acc[:, : 2 * IR] = m13
            else:
                acc[:, : 2 * IR] += m13
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(u - 1, t - 1), (u, t)], l_idx)
                elif t == T13 - 1:
                    egive(m, u, l_idx, t)
            else:
                egive(m, u, l_idx, t)
        h = acc[:, : 2 * IR]
    es_wait(m, i)
    has_es = "es" in m.W
    if has_es:
        h = h * scr["es"][s, :, : 2 * IR]
    mid_ = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:]))
    xb2[:, :IR] = mid_.astype(BF16)
    wcol = jnp.sum(jnp.where(lane_e8 == e, scr["wn"][...], 0.0), axis=1, keepdims=True)
    u2 = base + T13
    if T2 == 1:
        w2 = etake(m, "e2", u2)[...]
        xm = xb2[:, :IR]
        for j in range(H // CH):
            lo = 2 * IR + j * CH
            y = stream.mm(xm, w2[:, j * CH : (j + 1) * CH])[0:MR]
            y = rnd(y * scr["es"][s, :, lo : lo + CH]) if has_es else rnd(y)
            part[0:MR, j * CH : (j + 1) * CH] += wcol * y
        egive(m, u2, l_idx)
    elif not mp.unroll:
        acc[:, :H] = jnp.zeros((MB, H), F32)

        def b2(t, cc):
            u = u2 + t
            w = etake(m, "e2", u)
            k0 = pl.multiple_of(t * tk["e2"], tk["e2"])
            acc[:, :H] += stream.mm(xb2[:, pl.ds(k0, tk["e2"])], w[...])
            egive(m, u, l_idx)
            return cc

        lax.fori_loop(0, T2, b2, 0)
        y = (
            rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
            if has_es
            else rnd(acc[0:MR, :H])
        )
        part[0:MR, :] += wcol * y
    else:
        TK2 = tk["e2"]
        if not DIRECT0:
            acc[:, :H] = jnp.zeros((MB, H), F32)
        for t in range(T2):
            u = u2 + t
            w = etake(m, "e2", u)
            m2 = stream.mm(xb2[:, t * TK2 : (t + 1) * TK2], w[...])
            if DIRECT0 and t == 0:
                acc[:, :H] = m2
            else:
                acc[:, :H] += m2
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(u - 1, T13 + t - 1), (u, T13 + t)], l_idx)
                elif t == T2 - 1:
                    egive(m, u, l_idx, T13 + t)
            else:
                egive(m, u, l_idx, T13 + t)
        y = (
            rnd(acc[0:MR, :H] * scr["es"][s, :, 2 * IR :])
            if has_es
            else rnd(acc[0:MR, :H])
        )
        part[0:MR, :] += wcol * y
    return carry


def experts_piped(m: MoEEmit, l_idx, n_hit):
    mp, scr = m.mp, m.scr
    IR, T13, TE = mp.IR, mp.T13, mp.TE
    xb2 = scr["xb2"]

    @pl.when(n_hit > 0)
    def _():
        w13 = etake(m, "e13", 0)
        es_wait(m, 0)
        xb2[:, :IR] = gate_up(m, 0, w13)
        egive(m, 0, l_idx, 0)

        def body(i, carry):
            u13, u2 = (i + 1) * TE, i * TE + T13
            w13 = etake(m, "e13", u13)
            es_wait(m, i + 1)
            w2 = etake(m, "e2", u2)[...]
            mid = xb2[:, :IR]
            nxt = gate_up(m, i + 1, w13)
            down(m, i, mid, w2)
            xb2[:, :IR] = nxt
            egive(m, u13, l_idx, 0)
            egive(m, u2, l_idx, T13)
            return carry

        lax.fori_loop(0, n_hit - 1, body, 0)
        last = n_hit - 1
        u2 = last * TE + T13
        down(m, last, xb2[:, :IR], etake(m, "e2", u2)[...])
        egive(m, u2, l_idx, T13)


def experts_piped_ep(m: MoEEmit, l_idx, n_hit):
    ex, lane_e8, mp, scr, stream, tk = m.ex, m.lane_e8, m.mp, m.scr, m.stream, m.tk
    MB, MR = ex.MB, ex.MR
    H, IR, NSS, T13, T2, TE = mp.H, mp.IR, mp.NSS, mp.T13, mp.T2, mp.TE

    @pl.when(n_hit > 0)
    def _():
        acc, acc13 = scr["acc"], scr["acc13"]
        xb, xb2, part = scr["xb"], scr["xb2"], scr["part"]
        BN = 2 * IR
        TK13, TK2 = tk["e13"], tk["e2"]
        DIRECT0 = mp.r2_direct_store

        if not DIRECT0:
            acc13[...] = jnp.zeros((MB, BN), F32)
        for t in range(T13):
            w13 = etake(m, "e13", t)
            k0 = t * TK13
            mm13 = stream.mm(xb[:, k0 : k0 + TK13], w13[...])
            if DIRECT0 and t == 0:
                acc13[:, :BN] = mm13
            else:
                acc13[:, :BN] += mm13
            if mp.group_rel:
                if t % 2 == 1:
                    egive_group(m, [(t - 1, t - 1), (t, t)], l_idx)
                elif t == T13 - 1:
                    egive(m, t, l_idx, t)
            else:
                egive(m, t, l_idx, t)
        es_wait(m, 0)
        has_es = "es" in m.W
        h0 = acc13[:, :BN] * scr["es"][0, :, :BN] if has_es else acc13[:, :BN]
        mid0 = rnd(rnd(jax.nn.silu(rnd(h0[:, :IR]))) * rnd(h0[:, IR:]))
        xb2[:, :IR] = mid0.astype(BF16)

        def body_ep(i, _):
            e = scr["elist"][i]
            s_cur = i % NSS
            s_nxt = (i + 1) % NSS
            base_w2 = i * TE + T13
            base_nxt13 = (i + 1) * TE

            if not DIRECT0:
                acc[:, :H] = jnp.zeros((MB, H), F32)
                acc13[...] = jnp.zeros((MB, BN), F32)

            for s_step in range(T2):
                u2 = base_w2 + s_step
                t0 = 2 * s_step
                t1 = 2 * s_step + 1
                u13_0 = base_nxt13 + t0
                u13_1 = base_nxt13 + t1

                w2 = etake(m, "e2", u2)
                w13_0 = etake(m, "e13", u13_0)
                k2 = s_step * TK2
                k13_0 = t0 * TK13
                mm2 = stream.mm(xb2[:, k2 : k2 + TK2], w2[...])
                mm13_0 = stream.mm(xb[:, k13_0 : k13_0 + TK13], w13_0[...])
                if DIRECT0 and s_step == 0:
                    acc[:, :H] = mm2
                    acc13[:, :BN] = mm13_0
                else:
                    acc[:, :H] += mm2
                    acc13[:, :BN] += mm13_0
                if mp.group_rel:
                    egive_group(m, [(u2, T13 + s_step), (u13_0, t0)], l_idx)
                else:
                    egive(m, u2, l_idx, T13 + s_step)
                    egive(m, u13_0, l_idx, t0)

                w13_1 = etake(m, "e13", u13_1)
                k13_1 = t1 * TK13
                acc13[:, :BN] += stream.mm(xb[:, k13_1 : k13_1 + TK13], w13_1[...])
                egive(m, u13_1, l_idx, t1)

            wcol = jnp.sum(
                jnp.where(lane_e8 == e, scr["wn"][...], 0.0),
                axis=1,
                keepdims=True,
            )
            y = (
                rnd(acc[0:MR, :H] * scr["es"][s_cur, :, BN:])
                if has_es
                else rnd(acc[0:MR, :H])
            )
            part[0:MR, :] += wcol * y

            es_wait(m, i + 1)
            h_nxt = (
                acc13[:, :BN] * scr["es"][s_nxt, :, :BN] if has_es else acc13[:, :BN]
            )
            mid_nxt = rnd(rnd(jax.nn.silu(rnd(h_nxt[:, :IR]))) * rnd(h_nxt[:, IR:]))
            xb2[:, :IR] = mid_nxt.astype(BF16)
            return 0

        lax.fori_loop(0, n_hit - 1, body_ep, 0)

        i_last = n_hit - 1
        e_last = scr["elist"][i_last]
        s_last = i_last % NSS
        base_last_w2 = i_last * TE + T13
        if not DIRECT0:
            acc[:, :H] = jnp.zeros((MB, H), F32)
        for s_step in range(T2):
            u2 = base_last_w2 + s_step
            w2 = etake(m, "e2", u2)
            k2 = s_step * TK2
            mm2 = stream.mm(xb2[:, k2 : k2 + TK2], w2[...])
            if DIRECT0 and s_step == 0:
                acc[:, :H] = mm2
            else:
                acc[:, :H] += mm2
            if mp.group_rel:
                if s_step % 2 == 1:
                    egive_group(
                        m,
                        [(u2 - 1, T13 + s_step - 1), (u2, T13 + s_step)],
                        l_idx,
                    )
                elif s_step == T2 - 1:
                    egive(m, u2, l_idx, T13 + s_step)
            else:
                egive(m, u2, l_idx, T13 + s_step)
        wcol_last = jnp.sum(
            jnp.where(lane_e8 == e_last, scr["wn"][...], 0.0),
            axis=1,
            keepdims=True,
        )
        y_last = (
            rnd(acc[0:MR, :H] * scr["es"][s_last, :, BN:])
            if has_es
            else rnd(acc[0:MR, :H])
        )
        part[0:MR, :] += wcol_last * y_last
