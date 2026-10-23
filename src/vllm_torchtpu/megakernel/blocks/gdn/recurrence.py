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
"""GDN gated delta rule: key-major whole state and `NRB`-buffered row chunks."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ...ops import BF16, F32, I32, rnd, rup
from .params import GDNParams
from .states import rec_chunk


def gdn_rec(ctx: Any, gp: GDNParams, u: Any, ybuf: Any, slot: Any) -> None:
    """Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`)."""
    ex = ctx.ex
    scr = ctx.scr
    B, T, VER = ex.B, ex.T, ex.VER
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_OFF = gp.QKV, gp.Z, gp.BA, gp.A_OFF
    l2_eps = gp.l2_eps
    recb, sb = scr["recb"], scr["sbuf"]
    is_key_decay = gp.decay_type == "key"
    a_dim = vh * KD if is_key_decay else vh
    alog = ctx.sm(slot, "alog")[0, :a_dim]
    dtb = ctx.sm(slot, "dtb")[0, :a_dim]
    gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
    expand = vh // kh

    def cols(x):
        xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
        return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

    for b in range(B):
        src = b
        if VER:
            s_, t_ = divmod(b, T)
            urow_ = scr["ub"][s_, t_ : t_ + 1, :]
            q = urow_[0, : kh * KD].reshape(kh, KD)
            k = urow_[0, kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = urow_[0, 2 * kh * KD :].reshape(vh, VD)
            src = b if t_ == 0 else b - 1
        else:
            q = u[b, : kh * KD].reshape(kh, KD)
            k = u[b, kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = u[b, 2 * kh * KD :].reshape(vh, VD)
        q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
        q = q * KD**-0.5
        k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
        qc, kc_ = cols(q), cols(k)
        beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
        a = ybuf[b, A_OFF : A_OFF + a_dim]
        if gp.decay_law == "bounded_sigmoid":
            decay_flat = -F32(gp.decay_bound) * jax.nn.sigmoid(
                jnp.exp(alog) * (a + dtb)
            )
        else:
            decay_flat = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
        eg = (
            jnp.exp(decay_flat.reshape(vh, KD))[:, :, None]
            if is_key_decay
            else jnp.exp(decay_flat)[:, None, None]
        )
        sb[...] = recb[src * vh : (src + 1) * vh].astype(F32) * eg
        pred = jnp.sum(sb[...] * kc_, axis=1)
        delta = (v - pred) * beta[:, None]
        sb[...] = sb[...] + kc_ * delta[:, None, :]
        out = rnd(jnp.sum(sb[...] * qc, axis=1))
        recb[b * vh : (b + 1) * vh] = sb[...].astype(BF16)
        val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
        val = (gnw[None, :] * val.astype(BF16)).astype(F32)
        zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
        gate_val = jax.nn.sigmoid(zz) if gp.gate_type == "sigmoid" else jax.nn.silu(zz)
        scr["mix"][b, :Z] = rnd(val * gate_val).reshape(Z)


def gdn_rec_chunked(ctx: Any, gp: GDNParams, o: Any, slot: Any) -> None:
    """Gated delta rule over `NRB`-buffered row chunks (`GCH=True`)."""
    ex = ctx.ex
    scr, skip, sc = ctx.scr, ctx.skip, ctx.sc
    T, VER = ex.T, ex.VER
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_REL, BA_PAD = (
        gp.QKV,
        gp.Z,
        gp.BA,
        gp.A_OFF - gp.BA,
        gp.BA_PAD,
    )
    l2_eps = gp.l2_eps
    CB, NF, TR, NCH, NRB = gp.CB, gp.NF, gp.TR, gp.NCH, gp.NRB
    recb, sb, ybuf = scr["recb"], scr["sbuf"], scr["ybuf"]
    is_key_decay = gp.decay_type == "key"
    a_dim = vh * KD if is_key_decay else vh
    alog = ctx.sm(slot, "alog")[0, :a_dim]
    dtb = ctx.sm(slot, "dtb")[0, :a_dim]
    gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
    expand = vh // kh

    def colsk(x):
        return jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)

    def cols(x):
        return jnp.broadcast_to(colsk(x)[:, None], (kh, expand, KD, VD)).reshape(
            vh, KD, VD
        )

    HEAD_MAJOR = gp.head_major
    assert not HEAD_MAJOR or 2 * kh <= vh, (kh, vh)
    NCB = 2 if 4 * kh <= vh else 1
    hrow = lax.broadcasted_iota(I32, (vh, VD), 0)

    def rows(r0, n: int, s):
        n_pad = rup(n, 8)
        uc = ybuf[pl.ds(r0, n_pad), 0:QKV]
        bac = ybuf[pl.ds(r0, n_pad), BA : BA + BA_PAD]
        zc = ybuf[pl.ds(r0, n_pad), QKV : QKV + Z]
        st = recb.at[s]
        row_i = lax.broadcasted_iota(I32, (n_pad, 1), 0)
        blk = jnp.zeros((n_pad, Z), F32)
        for bb in range(n):
            h0 = bb * vh
            q = uc[bb, : kh * KD].reshape(kh, KD)
            k = uc[bb, kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = uc[bb, 2 * kh * KD :].reshape(vh, VD)
            q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
            q = q * KD**-0.5
            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
            beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
            a = bac[bb, A_REL : A_REL + a_dim]
            if gp.decay_law == "bounded_sigmoid":
                decay_flat = -F32(gp.decay_bound) * jax.nn.sigmoid(
                    jnp.exp(alog) * (a + dtb)
                )
            else:
                decay_flat = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
            eg = (
                jnp.exp(decay_flat.reshape(vh, KD))[:, :, None]
                if is_key_decay
                else jnp.exp(decay_flat)[:, None, None]
            )
            src = (
                ((n - n // T + bb // T) if bb % T == 0 else bb - 1) * vh if VER else h0
            )
            if not HEAD_MAJOR:
                # Default recurrence: whole-state update staged in `sb`.
                qc, kc_ = cols(q), cols(k)
                sb[...] = st[src : src + vh].astype(F32) * eg
                pred = jnp.sum(sb[...] * kc_, axis=1)
                delta = (v - pred) * beta[:, None]
                sb[...] = sb[...] + kc_ * delta[:, None, :]
                out = rnd(jnp.sum(sb[...] * qc, axis=1))
                st[h0 : h0 + vh] = sb[...].astype(BF16)
            else:
                # Head-major recurrence (opt-in `GDNTunables.gdn_head_major`):
                # per-head update with `colsk` K/Q double-buffered in `sb`.
                cb = (bb % NCB) * 2 * kh
                sb[cb : cb + kh] = colsk(k)
                sb[cb + kh : cb + 2 * kh] = colsk(q)
                beta2 = beta[:, None]
                out_acc = jnp.zeros((vh, VD), F32)
                for h in range(vh):
                    j = h // expand
                    kc_h = sb[cb + j : cb + j + 1]
                    s_h = st[src + h : src + h + 1].astype(F32) * eg[h : h + 1]
                    pred_h = jnp.sum(s_h * kc_h, axis=1)
                    delta_h = (v[h : h + 1] - pred_h) * beta2[h : h + 1]
                    s_h = s_h + kc_h * delta_h[:, None, :]
                    qc_h = sb[cb + kh + j : cb + kh + j + 1]
                    out_acc = jnp.where(hrow == h, jnp.sum(s_h * qc_h, axis=1), out_acc)
                    st[h0 + h : h0 + h + 1] = s_h.astype(BF16)
                out = rnd(out_acc)
            val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
            zz = zc[bb].reshape(vh, VD)
            gate_val = (
                jax.nn.sigmoid(zz) if gp.gate_type == "sigmoid" else jax.nn.silu(zz)
            )
            row = rnd(val * gate_val).reshape(1, Z)
            blk = jnp.where(row_i == bb, row, blk)
        scr["mix"][pl.ds(r0, n_pad), 0:Z] = blk

    def chunk(j, carry):
        s = lax.rem(j, NRB)
        if "gdn_state" not in skip:
            with sc("gdn_rec_wait"):
                rec_chunk(ctx, gp, o, j, s, False).wait()
        if "gdn" not in skip:
            rows(pl.multiple_of(j * CB, CB), CB, s)
        if "gdn_state" not in skip:
            rec_chunk(ctx, gp, o, j, s, True).start()
            jp, jn = j - 1, j - 1 + NRB

            @pl.when(jnp.logical_and(j >= 1, jn < NCH))
            def _():
                sp = lax.rem(jp, NRB)
                rec_chunk(ctx, gp, o, jp, sp, True).wait()
                if TR:

                    @pl.when(jn == NF)
                    def _():
                        rec_chunk(ctx, gp, o, NF, sp, False).start()

                    @pl.when(jn < NF)
                    def _():
                        rec_chunk(ctx, gp, o, jn, sp, False).start()
                else:
                    rec_chunk(ctx, gp, o, jn, sp, False).start()

        return carry

    lax.fori_loop(0, NF, chunk, 0)
    if TR:
        s = NF % NRB
        if "gdn_state" not in skip:
            with sc("gdn_rec_wait"):
                rec_chunk(ctx, gp, o, NF, s, False).wait()
        if "gdn" not in skip:
            rows(NF * CB, TR, s)
        if "gdn_state" not in skip:
            rec_chunk(ctx, gp, o, NF, s, True).start()
