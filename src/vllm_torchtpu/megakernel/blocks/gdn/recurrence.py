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

from ...ops import BF16, F32, I32, REC, rnd, rup
from .params import GDNParams
from .states import rec_chunk


def gate_act(gate: str) -> Any:
    """Activation applied to `z` on the gated output norm."""
    return {"silu": jax.nn.silu, "sigmoid": jax.nn.sigmoid}[gate]


def _qk_norms(x: Any, kh: int, KD: int, l2_eps: float) -> tuple[list, list]:
    """Per-head L2-normed queries (scaled by `KD**-0.5`) and keys of every row of `x
    [..., QKV]`: one lane reduction over all rows instead of one per row."""

    def l2n(y):
        return y * lax.rsqrt(jnp.sum(y * y, axis=-1, keepdims=True) + l2_eps)

    qn = [l2n(x[..., j * KD : (j + 1) * KD]) * KD**-0.5 for j in range(kh)]
    kn = [l2n(x[..., (kh + j) * KD : (kh + j + 1) * KD]) for j in range(kh)]
    return qn, kn


def _row_order(n: int, T: int, VER: bool) -> list[int]:
    """Verify rows request-interleaved (every request's row t, then t + 1): consecutive
    rows are independent state chains, so their updates overlap."""
    if not VER or n % T:
        return list(range(n))
    return [r * T + t for t in range(T) for r in range(n // T)]


def _gated_norm(out: Any, zz: Any, gnw: Any, eps: float, gate: str) -> Any:
    """Gated RMS norm of one head's outputs `out [rows, VD]` (gate input `zz`)."""
    val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + eps)
    val = (gnw[None, :] * val.astype(BF16)).astype(F32)
    return rnd(val * gate_act(gate)(zz))


def _head_update(
    st: Any, src: int, dst: int, eg: Any, kc: Any, qc: Any, v: Any, beta: Any
) -> Any:
    """Delta-rule update of one head's state `st[src]` into `st[dst]`, kept in vregs;
    returns the head's output `[1, VD]`."""
    s_h = st[src : src + 1].astype(F32) * eg
    pred = jnp.sum(s_h * kc, axis=1)
    s_h = s_h + kc * ((v - pred) * beta)[:, None, :]
    st[dst : dst + 1] = s_h.astype(REC)
    return jnp.sum(s_h * qc, axis=1)


def gdn_rec(ctx: Any, gp: GDNParams, u: Any, ybuf: Any, slot: Any) -> None:
    """Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`).

    Rows run in `_row_order`, each head's update in vregs (`_head_update`); the raw
    outputs are staged in `mix` and gated-normed for all rows after the loop."""
    ex = ctx.ex
    scr = ctx.scr
    B, T, VER = ex.B, ex.T, ex.VER
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_OFF = gp.QKV, gp.Z, gp.BA, gp.A_OFF
    recb, mix = scr["recb"], scr["mix"]
    alog = ctx.sm(slot, "alog")[0, :vh]
    dtb = ctx.sm(slot, "dtb")[0, :vh]
    gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
    expand = vh // kh

    def cols(x):
        xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
        return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

    qn, kn = _qk_norms(scr["ub"][:, 0:T, :] if VER else u[0:B], kh, KD, gp.l2_eps)
    for b in _row_order(B, T, VER):
        src = b
        if VER:
            s_, t_ = divmod(b, T)
            row = (s_, slice(t_, t_ + 1))
            v = scr["ub"][s_, t_ : t_ + 1, 2 * kh * KD :].reshape(vh, VD)
            src = b if t_ == 0 else b - 1
        else:
            row = slice(b, b + 1)
            v = u[b, 2 * kh * KD :].reshape(vh, VD)
        q = jnp.concatenate([x[row] for x in qn], axis=0)
        k = jnp.concatenate([x[row] for x in kn], axis=0)
        qc, kc_ = cols(q), cols(k)
        beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
        a = ybuf[b, A_OFF : A_OFF + vh]
        decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
        eg = jnp.exp(decay)[:, None, None]
        outs = [
            _head_update(
                recb,
                src * vh + h,
                b * vh + h,
                eg[h : h + 1],
                kc_[h : h + 1],
                qc[h : h + 1],
                v[h : h + 1],
                beta[h : h + 1, None],
            )
            for h in range(vh)
        ]
        mix[b, :Z] = rnd(jnp.concatenate(outs, axis=0)).reshape(Z)
    for h in range(vh):
        cs = slice(h * VD, (h + 1) * VD)
        mix[0:B, cs] = _gated_norm(
            mix[0:B, cs],
            ybuf[0:B, QKV + h * VD : QKV + (h + 1) * VD],
            gnw,
            gp.eps,
            gp.gate,
        )


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
    alog = ctx.sm(slot, "alog")[0, :vh]
    dtb = ctx.sm(slot, "dtb")[0, :vh]
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

    def rows(r0, n: int, s):
        n_pad = rup(n, 8)
        uc = ybuf[pl.ds(r0, n_pad), 0:QKV]
        bac = ybuf[pl.ds(r0, n_pad), BA : BA + BA_PAD]
        zc = ybuf[pl.ds(r0, n_pad), QKV : QKV + Z]
        st = recb.at[s]
        row_i = lax.broadcasted_iota(I32, (n_pad, 1), 0)
        blk = jnp.zeros((n_pad, Z), F32)
        qn, kn = _qk_norms(uc, kh, KD, l2_eps)
        for bb in _row_order(n, T, VER):
            h0 = bb * vh
            q = jnp.concatenate([x[bb : bb + 1] for x in qn], axis=0)
            k = jnp.concatenate([x[bb : bb + 1] for x in kn], axis=0)
            v = uc[bb, 2 * kh * KD :].reshape(vh, VD)
            beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
            a = bac[bb, A_REL : A_REL + vh]
            decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
            src = (
                ((n - n // T + bb // T) if bb % T == 0 else bb - 1) * vh if VER else h0
            )
            if not HEAD_MAJOR:
                # Default recurrence: whole-state update staged in `sb`.
                qc, kc_ = cols(q), cols(k)
                sb[...] = st[src : src + vh].astype(F32) * jnp.exp(decay)[:, None, None]
                pred = jnp.sum(sb[...] * kc_, axis=1)
                delta = (v - pred) * beta[:, None]
                sb[...] = sb[...] + kc_ * delta[:, None, :]
                out = rnd(jnp.sum(sb[...] * qc, axis=1))
                st[h0 : h0 + vh] = sb[...].astype(REC)
            else:
                # Head-major recurrence (opt-in `GDNTunables.gdn_head_major`):
                # per-head update with `colsk` K/Q double-buffered in `sb`.
                cb = (bb % NCB) * 2 * kh
                sb[cb : cb + kh] = colsk(k)
                sb[cb + kh : cb + 2 * kh] = colsk(q)
                eg = jnp.exp(decay)[:, None, None]
                outs = []
                for h in range(vh):
                    j = h // expand
                    outs.append(
                        _head_update(
                            st,
                            src + h,
                            h0 + h,
                            eg[h : h + 1],
                            sb[cb + j : cb + j + 1],
                            sb[cb + kh + j : cb + kh + j + 1],
                            v[h : h + 1],
                            beta[h : h + 1, None],
                        )
                    )
                out = rnd(jnp.concatenate(outs, axis=0))
            blk = jnp.where(row_i == bb, out.reshape(1, Z), blk)
        # The gated norm of every row at once, after the state chains; pad rows stay 0.
        blk = jnp.concatenate(
            [
                _gated_norm(
                    blk[:, h * VD : (h + 1) * VD],
                    zc[:, h * VD : (h + 1) * VD],
                    gnw,
                    gp.eps,
                    gp.gate,
                )
                for h in range(vh)
            ],
            axis=1,
        )
        scr["mix"][pl.ds(r0, n_pad), 0:Z] = jnp.where(row_i < n, blk, 0.0)

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
