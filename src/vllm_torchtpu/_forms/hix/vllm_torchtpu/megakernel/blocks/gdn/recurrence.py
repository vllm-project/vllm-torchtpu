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


def gdn_rec(ctx: Any, gp: GDNParams, u: Any, ybuf: Any, slot: Any) -> None:
    if int(ctx.ex.live_rows) >= 2 or gdnink_ws(ctx) or _k8_rec_skip(ctx): return (gdn_rec_live if int(ctx.ex.live_rows) >= 2 else gdn_rec_wshm if gdnink_ws(ctx) else gdn_rec_kskip)(ctx, gp, u, ybuf, slot)  # noqa: E701  Gated delta rule on key-major `recb [B * vh, KD, VD]` (`GCH=False`); q38 X-K-LIVE dispatch, same line; q38 L1-GDNINK dispatch, same line
    ex = ctx.ex
    scr = ctx.scr
    B, T, VER = ex.B, ex.T, ex.VER
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_OFF = gp.QKV, gp.Z, gp.BA, gp.A_OFF
    l2_eps = gp.l2_eps
    recb, sb = scr["recb"], scr["sbuf"]
    alog = ctx.sm(slot, "alog")[0, :vh]
    dtb = ctx.sm(slot, "dtb")[0, :vh]
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
        a = ybuf[b, A_OFF : A_OFF + vh]
        decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
        sb[...] = (
            recb[src * vh : (src + 1) * vh].astype(F32) * jnp.exp(decay)[:, None, None]
        )
        pred = jnp.sum(sb[...] * kc_, axis=1)
        delta = (v - pred) * beta[:, None]
        sb[...] = sb[...] + kc_ * delta[:, None, :]
        out = rnd(jnp.sum(sb[...] * qc, axis=1))
        recb[b * vh : (b + 1) * vh] = sb[...].astype(REC)
        val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
        val = (gnw[None, :] * val.astype(BF16)).astype(F32)
        zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
        scr["mix"][b, :Z] = rnd(val * gate_act(gp.gate)(zz)).reshape(Z)


def gdn_rec_chunked(ctx: Any, gp: GDNParams, o: Any, slot: Any) -> None:
    if int(ctx.ex.live_rows) >= 2: return gdn_rec_chunked_live(ctx, gp, o, slot)  # noqa: E701  Gated delta rule over `NRB`-buffered row chunks (`GCH=True`); q38 X-K-LIVE dispatch, same line
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
                    st[h0 + h : h0 + h + 1] = s_h.astype(REC)
                out = rnd(out_acc)
            val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
            zz = zc[bb].reshape(vh, VD)
            row = rnd(val * gate_act(gp.gate)(zz)).reshape(1, Z)
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


# q38 X-K-LIVE begin (L1 GDN half): pad requests skip the recurrence. Appended after every existing line, and the
# two dispatches above replace docstring lines, so the flag-off functions keep their lines and source locations.
# With `ctx.ex.live_rows` a request whose `live_rows` entry is 0 (a pad request: `q_len == 0`) reads and writes no
# `recb` state and its `mix` rows are zeroed (its outputs are discarded; zero keeps them finite). A live request's
# rows run the original row body unchanged, so live rows stay bit-identical (the CPU-interpret proofs).


def gdn_rec_live(ctx: Any, gp: GDNParams, u: Any, ybuf: Any, slot: Any) -> None:
    """`gdn_rec` with pad requests skipped.

    Verify (`T > 1`, the whole step): a `fori_loop` over 8-row chunks (aligned `ybuf` / `mix` blocks, as
    `gdn_rec_chunked`), with one `lax.cond` per request of a chunk around its `T` row bodies; the chunk's `mix` block
    is carried through the conds and a pad request leaves its rows zero. The row body is `gdn_rec`'s, with the
    request's state slots of the whole-state `recb` at dynamic row offsets. One copy of the request body in the
    program: 9004e0aa9's per-request `pl.when` (four unrolled regions at 16 rows) needed 222 KiB more scoped VMEM than
    the served 16-row bucket has (63.72M of 63.50M at rings 10:10, l1-live-aot-i2 j-f7cac4a2).
    Decode (`T == 1`): one `pl.when` per row (not used by the whole step)."""
    import functools

    from ...live_rows import req_live, row_live

    ex = ctx.ex
    scr = ctx.scr
    B, T, VER = ex.B, ex.T, ex.VER
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_OFF = gp.QKV, gp.Z, gp.BA, gp.A_OFF
    A_REL, BA_PAD = gp.A_OFF - gp.BA, gp.BA_PAD
    l2_eps = gp.l2_eps
    recb, sb = scr["recb"], scr["sbuf"]
    alog = ctx.sm(slot, "alog")[0, :vh]
    dtb = ctx.sm(slot, "dtb")[0, :vh]
    gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
    expand = vh // kh

    def cols(x):
        xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
        return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

    def state_update(q, k, v, beta, a, src, b):
        """`gdn_rec`'s recurrence on one row: reads `recb` row group `src`, writes row group `b` (static or traced)."""
        q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
        q = q * KD**-0.5
        k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
        qc, kc_ = cols(q), cols(k)
        decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
        sb[...] = recb[pl.ds(src * vh, vh)].astype(F32) * jnp.exp(decay)[:, None, None]
        pred = jnp.sum(sb[...] * kc_, axis=1)
        delta = (v - pred) * beta[:, None]
        sb[...] = sb[...] + kc_ * delta[:, None, :]
        out = rnd(jnp.sum(sb[...] * qc, axis=1))
        recb[pl.ds(b * vh, vh)] = sb[...].astype(REC)
        val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
        return (gnw[None, :] * val.astype(BF16)).astype(F32)

    if not VER:  # decode rows: one guarded row each (`gdn_rec`'s non-verify body)
        for b in range(B):

            @pl.when(req_live(ctx, b))
            def _(b=b):
                q = u[b, : kh * KD].reshape(kh, KD)
                k = u[b, kh * KD : 2 * kh * KD].reshape(kh, KD)
                v = u[b, 2 * kh * KD :].reshape(vh, VD)
                beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
                val = state_update(q, k, v, beta, ybuf[b, A_OFF : A_OFF + vh], b, b)
                zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
                scr["mix"][b, :Z] = rnd(val * gate_act(gp.gate)(zz)).reshape(Z)

            @pl.when(jnp.logical_not(req_live(ctx, b)))
            def _(b=b):
                scr["mix"][b : b + 1, :Z] = jnp.zeros((1, Z), scr["mix"].dtype)

        return

    CB = 8 if B % 8 == 0 else B
    assert CB % T == 0 and B % CB == 0, (B, T, CB)
    NC, NQ, n_pad = B // CB, CB // T, rup(CB, 8)

    def chunk(c, carry):
        r0 = pl.multiple_of(c * CB, CB) if NC > 1 else 0
        bac = ybuf[pl.ds(r0, n_pad), BA : BA + BA_PAD]
        zc = ybuf[pl.ds(r0, n_pad), QKV : QKV + Z]
        row_i = lax.broadcasted_iota(I32, (n_pad, 1), 0)

        def req_rows(blk, q0):
            s_ = c * NQ + q0  # the request (traced when NC > 1)
            for t_ in range(T):
                bb = q0 * T + t_  # row in the chunk (static)
                b = r0 + bb
                urow_ = scr["ub"][s_, t_ : t_ + 1, :]
                q = urow_[0, : kh * KD].reshape(kh, KD)
                k = urow_[0, kh * KD : 2 * kh * KD].reshape(kh, KD)
                v = urow_[0, 2 * kh * KD :].reshape(vh, VD)
                beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
                val = state_update(q, k, v, beta, bac[bb, A_REL : A_REL + vh], b if t_ == 0 else b - 1, b)
                zz = zc[bb].reshape(vh, VD)
                row = rnd(val * gate_act(gp.gate)(zz)).reshape(1, Z)
                blk = jnp.where(row_i == bb, row, blk)
            return blk

        blk = jnp.zeros((n_pad, Z), F32)
        for q0 in range(NQ):
            b0 = r0 + q0 * T
            blk = lax.cond(row_live(ctx, b0), functools.partial(req_rows, q0=q0), lambda blk: blk, blk)
        scr["mix"][pl.ds(r0, n_pad), 0:Z] = blk
        return carry

    if NC > 1:
        lax.fori_loop(0, NC, chunk, 0)
    else:
        chunk(0, 0)


def gdn_rec_chunked_live(ctx: Any, gp: GDNParams, o: Any, slot: Any) -> None:
    """`gdn_rec_chunked` with pad requests skipped: in each row chunk, one `lax.cond` per request around its
    `T` rows (the chunk's output block is carried through; a pad request leaves its rows zero). The state ring,
    its DMAs and their order are `gdn_rec_chunked`'s (chunk reads / write-backs move whole chunks)."""
    from ...live_rows import row_live

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
    hrow = lax.broadcasted_iota(I32, (vh, VD), 0)
    TQ = T if VER else 1  # rows per request

    def rows(r0, n: int, s):
        n_pad = rup(n, 8)
        uc = ybuf[pl.ds(r0, n_pad), 0:QKV]
        bac = ybuf[pl.ds(r0, n_pad), BA : BA + BA_PAD]
        zc = ybuf[pl.ds(r0, n_pad), QKV : QKV + Z]
        row_i = lax.broadcasted_iota(I32, (n_pad, 1), 0)

        def one(bb: int, blk):  # `gdn_rec_chunked`'s row body, unchanged (ref view taken inside the branch)
            st = recb.at[s]
            h0 = bb * vh
            q = uc[bb, : kh * KD].reshape(kh, KD)
            k = uc[bb, kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = uc[bb, 2 * kh * KD :].reshape(vh, VD)
            q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
            q = q * KD**-0.5
            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
            beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
            a = bac[bb, A_REL : A_REL + vh]
            decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
            src = (
                ((n - n // T + bb // T) if bb % T == 0 else bb - 1) * vh if VER else h0
            )
            if not HEAD_MAJOR:
                qc, kc_ = cols(q), cols(k)
                sb[...] = st[src : src + vh].astype(F32) * jnp.exp(decay)[:, None, None]
                pred = jnp.sum(sb[...] * kc_, axis=1)
                delta = (v - pred) * beta[:, None]
                sb[...] = sb[...] + kc_ * delta[:, None, :]
                out = rnd(jnp.sum(sb[...] * qc, axis=1))
                st[h0 : h0 + vh] = sb[...].astype(REC)
            else:
                cb = (bb % NCB) * 2 * kh
                sb[cb : cb + kh] = colsk(k)
                sb[cb + kh : cb + 2 * kh] = colsk(q)
                eg = jnp.exp(decay)[:, None, None]
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
                    st[h0 + h : h0 + h + 1] = s_h.astype(REC)
                out = rnd(out_acc)
            val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
            zz = zc[bb].reshape(vh, VD)
            row = rnd(val * gate_act(gp.gate)(zz)).reshape(1, Z)
            return jnp.where(row_i == bb, row, blk)

        blk = jnp.zeros((n_pad, Z), F32)
        for q0 in range(0, n, TQ):

            def live_rows_(blk, q0=q0):
                for bb in range(q0, q0 + TQ):
                    blk = one(bb, blk)
                return blk

            blk = lax.cond(row_live(ctx, r0 + q0), live_rows_, lambda blk: blk, blk)
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
# q38 X-K-LIVE end


# q38 L1-GDNINK begin (GDN-REC-WSHM, VLLM_TPU_MK_GDN_INK=1): the head-major state update on the whole-state path (<= 16 rows,
# `GCH=False`, `gdn_rec`). Appended after every existing line, and `gdn_rec`'s dispatch line above is the X-K-LIVE line
# extended in place, so the flag-off functions keep their lines and source locations (flag-off lowering identical).
# `gdn_rec_wshm` is `gdn_rec` with its whole-state `sbuf` staging of `[vh, KD, VD]` (96 f32 vregs at vh = 6, more than the 64
# vector registers) replaced by the per-head update of `gdn_rec_chunked`'s head-major branch (served above 16 rows since MM's
# VLLM_TPU_MK_GDN_HM; j-ed2d3a52): each head's `[1, KD, VD]` state, its key / query columns `colsk` (staged once per row in
# `sbuf`) and the reductions stay in vregs. Every element sees the same f32 operations in the same order as `gdn_rec`:
# state * exp(decay); pred = sum over KD of state * k; delta = (v - pred) * beta; state + k * delta; out = sum over KD of
# state * q; only the sums are taken per head (`[1, KD, VD]`) instead of over the `[vh, KD, VD]` block. X-K-LIVE mode 2
# keeps its own path (it takes precedence in the dispatch); mode 1 does not touch the recurrence.
import collections as _gink_collections  # noqa: E402
import os as _gink_os  # noqa: E402

GDNINK_ENV = "VLLM_TPU_MK_GDN_INK"
GDNINK_TRACES: "_gink_collections.Counter[str]" = _gink_collections.Counter()  # traces per variant (tests' engagement check)
_GDNINK_NOTED: set = set()


def gdnink_mode() -> int:
    """VLLM_TPU_MK_GDN_INK, a bit mask: 0 / unset = off; bit 1 = GDN-REC-WSHM (head-major update on the whole-state path);
    bit 2 = GDN-REL-FETCH (megakernel_runtime/qwen38_wholestep_call.py: the whole step built with `Tiles.rel_fetch`)."""
    v = _gink_os.environ.get(GDNINK_ENV, "0").strip()
    return int(v) if v.isdigit() else 0


def gdnink_ws(ctx: Any) -> bool:
    """`gdn_rec` (the whole-state path) takes `gdn_rec_wshm` (VLLM_TPU_MK_GDN_INK bit 1)."""
    return bool(gdnink_mode() & 1)


def _gdnink_note(variant: str, msg: str) -> None:
    """Count every trace of a variant; one engagement line per process and variant (logger "vllm." + module, as X-K-LIVE's)."""
    GDNINK_TRACES[variant] += 1
    if variant not in _GDNINK_NOTED:
        _GDNINK_NOTED.add(variant)
        import logging

        logging.getLogger("vllm." + __name__).info("q38 L1-GDNINK %s traced: %s", variant, msg)


def gdn_rec_wshm(ctx: Any, gp: GDNParams, u: Any, ybuf: Any, slot: Any) -> None:
    """`gdn_rec` with the head-major per-head state update (see the block comment above)."""
    ex = ctx.ex
    scr = ctx.scr
    B, T, VER = ex.B, ex.T, ex.VER
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_OFF = gp.QKV, gp.Z, gp.BA, gp.A_OFF
    l2_eps = gp.l2_eps
    recb, sb = scr["recb"], scr["sbuf"]
    alog = ctx.sm(slot, "alog")[0, :vh]
    dtb = ctx.sm(slot, "dtb")[0, :vh]
    gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
    expand = vh // kh
    assert 2 * kh <= vh, ("GDN-REC-WSHM stages the key / query columns in sbuf's vh slots", kh, vh)
    _gdnink_note("GDN-REC-WSHM", f"head-major state update on the whole-state path, {B} rows (T {T}), vh {vh}, kh {kh}")

    def colsk(x):
        return jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)

    hrow = lax.broadcasted_iota(I32, (vh, VD), 0)
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
        beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
        a = ybuf[b, A_OFF : A_OFF + vh]
        decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
        sb[0:kh] = colsk(k)
        sb[kh : 2 * kh] = colsk(q)
        eg = jnp.exp(decay)[:, None, None]
        beta2 = beta[:, None]
        out_acc = jnp.zeros((vh, VD), F32)
        for h in range(vh):
            j = h // expand
            kc_h = sb[j : j + 1]
            s_h = recb[src * vh + h : src * vh + h + 1].astype(F32) * eg[h : h + 1]
            pred_h = jnp.sum(s_h * kc_h, axis=1)
            delta_h = (v[h : h + 1] - pred_h) * beta2[h : h + 1]
            s_h = s_h + kc_h * delta_h[:, None, :]
            qc_h = sb[kh + j : kh + j + 1]
            out_acc = jnp.where(hrow == h, jnp.sum(s_h * qc_h, axis=1), out_acc)
            recb[b * vh + h : b * vh + h + 1] = s_h.astype(REC)
        out = rnd(out_acc)
        val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
        val = (gnw[None, :] * val.astype(BF16)).astype(F32)
        zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
        scr["mix"][b, :Z] = rnd(val * gate_act(gp.gate)(zz)).reshape(Z)
# q38 L1-GDNINK end


# q38 HIX-KERN K8 begin (VLLM_TPU_MK_KERN_REC_SKIP, default off; megakernel_runtime/qwen38_k8.py; appended, so the flag-off functions keep
# their lines except gdn_rec's dispatch line, extended in place as X-K-LIVE / L1-GDNINK did). Under X-K-LIVE mode 1 the whole-state
# recurrence `gdn_rec` runs all B rows, so a pad request's T rows are computed: 12 / 8 / 4 of the 16 rows at 1 / 2 / 3 live. Mode 2
# (VLLM_TPU_MK_LIVE_GDN_COMPUTE) skips them but is refused at <= 16 rows (9004e0aa9 / l1-live-aot-i2: its per-request regions overflowed
# the 16-row bucket's scoped VMEM). `gdn_rec_kskip` is `gdn_rec`'s VER row body, op for op and with the SAME STATIC row offsets (no
# dynamic row offset on any tiled ref: TILE-ALIGN; the rolled form 86d4dccc55 had dynamic single-row offsets on ybuf / mix), unrolled
# over the request slots, each request's T rows under `pl.when(req_live(ctx, s))`; a pad request writes zero `mix` rows (as
# `gdn_rec_live`) and leaves its `recb` rows as they are. Live rows see the same f32 operations in the same order on the same values (rows
# of a request in t order; the chain through `recb` bf16 unchanged), so their outputs, states and checkpoints match the flag-off kernel;
# pad rows' outputs are discarded downstream (X-K-LIVE's contract: write ids nulled, sidecar flush skips them).
GDN_KSKIP_TRACES: list = []  # (B, T, vh) per traced flag-on recurrence (tests / engagement)


def _k8_rec_skip(ctx: Any) -> bool:
    from vllm_torchtpu.megakernel_runtime.qwen38_k8 import rec_skip_on

    return bool(rec_skip_on()) and int(ctx.ex.live_rows) == 1 and bool(ctx.ex.VER)


def gdn_rec_kskip(ctx: Any, gp: GDNParams, u: Any, ybuf: Any, slot: Any) -> None:
    """`gdn_rec` (verify rows, X-K-LIVE mode 1) with pad requests skipped (block comment above)."""
    from ...live_rows import req_live

    ex = ctx.ex
    scr = ctx.scr
    B, T = ex.B, ex.T
    kh, vh, KD, VD = gp.kh, gp.vh, gp.KD, gp.VD
    QKV, Z, BA, A_OFF = gp.QKV, gp.Z, gp.BA, gp.A_OFF
    l2_eps = gp.l2_eps
    recb, sb = scr["recb"], scr["sbuf"]
    alog = ctx.sm(slot, "alog")[0, :vh]
    dtb = ctx.sm(slot, "dtb")[0, :vh]
    gnw = ctx.sm(slot, "gnorm")[0, :].astype(BF16)
    expand = vh // kh
    assert B % T == 0, (B, T)
    GDN_KSKIP_TRACES.append((B, T, vh))

    def cols(x):
        xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
        return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

    def rows(s_: int) -> None:
        for t_ in range(T):
            b = s_ * T + t_
            urow_ = scr["ub"][s_, t_ : t_ + 1, :]
            q = urow_[0, : kh * KD].reshape(kh, KD)
            k = urow_[0, kh * KD : 2 * kh * KD].reshape(kh, KD)
            v = urow_[0, 2 * kh * KD :].reshape(vh, VD)
            src = b if t_ == 0 else b - 1
            q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + l2_eps)
            q = q * KD**-0.5
            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + l2_eps)
            qc, kc_ = cols(q), cols(k)
            beta = rnd(jax.nn.sigmoid(ybuf[b, BA : BA + vh]))
            a = ybuf[b, A_OFF : A_OFF + vh]
            decay = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
            sb[...] = (
                recb[src * vh : (src + 1) * vh].astype(F32) * jnp.exp(decay)[:, None, None]
            )
            pred = jnp.sum(sb[...] * kc_, axis=1)
            delta = (v - pred) * beta[:, None]
            sb[...] = sb[...] + kc_ * delta[:, None, :]
            out = rnd(jnp.sum(sb[...] * qc, axis=1))
            recb[b * vh : (b + 1) * vh] = sb[...].astype(REC)
            val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + gp.eps)
            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
            zz = ybuf[b, QKV : QKV + Z].reshape(vh, VD)
            scr["mix"][b, :Z] = rnd(val * gate_act(gp.gate)(zz)).reshape(Z)

    for s in range(B // T):
        live = req_live(ctx, s)
        pl.when(live)(lambda s=s: rows(s))

        @pl.when(jnp.logical_not(live))
        def _(s=s):
            for t_ in range(T):
                scr["mix"][s * T + t_, :Z] = jnp.zeros((Z,), scr["mix"].dtype)
# q38 HIX-KERN K8 end
