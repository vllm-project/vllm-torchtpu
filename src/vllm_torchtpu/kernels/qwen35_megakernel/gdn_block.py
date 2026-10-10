"""GatedDeltaNet linear-attention recurrence block for the Qwen3.5 decode megakernel."""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .config import (
    BF16,
    CONV_STAGE_MIN_B,
    F32,
    FP8,
    I32,
    U32,
    rnd,
)
from .context import MKBuildContext
from .pool_dma import GDNPoolDMA


class GDNBlock:
    """GatedDeltaNet linear-attention recurrence block."""

    def __init__(self, ctx: MKBuildContext, rings, gdn_dma: GDNPoolDMA | None = None):
        self.ctx = ctx
        self.rings = rings
        self.gdn_dma = gdn_dma if gdn_dma is not None else GDNPoolDMA(ctx)

    def gdn(self, l, slot, g0):
        ctx = self.ctx
        c, scr, ST, O, raw_refs = ctx.c, ctx.scr, ctx.ST, ctx.O, ctx.raw_refs
        sc, sm = ctx.sc, ctx.sm
        dmatmul, gctx = self.rings.dmatmul, self.rings.gctx
        gdn_dma = self.gdn_dma
        AE, FUSED, GPR, GCH, VER = ctx.AE, ctx.FUSED, ctx.GPR, ctx.GCH, ctx.VER
        NGI, NGO, NS, T, B = ctx.NGI, ctx.NGO, ctx.NS, ctx.T, ctx.B
        QKV, Z, BA, H = ctx.QKV, ctx.Z, ctx.BA, ctx.H
        kh, vh, KD, VD = ctx.kh, ctx.vh, ctx.KD, ctx.VD
        A_conv, qkv_128, conv8_rows = ctx.A_conv, ctx.qkv_128, ctx.conv8_rows
        REC_PAGES, GDN_PAGES, GCONV_ROWS = ctx.REC_PAGES, ctx.GDN_PAGES, ctx.GCONV_ROWS
        SSM_U32_CAP, ssm_u32_rows = ctx.SSM_U32_CAP, ctx.ssm_u32_rows
        NRB, lay, nt, typed, skip = ctx.NRB, ctx.lay, ctx.nt, ctx.typed, ctx.skip

        o = l - (l + 1) // AE
        if FUSED:
            in_page = scr["gdn_in_page"]
            out_page = scr["gdn_out_page"]
            sb = scr["sbuf"]

            with sc("gdn_in"):
                for s_ in range(NGI):
                    gdn_dma.start_gdn_in(o, s_)
                y = dmatmul("lin_in", g0, scr["xb"], c.NLI, ctx=gctx(l, 0))
                ybuf = scr["ybuf"]
                ybuf[:, :c.NLI] = rnd(y * sm(slot, "in_s", c.NLI))

            ow, _ = lay["conv"]
            taps = [scr["sm"][slot, :, ow + t * QKV:ow + (t + 1) * QKV] for t in range(4)]
            alog = sm(slot, "alog")[0, :vh]
            dtb = sm(slot, "dtb")[0, :vh]
            gnw = sm(slot, "gnorm")[0, :].astype(BF16)
            expand = vh // kh
            row_iota_conv = lax.broadcasted_iota(I32, (A_conv, QKV), 0)
            # Conv page staging: the 3 live window rows occupy 3 * qkv_128 page rows; round
            # to the bf16 (16, 128) tile and stage them in sbuf slot 0 (free once the
            # token's state is cast), so the cast + store is tile aligned.
            conv_live = -(-(3 * qkv_128) // 16) * 16
            conv_fast = (
                ctx.conv_stage_min_b <= c.B
                and QKV % 128 == 0
                and VD == 128
                and conv_live <= KD
                and conv_live <= conv8_rows
            )

            def colsk(x):  # [kh, KD] -> [kh, KD, VD] holding x[j, i] at [j, i, :]
                return jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)

            # head-major state update: each value head's [KD, VD] state goes bf16 -> f32 ->
            # decay -> pred -> rank-1 update -> out -> bf16 in one pass held in vregs (no f32
            # sbuf round trips, no [vh, KD, VD] value carried across tokens). Token t reads
            # token t-1's state back from its bf16 checkpoint in out_page (bit-identical: the
            # same bf16 value the baseline carried). sbuf holds the q / k columns per kv head
            # (double-buffered across tokens when 4 * kh <= vh).
            assert 2 * kh <= vh, (kh, vh)
            NCB = 2 if 4 * kh <= vh else 1
            hrow = lax.broadcasted_iota(I32, (vh, VD), 0)
            HR2 = KD // 2  # u32 rows per head in the [vh * KD // 2, VD] u32 view

            for s in range(NS):
                slot_i = s % NGI
                slot_o = s % NGO
                with sc("gdn_rec_wait"):
                    act_in = gdn_dma.wait_gdn_in(slot_i)
                if typed:
                    u32_in = pltpu.bitcast(
                        in_page[slot_i, 0:REC_PAGES], U32
                    ).reshape(SSM_U32_CAP, 128)[:ssm_u32_rows, :]
                    raw_conv = (
                        pltpu.bitcast(in_page[slot_i, REC_PAGES], BF16)
                        .reshape(256, 128)[:conv8_rows, :]
                        .reshape(A_conv, QKV)
                        .astype(F32)
                    )
                    u32_in = jnp.where(act_in, u32_in, jnp.uint32(0))
                    s_bf16 = pltpu.bitcast(u32_in, BF16).reshape(vh, KD, VD)
                else:
                    in_view = (
                        in_page.at[slot_i, pl.ds(0, REC_PAGES)]
                        .bitcast(U32)
                        .reshape(SSM_U32_CAP, 128)
                    )
                    raw_conv = (
                        in_page.at[slot_i, REC_PAGES]
                        .bitcast(BF16)
                        .reshape(256, 128)[:conv8_rows, :]
                        .reshape(A_conv, QKV)
                        .astype(F32)
                    )
                raw_conv = jnp.where(act_in, raw_conv, 0.0)
                c0 = raw_conv[0:1, :]
                c1 = raw_conv[1:2, :]
                c2 = raw_conv[2:3, :]
                # the state is read from in_page inside the token loop: a refill of this
                # in_page slot (only when NS > NGI) is issued after the loop
                refill_in = s + NGI < NS
                if refill_in and typed:
                    gdn_dma.start_gdn_in(o, s + NGI)
                gdn_dma.wait_gdn_out(slot_o)
                if "gdn" not in skip:
                    with sc("gdn_rec"):
                        for t in range(T):
                            b = s * T + t
                            x_t = ybuf[b:b + 1, 0:QKV]
                            u_t = rnd(
                                jax.nn.silu(
                                    c0 * taps[0]
                                    + c1 * taps[1]
                                    + c2 * taps[2]
                                    + x_t * taps[3]
                                )
                            )
                            c0, c1, c2 = c1, c2, x_t
                            q = u_t[0, :kh * KD].reshape(kh, KD)
                            k = u_t[0, kh * KD:2 * kh * KD].reshape(kh, KD)
                            v = u_t[0, 2 * kh * KD:].reshape(vh, VD)
                            q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                            q = q * KD ** -0.5
                            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                            cb = (t % NCB) * 2 * kh
                            sb[cb:cb + kh] = colsk(k)
                            sb[cb + kh:cb + 2 * kh] = colsk(q)
                            beta = rnd(jax.nn.sigmoid(ybuf[b, BA:BA + vh]))
                            a = ybuf[b, BA + 64:BA + 64 + vh]
                            g = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                            eg = jnp.exp(g)[:, None, None]
                            beta2 = beta[:, None]
                            if not typed:
                                ov = (
                                    out_page.at[slot_o, pl.ds(t * GDN_PAGES, REC_PAGES)]
                                    .bitcast(U32)
                                    .reshape(SSM_U32_CAP, 128)
                                )
                                if t > 0:
                                    pv = (
                                        out_page.at[
                                            slot_o,
                                            pl.ds((t - 1) * GDN_PAGES, REC_PAGES),
                                        ]
                                        .bitcast(U32)
                                        .reshape(SSM_U32_CAP, 128)
                                    )
                            out_acc = jnp.zeros((vh, VD), F32)
                            new_heads = []
                            for h in range(vh):
                                j = h // expand
                                if typed:
                                    st_h = s_bf16[h:h + 1]
                                elif t == 0:
                                    u_h = jnp.where(
                                        act_in,
                                        in_view[h * HR2:(h + 1) * HR2, :],
                                        jnp.uint32(0),
                                    )
                                    st_h = pltpu.bitcast(u_h, BF16).reshape(1, KD, VD)
                                else:
                                    st_h = pltpu.bitcast(
                                        pv[h * HR2:(h + 1) * HR2, :], BF16
                                    ).reshape(1, KD, VD)
                                kc_h = sb[cb + j:cb + j + 1]
                                s_h = st_h.astype(F32) * eg[h:h + 1]
                                pred_h = jnp.sum(s_h * kc_h, axis=1)
                                delta_h = (v[h:h + 1] - pred_h) * beta2[h:h + 1]
                                s_h = s_h + kc_h * delta_h[:, None, :]
                                qc_h = sb[cb + kh + j:cb + kh + j + 1]
                                out_acc = jnp.where(
                                    hrow == h, jnp.sum(s_h * qc_h, axis=1), out_acc
                                )
                                nb_h = s_h.astype(BF16)
                                if typed:
                                    new_heads.append(nb_h)
                                else:
                                    ov[h * HR2:(h + 1) * HR2, :] = pltpu.bitcast(
                                        nb_h.reshape(KD, VD), U32
                                    )
                            out = rnd(out_acc)
                            if typed:
                                s_bf16 = jnp.concatenate(new_heads, axis=0)
                            val = out * lax.rsqrt(
                                jnp.mean(out * out, axis=-1, keepdims=True) + c.eps
                            )
                            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
                            zz = ybuf[b, QKV:QKV + Z].reshape(vh, VD)
                            scr["mix"][b, :Z] = rnd(val * jax.nn.silu(zz)).reshape(Z)

                            if typed:
                                u32_out = pltpu.bitcast(s_bf16.reshape(vh * KD, VD), U32)
                            if conv_fast:
                                # Conv window page rows r * qkv_128 + j = c_r lanes
                                # j*128 .. j*128+127; rows of c3 .. are zero. Stage the 3
                                # live rows with single-row f32 stores, cast once.
                                cst = sb.at[0]
                                for r_, cr in enumerate((c0, c1, c2)):
                                    for j in range(qkv_128):
                                        rr = r_ * qkv_128 + j
                                        cst[rr:rr + 1, :] = cr[:, j * 128:(j + 1) * 128]
                                cst[3 * qkv_128:conv_live, :] = jnp.zeros(
                                    (conv_live - 3 * qkv_128, 128), F32
                                )
                                conv_head = cst[0:conv_live, :].astype(BF16)
                                w_conv = None
                            else:
                                c0_f = jnp.broadcast_to(c0, (A_conv, QKV))
                                c1_f = jnp.broadcast_to(c1, (A_conv, QKV))
                                c2_f = jnp.broadcast_to(c2, (A_conv, QKV))
                                w_conv = jnp.where(
                                    row_iota_conv == 0,
                                    c0_f,
                                    jnp.where(
                                        row_iota_conv == 1,
                                        c1_f,
                                        jnp.where(row_iota_conv == 2, c2_f, 0.0),
                                    ),
                                ).astype(BF16)
                            if typed:
                                u32_p = (
                                    jnp.pad(
                                        u32_out,
                                        ((0, SSM_U32_CAP - ssm_u32_rows), (0, 0)),
                                    )
                                    if ssm_u32_rows < SSM_U32_CAP
                                    else u32_out
                                )
                                out_page[
                                    slot_o,
                                    t * GDN_PAGES : t * GDN_PAGES + REC_PAGES,
                                ] = pltpu.bitcast(
                                    u32_p.reshape(REC_PAGES, 2, 64, 1, 128), FP8
                                )
                                if conv_fast:
                                    conv_p = jnp.concatenate(
                                        [
                                            conv_head,
                                            jnp.zeros((conv8_rows - conv_live, 128), BF16),
                                        ],
                                        axis=0,
                                    )
                                else:
                                    conv_p = w_conv.reshape(conv8_rows, 128)
                                if conv8_rows < 256:
                                    conv_p = jnp.pad(
                                        conv_p, ((0, 256 - conv8_rows), (0, 0))
                                    )
                                out_page[slot_o, t * GDN_PAGES + REC_PAGES] = pltpu.bitcast(
                                    conv_p.reshape(2, 64, 2, 128), FP8
                                )
                            else:
                                # the state rows of slot t were written head by head above
                                cv = out_page.at[slot_o, t * GDN_PAGES + REC_PAGES].bitcast(
                                    BF16
                                ).reshape(256, 128)
                                if conv_fast:
                                    # Rows conv_live .. conv8_rows stay zero: gdn_out_page is
                                    # zeroed at the start of every call and only this store
                                    # writes conv pages (rows < conv_live).
                                    cv[0:conv_live, :] = conv_head
                                else:
                                    cv[:conv8_rows, :] = w_conv.reshape(conv8_rows, 128)
                if refill_in and not typed:
                    gdn_dma.start_gdn_in(o, s + NGI)
                gdn_dma.start_gdn_out(o, s)

            with sc("gdn_out"):
                scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
                yo = dmatmul(
                    "lin_out",
                    g0 + nt["lin_in"],
                    scr["xb2"],
                    H,
                    ctx=gctx(l, nt["lin_in"]),
                )
                return yo * sm(slot, "out_s")

        convb, recb, sem = scr["convb"], scr["recb"], scr["st_sem"]

        def rec_in():
            """Recurrent state loads. T > 1: sequence s's state goes to the slot of its token 0
            (row s * T), where its recurrence starts."""
            if not VER:
                return [pltpu.make_async_copy(ST["rec"].at[o], recb, sem.at[1])]
            return [
                pltpu.make_async_copy(
                    ST["rec"].at[o, pl.ds(s * vh, vh)],
                    recb.at[pl.ds(s * T * vh, vh)],
                    scr["rin_sem"].at[s],
                )
                for s in range(NS)
            ]

        with sc("gdn_in"):
            if GPR:  # staging only: independent of the convb / recb write-backs below
                gdn_dma.gpr_conv_start(o)
                for s_ in range(NRB):
                    gdn_dma.rec_chunk(o, s_, s_, False).start()

            @pl.when(o > 0)
            def _():
                pltpu.make_async_copy(
                    gdn_dma.conv_out_src(), O["conv"].at[0], sem.at[2]
                ).wait()
                if GCH:  # one write-back per buffer is still in flight
                    gdn_dma.rec_drain(0)
                else:
                    pltpu.make_async_copy(recb, O["rec"].at[0], sem.at[3]).wait()

            if not GPR:
                pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).start()
                if GCH:
                    for s_ in range(NRB):
                        gdn_dma.rec_chunk(o, s_, s_, False).start()
                else:
                    for d in rec_in():
                        d.start()
            y = dmatmul("lin_in", g0, scr["xb"], c.NLI, ctx=gctx(l, 0))
            ybuf = scr["ybuf"]
            ybuf[:, :c.NLI] = rnd(y * sm(slot, "in_s", c.NLI))
        with sc("gdn_conv"):
            if GPR:
                sub_ = lax.rem(o, ctx.G_PER_ATT)
                hr_rows_128 = c.HR * qkv_128
                conv_row_iota = lax.broadcasted_iota(I32, (c.HR, QKV), 0)
                for s_ in range(NS):
                    @pl.when(gdn_dma.is_gdn_in_active(sub_, s_))
                    def _():
                        gdn_dma.gpr_conv_copy(raw_refs[0], s_, 0, sem.at[0]).wait()
                for s_ in range(NS):
                    if typed:
                        raw_hr = pltpu.bitcast(scr["gpr_conv"][s_], BF16).reshape(
                            2 * GCONV_ROWS, 128
                        )
                    else:
                        raw_hr = scr["gpr_conv"].at[s_].bitcast(BF16).reshape(
                            2 * GCONV_ROWS, 128
                        )
                    raw_hr = raw_hr[:hr_rows_128, :].reshape(c.HR, QKV).astype(F32)
                    keep = jnp.logical_and(gdn_dma.is_gdn_in_active(sub_, s_), conv_row_iota < 3)
                    convb[s_] = jnp.where(keep, raw_hr, 0.0)
            else:
                pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).wait()
            ow, _ = lay["conv"]
            taps = [scr["sm"][slot, :, ow + t * QKV:ow + (t + 1) * QKV] for t in range(4)]
            u = None
            if VER:
                # history rows 3 + t <- token t's input; token t's window is rows t .. t+3
                # (the cached taps, then new inputs 0..t)
                us = []
                for s in range(NS):
                    convb[s, 3:3 + T, :] = ybuf[s * T:(s + 1) * T, 0:QKV]
                    w = [convb[s, k:k + T, :] for k in range(4)]
                    u_ = rnd(
                        jax.nn.silu(
                            w[0] * taps[0]
                            + w[1] * taps[1]
                            + w[2] * taps[2]
                            + w[3] * taps[3]
                        )
                    )
                    if GCH:
                        us.append(u_)
                    else:
                        scr["ub"][s, 0:T, :] = u_
                if GCH:
                    u = jnp.concatenate(us, axis=0)
            else:
                x = ybuf[0:B, 0:QKV]
                s0, s1, s2 = convb[0:B, :], convb[B:2 * B, :], convb[2 * B:3 * B, :]
                u = rnd(
                    jax.nn.silu(
                        s0 * taps[0] + s1 * taps[1] + s2 * taps[2] + x * taps[3]
                    )
                )
                convb[0:B, :] = s1
                convb[B:2 * B, :] = s2
                convb[2 * B:3 * B, :] = x
            if GPR:
                scr["convb16"][...] = convb[...].astype(BF16)
            pltpu.make_async_copy(
                gdn_dma.conv_out_src(), O["conv"].at[o], sem.at[2]
            ).start()
            if GCH:  # the chunk loop reads u rows from ybuf (x is consumed)
                ybuf[0:B, 0:QKV] = u
        if GCH:
            with sc("gdn_rec"):
                self.gdn_rec_chunked(o, slot)
        else:
            with sc("gdn_rec_wait"):
                for d in rec_in():
                    d.wait()
            if "gdn" not in skip:
                with sc("gdn_rec"):
                    self.gdn_rec(u, ybuf, slot)
        with sc("gdn_out"):
            if not GCH:
                pltpu.make_async_copy(recb, O["rec"].at[o], sem.at[3]).start()
            scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
            yo = dmatmul(
                "lin_out",
                g0 + nt["lin_in"],
                scr["xb2"],
                H,
                ctx=gctx(l, nt["lin_in"]),
            )
            return yo * sm(slot, "out_s")

    __call__ = gdn

    def gdn_rec(self, u, ybuf, slot):
        """Gated delta rule on the key-major bf16 states recb [B * vh, KD, VD] (in place), then
        the gated RMSNorm and SiLU(z) gate into mix[:B, :Z]. Key-major makes both state
        contractions sublane reductions and broadcasts delta along sublanes; q / k become
        lane-broadcast columns through one [VD, KD] transpose per key head.
        T > 1: token t of a sequence starts from token t-1's state (slot b - 1, just written
        as bf16) and token 0 from the loaded state (slot b); conv outputs come from ub."""
        ctx = self.ctx
        c, scr, sm = ctx.c, ctx.scr, ctx.sm
        B, T, VER = ctx.B, ctx.T, ctx.VER
        kh, vh, KD, VD = ctx.kh, ctx.vh, ctx.KD, ctx.VD
        QKV, Z, BA = ctx.QKV, ctx.Z, ctx.BA
        recb, sb = scr["recb"], scr["sbuf"]
        alog = sm(slot, "alog")[0, :vh]
        dtb = sm(slot, "dtb")[0, :vh]
        gnw = sm(slot, "gnorm")[0, :].astype(BF16)
        expand = vh // kh

        def cols(x):  # [kh, KD] -> [vh, KD, VD] holding x[h // expand, i] at [h, i, :]
            xc = jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)
            return jnp.broadcast_to(xc[:, None], (kh, expand, KD, VD)).reshape(vh, KD, VD)

        for b in range(B):
            src = b
            if VER:
                s_, t_ = divmod(b, T)
                urow_ = scr["ub"][s_, t_:t_ + 1, :]
                q = urow_[0, :kh * KD].reshape(kh, KD)
                k = urow_[0, kh * KD:2 * kh * KD].reshape(kh, KD)
                v = urow_[0, 2 * kh * KD:].reshape(vh, VD)
                src = b if t_ == 0 else b - 1
            else:
                q = u[b, :kh * KD].reshape(kh, KD)
                k = u[b, kh * KD:2 * kh * KD].reshape(kh, KD)
                v = u[b, 2 * kh * KD:].reshape(vh, VD)
            q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            q = q * KD ** -0.5
            k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            qc, kc_ = cols(q), cols(k)
            beta = rnd(jax.nn.sigmoid(ybuf[b, BA:BA + vh]))
            a = ybuf[b, BA + 64:BA + 64 + vh]
            g = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
            sb[...] = recb[src * vh:(src + 1) * vh].astype(F32) * jnp.exp(g)[:, None, None]
            pred = jnp.sum(sb[...] * kc_, axis=1)
            delta = (v - pred) * beta[:, None]
            sb[...] = sb[...] + kc_ * delta[:, None, :]
            out = rnd(jnp.sum(sb[...] * qc, axis=1))
            recb[b * vh:(b + 1) * vh] = sb[...].astype(BF16)
            val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + c.eps)
            val = (gnw[None, :] * val.astype(BF16)).astype(F32)
            zz = ybuf[b, QKV:QKV + Z].reshape(vh, VD)
            scr["mix"][b, :Z] = rnd(val * jax.nn.silu(zz)).reshape(Z)

    def gdn_rec_chunked(self, o, slot):
        """GCH: gdn_rec over row chunks. Chunk j of layer o lives in ring buffer j % NRB; the
        prologue (gdn) fetched chunks 0 .. NRB-1. Iteration j waits for chunk j, updates its
        rows, writes it back, then refills the buffer of chunk j - 1 (written back one
        iteration earlier) with chunk j - 1 + NRB. A fori_loop runs the NF full chunks, the
        TR-row tail chunk follows statically. The last NRB write-backs stay in flight until
        the next GatedDeltaNet layer (or the end of the kernel)."""
        ctx = self.ctx
        c, scr, sc, sm = ctx.c, ctx.scr, ctx.sc, ctx.sm
        gdn_dma = self.gdn_dma
        kh, vh, KD, VD = ctx.kh, ctx.vh, ctx.KD, ctx.VD
        QKV, Z, BA = ctx.QKV, ctx.Z, ctx.BA
        T, VER, skip = ctx.T, ctx.VER, ctx.skip
        CB, NF, TR, NCH, NRB = ctx.CB, ctx.NF, ctx.TR, ctx.NCH, ctx.NRB
        recb, sb, ybuf = scr["recb"], scr["sbuf"], scr["ybuf"]
        alog = sm(slot, "alog")[0, :vh]
        dtb = sm(slot, "dtb")[0, :vh]
        gnw = sm(slot, "gnorm")[0, :].astype(BF16)
        expand = vh // kh

        def colsk(x):  # [kh, KD] -> [kh, KD, VD] holding x[j, i] at [j, i, :]
            return jnp.swapaxes(jnp.broadcast_to(x[:, None, :], (kh, VD, KD)), 1, 2)

        # head-major state update (see the fused path): one pass per value head over its
        # [KD, VD] state, bf16 in / bf16 out; sbuf holds the q / k columns per kv head
        assert 2 * kh <= vh, (kh, vh)
        NCB = 2 if 4 * kh <= vh else 1
        hrow = lax.broadcasted_iota(I32, (vh, VD), 0)

        def rows(r0, n, s):  # rows r0 .. r0 + n - 1 (r0 8-aligned), states in buffer s
            uc = ybuf[pl.ds(r0, n), 0:QKV]
            bac = ybuf[pl.ds(r0, n), BA:BA + 128]
            zc = ybuf[pl.ds(r0, n), QKV:QKV + Z]
            st = recb.at[s]
            row_i = lax.broadcasted_iota(I32, (n, 1), 0)
            blk = jnp.zeros((n, Z), F32)
            for bb in range(n):
                h0 = bb * vh
                q = uc[bb, :kh * KD].reshape(kh, KD)
                k = uc[bb, kh * KD:2 * kh * KD].reshape(kh, KD)
                v = uc[bb, 2 * kh * KD:].reshape(vh, VD)
                q = q * lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                q = q * KD ** -0.5
                k = k * lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                cb = (bb % NCB) * 2 * kh
                sb[cb:cb + kh] = colsk(k)
                sb[cb + kh:cb + 2 * kh] = colsk(q)
                beta = rnd(jax.nn.sigmoid(bac[bb, 0:vh]))
                a = bac[bb, 64:64 + vh]
                g = -jnp.exp(alog) * jax.nn.softplus(a + dtb)
                eg = jnp.exp(g)[:, None, None]
                beta2 = beta[:, None]
                src = ((n - n // T + bb // T) if bb % T == 0 else bb - 1) * vh if VER else h0
                out_acc = jnp.zeros((vh, VD), F32)
                for h in range(vh):
                    j = h // expand
                    kc_h = sb[cb + j:cb + j + 1]
                    s_h = st[src + h:src + h + 1].astype(F32) * eg[h:h + 1]
                    pred_h = jnp.sum(s_h * kc_h, axis=1)
                    delta_h = (v[h:h + 1] - pred_h) * beta2[h:h + 1]
                    s_h = s_h + kc_h * delta_h[:, None, :]
                    qc_h = sb[cb + kh + j:cb + kh + j + 1]
                    out_acc = jnp.where(hrow == h, jnp.sum(s_h * qc_h, axis=1), out_acc)
                    st[h0 + h:h0 + h + 1] = s_h.astype(BF16)
                out = rnd(out_acc)
                val = out * lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + c.eps)
                val = (gnw[None, :] * val.astype(BF16)).astype(F32)
                zz = zc[bb].reshape(vh, VD)
                row = rnd(val * jax.nn.silu(zz)).reshape(1, Z)
                blk = jnp.where(row_i == bb, row, blk)
            scr["mix"][pl.ds(r0, n), 0:Z] = blk

        def chunk(j, carry):
            s = lax.rem(j, NRB)
            with sc("gdn_rec_wait"):
                gdn_dma.rec_chunk(o, j, s, False).wait()
            if "gdn" not in skip:
                rows(pl.multiple_of(j * CB, CB), CB, s)
            gdn_dma.rec_chunk(o, j, s, True).start()
            jp, jn = j - 1, j - 1 + NRB

            @pl.when(jnp.logical_and(j >= 1, jn < NCH))
            def _():
                sp = lax.rem(jp, NRB)
                gdn_dma.rec_chunk(o, jp, sp, True).wait()
                if TR:  # the refill may be the tail chunk (its own DMA size)
                    @pl.when(jn == NF)
                    def _():
                        gdn_dma.rec_chunk(o, NF, sp, False).start()

                    @pl.when(jn < NF)
                    def _():
                        gdn_dma.rec_chunk(o, jn, sp, False).start()
                else:
                    gdn_dma.rec_chunk(o, jn, sp, False).start()

            return carry

        lax.fori_loop(0, NF, chunk, 0)
        if TR:  # tail chunk NF: no refill follows (NRB >= 2)
            s = NF % NRB
            with sc("gdn_rec_wait"):
                gdn_dma.rec_chunk(o, NF, s, False).wait()
            if "gdn" not in skip:
                rows(NF * CB, TR, s)
            gdn_dma.rec_chunk(o, NF, s, True).start()
