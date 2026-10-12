"""Full and paged flash-decode attention block for the Qwen3.5 decode megakernel."""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .config import (
    BF16,
    F32,
    FP8,
    I32,
    NN,
    NT,
    U32,
    bswitch,
    pool_alias,
    rnd,
)
from .context import MKBuildContext
from .pool_dma import KVPoolDMA


class PagedAttentionBlock:
    """Full / paged flash-decode attention block."""

    def __init__(self, ctx: MKBuildContext, rings, kv_dma: KVPoolDMA | None = None):
        self.ctx = ctx
        self.rings = rings
        self.kv_dma = kv_dma if kv_dma is not None else KVPoolDMA(ctx)

    def group_copies(self, src, dst, ssem, rsem):
        """Remote copies src(off) -> dst on group peer me ^ off, off = 1 .. G-1. A peer's copy
        to this rank uses the same off, so it lands on rsem[off - 1] with the same size."""
        ctx = self.ctx
        me, G = ctx.me, ctx.G
        return [
            pltpu.make_async_remote_copy(
                src_ref=src(off),
                dst_ref=dst,
                send_sem=ssem.at[off - 1],
                recv_sem=rsem.at[off - 1],
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for off in range(1, G)
        ]

    def exchange(self, descs):
        for d in descs:
            d.start()
        for d in descs:
            d.wait_recv()
        for d in descs:
            d.wait_send()

    def init_worklist(self):
        """Populate this rank's KV work items in the step prologue."""
        ctx = self.ctx
        c, scr, ST, pos_ref = ctx.c, ctx.scr, ctx.ST, ctx.pos_ref
        NS, T, BLK, max_nb = ctx.NS, ctx.T, ctx.BLK, ctx.max_nb
        SPLIT, TS, gi = ctx.SPLIT, ctx.TS, ctx.gi
        n = jnp.int32(0)
        for b in range(NS):
            nb = jnp.minimum(jnp.maximum((pos_ref[b * T] + BLK - 1) // BLK, 0), max_nb)
            t0 = lax.rem(gi + (TS - b % TS), TS) if SPLIT else jnp.int32(0)
            cnt = jnp.maximum(nb - t0 + TS - 1, 0) // TS

            def put(j, carry, b=b, t0=t0, n=n):
                scr["wl_b"][n + j] = jnp.int32(b)
                scr["wl_t"][n + j] = t0 + j * TS
                if c.paged_kv:
                    pt_ref_ = ST["page_table"]
                    pb = BLK // c.page_size
                    p0 = (t0 + j * TS) * pb
                    k0 = (n + j) * pb
                    for p in range(pb):
                        col = jnp.minimum(p0 + p, c.max_pages_per_seq - 1)
                        scr["wl_kp"][k0 + p] = jnp.clip(
                            pt_ref_[b, col], 0, c.num_pages - 1
                        )
                return carry

            lax.fori_loop(0, cnt, put, 0)
            n = n + cnt
        scr["st"][1] = n

    def init_rope(self):
        """Populate step-level RoPE tables when QV and c.NATT > 0."""
        ctx = self.ctx
        c, scr, pos_ref = ctx.c, ctx.scr, ctx.pos_ref
        B, MR = ctx.B, ctx.MR
        if ctx.QV and c.NATT > 0:
            half_ = c.ROT // 2
            inv_ = 1.0 / (
                c.theta
                ** (
                    lax.broadcasted_iota(I32, (1, half_), 1).astype(F32)
                    * 2.0
                    / c.ROT
                )
            )
            rows_ = lax.broadcasted_iota(I32, (MR, half_), 0)
            pf = jnp.zeros((MR, half_), F32)
            for b in range(B):
                pf = jnp.where(rows_ == b, pos_ref[b].astype(F32), pf)
            ang_ = pf * inv_
            cos_t = rnd(jnp.cos(ang_))
            sin_t = rnd(jnp.sin(ang_))
            if ctx.ATT_QSWAP and c.ROT <= 128:
                def dup_(v):
                    parts = [v, v]
                    if c.ROT < 128:
                        parts.append(jnp.zeros((MR, 128 - c.ROT), F32))
                    return jnp.concatenate(parts, axis=-1)

                scr["rope"][0] = dup_(cos_t)
                scr["rope"][1] = dup_(sin_t)
            else:
                scr["rope"][0, :, :half_] = cos_t
                scr["rope"][1, :, :half_] = sin_t

    def attention(self, l, slot, g0, hook=None, ar0=None):
        ctx = self.ctx
        c, tiles, scr, ST, O = ctx.c, ctx.tiles, ctx.scr, ctx.ST, ctx.O
        pos_ref, raw_refs = ctx.pos_ref, ctx.raw_refs
        sc, sm, mm, qrow, rms_zc, arp_r = (
            ctx.sc,
            ctx.sm,
            ctx.mm,
            ctx.qrow,
            ctx.rms_zc,
            ctx.arp_r,
        )
        dmatmul, dtake, dview, dbanks_wait, arp_mm, flush_rel, fetch_rel = (
            self.rings.dmatmul,
            self.rings.dtake,
            self.rings.dview,
            self.rings.dbanks_wait,
            self.rings.arp_mm,
            self.rings.flush_rel,
            self.rings.fetch_rel,
        )
        kv_dma = self.kv_dma
        AE, B, T, NS, G, H = ctx.AE, ctx.B, ctx.T, ctx.NS, ctx.G, ctx.H
        nh, HD, NHH, QR, TQ = ctx.nh, ctx.HD, ctx.NHH, ctx.QR, ctx.TQ
        SPLIT, VER, PACK, QV, GQ = ctx.SPLIT, ctx.VER, ctx.PACK, ctx.QV, ctx.GQ
        SR, QG, MLW, NBP, PB, PW = ctx.SR, ctx.QG, ctx.MLW, ctx.NBP, ctx.PB, ctx.PW
        BLK, SUB, NKV, NKVR, KVE = ctx.BLK, ctx.SUB, ctx.NKV, ctx.NKVR, ctx.KVE
        KV_ITEM, EBROWS, PAIR_KV = ctx.KV_ITEM, ctx.EBROWS, ctx.PAIR_KV
        FUSED, LAZY_ATT, NBD = ctx.FUSED, ctx.LAZY_ATT, ctx.NBD
        nt, tk, skip, interpret, gi = ctx.nt, ctx.tk, ctx.skip, ctx.interpret, ctx.gi

        a = l // AE
        ksem = scr["kv_sem"]
        am, al, aacc = scr["am"], scr["al"], scr["aacc"]
        # KV stream: ring stage i % NKV holds work item i, block wl_t[i] of row wl_b[i] (listed
        # in the step prologue). Items depend only on pos: start before att_in.
        n_items = scr["st"][1]
        KR = NKVR if c.paged_kv else NKV  # ring stages (KVE: the NBE expert pool slots)
        if c.paged_kv:
            P_BLK = BLK // c.page_size

            if KVE:
                def kv_slot(s):  # expert pool slot s as (2, 64, BLK) u32 (K words, V words)
                    return pool_alias.view(
                        scr["epool"].at[pl.ds(s * EBROWS, KV_ITEM // 128)],
                        (2, 64, BLK),
                        U32,
                    )

                def kv_src(r, kp):  # raw page (2, 64, 4, ps) fp8 as (2, 64, ps) u32 words
                    return r.at[kp].bitcast(U32).reshape(2, 64, c.page_size)

                def kv_dst(s, p):
                    return kv_slot(s).at[:, :, pl.ds(p * c.page_size, c.page_size)]
            else:
                kvbuf = scr["kvbuf"]

                def kv_src(r, kp):
                    return r.at[kp]

                def kv_dst(s, p):
                    return kvbuf.at[s, :, :, :, pl.ds(p * c.page_size, c.page_size)]

            def kv_stage(s):  # whole ring stage s (destination of the item's P_BLK pages)
                return kv_slot(s) if KVE else kvbuf.at[s]

            def kv_start(i):
                @pl.when(jnp.logical_and(i < n_items, "kv" not in skip))
                def _():
                    s = i % KR
                    k0 = i * P_BLK
                    sem = ksem.at[s]
                    kp_tab = scr["wl_kp"]

                    def start_for_ref(r):
                        # unrolled (kv-dma-unroll): stage / semaphore addressing computed
                        # once; each page costs one SMEM load of its (step-prologue clamped)
                        # page id and a descriptor. Tail entries point at the null page; amask
                        # masks those columns.
                        for p in range(P_BLK):
                            kp = kp_tab[k0 + p]
                            pltpu.make_async_copy(kv_src(r, kp), kv_dst(s, p), sem).start()

                    bswitch(a, [lambda r=r: start_for_ref(r) for r in raw_refs])

            def kv_wait(i):
                if "kv" not in skip:
                    # the item's P_BLK page copies all signal ksem[s] and DMA semaphores count
                    # bytes: one wait sized to the whole stage (= sum of the page copies).
                    s = i % KR
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
                    zero = jnp.int32(0)  # waits need only the stage's sems / size
                    for d in kv_item(i, zero, zero):
                        d.wait()

            for i in range(NKV):
                kv_start(jnp.int32(i))

        n_ai, tk_ai = nt["att_in"], tk["att_in"]
        koff_g = nh * 2 * HD
        if n_ai <= min(ctx.dmm_unroll, NBD) and B not in ctx.DMM_UNROLL_OFF_B:
            GA = n_ai
        elif tiles.dense_acc_pair and n_ai % 2 == 0 and c.NAI > 512:
            GA = 2
        else:
            GA = 0
        GDEF = bool(
            ctx.ATT_GDEFER
            and QV
            and GA >= 2
            and HD % 128 == 0
            and c.NAI == koff_g + 2 * HD
            and "dense" not in skip
        )
        KVDEF = bool(GDEF and ctx.ATT_KVDEFER)

        def a_x(u):
            return scr["xb"][:, u * tk_ai : (u + 1) * tk_ai]

        def a_sum(ws, c0_, cw_):
            """att_in columns c0_ .. c0_ + cw_ in dmatmul's summation order."""
            p_ = None
            for u0 in range(0, n_ai, GA):
                st_ = mm(a_x(u0), ws[u0][:, c0_ : c0_ + cw_])
                for u in range(u0 + 1, u0 + GA):
                    st_ = st_ + mm(a_x(u), ws[u][:, c0_ : c0_ + cw_])
                p_ = st_ if p_ is None else p_ + st_
            return p_

        with sc("att_in"):
            ybuf = scr["ybuf"]
            if GDEF:
                # q (per head) and k / v columns now; the gate columns after the q exchange
                in_s_a = sm(slot, "in_s", c.NAI)
                if hook is not None:
                    # ARPIPE: q heads (and k / v unless KVDEF) own AR half first
                    ai_banks = dbanks_wait("att_in", g0, n_ai)
                    a_ws = [dview("att_in", b_) for b_ in ai_banks]
                    a_cols = [(h * 2 * HD, HD) for h in range(nh)]
                    if not KVDEF:
                        a_cols.append((koff_g, 2 * HD))
                    arp_mm("att_in", ai_banks, scr["xb"], a_cols, hook)
                    for c0_, cw_ in a_cols:
                        ybuf[:, c0_ : c0_ + cw_] = rnd(
                            arp_r(scr["acc"][:, c0_ : c0_ + cw_])
                            * in_s_a[:, c0_ : c0_ + cw_]
                        )
                else:
                    if getattr(ctx, "ATT_WGROUP", False) and n_ai <= ctx.NBD:
                        a_ws = self.rings.dtake_group(
                            [("att_in", g0 + u) for u in range(n_ai)]
                        )
                    else:
                        a_ws = [dtake("att_in", g0 + u) for u in range(n_ai)]
                    for h in range(nh):
                        c0_ = h * 2 * HD
                        ybuf[:, c0_ : c0_ + HD] = rnd(
                            a_sum(a_ws, c0_, HD) * in_s_a[:, c0_ : c0_ + HD]
                        )
                    if not KVDEF:  # KVDEF: k / v columns after the q exchange starts
                        ybuf[:, koff_g : koff_g + 2 * HD] = rnd(
                            a_sum(a_ws, koff_g, 2 * HD)
                            * in_s_a[:, koff_g : koff_g + 2 * HD]
                        )
            else:
                y = dmatmul(
                    "att_in",
                    g0,
                    scr["xb"],
                    c.NAI,
                    lazy=LAZY_ATT,
                    ctx=(a, AE - 1, 0, None),
                    hook=hook,
                )
                ybuf[:, :c.NAI] = rnd(arp_r(y) * sm(slot, "in_s", c.NAI))
        qn = sm(slot, "qn")
        kn = sm(slot, "kn")
        half = c.ROT // 2
        inv = 1.0 / (
            c.theta
            ** (lax.broadcasted_iota(I32, (1, half), 1).astype(F32) * 2.0 / c.ROT)
        )

        def rope(x, cos, sin):
            x1, x2, rest = x[:, :half], x[:, half:c.ROT], x[:, c.ROT:]
            return jnp.concatenate(
                (
                    rnd(rnd(x1 * cos) - rnd(x2 * sin)),
                    rnd(rnd(x2 * cos) + rnd(x1 * sin)),
                    rest,
                ),
                axis=-1,
            )

        koff = nh * 2 * HD
        rcs = scr["rcs"]
        with sc("att_qk"):
            if QV:
                # all B rows at once: one [B, HD] block per head (q, gate) and one for k / v; the
                # math per element is the per-row form's (same rnd points, row-wise rms)
                cos = scr["rope"][0, :B, :half]
                sin = scr["rope"][1, :B, :half]
                QSW = bool(ctx.ATT_QSWAP and c.ROT <= 128 and HD % 128 == 0)
                if QSW:
                    # partner lane of lane i < ROT: i + half (x1 lanes) / i - half (x2 lanes).
                    lane_ = lax.broadcasted_iota(I32, (B, 128), 1)
                    partner_ = jnp.where(lane_ < half, lane_ + half, lane_ - half)
                    from_a_ = pltpu.roll(lane_, half, 1) == partner_
                    in_rot_ = lane_ < c.ROT

                    def swap_(v):
                        return jnp.where(
                            from_a_,
                            pltpu.roll(v, half, 1),
                            pltpu.roll(v, 128 - half, 1),
                        )

                    cos_f = scr["rope"][0, :B, :]  # lanes half..ROT duplicate lanes 0..half-1
                    sin_f = scr["rope"][1, :B, :]
                    nsin_f = jnp.where(lane_ < half, -sin_f, sin_f)
                    w1_ = 1.0 + qn
                    w1s_ = swap_(jnp.broadcast_to(w1_[:, :128], (B, 128)))
                for h in range(nh):
                    q0_ = h * 2 * HD
                    if QSW:
                        # rms_zc / rope / scale of the per-row form, element for element
                        x_ = ybuf[:B, q0_ : q0_ + HD]
                        r_ = lax.rsqrt(
                            jnp.mean(x_ * x_, axis=-1, keepdims=True) + c.eps
                        )
                        n_ = rnd(x_ * r_ * w1_)
                        ns_ = rnd(swap_(x_[:, :128]) * r_ * w1s_)
                        n0_ = n_[:, :128]
                        ro_ = rnd(rnd(n0_ * cos_f) + rnd(ns_ * nsin_f))
                        qparts = [rnd(jnp.where(in_rot_, ro_, n0_) * HD**-0.5)]
                        for j in range(1, NHH):
                            qparts.append(
                                rnd(n_[:, j * 128 : (j + 1) * 128] * HD**-0.5)
                            )
                    else:
                        qh = rms_zc(ybuf[:B, q0_ : q0_ + HD], qn)
                        qh = rnd(rope(qh, cos, sin) * HD**-0.5)
                        qparts = [qh[:, j * 128 : (j + 1) * 128] for j in range(NHH)]
                    for j in range(NHH):  # row b, head h -> flattened row b * SR + h (== qrow)
                        scr["qg"][
                            0 if GQ else gi,
                            jnp.int32(j),
                            pl.ds(h, B, stride=SR),
                            :,
                        ] = qparts[j]
                if GQ:  # sequence s's (t, h) rows are flattened rows s * QG ..: one bf16 block
                    for sq in range(NS):
                        r0 = sq * QG
                        scr["qxb"][gi, sq] = jnp.concatenate(
                            [scr["qg"][0, j, r0:r0 + QG, :] for j in range(NHH)],
                            axis=-1,
                        ).astype(BF16)
            else:
                for b in range(B):  # q pass: the group q exchange starts right after it
                    pb = pos_ref[b]
                    ang = pb.astype(F32) * inv
                    cos, sin = rnd(jnp.cos(ang)), rnd(jnp.sin(ang))
                    rcs[0, b:b + 1, :] = cos
                    rcs[1, b:b + 1, :] = sin
                    qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                    q = rms_zc(qg[:, :HD], qn)
                    q = rnd(rope(q, cos, sin) * HD ** -0.5)
                    if PACK:
                        t_, o_ = qrow(b)
                        scr["qg"][gi, t_, o_:o_ + nh] = q
                    else:
                        if nh < 8:
                            q = jnp.concatenate((q, jnp.zeros((8 - nh, HD), F32)), axis=0)
                        if SPLIT or VER:
                            scr["qg"][gi, b] = q
                        else:
                            scr["qb"][b] = q.astype(BF16)
        qx = []
        if SPLIT:
            with sc("att_qgather"):
                qgb = scr["qxb"] if GQ else scr["qg"]
                qx = self.group_copies(
                    lambda off: qgb.at[gi], qgb.at[gi], scr["sq_ssem"], scr["sq_rsem"]
                )
                for d in qx:
                    d.start()
        # independent of the remote q: overlaps the q exchange round trip
        if FUSED:
            with sc("att_kvrd"):
                kv_dma.kvrd_issue(a)
        with sc("att_kvnew"):
            if QV:  # k / v / gate pass on [B, HD] blocks
                if KVDEF:
                    # deferred att_in k / v columns: MXU work inside the q exchange round trip
                    in_s_a = sm(slot, "in_s", c.NAI)
                    ybuf[:, koff_g : koff_g + 2 * HD] = rnd(
                        arp_r(a_sum(a_ws, koff_g, 2 * HD))
                        * in_s_a[:, koff_g : koff_g + 2 * HD]
                    )
                cos = scr["rope"][0, :B, :half]
                sin = scr["rope"][1, :B, :half]
                if not GDEF:  # GDEF: the gate columns are computed after this pass
                    for h in range(nh):
                        q0_ = h * 2 * HD
                        scr["gate"][:B, h * HD : (h + 1) * HD] = ybuf[
                            :B, q0_ + HD : q0_ + 2 * HD
                        ]
                k = rms_zc(ybuf[:B, koff:koff + HD], kn)
                v = ybuf[:B, koff + HD:koff + 2 * HD]
                k = rope(k, cos, sin)
                if c.paged_kv:
                    k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                    v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                    scr["knew_o"][:B, :] = k_fp8
                    scr["vnew_o"][:B, :] = v_fp8
                    scr["knew"][:B, :] = k_fp8.astype(F32)
                    scr["vnew"][:B, :] = v_fp8.astype(F32)
                else:
                    scr["knew"][:B, :] = k
                    scr["vnew"][:B, :] = v
            else:
                for b in range(B):  # k / v / gate pass
                    cos, sin = rcs[0, b:b + 1, :], rcs[1, b:b + 1, :]
                    qg = ybuf[b, :koff].reshape(nh, 2 * HD)
                    scr["gate"][b, :] = qg[:, HD:].reshape(nh * HD)
                    k = rms_zc(ybuf[b:b + 1, koff:koff + HD], kn)
                    v = ybuf[b:b + 1, koff + HD:koff + 2 * HD]
                    k = rope(k, cos, sin)
                    if c.paged_kv:
                        k_fp8 = jnp.clip(k, -448.0, 448.0).astype(FP8)
                        v_fp8 = jnp.clip(v, -448.0, 448.0).astype(FP8)
                        scr["knew_o"][b:b + 1, :] = k_fp8
                        scr["vnew_o"][b:b + 1, :] = v_fp8
                        scr["knew"][b:b + 1, :] = k_fp8.astype(F32)
                        scr["vnew"][b:b + 1, :] = v_fp8.astype(F32)
                    else:
                        scr["knew"][b:b + 1, :] = k
                        scr["vnew"][b:b + 1, :] = v
            if not FUSED:
                k_src = scr["knew_o"] if c.paged_kv else scr["knew"]
                v_src = scr["vnew_o"] if c.paged_kv else scr["vnew"]
                kv_out = (
                    pltpu.make_async_copy(
                        k_src, O["knew"].at[a], scr["kv_out_sem"].at[0]
                    ),
                    pltpu.make_async_copy(
                        v_src, O["vnew"].at[a], scr["kv_out_sem"].at[1]
                    ),
                )
                for d in kv_out:
                    d.start()
        if GDEF:
            # deferred att_in gate columns: MXU work inside the q exchange round trip
            with sc("att_gate_mm"):
                in_s_a = sm(slot, "in_s", c.NAI)
                for h in range(nh):
                    c0_ = h * 2 * HD + HD
                    g_ = rnd(
                        arp_r(a_sum(a_ws, c0_, HD)) * in_s_a[:, c0_ : c0_ + HD]
                    )
                    scr["gate"][:B, h * HD : (h + 1) * HD] = g_[:B]
                if not LAZY_ATT:  # att_in slots held until here: refill them now
                    for u in range(n_ai):
                        fetch_rel(g0 + u + NBD, a, AE - 1, u + NBD)
        if SPLIT or VER:
            with sc("att_qgather"):
                qgb = scr["qg"]
                for d in qx:
                    d.wait_recv()
                for d in qx:
                    d.wait_send()
                if GQ:  # KV row sq's query rows: (group rank g, token t, head h)
                    for sq in range(NS):
                        for g in range(G):
                            scr["qb"][sq, g * QG:(g + 1) * QG] = scr["qxb"][g, sq]

                def qg_rows(g, r0, n):  # n query rows from flattened row r0 of group rank g
                    if not QV:
                        t_, o_ = divmod(r0, 8)
                        return qgb[g, t_, o_:o_ + n, :]
                    return jnp.concatenate(
                        [qgb[g, j, r0:r0 + n, :] for j in range(NHH)], axis=-1
                    )

                def group_q(b):  # row b's G * nh group query heads, zero-padded to QR rows
                    if not SPLIT:
                        return qg_rows(0, b * 8, 8)
                    t_, o_ = qrow(b)
                    qa = jnp.concatenate(
                        [qg_rows(g, t_ * 8 + o_, nh) for g in range(G)], axis=0
                    )
                    if G * nh < QR:
                        qa = jnp.concatenate(
                            (qa, jnp.zeros((QR - G * nh, HD), F32)), 0
                        )
                    return qa

                for sq in range(NS if not GQ else 0):  # token t: query rows t * QR .. of qb[sq]
                    qs = [group_q(sq * T + t) for t in range(T)]
                    qs = qs[0] if T == 1 else jnp.concatenate(qs, axis=0)
                    scr["qb"][sq] = qs.astype(BF16)

        with sc("att_core"):
            AMASK = tiles.amask
            col = lax.broadcasted_iota(I32, (1 if AMASK else TQ, SUB), 1)
            col_v = (
                lax.broadcasted_iota(I32, (HD, SUB), 1)
                if (c.paged_kv and AMASK == 0)
                else None
            )
            am[...] = jnp.full(am.shape, -1e30, F32)
            al[...] = jnp.zeros(al.shape, F32)
            aacc[...] = jnp.zeros(aacc.shape, F32)

            def item_front(i):
                """Scores of work item i (KV row b = wl_b[i], block t = wl_t[i]) for the TQ
                query rows of its T tokens, masked to the cached positions (amask mode), with
                their row maxima and the (masked) V sub-blocks: everything that does not read
                row b's online-softmax state."""
                s = i % KR
                b, t = scr["wl_b"][i], scr["wl_t"][i]
                plen = pos_ref[b * T] if VER else pos_ref[b]
                qrows = scr["qb"][b]
                if c.paged_kv:
                    if interpret:
                        k_full = kvbuf[s, 0].reshape(HD, BLK)
                        v_full = kvbuf[s, 1].reshape(HD, BLK)
                    elif KVE:
                        kst = kv_slot(s)
                        k_u32 = kst[0]
                        v_u32 = kst[1]
                    else:
                        k_u32 = kvbuf.at[s, 0].bitcast(U32).reshape(64, BLK)[...]
                        v_u32 = kvbuf.at[s, 1].bitcast(U32).reshape(64, BLK)[...]
                    if not interpret:
                        k_full = pltpu.bitcast(k_u32, FP8)
                        if AMASK in (0, 3, 4):
                            v_full = pltpu.bitcast(v_u32, FP8)
                outs = []
                for j in range(BLK // SUB):
                    off = t * BLK + j * SUB
                    rem = plen - off
                    full_blk = off + SUB <= plen
                    if c.paged_kv:
                        kj = k_full if SUB == BLK else k_full[:, j * SUB:(j + 1) * SUB]
                        if AMASK == 0:
                            vj = v_full if SUB == BLK else v_full[:, j * SUB:(j + 1) * SUB]
                            vj = jnp.where(col_v + off < plen, vj, jnp.zeros_like(vj))
                        elif interpret:
                            vj = v_full if SUB == BLK else v_full[:, j * SUB:(j + 1) * SUB]
                            if AMASK not in (3, 4):
                                vj = jnp.where(col < rem, vj, jnp.zeros_like(vj))
                        elif AMASK == 1:
                            vu = v_u32 if SUB == BLK else v_u32[:, j * SUB:(j + 1) * SUB]
                            vu = jnp.where(col < rem, vu, jnp.uint32(0))
                            vj = pltpu.bitcast(vu, FP8)
                        elif AMASK == 2:
                            vu = v_u32 if SUB == BLK else v_u32[:, j * SUB:(j + 1) * SUB]
                            vu = lax.cond(
                                full_blk,
                                lambda: vu,
                                lambda: jnp.where(col < rem, vu, jnp.uint32(0)),
                            )
                            vj = pltpu.bitcast(vu, FP8)
                        else:
                            vj = v_full if SUB == BLK else v_full[:, j * SUB:(j + 1) * SUB]
                        scores = mm(qrows, kj, NN)
                    else:
                        kj = kbuf[s] if SUB == BLK else kbuf[s, j * SUB:(j + 1) * SUB]
                        vj = vbuf[s] if SUB == BLK else vbuf[s, j * SUB:(j + 1) * SUB]
                        scores = mm(qrows, kj, NT)
                    if AMASK == 0:
                        scores = jnp.where(col + off < plen, scores, -jnp.inf)
                    elif AMASK in (1, 4):
                        scores = jnp.where(col < rem, scores, -jnp.inf)
                    else:
                        scores = lax.cond(
                            full_blk,
                            lambda: scores,
                            lambda: jnp.where(col < rem, scores, -jnp.inf),
                        )
                    rmax = jnp.max(scores, axis=-1, keepdims=True)
                    outs.append((scores, rmax, vj))
                return b, outs

            def item_back(b, outs):
                """Online-softmax update of KV row b's state with item_front's sub-blocks."""
                for scores, rmax, vj in outs:
                    m_old = am[b]
                    m_new = jnp.maximum(m_old, rmax)
                    alpha = jnp.exp(m_old - m_new)
                    pr = jnp.exp(scores - m_new[:, 0:1])
                    pv = mm(pr.astype(BF16), vj, NT if c.paged_kv else NN)
                    al[b] = al[b] * alpha + jnp.sum(pr, axis=-1, keepdims=True)
                    aacc[b] = aacc[b] * alpha[:, 0:1] + pv
                    am[b] = m_new

            def item(i, carry):
                """Online-softmax step of KV row (sequence) b = wl_b[i] over its KV block
                t = wl_t[i], for the TQ query rows of its T tokens (cached positions only: the
                new tokens are added in att_new)."""
                kv_wait(i)
                if "attn" not in skip:
                    item_back(*item_front(i))
                kv_start(i + KR)
                return carry

            def item_pair(p, carry):
                """Items i0 = 2p and i1 = 2p + 1 in one barrier-free region: i1's K/V loads,
                QK matmul, mask and row max (item_front) do not read the softmax state, so the
                scheduler can overlap them with i0's exp / PV / state update. State updates
                stay in item order (i0 and i1 may share a KV row)."""
                i0 = 2 * p
                i1 = i0 + 1
                kv_wait(i0)
                kv_wait(i1)
                if "attn" not in skip:
                    f0 = item_front(i0)
                    f1 = item_front(i1)
                    item_back(*f0)
                    item_back(*f1)
                kv_start(i0 + KR)
                kv_start(i1 + KR)
                return carry

            if PAIR_KV:
                lax.fori_loop(0, lax.div(n_items, 2), item_pair, 0)

                @pl.when(lax.rem(n_items, 2) == 1)
                def _():
                    item(n_items - 1, 0)
            else:
                lax.fori_loop(0, n_items, item, 0)

        def gq_merge_start():
            psend, pri = scr["psend"], scr["pri"]
            with sc("att_merge"):
                # rows g * QG .. of every KV row's softmax state are group rank g's heads (all
                # T tokens): send that aacc block as is, plus one tile whose lanes
                # s * MLW .. (m) and 128 + s * MLW .. (l) hold sequence s's (m, l) (both are
                # lane-broadcast in am / al), to rank g in one contiguous copy
                lane = lax.broadcasted_iota(I32, (QG, 128), 1)
                for g in range(G):
                    rs = slice(g * QG, (g + 1) * QG)
                    for s_ in range(NS):
                        psend[g, s_] = aacc[s_, rs]
                    xm, xl = am[0, rs], al[0, rs]
                    for s_ in range(1, NS):
                        hi = lane >= s_ * MLW
                        xm = jnp.where(hi, am[s_, rs], xm)
                        xl = jnp.where(hi, al[s_, rs], xl)
                    psend[g, NS] = jnp.concatenate((xm, xl), axis=-1)
                pri[gi] = psend[gi]
                mx = self.group_copies(
                    lambda off: psend.at[gi ^ off],
                    pri.at[gi],
                    scr["sp_ssem"],
                    scr["sp_rsem"],
                )
                for d in mx:
                    d.start()
            return mx

        def slab_merge_start():
            with sc("att_merge"):
                # rows g * nh .. of a token's group query rows are group rank g's heads: send
                # them (with m in lanes HD.., l in HD + 64..) to rank g, merge what arrives
                pso, pri = scr["pso"], scr["pri"]
                lane_pw = lax.broadcasted_iota(I32, (nh, 128), 1)
                for b in range(B):
                    t_, o_ = qrow(b)
                    sq, q0 = b // T, (b % T) * QR  # row b's query rows in KV row sq
                    mb, lb = am[sq, q0:q0 + QR], al[sq, q0:q0 + QR]
                    ab = aacc[sq, q0:q0 + QR]
                    for g in range(G):
                        r0 = g * nh
                        pso[g, t_, o_:o_ + nh, 0:HD] = ab[r0:r0 + nh]
                        pso[g, t_, o_:o_ + nh, HD:PW] = jnp.where(
                            lane_pw < 64, mb[r0:r0 + nh], lb[r0:r0 + nh]
                        )
                pri[gi] = pso[gi]
                mx = self.group_copies(
                    lambda off: pso.at[gi ^ off],
                    pri.at[gi],
                    scr["sp_ssem"],
                    scr["sp_rsem"],
                )
                for d in mx:
                    d.start()
            return mx

        if SPLIT:
            mx = gq_merge_start() if GQ else slab_merge_start()
            pri = scr["pri"]
            # independent of the remote partials: overlaps the exchange round trip
            if FUSED:
                with sc("att_kvwr"):
                    kv_dma.kv_rmw(a)
            with sc("att_gate"):  # output gates of all rows (att_new reads them)
                scr["gate"][0:B, :] = rnd(jax.nn.sigmoid(scr["gate"][0:B, :]))
            with sc("att_merge"):
                for d in mx:
                    d.wait_recv()
                for d in mx:
                    d.wait_send()
                for t in range(NBP if not GQ else 0):  # PACK: one merge per tile of PB rows
                    parts = [pri[g, t] for g in range(G)]
                    mg = [x[:, HD:HD + 1] for x in parts]
                    m = mg[0]
                    for x in mg[1:]:
                        m = jnp.maximum(m, x)
                    wg = [jnp.exp(x - m) for x in mg]
                    lsum = sum(w * x[:, HD + 64:HD + 65] for w, x in zip(wg, parts))
                    asum = sum(w * x[:, :HD] for w, x in zip(wg, parts))
                    if not PACK:  # tile t is row t
                        sq, q0 = t // T, (t % T) * QR
                        am[sq, q0:q0 + 8] = jnp.broadcast_to(m, (8, 128))
                        al[sq, q0:q0 + 8] = jnp.broadcast_to(lsum, (8, 128))
                        aacc[sq, q0:q0 + 8] = asum
                        continue
                    for b in range(t * PB, min(B, (t + 1) * PB)):
                        o_ = qrow(b)[1]
                        sq, q0 = b // T, (b % T) * QR
                        am[sq, q0:q0 + nh] = jnp.broadcast_to(
                            m[o_:o_ + nh], (nh, 128)
                        )
                        al[sq, q0:q0 + nh] = jnp.broadcast_to(
                            lsum[o_:o_ + nh], (nh, 128)
                        )
                        aacc[sq, q0:q0 + nh] = asum[o_:o_ + nh]

        with sc("att_new"):
            if GQ:
                pri = scr["pri"]
                mlt = [pri[g, NS] for g in range(G)]
                row_q = lax.broadcasted_iota(I32, (QG, 1), 0)  # row r is token r // nh
                for sq in range(NS):
                    # merge of the G partials of own heads (same arithmetic as the slab merge)
                    f = sq * MLW
                    parts = [pri[g, sq] for g in range(G)]
                    mg = [x[:, f:f + 1] for x in mlt]
                    lg = [x[:, 128 + f:128 + f + 1] for x in mlt]
                    m = mg[0]
                    for x in mg[1:]:
                        m = jnp.maximum(m, x)
                    wg = [jnp.exp(x - m) for x in mg]
                    lsum = sum(w * x for w, x in zip(wg, lg))
                    asum = sum(w * x for w, x in zip(wg, parts))
                    r0 = sq * QG  # own (t, h) query rows (bf16-exact), flattened rows r0 ..
                    qblk = jnp.concatenate(
                        [scr["qg"][0, j, r0:r0 + QG, :] for j in range(NHH)], axis=-1
                    )
                    # new tokens, all T tokens' rows at once: row r (token r // nh) takes new
                    # token j's update iff j <= r // nh, so each row sees exactly the same update
                    # sequence as the per-token chain (j = 0 .. t)
                    m_old, l_old, a_old = m, lsum, asum
                    for j in range(T):
                        bj = sq * T + j
                        s_new = jnp.sum(
                            qblk * scr["knew"][bj:bj + 1, :], axis=-1, keepdims=True
                        )
                        m_new = jnp.maximum(m_old, s_new)
                        alpha = jnp.exp(m_old - m_new)
                        p_new = jnp.exp(s_new - m_new)
                        l_new = l_old * alpha + p_new
                        a_new = a_old * alpha + p_new * scr["vnew"][bj:bj + 1, :]
                        if j == 0:
                            m_old, l_old, a_old = m_new, l_new, a_new
                        else:
                            upd = row_q >= j * nh
                            m_old = jnp.where(upd, m_new, m_old)
                            l_old = jnp.where(upd, l_new, l_old)
                            a_old = jnp.where(upd, a_new, a_old)
                    ob = rnd(a_old / l_old)  # park rnd(out): flattened rows b * nh + h
                    for j in range(NHH):
                        scr["ost"][j, r0:r0 + QG, :] = ob[:, j * 128:(j + 1) * 128]
            else:
                NR = nh if PACK else 8  # softmax-state rows: rows 0..nh-1 are own heads
                for b in range(B):
                    # the new tokens' columns; rows q0 .. q0+NR-1 of KV row sq's softmax state are
                    # row b's (own heads in rows q0 .. q0+nh-1)
                    sq, tb = divmod(b, T)
                    q0 = tb * QR
                    m_old = am[sq, q0:q0 + NR][:, 0:1]
                    l_old = al[sq, q0:q0 + NR][:, 0:1]
                    a_old = aacc[sq, q0:q0 + NR]
                    if QV:  # PACK: rows b * SR .. + nh; else the row's 8-row tile (rows nh.. zero)
                        qf = jnp.concatenate(
                            [scr["qg"][gi, j, b * SR:b * SR + NR, :] for j in range(NHH)],
                            axis=-1,
                        )
                    elif PACK:
                        t_, o_ = qrow(b)
                        qf = scr["qg"][gi, t_, o_:o_ + nh]
                    else:
                        qf = (
                            scr["qg"][gi, b]
                            if SPLIT or VER
                            else scr["qb"][b].astype(F32)
                        )
                    if VER:  # token tb sees new tokens 0..tb of its sequence
                        for j in range(tb + 1):
                            bj = sq * T + j
                            s_new = jnp.sum(
                                qf * scr["knew"][bj:bj + 1, :], axis=-1, keepdims=True
                            )
                            m_new = jnp.maximum(m_old, s_new)
                            alpha = jnp.exp(m_old - m_new)
                            p_new = jnp.exp(s_new - m_new)
                            l_old = l_old * alpha + p_new
                            a_old = a_old * alpha + p_new * scr["vnew"][bj:bj + 1, :]
                            m_old = m_new
                        outb = a_old / l_old
                    else:
                        s_new = jnp.sum(
                            qf * scr["knew"][b:b + 1, :], axis=-1, keepdims=True
                        )
                        m_new = jnp.maximum(m_old, s_new)
                        alpha = jnp.exp(m_old - m_new)
                        p_new = jnp.exp(s_new - m_new)
                        l_tot = l_old * alpha + p_new
                        outb = (
                            a_old * alpha + p_new * scr["vnew"][b:b + 1, :]
                        ) / l_tot
                    if QV:  # park rnd(out) of the own heads as flattened rows b * SR + h
                        ob = rnd(outb[:nh])
                        for j in range(NHH):
                            scr["ost"][j, b * SR:b * SR + nh, :] = ob[
                                :, j * 128:(j + 1) * 128
                            ]
                        continue
                    for h in range(nh):
                        oh = rnd(outb[h:h + 1, :])
                        gh = scr["gate"][b:b + 1, h * HD:(h + 1) * HD]
                        if not SPLIT:
                            gh = rnd(jax.nn.sigmoid(gh))
                        scr["mix"][b:b + 1, h * HD:(h + 1) * HD] = rnd(oh * gh)
            if QV:  # gate, product and rounding on [B, HD] blocks: head h of all rows at once
                for h in range(nh):
                    oh = jnp.concatenate(
                        [
                            scr["ost"][jnp.int32(j), pl.ds(h, B, stride=SR), :]
                            for j in range(NHH)
                        ],
                        axis=-1,
                    )
                    gh = scr["gate"][:B, h * HD:(h + 1) * HD]
                    if not SPLIT:
                        gh = rnd(jax.nn.sigmoid(gh))
                    scr["mix"][:B, h * HD:(h + 1) * HD] = rnd(oh * gh)

        with sc("att_out"):
            if FUSED and not SPLIT:
                kv_dma.kv_rmw(a)
            elif not FUSED:
                for d in kv_out:
                    d.wait()
            NO = nh * HD
            scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
            yo = dmatmul(
                "att_out",
                g0 + nt["att_in"],
                scr["xb2"],
                H,
                lazy=LAZY_ATT,
                ctx=(a, AE - 1, nt["att_in"], None),
                split=ar0,
            )
        if LAZY_ATT:
            with sc("refill"):
                # g0 = tile 0 of block a's attention layer (lt = AE - 1): decode at trace time
                flush_rel(g0, a, AE - 1, 0, nt["att_in"] + nt["att_out"])
        return None if ar0 is not None else yo * sm(slot, "out_s")

    __call__ = attention
