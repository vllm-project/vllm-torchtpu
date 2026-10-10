"""GDN and KV pool DMA managers for the Qwen3.5 decode megakernel."""

from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .config import (
    BF16,
    F32,
    FP8,
    I32,
    U32,
    bswitch,
)
from .context import MKBuildContext


class GprRecIn:
    """GPR: rec_chunk input of chunk j (buffer s): the chunk's sequences' rec pages ->
    gpr_rec[s]; wait() also unpacks them (inactive: zeros) into the head slots of
    recb[s] the gathered-copy DMA fills (the last n // T slots)."""

    def __init__(self, dma: "GDNPoolDMA", o, j, s, n):
        self.dma = dma
        ctx = dma.ctx
        self.o, self.s, self.n = o, s, n
        self.nseq = n // (ctx.T * ctx.vh)
        self.seqs = [j * (ctx.CB // ctx.T) + q for q in range(self.nseq)]
        self.dsts = [ctx.scr["gpr_rec"].at[s, q] for q in range(self.nseq)]

    def start(self):
        ctx = self.dma.ctx
        self.dma.gpr_start(
            self.o,
            self.seqs,
            self.dsts,
            ctx.scr["rec_sem"].at[0, self.s],
            0,
            ctx.REC_PAGES,
        )

    def wait(self):
        ctx = self.dma.ctx
        scr = ctx.scr
        T, vh, KD, VD = ctx.T, ctx.vh, ctx.KD, ctx.VD
        SSM_U32_CAP, ssm_u32_rows = ctx.SSM_U32_CAP, ctx.ssm_u32_rows
        sub = lax.rem(self.o, ctx.G_PER_ATT)
        s, n = self.s, self.n
        for sq, dst in zip(self.seqs, self.dsts):
            self.dma.gpr_wait(sub, sq, dst, scr["rec_sem"].at[0, s], ctx.REC_PAGES)
        for q, sq in enumerate(self.seqs):
            if ctx.typed:
                u32 = pltpu.bitcast(scr["gpr_rec"][s, q], U32).reshape(
                    SSM_U32_CAP, 128
                )[:ssm_u32_rows, :]
            else:
                u32 = scr["gpr_rec"].at[s, q].bitcast(U32).reshape(
                    SSM_U32_CAP, 128
                )[:ssm_u32_rows, :]
            u32 = jnp.where(self.dma.is_gdn_in_active(sub, sq), u32, jnp.uint32(0))
            scr["recb"][s, pl.ds(n - n // T + q * vh, vh)] = pltpu.bitcast(
                u32, BF16
            ).reshape(vh, KD, VD)


class GDNPoolDMA:
    """GDN state and conv page DMA manager."""

    def __init__(self, ctx: MKBuildContext):
        self.ctx = ctx

    def is_seq_active(self, s):
        ctx = self.ctx
        c, ST = ctx.c, ctx.ST
        if c.act_mode == "ref":
            return ST["active_seqs"][s] != 0
        if c.act_mode == "from_ckpt":
            return (
                (ST["ckpt_indices"][0, s, 0] > 0)
                if ctx.ckpt_is_3d
                else (ST["ckpt_indices"][s, 0] > 0)
            )
        if c.act_mode == "pos":
            return ctx.pos_ref[s * ctx.T] >= 0
        return jnp.bool_(True)

    def gdn_roff(self, sub, s):
        ST = self.ctx.ST
        return ST["read_offsets"][sub, s] if self.ctx.roff_is_2d else ST["read_offsets"][s]

    def is_gdn_in_active(self, sub, sq):
        return jnp.logical_and(self.is_seq_active(sq), self.gdn_roff(sub, sq) >= 0)

    def gdn_pb_in(self, sub, s):
        ctx = self.ctx
        ST = ctx.ST
        GDN_PAGES, G_PER_ATT = ctx.GDN_PAGES, ctx.G_PER_ATT
        roff = self.gdn_roff(sub, s)
        if ctx.ckpt_is_3d:
            return ST["ckpt_indices"][sub, s, roff] * GDN_PAGES
        return ST["ckpt_indices"][s, roff] * (GDN_PAGES * G_PER_ATT) + sub * GDN_PAGES

    def gdn_pb_out(self, sub, s, t):
        ctx = self.ctx
        ST = ctx.ST
        GDN_PAGES, G_PER_ATT, ckpt_t_off = ctx.GDN_PAGES, ctx.G_PER_ATT, ctx.ckpt_t_off
        if ctx.ckpt_is_3d:
            return ST["ckpt_indices"][sub, s, t + ckpt_t_off] * GDN_PAGES
        return (
            ST["ckpt_indices"][s, t + ckpt_t_off] * (GDN_PAGES * G_PER_ATT)
            + sub * GDN_PAGES
        )

    def gdn_contig_ckpt(self, sub, s):
        ctx = self.ctx
        ST = ctx.ST
        T, ckpt_t_off = ctx.T, ctx.ckpt_t_off
        if not ctx.ckpt_is_3d or T <= 1:
            return jnp.bool_(T == 1)
        s0 = ST["ckpt_indices"][sub, s, ckpt_t_off]
        ok = jnp.bool_(True)
        for t in range(1, T):
            ok = jnp.logical_and(
                ok, ST["ckpt_indices"][sub, s, t + ckpt_t_off] == s0 + t
            )
        return ok

    def start_gdn_in(self, o_, s):
        ctx = self.ctx
        scr, raw_refs = ctx.scr, ctx.raw_refs
        NGI, G_PER_ATT, GDN_PAGES = ctx.NGI, ctx.G_PER_ATT, ctx.GDN_PAGES
        in_page = scr["gdn_in_page"]
        in_sem = scr["gdn_in_sem"]
        in_pend = scr["gdn_in_pend"]
        slot_i = s % NGI
        sub = lax.rem(o_, G_PER_ATT)
        roff = self.gdn_roff(sub, s)
        act_in = jnp.logical_and(self.is_seq_active(s), roff >= 0)
        in_pend[slot_i] = jnp.where(act_in, 1, 0).astype(I32)

        @pl.when(act_in)
        def _():
            g_pool = lax.div(o_, G_PER_ATT)
            pb = self.gdn_pb_in(sub, s)

            def do_in(r):
                pltpu.make_async_copy(
                    r.at[pl.ds(pb, GDN_PAGES)],
                    in_page.at[slot_i],
                    in_sem.at[slot_i],
                ).start()

            bswitch(g_pool, [lambda r=r: do_in(r) for r in raw_refs])

    def wait_gdn_in(self, slot_i):
        ctx = self.ctx
        scr, raw_refs, GDN_PAGES = ctx.scr, ctx.raw_refs, ctx.GDN_PAGES
        in_page = scr["gdn_in_page"]
        in_sem = scr["gdn_in_sem"]
        in_pend = scr["gdn_in_pend"]
        pend = in_pend[slot_i]

        @pl.when(pend != 0)
        def _():
            pltpu.make_async_copy(
                raw_refs[0].at[pl.ds(0, GDN_PAGES)],
                in_page.at[slot_i],
                in_sem.at[slot_i],
            ).wait()

        in_pend[slot_i] = jnp.int32(0)
        return pend != 0

    def start_gdn_out(self, o_, s):
        ctx = self.ctx
        scr, raw_refs = ctx.scr, ctx.raw_refs
        NGO, G_PER_ATT, GDN_PAGES, T = ctx.NGO, ctx.G_PER_ATT, ctx.GDN_PAGES, ctx.T
        out_page = scr["gdn_out_page"]
        out_sem = scr["gdn_out_sem"]
        out_mode = scr["gdn_out_mode"]
        slot_o = s % NGO
        act_out = self.is_seq_active(s)
        sub = lax.rem(o_, G_PER_ATT)
        g_pool = lax.div(o_, G_PER_ATT)
        if T == 1:
            out_mode[slot_o] = jnp.where(act_out, 1, 0).astype(I32)

            @pl.when(act_out)
            def _():
                pb0 = self.gdn_pb_out(sub, s, 0)

                def do_out(r):
                    pltpu.make_async_copy(
                        out_page.at[slot_o],
                        r.at[pl.ds(pb0, GDN_PAGES)],
                        out_sem.at[slot_o, 0],
                    ).start()

                bswitch(g_pool, [lambda r=r: do_out(r) for r in raw_refs])
        else:
            contig = self.gdn_contig_ckpt(sub, s)
            out_mode[slot_o] = jnp.where(
                act_out, jnp.where(contig, 1, 2), 0
            ).astype(I32)

            @pl.when(jnp.logical_and(act_out, contig))
            def _():
                pb0 = self.gdn_pb_out(sub, s, 0)

                def do_out_c(r):
                    pltpu.make_async_copy(
                        out_page.at[slot_o],
                        r.at[pl.ds(pb0, GDN_PAGES * T)],
                        out_sem.at[slot_o, 0],
                    ).start()

                bswitch(g_pool, [lambda r=r: do_out_c(r) for r in raw_refs])

            @pl.when(jnp.logical_and(act_out, jnp.logical_not(contig)))
            def _():
                def do_out_nc(r):
                    for t in range(T):
                        pb_t = self.gdn_pb_out(sub, s, t)
                        pltpu.make_async_copy(
                            out_page.at[slot_o, pl.ds(t * GDN_PAGES, GDN_PAGES)],
                            r.at[pl.ds(pb_t, GDN_PAGES)],
                            out_sem.at[slot_o, t],
                        ).start()

                bswitch(g_pool, [lambda r=r: do_out_nc(r) for r in raw_refs])

    fire_gdn_out = start_gdn_out

    def wait_gdn_out(self, slot_o):
        ctx = self.ctx
        scr, raw_refs, GDN_PAGES, T = ctx.scr, ctx.raw_refs, ctx.GDN_PAGES, ctx.T
        out_page = scr["gdn_out_page"]
        out_sem = scr["gdn_out_sem"]
        out_mode = scr["gdn_out_mode"]
        mode = out_mode[slot_o]

        @pl.when(mode == 1)
        def _():
            pltpu.make_async_copy(
                out_page.at[slot_o],
                raw_refs[0].at[pl.ds(0, GDN_PAGES * T)],
                out_sem.at[slot_o, 0],
            ).wait()

        if T > 1:
            @pl.when(mode == 2)
            def _():
                for t in range(T):
                    pltpu.make_async_copy(
                        out_page.at[slot_o, pl.ds(t * GDN_PAGES, GDN_PAGES)],
                        raw_refs[0].at[pl.ds(0, GDN_PAGES)],
                        out_sem.at[slot_o, t],
                    ).wait()

        out_mode[slot_o] = jnp.int32(0)

    def gpr_start(self, o, seqs, dsts, sem, first, npages):
        """GPR: start the DMAs of checkpoint pages first .. first + npages - 1 of GatedDeltaNet
        layer o's input state of each (active) sequence in seqs into dsts (one pool switch)."""
        ctx = self.ctx
        G_PER_ATT, raw_refs = ctx.G_PER_ATT, ctx.raw_refs
        sub = lax.rem(o, G_PER_ATT)
        g_pool = lax.div(o, G_PER_ATT)

        def issue(r):
            for sq, dst in zip(seqs, dsts):
                @pl.when(self.is_gdn_in_active(sub, sq))
                def _():
                    pb = self.gdn_pb_in(sub, sq) + first
                    pltpu.make_async_copy(r.at[pl.ds(pb, npages)], dst, sem).start()

        bswitch(g_pool, [lambda r=r: issue(r) for r in raw_refs])

    def gpr_wait(self, sub, sq, dst, sem, npages):
        raw_refs = self.ctx.raw_refs

        @pl.when(self.is_gdn_in_active(sub, sq))
        def _():
            pltpu.make_async_copy(raw_refs[0].at[pl.ds(0, npages)], dst, sem).wait()

    def gpr_conv_copy(self, r, sq, pb, sem):
        """GPR: the leading GCONV_ROWS (4,128)-rows of conv page pb ([pb, 0, :GCONV_ROWS])."""
        ctx = self.ctx
        return pltpu.make_async_copy(
            r.at[pb, 0, pl.ds(0, ctx.GCONV_ROWS)], ctx.scr["gpr_conv"].at[sq], sem
        )

    def gpr_conv_start(self, o):
        ctx = self.ctx
        G_PER_ATT, NS, REC_PAGES = ctx.G_PER_ATT, ctx.NS, ctx.REC_PAGES
        scr, raw_refs = ctx.scr, ctx.raw_refs
        sub = lax.rem(o, G_PER_ATT)
        g_pool = lax.div(o, G_PER_ATT)

        def issue(r):
            for sq in range(NS):
                @pl.when(self.is_gdn_in_active(sub, sq))
                def _():
                    self.gpr_conv_copy(
                        r, sq, self.gdn_pb_in(sub, sq) + REC_PAGES, scr["st_sem"].at[0]
                    ).start()

        bswitch(g_pool, [lambda r=r: issue(r) for r in raw_refs])

    def GprRecIn(self, o, j, s, n):
        return GprRecIn(self, o, j, s, n)

    def rec_chunk(self, o, j, s, out):
        """GCH: chunk j (state heads j * CBV ..; static j == NF: the TR-row tail) of
        GatedDeltaNet layer o's states, HBM -> ring buffer s (out: buffer s -> the output
        states). T > 1: a chunk holds n // (T * vh) sequences; their input states are placed
        in the last n // T head slots of buffer s, where each sequence's token 0 reads its
        initial state before those slots are overwritten by the last sequence's tokens."""
        ctx = self.ctx
        scr, ST, O = ctx.scr, ctx.ST, ctx.O
        TR, vh, NF, CBV, T = ctx.TR, ctx.vh, ctx.NF, ctx.CBV, ctx.T
        n = TR * vh if TR and isinstance(j, int) and j == NF else CBV
        r0 = j * CBV if isinstance(j, int) else pl.multiple_of(j * CBV, CBV)
        buf = scr["recb"].at[s] if n == CBV else scr["recb"].at[s, pl.ds(0, n)]
        if out:
            return pltpu.make_async_copy(
                buf, O["rec"].at[o, pl.ds(r0, n)], scr["rec_sem"].at[1, s]
            )
        if ctx.GPR:
            return GprRecIn(self, o, j, s, n)
        if ctx.VER:
            ni = n // T
            ri = (
                j * (CBV // T)
                if isinstance(j, int)
                else pl.multiple_of(j * (CBV // T), CBV // T)
            )
            return pltpu.make_async_copy(
                ST["rec"].at[o, pl.ds(ri, ni)],
                scr["recb"].at[s, pl.ds(n - ni, ni)],
                scr["rec_sem"].at[0, s],
            )
        return pltpu.make_async_copy(
            ST["rec"].at[o, pl.ds(r0, n)], buf, scr["rec_sem"].at[0, s]
        )

    def conv_out_src(self):
        ctx = self.ctx
        return ctx.scr["convb16"] if ctx.GPR else ctx.scr["convb"]

    def rec_drain(self, o):
        """GCH: wait for layer o's write-backs still in flight (its last NRB chunks)."""
        ctx = self.ctx
        NCH, NRB = ctx.NCH, ctx.NRB
        for j_ in range(NCH - NRB, NCH):
            self.rec_chunk(o, j_, j_ % NRB, True).wait()


class KVPoolDMA:
    """KV page table and fused scatter/read-modify-write DMA manager."""

    def __init__(self, ctx: MKBuildContext):
        self.ctx = ctx

    def is_seq_active(self, s):
        ctx = self.ctx
        c, ST = ctx.c, ctx.ST
        if c.act_mode == "ref":
            return ST["active_seqs"][s] != 0
        if c.act_mode == "from_ckpt":
            return (
                (ST["ckpt_indices"][0, s, 0] > 0)
                if ctx.ckpt_is_3d
                else (ST["ckpt_indices"][s, 0] > 0)
            )
        if c.act_mode == "pos":
            return ctx.pos_ref[s * ctx.T] >= 0
        return jnp.bool_(True)

    def seq_same_page(self, s):
        ctx = self.ctx
        c, T, pos_ref = ctx.c, ctx.T, ctx.pos_ref
        if T == 1:
            return jnp.bool_(True)
        b0 = s * T
        lp0 = lax.div(pos_ref[b0], c.page_size)
        ok = jnp.bool_(True)
        for t in range(1, T):
            ok = jnp.logical_and(ok, lax.div(pos_ref[b0 + t], c.page_size) == lp0)
        return ok

    def batch_to_i32_cols(self, tok_mr_hd):
        ctx = self.ctx
        c, MR = ctx.c, ctx.MR
        rows_bf16 = tok_mr_hd.astype(BF16)
        if MR < c.page_size:
            rows_bf16 = jnp.pad(rows_bf16, ((0, c.page_size - MR), (0, 0)))
        cols_fp8 = jnp.swapaxes(rows_bf16, 0, 1).astype(FP8)
        return pltpu.bitcast(cols_fp8, I32)

    def wait_kv_rd(self, s):
        ctx = self.ctx
        scr, raw_refs = ctx.scr, ctx.raw_refs
        pltpu.make_async_copy(
            raw_refs[0].at[0], scr["kv_scat_page"].at[s], scr["kv_rd_sem"].at[s]
        ).wait()

    def wait_kv_wr(self, s):
        ctx = self.ctx
        scr, raw_refs = ctx.scr, ctx.raw_refs
        wr_pend = scr["kv_wr_pend"]
        pend = wr_pend[s]

        @pl.when(pend != 0)
        def _():
            pltpu.make_async_copy(
                scr["kv_scat_page"].at[s], raw_refs[0].at[0], scr["kv_wr_sem"].at[s]
            ).wait()

        wr_pend[s] = jnp.int32(0)

    def kvrd_issue(self, a):
        """Fused pool I/O: start the read of each active sequence's write page (its T new
        tokens share one page). Depends only on pos, the page table and the stage's previous
        write-back, and no KV block read of this layer writes the page: issue it early
        (att-ici-overlap: inside the q exchange round trip)."""
        ctx = self.ctx
        c, scr, ST, pos_ref, raw_refs = (
            ctx.c,
            ctx.scr,
            ctx.ST,
            ctx.pos_ref,
            ctx.raw_refs,
        )
        NS, T = ctx.NS, ctx.T
        pt_ref = ST["page_table"]
        kv_scat = scr["kv_scat_page"]
        rd_sem = scr["kv_rd_sem"]
        rd_pend = scr["kv_rd_pend"]
        for s in range(NS):
            self.wait_kv_wr(s)
            act_s = self.is_seq_active(s)
            same_s = self.seq_same_page(s)
            do_rd = jnp.logical_and(act_s, same_s)
            rd_pend[s] = jnp.where(do_rd, 1, 0).astype(I32)

            @pl.when(do_rd)
            def _(s=s):
                lp0 = lax.div(pos_ref[s * T], c.page_size)
                kp0 = jnp.clip(pt_ref[s, lp0], 0, c.num_pages - 1)

                def issue_rd(r):
                    pltpu.make_async_copy(
                        r.at[kp0], kv_scat.at[s], rd_sem.at[s]
                    ).start()

                bswitch(a, [lambda r=r: issue_rd(r) for r in raw_refs])

    start_kv_rd = kvrd_issue

    def kv_rmw(self, a):
        """Fused pool I/O: write the T new tokens into each sequence's write page (read
        issued in kvrd_issue) and start its write-back. Needs only knew_o / vnew_o and the
        page read; every KV block read of this layer is done."""
        ctx = self.ctx
        c, scr, ST, pos_ref, raw_refs = (
            ctx.c,
            ctx.scr,
            ctx.ST,
            ctx.pos_ref,
            ctx.raw_refs,
        )
        NS, T, typed = ctx.NS, ctx.T, ctx.typed
        pt_ref = ST["page_table"]
        kv_scat = scr["kv_scat_page"]
        rd_sem, wr_sem = scr["kv_rd_sem"], scr["kv_wr_sem"]
        rd_pend, wr_pend = scr["kv_rd_pend"], scr["kv_wr_pend"]
        lane_iota_kv = lax.broadcasted_iota(I32, (64, c.page_size), 1)
        k_cols = self.batch_to_i32_cols(scr["knew_o"][...])
        v_cols = self.batch_to_i32_cols(scr["vnew_o"][...])
        for s in range(NS):
            act_s = self.is_seq_active(s)
            same_s = self.seq_same_page(s)
            do_fast = rd_pend[s] != 0
            rd_pend[s] = jnp.int32(0)

            @pl.when(do_fast)
            def _(s=s):
                self.wait_kv_rd(s)
                b0 = s * T
                if typed:
                    k_cur = pltpu.bitcast(kv_scat[s, 0], I32).reshape(64, c.page_size)
                    v_cur = pltpu.bitcast(kv_scat[s, 1], I32).reshape(64, c.page_size)
                else:
                    k_view = kv_scat.at[s, 0].bitcast(I32).reshape(64, c.page_size)
                    v_view = kv_scat.at[s, 1].bitcast(I32).reshape(64, c.page_size)
                    k_cur = k_view[...]
                    v_cur = v_view[...]
                for t in range(T):
                    # lane rotate (jnp.roll semantics) column b0+t onto lane
                    # pos % page_size instead of a masked 128-lane reduction
                    p_t = pos_ref[b0 + t]
                    ln_t = lax.rem(p_t, c.page_size)
                    sh_t = lax.rem(
                        ln_t + (c.page_size - (b0 + t) % c.page_size), c.page_size
                    )
                    mask_ln = lane_iota_kv == ln_t
                    k_cur = jnp.where(mask_ln, pltpu.roll(k_cols, sh_t, 1), k_cur)
                    v_cur = jnp.where(mask_ln, pltpu.roll(v_cols, sh_t, 1), v_cur)
                if typed:
                    kv_scat[s, 0] = pltpu.bitcast(
                        k_cur.reshape(64, 1, c.page_size), FP8
                    )
                    kv_scat[s, 1] = pltpu.bitcast(
                        v_cur.reshape(64, 1, c.page_size), FP8
                    )
                else:
                    k_view[...] = k_cur
                    v_view[...] = v_cur
                lp0 = lax.div(pos_ref[b0], c.page_size)
                kp0 = jnp.clip(pt_ref[s, lp0], 0, c.num_pages - 1)

                def issue_wr(r):
                    pltpu.make_async_copy(
                        kv_scat.at[s], r.at[kp0], wr_sem.at[s]
                    ).start()

                bswitch(a, [lambda r=r: issue_wr(r) for r in raw_refs])

            wr_pend[s] = jnp.where(do_fast, 1, 0).astype(I32)
            if T > 1:
                @pl.when(jnp.logical_and(act_s, jnp.logical_not(same_s)))
                def _(s=s):
                    b0 = s * T
                    for t in range(T):
                        p_t = pos_ref[b0 + t]
                        lp_t = lax.div(p_t, c.page_size)
                        ln_t = lax.rem(p_t, c.page_size)
                        kp_t = jnp.clip(pt_ref[s, lp_t], 0, c.num_pages - 1)

                        def rd_one(r, kp_t=kp_t):
                            pltpu.make_async_copy(
                                r.at[kp_t], kv_scat.at[s], rd_sem.at[s]
                            ).start()

                        bswitch(a, [lambda r=r: rd_one(r) for r in raw_refs])
                        # lane rotate (jnp.roll semantics) column b0+t onto lane
                        # ln_t while the page read is in flight
                        sh_t = lax.rem(
                            ln_t + (c.page_size - (b0 + t) % c.page_size), c.page_size
                        )
                        mask_ln = lane_iota_kv == ln_t
                        k_col = pltpu.roll(k_cols, sh_t, 1)
                        v_col = pltpu.roll(v_cols, sh_t, 1)
                        self.wait_kv_rd(s)
                        if typed:
                            k_cur = pltpu.bitcast(kv_scat[s, 0], I32).reshape(
                                64, c.page_size
                            )
                            v_cur = pltpu.bitcast(kv_scat[s, 1], I32).reshape(
                                64, c.page_size
                            )
                            k_cur = jnp.where(mask_ln, k_col, k_cur)
                            v_cur = jnp.where(mask_ln, v_col, v_cur)
                            kv_scat[s, 0] = pltpu.bitcast(
                                k_cur.reshape(64, 1, c.page_size), FP8
                            )
                            kv_scat[s, 1] = pltpu.bitcast(
                                v_cur.reshape(64, 1, c.page_size), FP8
                            )
                        else:
                            k_view = (
                                kv_scat.at[s, 0].bitcast(I32).reshape(64, c.page_size)
                            )
                            v_view = (
                                kv_scat.at[s, 1].bitcast(I32).reshape(64, c.page_size)
                            )
                            k_view[...] = jnp.where(mask_ln, k_col, k_view[...])
                            v_view[...] = jnp.where(mask_ln, v_col, v_view[...])

                        def wr_one(r, kp_t=kp_t):
                            pltpu.make_async_copy(
                                kv_scat.at[s], r.at[kp_t], wr_sem.at[s]
                            ).start()

                        bswitch(a, [lambda r=r: wr_one(r) for r in raw_refs])
                        pltpu.make_async_copy(
                            kv_scat.at[s], raw_refs[0].at[0], wr_sem.at[s]
                        ).wait()
