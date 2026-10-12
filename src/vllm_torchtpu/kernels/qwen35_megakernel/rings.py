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
"""Dense and expert VMEM weight-streaming rings and tiled matmul helpers."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import pool_alias
from .config import F32, bswitch
from .context import MKBuildContext


class DenseWeightRing:
    """Static-schedule dense weight VMEM ring (`fetch`, `take`/`dtake`, `give`/`dgive`, `dmatmul`)."""

    def __init__(self, ctx: MKBuildContext):
        self.ctx = ctx

    def dview(self, kind: str, bank):
        ctx = self.ctx
        if ctx.typed:
            return ctx.scr["dring_" + kind].at[bank]
        shp, dt = ctx.vshape[kind], jnp.dtype(ctx.vdtype[kind])
        return pool_alias.view(
            ctx.scr["dpool"].at[
                pl.ds(bank * ctx.BROWS, pool_alias.storage_size(shp, dt) // 128)
            ],
            shp,
            dt,
        )

    def dcopy(self, kind: str, src, bank):
        return pltpu.make_async_copy(src, self.dview(kind, bank), self.ctx.scr["dsem"].at[bank])

    def seg_issue(self, segs, r, ords, bank):
        W = self.ctx.W
        for kind, lo, hi, ordi in segs:
            @pl.when(jnp.logical_and(r >= lo, r < hi))
            def _(kind=kind, lo=lo, ordi=ordi):
                self.dcopy(kind, W[kind].at[ords[ordi], r - lo], bank).start()

    def fetch(self, g):
        """Issue dense stream tile g into bank g % NBD (static schedule)."""
        ctx = self.ctx
        T_TOTAL, skip = ctx.T_TOTAL, ctx.skip
        t_block, t_gdn, t_att, AE, NBD = ctx.t_block, ctx.t_gdn, ctx.t_att, ctx.AE, ctx.NBD
        nt = ctx.nt

        @pl.when(jnp.logical_and(g < T_TOTAL, "dense" not in skip))
        def _():
            blk = g // t_block
            sub = g - blk * t_block
            lt = jnp.minimum(sub // t_gdn, AE - 1)
            r = sub - lt * t_gdn
            layer = blk * AE + lt
            bank = g % NBD

            if AE > 1:
                @pl.when(lt < AE - 1)
                def _gdn():
                    a0 = nt["lin_in"]
                    a1 = a0 + nt["lin_out"]
                    a2 = a1 + nt["router"]
                    a3 = a2 + nt["sh13"]
                    self.seg_issue(
                        [
                            ("lin_in", 0, a0, 0),
                            ("lin_out", a0, a1, 0),
                            ("router", a1, a2, 1),
                            ("sh13", a2, a3, 1),
                            ("sh2", a3, t_gdn, 1),
                        ],
                        r,
                        (blk * (AE - 1) + lt, layer),
                        bank,
                    )

            @pl.when(lt == AE - 1)
            def _att():
                a0 = nt["att_in"]
                a1 = a0 + nt["att_out"]
                a2 = a1 + nt["router"]
                a3 = a2 + nt["sh13"]
                self.seg_issue(
                    [
                        ("att_in", 0, a0, 0),
                        ("att_out", a0, a1, 0),
                        ("router", a1, a2, 1),
                        ("sh13", a2, a3, 1),
                        ("sh2", a3, t_att, 1),
                    ],
                    r,
                    (blk, layer),
                    bank,
                )

    def dtake(self, kind: str, g):
        ctx = self.ctx
        bank = g % ctx.NBD
        if "dense" not in ctx.skip:
            with ctx.sc("dwait", 2):
                self.dcopy(kind, ctx.W[kind].at[0, 0], bank).wait()
        return self.dview(kind, bank)

    take = dtake

    def dtake_group(self, kinds_gs):
        """dtake for a group [(kind, g), ...] of consecutive tiles g0, g0+1, ...: all banks
        are computed before the first wait (DWAIT_HOIST), and the waits use a static dst view
        (the wait reads only the semaphore and the byte count)."""
        ctx = self.ctx
        if not getattr(ctx, "DWAIT_HOIST", False):
            return [self.dtake(k, g) for k, g in kinds_gs]
        banks = self.dbanks(kinds_gs[0][1], len(kinds_gs))
        if "dense" not in ctx.skip:
            with ctx.sc("dwait", 2):
                for (k, _), b in zip(kinds_gs, banks):
                    pltpu.make_async_copy(
                        ctx.W[k].at[0, 0],
                        self.dview(k, 0),
                        ctx.scr["dsem"].at[b],
                    ).wait()
        return [self.dview(k, b) for (k, _), b in zip(kinds_gs, banks)]

    def dbanks(self, g0, n: int):
        """Ring banks of tiles g0 .. g0+n-1 (n <= NBD, g0 >= 0): one rem, then a wrap."""
        NBD = self.ctx.NBD
        if isinstance(g0, int):
            return [(g0 + u) % NBD for u in range(n)]
        b0 = lax.rem(g0, NBD)
        out = [b0]
        for u in range(1, n):
            b = b0 + u
            out.append(jnp.where(b >= NBD, b - NBD, b))
        return out

    def dbanks_wait(self, kind: str, g0, n: int):
        """dtake_group([(kind, g0 + u) for u in range(n)]), returning the ring banks."""
        ctx = self.ctx
        banks = self.dbanks(g0, n)
        if "dense" not in ctx.skip:
            with ctx.sc("dwait", 2):
                for b in banks:
                    pltpu.make_async_copy(
                        ctx.W[kind].at[0, 0], self.dview(kind, 0), ctx.scr["dsem"].at[b]
                    ).wait()
        return banks

    def arp_mm(self, kind: str, banks, xsrc, cols, hook):
        """ARPIPE: acc[:, c0:c0 + cw] = xsrc @ W(kind) over all K tiles for (c0, cw) in cols:
        the K tiles of this core's all-reduce half first (a register sum per column chunk,
        stored), then hook() (round-2 wait, other half's residual + norm), then the other
        half's tiles (added). Tile s + h * ARP_HPT (s in ARP_S) is half h's; its bank and
        its xsrc lanes are picked at runtime from the core index."""
        mctx = self.ctx
        t_k = mctx.tk[kind]
        acc = mctx.scr["acc"]
        ARP_S, ARP_HPT, arp_c = mctx.ARP_S, mctx.ARP_HPT, mctx.arp_c

        def half_pass(other):
            xs, ws = [], []
            for s in ARP_S:
                b_lo, b_hi = banks[s], banks[s + ARP_HPT]
                if other:
                    b_lo, b_hi = b_hi, b_lo
                h = (1 - arp_c) if other else arp_c
                ws.append(self.dview(kind, jnp.where(arp_c == 0, b_lo, b_hi)))
                k0 = pl.multiple_of((s + h * ARP_HPT) * t_k, t_k)
                xs.append(xsrc[:, pl.ds(k0, t_k)])
            for c0, cw in cols:
                p = mctx.mm(xs[0], ws[0][:, c0 : c0 + cw])
                for i in range(1, len(xs)):
                    p = p + mctx.mm(xs[i], ws[i][:, c0 : c0 + cw])
                if other:
                    acc[:, c0 : c0 + cw] += p
                else:
                    acc[:, c0 : c0 + cw] = p

        half_pass(False)
        hook()
        half_pass(True)

    def dgive(self, g):
        self.fetch(g + self.ctx.NBD)

    give = dgive

    def flush(self, g0, n: int):
        """Refill the slots of tiles g0 .. g0+n-1, consumed lazily (no refill yet)."""
        NBD = self.ctx.NBD
        for t in range(n):
            self.fetch(g0 + t + NBD)

    def rel_decode(self, lt: int, r: int):
        """Trace-time decode of in-stream tile r (int) of layer type lt (int): (block offset,
        layer type, kind, ords index, r value of the kind's row 0). Mirrors fetch(g)."""
        ctx = self.ctx
        t_att, t_gdn, AE, nt = ctx.t_att, ctx.t_gdn, ctx.AE, ctx.nt
        b_off, lt_, base = 0, lt, 0
        while r - base >= (t_att if lt_ == AE - 1 else t_gdn):
            base += t_att if lt_ == AE - 1 else t_gdn
            lt_ += 1
            if lt_ == AE:
                lt_, b_off = 0, b_off + 1
        names = ("att_in", "att_out") if lt_ == AE - 1 else ("lin_in", "lin_out")
        lo = base
        for kind, ordi in ((names[0], 0), (names[1], 0), ("router", 1), ("sh13", 1), ("sh2", 1)):
            if r < lo + nt[kind]:
                break
            lo += nt[kind]
        return b_off, lt_, kind, ordi, lo

    def fetch_rel(self, g, blk, lt, r: int, dt=None, dt_vals=(0,), lts=None):
        """fetch(g) for g = tile r + dt of layer lt of block blk: r a trace-time int, dt None or
        a traced int taking only the trace-time values dt_vals, lt an int or traced (taking only
        the values lts). The (layer, kind) decode of tile g is resolved at trace time."""
        ctx = self.ctx
        AE, L, NBD, skip, W = ctx.AE, ctx.L, ctx.NBD, ctx.skip, ctx.W
        if not isinstance(lt, int):
            for lt_s in (range(AE) if lts is None else lts):
                pl.when(lt == lt_s)(
                    lambda lt_s=lt_s: self.fetch_rel(g, blk, lt_s, r, dt, dt_vals)
                )
            return
        if dt is None:
            dt_vals = (0,)
        groups = {}
        for v in dt_vals:
            groups.setdefault(self.rel_decode(lt, r + v), []).append(v)
        for (b_off, lt_, kind, ordi, lo), vs in groups.items():
            b = blk + b_off
            ords = ((b if lt_ == AE - 1 else b * (AE - 1) + lt_), b * AE + lt_)
            row = r - lo if dt is None else dt + (r - lo)
            cond = jnp.logical_and(b < L // AE, "dense" not in skip)
            if len(groups) > 1:
                hit = dt == vs[0]
                for v in vs[1:]:
                    hit = jnp.logical_or(hit, dt == v)
                cond = jnp.logical_and(cond, hit)

            @pl.when(cond)
            def _(kind=kind, ords=ords, ordi=ordi, row=row):
                self.dcopy(kind, W[kind].at[ords[ordi], row], lax.rem(g, NBD)).start()

    def flush_rel(self, g0, blk, lt, r0: int, n: int):
        """flush(g0, n) with g0 = tile r0 (trace-time int) of layer lt of block blk."""
        NBD = self.ctx.NBD
        for t in range(n):
            self.fetch_rel(g0 + t + NBD, blk, lt, r0 + t + NBD)

    def gctx(self, l, r0: int):
        """dmatmul ctx of a GDN layer l (layer type l % AE in 0 .. AE-2), kind at tile r0."""
        AE = self.ctx.AE
        return (l // AE, l % AE, r0, tuple(range(AE - 1)))

    def _give_tile(self, g0, t, u: int, t_vals, dmm_ctx):
        """Refill tile g0 + t + u (t int or traced with values t_vals, u int)."""
        NBD = self.ctx.NBD
        if dmm_ctx is None:
            self.dgive(g0 + t + u)
            return
        blk, lt, r0, lts = dmm_ctx
        if isinstance(t, int):
            self.fetch_rel(g0 + t + u + NBD, blk, lt, r0 + t + u + NBD, lts=lts)
        else:
            self.fetch_rel(
                g0 + t + u + NBD, blk, lt, r0 + u + NBD, dt=t, dt_vals=t_vals, lts=lts
            )

    def dmatmul_pair(
        self,
        kind: str,
        g0,
        xsrc,
        N: int,
        lazy: bool = False,
        ctx=None,
        ws_pre=None,
        hook=None,
        split=None,
    ):
        """Paired/unrolled dense matmul: acc[:, :N] = xsrc[:, :K] @ W(kind)."""
        mctx = self.ctx
        n_t, t_k = mctx.nt[kind], mctx.tk[kind]
        acc = mctx.scr["acc"]
        DMU = (
            min(mctx.dmm_unroll, mctx.NBD)
            if mctx.B not in mctx.DMM_UNROLL_OFF_B
            else 0
        )
        DCW = 512
        if n_t <= DMU:
            U = n_t
        else:
            U = 2 if n_t % 2 == 0 and N > DCW else 1
        chunks = [(c0, min(DCW, N - c0)) for c0 in range(0, N, DCW)]
        n_s = n_t // U
        if hook is not None:
            assert U == n_t and ws_pre is None and split is None
            banks = self.dbanks_wait(kind, g0, n_t)
            self.arp_mm(kind, banks, xsrc, chunks, hook)
            if not lazy:
                for u in range(n_t):
                    self._give_tile(g0, 0, u, (), ctx)
            return acc[:, :N]
        groups = [(chunks, None)]
        if split is not None:
            assert U == n_t and n_s == 1 and hook is None and ws_pre is None, kind
            assert sum(cw for cols_ in split["cols"] for _, cw in cols_) == N, kind
            groups = list(zip(split["cols"], split["hooks"]))
            split["pre"]()

        def step(t, first):
            if ws_pre is not None and isinstance(t, int) and t == 0 and U == n_t:
                ws = [b if not isinstance(b, int) else self.dview(kind, b) for b in ws_pre]
                xs = [xsrc[:, u * t_k : (u + 1) * t_k] for u in range(U)]
            else:
                ws, xs = [], []
                if getattr(mctx, "DWAIT_HOIST", False):
                    ws = self.dtake_group([(kind, g0 + t + u) for u in range(U)])
                for u in range(U):
                    if len(ws) <= u:
                        ws.append(self.dtake(kind, g0 + t + u))
                    k0 = (t + u) * t_k
                    if not isinstance(k0, int):
                        k0 = pl.multiple_of(k0, t_k)
                    xs.append(xsrc[:, pl.ds(k0, t_k)])
            for cols_, hk in groups:
                vals = []
                for c0, cw in cols_:
                    cs = slice(c0, c0 + cw) if isinstance(c0, int) else pl.ds(c0, cw)
                    p = mctx.mm(xs[0], ws[0][:, cs])
                    for u in range(1, U):
                        p = p + mctx.mm(xs[u], ws[u][:, cs])
                    if hk is not None:
                        vals.append(p)
                    elif first:
                        acc[:, cs] = p
                    else:
                        acc[:, cs] += p
                if hk is not None:
                    hk(vals)
            if not lazy:
                for u in range(U):
                    self._give_tile(g0, t, u, tuple(i * U for i in range(1, n_s)), ctx)

        n_s = n_t // U
        step(0, True)
        if n_s == 2:
            step(U, False)
        elif n_s > 2:
            def body_pair(i, carry):
                step(i * U, False)
                return carry

            lax.fori_loop(1, n_s, body_pair, 0)
        return acc[:, :N]

    def dmatmul(
        self,
        kind: str,
        g0,
        xsrc,
        N: int,
        lazy: bool = False,
        ctx=None,
        ws_pre=None,
        hook=None,
        split=None,
    ):
        """acc[:, :N] = xsrc[:, :K] @ W(kind), tiles g0 .. g0+nt-1 of the dense stream."""
        mctx = self.ctx
        n_t, t_k = mctx.nt[kind], mctx.tk[kind]
        acc = mctx.scr["acc"]
        DMU = (
            min(mctx.dmm_unroll, mctx.NBD)
            if mctx.B not in mctx.DMM_UNROLL_OFF_B
            else 0
        )
        if n_t <= DMU or mctx.tiles.dense_acc_pair:
            return self.dmatmul_pair(
                kind,
                g0,
                xsrc,
                N,
                lazy=lazy,
                ctx=ctx,
                ws_pre=ws_pre,
                hook=hook,
                split=split,
            )

        acc[:, :N] = jnp.zeros((mctx.MB, N), F32)

        def body(t, carry):
            g = g0 + t
            w = self.dtake(kind, g)
            k0 = pl.multiple_of(t * t_k, t_k)
            acc[:, :N] += mctx.mm(xsrc[:, pl.ds(k0, t_k)], w[...])
            if not lazy:
                self._give_tile(g0, t, 0, tuple(range(n_t)), ctx)
            return carry

        lax.fori_loop(0, n_t, body, 0)
        return acc[:, :N]


class ExpertWeightRing:
    """Dynamic-hit-list expert weight VMEM ring (`ecopy`, `etake`, `egive`, `egive_group`, `fetch_loop`)."""

    def __init__(self, ctx: MKBuildContext):
        self.ctx = ctx

    def eview(self, kind: str, bank):
        ctx = self.ctx
        if ctx.typed:
            return ctx.scr["ering_" + kind].at[bank]
        shp, dt = ctx.vshape[kind], jnp.dtype(ctx.vdtype[kind])
        return pool_alias.view(
            ctx.scr["epool"].at[
                pl.ds(bank * ctx.EBROWS, pool_alias.storage_size(shp, dt) // 128)
            ],
            shp,
            dt,
        )

    def ecopy(self, kind: str, src, bank):
        return pltpu.make_async_copy(src, self.eview(kind, bank), self.ctx.scr["esem"].at[bank])

    def fetch_e(self, u, l, t=None, fixed: bool = False):
        """Issue unit u into bank u % NBE. t: u's tile index when known at trace time."""
        ctx = self.ctx
        c, scr, W = ctx.c, ctx.scr, ctx.W
        TE, T13, NBE, NSS, ER, ECOPY = ctx.TE, ctx.T13, ctx.NBE, ctx.NSS, ctx.ER, ctx.ECOPY
        me = ctx.me
        if t is None and isinstance(u, int):
            t = u % TE
        i = u // TE if t is None else (u - t) // TE
        if ECOPY:
            if getattr(ctx, "PRIM_NOW_K", False):
                w_fix = scr["elist"][i] - me * ER
                w = w_fix if fixed else jnp.where(i < scr["st"][5], w_fix, scr["wlist"][i])
            else:
                w = scr["wlist"][i]
        else:
            e = scr["elist"][i]
            w = e if c.moe_tp else e - me * ER
        bank = u % NBE

        def es_copy():
            s = i % NSS
            pltpu.make_async_copy(
                W["es"].at[l, w], scr["es"].at[s], scr["es_sem"].at[s]
            ).start()

        def e13_start(tt):
            if c.stack_e:
                self.ecopy("e13", W["e13"].at[l, w, tt], bank).start()
            else:
                bswitch(
                    l,
                    [lambda r=r: self.ecopy("e13", r.at[w, tt], bank).start() for r in W["e13"]],
                )

        def e2_start(tt):
            if c.stack_e:
                self.ecopy("e2", W["e2"].at[l, w, tt], bank).start()
            else:
                bswitch(
                    l,
                    [lambda r=r: self.ecopy("e2", r.at[w, tt], bank).start() for r in W["e2"]],
                )

        if t is not None:
            if t < T13:
                e13_start(t)
            else:
                e2_start(t - T13)
            if t == 0:
                es_copy()
            return
        t = u - i * TE

        @pl.when(t < T13)
        def _():
            e13_start(t)

        @pl.when(t >= T13)
        def _():
            e2_start(t - T13)

        pl.when(t == 0)(es_copy)

    def etake(self, kind: str, u):
        ctx = self.ctx
        bank = u % ctx.NBE
        with ctx.sc("ewait", 2):
            w0 = ctx.W[kind].at[0, 0, 0] if ctx.c.stack_e else ctx.W[kind][0].at[0, 0]
            self.ecopy(kind, w0, bank).wait()
        return self.eview(kind, bank)

    def egive(self, u, l, t=None):
        ctx = self.ctx
        nu = u + ctx.NBE
        nt_ = None if t is None else (t + ctx.NBE) % ctx.TE

        @pl.when(nu < ctx.scr["st"][0])
        def _():
            self.fetch_e(nu, l, nt_)

    def _unit_meta(self, u, t):
        ctx = self.ctx
        c, scr = ctx.c, ctx.scr
        TE, NBE, ER, ECOPY, me = ctx.TE, ctx.NBE, ctx.ER, ctx.ECOPY, ctx.me
        if t is None and isinstance(u, int):
            t = u % TE
        dyn = not isinstance(u, int)
        if t is None:
            i = lax.div(u, jnp.int32(TE)) if dyn else u // TE
        else:
            i = lax.div(u - t, jnp.int32(TE)) if dyn else (u - t) // TE
        if ECOPY:
            lst = scr["wlist"]
        else:
            lst = scr["elist"]
        n_l = lst.shape[0]
        ic = i if isinstance(i, int) else jnp.minimum(i, n_l - 1)
        if ECOPY:
            if getattr(ctx, "PRIM_NOW_K", False):
                w_fix = scr["elist"][ic] - me * ER
                w = jnp.where(ic < scr["st"][5], w_fix, scr["wlist"][ic])
            else:
                w = lst[ic]
        else:
            e = lst[ic]
            w = e if c.moe_tp else e - me * ER
        td = None if t is not None else u - i * TE
        bank = lax.rem(u, jnp.int32(NBE)) if dyn else u % NBE
        return i, w, bank, t, td

    def _es_start(self, i, w, l):
        ctx = self.ctx
        NSS, W, scr = ctx.NSS, ctx.W, ctx.scr
        s = lax.rem(i, jnp.int32(NSS)) if not isinstance(i, int) else i % NSS
        pltpu.make_async_copy(W["es"].at[l, w], scr["es"].at[s], scr["es_sem"].at[s]).start()

    def _arm_issue(self, r13, r2, w, bank, t, td, gate):
        T13 = self.ctx.T13

        def g(p):
            return p if gate is None else jnp.logical_and(gate, p)

        if t is not None:
            def go():
                if t < T13:
                    self.ecopy("e13", r13.at[w, t], bank).start()
                else:
                    self.ecopy("e2", r2.at[w, t - T13], bank).start()

            if gate is None:
                go()
            else:
                pl.when(gate)(go)
            return

        @pl.when(g(td < T13))
        def _():
            self.ecopy("e13", r13.at[w, td], bank).start()

        @pl.when(g(td >= T13))
        def _():
            self.ecopy("e2", r2.at[w, td - T13], bank).start()

    def egive_group(self, items, l):
        """items: [(u, t)] consumed units; refills u + NBE (gated on < st) in ONE walk."""
        ctx = self.ctx
        if ctx.c.stack_e:
            for u, t in items:
                self.egive(u, l, t)
            return
        NBE, TE, W, scr = ctx.NBE, ctx.TE, ctx.W, ctx.scr
        st = scr["st"][0]
        metas = []
        for u, t in items:
            nu = u + NBE
            nt_ = None if t is None else (t + NBE) % TE
            metas.append(self._unit_meta(nu, nt_) + (nu < st,))

        def arm(r13, r2):
            for (_, w, bank, t, td, gate) in metas:
                self._arm_issue(r13, r2, w, bank, t, td, gate)

        bswitch(l, [lambda a=a, b=b: arm(a, b) for a, b in zip(W["e13"], W["e2"])])
        for (i, w, bank, t, td, gate) in metas:
            if t is not None:
                if t == 0:
                    pl.when(gate)(lambda i=i, w=w: self._es_start(i, w, l))
            else:
                pl.when(jnp.logical_and(gate, td == 0))(
                    lambda i=i, w=w: self._es_start(i, w, l)
                )

    def fetch_loop(self, nf, l):
        """Initial ring fill: units 0..nf-1 (nf <= NBE) in ONE bswitch walk (loop in the arm)."""
        W = self.ctx.W

        def arm(r13, r2):
            def body(k, carry):
                i, w, _, t, td = self._unit_meta(k, None)
                self._arm_issue(r13, r2, w, k, t, td, None)  # k < nf <= NBE: bank == k
                pl.when(td == 0)(lambda: self._es_start(i, w, l))
                return carry

            lax.fori_loop(0, nf, body, 0)

        bswitch(l, [lambda a=a, b=b: arm(a, b) for a, b in zip(W["e13"], W["e2"])])

    def fetch_range(self, k0, k1, l, fixed: bool = False):
        """Issue units k0 .. k1 - 1 into the ring."""
        ctx = self.ctx
        if ctx.c.stack_e:
            for k in range(ctx.NBE):
                @pl.when(jnp.logical_and(k >= k0, k < k1))
                def _(k=k):
                    self.fetch_e(k, l, fixed=fixed)
        else:
            lax.fori_loop(
                k0,
                k1,
                lambda k, _: (self.fetch_e(k, l, fixed=fixed), 0)[1],
                0,
            )

    def es_wait(self, i):
        ctx = self.ctx
        s = i % ctx.NSS
        with ctx.sc("swait", 2):
            pltpu.make_async_copy(
                ctx.W["es"].at[0, 0], ctx.scr["es"].at[s], ctx.scr["es_sem"].at[s]
            ).wait()


class WeightRings(DenseWeightRing, ExpertWeightRing):
    """Combined dense + expert weight rings facade."""

    def __init__(self, ctx: MKBuildContext):
        DenseWeightRing.__init__(self, ctx)
        ExpertWeightRing.__init__(self, ctx)
        self.dense = self
        self.expert = self
